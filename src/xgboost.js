// Move Node class outside
class Node {
  constructor() {
    this.featureIndex = null;
    this.threshold = null;
    this.left = null;
    this.right = null;
    this.value = null;
    this.isLeaf = false;
  }
}

class XGBoost {
  constructor(params = {}) {
    this.learningRate = params.learningRate || 0.3;
    this.maxDepth = params.maxDepth || 4;
    this.minChildWeight = params.minChildWeight || 1;
    this.numRounds = params.numRounds || 100;
    this.trees = [];
  }

  fit(X, y) {
    let predictions = new Array(y.length).fill(0.5); // Initialize with base prediction
    
    for (let i = 0; i < this.numRounds; i++) {
      // Calculate gradients (for binary classification)
      const gradients = y.map((actual, idx) => {
        const pred = predictions[idx];
        return actual - pred; // Simple gradient for logistic loss
      });

      // Build tree using gradients
      const tree = this._buildTree(X, gradients, 0);
      this.trees.push({ root: tree });

      // Update predictions
      predictions = predictions.map((pred, idx) => {
        const update = this._predict(X[idx], tree);
        return pred + this.learningRate * update;
      });
    }
  }

  _buildTree(X, gradients, depth) {
    const node = new Node();
    
    // Calculate sum of gradients and count of samples
    const sumGrad = gradients.reduce((a, b) => a + b, 0);
    const count = gradients.length;
    
    // Check stopping conditions
    if (depth >= this.maxDepth || count < this.minChildWeight || Math.abs(sumGrad) < 1e-10) {
      node.isLeaf = true;
      node.value = sumGrad / (count + 1e-10); // Avoid division by zero
      return node;
    }

    // Find best split
    let bestGain = 0;
    let bestSplit = null;

    for (let featureIdx = 0; featureIdx < X[0].length; featureIdx++) {
      const sortedIndices = Array.from({length: X.length}, (_, i) => i)
        .sort((a, b) => X[a][featureIdx] - X[b][featureIdx]);

      let leftSum = 0;
      let leftCount = 0;
      let rightSum = sumGrad;
      let rightCount = count;

      for (let i = 0; i < sortedIndices.length - 1; i++) {
        const idx = sortedIndices[i];
        leftSum += gradients[idx];
        rightSum -= gradients[idx];
        leftCount++;
        rightCount--;

        if (X[sortedIndices[i]][featureIdx] === X[sortedIndices[i + 1]][featureIdx]) {
          continue;
        }

        const gain = (leftSum * leftSum) / (leftCount + 1e-10) + 
                   (rightSum * rightSum) / (rightCount + 1e-10) - 
                   (sumGrad * sumGrad) / (count + 1e-10);

        if (gain > bestGain) {
          bestGain = gain;
          bestSplit = {
            featureIndex: featureIdx,
            threshold: (X[sortedIndices[i]][featureIdx] + X[sortedIndices[i + 1]][featureIdx]) / 2,
            leftIndices: sortedIndices.slice(0, i + 1),
            rightIndices: sortedIndices.slice(i + 1)
          };
        }
      }
    }

    if (!bestSplit || bestGain < 1e-10) {
      node.isLeaf = true;
      node.value = sumGrad / (count + 1e-10);
      return node;
    }

    node.featureIndex = bestSplit.featureIndex;
    node.threshold = bestSplit.threshold;

    // Recursively build left and right subtrees
    const leftX = bestSplit.leftIndices.map(i => X[i]);
    const leftGrad = bestSplit.leftIndices.map(i => gradients[i]);
    node.left = this._buildTree(leftX, leftGrad, depth + 1);

    const rightX = bestSplit.rightIndices.map(i => X[i]);
    const rightGrad = bestSplit.rightIndices.map(i => gradients[i]);
    node.right = this._buildTree(rightX, rightGrad, depth + 1);

    return node;
  }

  predictSingle(x) {
    let pred = 0.5; // base prediction
    for (const tree of this.trees) {
      pred += this.learningRate * this._predict(x, tree.root);
    }
    return 1 / (1 + Math.exp(-pred)); // sigmoid for probability
  }

  predictBatch(X) {
    return X.map(x => this.predictSingle(x));
  }

  _predict(x, node) {
    if (node.isLeaf) return node.value;
    return x[node.featureIndex] <= node.threshold ? 
      this._predict(x, node.left) : 
      this._predict(x, node.right);
  }

  getFeatureImportance() {
    if (!this.trees || !this.trees.length) {
      return [];
    }
    
    // Get number of features from training data
    const numFeatures = this.trees[0].root ? this._getNumFeatures(this.trees[0].root) : 0;
    const importance = new Array(numFeatures).fill(0);
    
    this.trees.forEach(tree => {
      this._traverseTreeForImportance(tree.root, importance);
    });
    
    return importance;
  }

  _getNumFeatures(node) {
    if (!node) return 0;
    if (node.isLeaf) return 0;
    return Math.max(
      node.featureIndex + 1,
      this._getNumFeatures(node.left),
      this._getNumFeatures(node.right)
    );
  }

  _traverseTreeForImportance(node, importance) {
    if (!node || node.isLeaf) return;
    
    importance[node.featureIndex]++;
    this._traverseTreeForImportance(node.left, importance);
    this._traverseTreeForImportance(node.right, importance);
  }

  toJSON() {
    return {
        trees: this.trees,
        params: {
            learningRate: this.learningRate,
            maxDepth: this.maxDepth,
            minChildWeight: this.minChildWeight,
            numRounds: this.numRounds
        }
    };
  }

  static fromJSON(json) {
    const model = new XGBoost(json.params);
    model.trees = json.trees;
    return model;
  }
}

// Add this export statement at the end
module.exports = { XGBoost };
