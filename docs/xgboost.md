# XGBoost.js Documentation

## Introduction

XGBoost.js is my JavaScript version of the XGBoost algorithm. It handles classification and regression tasks, letting you train models and make predictions directly in your JavaScript projects, whether on the server or in the browser.


## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Basic Classification](#basic-classification)
  - [Model Serialization](#model-serialization)
  - [Feature Importance](#feature-importance)
- [Advanced Usage](#advanced-usage)
  - [Handling Multiclass Classification](#handling-multiclass-classification)
  - [Integrating with Web Applications](#integrating-with-web-applications)
- [Real-Life Examples](#real-life-examples)
  - [Predicting Housing Prices](#predicting-housing-prices)
  - [Customer Churn Prediction](#customer-churn-prediction)
- [Testing](#testing)
- [Conclusion](#conclusion)

## Installation

Ensure you have [Node.js](https://nodejs.org/) or [Bun](https://bun.sh/) installed. Install the library with a package manager:

nodejs:
```bash
npm install @fractal-solutions/xgboost-js
```

bun:
```bash
bun add @fractal-solutions/xgboost-js
```

You can clone or download the repository and include the `xgboost.js` module in your project:

```javascript
const { XGBoost } = require('./xgboost.js');
```

## Getting Started

### Basic Classification

Learn how to train a simple binary classification model using XGBoost.js.

#### Step 1: Prepare Your Data

Organize your training data into feature matrices and label vectors.

```javascript
const { XGBoost } = require('./xgboost.js');

// Realistic sample training data
// Each sample consists of two features:
// - Age (in years)
// - Annual Income (in USD)
const X_train = [
    [25, 50000],
    [30, 60000],
    [45, 80000],
    [35, 70000],
    [50, 90000],
    [23, 48000],
    [40, 75000],
    [29, 62000],
    [33, 68000],
    [38, 72000],
    [27, 53000],
    [42, 77000],
    [31, 61000],
    [36, 69000],
    [48, 85000],
    [22, 47000],
    [39, 73000],
    [34, 66000],
    [28, 59000],
    [46, 82000],
];

// Labels corresponding to the training data
// 0: Did Not Purchase
// 1: Purchased
const y_train = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1];
```

#### Step 2: Initialize and Train the Model

Configure the model parameters and train using the `fit` method. Understanding the hyperparameters is crucial for optimizing the model's performance:

- **learningRate**: Controls the contribution of each tree to the final model. A smaller value makes the learning process more robust but requires more trees.
- **maxDepth**: Sets the maximum depth of each tree. Deeper trees can capture more complex patterns but may lead to overfitting.
- **minChildWeight**: Specifies the minimum sum of instance weights (hessian) needed in a child. It helps prevent overfitting by controlling the complexity of the trees.
- **numRounds**: Determines the number of boosting rounds or the number of trees to be built. More rounds can improve performance but increase computational cost.

```javascript
// Initialize the model with parameters and explain each hyperparameter
const model = new XGBoost({
    learningRate: 0.3,       // Determines the step size at each iteration
    maxDepth: 4,             // Maximum depth of a tree
    minChildWeight: 1,       // Minimum sum of instance weight (hessian) needed in a child
    numRounds: 100            // Number of boosting rounds
});

// Train the model with the training data
model.fit(X_train, y_train);
```

 // Start of Selection
#### Step 3: Make Predictions

Use the trained model to predict outcomes on new data. Ensure that the test data follows the same feature structure as the training data.

```javascript
// Sample test data following the same feature structure as the training data
const X_test = [
    [20, 45000],
    [35, 75000],
    [40, 82000],
];

// Predict probabilities for the test data using predictBatch
const predictionsBatch = model.predictBatch(X_test);

// Predict probabilities for the test data using predictSingle
const predictionsSingle = X_test.map(x => model.predictSingle(x));

console.log('Batch Predictions:', predictionsBatch); // Outputs an array of probabilities indicating the likelihood of purchase
console.log('Single Predictions:', predictionsSingle); // Outputs an array of probabilities indicating the likelihood of purchase
```


### Model Serialization

Save your trained model and load it later without retraining.

```javascript
// Serialize the model
const serialized = model.toJSON();

// Save 'serialized' to a file or database as needed

// Later, deserialize the model
const deserializedModel = XGBoost.fromJSON(serialized);

// Use the deserialized model for predictions
const newPredictions = deserializedModel.predictBatch(X_test);

console.log(newPredictions);
```


### Feature Importance

Understanding which features contribute most to your model's predictions is essential for interpreting the results and making informed decisions. XGBoost provides a method to retrieve feature importance scores, allowing you to identify and focus on the most influential features in your dataset.

```javascript
// Retrieve feature importance scores
const importance = model.getFeatureImportance();

// Assuming you have an array of feature names corresponding to your dataset
const featureNames = ['feature1', 'feature2', 'feature3', 'feature4'];

// Combine feature names with their importance scores
const featureImportance = featureNames.map((name, index) => ({
    feature: name,
    importance: importance[index]
}));

// Sort features by importance in descending order
featureImportance.sort((a, b) => b.importance - a.importance);

// Display the feature importances
console.log('Feature Importances:');
featureImportance.forEach(({ feature, importance }) => {
    console.log(`${feature}: ${importance}`);
});
```

**Explanation:**

1. **Retrieving Importance Scores:**
   - The `model.getFeatureImportance()` method returns an array where each element represents the importance score of a corresponding feature. The importance is typically based on how frequently a feature is used to split the data across all trees in the model.

2. **Mapping Feature Names:**
   - To make the importance scores more interpretable, especially when dealing with multiple features, it's helpful to map these scores to their respective feature names. This assumes you have an array `featureNames` that lists all feature names in the same order as they were used during training.

3. **Sorting Features:**
   - Sorting the features in descending order of their importance scores allows you to quickly identify which features have the most significant impact on the model's predictions.

4. **Displaying the Results:**
   - The final console log presents a clear and organized view of feature importances, making it easier to interpret and analyze the model's behavior.

**Example Output:**
```
Feature Importances:
age: 25
income: 18
education: 10
gender: 5
```

In this example, the `age` feature is the most influential, followed by `income`, `education`, and `gender`. Such insights can guide feature selection, data collection priorities, and provide explanations for model decisions.

## Advanced Usage

### Handling Multiclass Classification

Extend XGBoost.js to handle multiclass classification tasks by adjusting label encoding and prediction interpretation.

```javascript
// Example setup for multiclass classification with 3 classes
const y_train = [0, 1, 2, 1, 0, 2];

// Initialize the model with multiclass parameters
const model = new XGBoost({
    learningRate: 0.1,
    maxDepth: 6,
    minChildWeight: 1,
    numRounds: 200
});

// Train the model
model.fit(X_train, y_train);

// Predict class probabilities
const predictions = model.predictBatch(X_test);

console.log(predictions);
```

### Integrating with Web Applications

Leverage XGBoost.js in frontend applications for real-time predictions.

```html
<!DOCTYPE html>
<html>
<head>
    <title>XGBoost.js Integration</title>
    <script src="xgboost.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // Initialize and load the model
            const model = new XGBoost({
                learningRate: 0.3,
                maxDepth: 4,
                minChildWeight: 1,
                numRounds: 100
            });

            // Example prediction on user input
            const userInput = [2.5, 3.5];
            const prediction = model.predictSingle(userInput);

            console.log('Prediction:', prediction);
        });
    </script>
</head>
<body>
    <h1>XGBoost.js in Web App</h1>
</body>
</html>
```

## Real-Life Examples

### Predicting Housing Prices

Use XGBoost.js to predict housing prices based on features like size, location, and number of bedrooms.

```javascript
const { XGBoost } = require('./xgboost.js');

// Sample training data
// Each entry in X_train represents a house with two features:
// - Size in square feet
// - Number of bedrooms
const X_train = [
    [1500, 3],
    [1600, 3],
    [1700, 4],
    [1875, 3],
    [1100, 2],
    [1550, 4],
    [2350, 4],
    [2450, 5],
];
// Target prices corresponding to each house in X_train
const y_train = [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000];

// Initialize and train the XGBoost model with specified hyperparameters
const model = new XGBoost({
    learningRate: 0.05,      // Step size shrinkage to prevent overfitting
    maxDepth: 5,             // Maximum depth of a tree
    minChildWeight: 1,       // Minimum sum of instance weight (hessian) needed in a child
    numRounds: 500           // Number of boosting rounds
});
model.fit(X_train, y_train);

// New housing data for prediction
// Each entry in X_new has the same features as the training data:
// - Size in square feet
// - Number of bedrooms
const X_new = [
    [2000, 3],
    [1600, 2],
];
// Predict housing prices for the new data
const predictedPrices = model.predictBatch(X_new);

console.log('Predicted Prices:', predictedPrices);
```

### Customer Churn Prediction

Predict whether customers will churn based on their usage patterns and demographics.

```javascript
const { XGBoost } = require('./xgboost.js');

// Sample data
const X_train = [
    [1, 34, 50000],
    [0, 45, 60000],
    [1, 23, 40000],
    [0, 35, 65000],
    [1, 52, 70000],
    [0, 46, 55000],
];
const y_train = [1, 0, 1, 0, 0, 0];

// Initialize and train the model
const model = new XGBoost({
    learningRate: 0.2,
    maxDepth: 3,
    minChildWeight: 1,
    numRounds: 150
});
model.fit(X_train, y_train);

// Predict churn probabilities
const X_new = [
    [1, 30, 48000],
    [0, 50, 62000],
];
const churnProbabilities = model.predictBatch(X_new);

console.log('Churn Probabilities:', churnProbabilities);
```

## Testing

Utilize the provided `xgboosttest.js` to run comprehensive tests ensuring the reliability and accuracy of your models.

```javascript
// Run tests
const tester = new XGBoostTester();
tester.runTests().catch(console.error);
```

## Conclusion

XGBoost.js offers a robust and flexible solution for integrating gradient boosting algorithms into JavaScript applications. Whether you're building predictive models for web applications, data analysis, or real-time decision-making systems, XGBoost.js provides the tools necessary to implement efficient and accurate machine learning solutions.

For more advanced features and customization, refer to the test suite in `xgboosttest.js` and explore additional methods within the `XGBoost` class.

```
