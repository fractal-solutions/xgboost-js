const { XGBoost } = require('./xgboost.js');  

// Extended test suite for XGBoost implementation


class XGBoostTester {
    constructor() {
        this.testResults = [];
    }

    // Utility function to generate different types of synthetic datasets
    generateDataset(type = 'binary', n = 1000) {
        switch(type) {
            case 'binary':
                return this.generateBinaryData(n);
            case 'nonlinear':
                return this.generateNonlinearData(n);
            case 'noisy':
                return this.generateNoisyData(n);
            default:
                throw new Error('Unknown dataset type');
        }
    }

    generateBinaryData(n) {
        const X = [];
        const y = [];
        for (let i = 0; i < n; i++) {
            const x1 = Math.random() * 10;
            const x2 = Math.random() * 10;
            X.push([x1, x2]);
            y.push(x1 + x2 > 10 ? 1 : 0);
        }
        return [X, y];
    }

    generateNonlinearData(n) {
        const X = [];
        const y = [];
        for (let i = 0; i < n; i++) {
            const x1 = Math.random() * 10 - 5;
            const x2 = Math.random() * 10 - 5;
            X.push([x1, x2]);
            // Circular decision boundary
            y.push(x1 * x1 + x2 * x2 < 16 ? 1 : 0);
        }
        return [X, y];
    }

    generateNoisyData(n) {
        const X = [];
        const y = [];
        for (let i = 0; i < n; i++) {
            const x1 = Math.random() * 10;
            const x2 = Math.random() * 10;
            X.push([x1, x2]);
            // Add 10% noise to labels
            const trueLabel = x1 + x2 > 10 ? 1 : 0;
            y.push(Math.random() < 0.1 ? 1 - trueLabel : trueLabel);
        }
        return [X, y];
    }

    calculateMetrics(predictions, actual) {
        const binaryPreds = predictions.map(p => p >= 0.5 ? 1 : 0);
        let tp = 0, fp = 0, tn = 0, fn = 0;

        for (let i = 0; i < actual.length; i++) {
            if (actual[i] === 1 && binaryPreds[i] === 1) tp++;
            if (actual[i] === 0 && binaryPreds[i] === 1) fp++;
            if (actual[i] === 0 && binaryPreds[i] === 0) tn++;
            if (actual[i] === 1 && binaryPreds[i] === 0) fn++;
        }

        const accuracy = (tp + tn) / (tp + tn + fp + fn);
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;

        return { accuracy, precision, recall, f1 };
    }

    async runTests() {
        await this.testBasicClassification();
        //await this.testNonlinearClassification();
        //await this.testNoisyData();
        await this.testModelSerialization();
        await this.testFeatureImportance();
        this.printResults();
    }

    async testBasicClassification() {
        try {
            const [X_train, y_train] = this.generateDataset('binary', 1000);
            const [X_test, y_test] = this.generateDataset('binary', 200);

            const model = new XGBoost({
                learningRate: 0.3,
                maxDepth: 4,
                minChildWeight: 1,
                numRounds: 100
            });

            model.fit(X_train, y_train);
            const predictions = model.predictBatch(X_test);
            const metrics = this.calculateMetrics(predictions, y_test);

            this.testResults.push({
                name: 'Basic Classification',
                passed: metrics.accuracy > 0.7,
                metrics
            });
        } catch (error) {
            this.testResults.push({
                name: 'Basic Classification',
                passed: false,
                error: error.message
            });
        }
    }

    async testModelSerialization() {
        try {
            const [X_train, y_train] = this.generateDataset('binary', 100);
            const model = new XGBoost({ numRounds: 10 });
            model.fit(X_train, y_train);

            // Test serialization
            const serialized = model.toJSON();
            const deserializedModel = XGBoost.fromJSON(serialized);

            // Compare predictions
            const original_preds = model.predictBatch(X_train);
            const deserialized_preds = deserializedModel.predictBatch(X_train);

            const arePredsEqual = original_preds.every((pred, i) => 
                Math.abs(pred - deserialized_preds[i]) < 1e-6
            );

            this.testResults.push({
                name: 'Model Serialization',
                passed: arePredsEqual,
                details: arePredsEqual ? 'Predictions match after serialization' : 'Predictions differ'
            });
        } catch (error) {
            this.testResults.push({
                name: 'Model Serialization',
                passed: false,
                error: error.message
            });
        }
    }

    async testFeatureImportance() {
        try {
            const [X_train, y_train] = this.generateDataset('binary', 100);
            const model = new XGBoost({ 
                numRounds: 10,
                learningRate: 0.3,
                maxDepth: 3,
                minChildWeight: 1
            });
            
            model.fit(X_train, y_train);
            const importance = model.getFeatureImportance();
            
            // We know we have 2 features in our synthetic data
            const isValid = importance.length === 2 && 
                           importance.every(v => typeof v === 'number');

            this.testResults.push({
                name: 'Feature Importance',
                passed: isValid,
                details: `Feature importance scores: ${importance.join(', ')}`
            });
        } catch (error) {
            this.testResults.push({
                name: 'Feature Importance',
                passed: false,
                error: error.message
            });
        }
    }

    printResults() {
        console.log('\n=== XGBoost Test Results ===\n');
        this.testResults.forEach(result => {
            console.log(`${result.name}: ${result.passed ? '✅ PASSED' : '❌ FAILED'}`);
            if (result.metrics) {
                console.log('Metrics:', {
                    accuracy: result.metrics.accuracy.toFixed(4),
                    precision: result.metrics.precision.toFixed(4),
                    recall: result.metrics.recall.toFixed(4),
                    f1: result.metrics.f1.toFixed(4)
                });
            }
            if (result.details) {
                console.log('Details:', result.details);
            }
            if (result.error) {
                console.log('Error:', result.error);
            }
            console.log('---');
        });
    }
}

// Run the tests
const tester = new XGBoostTester();
tester.runTests().catch(console.error); 