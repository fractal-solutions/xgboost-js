# xgboost-js

A pure JavaScript implementation of XGBoost for both Node.js and browser environments.

## Installation

```bash
npm install @fractal-solutions/xgboost-js
```

## Usage

```javascript
const { XGBoost } = require('@fractal-solutions/xgboost-js');

// Initialize the model
const model = new XGBoost({
    learningRate: 0.3,
    maxDepth: 4,
    minChildWeight: 1,
    numRounds: 100
});

// Train the model
model.fit(X_train, y_train);

// Make predictions
const predictions = model.predictBatch(X_test);
```

## Features

- Binary classification
- Model serialization
- Feature importance calculation
- Comprehensive test suite
- Pure JavaScript implementation
- Works in both Node.js and browser environments

## Documentation

For full documentation, see [docs/xgboost.md](docs/xgboost.md)

## License

MIT © Fractal Solutions 
