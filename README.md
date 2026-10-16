# ADVANCED AI TRADING BOT

An advanced Python-based forex trading system that combines machine learning, automated strategy evaluation, probability-based trade selection, and MetaTrader 5 integration to develop and execute algorithmic trading strategies across 19 financial instruments.

The system uses LightGBM multiclass classification to analyze market data and classify potential price movements. It includes a complete workflow for dataset preparation, feature engineering, model training, threshold optimization, forward testing, and live trade execution.

## Overview

The ADVANCED AI TRADING BOT project is designed to explore the application of machine learning to algorithmic forex trading through a structured pipeline that covers model development, strategy evaluation, and automated execution.

The system incorporates 19 individually trained multiclass models, with each model corresponding to a specific trading instrument.

Its major components include:

- LightGBM multiclass machine learning.
- Five-class price movement classification.
- Automated feature engineering and dataset preparation.
- Optuna-based hyperparameter optimization.
- SHAP-based model interpretability.
- Model evaluation and feature selection.
- Probability-based directional trade thresholds.
- Forward testing and strategy performance analysis.
- Automated MetaTrader 5 trade execution.
- Dynamic position sizing based on account balance and risk settings.
- Stop-loss and take-profit management.
- Partial profit-taking and stop-loss adjustment.
- Spread filtering and currency correlation checks.
- Multiprocessing-based execution across multiple instruments.

## Supported Trading Instruments

The system contains 19 trained models covering the following instruments:

| No. | Trading Instrument |
|---:|---|
| 1 | AUDCAD |
| 2 | AUDJPY |
| 3 | AUDUSD |
| 4 | BTCUSD |
| 5 | CADJPY |
| 6 | ETHUSD |
| 7 | EURGBP |
| 8 | EURJPY |
| 9 | EURUSD |
| 10 | GBPCAD |
| 11 | GBPJPY |
| 12 | GBPUSD |
| 13 | NZDJPY |
| 14 | NZDUSD |
| 15 | USDCAD |
| 16 | USDCHF |
| 17 | USDJPY |
| 18 | XAGUSD |
| 19 | XAUUSD |

**Note:** The exact symbol names used by MetaTrader 5 depend on the broker. Some brokers append suffixes or use different symbol naming conventions.

## Project Architecture

The project is organized into three primary directories.

```text
ADVANCED-AI-TRADING-BOT/
│
├── ALL_MODELS/
│   ├── HL_MAIN_TARGET_5CLASS_10M_AUDCAD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_AUDJPY_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_AUDUSD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_BTCUSD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_CADJPY_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_ETHUSD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_EURGBP_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_EURJPY_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_EURUSD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_GBPCAD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_GBPJPY_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_GBPUSD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_NZDJPY_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_NZDUSD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_USDCAD_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_USDCHF_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_USDJPY_model.pkl
│   ├── HL_MAIN_TARGET_5CLASS_10M_XAGUSD_model.pkl
│   └── HL_MAIN_TARGET_5CLASS_10M_XAUUSD_model.pkl
│
├── CSV_FILES/
│   ├── AUDCAD datasets
│   ├── AUDJPY datasets
│   ├── AUDUSD datasets
│   ├── BTCUSD datasets
│   ├── CADJPY datasets
│   ├── ETHUSD datasets
│   ├── EURGBP datasets
│   ├── EURJPY datasets
│   ├── EURUSD datasets
│   ├── GBPCAD datasets
│   ├── GBPJPY datasets
│   ├── GBPUSD datasets
│   ├── NZDJPY datasets
│   ├── NZDUSD datasets
│   ├── USDCAD datasets
│   ├── USDCHF datasets
│   ├── USDJPY datasets
│   ├── XAGUSD datasets
│   └── XAUUSD datasets
│
└── PY_FILES/
    ├── CONNECT_MT5.py
    ├── FOREX_TRADING.py
    ├── func.py
    ├── GET_MT5_DATASET.py
    ├── LOAD_MODEL_INFOS.py
    ├── MT5_DATA_CLEANING.py
    ├── MULTICLASS_FIND_THRESHOLD.py
    ├── MULTICLASS_FORWARD_TEST.py
    ├── MULTICLASS_LIVE_TRADE.py
    ├── MULTICLASS_TRAIN_MODEL.py
    ├── SYMBOL_DATA.py
    └── SYMBOL_POINTS.py
```

The directory structure above illustrates the organization of the project. The exact filenames, file extensions, and dataset names may differ depending on the files included in the repository.

# Core Components

## 1. ALL_MODELS

This directory contains the trained machine learning models used by the trading system.

Each model corresponds to an individual trading instrument and is saved as a serialized Python object.

The model bundles contain the trained estimator, selected input features, and target settings required for prediction and trade management.

The system loads the appropriate model according to the selected instrument and configured timeframe.

## 2. CSV_FILES

This directory contains the historical market datasets used for model development and strategy evaluation.

The datasets support the preparation of training and testing data, feature engineering, model validation, and forward testing.

The quality and representativeness of these datasets are important factors in evaluating the reliability of the resulting trading strategies.

## 3. PY_FILES

This directory contains the Python scripts responsible for data acquisition, preprocessing, model development, evaluation, and trading execution.

### CONNECT_MT5.py

Establishes a connection to MetaTrader 5 for connectivity testing and basic integration checks.

### FOREX_TRADING.py

Acts as the main controller for the trading system.

It coordinates the execution of the live-trading engine and manages the system's overall operation across configured instruments.

### func.py

Contains shared helper functions used throughout the project.

These functions support trading operations, data processing, risk calculations, model prediction, trade management, and other reusable tasks.

### GET_MT5_DATASET.py

Retrieves historical market data from MetaTrader 5 for dataset preparation and analysis.

### LOAD_MODEL_INFOS.py

Loads trained model information for inspection and testing.

It helps examine saved models, their selected features, and their associated target settings.

### MT5_DATA_CLEANING.py

Cleans and structures market data downloaded from MetaTrader 5.

It prepares the raw datasets for subsequent feature engineering, training, and evaluation.

### MULTICLASS_FIND_THRESHOLD.py

Evaluates candidate probability thresholds to identify settings suitable for further trading-strategy evaluation.

Depending on the evaluation configuration, the analysis can include metrics such as win rate, maximum drawdown, and Sharpe ratio.

The objective is to assess how different thresholds affect trade selection and strategy performance.

### MULTICLASS_FORWARD_TEST.py

Performs forward testing using selected threshold settings.

It evaluates simulated trading outcomes and produces information that can be used to examine trade frequency, win rate, drawdown, and other strategy performance characteristics.

Forward-test results should not be interpreted as proof of future profitability or as equivalent to verified live-trading results.

### MULTICLASS_LIVE_TRADE.py

Contains the main live-trading execution logic.

Its responsibilities include:

- Loading the appropriate trained model.
- Obtaining multiclass predictions and class probabilities.
- Combining directional class probabilities.
- Applying configured UP and DOWN probability thresholds.
- Determining movement-based stop-loss and take-profit distances.
- Applying spread filters.
- Checking currency correlation and existing positions.
- Running entry-confirmation logic.
- Calculating position sizes based on configured account risk.
- Placing BUY and SELL orders through MetaTrader 5.
- Monitoring open positions.
- Managing partial profit-taking.
- Adjusting stop-loss levels according to configured trade-management rules.

The engine is designed to work with the shared helper functions in `func.py`.

### MULTICLASS_TRAIN_MODEL.py

Contains the model-development pipeline.

Its responsibilities include:

- Dataset analysis.
- Input feature calculation.
- Feature engineering.
- Five-class target construction.
- Feature selection.
- LightGBM model training.
- Optuna-based hyperparameter optimization.
- SHAP-based feature analysis.
- Model evaluation and validation.
- Saving trained models and associated configuration information.

### SYMBOL_DATA.py

Provides instrument-specific information used by the system, such as balance-related settings, pip information, spread characteristics, minimum distance constraints, and volume-related properties, depending on the implementation.

### SYMBOL_POINTS.py

Retrieves or defines symbol-specific trading properties, including price precision, point size, tick size, and tick value.

These properties support price calculations, feature engineering, and trade-risk calculations.

# Machine Learning Architecture

## LightGBM Multiclass Classification

The system uses LightGBM to classify market observations into five price-movement categories.

Rather than producing only a binary BUY or SELL prediction, the model estimates a probability distribution over the five classes.

## Five-Class Target Structure

| Class | Label | Interpretation |
|---:|---|---|
| 0 | Strong DOWN | Stronger dominant downward excursion |
| 1 | Moderate DOWN | Moderate dominant downward excursion |
| 2 | Small / NEUTRAL | Smaller or ambiguous dominant movement |
| 3 | Moderate UP | Moderate dominant upward excursion |
| 4 | Strong UP | Stronger dominant upward excursion |

The target-generation process examines future price excursions over a configured prediction horizon.

The classification thresholds are derived from the movement distribution using configured quantiles.

The target describes historical future price behavior according to the target-generation rules. It does not guarantee that a particular trade will reach its take-profit before its stop-loss.

## Probability-Based Directional Thresholds

The live-trading engine combines the probabilities of the directional classes.

The UP probability is calculated as:

```python
up_probability = (
    all_class_probabilities[3]
    + all_class_probabilities[4]
)
```

The DOWN probability is calculated as:

```python
down_probability = (
    all_class_probabilities[0]
    + all_class_probabilities[1]
)
```

The neutral class remains separate:

```python
neutral_probability = all_class_probabilities[2]
```

The configured threshold determines whether the combined directional probability qualifies for further trade-entry evaluation.

For example, with a threshold of 0.50:

```python
if up_probability >= 0.50:
    # UP direction qualifies for further evaluation

elif down_probability >= 0.50:
    # DOWN direction qualifies for further evaluation
```

The threshold is a configurable strategy parameter, not a guarantee of trade success.

## Predicted Class and TP/SL Calculation

The system distinguishes between directional probability and the predicted class used to calculate movement-based trade distances.

The combined directional probabilities determine whether a direction meets the configured threshold.

The highest-probability individual class supplies the class-specific median movement percentage used by the existing TP/SL calculation.

For example, when Class 3 is the highest-probability class, the system uses the median movement percentage associated with Class 3, even when Class 4 contributes additional probability to the combined UP score.

This separation allows the directional threshold and the movement-distance calculation to serve different purposes.

The resulting distances are then adjusted using the configured TP and SL ratios.

Actual trading outcomes still depend on execution prices, spread, market behavior, and trade-management rules.

# Strategy Evaluation

The project includes dedicated scripts for threshold evaluation and forward testing.

Depending on the configuration, the analysis can examine:

- Win rate.
- Maximum drawdown.
- Sharpe ratio.
- Number of executed trades.
- Threshold sensitivity.
- Instrument-specific performance.
- Trade timing and duration.
- Differences between strategy configurations.

A high classification score or win rate alone is insufficient to establish profitability. Transaction costs, position sizing, drawdown, execution assumptions, and the consistency of results across unseen data must also be considered.

Threshold selection should be performed without using the final evaluation period to tune the strategy. This helps reduce the risk of overfitting.

# Risk Management and Trade Execution

The live-trading engine includes configurable mechanisms intended to manage execution and trading risk.

These include:

- Position sizing based on configured account risk.
- Stop-loss and take-profit placement.
- Spread-based trade filtering.
- Currency correlation checks.
- Existing-position checks.
- Entry confirmation.
- Partial profit-taking.
- Stop-loss adjustment.
- Trade monitoring.

The behavior of these mechanisms depends on the actual implementation, broker specifications, account conditions, and configured parameters.

Risk controls can reduce certain risks, but they cannot eliminate market risk, slippage, execution failures, or losses.

# Technology Stack

The project uses Python and the following technologies, where applicable to the installed environment:

- Python
- LightGBM
- MetaTrader 5 Python integration
- Pandas
- NumPy
- Scikit-learn
- Optuna
- SHAP
- Joblib

Additional dependencies may be required by individual scripts.

# Installation and Setup

## Prerequisites

Before running the project, prepare the following:

1. A compatible Python environment.
2. A supported MetaTrader 5 terminal installation.
3. A broker account that supports the instruments you intend to trade.
4. The required Python dependencies.
5. The trained models and datasets required by the selected workflow.

The MetaTrader 5 terminal must be configured correctly for the intended account and trading environment.

## Install Dependencies

Create and activate a virtual environment if appropriate for your setup.

Install the packages required by the scripts.

For example:

```bash
pip install MetaTrader5 pandas numpy scikit-learn lightgbm optuna shap joblib
```

Other dependencies may be required depending on the functions implemented in `func.py` and the remaining project scripts.

For reproducibility, use compatible package versions and maintain a dependency file such as `requirements.txt`.

## Configure the Project

Before execution:

1. Verify the directory paths used by the scripts.
2. Confirm that the trained models are stored in the expected location.
3. Ensure the CSV datasets match the expected schemas.
4. Verify the broker's symbol names and suffixes.
5. Check symbol-specific point, tick, volume, and minimum-distance settings.
6. Configure the intended timeframe and trading-risk parameters.
7. Confirm that the MetaTrader 5 terminal is connected to the correct account.

Do not place account passwords, API credentials, or other secrets directly in publicly accessible source files.

## Running the System

The exact execution commands depend on the entry points and import structure used in the project.

The general workflow is:

1. Retrieve historical market data.
2. Clean and prepare the datasets.
3. Train and evaluate the multiclass models.
4. Save the trained models and their settings.
5. Evaluate candidate probability thresholds.
6. Forward-test the selected strategy configuration.
7. Verify the trading logic in a demo or other controlled environment.
8. Start the live-trading controller only after the relevant checks have been completed.

The scripts should be executed from a working directory consistent with their file paths and imports.

# Important Considerations

## Broker Compatibility

Different brokers may use different symbol names, contract specifications, spreads, tick values, and minimum trade volumes.

Verify these properties before using the system with a new broker.

## Model and Dataset Compatibility

A trained model must receive the features expected during training, in the correct order and with compatible data types.

Using a model with incompatible feature calculations or a mismatched dataset can invalidate its predictions.

## Forward Testing Versus Live Trading

Forward testing is useful for evaluating a strategy under defined simulation assumptions, but it does not guarantee that live execution will produce identical results.

Differences can arise from spreads, slippage, market execution, timing, broker rules, and other conditions.

## Model Reproducibility

Model performance can depend on the training dataset, target-generation parameters, selected features, package versions, and training configuration.

Retain the relevant settings and evaluation records when retraining models.

# Disclaimer

This project is provided for educational, research, and software-development purposes.

Forex, cryptocurrency, and precious-metal trading involve substantial financial risk. Machine learning predictions, historical performance, and forward-test results do not guarantee future results or profitability.

The software may place orders through MetaTrader 5 when configured to do so. Users are responsible for verifying the code, broker compatibility, account configuration, risk settings, and execution behavior before using it with real funds.

Use appropriate testing and risk controls. The project is not financial advice, and the author does not guarantee trading performance or protection against financial losses.

# Author

**Ezee Kits (Peter)**

Electrical and Electronics Engineer | Python Developer

Areas of interest:

- Machine Learning
- Algorithmic Trading
- Financial Data Analysis
- Python Automation
- Data Engineering
- Artificial Intelligence

GitHub: [Ezee-Kits](https://github.com/Ezee-Kits)

Portfolio: https://ezee-kits-portfolio.onrender.com/

YouTube: [@EzeeKits](https://www.youtube.com/@EzeeKits)
