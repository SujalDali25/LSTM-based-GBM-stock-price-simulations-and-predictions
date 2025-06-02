# LSTM-based-GBM-stock-price-simulations-and-predictions

A hybrid machine learning model combining LSTM with multi-head attention and Geometric Brownian Motion simulation to predict stock price, volatility, and drift using advanced technical indicators.

Overview
This project trains a deep learning model on 5 years of historical stock data, enriched with over 20 technical features, to simultaneously predict future stock price, volatility, and drift. It integrates these predictions into an enhanced GBM simulation, providing more realistic future price paths than traditional models.

Features
Fetches and cleans stock data from Yahoo Finance

Computes a rich set of technical indicators (RSI, MACD, Bollinger Bands, moving averages, etc.)

Prepares sequences for LSTM input

Uses a multi-layer LSTM with multi-head attention for improved temporal feature extraction

Predicts price, volatility, and drift jointly

Runs ML-driven GBM simulations for probabilistic future price paths

Visualizes training, prediction accuracy, volatility/drift, risk metrics, and technical analysis charts


Usage
Modify the ticker symbol and hyperparameters in main.py if needed

Run the training and evaluation:
python main.py

The script outputs training logs, performance metrics, and generates detailed plots

Output
![prediction price](https://github.com/user-attachments/assets/23362467-9dd4-4698-8130-7b037addc5d1)
Predicted Price
The model predicts next-day stock prices by learning complex temporal patterns from a rich set of technical indicators. You will see plots comparing actual historical prices against the model’s predicted prices, showing how well the model fits and forecasts future values.

![simulations](https://github.com/user-attachments/assets/93862422-a237-4ea8-acee-991b0d1754a9)
Simulations
Using the predicted volatility and drift, the project runs enhanced Geometric Brownian Motion (GBM) simulations to generate multiple possible future price paths. This approach provides a probabilistic view of potential stock price trajectories rather than a single deterministic forecast.

The ML-enhanced GBM uses dynamic parameters predicted by the model, adapting the volatility and drift over time.

For comparison, traditional GBM simulations with static historical parameters are also generated.

Visualizations include confidence intervals and percentile ranges to communicate uncertainty and risk.

These outputs together help investors and analysts understand both expected price movements and the range of possible outcomes, supporting more informed decision-making.

Project Structure
main.py — Entry point for training and analysis

model.py — Hybrid LSTM-attention model definition

data_utils.py — Data fetching and feature engineering

simulation.py — GBM simulation functions

visualization.py — Plotting and results visualization

requirements.txt — Required Python packages

Results
Training loss curves

Actual vs predicted price comparisons

Error distribution histograms

Predicted volatility and drift scatter plots

Comparison of traditional and ML-enhanced GBM simulations

Risk percentile analysis and profit probabilities

Technical indicator visualizations for price, volume, RSI, and volatility

Limitations
Assumes clean, sufficient historical data

Volatility and drift predictions require further validation

Model hyperparameters are fixed; tuning may improve performance

Not designed for real-time or multi-stock batch inference yet

Future Work
Hyperparameter tuning and automated search

Explainability and attention visualization

Multi-stock and portfolio-level modeling

Real-time data streaming and deployment via API or cloud

Additional validation metrics for volatility and drift
