# Stock Price Prediction with LSTM Neural Network

This Jupyter Notebook demonstrates how to predict stock closing prices using a Long Short-Term Memory (LSTM) neural network in PyTorch. The workflow covers data loading, preprocessing, model building, training, evaluation, and visualization.

---

## Features

- **Data Loading:**
  Fetches historical stock data for a user-specified ticker using the Stooq data source.

- **Data Preprocessing:**

  - Sorts data by date.
  - Scales closing prices using `StandardScaler` for better neural network performance.
  - Creates overlapping sliding windows of past closing prices for time series modeling.

- **Model Architecture:**

  - Defines an LSTM-based neural network with configurable hidden layers and dropout.
  - Uses a fully connected layer to output the predicted closing price.

- **Training:**

  - Splits data into training and test sets (80/20).
  - Trains the model using Mean Squared Error loss and the Adam optimizer.
  - Tracks and prints training loss and timing.

- **Evaluation:**

  - Makes predictions on the test set.
  - Inverse-transforms predictions and actual values to original price scale.
  - Calculates and prints Root Mean Squared Error (RMSE) for both train and test sets.

- **Visualization:**
  - Plots actual vs. predicted prices for the test set.
  - Shows prediction variance and RMSE on a separate subplot.
  - X-axis can be set to display monthly or yearly ticks for clarity.

---

## How to Use

1. **Set the Ticker:**
   Change the `ticker` variable near the top of the notebook to the stock symbol you want to predict (e.g., `'AAPL'`).

2. **Run All Cells:**
   Execute each cell in order. The notebook will:

   - Download and preprocess the data.
   - Train the LSTM model.
   - Evaluate and visualize the results.

3. **Adjust Parameters (Optional):**
   - Change `seq_length` to use a different window size.
   - Modify LSTM parameters (hidden size, number of layers, dropout) in the `PredictionModel` class.
   - Adjust training parameters (number of epochs, learning rate, etc.).

---

## Requirements

- Python 3.x
- Jupyter Notebook
- PyTorch
- pandas, numpy, matplotlib, scikit-learn
- pandas_datareader, yfinance

Install requirements with:

```bash
pip install torch pandas numpy matplotlib scikit-learn pandas_datareader yfinance
```

---

## Notes

- GPU acceleration is used if available (CUDA, MPS, or ROCm).
- The notebook uses normalized data for training and inverse-transforms predictions for interpretability.
- The model predicts the next day's closing price based on a window of previous days.

---

## Example Output

- **Line plot** of actual vs. predicted closing prices for the test period.
- **Variance plot** showing the difference between predictions and actual prices, with RMSE reference.

---

## License

This notebook is for educational purposes.
Feel free to modify and use it for your own stock prediction experiments!
