# Money Laundering Detection - Streamlit App

A comprehensive Streamlit web application for detecting money laundering transactions using ensemble machine learning models.

## 🚀 Features

- **Multiple Ensemble Methods**: Hybrid voting, majority voting, weighted voting, and soft voting
- **Excel/CSV Data Upload**: Easy data upload with automatic preprocessing
- **Interactive Dashboard**: Real-time visualization of predictions and risk scores
- **Model Comparison**: Compare individual model predictions with ensemble results
- **Downloadable Results**: Export predictions and suspicious transactions
- **Performance Metrics**: Accuracy, precision, recall, F1-score with confusion matrix

## 📋 Prerequisites

### Required Models
Ensure you have the following trained model files in your directory:
- `lightgbm_model_20250721_v1.pkl`
- `randomforest_model_20250721_v1.pkl` 
- `xgboost_model_20250721_v1.pkl`
- `trained_models/scaler.pkl` (optional, will create new one if missing)

### Required Data Format
Your Excel/CSV file should contain these columns:
- `Date`: Transaction date (YYYY-MM-DD format)
- `Time`: Transaction time (HH:MM:SS format) 
- `Sender_account`: Sender account number
- `Receiver_account`: Receiver account number
- `Payment_currency`: Currency of payment
- `Received_currency`: Currency received
- `Sender_bank_location`: Location of sender bank
- `Receiver_bank_location`: Location of receiver bank
- `Payment_type`: Type of payment method
- `Amount`: Transaction amount
- `Is_laundering`: (Optional) Ground truth labels for evaluation

## 🛠️ Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Sample Data** (Optional)
   ```bash
   python generate_test_excel.py
   ```

## 🏃‍♂️ Running the App

### Option 1: Using Python Script
```bash
python run_app.py
```

### Option 2: Direct Streamlit Command
```bash
streamlit run streamlit_app.py
```

### Option 3: Windows Batch File
```bash
run_streamlit.bat
```

## 📊 Using the App

### 1. Configuration
- **Select Ensemble Method**: Choose from hybrid, majority, weighted, or soft voting
- **Adjust Model Weights**: Fine-tune individual model contributions (for weighted/hybrid methods)

### 2. Data Upload
- **Upload File**: Select your Excel or CSV file
- **Generate Sample**: Create test data using the built-in generator
- **Preview Data**: Review uploaded data structure and statistics

### 3. Analysis
- **View Predictions**: See risk scores and binary predictions for each transaction
- **Interactive Visualizations**: 
  - Risk score distribution histogram
  - Prediction summary pie chart
  - Individual model comparison lines
- **Filter Results**: Show only suspicious, safe, high-risk, or medium-risk transactions

### 4. Results Export
- **Download Full Results**: Complete prediction results with risk scores
- **Download Suspicious Only**: Filter for flagged transactions only
- **Performance Metrics**: If ground truth available, view model performance

## 🎯 Ensemble Methods

### Hybrid Voting (Recommended)
- **Combines**: Soft voting (probability averaging) + Hard voting (binary majority)
- **Weights**: Configurable for both approach combination and individual models
- **Threshold**: Adaptive threshold for final classification

### Majority Voting
- **Method**: Simple majority rule (≥2 out of 3 models predict positive)
- **Best for**: Balanced datasets with clear decision boundaries

### Weighted Majority Voting  
- **Method**: Weighted combination based on model performance
- **Best for**: When you know relative model strengths

### Majority Soft Voting
- **Method**: Probability-based majority with configurable threshold
- **Best for**: Fine-tuned threshold optimization

## 📈 Interpreting Results

### Risk Scores
- **0-10%**: Very Low Risk (Safe)
- **10-30%**: Low Risk (Safe) 
- **30-60%**: Medium Risk (Investigate)
- **60-100%**: High Risk (Suspicious)

### Ensemble Benefits
- **Reduced False Positives**: Multiple models reduce individual model errors
- **Improved Robustness**: Less sensitive to individual model weaknesses
- **Higher Confidence**: Consensus predictions are more reliable

## 🔧 Customization

### Model Weights
Adjust in the sidebar based on your model validation performance:
```python
# Example: If LightGBM performs best
weights = [0.5, 0.3, 0.2]  # [LightGBM, RandomForest, XGBoost]
```

### Thresholds
Modify threshold values in the code:
```python
# Hybrid voting threshold
threshold = 0.3  # Lower = more sensitive

# Individual model thresholds
MODEL_THRESHOLDS = {
    "randomforest": 0.3,
    "xgboost": 0.7, 
    "lightgbm": 0.6,
}
```

## 🐛 Troubleshooting

### Common Issues

1. **"No trained models found"**
   - Ensure model files are in the correct directory
   - Check file names match expected patterns

2. **"Error in preprocessing"**
   - Verify your data has all required columns
   - Check date/time formats are correct
   - Remove any completely empty rows

3. **"Scaler not found"**
   - App will create new scaler automatically
   - For consistent results, train and save scaler with models

4. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Ensure you're in the correct directory

### Performance Tips
- **Large files**: Consider processing in batches for very large datasets
- **Memory usage**: Close other applications if processing many transactions
- **Speed**: Hybrid voting is fastest, individual model comparison takes longer

## 📁 File Structure
```
├── streamlit_app.py           # Main Streamlit application
├── generate_test_excel.py     # Sample data generator
├── run_app.py                 # App launcher script
├── run_streamlit.bat          # Windows batch launcher
├── requirements.txt           # Python dependencies
├── src/
│   └── ensemble.py           # Ensemble voting methods
├── data/
│   ├── test_data.xlsx        # Generated sample data
│   └── test_data.csv         # Sample data (CSV backup)
└── trained_models/           # Model files directory
    ├── lightgbm_model_20250721_v1.pkl
    ├── randomforest_model_20250721_v1.pkl
    ├── xgboost_model_20250721_v1.pkl
    └── scaler.pkl
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## 📄 License

This project is for educational and research purposes in financial crime detection.
