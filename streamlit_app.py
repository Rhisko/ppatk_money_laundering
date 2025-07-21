import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import ensemble methods
from src.ensemble import hybrid_voting, majority_voting, weighted_majority_voting, majority_soft_voting

# Set page config
st.set_page_config(
    page_title="Money Laundering Detection - Ensemble Models",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.suspicious-alert {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.safe-alert {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load trained models"""
    models = {}
    model_files = {
        'lightgbm': 'lightgbm_model_20250721_v1.pkl',
        'randomforest': 'randomforest_model_20250721_v1.pkl', 
        'xgboost': 'xgboost_model_20250721_v1.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = os.path.join('trained_models', filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
        else:
            # Try alternative path
            if os.path.exists(filename):
                models[name] = joblib.load(filename)
    
    return models

@st.cache_data
def load_scaler():
    """Load the trained scaler"""
    scaler_path = 'trained_models/scaler.pkl'
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        st.warning("Scaler not found. Using default StandardScaler.")
        return StandardScaler()

def preprocess_uploaded_data(df):
    """Preprocess uploaded data similar to training data"""
    try:
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = df_processed.dropna()
        
        # Feature Engineering: Extract Time features
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
        df_processed['weekday'] = df_processed['Date'].dt.weekday
        df_processed['hour'] = pd.to_datetime(df_processed['Time'], format='%H:%M:%S').dt.hour
        
        # Drop unnecessary columns
        df_processed = df_processed.drop(['Time', 'Date'], axis=1)
        
        # Log transformation for 'Amount'
        df_processed['log_amount'] = np.log1p(df_processed['Amount'])
        df_processed = df_processed.drop(['Amount'], axis=1)
        
        # Encode categorical variables
        categorical_cols = [
            'Sender_account', 'Receiver_account',
            'Payment_currency', 'Received_currency', 
            'Sender_bank_location', 'Receiver_bank_location',
            'Payment_type'
        ]
        
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        # Select features used in training
        feature_cols = [
            'Sender_account', 'Receiver_account',
            'Payment_currency', 'Received_currency',
            'Sender_bank_location', 'Receiver_bank_location', 
            'Payment_type', 'weekday', 'hour', 'log_amount'
        ]
        
        X = df_processed[feature_cols]
        
        # Get target if exists
        y = None
        if 'Is_laundering' in df_processed.columns:
            y = df_processed['Is_laundering'].astype(int)
        
        return X, y
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None

def predict_ensemble(models, X, ensemble_method='hybrid', weights=None):
    """Make predictions using ensemble of models"""
    predictions = []
    probabilities = []
    
    # Get predictions from each model
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)[:, 1]  # Get probability of positive class
            else:
                prob = model.predict(X)
            probabilities.append(prob)
            
            # Binary predictions
            pred = (prob > 0.5).astype(int)
            predictions.append(pred)
            
        except Exception as e:
            st.error(f"Error predicting with {name}: {str(e)}")
            continue
    
    if not probabilities:
        return None, None, None
    
    # Apply ensemble method
    if ensemble_method == 'majority':
        ensemble_pred = majority_voting(predictions)
        ensemble_prob = np.mean(probabilities, axis=0)
    elif ensemble_method == 'weighted':
        if weights is None:
            weights = [1/len(models)] * len(models)
        ensemble_pred = weighted_majority_voting(predictions, weights)
        ensemble_prob = np.average(probabilities, axis=0, weights=weights)
    elif ensemble_method == 'hybrid':
        if weights is None:
            weights = [0.3, 0.5, 0.3]  # Default weights
        ensemble_pred, ensemble_prob = hybrid_voting(
            probabilities, threshold=0.3, weights=weights, return_score=True
        )
    else:  # majority_soft
        ensemble_pred = majority_soft_voting(probabilities, threshold=0.3)
        ensemble_prob = np.mean(probabilities, axis=0)
    
    return ensemble_pred, ensemble_prob, probabilities

def main():
    st.markdown('<h1 class="main-header">🏦 Money Laundering Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Ensemble Model Testing with Excel Data Upload")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
        scaler = load_scaler()
    
    if not models:
        st.error("No trained models found! Please ensure model files are in the correct directory.")
        st.stop()
    
    st.sidebar.success(f"✅ Loaded {len(models)} models: {', '.join(models.keys())}")
    
    # Ensemble method selection
    ensemble_method = st.sidebar.selectbox(
        "Select Ensemble Method:",
        ['hybrid', 'majority', 'weighted', 'majority_soft'],
        help="Choose the ensemble voting method for combining model predictions"
    )
    
    # Model weights (for weighted ensemble)
    if ensemble_method == 'weighted' or ensemble_method == 'hybrid':
        st.sidebar.subheader("Model Weights")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            w_lgb = st.slider("LightGBM", 0.0, 1.0, 0.3, 0.1)
        with col2:
            w_rf = st.slider("Random Forest", 0.0, 1.0, 0.5, 0.1)
        with col3:
            w_xgb = st.slider("XGBoost", 0.0, 1.0, 0.3, 0.1)
        weights = [w_lgb, w_rf, w_xgb]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        st.sidebar.text(f"Normalized: {[f'{w:.2f}' for w in weights]}")
    else:
        weights = None
    
    # File upload
    st.subheader("📁 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your transaction data for money laundering detection"
    )
    
    # Sample data generation
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("💡 **Need test data?** Use the button below to generate sample data, or upload your own Excel/CSV file.")
    with col2:
        if st.button("🎲 Generate Sample Data", type="secondary"):
            with st.spinner("Generating sample data..."):
                # Generate and save sample data
                os.system("python generate_test_excel.py")
                st.success("✅ Sample data generated! Check `data/test_data.xlsx`")
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} rows of data")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                if 'Is_laundering' in df.columns:
                    suspicious_count = df['Is_laundering'].sum()
                    st.metric("Known Suspicious", suspicious_count)
                else:
                    st.metric("Known Labels", "Not Available")
            with col3:
                st.metric("Features", len(df.columns))
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(10))
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                X, y = preprocess_uploaded_data(df)
            
            if X is None:
                st.error("Failed to preprocess data. Please check your file format.")
                st.stop()
            
            # Scale features
            try:
                X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
            except:
                # If scaler fails, fit new one
                X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
                st.warning("Used new scaler - results may differ from training.")
            
            # Make predictions
            st.subheader("🔍 Model Predictions")
            
            with st.spinner("Running ensemble prediction..."):
                ensemble_pred, ensemble_prob, individual_probs = predict_ensemble(
                    models, X_scaled, ensemble_method, weights
                )
            
            if ensemble_pred is None:
                st.error("Failed to make predictions")
                st.stop()
            
            # Results
            suspicious_mask = ensemble_pred == 1
            suspicious_count = np.sum(suspicious_mask)
            safe_count = len(ensemble_pred) - suspicious_count
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚨 Suspicious Detected", suspicious_count)
            with col2:
                st.metric("✅ Safe Transactions", safe_count)
            with col3:
                detection_rate = (suspicious_count / len(df)) * 100
                st.metric("Detection Rate", f"{detection_rate:.1f}%")
            with col4:
                avg_risk_score = np.mean(ensemble_prob) * 100
                st.metric("Avg Risk Score", f"{avg_risk_score:.1f}%")
            
            # Visualizations
            st.subheader("📈 Analysis Dashboard")
            
            # Risk score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    x=ensemble_prob * 100,
                    bins=20,
                    title="Risk Score Distribution",
                    labels={'x': 'Risk Score (%)', 'y': 'Count'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.add_vline(
                    x=30, line_dash="dash", line_color="red",
                    annotation_text="Threshold (30%)"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Prediction summary
                pred_summary = pd.DataFrame({
                    'Category': ['Safe', 'Suspicious'],
                    'Count': [safe_count, suspicious_count]
                })
                fig_pie = px.pie(
                    pred_summary, values='Count', names='Category',
                    title="Prediction Summary",
                    color_discrete_map={'Safe': '#4caf50', 'Suspicious': '#f44336'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Individual model comparison
            if len(individual_probs) == 3:
                model_names = ['LightGBM', 'Random Forest', 'XGBoost']
                model_df = pd.DataFrame({
                    'Transaction': range(len(df)),
                    'LightGBM': individual_probs[0] * 100,
                    'Random Forest': individual_probs[1] * 100,
                    'XGBoost': individual_probs[2] * 100,
                    'Ensemble': ensemble_prob * 100
                })
                
                fig_comparison = px.line(
                    model_df.melt(id_vars=['Transaction'], var_name='Model', value_name='Risk_Score'),
                    x='Transaction', y='Risk_Score', color='Model',
                    title="Model Risk Score Comparison",
                    labels={'Risk_Score': 'Risk Score (%)'}
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Detailed results
            st.subheader("📋 Detailed Results")
            
            # Prepare results dataframe
            results_df = df.copy()
            results_df['Risk_Score'] = ensemble_prob * 100
            results_df['Prediction'] = ['Suspicious' if p == 1 else 'Safe' for p in ensemble_pred]
            results_df['Risk_Level'] = pd.cut(
                ensemble_prob * 100, 
                bins=[0, 10, 30, 60, 100], 
                labels=['Very Low', 'Low', 'Medium', 'High']
            )
            
            # Add individual model scores
            if len(individual_probs) == 3:
                results_df['LightGBM_Score'] = individual_probs[0] * 100
                results_df['RandomForest_Score'] = individual_probs[1] * 100
                results_df['XGBoost_Score'] = individual_probs[2] * 100
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                show_filter = st.selectbox(
                    "Filter Results:",
                    ['All', 'Suspicious Only', 'Safe Only', 'High Risk (>60%)', 'Medium Risk (30-60%)']
                )
            with col2:
                sort_by = st.selectbox(
                    "Sort by:",
                    ['Risk_Score', 'Amount', 'Date'],
                    index=0
                )
            
            # Apply filters
            if show_filter == 'Suspicious Only':
                filtered_df = results_df[results_df['Prediction'] == 'Suspicious']
            elif show_filter == 'Safe Only':
                filtered_df = results_df[results_df['Prediction'] == 'Safe']
            elif show_filter == 'High Risk (>60%)':
                filtered_df = results_df[results_df['Risk_Score'] > 60]
            elif show_filter == 'Medium Risk (30-60%)':
                filtered_df = results_df[(results_df['Risk_Score'] >= 30) & (results_df['Risk_Score'] <= 60)]
            else:
                filtered_df = results_df
            
            # Sort results
            if sort_by in filtered_df.columns:
                filtered_df = filtered_df.sort_values(sort_by, ascending=False)
            
            st.dataframe(
                filtered_df.style.format({
                    'Risk_Score': '{:.1f}%',
                    'LightGBM_Score': '{:.1f}%' if 'LightGBM_Score' in filtered_df.columns else None,
                    'RandomForest_Score': '{:.1f}%' if 'RandomForest_Score' in filtered_df.columns else None,
                    'XGBoost_Score': '{:.1f}%' if 'XGBoost_Score' in filtered_df.columns else None,
                }).background_gradient(subset=['Risk_Score'], cmap='Reds'),
                use_container_width=True
            )
            
            # Download results
            st.subheader("💾 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Full Results (CSV)",
                    data=csv_data,
                    file_name=f"money_laundering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                suspicious_only = results_df[results_df['Prediction'] == 'Suspicious']
                if len(suspicious_only) > 0:
                    suspicious_csv = suspicious_only.to_csv(index=False)
                    st.download_button(
                        label="🚨 Download Suspicious Only (CSV)",
                        data=suspicious_csv,
                        file_name=f"suspicious_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # Model performance (if ground truth available)
            if y is not None:
                st.subheader("🎯 Model Performance")
                from sklearn.metrics import classification_report, confusion_matrix
                
                # Calculate metrics
                accuracy = np.mean(ensemble_pred == y)
                precision = np.sum((ensemble_pred == 1) & (y == 1)) / np.sum(ensemble_pred == 1) if np.sum(ensemble_pred == 1) > 0 else 0
                recall = np.sum((ensemble_pred == 1) & (y == 1)) / np.sum(y == 1) if np.sum(y == 1) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.3f}")
                with col2:
                    st.metric("Precision", f"{precision:.3f}")
                with col3:
                    st.metric("Recall", f"{recall:.3f}")
                with col4:
                    st.metric("F1-Score", f"{f1:.3f}")
                
                # Confusion matrix
                cm = confusion_matrix(y, ensemble_pred)
                fig_cm = px.imshow(
                    cm, text_auto=True, aspect="auto",
                    title="Confusion Matrix",
                    labels={'x': 'Predicted', 'y': 'Actual'},
                    x=['Safe', 'Suspicious'], y=['Safe', 'Suspicious']
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure your file has the required columns and format.")

if __name__ == "__main__":
    main()
