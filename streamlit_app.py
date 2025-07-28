import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import defaultdict, Counter

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
        'lightgbm': 'lightgbm_model_20250721_v2.pkl',
        'randomforest': 'randomforest_model_20250721_v2.pkl',
        'xgboost': 'xgboost_model_20250721_v2.pkl'
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

def predict_ensemble(models, X, ensemble_method='hybrid', weights=None, model_thresholds=None):
    """Make predictions using ensemble of models following main.py logic"""
    predictions = []
    probabilities = []
    individual_scores = {}
    
    # Default model thresholds from main.py
    if model_thresholds is None:
        model_thresholds = {
            "randomforest": 0.3,
            "xgboost": 0.7,
            "lightgbm": 0.6,
        }
    
    # Get predictions from each model
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)[:, 1]  # Get probability of positive class
            else:
                prob = model.predict(X)
            
            probabilities.append(prob)
            individual_scores[f'{name}_score'] = prob
            
            # Binary predictions using model-specific thresholds
            threshold = model_thresholds.get(name, 0.5)
            pred = (prob > threshold).astype(int)
            predictions.append(pred)
            
        except Exception as e:
            st.error(f"Error predicting with {name}: {str(e)}")
            continue
    
    if not probabilities:
        return None, None, None, None
    
    # Apply ensemble method following main.py logic
    if ensemble_method == 'majority':
        ensemble_pred = majority_voting(predictions)
        ensemble_scores = np.mean(probabilities, axis=0)
    elif ensemble_method == 'weighted':
        if weights is None:
            weights = [1/len(models)] * len(models)
        ensemble_pred = weighted_majority_voting(predictions, weights)
        ensemble_scores = np.average(probabilities, axis=0, weights=weights)
    elif ensemble_method == 'hybrid':
        if weights is None:
            weights = [0.3, 0.5, 0.3]  # Default weights from main.py
        # Use hybrid_voting with return_score=True and auto_tune=True like main.py
        ensemble_pred, ensemble_scores = hybrid_voting(
            probabilities, threshold=0.3, weights=weights, return_score=True, auto_tune=True
        )
    else:  # majority_soft
        ensemble_pred = majority_soft_voting(probabilities, threshold=0.3)
        ensemble_scores = np.mean(probabilities, axis=0)
    
    return ensemble_pred, ensemble_scores, probabilities, individual_scores

def create_transaction_network(df, risk_scores, min_risk_threshold=30):
    """Create network graph from transaction data"""
    # Filter transactions above risk threshold
    high_risk_mask = risk_scores >= (min_risk_threshold / 100)
    df_filtered = df[high_risk_mask].copy()
    
    if len(df_filtered) == 0:
        return None, None, None
    
    # Create network graph
    G = nx.Graph()
    
    # Track entity statistics
    entity_stats = defaultdict(lambda: {
        'transactions': 0, 
        'total_amount': 0, 
        'avg_risk': 0, 
        'max_risk': 0,
        'currencies': set(),
        'locations': set(),
        'payment_types': set()
    })
    
    # Add nodes and edges
    for idx, row in df_filtered.iterrows():
        sender = f"S_{row.get('Sender_account', 'Unknown')}"
        receiver = f"R_{row.get('Receiver_account', 'Unknown')}"
        amount = row.get('Amount', 0)
        risk = risk_scores[idx] * 100
        
        # Add nodes if not exist
        if not G.has_node(sender):
            G.add_node(sender, type='sender', label=sender.replace('S_', ''))
        if not G.has_node(receiver):
            G.add_node(receiver, type='receiver', label=receiver.replace('R_', ''))
        
        # Add edge with transaction details
        if G.has_edge(sender, receiver):
            # Update existing edge
            G[sender][receiver]['weight'] += amount
            G[sender][receiver]['transactions'] += 1
            G[sender][receiver]['avg_risk'] = (G[sender][receiver]['avg_risk'] + risk) / 2
            G[sender][receiver]['max_risk'] = max(G[sender][receiver]['max_risk'], risk)
        else:
            # Create new edge
            G.add_edge(sender, receiver, 
                      weight=amount, 
                      transactions=1, 
                      avg_risk=risk, 
                      max_risk=risk)
        
        # Update entity statistics
        for entity in [sender, receiver]:
            entity_stats[entity]['transactions'] += 1
            entity_stats[entity]['total_amount'] += amount
            entity_stats[entity]['avg_risk'] = (entity_stats[entity]['avg_risk'] + risk) / 2
            entity_stats[entity]['max_risk'] = max(entity_stats[entity]['max_risk'], risk)
            
            if 'Payment_currency' in row:
                entity_stats[entity]['currencies'].add(str(row['Payment_currency']))
            if 'Sender_bank_location' in row:
                entity_stats[entity]['locations'].add(str(row['Sender_bank_location']))
            if 'Payment_type' in row:
                entity_stats[entity]['payment_types'].add(str(row['Payment_type']))
    
    return G, entity_stats, df_filtered

def create_network_visualization(G, entity_stats):
    """Create interactive network visualization using Plotly"""
    if G is None or len(G.nodes()) == 0:
        return None
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        edge_data = G[edge[0]][edge[1]]
        edge_info.append(
            f"From: {edge[0].replace('S_', '').replace('R_', '')}<br>"
            f"To: {edge[1].replace('S_', '').replace('R_', '')}<br>"
            f"Transactions: {edge_data['transactions']}<br>"
            f"Total Amount: ${edge_data['weight']:,.2f}<br>"
            f"Avg Risk: {edge_data['avg_risk']:.1f}%<br>"
            f"Max Risk: {edge_data['max_risk']:.1f}%"
        )
    
    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_color = []
    node_size = []
    node_symbol = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node info
        stats = entity_stats[node]
        node_text.append(node.replace('S_', '').replace('R_', ''))
        
        node_info.append(
            f"Entity: {node.replace('S_', '').replace('R_', '')}<br>"
            f"Type: {'Sender' if node.startswith('S_') else 'Receiver'}<br>"
            f"Transactions: {stats['transactions']}<br>"
            f"Total Amount: ${stats['total_amount']:,.2f}<br>"
            f"Avg Risk: {stats['avg_risk']:.1f}%<br>"
            f"Max Risk: {stats['max_risk']:.1f}%<br>"
            f"Currencies: {len(stats['currencies'])}<br>"
            f"Locations: {len(stats['locations'])}"
        )
        
        # Color based on risk level
        risk_score = stats['max_risk']
        if risk_score >= 80:
            node_color.append('#d32f2f')  # Red - Critical
        elif risk_score >= 60:
            node_color.append('#f57c00')  # Orange - High
        elif risk_score >= 40:
            node_color.append('#fbc02d')  # Yellow - Medium
        else:
            node_color.append('#388e3c')  # Green - Low
        
        # Size based on transaction volume
        node_size.append(max(10, min(50, stats['transactions'] * 3)))
        
        # Symbol based on type
        node_symbol.append('circle' if node.startswith('S_') else 'square')
    
    # Node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white'),
            symbol=node_symbol
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='AML Network Analysis - Entity Relationships',
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Node size = transaction volume | Color = risk level | Circle = Sender | Square = Receiver",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color='gray', size=10)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                    ))
    
    return fig

def create_risk_analytics_dashboard(df, ensemble_scores, ensemble_pred):
    """Create comprehensive risk analytics dashboard"""
    results_df = df.copy()
    results_df['Risk_Score'] = ensemble_scores * 100
    results_df['Prediction'] = ensemble_pred
    results_df['Risk_Level'] = pd.cut(
        ensemble_scores * 100, 
        bins=[0, 30, 50, 70, 100], 
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Risk vs Amount Analysis', 'Transaction Volume by Time',
            'Risk by Payment Type', 'Geographic Risk Distribution',
            'Currency Risk Analysis', 'Daily Risk Trends'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Risk vs Amount scatter
    if 'Amount' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Amount'],
                y=results_df['Risk_Score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=results_df['Risk_Score'],
                    colorscale='Reds',
                    showscale=False
                ),
                name='Risk vs Amount'
            ),
            row=1, col=1
        )
    
    # 2. Transaction volume by hour
    if 'hour' in results_df.columns or 'Time' in results_df.columns:
        if 'hour' not in results_df.columns and 'Time' in results_df.columns:
            results_df['hour'] = pd.to_datetime(results_df['Time'], format='%H:%M:%S').dt.hour
        
        if 'hour' in results_df.columns:
            hourly_stats = results_df.groupby('hour').agg({
                'Risk_Score': 'mean',
                'Prediction': 'count'
            }).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=hourly_stats['hour'],
                    y=hourly_stats['Prediction'],
                    name='Transaction Count',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
    
    # 3. Risk by Payment Type
    if 'Payment_type' in results_df.columns:
        payment_risk = results_df.groupby('Payment_type')['Risk_Score'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=payment_risk['Payment_type'],
                y=payment_risk['Risk_Score'],
                name='Avg Risk by Payment Type',
                marker_color='orange'
            ),
            row=1, col=3
        )
    
    # 4. Geographic risk
    if 'Sender_bank_location' in results_df.columns:
        geo_risk = results_df.groupby('Sender_bank_location').agg({
            'Risk_Score': 'mean',
            'Prediction': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=geo_risk['Sender_bank_location'],
                y=geo_risk['Risk_Score'],
                mode='markers',
                marker=dict(
                    size=geo_risk['Prediction'] * 8 + 15,  # Better size scaling
                    color=geo_risk['Risk_Score'],  # Color based on risk score
                    colorscale='Viridis',  # More distinguishable color scale
                    opacity=0.9,  # Increased opacity
                    line=dict(width=2, color='darkblue'),  # Add border for better visibility
                    showscale=False
                ),
                name='Geographic Risk',
                text=[f"Location: {loc}<br>Risk: {risk:.1f}%<br>Count: {count}" 
                      for loc, risk, count in zip(geo_risk['Sender_bank_location'], 
                                                 geo_risk['Risk_Score'], 
                                                 geo_risk['Prediction'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 5. Currency risk
    if 'Payment_currency' in results_df.columns:
        currency_risk = results_df.groupby('Payment_currency')['Risk_Score'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=currency_risk['Payment_currency'],
                y=currency_risk['Risk_Score'],
                name='Risk by Currency',
                marker_color='purple'
            ),
            row=2, col=2
        )
    
    # 6. Daily trends
    if 'Date' in results_df.columns:
        try:
            results_df['Date'] = pd.to_datetime(results_df['Date'])
            daily_risk = results_df.groupby('Date').agg({
                'Risk_Score': 'mean',
                'Prediction': 'sum'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_risk['Date'],
                    y=daily_risk['Risk_Score'],
                    mode='lines+markers',
                    name='Daily Risk Trend',
                    line=dict(color='green')
                ),
                row=2, col=3
            )
        except:
            pass  # Skip if date parsing fails
    
    fig.update_layout(height=800, showlegend=False, title_text="AML Risk Analytics Dashboard")
    return fig

def create_entity_analysis(df, risk_scores, ensemble_pred):
    """Create entity-based analysis"""
    # Analyze high-risk entities
    high_risk_threshold = 0.6
    high_risk_mask = risk_scores >= high_risk_threshold
    
    entity_analysis = {
        'high_risk_senders': Counter(),
        'high_risk_receivers': Counter(),
        'suspicious_patterns': []
    }
    
    # Analyze senders and receivers
    for idx, row in df[high_risk_mask].iterrows():
        if 'Sender_account' in row:
            entity_analysis['high_risk_senders'][row['Sender_account']] += 1
        if 'Receiver_account' in row:
            entity_analysis['high_risk_receivers'][row['Receiver_account']] += 1
    
    # Find patterns
    # 1. Frequent high-risk senders
    top_senders = entity_analysis['high_risk_senders'].most_common(5)
    # 2. Frequent high-risk receivers
    top_receivers = entity_analysis['high_risk_receivers'].most_common(5)
    
    return entity_analysis, top_senders, top_receivers

def main():
    """
    Main Streamlit application for Money Laundering Detection
    
    Updated to follow main.py logic:
    - Uses model-specific thresholds (RF: 0.3, XGB: 0.7, LightGBM: 0.6)
    - Implements hybrid_voting with auto_tune=True and return_score=True
    - Saves hybrid_anomaly_scores.csv with same format as main.py
    - Includes anomaly score ranking functionality
    - Provides comprehensive Excel export with multiple sheets
    """
    st.markdown('<h1 class="main-header">🏦 Money Laundering Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Ensemble Model Testing with Excel Data Upload")
    st.info("📋 **Enhanced with main.py logic**: Model-specific thresholds, hybrid voting, and anomaly score ranking")
    
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
    
    # Show model thresholds from main.py
    st.sidebar.subheader("🎯 Model Thresholds")
    st.sidebar.write("Following main.py configuration:")
    model_thresholds = {
        "randomforest": 0.3,
        "xgboost": 0.7,
        "lightgbm": 0.6,
    }
    for model, threshold in model_thresholds.items():
        st.sidebar.write(f"• **{model.title()}**: {threshold}")
    
    st.sidebar.markdown("---")
    
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
                ensemble_pred, ensemble_scores, individual_probs, individual_scores = predict_ensemble(
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
                avg_risk_score = np.mean(ensemble_scores) * 100
                st.metric("Avg Risk Score", f"{avg_risk_score:.1f}%")
            
            # Visualizations
            st.subheader("� AML Analytics Dashboard")
            
            # Network Analysis
            st.subheader("🕸️ Entity Network Graph")
            
            # Network analysis controls
            col1, col2, col3 = st.columns(3)
            with col1:
                min_risk_threshold = st.slider(
                    "Minimum Risk Threshold (%)", 
                    0, 100, 30, 5,
                    help="Only show transactions above this risk level"
                )
            with col2:
                show_node_labels = st.checkbox("Show Node Labels", True)
            with col3:
                network_layout = st.selectbox(
                    "Network Layout", 
                    ["Spring", "Circular", "Random"],
                    help="Choose how to arrange the network nodes"
                )
            
            # Create network graph
            with st.spinner("Building network graph..."):
                G, entity_stats, df_network = create_transaction_network(
                    df, ensemble_scores, min_risk_threshold
                )
            
            if G is not None and len(G.nodes()) > 0:
                # Network statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔗 Network Entities", len(G.nodes()))
                with col2:
                    st.metric("↔️ Connections", len(G.edges()))
                with col3:
                    st.metric("🌐 Network Density", f"{nx.density(G):.3f}")
                with col4:
                    # Find most connected entity
                    degrees = dict(G.degree())
                    if degrees:
                        max_degree_node = max(degrees, key=degrees.get)
                        st.metric("🎯 Hub Entity", max_degree_node.replace('S_', '').replace('R_', ''))
                
                # Network visualization
                network_fig = create_network_visualization(G, entity_stats)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True, height=600)
                
                # Entity ranking table
                st.subheader("🏆 High-Risk Entity Rankings")
                
                entity_rankings = []
                for entity, stats in entity_stats.items():
                    entity_rankings.append({
                        'Entity': entity.replace('S_', '').replace('R_', ''),
                        'Type': 'Sender' if entity.startswith('S_') else 'Receiver',
                        'Transactions': stats['transactions'],
                        'Total_Amount': stats['total_amount'],
                        'Avg_Risk': stats['avg_risk'],
                        'Max_Risk': stats['max_risk'],
                        'Currencies': len(stats['currencies']),
                        'Locations': len(stats['locations'])
                    })
                
                entity_df = pd.DataFrame(entity_rankings)
                entity_df = entity_df.sort_values('Max_Risk', ascending=False)
                
                st.dataframe(
                    entity_df.style.format({
                        'Total_Amount': '${:,.2f}',
                        'Avg_Risk': '{:.1f}%',
                        'Max_Risk': '{:.1f}%'
                    }).background_gradient(subset=['Max_Risk'], cmap='Reds'),
                    use_container_width=True
                )
            else:
                st.warning(f"No network entities found above {min_risk_threshold}% risk threshold. Try lowering the threshold.")
            
            # Risk Analytics Dashboard
            st.subheader("📊 Risk Analytics Dashboard")
            
            # Create risk analytics dashboard
            risk_dashboard = create_risk_analytics_dashboard(df, ensemble_scores, ensemble_pred)
            st.plotly_chart(risk_dashboard, use_container_width=True)
            
            # Risk level distribution
            col1, col2 = st.columns(2)
            
            with col1:
                risk_levels = pd.cut(
                    ensemble_scores * 100, 
                    bins=[0, 30, 50, 70, 100], 
                    labels=['Low', 'Medium', 'High', 'Critical']
                )
                risk_counts = risk_levels.value_counts()
                
                fig_risk_dist = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title="Risk Level Distribution",
                    labels={'x': 'Risk Level', 'y': 'Count'},
                    color=risk_counts.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_risk_dist, use_container_width=True)
            
            with col2:
                # Top risky transactions
                top_risky = df.copy()
                top_risky['Risk_Score'] = ensemble_scores * 100
                top_risky = top_risky.nlargest(10, 'Risk_Score')
                
                fig_top_risky = px.bar(
                    x=range(len(top_risky)),
                    y=top_risky['Risk_Score'],
                    title="Top 10 Riskiest Transactions",
                    labels={'x': 'Transaction Index', 'y': 'Risk Score (%)'},
                    color=top_risky['Risk_Score'],
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_top_risky, use_container_width=True)
            
            # Pattern Detection
            st.subheader("🎯 Pattern Detection")
            
            # Entity analysis
            entity_analysis, top_senders, top_receivers = create_entity_analysis(
                df, ensemble_scores, ensemble_pred
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🚨 Most Suspicious Senders**")
                if top_senders:
                    sender_df = pd.DataFrame(top_senders, columns=['Sender', 'High_Risk_Transactions'])
                    st.dataframe(sender_df, use_container_width=True)
                else:
                    st.info("No high-risk senders detected")
            
            with col2:
                st.write("**📥 Most Suspicious Receivers**")
                if top_receivers:
                    receiver_df = pd.DataFrame(top_receivers, columns=['Receiver', 'High_Risk_Transactions'])
                    st.dataframe(receiver_df, use_container_width=True)
                else:
                    st.info("No high-risk receivers detected")
            
            # Transaction patterns
            st.subheader("🔍 Transaction Patterns")
            
            patterns_found = []
            
            # Pattern 1: High frequency transactions
            transaction_counts = df.groupby(['Sender_account', 'Receiver_account']).size().reset_index(name='Transaction_Count')
            high_freq_pairs = transaction_counts[transaction_counts['Transaction_Count'] >= 3]
            
            if len(high_freq_pairs) > 0:
                patterns_found.append(f"Found {len(high_freq_pairs)} high-frequency transaction pairs")
                
                # Show high frequency pairs table
                st.write("**🔄 High-Frequency Transaction Pairs**")
                high_freq_display = high_freq_pairs.sort_values('Transaction_Count', ascending=False)
                st.dataframe(
                    high_freq_display.style.background_gradient(subset=['Transaction_Count'], cmap='Oranges'),
                    use_container_width=True
                )
                
                # Visualization for high frequency pairs
                if len(high_freq_display) <= 20:  # Only show chart if not too many pairs
                    fig_freq = px.bar(
                        high_freq_display.head(10),
                        x='Transaction_Count',
                        y=[f"{row['Sender_account']} → {row['Receiver_account']}" for _, row in high_freq_display.head(10).iterrows()],
                        orientation='h',
                        title="Top 10 High-Frequency Transaction Pairs",
                        labels={'Transaction_Count': 'Number of Transactions', 'y': 'Transaction Pair'},
                        color='Transaction_Count',
                        color_continuous_scale='Oranges'
                    )
                    fig_freq.update_layout(height=400)
                    st.plotly_chart(fig_freq, use_container_width=True)
            
            # Pattern 2: Round amount transactions
            if 'Amount' in df.columns:
                round_amounts = df[df['Amount'] % 1000 == 0].copy()
                if len(round_amounts) > 0:
                    patterns_found.append(f"Found {len(round_amounts)} round amount transactions")
                    
                    # Show round amounts analysis
                    st.write("**💰 Round Amount Transactions**")
                    round_amounts['Risk_Score'] = ensemble_scores[round_amounts.index] * 100
                    round_summary = round_amounts.groupby('Amount').agg({
                        'Sender_account': 'count',
                        'Risk_Score': 'mean'
                    }).rename(columns={'Sender_account': 'Count'}).reset_index()
                    round_summary = round_summary.sort_values('Count', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(
                            round_summary.head(10).style.format({
                                'Amount': '${:,.0f}',
                                'Risk_Score': '{:.1f}%'
                            }).background_gradient(subset=['Risk_Score'], cmap='Reds'),
                            use_container_width=True
                        )
                    
                    with col2:
                        if len(round_summary) > 0:
                            fig_round = px.scatter(
                                round_summary.head(15),
                                x='Amount',
                                y='Risk_Score',
                                size='Count',
                                title="Round Amounts vs Risk Score",
                                labels={'Risk_Score': 'Average Risk Score (%)', 'Amount': 'Transaction Amount'},
                                color='Risk_Score',
                                color_continuous_scale='Reds'
                            )
                            st.plotly_chart(fig_round, use_container_width=True)
            
            # Pattern 3: Same day multiple transactions
            if 'Date' in df.columns:
                df_copy = df.copy()
                df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                same_day_counts = df_copy.groupby(['Date', 'Sender_account']).size().reset_index(name='Daily_Transactions')
                multiple_same_day = same_day_counts[same_day_counts['Daily_Transactions'] >= 3]
                
                if len(multiple_same_day) > 0:
                    patterns_found.append(f"Found {len(multiple_same_day)} entities with multiple transactions on same day")
                    
                    # Show same day multiple transactions
                    st.write("**📅 Same-Day Multiple Transactions**")
                    
                    # Add risk scores for these entities
                    entity_risk_map = {}
                    for idx, row in df_copy.iterrows():
                        entity = row['Sender_account']
                        if entity not in entity_risk_map:
                            entity_risk_map[entity] = []
                        entity_risk_map[entity].append(ensemble_scores[idx] * 100)
                    
                    # Calculate average risk for each entity
                    multiple_same_day['Avg_Risk_Score'] = multiple_same_day['Sender_account'].apply(
                        lambda x: np.mean(entity_risk_map.get(x, [0]))
                    )
                    
                    multiple_same_day_display = multiple_same_day.sort_values('Daily_Transactions', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(
                            multiple_same_day_display.head(15).style.format({
                                'Date': lambda x: x.strftime('%Y-%m-%d'),
                                'Avg_Risk_Score': '{:.1f}%'
                            }).background_gradient(subset=['Avg_Risk_Score'], cmap='Reds'),
                            use_container_width=True
                        )
                    
                    with col2:
                        # Timeline chart of same-day multiple transactions
                        daily_pattern = multiple_same_day_display.groupby('Date').agg({
                            'Daily_Transactions': 'sum',
                            'Sender_account': 'count'
                        }).rename(columns={'Sender_account': 'Unique_Entities'}).reset_index()
                        
                        fig_timeline = px.line(
                            daily_pattern,
                            x='Date',
                            y='Daily_Transactions',
                            title="Timeline of High-Activity Days",
                            labels={'Daily_Transactions': 'Total Transactions', 'Date': 'Date'},
                            markers=True
                        )
                        fig_timeline.add_scatter(
                            x=daily_pattern['Date'],
                            y=daily_pattern['Unique_Entities'],
                            mode='lines+markers',
                            name='Unique Entities',
                            yaxis='y2'
                        )
                        fig_timeline.update_layout(
                            yaxis2=dict(overlaying='y', side='right', title='Unique Entities'),
                            height=400
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Summary of patterns found
            if patterns_found:
                st.write("**📋 Pattern Summary:**")
                for pattern in patterns_found:
                    st.warning(f"⚠️ {pattern}")
            else:
                st.success("✅ No suspicious patterns detected")
            
            # Transaction Analysis
            st.subheader("📈 Transaction Analysis")
            
            # Risk score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    x=ensemble_scores * 100,
                    nbins=20,
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
                    'Ensemble': ensemble_scores * 100
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
            
            # Prepare results dataframe following main.py format
            results_df = df.copy()
            results_df['Risk_Score'] = ensemble_scores * 100
            results_df['Hybrid_Score'] = ensemble_scores  # Following main.py naming convention
            results_df['Prediction'] = ['Suspicious' if p == 1 else 'Safe' for p in ensemble_pred]
            results_df['Risk_Level'] = pd.cut(
                ensemble_scores * 100, 
                bins=[0, 10, 30, 60, 100], 
                labels=['Very Low', 'Low', 'Medium', 'High']
            )
            
            # Add individual model scores
            if len(individual_probs) == 3:
                results_df['LightGBM_Score'] = individual_probs[0] * 100
                results_df['RandomForest_Score'] = individual_probs[1] * 100
                results_df['XGBoost_Score'] = individual_probs[2] * 100
            
            # Add individual model scores from individual_scores dict
            for score_name, score_values in individual_scores.items():
                results_df[score_name.replace('_score', '_Score')] = score_values * 100
            
            # Create anomaly ranking following main.py logic
            if y is not None:
                # Import ranking function
                try:
                    from src.anomaly_score_ranking import rank_anomaly_scores
                    top_suspicious = rank_anomaly_scores(y, ensemble_scores, top_n=100)
                    
                    st.subheader("🎯 Top Suspicious Transactions (Ranked by Anomaly Score)")
                    st.write("**Top 10 suspicious transactions based on hybrid anomaly scoring:**")
                    
                    # Display top 10 suspicious transactions
                    # Check if the required columns exist before formatting
                    if 'anomaly_score' in top_suspicious.columns:
                        format_dict = {'anomaly_score': '{:.4f}'}
                        if 'rank' in top_suspicious.columns:
                            format_dict['rank'] = '{:.0f}'
                        
                        st.dataframe(
                            top_suspicious.head(10).style.format(format_dict).background_gradient(subset=['anomaly_score'], cmap='Reds'),
                            use_container_width=True
                        )
                    else:
                        # Fallback display without special formatting
                        st.dataframe(top_suspicious.head(10), use_container_width=True)
                    
                    # Add download button for top suspicious transactions
                    suspicious_csv = top_suspicious.to_csv(index=False)
                    today_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📥 Download Top Suspicious Transactions",
                        data=suspicious_csv,
                        file_name=f"top_suspicious_transactions_{today_str}.csv",
                        mime="text/csv"
                    )
                    
                except ImportError:
                    st.warning("Anomaly ranking function not available. Showing standard results.")
                except Exception as e:
                    st.warning(f"Could not generate anomaly ranking: {str(e)}. Showing standard results.")
            
            # Save hybrid scores like main.py
            df_scores = pd.DataFrame({
                "label": y if y is not None else np.nan,
                "hybrid_score": ensemble_scores
            })
            
            # Display hybrid scores summary
            with st.expander("📊 Hybrid Anomaly Scores Summary", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Score", f"{ensemble_scores.mean():.4f}")
                with col2:
                    st.metric("Max Score", f"{ensemble_scores.max():.4f}")
                with col3:
                    st.metric("Min Score", f"{ensemble_scores.min():.4f}")
                
                st.dataframe(
                    df_scores.describe().style.format('{:.4f}'),
                    use_container_width=True
                )
            
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
            col1, col2, col3 = st.columns(3)
            
            today_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with col1:
                # Full results CSV
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Full Results (CSV)",
                    data=csv_data,
                    file_name=f"money_laundering_results_{today_str}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Suspicious transactions only
                suspicious_only = results_df[results_df['Prediction'] == 'Suspicious']
                if len(suspicious_only) > 0:
                    suspicious_csv = suspicious_only.to_csv(index=False)
                    st.download_button(
                        label="🚨 Download Suspicious Only (CSV)",
                        data=suspicious_csv,
                        file_name=f"suspicious_transactions_{today_str}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No suspicious transactions detected")
            
            # Save hybrid scores CSV like main.py - moved outside columns
            st.markdown("---")
            hybrid_scores_csv = df_scores.to_csv(index=False)
            st.download_button(
                label="🔢 Download Hybrid Anomaly Scores (CSV)",
                data=hybrid_scores_csv,
                file_name=f"hybrid_anomaly_scores_{today_str}.csv",
                mime="text/csv",
                help="Hybrid anomaly scores in the same format as main.py output"
            )
            
            # Model performance (if ground truth available)
            if y is not None:
                st.subheader("🎯 Model Performance")
                try:
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
                    st.error(f"Error calculating model performance: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure your file has the required columns and format.")

if __name__ == "__main__":
    main()
