import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from PIL import Image
import time
import altair as alt

# Configure page
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .sidebar-content {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classifier</h1>
    <p>Advanced ML-powered classification of satellite imagery</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3>🎛️ Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["CNN Model", "ResNet50", "VGG16", "Custom Model"],
        key="model_select"
    )
    
    # Image processing options
    st.markdown("### 🔧 Processing Options")
    image_size = st.slider("Image Size", 64, 512, 255, step=32)
    batch_size = st.slider("Batch Size", 16, 128, 32, step=16)
    
    # Data augmentation
    st.markdown("### 🔄 Data Augmentation")
    rotation = st.checkbox("Rotation", value=True)
    flip = st.checkbox("Horizontal Flip", value=True)
    zoom = st.checkbox("Zoom", value=True)
    
    # Training parameters
    st.markdown("### 📊 Training Parameters")
    epochs = st.slider("Epochs", 10, 100, 25)
    learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
    
    # Action buttons
    st.markdown("### 🚀 Actions")
    if st.button("🔄 Retrain Model", key="retrain"):
        st.success("Model retraining initiated!")
    
    if st.button("📊 Generate Report", key="report"):
        st.info("Report generation started...")
    
    if st.button("💾 Save Configuration", key="save"):
        st.success("Configuration saved!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Model Performance Metrics
    st.markdown("## 📈 Model Performance")
    
    # Create sample metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">Accuracy</h3>
            <h2 style="color: #333; margin-bottom: 0;">94.2%</h2>
            <p style="color: #666; font-size: 0.9rem;">+2.1% from last run</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #f093fb; margin-bottom: 0.5rem;">Precision</h3>
            <h2 style="color: #333; margin-bottom: 0;">92.8%</h2>
            <p style="color: #666; font-size: 0.9rem;">+1.5% from last run</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #4facfe; margin-bottom: 0.5rem;">Recall</h3>
            <h2 style="color: #333; margin-bottom: 0;">93.5%</h2>
            <p style="color: #666; font-size: 0.9rem;">+0.8% from last run</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #43e97b; margin-bottom: 0.5rem;">F1-Score</h3>
            <h2 style="color: #333; margin-bottom: 0;">93.1%</h2>
            <p style="color: #666; font-size: 0.9rem;">+1.2% from last run</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Training History Charts
    st.markdown("## 📊 Training History")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Generate sample training data
        epochs_data = list(range(1, 26))
        train_acc = [0.3 + 0.65 * (1 - np.exp(-i/5)) + np.random.normal(0, 0.02) for i in epochs_data]
        val_acc = [0.28 + 0.64 * (1 - np.exp(-i/5)) + np.random.normal(0, 0.03) for i in epochs_data]
        
        acc_df = pd.DataFrame({
            'Epoch': epochs_data + epochs_data,
            'Accuracy': train_acc + val_acc,
            'Type': ['Training'] * len(epochs_data) + ['Validation'] * len(epochs_data)
        })
        
        acc_chart = alt.Chart(acc_df).mark_line(strokeWidth=3).add_selection(
            alt.selection_interval(bind='scales')
        ).encode(
            x=alt.X('Epoch:Q', title='Epoch'),
            y=alt.Y('Accuracy:Q', title='Accuracy', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Type:N', scale=alt.Scale(range=['#667eea', '#f093fb'])),
            tooltip=['Epoch', 'Accuracy', 'Type']
        ).properties(
            width=350,
            height=300,
            title='Training & Validation Accuracy'
        ).configure_title(
            fontSize=16,
            anchor='start'
        )
        
        st.altair_chart(acc_chart, use_container_width=True)
    
    with chart_col2:
        # Generate sample loss data
        train_loss = [2.5 * np.exp(-i/3) + np.random.normal(0, 0.1) for i in epochs_data]
        val_loss = [2.6 * np.exp(-i/3) + np.random.normal(0, 0.15) for i in epochs_data]
        
        loss_df = pd.DataFrame({
            'Epoch': epochs_data + epochs_data,
            'Loss': train_loss + val_loss,
            'Type': ['Training'] * len(epochs_data) + ['Validation'] * len(epochs_data)
        })
        
        loss_chart = alt.Chart(loss_df).mark_line(strokeWidth=3).add_selection(
            alt.selection_interval(bind='scales')
        ).encode(
            x=alt.X('Epoch:Q', title='Epoch'),
            y=alt.Y('Loss:Q', title='Loss'),
            color=alt.Color('Type:N', scale=alt.Scale(range=['#4facfe', '#43e97b'])),
            tooltip=['Epoch', 'Loss', 'Type']
        ).properties(
            width=350,
            height=300,
            title='Training & Validation Loss'
        ).configure_title(
            fontSize=16,
            anchor='start'
        )
        
        st.altair_chart(loss_chart, use_container_width=True)
    
    # Class Distribution
    st.markdown("## 🌍 Dataset Distribution")
    
    class_data = pd.DataFrame({
        'Class': ['Cloudy', 'Desert', 'Green Area', 'Water'],
        'Count': [1250, 1180, 1320, 1100],
        'Percentage': [25.0, 23.6, 26.4, 22.0]
    })
    
    dist_chart = alt.Chart(class_data).mark_bar().encode(
        x=alt.X('Class:N', title='Land Cover Class'),
        y=alt.Y('Count:Q', title='Number of Images'),
        color=alt.Color('Class:N', scale=alt.Scale(range=['#667eea', '#f093fb', '#43e97b', '#4facfe'])),
        tooltip=['Class', 'Count', 'Percentage']
    ).properties(
        width=600,
        height=300,
        title='Distribution of Classes in Dataset'
    ).configure_title(
        fontSize=16,
        anchor='start'
    )
    
    st.altair_chart(dist_chart, use_container_width=True)

with col2:
    # Model Information
    st.markdown("""
    <div class="info-card">
        <h3>🤖 Model Information</h3>
        <p><strong>Architecture:</strong> CNN</p>
        <p><strong>Input Size:</strong> 255×255×3</p>
        <p><strong>Parameters:</strong> 1.2M</p>
        <p><strong>Training Time:</strong> 45 min</p>
        <p><strong>Last Updated:</strong> 2 hours ago</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Information
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Dataset Overview</h3>
        <p><strong>Total Images:</strong> 4,850</p>
        <p><strong>Classes:</strong> 4</p>
        <p><strong>Resolution:</strong> 255×255</p>
        <p><strong>Format:</strong> RGB</p>
        <p><strong>Size:</strong> 125 MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recent Predictions
    st.markdown("### 🔍 Recent Predictions")
    
    # Sample predictions
    predictions_data = [
        {"image": "sat_001.jpg", "predicted": "Water", "confidence": 0.94},
        {"image": "sat_002.jpg", "predicted": "Desert", "confidence": 0.87},
        {"image": "sat_003.jpg", "predicted": "Green Area", "confidence": 0.92},
        {"image": "sat_004.jpg", "predicted": "Cloudy", "confidence": 0.89},
        {"image": "sat_005.jpg", "predicted": "Water", "confidence": 0.96}
    ]
    
    for pred in predictions_data:
        confidence_color = "#43e97b" if pred["confidence"] > 0.9 else "#f093fb" if pred["confidence"] > 0.8 else "#4facfe"
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {confidence_color};">
            <strong>{pred['image']}</strong><br>
            <span style="color: #666;">{pred['predicted']} • {pred['confidence']:.2%}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔮 Test New Image", key="test_image"):
        with st.spinner("Processing image..."):
            time.sleep(2)
            st.success("Image classified as: **Water** (95.2% confidence)")
    
    if st.button("📁 Load Dataset", key="load_dataset"):
        with st.spinner("Loading dataset..."):
            time.sleep(1.5)
            st.success("Dataset loaded successfully!")
    
    if st.button("🔄 Refresh Metrics", key="refresh"):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit • Last updated: July 2025</p>
</div>
""", unsafe_allow_html=True)

# Add some interactivity with expandable sections
with st.expander("🔧 Advanced Settings"):
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
        st.slider("Dropout Rate", 0.0, 0.8, 0.5)
    with col2:
        st.selectbox("Loss Function", ["Categorical Crossentropy", "Sparse Categorical Crossentropy"])
        st.slider("L2 Regularization", 0.0, 0.1, 0.01)

with st.expander("📊 Confusion Matrix"):
    # Create a sample confusion matrix
    cm_data = np.array([[245, 12, 8, 5],
                       [15, 238, 10, 7],
                       [6, 14, 252, 8],
                       [3, 9, 12, 246]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cloudy', 'Desert', 'Green Area', 'Water'],
                yticklabels=['Cloudy', 'Desert', 'Green Area', 'Water'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)

with st.expander("🚀 Performance Tips"):
    st.markdown("""
    <div class="success-card">
        <h4>💡 Optimization Tips</h4>
        <ul>
            <li>Use data augmentation to improve generalization</li>
            <li>Implement early stopping to prevent overfitting</li>
            <li>Consider transfer learning for better performance</li>
            <li>Monitor validation metrics during training</li>
            <li>Use appropriate batch sizes for your hardware</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
