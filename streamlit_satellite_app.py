import streamlit as st
import pandas as pd
# import numpy as npimport plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import base64
from io import BytesIO
import random

# Page configuration
st.set_page_config(
    page_title="SatelliteVision AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Card styles */
    .card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 4px;
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Title styling */
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.8s ease;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: 12px;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data():
    """Generate sample satellite image classification data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Sample classification data
    classifications = []
    for date in dates:
        for class_name in ['Cloudy', 'Desert', 'Green_Area', 'Water']:
            count = random.randint(50, 200)
            accuracy = random.uniform(0.85, 0.98)
            classifications.append({
                'date': date,
                'class': class_name,
                'count': count,
                'accuracy': accuracy
            })
    
    return pd.DataFrame(classifications)

# Sidebar
with st.sidebar:
    st.markdown("### 🛰️ Navigation")
    
    page = st.selectbox(
        "Select Page",
        ["Dashboard", "Classification", "Analytics", "Model Training", "Settings"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Images", "12,847", "↗️ 234")
    with col2:
        st.metric("Accuracy", "94.2%", "↗️ 2.1%")
    
    st.markdown("---")
    
    st.markdown("### 🎛️ Controls")
    
    # Theme selector
    theme = st.selectbox("Theme", ["Ocean", "Forest", "Desert", "Arctic"])
    
    # Notification toggle
    notifications = st.toggle("Enable Notifications", value=True)
    
    # Auto-refresh
    auto_refresh = st.toggle("Auto Refresh", value=False)
    
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30)

# Main content area
if page == "Dashboard":
    # Header
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="main-title">🛰️ SatelliteVision AI</h1>
        <p class="subtitle">Advanced Satellite Image Classification Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">12,847</div>
            <div class="metric-label">Images Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="metric-value">94.2%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
            <div class="metric-value">4</div>
            <div class="metric-label">Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
            <div class="metric-value">2.3s</div>
            <div class="metric-label">Avg Process Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts section
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Classification Trends")
        
        # Sample trend data
        df_sample = generate_sample_data()
        daily_counts = df_sample.groupby(['date', 'class'])['count'].sum().reset_index()
        
        fig = px.line(
            daily_counts, 
            x='date', 
            y='count', 
            color='class',
            title="Daily Classification Counts",
            color_discrete_map={
                'Cloudy': '#3498db',
                'Desert': '#f39c12',
                'Green_Area': '#2ecc71',
                'Water': '#9b59b6'
            }
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            title_font_size=16,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 Accuracy by Class")
        
        # Sample accuracy data
        accuracy_data = df_sample.groupby('class')['accuracy'].mean().reset_index()
        
        fig = px.bar(
            accuracy_data,
            x='class',
            y='accuracy',
            title="Model Accuracy by Classification",
            color='accuracy',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            title_font_size=16,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent activity
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Recent Activity")
    
    # Sample activity data
    activity_data = {
        'Time': ['2 minutes ago', '15 minutes ago', '1 hour ago', '3 hours ago', '1 day ago'],
        'Activity': [
            'Classified 45 new satellite images',
            'Model training completed with 94.2% accuracy',
            'Processed batch of 120 desert images',
            'Updated water classification parameters',
            'Exported classification results to CSV'
        ],
        'Status': ['✅ Completed', '✅ Completed', '✅ Completed', '✅ Completed', '✅ Completed']
    }
    
    activity_df = pd.DataFrame(activity_data)
    st.dataframe(activity_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Classification":
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="main-title">🔍 Image Classification</h1>
        <p class="subtitle">Upload and classify satellite images in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📁 Upload Images")
        
        uploaded_files = st.file_uploader(
            "Choose satellite images...",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg'],
            help="Upload satellite images for classification"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
            
            for i, file in enumerate(uploaded_files):
                st.write(f"📸 {file.name}")
                
                st.image(file, caption=f"🖼️ Preview: {file.name}", use_column_width=True)
                
                # Simulate processing
                progress_bar = st.progress(0)
                for j in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(j + 1)
                
                # Simulate classification result
                classes = ['Cloudy', 'Desert', 'Green_Area', 'Water']
                confidence = random.uniform(0.85, 0.98)
                predicted_class = random.choice(classes)
                
                st.success(f"🎯 Classification: **{predicted_class}** (Confidence: {confidence:.2%})")
                
                if i >= 2:  # Limit display to first 3 files
                    st.info(f"... and {len(uploaded_files) - 3} more files")
                    break
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Classification Results")
        
        # Sample classification distribution
        class_distribution = {
            'Class': ['Cloudy', 'Desert', 'Green_Area', 'Water'],
            'Count': [234, 456, 123, 789],
            'Percentage': [15.6, 30.4, 8.2, 52.6]
        }
        
        fig = px.pie(
            values=class_distribution['Count'],
            names=class_distribution['Class'],
            title="Classification Distribution",
            color_discrete_map={
                'Cloudy': '#3498db',
                'Desert': '#f39c12',
                'Green_Area': '#2ecc71',
                'Water': '#9b59b6'
            }
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Batch processing
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Batch Processing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Batch Processing", type="primary"):
            st.success("Batch processing initiated!")
    
    with col2:
        if st.button("⏸️ Pause Processing"):
            st.warning("Processing paused.")
    
    with col3:
        if st.button("📊 Export Results"):
            st.info("Results exported to CSV!")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Analytics":
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="main-title">📊 Analytics Dashboard</h1>
        <p class="subtitle">Comprehensive insights and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Time series analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Time Series Analysis")
    
    df_sample = generate_sample_data()
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Classification Volume", "Accuracy Trends", "Class Distribution", "Processing Time"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    
    # Volume chart
    daily_volume = df_sample.groupby('date')['count'].sum().reset_index()
    fig.add_trace(
        go.Scatter(x=daily_volume['date'], y=daily_volume['count'], name='Volume'),
        row=1, col=1
    )
    
    # Accuracy trends
    daily_accuracy = df_sample.groupby('date')['accuracy'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=daily_accuracy['date'], y=daily_accuracy['accuracy'], name='Accuracy'),
        row=1, col=2
    )
    
    # Class distribution
    class_totals = df_sample.groupby('class')['count'].sum()
    fig.add_trace(
        go.Pie(labels=class_totals.index, values=class_totals.values, name="Distribution"),
        row=2, col=1
    )
    
    # Processing time simulation
    processing_times = np.random.normal(2.3, 0.5, 100)
    fig.add_trace(
        go.Histogram(x=processing_times, name='Processing Time'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Model Performance")
        
        # Confusion matrix simulation
        confusion_data = np.random.randint(50, 200, size=(4, 4))
        confusion_df = pd.DataFrame(
            confusion_data,
            index=['Cloudy', 'Desert', 'Green_Area', 'Water'],
            columns=['Cloudy', 'Desert', 'Green_Area', 'Water']
        )
        
        fig = px.imshow(
            confusion_df,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Classification Metrics")
        
        metrics_data = {
            'Class': ['Cloudy', 'Desert', 'Green_Area', 'Water'],
            'Precision': [0.94, 0.92, 0.96, 0.95],
            'Recall': [0.93, 0.94, 0.95, 0.93],
            'F1-Score': [0.935, 0.930, 0.955, 0.940]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Average metrics
        st.metric("Average Precision", f"{np.mean(metrics_data['Precision']):.3f}")
        st.metric("Average Recall", f"{np.mean(metrics_data['Recall']):.3f}")
        st.metric("Average F1-Score", f"{np.mean(metrics_data['F1-Score']):.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Model Training":
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="main-title">🤖 Model Training</h1>
        <p class="subtitle">Train and optimize your satellite image classification model</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎛️ Training Configuration")
        
        # Training parameters
        col_a, col_b = st.columns(2)
        
        with col_a:
            epochs = st.slider("Number of Epochs", 1, 100, 25)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            learning_rate = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1], value=0.001)
        
        with col_b:
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"], index=0)
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2)
            data_augmentation = st.checkbox("Enable Data Augmentation", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Training controls
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🚀 Training Controls")
        
        col_x, col_y, col_z = st.columns(3)
        
        with col_x:
            start_training = st.button("▶️ Start Training", type="primary")
        
        with col_y:
            stop_training = st.button("⏹️ Stop Training")
        
        with col_z:
            save_model = st.button("💾 Save Model")
        
        if start_training:
            st.success("🎯 Training started!")
            
            # Simulate training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(epochs):
                # Simulate training metrics
                train_loss = 2.5 * np.exp(-i/10) + 0.1 + np.random.normal(0, 0.05)
                val_loss = 2.3 * np.exp(-i/10) + 0.15 + np.random.normal(0, 0.05)
                train_acc = 1 - np.exp(-i/5) * 0.8 + np.random.normal(0, 0.01)
                val_acc = 1 - np.exp(-i/5) * 0.85 + np.random.normal(0, 0.01)
                
                progress_bar.progress((i + 1) / epochs)
                status_text.text(f'Epoch {i+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}')
                
                time.sleep(0.1)
            
            st.success("✅ Training completed!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Training History")
        
        # Simulate training history
        epochs_range = list(range(1, 26))
        train_loss = [2.5 * np.exp(-i/10) + 0.1 + np.random.normal(0, 0.05) for i in epochs_range]
        val_loss = [2.3 * np.exp(-i/10) + 0.15 + np.random.normal(0, 0.05) for i in epochs_range]
        
        history_df = pd.DataFrame({
            'Epoch': epochs_range,
            'Training Loss': train_loss,
            'Validation Loss': val_loss
        })
        
        fig = px.line(
            history_df,
            x='Epoch',
            y=['Training Loss', 'Validation Loss'],
            title="Training History"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model info
        st.markdown("### 📋 Model Information")
        st.write("**Architecture:** CNN")
        st.write("**Parameters:** 1,234,567")
        st.write("**Model Size:** 12.3 MB")
        st.write("**Last Training:** 2 hours ago")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Settings":
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="main-title">⚙️ Settings</h1>
        <p class="subtitle">Configure your application preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # General settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎛️ General Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Language", ["English", "Spanish", "French", "German"])
        st.selectbox("Time Zone", ["UTC", "EST", "PST", "GMT"])
        st.slider("Max Upload Size (MB)", 1, 100, 50)
    
    with col2:
        st.checkbox("Enable Auto-Save", value=True)
        st.checkbox("Show Advanced Options", value=False)
        st.checkbox("Enable Debug Mode", value=False)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Default Model", ["SatelliteNet v1.0", "SatelliteNet v2.0", "Custom Model"])
        st.slider("Confidence Threshold", 0.5, 1.0, 0.8)
        st.selectbox("Image Size", ["224x224", "255x255", "512x512"])
    
    with col2:
        st.checkbox("Enable GPU Acceleration", value=True)
        st.checkbox("Use Mixed Precision", value=False)
        st.selectbox("Batch Processing Mode", ["Sequential", "Parallel"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # API settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔗 API Settings")
    
    api_key = st.text_input("API Key", value="sk-...", type="password")
    st.selectbox("API Version", ["v1", "v2", "v3"])
    st.slider("Rate Limit (requests/min)", 10, 1000, 100)
    
    if st.button("🔄 Regenerate API Key"):
        st.success("New API key generated!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Save settings
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Default"):
            st.warning("Settings reset to default values!")
    
    with col3:
        if st.button("📤 Export Settings"):
            st.info("Settings exported to JSON!")

# Auto-refresh functionality
if page == "Dashboard" and auto_refresh:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    if time.time() - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = time.time()
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
    <p>🛰️ SatelliteVision AI v1.0 | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)