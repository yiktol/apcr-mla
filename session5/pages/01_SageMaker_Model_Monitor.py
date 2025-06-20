
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import json
import time
import uuid
from datetime import datetime, timedelta
import random
import networkx as nx
from PIL import Image
import io
import base64
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from plotly.subplots import make_subplots



def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
        
    if 'drift_data' not in st.session_state:
        st.session_state.drift_data = generate_drift_data()
        
    if 'model_performance' not in st.session_state:
        st.session_state.model_performance = generate_model_performance_data()
        
    if 'model_test_results' not in st.session_state:
        st.session_state.model_test_results = None
    
    if 'traffic_distribution' not in st.session_state:
        st.session_state.traffic_distribution = [70, 30]

    if 'quiz_attempted' not in st.session_state:
        st.session_state['quiz_attempted'] = False
    if 'quiz_score' not in st.session_state:
        st.session_state['quiz_score'] = 0

def reset_session():
    """
    Reset the session state
    """
    # Keep only user_id and reset all other state
    user_id = st.session_state.user_id
    st.session_state.clear()
    st.session_state.user_id = user_id


# Data generation functions
def generate_drift_data():
    """
    Generate example data for data drift visualization
    """
    # Baseline data (training data distribution)
    np.random.seed(42)
    baseline_data = np.random.normal(loc=5, scale=1, size=1000)
    
    # Current data with drift
    current_data_slight_drift = np.random.normal(loc=5.2, scale=1.1, size=1000)
    current_data_medium_drift = np.random.normal(loc=5.5, scale=1.3, size=1000)
    current_data_large_drift = np.random.normal(loc=6, scale=1.5, size=1000)
    
    # Concept drift (change in relationship between features and target)
    x = np.linspace(0, 10, 1000)
    
    # Original relationship: y = 2x + random noise
    baseline_y = 2 * x + np.random.normal(0, 1, 1000)
    
    # Drifted relationship: y = 2x + 3 + random noise (shift in intercept)
    drift_y = 2 * x + 3 + np.random.normal(0, 1, 1000)
    
    # Drifted relationship: y = 3x - 1 + random noise (change in slope)
    concept_drift_y = 3 * x - 1 + np.random.normal(0, 1, 1000)
    
    return {
        'baseline': baseline_data,
        'slight_drift': current_data_slight_drift,
        'medium_drift': current_data_medium_drift,
        'large_drift': current_data_large_drift,
        'feature_x': x,
        'baseline_y': baseline_y,
        'drift_y': drift_y,
        'concept_drift_y': concept_drift_y
    }


def generate_model_performance_data():
    """
    Generate example data for model performance metrics over time
    """
    # Define timeline
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    
    # Generate accuracy with degradation over time
    accuracies = [0.92 - 0.001 * i + 0.02 * np.sin(i/10) for i in range(60)]
    
    # Generate F1 scores with similar pattern
    f1_scores = [0.90 - 0.0015 * i + 0.02 * np.sin(i/10) for i in range(60)]
    
    # Generate data quality score
    data_quality = [0.98 - 0.003 * i + 0.01 * np.sin(i/8) for i in range(60)]
    
    # Generate bias metrics
    bias_metric = [0.05 + 0.001 * i + 0.01 * np.sin(i/9) for i in range(60)]
    
    return pd.DataFrame({
        'date': dates,
        'accuracy': accuracies,
        'f1_score': f1_scores,
        'data_quality': data_quality,
        'bias_metric': bias_metric
    })


def generate_inference_data(num_samples=100, drift_level=0):
    """
    Generate mock inference data with optional drift
    """
    np.random.seed(int(time.time()))
    
    # Base features
    feature1 = np.random.normal(5 + drift_level, 1 + drift_level * 0.2, num_samples)
    feature2 = np.random.normal(2 - drift_level * 0.5, 0.5 + drift_level * 0.1, num_samples)
    feature3 = np.random.normal(10 + drift_level * 2, 2 + drift_level * 0.3, num_samples)
    feature4 = np.random.normal(1 + drift_level * 0.1, 0.2 + drift_level * 0.05, num_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4
    })
    
    # Generate predictions
    if drift_level == 0:
        # No drift condition
        predictions = 0.7 * feature1 - 1.2 * feature2 + 0.3 * feature3 + 2 * feature4
    else:
        # With drift, change the relationship
        predictions = 0.6 * feature1 - 1.0 * feature2 + 0.4 * feature3 + (2.2 - drift_level * 0.3) * feature4
    
    # Add some noise to predictions
    predictions = predictions + np.random.normal(0, 1, num_samples)
    
    # Classify predictions
    df['prediction'] = (predictions > np.mean(predictions)).astype(int)
    
    # Calculate ground truth (changes with drift)
    if drift_level < 0.5:
        # Low drift - ground truth is close to prediction
        df['actual'] = df['prediction']
        # Introduce some errors (5-10%)
        error_rate = 0.05 + drift_level * 0.1
        errors = np.random.choice(
            [True, False], 
            size=num_samples, 
            p=[error_rate, 1-error_rate]
        )
        df.loc[errors, 'actual'] = 1 - df.loc[errors, 'prediction']
    else:
        # High drift - ground truth diverges more from prediction
        error_rate = 0.1 + drift_level * 0.3
        errors = np.random.choice(
            [True, False], 
            size=num_samples, 
            p=[error_rate, 1-error_rate]
        )
        df['actual'] = df['prediction']
        df.loc[errors, 'actual'] = 1 - df.loc[errors, 'prediction']
    
    # Add timing information
    current_time = datetime.now()
    df['timestamp'] = [current_time - timedelta(seconds=i) for i in range(num_samples)]
    
    return df


def run_model_test(traffic_distribution, test_duration=10):
    """
    Run a simulated model test with specified traffic distribution
    """
    # Parse traffic distribution
    model_a_traffic = traffic_distribution[0] / 100
    model_b_traffic = traffic_distribution[1] / 100
    
    # Generate total number of inferences during test period
    total_inferences = 1000
    
    # Distribute traffic between models
    model_a_count = int(total_inferences * model_a_traffic)
    model_b_count = total_inferences - model_a_count
    
    # Generate inference data for each model
    model_a_data = generate_inference_data(model_a_count, drift_level=0.2)
    model_b_data = generate_inference_data(model_b_count, drift_level=0.5)
    
    # Calculate metrics for each model
    model_a_metrics = calculate_model_metrics(model_a_data)
    model_b_metrics = calculate_model_metrics(model_b_data)
    
    # Generate hourly metrics for both models
    hourly_metrics = generate_hourly_metrics(model_a_traffic, model_b_traffic)
    
    return {
        "model_a_metrics": model_a_metrics,
        "model_b_metrics": model_b_metrics,
        "model_a_traffic": model_a_traffic,
        "model_b_traffic": model_b_traffic,
        "model_a_sample": model_a_data.sample(min(5, model_a_count)),
        "model_b_sample": model_b_data.sample(min(5, model_b_count)),
        "hourly_metrics": hourly_metrics,
        "test_duration": test_duration
    }


def calculate_model_metrics(df):
    """
    Calculate model performance metrics from data
    """
    # Calculate accuracy
    accuracy = (df['prediction'] == df['actual']).mean()
    
    # Calculate true positives, false positives, etc.
    tp = ((df['prediction'] == 1) & (df['actual'] == 1)).sum()
    fp = ((df['prediction'] == 1) & (df['actual'] == 0)).sum()
    tn = ((df['prediction'] == 0) & (df['actual'] == 0)).sum()
    fn = ((df['prediction'] == 0) & (df['actual'] == 1)).sum()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Data quality metrics
    missing_rate = df.isna().mean().mean()
    data_quality_score = 1 - missing_rate
    
    # Calculate feature drift using statistical tests
    feature_stats = {}
    for feature in ['feature1', 'feature2', 'feature3', 'feature4']:
        # Simulate baseline data
        baseline_mean = 5 if feature == 'feature1' else (2 if feature == 'feature2' else (10 if feature == 'feature3' else 1))
        baseline_std = 1 if feature == 'feature1' else (0.5 if feature == 'feature2' else (2 if feature == 'feature3' else 0.2))
        
        # Generate fake baseline data
        baseline_data = np.random.normal(baseline_mean, baseline_std, 500)
        
        # Calculate KS test statistic
        ks_stat, p_value = stats.ks_2samp(baseline_data, df[feature])
        
        feature_stats[feature] = {
            'drift_score': ks_stat,
            'p_value': p_value,
            'mean': df[feature].mean(),
            'std': df[feature].std(),
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'data_quality': data_quality_score,
        'feature_stats': feature_stats,
        'inference_count': len(df)
    }


def generate_hourly_metrics(model_a_traffic, model_b_traffic):
    """
    Generate hourly metrics for the test period
    """
    hours = 10
    
    # Generate time points
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
    
    # Model A metrics - with some variance but overall stable
    model_a_accuracy = [0.92 + random.uniform(-0.03, 0.02) for _ in range(hours)]
    model_a_latency = [120 + random.uniform(-20, 30) for _ in range(hours)]
    model_a_errors = [random.randint(0, 5) for _ in range(hours)]
    
    # Model B metrics - with some variance and deterioration
    model_b_accuracy = [0.88 + random.uniform(-0.05, 0.02) - i*0.005 for i in range(hours)]
    model_b_latency = [150 + random.uniform(-25, 40) + i*5 for i in range(hours)]
    model_b_errors = [random.randint(1, 8) + i for i in range(hours)]
    
    # Combined metrics based on traffic distribution
    combined_accuracy = [a*model_a_traffic + b*model_b_traffic 
                        for a, b in zip(model_a_accuracy, model_b_accuracy)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'model_a_accuracy': model_a_accuracy,
        'model_a_latency': model_a_latency,
        'model_a_errors': model_a_errors,
        'model_b_accuracy': model_b_accuracy,
        'model_b_latency': model_b_latency,
        'model_b_errors': model_b_errors,
        'combined_accuracy': combined_accuracy
    })
    
    return df


def load_lottie_url(url: str):
    """
    Load lottie animation from URL
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


# Visualization functions
def plot_data_drift(baseline, current):
    """
    Create a visualization of data drift between baseline and current data
    """
    fig = plt.figure(figsize=(10, 5))
    
    sns.kdeplot(baseline, fill=True, alpha=0.5, label='Baseline (Training)', color='#00A1C9')
    sns.kdeplot(current, fill=True, alpha=0.5, label='Current (Inference)', color='#FF9900')
    
    plt.title('Data Distribution Comparison', fontsize=14)
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    
    return fig


def plot_concept_drift(x, baseline_y, drift_y):
    """
    Create a visualization of concept drift between baseline and current data
    """
    fig = plt.figure(figsize=(10, 5))
    
    plt.scatter(x, baseline_y, alpha=0.3, label='Baseline Relationship', color='#00A1C9')
    plt.scatter(x, drift_y, alpha=0.3, label='Current Relationship', color='#FF9900')
    
    # Add trendlines
    baseline_poly = np.polyfit(x, baseline_y, 1)
    drift_poly = np.polyfit(x, drift_y, 1)
    
    baseline_trend_y = np.polyval(baseline_poly, x)
    drift_trend_y = np.polyval(drift_poly, x)
    
    plt.plot(x, baseline_trend_y, color='#00A1C9', linewidth=3)
    plt.plot(x, drift_trend_y, color='#FF9900', linewidth=3)
    
    plt.title('Concept Drift Visualization', fontsize=14)
    plt.xlabel('Feature X')
    plt.ylabel('Target Y')
    plt.legend()
    
    return fig


def plot_model_performance(performance_df):
    """
    Create a visualization of model performance over time
    """
    fig = go.Figure()
    
    # Add accuracy line
    fig.add_trace(go.Scatter(
        x=performance_df['date'],
        y=performance_df['accuracy'],
        mode='lines',
        name='Accuracy',
        line=dict(color='#00A1C9', width=3)
    ))
    
    # Add F1 score line
    fig.add_trace(go.Scatter(
        x=performance_df['date'],
        y=performance_df['f1_score'],
        mode='lines',
        name='F1 Score',
        line=dict(color='#FF9900', width=3)
    ))
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=performance_df['date'].min(),
        y0=0.85,
        x1=performance_df['date'].max(),
        y1=0.85,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add annotation for threshold
    fig.add_annotation(
        x=performance_df['date'].max(),
        y=0.85,
        text="Alert Threshold",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-30
    )
    
    # Configure layout
    fig.update_layout(
        title="Model Performance Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Metric Value",
        yaxis=dict(range=[0.75, 1.0]),
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_monitor_dashboard(test_results):
    """
    Create a dashboard visualization of model monitoring results
    """
    if test_results is None:
        return None
    
    # Unpack results
    model_a_metrics = test_results["model_a_metrics"]
    model_b_metrics = test_results["model_b_metrics"]
    hourly_metrics = test_results["hourly_metrics"]
    
    # Create performance comparison chart
    metrics_fig = go.Figure()
    
    # Add metrics for Model A
    metrics_fig.add_trace(go.Bar(
        x=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Data Quality'],
        y=[
            model_a_metrics['accuracy'], 
            model_a_metrics['precision'], 
            model_a_metrics['recall'],
            model_a_metrics['f1'],
            model_a_metrics['data_quality']
        ],
        name='Model A',
        marker_color='#00A1C9'
    ))
    
    # Add metrics for Model B
    metrics_fig.add_trace(go.Bar(
        x=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Data Quality'],
        y=[
            model_b_metrics['accuracy'], 
            model_b_metrics['precision'], 
            model_b_metrics['recall'],
            model_b_metrics['f1'],
            model_b_metrics['data_quality']
        ],
        name='Model B',
        marker_color='#FF9900'
    ))
    
    # Configure layout
    metrics_fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create time-series chart for accuracy
    time_fig = go.Figure()
    
    # Add accuracy for Model A
    time_fig.add_trace(go.Scatter(
        x=hourly_metrics['timestamp'],
        y=hourly_metrics['model_a_accuracy'],
        mode='lines+markers',
        name='Model A Accuracy',
        line=dict(color='#00A1C9', width=3)
    ))
    
    # Add accuracy for Model B
    time_fig.add_trace(go.Scatter(
        x=hourly_metrics['timestamp'],
        y=hourly_metrics['model_b_accuracy'],
        mode='lines+markers',
        name='Model B Accuracy',
        line=dict(color='#FF9900', width=3)
    ))
    
    # Add combined accuracy based on traffic distribution
    time_fig.add_trace(go.Scatter(
        x=hourly_metrics['timestamp'],
        y=hourly_metrics['combined_accuracy'],
        mode='lines+markers',
        name='Combined Accuracy',
        line=dict(color='#59BA47', width=3, dash='dash')
    ))
    
    # Configure layout
    time_fig.update_layout(
        title="Model Accuracy Over Time",
        xaxis_title="Time",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0.8, 1.0]),
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create latency comparison chart
    latency_fig = go.Figure()
    
    # Add latency for Model A
    latency_fig.add_trace(go.Scatter(
        x=hourly_metrics['timestamp'],
        y=hourly_metrics['model_a_latency'],
        mode='lines+markers',
        name='Model A Latency',
        line=dict(color='#00A1C9', width=3)
    ))
    
    # Add latency for Model B
    latency_fig.add_trace(go.Scatter(
        x=hourly_metrics['timestamp'],
        y=hourly_metrics['model_b_latency'],
        mode='lines+markers',
        name='Model B Latency',
        line=dict(color='#FF9900', width=3)
    ))
    
    # Configure layout
    latency_fig.update_layout(
        title="Model Latency Over Time (ms)",
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create feature drift chart - radar chart
    # Get feature drift scores
    features = ['feature1', 'feature2', 'feature3', 'feature4']
    model_a_drift = [model_a_metrics['feature_stats'][f]['drift_score'] for f in features]
    model_b_drift = [model_b_metrics['feature_stats'][f]['drift_score'] for f in features]
    
    # Create radar chart for feature drift
    drift_fig = go.Figure()
    
    drift_fig.add_trace(go.Scatterpolar(
        r=model_a_drift,
        theta=features,
        fill='toself',
        name='Model A',
        line_color='#00A1C9'
    ))
    
    drift_fig.add_trace(go.Scatterpolar(
        r=model_b_drift,
        theta=features,
        fill='toself',
        name='Model B',
        line_color='#FF9900'
    ))
    
    drift_fig.update_layout(
        title="Feature Drift (KS Statistic)",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(model_a_drift), max(model_b_drift)) * 1.1]
            )),
        height=400,
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=20),
    )
    
    return {
        "metrics_fig": metrics_fig,
        "time_fig": time_fig,
        "latency_fig": latency_fig,
        "drift_fig": drift_fig
    }


def create_monitor_architecture():
    """
    Create a visualization of SageMaker Model Monitor architecture
    """
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("Training Data", type="data", color="#00A1C9")
    G.add_node("Model Training", type="process", color="#FF9900")
    G.add_node("Baseline Statistics", type="data", color="#00A1C9")
    G.add_node("Constraints", type="data", color="#00A1C9")
    G.add_node("Trained Model", type="model", color="#59BA47")
    G.add_node("SageMaker Endpoint", type="endpoint", color="#232F3E")
    G.add_node("Inference Requests", type="data", color="#00A1C9")
    G.add_node("Inference Responses", type="data", color="#00A1C9")
    G.add_node("Captured Data", type="data", color="#00A1C9")
    G.add_node("Model Monitor", type="monitor", color="#D13212")
    G.add_node("Data Quality", type="metric", color="#545B64")
    G.add_node("Model Quality", type="metric", color="#545B64")
    G.add_node("Bias Drift", type="metric", color="#545B64")
    G.add_node("Feature Attribution", type="metric", color="#545B64")
    G.add_node("CloudWatch Alerts", type="alert", color="#FF9900")
    
    # Add edges
    G.add_edge("Training Data", "Model Training")
    G.add_edge("Model Training", "Trained Model")
    G.add_edge("Training Data", "Baseline Statistics")
    G.add_edge("Baseline Statistics", "Constraints")
    G.add_edge("Trained Model", "SageMaker Endpoint")
    G.add_edge("Inference Requests", "SageMaker Endpoint")
    G.add_edge("SageMaker Endpoint", "Inference Responses")
    G.add_edge("SageMaker Endpoint", "Captured Data")
    G.add_edge("Captured Data", "Model Monitor")
    G.add_edge("Constraints", "Model Monitor")
    G.add_edge("Model Monitor", "Data Quality")
    G.add_edge("Model Monitor", "Model Quality")
    G.add_edge("Model Monitor", "Bias Drift")
    G.add_edge("Model Monitor", "Feature Attribution")
    G.add_edge("Data Quality", "CloudWatch Alerts")
    G.add_edge("Model Quality", "CloudWatch Alerts")
    G.add_edge("Bias Drift", "CloudWatch Alerts")
    G.add_edge("Feature Attribution", "CloudWatch Alerts")
    
    return G


def draw_monitor_architecture(G):
    """
    Draw the SageMaker Model Monitor architecture
    """
    plt.figure(figsize=(14, 10))
    
    # Define positions for nodes
    pos = {
        "Training Data": (1, 8),
        "Model Training": (3, 8),
        "Trained Model": (5, 8),
        "SageMaker Endpoint": (7, 8),
        "Baseline Statistics": (1, 6),
        "Constraints": (3, 6),
        "Inference Requests": (7, 10),
        "Inference Responses": (9, 8),
        "Captured Data": (7, 6),
        "Model Monitor": (5, 4),
        "Data Quality": (3, 2),
        "Model Quality": (5, 2),
        "Bias Drift": (7, 2),
        "Feature Attribution": (9, 2),
        "CloudWatch Alerts": (6, 0),
    }
    
    # Get node types and colors
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    # Create node shapes dictionary
    node_shapes = {
        "data": "s",  # square
        "process": "o",  # circle
        "model": "h",  # hexagon
        "endpoint": "d",  # diamond
        "monitor": "^",  # triangle up
        "metric": "p",  # pentagon
        "alert": "*",  # star
    }
    
    # Draw the graph components grouped by node type
    for node_type, shape in node_shapes.items():
        # Filter nodes of this type
        nodes = [n for n in G.nodes() if G.nodes[n]['type'] == node_type]
        if not nodes:
            continue
            
        # Get positions and colors for this node type
        pos_subset = {n: pos[n] for n in nodes}
        color_subset = [G.nodes[n]['color'] for n in nodes]
        
        # Draw nodes of this type
        nx.draw_networkx_nodes(
            G, pos_subset, 
            nodelist=nodes,
            node_color=color_subset, 
            node_shape=shape,
            node_size=2000
        )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        edge_color='#aaaaaa', 
        width=1.5,
        arrowsize=20,
        arrowstyle='->'
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, 
        font_size=12, 
        font_weight='bold',
        font_color='white'
    )
    
    # Add a legend for node types
    legend_shapes = []
    legend_labels = []
    
    for node_type, shape in node_shapes.items():
        # Check if this node type exists in the graph
        if any(G.nodes[n]['type'] == node_type for n in G.nodes()):
            # Find a color for this node type
            color = next((G.nodes[n]['color'] for n in G.nodes() if G.nodes[n]['type'] == node_type), '#000000')
            
            legend_shapes.append(
                plt.Line2D([0], [0], marker=shape, color='w', markerfacecolor=color, markersize=15, label=node_type.title())
            )
            legend_labels.append(node_type.title())
    
    plt.legend(legend_shapes, legend_labels, loc='upper right', fontsize=12)
    
    plt.title("Amazon SageMaker Model Monitor Architecture", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()


# Main application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="SageMaker Model Monitor Learning Hub",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # AWS Color Scheme
    AWS_COLORS = {
        "orange": "#FF9900",
        "teal": "#00A1C9", 
        "blue": "#232F3E",
        "gray": "#E9ECEF",
        "light_gray": "#F8F9FA",
        "white": "#FFFFFF",
        "dark_gray": "#545B64",
        "green": "#59BA47",
        "red": "#D13212"
    }
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {
            background-color: #F8F9FA;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #F8F9FA;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            white-space: pre-wrap;
            border-radius: 6px;
            font-weight: 600;
            background-color: #FFFFFF;
            color: #232F3E;
            border: 1px solid #E9ECEF;
            padding: 5px 15px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF9900 !important;
            color: #FFFFFF !important;
            border: 1px solid #FF9900 !important;
        }
        .stButton button {
            background-color: #FF9900;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 8px 16px;
        }
        .stButton button:hover {
            background-color: #EC7211;
        }
        .info-box {
            background-color: #E6F2FF;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #00A1C9;
        }
        .warning-box {
            background-color: #FFF8E6;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #FF9900;
        }
        .code-box {
            background-color: #232F3E;
            color: #FFFFFF;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            margin: 15px 0;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 0;
            right: 0;
            left: 0;
            background-color: #232F3E;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 12px;
        }
        h1, h2, h3 {
            color: #232F3E;
        }
        .status-card {
            border: 1px solid #E9ECEF;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            background-color: #FFFFFF;
            transition: transform 0.2s;
        }
        .status-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .status-1 {
            border-left: 5px solid #59BA47;
        }
        .status-2 {
            border-left: 5px solid #D13212;
        }
        .status-3 {
            border-left: 5px solid #FF9900;
        }
        .metrics-table th {
            font-weight: normal;
            color: #545B64;
        }
        .metrics-table td {
            font-weight: bold;
        }
        .stSlider [data-baseweb=slider] {
            margin-top: 3rem;
        }
        .monitor-column {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .tooltip-header {
            font-weight: bold;
            text-decoration: underline;
            display: inline-block;
            color: #00A1C9;
            cursor: help;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #FF9900;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #232F3E;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #232F3E;
            margin-top: 0.8rem;
            margin-bottom: 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Function to display custom header
    def custom_header(text, level="main"):
        if level == "main":
            st.markdown(f'<div class="main-header">{text}</div>', unsafe_allow_html=True)
        elif level == "sub":
            st.markdown(f'<div class="sub-header">{text}</div>', unsafe_allow_html=True)
        elif level == "section":
            st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.title("Session Management")
        st.info(f"User ID: {st.session_state.user_id}")
        
        if st.button("Reset Session"):
            reset_session()
            st.rerun()
        
        st.divider()     
        # Information about the application
        with st.expander("About this application", expanded=False):
            st.markdown("""
                This e-learning application demonstrates how Amazon SageMaker Model Monitor works 
                to detect and visualize model drift in machine learning deployments.
                
                Navigate through the tabs to learn about different types of drift, 
                monitoring strategies, and how to test models with different traffic distributions.
            """)
            
            # Load lottie animation
            lottie_url = "https://assets4.lottiefiles.com/packages/lf20_qp1q7mct.json"
            lottie_json = load_lottie_url(lottie_url)
            if lottie_json:
                st_lottie(lottie_json, height=200, key="sidebar_animation")
            
            # Additional resources section
            st.subheader("Additional Resources")
            st.markdown("""
                - [SageMaker Model Monitor Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
                - [Model Drift Detection Best Practices](https://aws.amazon.com/blogs/machine-learning/best-practices-for-detecting-and-handling-model-drift-on-amazon-sagemaker/)
                - [SageMaker Model Monitoring Examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/sagemaker_model_monitor)
            """)
    
    # Main app header
    st.title("Amazon SageMaker Model Monitor Learning Hub")
    st.markdown("Explore how to detect, visualize, and mitigate model drift in production machine learning models.")
    
    # Tab-based navigation with emoji
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 SageMaker Model Monitor", 
        "🔍 Types of Model Drift",
        "🛠️ Model Testing Strategies",
        "🚦 Test Models with Traffic Distribution",
        "📝 Knowledge Check"
    ])
    
    # TAB 1: SAGEMAKER MODEL MONITOR
    with tab1:
        st.header("Amazon SageMaker Model Monitor")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Amazon SageMaker Model Monitor provides tools to continuously monitor the quality of machine learning models in production. 
            It helps detect concept drift and data quality issues in deployed models.
            
            **Key capabilities:**
            - Monitors data quality
            - Detects concept drift in model predictions
            - Tracks model bias drift over time
            - Monitors feature attribution drift
            - Sends alerts when thresholds are breached
            """)
            
            st.markdown("""
            <div class="info-box">
            <h3>Why Model Monitoring Matters</h3>
            <p>Models that perform well during development may degrade over time in production due to:</p>
            <ul>
                <li>Changes in input data patterns</li>
                <li>Concept drift where relationships between features and target change</li>
                <li>Data quality issues in production inputs</li>
                <li>Seasonal or temporal variations in data</li>
            </ul>
            <p>Continuous monitoring helps ensure model predictions remain reliable and accurate over time.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Monitoring diagram
            monitoring_architecture = create_monitor_architecture()
            fig = draw_monitor_architecture(monitoring_architecture)
            st.pyplot(fig)
            st.caption("SageMaker Model Monitor Architecture")
        
        st.subheader("How SageMaker Model Monitor Works")
        
        # Three columns for three steps
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>1️⃣ Baseline Creation</h3>
                <p>During model development, SageMaker automatically analyzes your training data to:</p>
                <ul>
                    <li>Generate baseline statistics</li>
                    <li>Create constraints to validate future data</li>
                    <li>Define expected data and prediction distributions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
                <h3>2️⃣ Data Capture</h3>
                <p>When deployed, the model endpoint:</p>
                <ul>
                    <li>Captures inference requests and responses</li>
                    <li>Stores data in S3 for analysis</li>
                    <li>Supports sampling to reduce storage costs</li>
                    <li>Captures metadata and prediction results</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="card">
                <h3>3️⃣ Continuous Monitoring</h3>
                <p>SageMaker periodically:</p>
                <ul>
                    <li>Analyses captured data for drift</li>
                    <li>Compares against baseline statistics</li>
                    <li>Generates monitoring reports</li>
                    <li>Triggers alerts when violations occur</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Monitor Types")
        
        # Four columns for four monitor types
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="status-card status-1">
                <h4>Data Quality Monitoring</h4>
                <p>Detects changes in data statistics, missing values, and data type inconsistencies.</p>
                <ul>
                    <li>Feature distributions</li>
                    <li>Missing values</li>
                    <li>Type mismatches</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="status-card status-3">
                <h4>Model Quality Monitoring</h4>
                <p>Tracks model performance metrics when ground truth is available.</p>
                <ul>
                    <li>Accuracy</li>
                    <li>Precision/Recall</li>
                    <li>F1 Score</li>
                    <li>AUC</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="status-card status-2">
                <h4>Bias Drift Monitoring</h4>
                <p>Detects changes in model bias against sensitive groups.</p>
                <ul>
                    <li>Demographic parity</li>
                    <li>Equal opportunity</li>
                    <li>Disparate impact</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="status-card status-3">
                <h4>Feature Attribution Monitoring</h4>
                <p>Tracks changes in feature importance over time.</p>
                <ul>
                    <li>SHAP values</li>
                    <li>Feature importance</li>
                    <li>Attribution shift</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Example code for setting up monitoring
        st.subheader("Example: Setting up Model Monitor")
        
        st.code('''
# Create data baseline
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor

# Configure data capture for the endpoint
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://my-bucket/monitor-data/'
)

# Deploy model with data capture enabled
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    data_capture_config=data_capture_config
)

# Create model monitor
my_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

# Create the baseline
my_monitor.suggest_baseline(
    baseline_dataset='s3://my-bucket/training-data/training.csv',
    dataset_format=DatasetFormat.csv()
)

# Create a monitoring schedule that runs hourly
my_monitor.create_monitoring_schedule(
    monitor_schedule_name='my-monitoring-schedule',
    endpoint_input=predictor.endpoint_name,
    schedule_expression='cron(0 * ? * * *)'
)
        ''')
        
        # Sample performance monitoring visualization
        st.subheader("Sample Model Performance Monitoring")
        
        performance_fig = plot_model_performance(st.session_state.model_performance)
        st.plotly_chart(performance_fig, use_container_width=True)
        
        st.markdown("""
        <div class="warning-box">
        <h3>Best Practices</h3>
        <p>For effective model monitoring:</p>
        <ul>
            <li><strong>Establish meaningful baselines</strong> using representative training data</li>
            <li><strong>Set appropriate thresholds</strong> that balance false alarms with drift detection</li>
            <li><strong>Monitor gradually</strong> - start with data quality, then add more advanced monitoring</li>
            <li><strong>Create automated workflows</strong> to respond to detected drift</li>
            <li><strong>Schedule regular reviews</strong> of monitoring results and thresholds</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 2: TYPES OF MODEL DRIFT
    with tab2:
        st.header("Types of Model Drift")
        
        st.markdown("""
        Model drift occurs when a model's performance deteriorates over time due to changes in data patterns.
        Understanding different types of drift helps in detecting and mitigating them effectively.
        """)
        
        # Types of drift tabs
        drift_tab1, drift_tab2, drift_tab3, drift_tab4 = st.tabs([
            "Data Drift", 
            "Concept Drift", 
            "Prediction Drift",
            "Feature Importance Drift"
        ])
        
        # DATA DRIFT
        with drift_tab1:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("What is Data Drift?")
                st.markdown("""
                Data drift occurs when the statistical properties of the input data change over time, 
                causing the model's predictions to become less accurate. This is also known as covariate shift.
                
                **Common causes:**
                - Seasonal variations
                - Changes in data collection methods
                - Changes in user behavior
                - System upgrades affecting data generation
                """)
                
                st.markdown("""
                <div class="info-box">
                <h3>Detection Methods</h3>
                <p>Statistical tests to detect data drift:</p>
                <ul>
                    <li><strong>Kolmogorov-Smirnov (K-S) test</strong>: Measures the maximum difference between two cumulative distribution functions</li>
                    <li><strong>Population Stability Index (PSI)</strong>: Measures how much a distribution has changed</li>
                    <li><strong>Jensen-Shannon Distance</strong>: Measures the similarity between two probability distributions</li>
                    <li><strong>Wasserstein Distance</strong>: Measures the distance between two probability distributions</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Select drift level for visualization
                drift_level = st.select_slider(
                    "Select drift level for visualization:",
                    options=["No Drift", "Slight Drift", "Medium Drift", "Large Drift"],
                    value="Medium Drift"
                )
                
                # Map selection to data
                drift_data_map = {
                    "No Drift": st.session_state.drift_data['baseline'],
                    "Slight Drift": st.session_state.drift_data['slight_drift'],
                    "Medium Drift": st.session_state.drift_data['medium_drift'],
                    "Large Drift": st.session_state.drift_data['large_drift']
                }
                
                # Plot data drift
                drift_fig = plot_data_drift(st.session_state.drift_data['baseline'], drift_data_map[drift_level])
                st.pyplot(drift_fig)
                st.caption("Comparison of baseline (training) and current (inference) data distributions")
            
            # Data drift mitigation strategies
            st.subheader("Mitigation Strategies")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>Regular Retraining</h4>
                    <p>Periodically retrain models with recent data to adapt to changing patterns.</p>
                    <p><strong>Example:</strong> Implement automated retraining pipelines that trigger when drift exceeds thresholds.</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Online Learning</h4>
                    <p>Use models that can update incrementally as new data arrives.</p>
                    <p><strong>Example:</strong> Implement reinforcement learning systems that adapt to new patterns in real-time.</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div class="card">
                    <h4>Robust Feature Engineering</h4>
                    <p>Design features that are more stable and resistant to drift.</p>
                    <p><strong>Example:</strong> Use feature normalization techniques or relative features instead of absolute values.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # CONCEPT DRIFT
        with drift_tab2:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("What is Concept Drift?")
                st.markdown("""
                Concept drift occurs when the relationship between input features and the target variable changes over time.
                This means the very concept that the model is trying to predict has changed.
                
                **Common causes:**
                - Changing customer preferences
                - Economic or market changes
                - Regulatory changes
                - Competitive landscape changes
                """)
                
                st.markdown("""
                <div class="info-box">
                <h3>Types of Concept Drift</h3>
                <ul>
                    <li><strong>Sudden Drift</strong>: Abrupt change from one concept to another</li>
                    <li><strong>Gradual Drift</strong>: Slow transition from one concept to another</li>
                    <li><strong>Incremental Drift</strong>: Concept changes incrementally over time</li>
                    <li><strong>Recurring Drift</strong>: Previously seen concepts reappear (e.g., seasonal patterns)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Select concept drift type
                drift_type = st.radio(
                    "Select concept drift type:",
                    ["Baseline vs Drift (Intercept Shift)", "Baseline vs Concept Change (Slope Shift)"]
                )
                
                # Plot concept drift
                if drift_type == "Baseline vs Drift (Intercept Shift)":
                    concept_fig = plot_concept_drift(
                        st.session_state.drift_data['feature_x'], 
                        st.session_state.drift_data['baseline_y'], 
                        st.session_state.drift_data['drift_y']
                    )
                else:
                    concept_fig = plot_concept_drift(
                        st.session_state.drift_data['feature_x'], 
                        st.session_state.drift_data['baseline_y'], 
                        st.session_state.drift_data['concept_drift_y']
                    )
                
                st.pyplot(concept_fig)
                st.caption("Visualization of the changing relationship between feature and target")
            
            # Concept drift detection
            st.subheader("Detecting Concept Drift")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>Error-Based Methods</h4>
                    <p>Monitor changes in model performance metrics over time.</p>
                    <ul>
                        <li>Track accuracy, precision, recall, F1 score</li>
                        <li>Implement statistical process control (SPC) to detect significant changes</li>
                        <li>Use CUSUM (Cumulative Sum) charts to detect small, persistent changes</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Distribution-Based Methods</h4>
                    <p>Monitor changes in prediction distributions over time.</p>
                    <ul>
                        <li>Compare prediction distributions between time periods</li>
                        <li>Use Kullback-Leibler divergence to measure distribution differences</li>
                        <li>Implement Hellinger distance for probability distributions</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Concept drift mitigation
            st.subheader("Mitigation Strategies")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>Ensemble Methods</h4>
                    <p>Use multiple models to adapt to changing concepts.</p>
                    <p><strong>Example:</strong> Weighted ensemble where weights adjust based on recent performance.</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Sliding Window Approaches</h4>
                    <p>Train models on recent data windows to adapt to current concepts.</p>
                    <p><strong>Example:</strong> ADWIN (Adaptive Windowing) that adjusts window size based on detected changes.</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div class="card">
                    <h4>Concept Evolution Tracking</h4>
                    <p>Maintain multiple models representing different concepts.</p>
                    <p><strong>Example:</strong> Model versioning with automated selection based on recent performance.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # PREDICTION DRIFT
        with drift_tab3:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("What is Prediction Drift?")
                st.markdown("""
                Prediction drift occurs when the statistical properties of the model's predictions change over time,
                even if the input data remains stable. This can indicate problems with the model or changes in how
                the model processes data.
                
                **Key indicators:**
                - Changes in the distribution of predicted values
                - Shifts in prediction confidence scores
                - Changes in error patterns
                - Increased bias in predictions for certain groups
                """)
                
                st.markdown("""
                <div class="info-box">
                <h3>Detection Methods</h3>
                <p>Ways to detect prediction drift:</p>
                <ul>
                    <li><strong>Statistical tests</strong> on prediction distributions over time</li>
                    <li><strong>Tracking prediction statistics</strong> (mean, variance, quantiles)</li>
                    <li><strong>Monitoring label distributions</strong> when available</li>
                    <li><strong>Confusion matrix evolution</strong> to track changing error patterns</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create a simple prediction drift visualization
                fig = go.Figure()
                
                # Generate example prediction distributions for visualization
                x = np.linspace(-3, 3, 200)
                
                # Baseline prediction distribution (normal)
                y1 = stats.norm.pdf(x, 0, 1)
                
                # Drifted prediction - shift and wider variance
                y2 = stats.norm.pdf(x, 0.5, 1.2)
                
                # Drifted prediction - bimodal
                y3 = 0.6 * stats.norm.pdf(x, -1, 0.8) + 0.4 * stats.norm.pdf(x, 1.5, 0.7)
                
                fig.add_trace(go.Scatter(
                    x=x, y=y1, 
                    mode='lines', 
                    name='Baseline Predictions',
                    line=dict(color='#00A1C9', width=3)
                ))
                
                # Add dropdown to switch between different drift patterns
                fig.add_trace(go.Scatter(
                    x=x, y=y2,
                    mode='lines',
                    name='Shifted Predictions',
                    line=dict(color='#FF9900', width=3),
                    visible=True
                ))
                
                fig.add_trace(go.Scatter(
                    x=x, y=y3,
                    mode='lines',
                    name='Pattern Change',
                    line=dict(color='#D13212', width=3),
                    visible=False
                ))
                
                # Add dropdown
                fig.update_layout(
                    updatemenus=[
                        dict(
                            active=0,
                            buttons=list([
                                dict(
                                    label="Shift in Distribution",
                                    method="update",
                                    args=[{"visible": [True, True, False]},
                                         {"title": "Prediction Drift: Shift in Distribution"}]
                                ),
                                dict(
                                    label="Pattern Change",
                                    method="update",
                                    args=[{"visible": [True, False, True]},
                                         {"title": "Prediction Drift: Pattern Change"}]
                                )
                            ]),
                            direction="down",
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.1,
                            xanchor="left",
                            y=1.15,
                            yanchor="top"
                        ),
                    ]
                )
                
                fig.update_layout(
                    title="Prediction Distribution Over Time",
                    xaxis_title="Prediction Value",
                    yaxis_title="Density",
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Impact of prediction drift
            st.subheader("Impact of Prediction Drift")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="warning-box">
                <h3>Business Impact</h3>
                <p>Prediction drift can lead to:</p>
                <ul>
                    <li>Decreased customer satisfaction</li>
                    <li>Revenue loss from poor recommendations</li>
                    <li>Increased operational costs from incorrect resource allocation</li>
                    <li>Compliance issues when models make unfair decisions</li>
                    <li>Loss of trust in ML systems</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Case Study: E-commerce Recommendation System</h4>
                    <p>An online retailer's recommendation system showed prediction drift during holiday season:</p>
                    <ul>
                        <li>Confidence scores shifted higher despite decreased accuracy</li>
                        <li>Recommendations became less diverse</li>
                        <li>Click-through rates dropped by 23%</li>
                    </ul>
                    <p><strong>Solution:</strong> Implemented seasonal model variants and continuous monitoring of recommendation quality metrics.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # FEATURE IMPORTANCE DRIFT
        with drift_tab4:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("What is Feature Importance Drift?")
                st.markdown("""
                Feature importance drift occurs when the relative importance of features in a model changes over time.
                This can indicate changes in the underlying relationships or emergent patterns in the data.
                
                **Why it matters:**
                - Reveals changing dynamics in your problem domain
                - Helps identify features that need more monitoring
                - Guides feature engineering efforts
                - Indicates when model architecture should be reconsidered
                """)
                
                st.markdown("""
                <div class="info-box">
                <h3>Detection Methods</h3>
                <p>Ways to detect feature importance drift:</p>
                <ul>
                    <li><strong>SHAP value monitoring</strong> over time</li>
                    <li><strong>Permutation importance</strong> tracking</li>
                    <li><strong>Model coefficients</strong> for linear models</li>
                    <li><strong>Feature attribution</strong> methods for black-box models</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create a feature importance drift visualization
                # Generate sample data for feature importance over time
                features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
                importance_t0 = [0.35, 0.25, 0.20, 0.15, 0.05]
                importance_t1 = [0.30, 0.15, 0.25, 0.20, 0.10]
                importance_t2 = [0.20, 0.10, 0.30, 0.25, 0.15]
                
                # Create figure
                fig = go.Figure()
                
                # Use dropdown to switch time periods
                fig.add_trace(go.Bar(
                    x=features,
                    y=importance_t0,
                    name='Baseline',
                    marker_color='#00A1C9'
                ))
                
                fig.add_trace(go.Bar(
                    x=features,
                    y=importance_t1,
                    name='Month 1',
                    marker_color='#FF9900',
                    visible=False
                ))
                
                fig.add_trace(go.Bar(
                    x=features,
                    y=importance_t2,
                    name='Month 2',
                    marker_color='#D13212',
                    visible=False
                ))
                
                # Add dropdown menu
                fig.update_layout(
                    updatemenus=[
                        dict(
                            active=0,
                            buttons=list([
                                dict(
                                    label="Baseline",
                                    method="update",
                                    args=[{"visible": [True, False, False]},
                                          {"title": "Feature Importance: Baseline"}]
                                ),
                                dict(
                                    label="Month 1",
                                    method="update",
                                    args=[{"visible": [True, True, False]},
                                          {"title": "Feature Importance: Baseline vs Month 1"}]
                                ),
                                dict(
                                    label="Month 2",
                                    method="update",
                                    args=[{"visible": [True, True, True]},
                                          {"title": "Feature Importance: Baseline vs Month 1 vs Month 2"}]
                                )
                            ]),
                            direction="down",
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.1,
                            xanchor="left",
                            y=1.15,
                            yanchor="top"
                        ),
                    ]
                )
                
                fig.update_layout(
                    title="Feature Importance Over Time",
                    xaxis_title="Feature",
                    yaxis_title="Importance",
                    barmode='group',
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.info("Notice how Feature C becomes increasingly important over time, while Feature A and B decrease in importance.")
            
            # Feature importance drift monitoring with SageMaker
            st.subheader("Monitoring Feature Attribution with SageMaker")
            
            st.code('''
# Set up a feature attribution monitor
from sagemaker.clarify import DataConfig, ModelConfig, ModelPredictedLabelConfig
from sagemaker.model_monitor import ClarifyModelMonitor

# Configure the model and data
data_config = DataConfig(
    s3_data_input_path='s3://my-bucket/monitor-input/',
    s3_output_path='s3://my-bucket/monitor-output/',
    label='target',
    headers=['feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e', 'target']
)

model_config = ModelConfig(
    model_name='my-model',
    instance_type='ml.m5.xlarge',
    instance_count=1
)

predicted_label_config = ModelPredictedLabelConfig(
    probability_threshold=0.5
)

# Create a feature attribution monitor
feature_monitor = ClarifyModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

# Create a monitoring schedule
feature_monitor.create_monitoring_schedule(
    monitor_schedule_name='feature-attribution-monitor',
    endpoint_input=endpoint_name,
    record_preprocessor_script='s3://my-bucket/preprocessor.py',
    post_analytics_processor_script='s3://my-bucket/postprocessor.py',
    output_s3_uri='s3://my-bucket/monitor-output/',
    statistics=baseline_stats,
    constraints=baseline_constraints,
    schedule_cron_expression='cron(0 * ? * * *)'
)
            ''')
            
            # Case study
            st.markdown("""
            <div class="card">
                <h4>Case Study: Financial Fraud Detection</h4>
                <p>A bank's fraud detection system experienced feature importance drift:</p>
                <ul>
                    <li>Initially, transaction amount and time were top predictive features</li>
                    <li>Over time, location data and device information became more important</li>
                    <li>This shift coincided with fraudsters changing tactics to use stolen credentials from trusted devices</li>
                </ul>
                <p><strong>Solution:</strong> The bank implemented feature importance monitoring to automatically detect emerging fraud patterns and adjust feature engineering.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 3: MODEL TESTING STRATEGIES
    with tab3:
        st.header("Model Testing Strategies")
        
        st.markdown("""
        Effective model testing strategies are crucial for detecting drift early and ensuring model reliability.
        SageMaker Model Monitor provides several approaches to test and validate models before and during deployment.
        """)
        
        # Testing strategies
        st.subheader("Key Testing Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Shadow Deployment</h4>
                <p>Deploy a new model version alongside the production model to compare performance without affecting users.</p>
                <p><strong>How it works:</strong></p>
                <ol>
                    <li>Deploy new model version as a shadow endpoint</li>
                    <li>Route a copy of production traffic to shadow model</li>
                    <li>Compare predictions and performance metrics</li>
                    <li>Validate model before full deployment</li>
                </ol>
                <p><strong>Benefits:</strong> Zero risk to users, direct comparison with current model, real-world validation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>A/B Testing</h4>
                <p>Route a percentage of traffic to different model versions to compare performance with real users.</p>
                <p><strong>How it works:</strong></p>
                <ol>
                    <li>Deploy multiple model variants to the same endpoint</li>
                    <li>Configure traffic distribution between variants</li>
                    <li>Collect performance metrics for each variant</li>
                    <li>Gradually shift traffic to better performing models</li>
                </ol>
                <p><strong>Benefits:</strong> Direct user feedback, statistical validation, gradual rollout</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Canary Deployment</h4>
                <p>Deploy a new model to a small percentage of users before full rollout.</p>
                <p><strong>How it works:</strong></p>
                <ol>
                    <li>Deploy new model version to production</li>
                    <li>Route a small percentage (e.g., 5%) of traffic to it</li>
                    <li>Monitor performance closely</li>
                    <li>Gradually increase traffic if performance is good</li>
                    <li>Roll back immediately if issues arise</li>
                </ol>
                <p><strong>Benefits:</strong> Minimizes impact of issues, controlled exposure, easier rollback</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>Champion/Challenger</h4>
                <p>Continuously test new model candidates against your current best model.</p>
                <p><strong>How it works:</strong></p>
                <ol>
                    <li>Maintain current best model as "champion"</li>
                    <li>Deploy promising new models as "challengers"</li>
                    <li>Allocate small traffic portions to challengers</li>
                    <li>Compare performance metrics</li>
                    <li>Promote challenger to champion if it outperforms</li>
                </ol>
                <p><strong>Benefits:</strong> Continuous improvement, competitive evaluation, performance focus</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Implementation with SageMaker
        st.subheader("Implementation with SageMaker")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Amazon SageMaker provides built-in capabilities for implementing these testing strategies:
            
            1. **SageMaker Multi-Model Endpoints** - Host multiple models on a single endpoint
            2. **SageMaker Model Registry** - Version and track model artifacts
            3. **Production Variants** - Configure traffic distribution between models 
            4. **Automatic Scaling** - Scale endpoints based on traffic patterns
            5. **Model Monitor** - Track drift and performance for all variants
            
            These capabilities make it easy to implement sophisticated testing strategies without building custom infrastructure.
            """)
            
            # Example code
            st.code('''
# A/B testing with production variants
import boto3
from sagemaker.model import Model

# Create client
sagemaker_client = boto3.client('sagemaker')

# Create models
model_a = Model(
    image_uri='<ecr-image-uri>',
    model_data='s3://my-bucket/model-a/model.tar.gz',
    role='<role-arn>'
)

model_b = Model(
    image_uri='<ecr-image-uri>',
    model_data='s3://my-bucket/model-b/model.tar.gz',
    role='<role-arn>'
)

# Deploy models as separate production variants
model_a.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='my-ab-endpoint',
    production_variants=[
        {
            'VariantName': 'ModelA',
            'ModelName': 'model-a',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge',
            'InitialVariantWeight': 0.7
        },
        {
            'VariantName': 'ModelB',
            'ModelName': 'model-b',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge',
            'InitialVariantWeight': 0.3
        }
    ]
)

# Update variant weights as needed
sagemaker_client.update_endpoint_weights_and_capacities(
    EndpointName='my-ab-endpoint',
    DesiredWeightsAndCapacities=[
        {
            'VariantName': 'ModelA',
            'DesiredWeight': 0.5
        },
        {
            'VariantName': 'ModelB',
            'DesiredWeight': 0.5
        }
    ]
)
            ''')
        
        with col2:
            # Add visualization showing A/B testing or canary deployment
            
            # Create sample data for A/B testing visualization
            fig = go.Figure()
            
            # Timeline
            days = 10
            x = list(range(1, days + 1))
            
            # Model A traffic (starts at 80%, decreases to 50%)
            model_a_traffic = [80 - (30 / days) * i for i in range(days)]
            
            # Model B traffic (starts at 20%, increases to 50%)
            model_b_traffic = [20 + (30 / days) * i for i in range(days)]
            
            # Model A metrics (stable, around 0.92)
            model_a_metrics = [0.92 + 0.01 * np.sin(i) for i in range(days)]
            
            # Model B metrics (improving from 0.89 to 0.94)
            model_b_metrics = [0.89 + (0.05 / days) * i for i in range(days)]
            
            # Create plot with dual y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add traces for traffic
            fig.add_trace(
                go.Scatter(
                    x=x, y=model_a_traffic,
                    name="Model A Traffic %",
                    line=dict(color='#00A1C9', width=3, dash='dash'),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x, y=model_b_traffic,
                    name="Model B Traffic %",
                    line=dict(color='#FF9900', width=3, dash='dash'),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            # Add traces for metrics
            fig.add_trace(
                go.Scatter(
                    x=x, y=model_a_metrics,
                    name="Model A Accuracy",
                    line=dict(color='#00A1C9', width=3),
                    mode='lines+markers'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x, y=model_b_metrics,
                    name="Model B Accuracy",
                    line=dict(color='#FF9900', width=3),
                    mode='lines+markers'
                ),
                secondary_y=False
            )
            
            # Add figure layout
            fig.update_layout(
                title="A/B Testing: Traffic Distribution and Model Performance",
                xaxis=dict(title="Day"),
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Model Accuracy", secondary_y=False, range=[0.85, 0.95])
            fig.update_yaxes(title_text="Traffic Distribution (%)", secondary_y=True, range=[0, 100])
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Visualization of A/B testing with gradual traffic shift as Model B shows improving performance")
        
        # Best practices
        st.subheader("Testing Best Practices")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="status-card status-1">
                <h4>Define Clear Metrics</h4>
                <ul>
                    <li>Identify key performance indicators</li>
                    <li>Set precise success criteria</li>
                    <li>Monitor business and technical metrics</li>
                    <li>Define statistical significance thresholds</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="status-card status-3">
                <h4>Start Small, Scale Gradually</h4>
                <ul>
                    <li>Begin with low-risk segments</li>
                    <li>Increase traffic gradually</li>
                    <li>Monitor closely during scaling</li>
                    <li>Have rollback procedures ready</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="status-card status-2">
                <h4>Implement Automated Guardrails</h4>
                <ul>
                    <li>Set automatic alerts for issues</li>
                    <li>Define traffic cutoffs for poor performance</li>
                    <li>Create auto-rollback triggers</li>
                    <li>Log all test results for analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Advanced SageMaker monitoring
        st.subheader("Advanced SageMaker Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h3>Custom Monitoring with Processing Jobs</h3>
            <p>For advanced monitoring scenarios, SageMaker Processing Jobs allow you to:</p>
            <ul>
                <li>Run custom analysis scripts on captured data</li>
                <li>Implement domain-specific drift detection algorithms</li>
                <li>Generate custom visualizations and reports</li>
                <li>Schedule jobs to run on your preferred frequency</li>
                <li>Integrate with your existing monitoring workflow</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.code('''
# Example: Custom monitoring with Processing Jobs
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# Create processor
processor = SKLearnProcessor(
    framework_version='0.23-1',
    role='<role-arn>',
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Run processing job
processor.run(
    code='custom_monitor.py',
    inputs=[
        ProcessingInput(
            source='s3://my-bucket/endpoint-data/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='results',
            source='/opt/ml/processing/output',
            destination='s3://my-bucket/monitoring-results/'
        )
    ]
)
            ''')
            
        with col2:
            st.markdown("""
            <div class="warning-box">
            <h3>Automated Response Workflows</h3>
            <p>When drift is detected, automate your response:</p>
            <ul>
                <li><strong>Amazon EventBridge</strong> can trigger workflows on monitoring alerts</li>
                <li><strong>AWS Step Functions</strong> can orchestrate retraining and deployment</li>
                <li><strong>AWS Lambda</strong> can perform automated traffic rebalancing</li>
                <li><strong>Amazon SNS</strong> can notify stakeholders of detected issues</li>
            </ul>
            <p>This allows for a closed-loop system that can self-heal when drift is detected.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.image("https://d1.awsstatic.com/reInvent/reinvent-2022/mls201/SageMaker-Model-Monitor-How-It-Works.2f7a5ebb1478214486ea902b0dc1ddcc556f2185.png", 
                    caption="SageMaker Model Monitor Architecture", 
                    use_container_width=True)
            
    
    # TAB 4: TEST MODELS WITH TRAFFIC DISTRIBUTION
    with tab4:
        st.header("Test Models with Traffic Distribution")
        
        st.markdown("""
        In this interactive section, you can experiment with different traffic distributions 
        between model variants and observe how it affects overall model performance and reliability.
        """)
        
        # Configuration section
        st.subheader("Configure Traffic Distribution")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Traffic distribution slider
            st.markdown("#### Traffic Distribution (%) between Model A and Model B")
            
            # Use the slider to distribute traffic between models
            model_a_traffic = st.slider("Model A Traffic %", 0, 100, st.session_state.traffic_distribution[0], 10)
            model_b_traffic = 100 - model_a_traffic
            
            # Update session state
            st.session_state.traffic_distribution = [model_a_traffic, model_b_traffic]
            
            # Display the current distribution
            col1a, col1b = st.columns(2)
            
            with col1a:
                st.metric(
                    label="Model A Traffic",
                    value=f"{model_a_traffic}%",
                    delta=None
                )
                
                st.markdown("""
                <div class="status-card status-1">
                    <h5>Model A Characteristics</h5>
                    <ul>
                        <li>Established model with stable performance</li>
                        <li>Well understood behavior</li>
                        <li>Higher latency (avg. 120ms)</li>
                        <li>Less susceptible to data drift</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col1b:
                st.metric(
                    label="Model B Traffic",
                    value=f"{model_b_traffic}%",
                    delta=None
                )
                
                st.markdown("""
                <div class="status-card status-3">
                    <h5>Model B Characteristics</h5>
                    <ul>
                        <li>Newer model with potentially better accuracy</li>
                        <li>Less established behavior</li>
                        <li>Lower latency (avg. 90ms)</li>
                        <li>More sensitive to certain types of data</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Test duration and options
            st.markdown("#### Test Configuration")
            
            test_duration = st.slider("Test Duration (minutes)", 5, 60, 10, 5)
            
            drift_scenario = st.selectbox(
                "Drift Scenario during Test",
                ["No significant drift", "Gradual feature drift", "Sudden concept drift"]
            )
            
            # Run test button
            if st.button("Run Model Test", type="primary"):
                # Show progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate test execution with progress updates
                for i in range(100):
                    # Update progress bar
                    progress_bar.progress(i + 1)
                    
                    # Update status text periodically
                    if i % 20 == 0:
                        phase = ["Initializing test", "Collecting data", "Processing results", "Analyzing metrics", "Finalizing report"][i // 20]
                        status_text.text(f"{phase}... ({i+1}%)")
                    
                    # Short delay to simulate processing
                    time.sleep(0.05)
                
                # Complete the test
                status_text.text("Test completed! Displaying results...")
                
                # Run the actual test and store results
                st.session_state.model_test_results = run_model_test(st.session_state.traffic_distribution, test_duration)
                
                # Delay to show the completion message
                time.sleep(1)
                
                # Clear the progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show success message
                st.success("Test completed successfully! See results below.")
        
        # Display test results if available
        if st.session_state.model_test_results is not None:
            st.header("Test Results")
            
            # Get test results
            test_results = st.session_state.model_test_results
            
            # Create dashboard visualizations
            dashboard_figs = create_monitor_dashboard(test_results)
            
            if dashboard_figs:
                # Display performance comparison
                st.subheader("Model Performance Comparison")
                st.plotly_chart(dashboard_figs["metrics_fig"], use_container_width=True)
                
                # Display accuracy over time
                st.subheader("Model Accuracy Over Time")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(dashboard_figs["time_fig"], use_container_width=True)
                
                with col2:
                    # Calculate combined metrics based on traffic distribution
                    model_a_weight = test_results["model_a_traffic"]
                    model_b_weight = test_results["model_b_traffic"]
                    
                    combined_accuracy = model_a_weight * test_results["model_a_metrics"]["accuracy"] + model_b_weight * test_results["model_b_metrics"]["accuracy"]
                    combined_f1 = model_a_weight * test_results["model_a_metrics"]["f1"] + model_b_weight * test_results["model_b_metrics"]["f1"]
                    
                    # Display metrics
                    st.subheader("Combined Model Performance")
                    st.metric(
                        label="Combined Accuracy",
                        value=f"{combined_accuracy:.2%}",
                        delta=f"{combined_accuracy - 0.90:.2%}" if combined_accuracy > 0.90 else f"{combined_accuracy - 0.90:.2%}"
                    )
                    
                    st.metric(
                        label="Combined F1 Score",
                        value=f"{combined_f1:.2%}",
                        delta=f"{combined_f1 - 0.88:.2%}" if combined_f1 > 0.88 else f"{combined_f1 - 0.88:.2%}"
                    )
                    
                    # Recommendation based on results
                    st.markdown("### Recommendation")
                    
                    if combined_accuracy > 0.92:
                        st.markdown("""
                        <div class="status-card status-1">
                            <h4>✅ Optimal Configuration</h4>
                            <p>Current traffic distribution is performing well. Continue monitoring for potential drift.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif combined_accuracy > 0.89:
                        st.markdown("""
                        <div class="status-card status-3">
                            <h4>⚠️ Acceptable Configuration</h4>
                            <p>Performance is acceptable but could be improved. Consider adjusting traffic distribution.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="status-card status-2">
                            <h4>❌ Suboptimal Configuration</h4>
                            <p>Current traffic distribution is not performing well. Adjust traffic to favor better performing model.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display latency comparison
                st.subheader("Model Latency Comparison")
                st.plotly_chart(dashboard_figs["latency_fig"], use_container_width=True)
                
                # Display drift comparison
                st.subheader("Feature Drift Analysis")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(dashboard_figs["drift_fig"], use_container_width=True)
                
                with col2:
                    # Add interpretation of drift results
                    st.markdown("### Drift Detection")
                    
                    # Check if any feature has high drift in either model
                    max_drift_a = max([test_results["model_a_metrics"]["feature_stats"][f]["drift_score"] for f in ["feature1", "feature2", "feature3", "feature4"]])
                    max_drift_b = max([test_results["model_b_metrics"]["feature_stats"][f]["drift_score"] for f in ["feature1", "feature2", "feature3", "feature4"]])
                    
                    if max(max_drift_a, max_drift_b) > 0.2:
                        st.markdown("""
                        <div class="status-card status-2">
                            <h4>⚠️ Significant Drift Detected</h4>
                            <p>At least one feature shows significant drift from the baseline distribution.</p>
                            <p><strong>Recommendation:</strong> Investigate feature drift and consider retraining models.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif max(max_drift_a, max_drift_b) > 0.1:
                        st.markdown("""
                        <div class="status-card status-3">
                            <h4>⚠️ Moderate Drift Detected</h4>
                            <p>Some features show moderate drift from the baseline distribution.</p>
                            <p><strong>Recommendation:</strong> Continue monitoring and prepare for potential retraining.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="status-card status-1">
                            <h4>✅ No Significant Drift</h4>
                            <p>Features are stable and consistent with baseline distributions.</p>
                            <p><strong>Recommendation:</strong> Continue with current configuration.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Sample data display
                st.subheader("Sample Inference Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Model A Sample Data")
                    st.dataframe(test_results["model_a_sample"].reset_index(drop=True))
                
                with col2:
                    st.markdown("#### Model B Sample Data")
                    st.dataframe(test_results["model_b_sample"].reset_index(drop=True))
            
            # Key insights section
            st.subheader("Key Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>Performance Insights</h4>
                    <p>Model A shows stable performance but doesn't adapt as quickly to changing patterns.</p>
                    <p>Model B has slightly higher overall accuracy but is more sensitive to data drift.</p>
                    <p>Optimal traffic split depends on your tolerance for risk vs. reward.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Operational Insights</h4>
                    <p>Model A has higher average latency but more consistent response times.</p>
                    <p>Model B is faster but shows more latency variance under load.</p>
                    <p>Consider peak traffic requirements when setting distribution.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="card">
                    <h4>Monitoring Insights</h4>
                    <p>Feature drift is more pronounced in Model B, suggesting it may need more frequent retraining.</p>
                    <p>Consider implementing automated retraining triggers based on drift magnitude.</p>
                    <p>Set up alerts for significant accuracy drops in either model.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show instructions if no test has been run
            st.info("Configure the traffic distribution between models and click 'Run Model Test' to see results.")

    # TAB 5: KNOWLEDGE CHECK
    with tab5:
        custom_header("Test Your Knowledge")
        
        st.markdown("""
        This quiz will test your understanding of the key concepts covered in Amazon SageMaker Model Monitor.
        Answer the following questions to evaluate your knowledge of model drift and monitoring strategies.
        """)
        
        # Define quiz questions
        questions = [
            {
                "question": "Which type of drift occurs when the statistical properties of the input data change over time?",
                "options": ["Concept Drift", "Data Drift", "Prediction Drift", "Feature Importance Drift"],
                "correct": "Data Drift",
                "explanation": "Data drift occurs when the statistical properties of the input data change over time, causing the model's predictions to become less accurate. This is also known as covariate shift."
            },
            {
                "question": "Which SageMaker Model Monitor capability is used to detect changes in how features contribute to model predictions?",
                "options": ["Data Quality Monitoring", "Model Quality Monitoring", "Bias Drift Monitoring", "Feature Attribution Monitoring"],
                "correct": "Feature Attribution Monitoring",
                "explanation": "Feature Attribution Monitoring in SageMaker is used to track changes in feature importance over time, detecting when the contribution of individual features to model predictions differs from what was observed during training."
            },
            {
                "question": "Which testing strategy involves deploying a new model alongside the production model without affecting users?",
                "options": ["A/B Testing", "Canary Deployment", "Shadow Deployment", "Champion/Challenger"],
                "correct": "Shadow Deployment",
                "explanation": "Shadow Deployment involves deploying a new model version alongside the production model to compare performance without affecting users. The shadow model receives a copy of the production traffic but its predictions are not returned to users."
            },
            {
                "question": "Which type of drift occurs when the relationship between input features and the target variable changes over time?",
                "options": ["Concept Drift", "Data Drift", "Prediction Drift", "Feature Importance Drift"],
                "correct": "Concept Drift",
                "explanation": "Concept drift occurs when the relationship between input features and the target variable changes over time. This means the very concept that the model is trying to predict has changed."
            },
            {
                "question": "Which of the following is a key step in setting up SageMaker Model Monitor?",
                "options": ["Creating a data baseline", "Changing the model architecture", "Implementing A/B testing", "Designing a new UI"],
                "correct": "Creating a data baseline",
                "explanation": "Creating a data baseline is a key step in setting up SageMaker Model Monitor. The baseline statistics are generated from training data and used to define constraints against which future data will be validated."
            },
            {
                "question": "What is a common method for detecting data drift in model inputs?",
                "options": ["LSTM networks", "Kolmogorov-Smirnov (K-S) test", "Regularization", "Hyperparameter tuning"],
                "correct": "Kolmogorov-Smirnov (K-S) test",
                "explanation": "The Kolmogorov-Smirnov (K-S) test is a statistical method commonly used to detect data drift by measuring the maximum difference between two cumulative distribution functions, comparing baseline and current data distributions."
            },
            {
                "question": "Which deployment strategy routes a small percentage of traffic to a new model before full rollout?",
                "options": ["Blue-Green Deployment", "Canary Deployment", "Shadow Deployment", "Rolling Deployment"],
                "correct": "Canary Deployment",
                "explanation": "Canary Deployment involves routing a small percentage of traffic (e.g., 5%) to a new model before full rollout. This allows monitoring performance with limited user impact and easy rollback if issues arise."
            }
        ]
        
        # Check if the quiz has been attempted
        if not st.session_state['quiz_attempted']:
            # Create a form for the quiz
            with st.form("quiz_form"):
                st.markdown("### Answer the following questions:")
                
                # Track user answers
                user_answers = []
                
                # Display 5 random questions
                np.random.seed(42)  # For reproducibility
                selected_questions = np.random.choice(questions, size=5, replace=False)
                
                # Display each question
                for i, q in enumerate(selected_questions):
                    st.markdown(f"**Question {i+1}:** {q['question']}")
                    answer = st.radio(f"Select your answer for question {i+1}:", q['options'], key=f"q{i}", index=None)
                    user_answers.append((answer, q['correct'], q['explanation']))
                
                # Submit button
                submitted = st.form_submit_button("Submit Quiz")
                
                if submitted:
                    # Calculate score
                    score = sum([1 for ua, corr, _ in user_answers if ua == corr])
                    st.session_state['quiz_score'] = score
                    st.session_state['quiz_attempted'] = True
                    st.session_state['quiz_answers'] = user_answers
                    st.rerun()
        else:
            # Display results
            score = st.session_state['quiz_score']
            user_answers = st.session_state.get('quiz_answers', [])
            
            st.markdown(f"### Your Score: {score}/5")
            
            if score == 5:
                st.success("🎉 Perfect score! You've mastered the concepts of SageMaker Model Monitor!")
            elif score >= 3:
                st.success("👍 Good job! You have a solid understanding of model monitoring concepts.")
            else:
                st.warning("📚 You might want to review the content again to strengthen your understanding of model monitoring.")
            
            # Show correct answers
            st.markdown("### Review Questions and Answers:")
            
            for i, (user_answer, correct_answer, explanation) in enumerate(user_answers):
                st.markdown(f"**Question {i+1}**")
                st.markdown(f"**Your answer:** {user_answer}")
                
                if user_answer == correct_answer:
                    st.markdown(f"**✅ Correct!**")
                else:
                    st.markdown(f"**❌ Incorrect. The correct answer is:** {correct_answer}")
                
                st.markdown(f"**Explanation:** {explanation}")
                
                if i < len(user_answers) - 1:
                    st.markdown("---")
            
            # Option to retake the quiz
            if st.button("Retake Quiz"):
                st.session_state['quiz_attempted'] = False
                st.rerun()



    # Add footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
