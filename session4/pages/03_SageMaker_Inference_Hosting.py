
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import uuid
import random
from datetime import datetime, timedelta
from PIL import Image
import io
import base64
from streamlit_lottie import st_lottie
import requests
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union


# Custom utility functions
def load_lottieurl(url: str) -> Dict:
    """
    Load a Lottie animation from a URL
    
    Args:
        url: URL of the Lottie animation JSON
    
    Returns:
        Lottie animation as a dictionary, or None if loading fails
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def create_inference_flow_diagram(diagram_type: str, fig_width: int = 10, fig_height: int = 6) -> plt.Figure:
    """
    Create a flow diagram showing the inference process for different endpoint types
    
    Args:
        diagram_type: Type of endpoint to illustrate (multi_model, multi_container, pipeline)
        fig_width: Figure width in inches
        fig_height: Figure height in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # AWS Colors
    AWS_COLORS = {
        "orange": "#FF9900",
        "teal": "#00A1C9", 
        "blue": "#232F3E",
        "gray": "#E9ECEF",
        "light_gray": "#F8F9FA",
        "white": "#FFFFFF",
        "dark_gray": "#545B64",
        "green": "#59BA47",
        "red": "#D13212",
        "purple": "#9D1F63"
    }
    
    # Common elements
    # Client
    client = plt.Rectangle((0.5, 4.5), 1.5, 1, fc=AWS_COLORS["blue"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
    ax.add_patch(client)
    ax.text(1.25, 5, "Client", ha='center', va='center', color='white', fontweight='bold')
    
    if diagram_type == "multi_model":
        # Multi-model endpoint
        endpoint = plt.Rectangle((3.5, 4), 3, 2, fc=AWS_COLORS["orange"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(endpoint)
        ax.text(5, 5, "Multi-Model Endpoint", ha='center', va='center', color='white', fontweight='bold')
        
        # Model server
        model_server = plt.Rectangle((4, 4.2), 2, 0.5, fc=AWS_COLORS["white"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(model_server)
        ax.text(5, 4.45, "Model Server", ha='center', va='center', color='black', fontsize=9)
        
        # S3 bucket
        s3 = plt.Rectangle((3.5, 1), 3, 1, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(s3)
        ax.text(5, 1.5, "S3 Model Repository", ha='center', va='center', color='white', fontweight='bold')
        
        # Model A, B, C (in S3)
        model_a_s3 = plt.Rectangle((3.7, 1.2), 0.7, 0.5, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        model_b_s3 = plt.Rectangle((4.6, 1.2), 0.7, 0.5, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        model_c_s3 = plt.Rectangle((5.5, 1.2), 0.7, 0.5, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(model_a_s3)
        ax.add_patch(model_b_s3)
        ax.add_patch(model_c_s3)
        ax.text(4.05, 1.45, "Model A", ha='center', va='center', fontsize=8)
        ax.text(4.95, 1.45, "Model B", ha='center', va='center', fontsize=8)
        ax.text(5.85, 1.45, "Model C", ha='center', va='center', fontsize=8)
        
        # Model cache
        cache = plt.Rectangle((4, 2.5), 2, 0.8, fc=AWS_COLORS["green"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(cache)
        ax.text(5, 2.9, "Model Cache", ha='center', va='center', color='white', fontweight='bold')
        
        # Model A, B in cache
        model_a_cache = plt.Rectangle((4.2, 2.6), 0.7, 0.5, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        model_b_cache = plt.Rectangle((5.1, 2.6), 0.7, 0.5, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(model_a_cache)
        ax.add_patch(model_b_cache)
        ax.text(4.55, 2.85, "Model A", ha='center', va='center', fontsize=8)
        ax.text(5.45, 2.85, "Model B", ha='center', va='center', fontsize=8)
        
        # Arrows
        # Client to endpoint
        ax.arrow(2.2, 5, 1.1, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(2.7, 5.2, "Request with\nmodel name", ha='center', va='bottom', fontsize=8)
        
        # Model retrieval paths
        # 1. Check cache
        ax.arrow(5, 4, 0, -0.5, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(5.3, 3.7, "1. Check\ncache", ha='left', va='center', fontsize=8)
        
        # 2. Load from S3 if not in cache
        ax.arrow(5, 2.3, 0, -0.5, head_width=0.15, head_length=0.15, 
                linestyle='dashed', fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(5.3, 2, "2. Load from S3\nif needed", ha='left', va='center', fontsize=8)
        
        # Response to client
        ax.arrow(3.5, 4.5, -1.3, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"])
        ax.text(2.7, 4.3, "Response", ha='center', va='top', color=AWS_COLORS["green"], fontsize=8)
        
        # Add new request with different model
        ax.arrow(1.5, 4.3, 2, -1, head_width=0.15, head_length=0.15, 
                linestyle='dotted', fc=AWS_COLORS["blue"], ec=AWS_COLORS["blue"])
        ax.text(2.7, 3.8, "Different model request", ha='center', va='center', color=AWS_COLORS["blue"], fontsize=8)
        
        # Add text highlighting key characteristics
        ax.text(8.5, 4, "• Host multiple models on\n  a single endpoint\n• Dynamic model loading\n• Memory-efficient\n• Ideal for serving many\n  similar models", 
               ha='left', va='center', fontsize=10, bbox=dict(facecolor=AWS_COLORS["light_gray"], alpha=0.5))
        
    elif diagram_type == "multi_container":
        # Multi-container endpoint
        endpoint = plt.Rectangle((3.5, 4), 3, 2, fc=AWS_COLORS["orange"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(endpoint)
        ax.text(5, 5.5, "Multi-Container Endpoint", ha='center', va='center', color='white', fontweight='bold')
        
        # Production variant containers
        container1 = plt.Rectangle((3.8, 4.7), 1.2, 0.6, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        container2 = plt.Rectangle((5, 4.7), 1.2, 0.6, fc=AWS_COLORS["green"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(container1)
        ax.add_patch(container2)
        ax.text(4.4, 5, "Model A\nContainer", ha='center', va='center', color='white', fontsize=8)
        ax.text(5.6, 5, "Model B\nContainer", ha='center', va='center', color='white', fontsize=8)
        
        # Inference results
        result1 = plt.Rectangle((3.8, 4.1), 1.2, 0.4, fc=AWS_COLORS["white"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        result2 = plt.Rectangle((5, 4.1), 1.2, 0.4, fc=AWS_COLORS["white"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(result1)
        ax.add_patch(result2)
        ax.text(4.4, 4.3, "Results 70%", ha='center', va='center', fontsize=8)
        ax.text(5.6, 4.3, "Results 30%", ha='center', va='center', fontsize=8)
        
        # Traffic split controller
        controller = plt.Rectangle((4, 3), 2, 0.6, fc=AWS_COLORS["blue"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(controller)
        ax.text(5, 3.3, "Traffic Split Controller", ha='center', va='center', color='white', fontsize=9)
        
        # CloudWatch monitoring
        cloudwatch = plt.Rectangle((7.5, 3), 2, 1, fc=AWS_COLORS["purple"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(cloudwatch)
        ax.text(8.5, 3.5, "CloudWatch\nMonitoring", ha='center', va='center', color='white', fontweight='bold')
        
        # Arrows
        # Client to endpoint
        ax.arrow(2.2, 5, 1.1, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(2.7, 5.2, "Request", ha='center', va='bottom', fontsize=8)
        
        # Traffic split to containers
        ax.arrow(5, 4, 0, -0.2, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Controller to both models (showing traffic split)
        arrow1 = plt.Arrow(4.7, 3.6, -0.1, 0.4, width=0.15, fc=AWS_COLORS["dark_gray"], alpha=0.8)
        arrow2 = plt.Arrow(5.3, 3.6, 0.1, 0.4, width=0.15, fc=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
        ax.text(4.6, 3.8, "70%", ha='center', va='center', fontsize=8)
        ax.text(5.4, 3.8, "30%", ha='center', va='center', fontsize=8)
        
        # Response to client
        ax.arrow(3.5, 4.5, -1.3, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"])
        ax.text(2.7, 4.3, "Response", ha='center', va='top', color=AWS_COLORS["green"], fontsize=8)
        
        # Monitoring arrows
        ax.arrow(6.6, 4.5, 0.7, -1, head_width=0.15, head_length=0.15, 
                linestyle='dotted', fc=AWS_COLORS["purple"], ec=AWS_COLORS["purple"])
        ax.text(7, 4, "Metrics", ha='center', va='center', color=AWS_COLORS["purple"], fontsize=8)
        
        # Add text highlighting key characteristics
        ax.text(8, 5, "• Host multiple models with\n  different containers\n• A/B testing capabilities\n• Traffic splitting\n• Easily compare model\n  performance", 
               ha='left', va='center', fontsize=10, bbox=dict(facecolor=AWS_COLORS["light_gray"], alpha=0.5))
        
    else:  # Inference pipeline
        # Inference pipeline endpoint
        endpoint = plt.Rectangle((3.5, 4), 5, 1.5, fc=AWS_COLORS["orange"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(endpoint)
        ax.text(6, 5.2, "Inference Pipeline Endpoint", ha='center', va='center', color='white', fontweight='bold')
        
        # Pipeline containers
        container1 = plt.Rectangle((4, 4.3), 1, 0.8, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        container2 = plt.Rectangle((5.5, 4.3), 1, 0.8, fc=AWS_COLORS["blue"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        container3 = plt.Rectangle((7, 4.3), 1, 0.8, fc=AWS_COLORS["green"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(container1)
        ax.add_patch(container2)
        ax.add_patch(container3)
        ax.text(4.5, 4.7, "Pre-processing\nContainer", ha='center', va='center', color='white', fontsize=8)
        ax.text(6, 4.7, "Inference\nContainer", ha='center', va='center', color='white', fontsize=8)
        ax.text(7.5, 4.7, "Post-processing\nContainer", ha='center', va='center', color='white', fontsize=8)
        
        # Data flow
        arrow1 = plt.Arrow(5.1, 4.7, 0.3, 0, width=0.1, fc=AWS_COLORS["dark_gray"])
        arrow2 = plt.Arrow(6.6, 4.7, 0.3, 0, width=0.1, fc=AWS_COLORS["dark_gray"])
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
        
        # Data transformation visualizations
        raw_data = plt.Rectangle((4.5, 2.8), 1, 0.6, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        processed_data = plt.Rectangle((6, 2.8), 1, 0.6, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        results = plt.Rectangle((7.5, 2.8), 1, 0.6, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(raw_data)
        ax.add_patch(processed_data)
        ax.add_patch(results)
        ax.text(5, 3.1, "Raw Data", ha='center', va='center', fontsize=8)
        ax.text(6.5, 3.1, "Processed\nData", ha='center', va='center', fontsize=8)
        ax.text(8, 3.1, "Final\nResults", ha='center', va='center', fontsize=8)
        
        # Data transformation arrows
        ax.arrow(4.5, 4.3, 0, -0.7, head_width=0.1, head_length=0.1, 
                linestyle='dotted', fc=AWS_COLORS["teal"], ec=AWS_COLORS["teal"])
        ax.arrow(6, 4.3, 0, -0.7, head_width=0.1, head_length=0.1, 
                linestyle='dotted', fc=AWS_COLORS["blue"], ec=AWS_COLORS["blue"])
        ax.arrow(7.5, 4.3, 0, -0.7, head_width=0.1, head_length=0.1, 
                linestyle='dotted', fc=AWS_COLORS["green"], ec=AWS_COLORS["green"])
        
        # Arrows
        # Client to endpoint
        ax.arrow(2.2, 5, 1.1, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(2.7, 5.2, "Request", ha='center', va='bottom', fontsize=8)
        
        # Response to client
        ax.arrow(3.5, 4.7, -1.3, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"])
        ax.text(2.7, 4.5, "Response", ha='center', va='top', color=AWS_COLORS["green"], fontsize=8)
        
        # Add text highlighting key characteristics
        ax.text(4.5, 1.5, "• Sequential processing pipeline\n• Data transformation between containers\n• Specialized containers for different tasks\n• Ideal for complex ML workflows", 
               ha='left', va='center', fontsize=10, bbox=dict(facecolor=AWS_COLORS["light_gray"], alpha=0.5))
        
    return fig


def create_performance_chart(endpoint_type: str, instance_count: int = 2) -> go.Figure:
    """
    Create a performance comparison chart for different endpoint types
    
    Args:
        endpoint_type: Type of endpoint (multi_model, multi_container, pipeline)
        instance_count: Number of instances for calculation
        
    Returns:
        Plotly Figure object
    """
    # AWS Colors
    AWS_COLORS = {
        "orange": "#FF9900",
        "teal": "#00A1C9", 
        "blue": "#232F3E",
        "gray": "#E9ECEF",
        "light_gray": "#F8F9FA",
        "white": "#FFFFFF",
        "dark_gray": "#545B64",
        "green": "#59BA47",
        "red": "#D13212",
        "purple": "#9D1F63"
    }
    
    # Define base metrics
    if endpoint_type == "multi_model":
        # Define comparison metrics for multi-model vs traditional endpoints
        model_counts = [1, 5, 10, 20, 50]
        traditional_costs = [instance_count * 1.0 * m for m in model_counts]  # 1.0 per instance hour per model
        multi_model_costs = [instance_count * 1.2] * 5  # Flat line, slightly higher instance cost
        
        # Create the figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=model_counts,
            y=traditional_costs,
            mode='lines+markers',
            name='Traditional Endpoints',
            line=dict(color=AWS_COLORS["blue"], width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=model_counts,
            y=multi_model_costs,
            mode='lines+markers',
            name='Multi-Model Endpoint',
            line=dict(color=AWS_COLORS["orange"], width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Cost Comparison: Multi-Model vs Traditional Endpoints",
            xaxis_title="Number of Models",
            yaxis_title="Hourly Cost ($)",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add cost savings annotations
        for i, m in enumerate(model_counts):
            if i > 0:  # Skip the first point
                savings = traditional_costs[i] - multi_model_costs[i]
                savings_pct = (savings / traditional_costs[i]) * 100
                
                fig.add_annotation(
                    x=m,
                    y=(traditional_costs[i] + multi_model_costs[i]) / 2,
                    text=f"{savings_pct:.0f}% savings",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=AWS_COLORS["green"],
                    ax=20,
                    ay=0
                )
        
    elif endpoint_type == "multi_container":
        # Define metrics for multi-container (A/B testing)
        days = list(range(1, 15))
        model_a_accuracy = [0.82 + i*0.001 for i in range(14)]  # Slow improvement
        model_b_accuracy = [0.79 + i*0.005 for i in range(14)]  # Faster improvement
        
        # Create the figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days,
            y=model_a_accuracy,
            mode='lines+markers',
            name='Model A (70% traffic)',
            line=dict(color=AWS_COLORS["blue"], width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=days,
            y=model_b_accuracy,
            mode='lines+markers',
            name='Model B (30% traffic)',
            line=dict(color=AWS_COLORS["orange"], width=3),
            marker=dict(size=8)
        ))
        
        crossover_day = None
        for i in range(1, len(days)):
            if model_a_accuracy[i-1] >= model_b_accuracy[i-1] and model_a_accuracy[i] < model_b_accuracy[i]:
                crossover_day = days[i]
                break
        
        if crossover_day:
            fig.add_shape(
                type="line",
                x0=crossover_day,
                x1=crossover_day,
                y0=min(min(model_a_accuracy), min(model_b_accuracy)),
                y1=max(max(model_a_accuracy), max(model_b_accuracy)),
                line=dict(color=AWS_COLORS["green"], dash="dash")
            )
            
            fig.add_annotation(
                x=crossover_day,
                y=model_b_accuracy[crossover_day-1],
                text="Crossover Point:\nModel B outperforms Model A",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=AWS_COLORS["green"],
                ax=50,
                ay=0
            )
        
        fig.update_layout(
            title="A/B Testing Performance Comparison",
            xaxis_title="Days",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0.78, 0.9]),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
    else:  # Inference pipeline
        # Define metrics for pipeline vs single container
        processing_steps = ["Raw Input", "Pre-processing", "Inference", "Post-processing", "Final Output"]
        
        # Time taken for each step in ms
        pipeline_times = [0, 50, 120, 35, 0]  # Processing steps only
        single_container_times = [0, 80, 150, 60, 0]  # Same steps but slower in single container
        
        pipeline_cumulative = [sum(pipeline_times[:i+1]) for i in range(len(pipeline_times))]
        single_cumulative = [sum(single_container_times[:i+1]) for i in range(len(single_container_times))]
        
        # Create the figure
        fig = go.Figure()
        
        # Add lines connecting the points
        fig.add_trace(go.Scatter(
            x=processing_steps,
            y=pipeline_cumulative,
            mode='lines+markers',
            name='Pipeline Endpoint',
            line=dict(color=AWS_COLORS["orange"], width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=processing_steps,
            y=single_cumulative,
            mode='lines+markers',
            name='Single Container',
            line=dict(color=AWS_COLORS["blue"], width=3),
            marker=dict(size=10)
        ))
        
        # Add the performance improvement annotation
        improvement = (single_cumulative[-1] - pipeline_cumulative[-1]) / single_cumulative[-1] * 100
        
        fig.add_annotation(
            x=processing_steps[-1],
            y=(pipeline_cumulative[-1] + single_cumulative[-1]) / 2,
            text=f"{improvement:.1f}% faster",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=AWS_COLORS["green"],
            ax=-40,
            ay=0
        )
        
        fig.update_layout(
            title="Processing Time Comparison",
            xaxis_title="Processing Step",
            yaxis_title="Cumulative Time (ms)",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
    
    return fig


def generate_model_metrics(model_name: str) -> Dict[str, float]:
    """
    Generate random model metrics for demonstration
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of model metrics
    """
    # Seed with model name for consistent results
    random.seed(hash(model_name) % 10000)
    
    base_accuracy = 0.85
    base_latency = 100
    
    # Adjust metrics based on model name
    if "xgboost" in model_name.lower():
        base_accuracy += random.uniform(0.02, 0.05)
        base_latency -= random.uniform(20, 40)
    elif "bert" in model_name.lower():
        base_accuracy += random.uniform(0.03, 0.07)
        base_latency += random.uniform(50, 100)
    elif "resnet" in model_name.lower():
        base_accuracy += random.uniform(0.01, 0.04)
        base_latency += random.uniform(10, 30)
    
    # Add some randomness
    accuracy = min(0.99, max(0.8, base_accuracy + random.uniform(-0.02, 0.02)))
    precision = min(0.99, max(0.8, accuracy + random.uniform(-0.03, 0.02)))
    recall = min(0.99, max(0.8, accuracy + random.uniform(-0.04, 0.03)))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    latency = max(10, base_latency + random.uniform(-10, 20))
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "latency": round(latency, 1)
    }


def generate_model_list(list_type: str) -> List[Dict]:
    """
    Generate a list of models for the multi-model endpoint demo
    
    Args:
        list_type: Type of list to generate (xgboost, bert, mixed)
        
    Returns:
        List of model dictionaries
    """
    if list_type == "xgboost":
        models = [
            {"name": "xgboost-churn-v1", "type": "XGBoost", "size_mb": 5},
            {"name": "xgboost-churn-v2", "type": "XGBoost", "size_mb": 7},
            {"name": "xgboost-fraud-v1", "type": "XGBoost", "size_mb": 12},
            {"name": "xgboost-recommender-v1", "type": "XGBoost", "size_mb": 15},
            {"name": "xgboost-credit-scoring-v1", "type": "XGBoost", "size_mb": 8},
            {"name": "xgboost-credit-scoring-v2", "type": "XGBoost", "size_mb": 9},
        ]
    elif list_type == "bert":
        models = [
            {"name": "bert-sentiment-v1", "type": "PyTorch", "size_mb": 438},
            {"name": "bert-qa-v1", "type": "PyTorch", "size_mb": 427},
            {"name": "bert-summarization-v1", "type": "PyTorch", "size_mb": 445},
        ]
    else:  # mixed
        models = [
            {"name": "xgboost-churn-v1", "type": "XGBoost", "size_mb": 5},
            {"name": "resnet-18-v1", "type": "PyTorch", "size_mb": 44},
            {"name": "bert-base-v1", "type": "PyTorch", "size_mb": 438},
            {"name": "lightgbm-fraud-v1", "type": "LightGBM", "size_mb": 7},
            {"name": "tensorflow-recommender-v1", "type": "TensorFlow", "size_mb": 65},
        ]
    
    # Add metrics to each model
    for model in models:
        model["metrics"] = generate_model_metrics(model["name"])
        
    return models


def create_multi_container_config(config_type: str) -> Dict:
    """
    Generate configuration for multi-container endpoint demo
    
    Args:
        config_type: Type of configuration (ab_test, blue_green, canary)
        
    Returns:
        Dictionary of configuration
    """
    if config_type == "ab_test":
        return {
            "variants": [
                {
                    "name": "ModelA",
                    "container": "xgboost:1.3-1",
                    "instance_type": "ml.m5.large",
                    "initial_weight": 50,
                    "description": "Current production model",
                    "metrics": generate_model_metrics("xgboost-current")
                },
                {
                    "name": "ModelB",
                    "container": "xgboost:1.5-1",
                    "instance_type": "ml.m5.large",
                    "initial_weight": 50,
                    "description": "New model with improved features",
                    "metrics": generate_model_metrics("xgboost-improved")
                }
            ],
            "description": "Split traffic 50/50 between current and new model to compare performance"
        }
    elif config_type == "blue_green":
        return {
            "variants": [
                {
                    "name": "Blue",
                    "container": "tensorflow:2.5.1",
                    "instance_type": "ml.c5.xlarge",
                    "initial_weight": 90,
                    "description": "Current production model",
                    "metrics": generate_model_metrics("tensorflow-prod")
                },
                {
                    "name": "Green",
                    "container": "tensorflow:2.7.0",
                    "instance_type": "ml.c5.xlarge",
                    "initial_weight": 10,
                    "description": "New model version for gradual transition",
                    "metrics": generate_model_metrics("tensorflow-new")
                }
            ],
            "description": "Gradually shift traffic from blue (current) to green (new) deployment"
        }
    else:  # canary
        return {
            "variants": [
                {
                    "name": "Production",
                    "container": "pytorch:1.10.0",
                    "instance_type": "ml.g4dn.xlarge",
                    "initial_weight": 95,
                    "description": "Stable production model",
                    "metrics": generate_model_metrics("pytorch-stable")
                },
                {
                    "name": "Canary",
                    "container": "pytorch:1.11.0",
                    "instance_type": "ml.g4dn.xlarge",
                    "initial_weight": 5,
                    "description": "Experimental model with minimal traffic",
                    "metrics": generate_model_metrics("pytorch-experimental")
                }
            ],
            "description": "Test new model with minimal 5% traffic before wider deployment"
        }
    

def create_pipeline_config(pipeline_type: str) -> Dict:
    """
    Generate configuration for inference pipeline demo
    
    Args:
        pipeline_type: Type of pipeline (nlp, vision, tabular)
        
    Returns:
        Dictionary of pipeline configuration
    """
    if pipeline_type == "nlp":
        return {
            "name": "NLP Processing Pipeline",
            "containers": [
                {
                    "name": "text-preprocessing",
                    "image": "text-processors:latest",
                    "description": "Tokenization, text cleaning, and feature extraction",
                    "input": "Raw text",
                    "output": "Preprocessed text features"
                },
                {
                    "name": "bert-model",
                    "image": "huggingface-pytorch:latest",
                    "description": "BERT-based language model for inference",
                    "input": "Preprocessed text features",
                    "output": "Raw BERT embeddings and predictions"
                },
                {
                    "name": "postprocessing",
                    "image": "output-processors:latest",
                    "description": "Convert model outputs to human-readable results",
                    "input": "Raw BERT embeddings and predictions",
                    "output": "Formatted results with confidence scores"
                }
            ],
            "use_case": "Sentiment analysis, text classification, and entity recognition",
            "example_input": "This product exceeded my expectations in every way!",
            "example_output": {"sentiment": "POSITIVE", "confidence": 0.97, "entities": [{"type": "PRODUCT", "text": "product", "confidence": 0.83}]}
        }
    elif pipeline_type == "vision":
        return {
            "name": "Computer Vision Pipeline",
            "containers": [
                {
                    "name": "image-preprocessing",
                    "image": "image-processors:latest",
                    "description": "Image resizing, normalization, and augmentation",
                    "input": "Raw image",
                    "output": "Normalized image tensor"
                },
                {
                    "name": "object-detection",
                    "image": "pytorch-vision:latest",
                    "description": "Object detection model based on YOLO",
                    "input": "Normalized image tensor",
                    "output": "Raw bounding boxes and class probabilities"
                },
                {
                    "name": "visualization",
                    "image": "vision-postprocessors:latest",
                    "description": "Convert detections to annotated images and metadata",
                    "input": "Raw bounding boxes and class probabilities",
                    "output": "Annotated image and structured detection results"
                }
            ],
            "use_case": "Object detection, image classification, and scene understanding",
            "example_input": "[Image data]",
            "example_output": {"objects": 5, "classes": ["person", "car", "dog"], "confidences": [0.98, 0.95, 0.87]}
        }
    else:  # tabular
        return {
            "name": "Tabular Data Pipeline",
            "containers": [
                {
                    "name": "feature-engineering",
                    "image": "sklearn-preprocessor:latest",
                    "description": "Feature scaling, encoding, and transformation",
                    "input": "Raw tabular data",
                    "output": "Engineered feature matrix"
                },
                {
                    "name": "xgboost-predictor",
                    "image": "xgboost-inference:latest",
                    "description": "Gradient boosting model for prediction",
                    "input": "Engineered feature matrix",
                    "output": "Raw prediction scores"
                },
                {
                    "name": "explanation-generator",
                    "image": "shap-explainer:latest",
                    "description": "Generate prediction explanations using SHAP values",
                    "input": "Raw prediction scores",
                    "output": "Predictions with feature importance explanations"
                }
            ],
            "use_case": "Credit scoring, churn prediction, and fraud detection",
            "example_input": {"age": 34, "income": 75000, "credit_score": 720},
            "example_output": {"prediction": "Low Risk", "probability": 0.92, "key_factors": ["credit_score", "income"]}
        }


def generate_sample_code(endpoint_type: str) -> str:
    """
    Generate sample code for different endpoint types
    
    Args:
        endpoint_type: Type of endpoint (multi_model, multi_container, inference_pipeline)
        
    Returns:
        String containing sample Python code
    """
    if endpoint_type == "multi_model":
        return '''
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.multidatamodel import MultiDataModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define container
container = "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.3-1"

# Create the multi-model
multi_model = MultiDataModel(
    name="my-multi-model-endpoint",
    model_data_prefix="s3://my-bucket/my-models/",
    image_uri=container,
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy the multi-model endpoint
predictor = multi_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)

# Add models to the endpoint
multi_model.add_model(model_data_source="s3://my-bucket/my-models/xgboost-churn.tar.gz")
multi_model.add_model(model_data_source="s3://my-bucket/my-models/xgboost-fraud.tar.gz")
multi_model.add_model(model_data_source="s3://my-bucket/my-models/xgboost-propensity.tar.gz")

# Create a runtime client to invoke the endpoint
runtime_client = boto3.client('sagemaker-runtime')

# Invoke a specific model on the endpoint
response = runtime_client.invoke_endpoint(
    EndpointName="my-multi-model-endpoint",
    ContentType="text/csv",
    TargetModel="xgboost-churn.tar.gz",
    Body="0.5,0.2,0.1,0.8,0.4"
)

# Parse and print the result
result = response['Body'].read().decode()
print(f"Prediction: {result}")

# Invoke a different model on the same endpoint
response = runtime_client.invoke_endpoint(
    EndpointName="my-multi-model-endpoint",
    ContentType="text/csv",
    TargetModel="xgboost-fraud.tar.gz",
    Body="0.1,0.9,0.3,0.2,0.5"
)

# Parse and print the result
result = response['Body'].read().decode()
print(f"Prediction: {result}")

# List models currently loaded in the container's memory
list_response = multi_model.list_models()
print(list_response)

# When done, delete the endpoint
predictor.delete_endpoint()
'''.strip()

    elif endpoint_type == "multi_container":
        return '''
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define container info for Model A
container_a = {
    "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.3-1",
    "ModelDataUrl": "s3://my-bucket/model-a/model.tar.gz"
}

# Define container info for Model B
container_b = {
    "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1",
    "ModelDataUrl": "s3://my-bucket/model-b/model.tar.gz"
}

# Create Model A
model_a = Model(
    name="model-a",
    image_uri=container_a["Image"],
    model_data=container_a["ModelDataUrl"],
    role=role,
    sagemaker_session=sagemaker_session
)

# Create Model B
model_b = Model(
    name="model-b",
    image_uri=container_b["Image"],
    model_data=container_b["ModelDataUrl"],
    role=role,
    sagemaker_session=sagemaker_session
)

# Define production variants for A/B testing
production_variants = [
    {
        "VariantName": "ModelA",
        "ModelName": "model-a",
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.large",
        "InitialVariantWeight": 0.7  # 70% of traffic
    },
    {
        "VariantName": "ModelB",
        "ModelName": "model-b",
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.large",
        "InitialVariantWeight": 0.3  # 30% of traffic
    }
]

# Create the endpoint configuration
endpoint_config_response = sagemaker_session.sagemaker_client.create_endpoint_config(
    EndpointConfigName="ab-test-config",
    ProductionVariants=production_variants
)

# Create the endpoint
endpoint_response = sagemaker_session.sagemaker_client.create_endpoint(
    EndpointName="ab-test-endpoint",
    EndpointConfigName="ab-test-config"
)

# Wait for endpoint to be ready
print("Waiting for endpoint to be in service...")
sagemaker_session.wait_for_endpoint("ab-test-endpoint")

# Create a predictor for the endpoint
predictor = Predictor(
    endpoint_name="ab-test-endpoint",
    sagemaker_session=sagemaker_session
)

# Invoke the endpoint - this will automatically route to either Model A or Model B
# based on the configured traffic split
result = predictor.predict("0.5,0.2,0.1,0.8,0.4")
print(f"Prediction: {result}")

# Update variant weights after comparing performance
update_response = sagemaker_session.sagemaker_client.update_endpoint_weights_and_capacities(
    EndpointName="ab-test-endpoint",
    DesiredWeightsAndCapacities=[
        {
            "VariantName": "ModelA",
            "DesiredWeight": 0.2  # Reduce to 20%
        },
        {
            "VariantName": "ModelB",
            "DesiredWeight": 0.8  # Increase to 80%
        }
    ]
)

# When done testing, delete the endpoint
sagemaker_session.sagemaker_client.delete_endpoint(EndpointName="ab-test-endpoint")
sagemaker_session.sagemaker_client.delete_endpoint_config(EndpointConfigName="ab-test-config")
'''.strip()

    else:  # inference_pipeline
        return '''
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.predictor import Predictor

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define container for preprocessing
preprocessor_container = {
    "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/preprocessor:latest",
    "Environment": {
        "NORMALIZE_INPUT": "True",
        "FEATURE_COLUMNS": "age,income,credit_score,debt_ratio"
    }
}

# Define container for the model
model_container = {
    "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1",
    "ModelDataUrl": "s3://my-bucket/model/xgboost-model.tar.gz"
}

# Define container for postprocessing
postprocessor_container = {
    "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/postprocessor:latest",
    "Environment": {
        "THRESHOLD": "0.5",
        "OUTPUT_FORMAT": "json"
    }
}

# Create the preprocessing model
preprocessor_model = Model(
    image_uri=preprocessor_container["Image"],
    role=role,
    env=preprocessor_container["Environment"],
    sagemaker_session=sagemaker_session
)

# Create the main model
primary_model = Model(
    image_uri=model_container["Image"],
    model_data=model_container["ModelDataUrl"],
    role=role,
    sagemaker_session=sagemaker_session
)

# Create the postprocessing model
postprocessor_model = Model(
    image_uri=postprocessor_container["Image"],
    role=role,
    env=postprocessor_container["Environment"],
    sagemaker_session=sagemaker_session
)

# Create the pipeline model
pipeline_model = PipelineModel(
    name="inference-pipeline",
    role=role,
    models=[preprocessor_model, primary_model, postprocessor_model],
    sagemaker_session=sagemaker_session
)

# Deploy the pipeline model to an endpoint
predictor = pipeline_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="pipeline-endpoint"
)

# Invoke the pipeline endpoint with raw data
# The data will flow through preprocessing → model → postprocessing
raw_data = {
    "age": 42,
    "income": 60000,
    "credit_score": 720,
    "debt_ratio": 0.3
}

result = predictor.predict(raw_data)
print(f"Pipeline result: {result}")

# When done, clean up
predictor.delete_endpoint()
'''.strip()


def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'selected_model_list' not in st.session_state:
        st.session_state.selected_model_list = "mixed"
    
    if 'model_list' not in st.session_state:
        st.session_state.model_list = generate_model_list(st.session_state.selected_model_list)
    
    if 'loaded_models' not in st.session_state:
        st.session_state.loaded_models = st.session_state.model_list[:2]  # First two models loaded by default
    
    if 'multi_container_config' not in st.session_state:
        st.session_state.multi_container_config = create_multi_container_config("ab_test")
    
    if 'pipeline_config' not in st.session_state:
        st.session_state.pipeline_config = create_pipeline_config("nlp")
        
    if 'selected_instance_count' not in st.session_state:
        st.session_state.selected_instance_count = 2


def reset_session():
    """
    Reset the session state
    """
    # Keep only user_id and reset all other state
    user_id = st.session_state.user_id
    st.session_state.clear()
    st.session_state.user_id = user_id


# Main application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="SageMaker Inference Hosting",
        page_icon="🚀",
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
        "red": "#D13212",
        "purple": "#9D1F63"
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
        .comparison-table th {
            background-color: #E9ECEF;
        }
        .comparison-table td, .comparison-table th {
            padding: 8px;
            border: 1px solid #ddd;
        }
        .comparison-table tr:hover {
            background-color: #f5f5f5;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            grid-gap: 10px;
        }
        .metric-item {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }
        .model-card {
            border: 1px solid #E9ECEF;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            transition: all 0.2s;
        }
        .model-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .model-card.loaded {
            border-left: 5px solid #59BA47;
        }
        .model-card.not-loaded {
            border-left: 5px solid #E9ECEF;
        }
        .container-card {
            border: 1px solid #E9ECEF;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: white;
        }
        .variant-card {
            border-left: 5px solid #00A1C9;
            background-color: #F8F9FA;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 0 5px 5px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for session management
    with st.sidebar:
        st.markdown("### Session Management")
        st.info(f"User ID: {st.session_state.user_id}")
        
        if st.button("🔄 Reset Session"):
            reset_session()
            st.rerun()
        
        st.divider()
        
        # Information about the application
        with st.expander("📚 About This App", expanded=False):
            st.markdown("""
                This interactive learning application demonstrates Amazon SageMaker's 
                advanced inference hosting options. Explore each tab to understand the 
                different model hosting strategies and their use cases.
            """)
            
            # AWS learning resources
            st.subheader("Additional Resources")
            st.markdown("""
                - [SageMaker Inference Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
                - [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
                - [AWS Training and Certification](https://aws.amazon.com/training/)
            """)
    
    # Main app header
    st.title("Amazon SageMaker Inference Hosting Options")
    st.markdown("""
    Learn about the advanced deployment options available in Amazon SageMaker 
    for serving machine learning models efficiently and cost-effectively.
    """)
    
    # Animation for the header
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_rqfktejl.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=200, key="header_animation")
    
    # Tab-based navigation with emoji
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Overview", 
        "📚 Multi-Model Endpoints", 
        "🔀 Multi-Container Endpoints",
        "🔄 Inference Pipelines"
    ])
    
    # OVERVIEW TAB
    with tab1:
        st.header("SageMaker Inference Hosting Overview")
        
        st.markdown("""
        Amazon SageMaker provides several advanced options for hosting your machine learning models.
        This application explores three powerful deployment strategies that help you optimize costs,
        manage multiple models efficiently, and create complex inference pipelines.
        """)
        
        # Comparison table
        st.subheader("Inference Hosting Options Comparison")
        
        comparison_data = {
            "Feature": [
                "Multiple Models on Single Endpoint", 
                "Multiple Containers", 
                "Sequential Processing",
                "Dynamic Model Loading",
                "A/B Testing",
                "Cost Efficiency for Many Models",
                "Complex Pre/Post Processing"
            ],
            "Multi-Model Endpoints": [
                "✅", 
                "❌", 
                "❌", 
                "✅", 
                "❌", 
                "✅",
                "❌"
            ],
            "Multi-Container Endpoints": [
                "✅", 
                "✅", 
                "❌", 
                "❌", 
                "✅", 
                "❌",
                "❌" 
            ],
            "Inference Pipelines": [
                "❌", 
                "✅", 
                "✅", 
                "❌", 
                "❌", 
                "❌",
                "✅"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Benefits of advanced hosting options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Benefits of Advanced Hosting
            
            - **Reduced Infrastructure Costs**: Host multiple models on the same endpoint
            - **Simplified Management**: Centralized deployment and monitoring
            - **Enhanced Experimentation**: Easy A/B testing and model comparisons
            - **Specialized Processing**: Dedicated containers for different stages
            - **Lower Latency**: Pre-warming and caching for faster predictions
            - **Better Resource Utilization**: Share compute across models
            """)
        
        with col2:
            st.image("https://d1.awsstatic.com/reInvent2021/pdx322-slide-9.76a3636b0718e5c3a466adbd8c1edafaf23b5af5.png",
                    caption="SageMaker Hosting Architectures", use_container_width=True)
        
        # Create a visual overview
        st.subheader("Visual Overview of Hosting Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            multi_model_fig = create_inference_flow_diagram("multi_model", 6, 4)
            st.pyplot(multi_model_fig)
            st.markdown("""
            #### Multi-Model Endpoints
            
            Host many similar models with dynamic loading and unloading to minimize resource usage.
            """)
        
        with col2:
            multi_container_fig = create_inference_flow_diagram("multi_container", 6, 4)
            st.pyplot(multi_container_fig)
            st.markdown("""
            #### Multi-Container Endpoints
            
            Run multiple containers on a single endpoint with traffic splitting for A/B testing.
            """)
            
        with col3:
            pipeline_fig = create_inference_flow_diagram("pipeline", 6, 4)
            st.pyplot(pipeline_fig)
            st.markdown("""
            #### Inference Pipelines
            
            Create sequential processing workflows with specialized containers.
            """)
        
        # Use case comparison
        st.subheader("When to Use Each Hosting Option")
        
        use_cases = {
            "Use Case": [
                "Hosting multiple similar models cost-effectively",
                "Performing A/B testing between models",
                "Gradual model rollout (canary deployment)",
                "Processing complex data transformations",
                "Deploying with multiple framework requirements",
                "Generating model explanations via post-processing",
                "Serving hundreds of models with low overall traffic"
            ],
            "Best Option": [
                "Multi-Model Endpoint",
                "Multi-Container Endpoint",
                "Multi-Container Endpoint",
                "Inference Pipeline",
                "Inference Pipeline",
                "Inference Pipeline",
                "Multi-Model Endpoint"
            ],
            "Why": [
                "Dynamically loads models into memory as needed",
                "Provides traffic splitting capabilities",
                "Allows controlled traffic shifting between variants",
                "Enables specialized containers for each processing stage",
                "Each container can use a different framework",
                "Add explainability without modifying the model",
                "Most cost-effective for many rarely-used models"
            ]
        }
        
        use_cases_df = pd.DataFrame(use_cases)
        st.dataframe(use_cases_df, use_container_width=True)
        
        # Instance count configurator for performance charts
        st.subheader("Cost & Performance Comparison")
        
        instance_count = st.slider(
            "Number of Instances", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.selected_instance_count,
            help="See how costs change with different numbers of instances"
        )
        st.session_state.selected_instance_count = instance_count
        
        # Performance charts for each type
        col1, col2 = st.columns(2)
        
        with col1:
            multi_model_fig = create_performance_chart("multi_model", instance_count)
            st.plotly_chart(multi_model_fig, use_container_width=True)
        
        with col2:
            multi_container_fig = create_performance_chart("multi_container")
            st.plotly_chart(multi_container_fig, use_container_width=True)
        
        pipeline_fig = create_performance_chart("pipeline")
        st.plotly_chart(pipeline_fig, use_container_width=True)
        
        # Common considerations
        st.markdown("""
        ### Key Considerations When Choosing a Hosting Option
        
        1. **Number of Models**: How many models do you need to host?
        2. **Model Similarity**: Are your models based on the same framework?
        3. **Request Patterns**: Do you have predictable or sporadic traffic?
        4. **Memory Requirements**: How large are your model artifacts?
        5. **Processing Needs**: Do you need complex pre/post-processing?
        6. **Testing Strategy**: Are you doing A/B testing or canary deployments?
        7. **Cost Sensitivity**: How important is minimizing infrastructure costs?
        """)
        
    # MULTI-MODEL ENDPOINT TAB
    with tab2:
        st.header("📚 Multi-Model Endpoints")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Multi-Model Endpoints (MMEs) allow you to deploy multiple models behind a single SageMaker endpoint.
            Models are dynamically loaded and unloaded based on usage, helping you efficiently host many models
            while only paying for a single endpoint.
            
            **Key features:**
            - **Cost-efficient hosting** for many similar models
            - **Dynamic loading** of models from S3 to endpoint container memory
            - **Intelligent caching** of frequently used models
            - **Common container environment** across all models
            """)
        
        with col2:
            st.image("images/multi_model.png",
                    caption="Multi-Model Endpoint Architecture", use_container_width=True)
        
        # How MME works
        st.subheader("How Multi-Model Endpoints Work")
        
        mme_fig = create_inference_flow_diagram("multi_model")
        st.pyplot(mme_fig)
        
        # Workflow description
        st.markdown("""
        ### Multi-Model Endpoint Workflow
        
        1. **Model Preparation**: Package and upload multiple model artifacts to S3
        2. **Endpoint Creation**: Deploy a multi-model endpoint pointing to your S3 model repository
        3. **Inference Request**: Client specifies target model name with each request
        4. **Dynamic Loading**: Container checks if model is loaded in memory
            - If not loaded, downloads from S3 and loads into memory
            - If memory is full, may unload less frequently used models
        5. **Inference Processing**: Container uses the specified model to generate prediction
        6. **Response**: Results returned to the client
        
        As traffic patterns change, different models may be loaded or unloaded automatically.
        """)
        
        # Interactive MME demo
        st.subheader("Interactive Demo: Multi-Model Endpoint")
        
        # Model list selector
        model_list_type = st.radio(
            "Select model repository type:", 
            ["Mixed Models", "Similar XGBoost Models", "Large BERT Models"],
            horizontal=True,
            index=0 if st.session_state.selected_model_list == "mixed" else
                  1 if st.session_state.selected_model_list == "xgboost" else 2
        )
        
        # Update model list based on selection
        if model_list_type == "Mixed Models" and st.session_state.selected_model_list != "mixed":
            st.session_state.selected_model_list = "mixed"
            st.session_state.model_list = generate_model_list("mixed")
            st.session_state.loaded_models = st.session_state.model_list[:2]
        elif model_list_type == "Similar XGBoost Models" and st.session_state.selected_model_list != "xgboost":
            st.session_state.selected_model_list = "xgboost"
            st.session_state.model_list = generate_model_list("xgboost")
            st.session_state.loaded_models = st.session_state.model_list[:2]
        elif model_list_type == "Large BERT Models" and st.session_state.selected_model_list != "bert":
            st.session_state.selected_model_list = "bert"
            st.session_state.model_list = generate_model_list("bert")
            st.session_state.loaded_models = st.session_state.model_list[:2]
        
        # Display endpoint configuration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Show endpoint configuration
            st.markdown("### Endpoint Configuration")
            
            endpoint_params = {
                "Instance Type": "ml.m5.xlarge",
                "Memory (GB)": 16,
                "vCPUs": 4,
                "Max Models in Memory": 3 if st.session_state.selected_model_list == "bert" else 6,
                "Instance Count": 1,
                "Auto Scaling": "Enabled",
                "Model Cache Size (MB)": 12000 if st.session_state.selected_model_list == "bert" else 5000
            }
            
            endpoint_df = pd.DataFrame({"Parameter": list(endpoint_params.keys()),
                                       "Value": list(endpoint_params.values())})
            st.table(endpoint_df)
            
        with col2:
            # Show model repository info
            st.markdown("### Model Repository")
            
            # Calculate total size
            total_size = sum(model["size_mb"] for model in st.session_state.model_list)
            
            repo_stats = {
                "S3 Location": "s3://my-model-bucket/models/",
                "Models Available": len(st.session_state.model_list),
                "Total Size": f"{total_size} MB",
                "Average Model Size": f"{total_size / len(st.session_state.model_list):.1f} MB",
                "Largest Model": f"{max(model['size_mb'] for model in st.session_state.model_list)} MB",
                "Smallest Model": f"{min(model['size_mb'] for model in st.session_state.model_list)} MB"
            }
            
            repo_df = pd.DataFrame({"Metric": list(repo_stats.keys()),
                                  "Value": list(repo_stats.values())})
            st.table(repo_df)
            
        # Model selection
        st.subheader("Available Models")
        
        # Display models as cards in 2 columns
        col1, col2 = st.columns(2)
        
        for i, model in enumerate(st.session_state.model_list):
            # Alternate between columns
            with col1 if i % 2 == 0 else col2:
                is_loaded = any(lm["name"] == model["name"] for lm in st.session_state.loaded_models)
                model_class = "model-card loaded" if is_loaded else "model-card not-loaded"
                
                st.markdown(f"""
                <div class="{model_class}">
                    <h4>{model["name"]}</h4>
                    <p><strong>Type:</strong> {model["type"]}</p>
                    <p><strong>Size:</strong> {model["size_mb"]} MB</p>
                    <p><strong>Status:</strong> {"📥 Loaded in Memory" if is_loaded else "📦 Available in S3"}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Inference simulation
        st.subheader("Inference Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Model selection for inference
            selected_model_name = st.selectbox(
                "Select model to invoke:",
                [model["name"] for model in st.session_state.model_list]
            )
            
            selected_model = next((m for m in st.session_state.model_list if m["name"] == selected_model_name), None)
            
            # Add a button to invoke the model
            if st.button("Invoke Model"):
                with st.spinner("Sending request to endpoint..."):
                    time.sleep(1)  # Simulate network latency
                    
                    # Check if model is already loaded
                    is_loaded = any(lm["name"] == selected_model_name for lm in st.session_state.loaded_models)
                    
                    # If not loaded, load it (possibly unload another model)
                    if not is_loaded:
                        max_models = endpoint_params["Max Models in Memory"]
                        
                        if len(st.session_state.loaded_models) >= max_models:
                            # Remove least recently used model (first in list)
                            st.session_state.loaded_models.pop(0)
                        
                        # Add the newly loaded model
                        st.session_state.loaded_models.append(selected_model)
                        
                        # Add loading time delay
                        time.sleep(1.5)  # Longer delay for loading
                        
                        st.success(f"Model {selected_model_name} loaded into memory.")
                    
                    # Show the inference result
                    st.session_state.latest_result = selected_model["metrics"]
                    
                    # Update list order (move used model to end as most recently used)
                    if is_loaded:
                        # Find and remove the model
                        for i, lm in enumerate(st.session_state.loaded_models):
                            if lm["name"] == selected_model_name:
                                st.session_state.loaded_models.pop(i)
                                break
                        
                        # Add it back at the end
                        st.session_state.loaded_models.append(selected_model)
        
        with col2:
            # Display memory usage visualization
            st.markdown("### Endpoint Memory Usage")
            
            # Calculate memory usage
            total_memory = endpoint_params["Memory (GB)"] * 1024  # Convert to MB
            used_memory = sum(model["size_mb"] for model in st.session_state.loaded_models)
            overhead_memory = 500  # Reserved for container and runtime
            available_memory = total_memory - used_memory - overhead_memory
            
            # Create a bar chart showing memory usage
            memory_data = pd.DataFrame([
                {"Category": "Used by Models", "Memory (MB)": used_memory, "Color": AWS_COLORS["orange"]},
                {"Category": "System Overhead", "Memory (MB)": overhead_memory, "Color": AWS_COLORS["gray"]},
                {"Category": "Available", "Memory (MB)": available_memory, "Color": AWS_COLORS["green"]}
            ])
            
            fig = px.bar(memory_data, x="Memory (MB)", y="Category", orientation='h',
                         color="Category", color_discrete_map={"Used by Models": AWS_COLORS["orange"], 
                                                             "System Overhead": AWS_COLORS["gray"],
                                                             "Available": AWS_COLORS["green"]})
            
            fig.update_layout(
                title=f"Memory Usage ({used_memory} MB used of {total_memory} MB total)",
                showlegend=False,
                height=150,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display loaded models
            st.markdown("### Currently Loaded Models:")
            
            if st.session_state.loaded_models:
                loaded_models_df = pd.DataFrame([
                    {"Model": lm["name"], "Type": lm["type"], "Size (MB)": lm["size_mb"]}
                    for lm in st.session_state.loaded_models
                ])
                st.dataframe(loaded_models_df, use_container_width=True)
            else:
                st.info("No models currently loaded")
                
            # Display inference results if available
            if "latest_result" in st.session_state:
                st.markdown("### Latest Inference Result:")
                
                metrics = st.session_state.latest_result
                cols = st.columns(len(metrics))
                
                for i, (metric, value) in enumerate(metrics.items()):
                    with cols[i]:
                        st.metric(metric.replace("_", " ").title(), value)
        
        # Cost analysis
        st.subheader("Cost Analysis")
        
        # Create a cost comparison chart
        num_models = [1, 5, 10, 20, 50, 100]
        
        instance_params = {
        "Instance Count": 1,
        "Instance Type": "ml.p3.2xlarge",
        "GPU Count": 1,
        "Memory": "16 GiB",
        "vCPUs": 8
    }

        
        mme_costs = [instance_params["Instance Count"] * 0.30 * 24 * 30 for _ in num_models]  # $0.30 per hour per instance
        traditional_costs = [n * 0.30 * 24 * 30 for n in num_models]  # Individual endpoints
        
        cost_data = pd.DataFrame({
            "Number of Models": num_models,
            "Multi-Model Endpoint": mme_costs,
            "Individual Endpoints": traditional_costs
        })
        
        cost_data_melted = pd.melt(
            cost_data, 
            id_vars=["Number of Models"], 
            var_name="Endpoint Type",
            value_name="Monthly Cost ($)"
        )
        
        # Plot with Altair
        cost_chart = alt.Chart(cost_data_melted).mark_line(point=True).encode(
            x=alt.X('Number of Models:Q'),
            y=alt.Y('Monthly Cost ($):Q'),
            color=alt.Color('Endpoint Type:N', 
                          scale=alt.Scale(domain=['Multi-Model Endpoint', 'Individual Endpoints'],
                                         range=[AWS_COLORS["orange"], AWS_COLORS["blue"]]))
        ).properties(
            width=700,
            height=400,
            title="Monthly Cost Comparison: Multi-Model vs. Individual Endpoints"
        )
        
        st.altair_chart(cost_chart, use_container_width=True)
        
        # ROI analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Let user select number of models
            selected_models = st.slider("Number of Models", min_value=1, max_value=100, value=10)
            
            # Calculate costs
            traditional_cost = selected_models * 0.30 * 24 * 30
            mme_cost = 0.30 * 24 * 30  # One instance
            savings = traditional_cost - mme_cost
            savings_percent = (savings / traditional_cost) * 100 if traditional_cost > 0 else 0
        
        with col2:
            # Display the cost comparison
            st.metric("Traditional Endpoints Cost", f"${traditional_cost:,.2f}/month")
            st.metric("Multi-Model Endpoint Cost", f"${mme_cost:,.2f}/month")
            st.metric("Monthly Savings", f"${savings:,.2f} ({savings_percent:.1f}%)")
        
        # Best practices
        st.subheader("Best Practices for Multi-Model Endpoints")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### When to Use
            
            - When hosting **many similar models** (same framework)
            - For **cost optimization** of multiple endpoints
            - When models have **similar resource requirements**
            - For models with **intermittent traffic** patterns
            - When hosting **many small specialized models** (e.g., per-customer models)
            """)
        
        with col2:
            st.markdown("""
            ### Implementation Tips
            
            - Group models by **framework and resource needs**
            - Consider **model loading times** for latency-sensitive applications
            - Use **model parallelism** for very large models
            - Monitor **cache hit rates** to optimize instance size
            - Set up **auto-scaling** based on instance metrics
            - Test with **peak load scenarios** to determine instance requirements
            """)
        
        # Example code
        st.subheader("Sample Code: Deploying a Multi-Model Endpoint")
        
        st.code(generate_sample_code("multi_model"), language="python")
    
    # MULTI-CONTAINER ENDPOINT TAB
    with tab3:
        st.header("🔀 Multi-Container Endpoints")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Multi-Container Endpoints allow you to deploy multiple containerized models behind a single endpoint,
            with the ability to split traffic between them. This is ideal for A/B testing, canary deployments,
            and blue/green deployments of machine learning models.
            
            **Key features:**
            - **Traffic splitting** between multiple model variants
            - **A/B testing** for model comparison
            - **Canary deployments** for safe model updates
            - **Production variant isolation** for resilience
            - **Independent scaling** for different model variants
            """)
        
        with col2:
            st.image("images/multi_container.png",
                    caption="Multi-Container Endpoint Architecture", use_container_width=True)
        
        # How it works
        st.subheader("How Multi-Container Endpoints Work")
        
        mce_fig = create_inference_flow_diagram("multi_container")
        st.pyplot(mce_fig)
        
        # Workflow description
        st.markdown("""
        ### Multi-Container Endpoint Workflow
        
        1. **Model Preparation**: Create separate containerized models with your chosen frameworks
        2. **Configure Production Variants**: Define each variant with its own container, instance type, and traffic weight
        3. **Endpoint Creation**: Deploy all variants behind a single endpoint URL
        4. **Request Routing**: SageMaker automatically routes traffic based on specified weights
        5. **Monitoring**: Track performance metrics for each variant independently
        6. **Weight Adjustment**: Dynamically adjust traffic distribution based on performance
        
        This approach enables robust experimentation and safe deployment strategies for machine learning models.
        """)
        
        # Interactive Multi-Container demo
        st.subheader("Interactive Demo: Multi-Container Endpoint")
        
        # Configuration selector
        config_types = ["A/B Testing", "Blue/Green Deployment", "Canary Release"]
        selected_config = st.radio(
            "Select deployment strategy:", 
            config_types,
            horizontal=True,
            index=0
        )
        
        # Update configuration based on selection
        config_type_map = {"A/B Testing": "ab_test", "Blue/Green Deployment": "blue_green", "Canary Release": "canary"}
        selected_config_key = config_type_map[selected_config]
        
        if "multi_container_config" not in st.session_state or st.session_state.multi_container_config["variants"][0]["name"] != {
            "ab_test": "ModelA", 
            "blue_green": "Blue", 
            "canary": "Production"
        }[selected_config_key]:
            st.session_state.multi_container_config = create_multi_container_config(selected_config_key)
        
        # Display config info
        st.markdown(f"### {selected_config} Configuration")
        st.markdown(st.session_state.multi_container_config["description"])
        
        # Display variants
        col1, col2 = st.columns(2)
        
        # Extract variants for easier access
        variants = st.session_state.multi_container_config["variants"]
        
        # Display variant information
        for i, variant in enumerate(variants):
            with col1 if i == 0 else col2:
                st.markdown(f"""
                <div class="container-card">
                    <h4>{variant["name"]} Variant</h4>
                    <div class="variant-card">
                        <p><strong>Container:</strong> {variant["container"]}</p>
                        <p><strong>Instance Type:</strong> {variant["instance_type"]}</p>
                        <p><strong>Traffic Weight:</strong> {variant["initial_weight"]}%</p>
                        <p><strong>Description:</strong> {variant["description"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Traffic allocation slider
        st.markdown("### Traffic Distribution")
        
        if selected_config == "A/B Testing":
            # For A/B testing, allow full adjustment between variants
            model_a_traffic = st.slider(
                "Model A Traffic Percentage", 
                min_value=0, 
                max_value=100,
                value=variants[0]["initial_weight"],
                step=5
            )
            
            model_b_traffic = 100 - model_a_traffic
            
            # Update the variant weights
            variants[0]["initial_weight"] = model_a_traffic
            variants[1]["initial_weight"] = model_b_traffic
            
            # Create a donut chart showing traffic distribution
            traffic_data = pd.DataFrame({
                "Variant": [variants[0]["name"], variants[1]["name"]],
                "Traffic (%)": [model_a_traffic, model_b_traffic]
            })
            
            fig = go.Figure(data=[go.Pie(
                labels=traffic_data["Variant"],
                values=traffic_data["Traffic (%)"],
                hole=.4,
                marker_colors=[AWS_COLORS["blue"], AWS_COLORS["orange"]]
            )])
            
            fig.update_layout(
                title=f"Traffic Distribution: {model_a_traffic}% vs {model_b_traffic}%",
                annotations=[dict(text="Traffic Split", x=0.5, y=0.5, font_size=15, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif selected_config == "Blue/Green Deployment":
            # For Blue/Green, allow shifting from blue to green
            blue_traffic = st.slider(
                "Blue (Current) Traffic Percentage", 
                min_value=0, 
                max_value=100,
                value=variants[0]["initial_weight"],
                step=10
            )
            
            green_traffic = 100 - blue_traffic
            
            # Update the variant weights
            variants[0]["initial_weight"] = blue_traffic
            variants[1]["initial_weight"] = green_traffic
            
            # Create a progress bar visualization
            st.progress(green_traffic / 100)
            st.markdown(f"**Deployment Progress**: {green_traffic}% shifted to Green (new) variant")
            
            # Create a state indicator
            deployment_state = "Planning" if green_traffic == 0 else "Rollback" if green_traffic < 20 else "Testing" if green_traffic < 60 else "Finalizing" if green_traffic < 100 else "Completed"
            
            st.info(f"Deployment State: **{deployment_state}**")
            
        else:  # Canary
            # For Canary, allow just a small percentage to the canary variant
            production_traffic = st.slider(
                "Production Traffic Percentage", 
                min_value=80, 
                max_value=100,
                value=variants[0]["initial_weight"],
                step=1
            )
            
            canary_traffic = 100 - production_traffic
            
            # Update the variant weights
            variants[0]["initial_weight"] = production_traffic
            variants[1]["initial_weight"] = canary_traffic
            
            # Create a small multiples visualization
            canary_data = pd.DataFrame({
                "Variant": ["Production", "Canary"],
                "Traffic (%)": [production_traffic, canary_traffic]
            })
            
            fig = px.bar(canary_data, x="Variant", y="Traffic (%)", 
                        color="Variant", color_discrete_map={"Production": AWS_COLORS["blue"], "Canary": AWS_COLORS["orange"]})
            
            fig.update_layout(
                title=f"Canary Testing: {canary_traffic}% traffic to experimental variant",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance monitoring
        st.subheader("Performance Comparison")
        
        # Create tabs for different metrics
        metric_tab1, metric_tab2, metric_tab3 = st.tabs(["Accuracy", "Latency", "Custom Metrics"])
        
        with metric_tab1:
            # Get metrics for variants
            metrics = [variant["metrics"] for variant in variants]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{variants[0]['name']} Accuracy", f"{metrics[0]['accuracy']:.4f}")
                st.metric(f"{variants[0]['name']} Precision", f"{metrics[0]['precision']:.4f}")
                st.metric(f"{variants[0]['name']} Recall", f"{metrics[0]['recall']:.4f}")
            
            with col2:
                accuracy_diff = metrics[1]['accuracy'] - metrics[0]['accuracy']
                precision_diff = metrics[1]['precision'] - metrics[0]['precision']
                recall_diff = metrics[1]['recall'] - metrics[0]['recall']
                
                st.metric(f"{variants[1]['name']} Accuracy", f"{metrics[1]['accuracy']:.4f}", 
                         f"{accuracy_diff:.4f}")
                st.metric(f"{variants[1]['name']} Precision", f"{metrics[1]['precision']:.4f}", 
                         f"{precision_diff:.4f}")
                st.metric(f"{variants[1]['name']} Recall", f"{metrics[1]['recall']:.4f}", 
                         f"{recall_diff:.4f}")
            
            # Create comparison chart
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                f"{variants[0]['name']}": [metrics[0]['accuracy'], metrics[0]['precision'], 
                                          metrics[0]['recall'], metrics[0]['f1_score']],
                f"{variants[1]['name']}": [metrics[1]['accuracy'], metrics[1]['precision'], 
                                          metrics[1]['recall'], metrics[1]['f1_score']]
            })
            
            metrics_melted = pd.melt(metrics_df, id_vars=['Metric'], var_name='Variant', value_name='Score')
            
            chart = alt.Chart(metrics_melted).mark_bar().encode(
                x=alt.X('Metric:N'),
                y=alt.Y('Score:Q', scale=alt.Scale(domain=[0.08, 1])),
                color=alt.Color('Variant:N', 
                              scale=alt.Scale(domain=[variants[0]['name'], variants[1]['name']], 
                                            range=[AWS_COLORS["blue"], AWS_COLORS["orange"]])),
                column=alt.Column('Variant:N')
            ).properties(
                width=200,
                title="Accuracy Metrics Comparison"
            )
            
            st.altair_chart(chart, use_container_width=True)
        
        with metric_tab2:
            # Latency comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{variants[0]['name']} Latency", f"{metrics[0]['latency']:.1f} ms")
            
            with col2:
                latency_diff = metrics[0]['latency'] - metrics[1]['latency']  # Reversed because lower is better
                st.metric(f"{variants[1]['name']} Latency", f"{metrics[1]['latency']:.1f} ms", 
                         f"{latency_diff:.1f} ms")
            
            # Generate some random latency distributions
            np.random.seed(42)
            variant1_latencies = np.random.normal(metrics[0]['latency'], metrics[0]['latency'] * 0.1, 1000)
            variant2_latencies = np.random.normal(metrics[1]['latency'], metrics[1]['latency'] * 0.1, 1000)
            
            # Create histogram for comparison
            latency_df = pd.DataFrame({
                variants[0]['name']: variant1_latencies,
                variants[1]['name']: variant2_latencies
            })
            
            latency_melted = pd.melt(latency_df, var_name='Variant', value_name='Latency (ms)')
            
            latency_chart = alt.Chart(latency_melted).mark_area(
                opacity=0.5,
                interpolate='step'
            ).encode(
                alt.X('Latency (ms):Q', bin=alt.Bin(maxbins=30)),
                alt.Y('count():Q'),
                alt.Color('Variant:N',
                        scale=alt.Scale(domain=[variants[0]['name'], variants[1]['name']], 
                                      range=[AWS_COLORS["blue"], AWS_COLORS["orange"]]))
            ).properties(
                width=700,
                height=300,
                title="Latency Distribution Comparison"
            )
            
            st.altair_chart(latency_chart, use_container_width=True)
            
            # Show p95, p99 latencies
            col1, col2 = st.columns(2)
            
            p95_v1 = np.percentile(variant1_latencies, 95)
            p95_v2 = np.percentile(variant2_latencies, 95)
            p99_v1 = np.percentile(variant1_latencies, 99)
            p99_v2 = np.percentile(variant2_latencies, 99)
            
            with col1:
                st.metric(f"{variants[0]['name']} p95 Latency", f"{p95_v1:.1f} ms")
                st.metric(f"{variants[0]['name']} p99 Latency", f"{p99_v1:.1f} ms")
            
            with col2:
                p95_diff = p95_v1 - p95_v2
                p99_diff = p99_v1 - p99_v2
                
                st.metric(f"{variants[1]['name']} p95 Latency", f"{p95_v2:.1f} ms", 
                         f"{p95_diff:.1f} ms")
                st.metric(f"{variants[1]['name']} p99 Latency", f"{p99_v2:.1f} ms", 
                         f"{p99_diff:.1f} ms")
        
        with metric_tab3:
            # Business metrics that would be relevant (simulated)
            st.markdown("### Business Impact Metrics")
            st.info("These metrics would be collected by integrating with your business applications.")
            
            # Generate some simulated business metrics
            business_metrics = {
                variants[0]['name']: {
                    "Conversion Rate": 0.034,
                    "Avg Session Time (sec)": 145,
                    "Click-Through Rate": 0.068,
                    "Bounce Rate": 0.22
                },
                variants[1]['name']: {
                    "Conversion Rate": 0.039,
                    "Avg Session Time (sec)": 158,
                    "Click-Through Rate": 0.072,
                    "Bounce Rate": 0.19
                }
            }
            
            # Show business metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Conversion Rate", f"{business_metrics[variants[0]['name']]['Conversion Rate']:.1%}")
                st.metric("Click-Through Rate", f"{business_metrics[variants[0]['name']]['Click-Through Rate']:.1%}")
            
            with col2:
                conversion_diff = business_metrics[variants[1]['name']]['Conversion Rate'] - business_metrics[variants[0]['name']]['Conversion Rate']
                ctr_diff = business_metrics[variants[1]['name']]['Click-Through Rate'] - business_metrics[variants[0]['name']]['Click-Through Rate']
                
                st.metric("Conversion Rate (New)", f"{business_metrics[variants[1]['name']]['Conversion Rate']:.1%}", 
                         f"{conversion_diff:.1%}")
                st.metric("Click-Through Rate (New)", f"{business_metrics[variants[1]['name']]['Click-Through Rate']:.1%}", 
                         f"{ctr_diff:.1%}")
            
            # Create statistical significance indicator
            p_value = 0.02 if conversion_diff > 0.003 else 0.11
            
            if p_value < 0.05:
                st.success(f"✅ The improvement in conversion rate is statistically significant (p = {p_value:.3f})")
            else:
                st.warning(f"⚠️ The improvement in conversion rate is not yet statistically significant (p = {p_value:.3f})")
        
        # Deployment decision section
        st.subheader("Deployment Decision")
        
        metrics = [variant["metrics"] for variant in variants]
        
        # Calculate decision score based on metrics
        metric_improvements = {
            "accuracy": metrics[1]["accuracy"] - metrics[0]["accuracy"],
            "latency": (metrics[0]["latency"] - metrics[1]["latency"]) / metrics[0]["latency"],  # Percentage improvement
            "f1": metrics[1]["f1_score"] - metrics[0]["f1_score"]
        }
        
        # Decision logic
        if sum(metric_improvements.values()) > 0.05:  # Significant improvement
            decision = "Deploy New Model"
            confidence = "High"
            color = AWS_COLORS["green"]
        elif sum(metric_improvements.values()) > 0.01:  # Modest improvement
            decision = "Deploy New Model"
            confidence = "Medium"
            color = AWS_COLORS["teal"]
        elif sum(metric_improvements.values()) > -0.01:  # Neutral
            decision = "Continue Testing"
            confidence = "Low"
            color = AWS_COLORS["orange"]
        else:  # Negative impact
            decision = "Keep Current Model"
            confidence = "High"
            color = AWS_COLORS["red"]
        
        # Display decision
        st.markdown(f"""
        <div style="padding:15px; border-radius:5px; background-color:{color}; color:white;">
        <h3 style="margin: 0;">Recommended Decision: {decision}</h3>
        <p style="margin: 5px 0 0 0;">Confidence: {confidence}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show metrics that influenced the decision
        st.markdown("### Key Metrics Influencing Decision")
        
        for metric, value in metric_improvements.items():
            if metric == "latency":
                # For latency, format as percentage improvement
                st.metric(f"{metric.title()} Improvement", f"{value*100:.1f}%")
            else:
                st.metric(f"{metric.title()} Improvement", f"{value:.4f}")
        
        # Best practices
        st.subheader("Best Practices for Multi-Container Endpoints")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### When to Use
            
            - For **A/B testing** between model versions
            - During **gradual model rollouts** (blue/green)
            - For **testing new frameworks** with minimal risk
            - When **comparing performance** of different implementations
            - For **canary testing** with minimal traffic exposure
            """)
        
        with col2:
            st.markdown("""
            ### Implementation Tips
            
            - Start with **small traffic allocations** to new variants
            - Set up detailed **CloudWatch monitoring** for each variant
            - Define clear **success metrics** before testing
            - Use **automated rollbacks** if issues are detected
            - Consider **instance right-sizing** for each variant
            - Implement **statistical significance testing**
            """)
        
        # Sample code
        st.subheader("Sample Code: Deploying a Multi-Container Endpoint")
        
        st.code(generate_sample_code("multi_container"), language="python")
    
    # INFERENCE PIPELINE TAB
    with tab4:
        st.header("🔄 Inference Pipelines")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Inference Pipelines allow you to chain together multiple containers that process data sequentially
            as a single SageMaker endpoint. This is ideal for scenarios requiring pre-processing or post-processing 
            steps around your core ML model.
            
            **Key features:**
            - **Sequential processing** of data through multiple containers
            - **Specialized containers** for different processing stages
            - **Mixed framework support** in a single inference workflow
            - **End-to-end data transformation** pipeline
            - **Simplified integration** with preprocessing/postprocessing
            """)
        
        with col2:
            st.image("images/inference_pipeline.png",
                    caption="Inference Pipeline Flow", use_container_width=True)
        
        # How it works
        st.subheader("How Inference Pipelines Work")
        
        pipeline_fig = create_inference_flow_diagram("pipeline")
        st.pyplot(pipeline_fig)
        
        # Workflow description
        st.markdown("""
        ### Inference Pipeline Workflow
        
        1. **Configure Containers**: Set up specialized containers for each stage of processing
        2. **Define Pipeline Sequence**: Create a pipeline model with containers in processing order
        3. **Deploy Endpoint**: Deploy the pipeline as a single SageMaker endpoint
        4. **Request Processing**:
            - Raw data enters the first container for preprocessing
            - Preprocessed data passes to the model container for inference
            - Model output passes to post-processing container for final formatting
        5. **Response**: Final processed results returned to the client
        
        Each container is responsible for transforming data into the format expected by the next container.
        """)
        
        # Interactive Pipeline demo
        st.subheader("Interactive Demo: Inference Pipeline")
        
        # Pipeline type selector
        pipeline_types = ["NLP Pipeline", "Computer Vision Pipeline", "Tabular Data Pipeline"]
        selected_pipeline = st.radio(
            "Select pipeline type:", 
            pipeline_types,
            horizontal=True,
            index=0
        )
        
        # Update pipeline based on selection
        pipeline_type_map = {
            "NLP Pipeline": "nlp", 
            "Computer Vision Pipeline": "vision", 
            "Tabular Data Pipeline": "tabular"
        }
        
        selected_pipeline_key = pipeline_type_map[selected_pipeline]
        
        if ("pipeline_config" not in st.session_state or 
            st.session_state.pipeline_config["name"] != 
            {"nlp": "NLP Processing Pipeline", 
             "vision": "Computer Vision Pipeline", 
             "tabular": "Tabular Data Pipeline"}[selected_pipeline_key]):
            st.session_state.pipeline_config = create_pipeline_config(selected_pipeline_key)
        
        # Display pipeline information
        pipeline_config = st.session_state.pipeline_config
        
        st.markdown(f"### {pipeline_config['name']}")
        st.markdown(f"**Use Case:** {pipeline_config['use_case']}")
        
        # Create a visual diagram of the pipeline
        st.markdown("### Pipeline Flow")
        
        # Create pipeline diagram with Plotly
        containers = pipeline_config["containers"]
        
        pipeline_graph = go.Figure()
        
        # Add node for each container
        x_positions = [1, 3, 5]
        
        for i, container in enumerate(containers):
            pipeline_graph.add_trace(go.Scatter(
                x=[x_positions[i]],
                y=[1],
                mode="markers+text",
                marker=dict(
                    size=60,
                    color=[AWS_COLORS["teal"], AWS_COLORS["blue"], AWS_COLORS["green"]][i % 3],
                    symbol="square",
                    line=dict(width=2, color="white")
                ),
                text=[container["name"]],
                textposition="middle center",
                textfont=dict(color="white", size=11),
                hovertext=[container["description"]],
                hoverinfo="text",
                name=container["name"]
            ))
        
        # Add data nodes
        data_x = [0, 2, 4, 6]
        # data_names = ["Input", container["input"] for container in containers[0:2]], containers[1]["output"], containers[2]["output"]
        
        pipeline_graph.add_trace(go.Scatter(
            x=data_x,
            y=[0, 0, 0, 0],
            mode="markers+text",
            marker=dict(
                size=50,
                color=[AWS_COLORS["gray"]],
                symbol="circle",
                line=dict(width=2, color="white")
            ),
            text=["Raw Input", "Preprocessed", "Model Output", "Final Output"],
            textposition="bottom center",
            textfont=dict(size=10),
            hoverinfo="skip",
            showlegend=False
        ))
        
        # Add connecting arrows
        for i in range(len(data_x)-1):
            pipeline_graph.add_shape(
                type="line",
                x0=data_x[i] + 0.3,
                y0=0,
                x1=data_x[i+1] - 0.3,
                y1=0,
                line=dict(color=AWS_COLORS["dark_gray"], width=2, dash="solid"),
                xref="x",
                yref="y"
            )
            
            # Also connect to containers
            pipeline_graph.add_shape(
                type="line",
                x0=data_x[i],
                y0=0.1,
                x1=x_positions[i],
                y1=0.9,
                line=dict(color=AWS_COLORS["dark_gray"], width=2, dash="dot"),
                xref="x",
                yref="y"
            )
        
        # Connect last container to output
        pipeline_graph.add_shape(
            type="line",
            x0=x_positions[2],
            y0=0.9,
            x1=data_x[3],
            y1=0.1,
            line=dict(color=AWS_COLORS["dark_gray"], width=2, dash="dot"),
            xref="x",
            yref="y"
        )
        
        # Update layout
        pipeline_graph.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            width=800,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, 6.5]
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, 1.5]
            )
        )
        
        st.plotly_chart(pipeline_graph, use_container_width=True)
        
        # Container details
        st.subheader("Container Details")
        
        # Show each container and its details
        for i, container in enumerate(containers):
            with st.expander(f"{i+1}. {container['name']} Container", expanded=(i==0)):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"**Image:** `{container['image']}`")
                    st.markdown(f"**Description:** {container['description']}")
                
                with col2:
                    st.markdown(f"**Input:** {container['input']}")
                    st.markdown(f"**Output:** {container['output']}")
        
        # Data transformation visualization
        st.subheader("Data Transformation Flow")
        
        # Create an example request and trace it through the pipeline
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if selected_pipeline_key == "nlp":
                example_input = st.text_input(
                    "Enter example text:", 
                    value=pipeline_config['example_input']
                )
            elif selected_pipeline_key == "vision":
                st.image("https://images.unsplash.com/photo-1518791841217-8f162f1e1131?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=60",
                        caption="Sample input image", use_container_width=True)
                example_input = "[Image data]"
            else:  # tabular
                input_dict = pipeline_config['example_input']
                
                example_age = st.slider("Age", 18, 80, input_dict['age'])
                example_income = st.slider("Income", 10000, 200000, input_dict['income'], step=5000)
                example_credit_score = st.slider("Credit Score", 300, 850, input_dict['credit_score'])
                
                example_input = {
                    'age': example_age,
                    'income': example_income,
                    'credit_score': example_credit_score
                }
            
            # Add a button to process the input
            process_clicked = st.button("Process Input")
            
            if process_clicked:
                # Store the input and processing state
                st.session_state.last_input = example_input
                st.session_state.processed = True
            
        with col2:
            if 'processed' in st.session_state and st.session_state.processed:
                # Show the processing steps
                timeline_data = []
                
                # Add input step
                timeline_data.append({
                    "Time (ms)": 0,
                    "Step": "Input",
                    "Data": str(st.session_state.last_input)
                })
                
                # Preprocessing step
                preprocess_time = 50 if selected_pipeline_key == "nlp" else 120 if selected_pipeline_key == "vision" else 30
                preprocessed_data = "Tokenized and normalized text features" if selected_pipeline_key == "nlp" else "Normalized image tensor" if selected_pipeline_key == "vision" else "Scaled features, encoded categories"
                
                timeline_data.append({
                    "Time (ms)": preprocess_time,
                    "Step": "Preprocessing",
                    "Data": preprocessed_data
                })
                
                # Inference step
                inference_time = 120 if selected_pipeline_key == "nlp" else 180 if selected_pipeline_key == "vision" else 80
                inference_data = "Raw sentiment scores, token embeddings" if selected_pipeline_key == "nlp" else "Raw bounding boxes, class probabilities" if selected_pipeline_key == "vision" else "Model prediction scores"
                
                timeline_data.append({
                    "Time (ms)": preprocess_time + inference_time,
                    "Step": "Model Inference",
                    "Data": inference_data
                })
                
                # Postprocessing step
                postprocess_time = 35 if selected_pipeline_key == "nlp" else 60 if selected_pipeline_key == "vision" else 25
                result = pipeline_config['example_output']
                
                timeline_data.append({
                    "Time (ms)": preprocess_time + inference_time + postprocess_time,
                    "Step": "Postprocessing",
                    "Data": str(result)
                })
                
                # Convert to DataFrame
                timeline_df = pd.DataFrame(timeline_data)
                
                # Create a visualization of the processing timeline
                timeline_chart = alt.Chart(timeline_df).mark_line(point=True).encode(
                    x=alt.X('Time (ms):Q', title='Processing Time (ms)'),
                    y=alt.Y('Step:N', title=None, sort=None)
                ).properties(
                    width=500,
                    height=200,
                    title='Processing Timeline'
                )
                
                st.altair_chart(timeline_chart, use_container_width=True)
                
                # Show the final result
                st.subheader("Final Output")
                st.json(result)
            else:
                st.info("Click 'Process Input' to see how data flows through the pipeline.")
        
        # Advantage demonstration
        st.subheader("Benefits of Pipeline Architecture")
        
        # Create a performance comparison
        perf_fig = create_performance_chart("pipeline")
        st.plotly_chart(perf_fig, use_container_width=True, key='perf_fig')
        
        # Highlight specific benefits
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Development Benefits
            
            - **Specialized Development**: Different teams can focus on their expertise areas
            - **Component Reusability**: Preprocessing and postprocessing components can be reused
            - **Simplified Testing**: Each component can be tested independently
            - **Framework Flexibility**: Mix different ML frameworks in one pipeline
            - **Reduced Coupling**: Cleaner interfaces between components
            """)
        
        with col2:
            st.markdown("""
            ### Operational Benefits
            
            - **Single Endpoint Management**: One endpoint to monitor and maintain
            - **End-to-End Optimization**: Holistic performance monitoring
            - **Unified Logging**: All processing steps in one log stream
            - **Simplified Deployment**: Deploy entire pipeline with one API call
            - **Consistent Environment**: Guaranteed compatibility between stages
            """)
        
        # Best practices
        st.subheader("Best Practices for Inference Pipelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### When to Use
            
            - When you need **complex preprocessing** before inference
            - For **post-processing** or results formatting
            - To **generate explanations** for model predictions
            - When using **multiple frameworks** for different stages
            - For **standardizing input** across different clients
            """)
        
        with col2:
            st.markdown("""
            ### Implementation Tips
            
            - **Define clear interfaces** between containers
            - **Test each container** independently before combining
            - **Optimize data serialization** between containers
            - **Monitor processing time** at each stage
            - **Size instances** appropriately for all containers
            - **Consider container overhead** in resource planning
            """)
        
        # Sample code
        st.subheader("Sample Code: Creating an Inference Pipeline")
        
        st.code(generate_sample_code("inference_pipeline"), language="python")

    # Add footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    main()
