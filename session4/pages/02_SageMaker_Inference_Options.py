
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


def generate_random_latency_data(
    mean: float = 100, 
    std: float = 20, 
    n_samples: int = 100, 
    min_val: float = 10
) -> List[float]:
    """
    Generate random latency data with a normal distribution
    
    Args:
        mean: Mean latency in milliseconds
        std: Standard deviation in milliseconds
        n_samples: Number of samples to generate
        min_val: Minimum latency value
        
    Returns:
        List of latency values
    """
    # Generate normally distributed latencies but ensure they're above min_val
    latencies = np.random.normal(mean, std, n_samples)
    return [max(min_val, l) for l in latencies]


def create_latency_distribution_chart(
    latencies: List[float], 
    title: str = "Latency Distribution", 
    color: str = "#FF9900"
) -> alt.Chart:
    """
    Create an Altair chart showing latency distribution
    
    Args:
        latencies: List of latency values
        title: Chart title
        color: Color for the histogram bars
        
    Returns:
        Altair chart object
    """
    df = pd.DataFrame({"latency": latencies})
    
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('latency:Q', bin=alt.Bin(maxbins=20), title='Latency (ms)'),
        alt.Y('count()', title='Count'),
        color=alt.value(color)
    ).properties(
        title=title,
        width='container',
        height=300
    )
    
    return chart


def generate_random_inference_metrics(
    inference_type: str, 
    n_datapoints: int = 30
) -> pd.DataFrame:
    """
    Generate random metrics for inference visualizations
    
    Args:
        inference_type: Type of inference (realtime, async, batch, serverless)
        n_datapoints: Number of time points to generate
        
    Returns:
        DataFrame with inference metrics
    """
    np.random.seed(42 + hash(inference_type) % 100)  # Different seed per inference type
    
    # Base characteristics for different inference types
    characteristics = {
        "realtime": {
            "latency_mean": 100,  # ms
            "latency_std": 30,
            "throughput_mean": 100,  # requests/second
            "throughput_std": 20,
            "cost_mean": 1.5,  # $ per hour
            "cost_std": 0.2,
            "scalability": "medium"
        },
        "async": {
            "latency_mean": 60000,  # 1 minute in ms
            "latency_std": 30000,
            "throughput_mean": 500,
            "throughput_std": 100,
            "cost_mean": 1.0,
            "cost_std": 0.15,
            "scalability": "high"
        },
        "batch": {
            "latency_mean": 300000,  # 5 minutes in ms
            "latency_std": 90000,
            "throughput_mean": 2000,
            "throughput_std": 300,
            "cost_mean": 0.8,
            "cost_std": 0.1,
            "scalability": "very high"
        },
        "serverless": {
            "latency_mean": 150,  # ms
            "latency_std": 50,
            "throughput_mean": 80,
            "throughput_std": 20,
            "cost_mean": 0.5,
            "cost_std": 0.1,
            "scalability": "auto"
        }
    }
    
    char = characteristics[inference_type]
    
    # Generate timestamps (one per day for n_datapoints days)
    base_date = datetime.now() - timedelta(days=n_datapoints)
    dates = [base_date + timedelta(days=i) for i in range(n_datapoints)]
    
    # Generate metrics
    latencies = np.random.normal(char["latency_mean"], char["latency_std"], n_datapoints)
    throughputs = np.random.normal(char["throughput_mean"], char["throughput_std"], n_datapoints)
    costs = np.random.normal(char["cost_mean"], char["cost_std"], n_datapoints)
    
    # Ensure values are reasonable
    latencies = np.maximum(latencies, 10)  # Minimum 10ms latency
    throughputs = np.maximum(throughputs, 10)  # Minimum 10 req/s
    costs = np.maximum(costs, 0.1)  # Minimum $0.1/hour
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "latency": latencies,
        "throughput": throughputs,
        "cost": costs,
        "type": inference_type
    })
    
    return df


def create_time_series_chart(
    df: pd.DataFrame, 
    metric: str, 
    color: str = "#FF9900", 
    title: Optional[str] = None
) -> alt.Chart:
    """
    Create an Altair time series chart for inference metrics
    
    Args:
        df: DataFrame with metrics
        metric: Metric to plot (latency, throughput, cost)
        color: Line color
        title: Chart title (optional)
        
    Returns:
        Altair chart object
    """
    if title is None:
        title = f"{metric.capitalize()} over time"
    
    chart = alt.Chart(df).mark_line(point=True).encode(
        alt.X('date:T', title='Date'),
        alt.Y(f'{metric}:Q', title=f'{metric.capitalize()}'),
        color=alt.value(color)
    ).properties(
        title=title,
        width='container',
        height=300
    )
    
    return chart


def create_comparison_chart(
    dfs: Dict[str, pd.DataFrame], 
    metric: str, 
    colors: Dict[str, str]
) -> alt.Chart:
    """
    Create an Altair chart comparing a metric across inference types
    
    Args:
        dfs: Dictionary of DataFrames with inference types as keys
        metric: Metric to compare
        colors: Dictionary of colors with inference types as keys
        
    Returns:
        Altair chart object
    """
    # Combine all dataframes
    combined_df = pd.concat(dfs.values())
    
    # Create chart
    chart = alt.Chart(combined_df).mark_line().encode(
        alt.X('date:T', title='Date'),
        alt.Y(f'{metric}:Q', title=f'{metric.capitalize()}'),
        alt.Color('type:N', title='Inference Type',
                 scale=alt.Scale(domain=list(colors.keys()),
                                range=list(colors.values())))
    ).properties(
        title=f"{metric.capitalize()} Comparison",
        width='container',
        height=400
    )
    
    return chart


def create_inference_flow_diagram(
    inference_type: str, 
    fig_width: int = 800, 
    fig_height: int = 400
) -> plt.Figure:
    """
    Create a flow diagram showing the inference process for a given inference type
    
    Args:
        inference_type: Type of inference (realtime, async, batch, serverless)
        fig_width: Figure width in pixels
        fig_height: Figure height in pixels
        
    Returns:
        Matplotlib Figure object
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=(fig_width/100, fig_height/100), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
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
        "red": "#D13212"
    }
    
    # Common elements
    # Client
    client = plt.Rectangle((0.5, 3.5), 1.5, 1, fc=AWS_COLORS["blue"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
    ax.add_patch(client)
    ax.text(1.25, 4, "Client", ha='center', va='center', color='white', fontweight='bold')
    
    if inference_type == "realtime":
        # Real-time components
        endpoint = plt.Rectangle((5, 3.5), 2, 1, fc=AWS_COLORS["orange"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(endpoint)
        ax.text(6, 4, "Real-time Endpoint", ha='center', va='center', color='white', fontweight='bold')
        
        # Arrows
        ax.arrow(2.2, 4, 2.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(3.5, 4.2, "Request", ha='center', va='bottom')
        
        ax.arrow(5, 3.75, -2.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"])
        ax.text(3.5, 3.6, "Response (Synchronous)", ha='center', va='top', color=AWS_COLORS["green"])
        
        # Container(s)
        container = plt.Rectangle((8, 3.5), 1.5, 1, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(container)
        ax.text(8.75, 4, "Model\nContainer", ha='center', va='center', color='white', fontweight='bold')
        
        # Connection between endpoint and container
        ax.arrow(7.1, 4, 0.7, 0, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.arrow(8, 3.75, -0.7, 0, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Add text highlighting key characteristics
        ax.text(5, 2.5, "• Low latency, synchronous responses\n• Always running\n• Pay for provisioned capacity\n• Ideal for interactive applications", 
               fontsize=10, ha='center', va='center', bbox=dict(facecolor=AWS_COLORS["light_gray"], alpha=0.5))
        
    elif inference_type == "async":
        # Async components
        endpoint = plt.Rectangle((4, 3.5), 2, 1, fc=AWS_COLORS["orange"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(endpoint)
        ax.text(5, 4, "Async Endpoint", ha='center', va='center', color='white', fontweight='bold')
        
        # S3 bucket
        s3_in = plt.Rectangle((4, 1.5), 1, 1, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(s3_in)
        ax.text(4.5, 2, "S3 Input", ha='center', va='center', color='white', fontweight='bold')
        
        s3_out = plt.Rectangle((7, 1.5), 1, 1, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(s3_out)
        ax.text(7.5, 2, "S3 Output", ha='center', va='center', color='white', fontweight='bold')
        
        # SQS
        sqs = plt.Rectangle((5.5, 1.5), 1, 1, fc=AWS_COLORS["green"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(sqs)
        ax.text(6, 2, "Queue", ha='center', va='center', color='white', fontweight='bold')
        
        # Container
        container = plt.Rectangle((8, 3.5), 1.5, 1, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(container)
        ax.text(8.75, 4, "Model\nContainer", ha='center', va='center', color='white', fontweight='bold')
        
        # Arrows
        # Client to endpoint
        ax.arrow(2.2, 4, 1.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(3, 4.2, "Submit Job", ha='center', va='bottom')
        
        # Endpoint to client
        ax.arrow(4, 3.75, -1.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"])
        ax.text(3, 3.6, "Job ID", ha='center', va='top', color=AWS_COLORS["green"])
        
        # Endpoint to S3
        ax.arrow(4.5, 3.5, 0, -0.8, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # S3 to Queue
        ax.arrow(5.1, 2, 0.3, 0, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Queue to endpoint
        ax.arrow(6, 2.5, 0, 0.8, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Endpoint to container
        ax.arrow(6.1, 4, 1.7, 0, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Container to S3 output
        ax.arrow(8.5, 3.5, -0.5, -0.8, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Client checking S3
        ax.arrow(1.25, 3.5, 5.5, -1.5, head_width=0.1, head_length=0.1, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"], 
                linestyle='dashed')
        ax.text(4, 2.7, "Check Result", ha='center', va='center', color=AWS_COLORS["green"], fontsize=9)
        
        # Add text highlighting key characteristics
        ax.text(2, 0.7, "• Asynchronous processing\n• Queued requests\n• Results available in S3\n• Ideal for large payloads or long processing times", 
               fontsize=10, ha='left', va='center', bbox=dict(facecolor=AWS_COLORS["light_gray"], alpha=0.5))
        
    elif inference_type == "batch":
        # Batch components
        batch_job = plt.Rectangle((5, 3.5), 2, 1, fc=AWS_COLORS["orange"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(batch_job)
        ax.text(6, 4, "Batch Transform Job", ha='center', va='center', color='white', fontweight='bold')
        
        # S3 bucket
        s3_in = plt.Rectangle((3, 1.5), 1, 1, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(s3_in)
        ax.text(3.5, 2, "S3 Input", ha='center', va='center', color='white', fontweight='bold')
        
        s3_out = plt.Rectangle((9, 1.5), 1, 1, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(s3_out)
        ax.text(9.5, 2, "S3 Output", ha='center', va='center', color='white', fontweight='bold')
        
        # Worker nodes
        for i in range(3):
            worker = plt.Rectangle((5.5 + i*1.5, 2), 1, 0.8, fc=AWS_COLORS["green"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
            ax.add_patch(worker)
            ax.text(6 + i*1.5, 2.4, f"Worker {i+1}", ha='center', va='center', color='white', fontweight='bold')
        
        # Arrows
        # Client to batch job
        ax.arrow(2.2, 4, 2.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(3.5, 4.2, "Configure & Start", ha='center', va='bottom')
        
        # Batch job to client
        ax.arrow(5, 3.75, -2.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"])
        ax.text(3.5, 3.6, "Job Status", ha='center', va='top', color=AWS_COLORS["green"])
        
        # Batch job to input
        ax.arrow(5.5, 3.5, -1.5, -1, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(4.7, 3, "Read Data", ha='center', va='center', fontsize=9)
        
        # Batch job to workers
        for i in range(3):
            ax.arrow(6, 3.5, 0.2 + i*1.5, -0.5, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Workers to output
        for i in range(3):
            ax.arrow(6.5 + i*1.5, 2, 2 - i*1.5, 0, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Client checking S3 output
        ax.arrow(1.25, 3.5, 7.5, -1.5, head_width=0.1, head_length=0.1, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"], 
                linestyle='dashed')
        ax.text(5, 2.5, "Check Results", ha='center', va='center', color=AWS_COLORS["green"], fontsize=9)
        
        # Add text highlighting key characteristics
        ax.text(2, 0.7, "• Process entire datasets at once\n• Highly scalable with multiple workers\n• Cost-effective for large datasets\n• Ideal for offline processing", 
               fontsize=10, ha='left', va='center', bbox=dict(facecolor=AWS_COLORS["light_gray"], alpha=0.5))
        
    else:  # serverless
        # Serverless components
        endpoint = plt.Rectangle((5, 3.5), 2, 1, fc=AWS_COLORS["orange"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
        ax.add_patch(endpoint)
        ax.text(6, 4, "Serverless Endpoint", ha='center', va='center', color='white', fontweight='bold')
        
        # Auto-scaling container fleet
        container_group = plt.Rectangle((7.5, 1.5), 2, 1.5, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.4)
        ax.add_patch(container_group)
        ax.text(8.5, 2.5, "Auto-scaling\nContainer Fleet", ha='center', va='center', fontsize=9)
        
        for i in range(3):
            container = plt.Rectangle((7.7 + i*0.6, 1.7), 0.5, 0.5, fc=AWS_COLORS["teal"], ec=AWS_COLORS["dark_gray"], alpha=0.8)
            ax.add_patch(container)
            
        # Arrows
        # Client to endpoint
        ax.arrow(2.2, 4, 2.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        ax.text(3.5, 4.2, "Request", ha='center', va='bottom')
        
        # Endpoint to client
        ax.arrow(5, 3.75, -2.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["green"], ec=AWS_COLORS["green"])
        ax.text(3.5, 3.6, "Response", ha='center', va='top', color=AWS_COLORS["green"])
        
        # Endpoint to container fleet
        ax.arrow(6, 3.5, 1.5, -1, head_width=0.1, head_length=0.1, fc=AWS_COLORS["dark_gray"], ec=AWS_COLORS["dark_gray"])
        
        # Elastic scaling visualization
        elastic_arrow1 = plt.Arrow(8, 1.2, 0, -0.5, width=0.1, fc=AWS_COLORS["green"])
        elastic_arrow2 = plt.Arrow(8.5, 1.2, 0, -0.5, width=0.1, fc=AWS_COLORS["green"])
        elastic_arrow3 = plt.Arrow(9, 1.2, 0, -0.5, width=0.1, fc=AWS_COLORS["green"])
        ax.add_patch(elastic_arrow1)
        ax.add_patch(elastic_arrow2)
        ax.add_patch(elastic_arrow3)
        
        # Add text highlighting key characteristics
        ax.text(3, 2, "• Auto-scaling to zero when idle\n• Pay only for inference time used\n• No infrastructure management\n• Ideal for variable or unpredictable workloads", 
               fontsize=10, ha='left', va='center', bbox=dict(facecolor=AWS_COLORS["light_gray"], alpha=0.5))
        
        # Add cost efficiency indicator
        cost_scale = plt.Rectangle((5, 0.7), 4, 0.3, fc=AWS_COLORS["light_gray"], ec=AWS_COLORS["dark_gray"], alpha=0.4)
        ax.add_patch(cost_scale)
        
        # Gradient from low to high usage
        for i in range(40):
            cost_bar = plt.Rectangle((5 + i*0.1, 0.7), 0.1, 0.3, fc=plt.cm.RdYlGn_r(i/40), alpha=0.7)
            ax.add_patch(cost_bar)
        
        ax.text(5, 0.5, "Low Usage", ha='left', va='top', fontsize=8)
        ax.text(9, 0.5, "High Usage", ha='right', va='top', fontsize=8)
        ax.text(7, 0.5, "Cost Efficiency", ha='center', va='top', fontsize=9, fontweight='bold')
    
    return fig


def create_case_study_data(
    case_study_type: str
) -> Dict[str, Any]:
    """
    Generate data for different inference case studies
    
    Args:
        case_study_type: Type of case study (e-commerce, healthcare, financial, manufacturing)
        
    Returns:
        Dictionary with case study data
    """
    case_studies = {
        "e-commerce": {
            "title": "E-commerce Product Recommendations",
            "description": """
            An e-commerce platform needs to provide personalized product recommendations to users 
            browsing their website. The recommendations need to be generated in real-time as users 
            navigate through the site.
            """,
            "requirements": [
                "Low latency responses (< 100ms)",
                "High throughput during peak shopping events",
                "Real-time personalization",
                "Cost efficiency during normal operations"
            ],
            "solution": "Real-Time Inference with Auto-Scaling",
            "solution_details": """
            A real-time SageMaker endpoint provides the low-latency recommendations needed for 
            interactive user experiences. To manage costs while handling peak loads, the endpoint 
            uses auto-scaling policies that adjust capacity based on traffic patterns.
            """,
            "inference_type": "realtime",
            "model_type": "XGBoost Recommender",
            "metrics": {
                "latency": "85ms (p95)",
                "throughput": "Up to 1,200 requests/second during peak",
                "cost_savings": "42% compared to fixed capacity"
            },
            "scaling_policy": "Target tracking based on model CPU utilization (65%)",
            "architecture_notes": """
            - Endpoint instances: ml.c6g.xlarge (cost-efficient ARM-based instances)
            - Auto-scaling group: 2-20 instances
            - CloudFront for request caching
            - Feature store for real-time user data
            """
        },
        "healthcare": {
            "title": "Medical Image Analysis",
            "description": """
            A healthcare provider processes thousands of medical images (X-rays, MRIs, CT scans) 
            overnight for analysis. The analysis doesn't need to be immediate but must be 
            completed by the morning for physician review.
            """,
            "requirements": [
                "Process large image files (up to 2GB each)",
                "Complete analysis of 5,000+ images overnight",
                "Cost-efficient processing",
                "Reliable batch processing"
            ],
            "solution": "Batch Transform",
            "solution_details": """
            A daily batch transform job processes all accumulated medical images during off-hours.
            The job uses distributed processing with multiple compute instances to ensure all 
            images are analyzed by morning. Results are stored in S3 for physician access.
            """,
            "inference_type": "batch",
            "model_type": "Medical Vision Transformer",
            "metrics": {
                "processing_time": "4.5 hours for 5,000 images",
                "accuracy": "97.8% diagnostic agreement with radiologists",
                "cost_savings": "68% compared to on-demand processing"
            },
            "architecture_notes": """
            - Compute: 10x ml.g4dn.xlarge instances (GPU acceleration)
            - S3 storage with lifecycle policies
            - Input splitting: 100 images per mini-batch
            - Results integrated with healthcare data system via AWS HealthLake
            """
        },
        "financial": {
            "title": "Credit Card Fraud Detection",
            "description": """
            A financial services company needs to analyze credit card transactions for potential 
            fraud. The system must handle spikes in transaction volume while minimizing 
            infrastructure costs during quiet periods.
            """,
            "requirements": [
                "Handle variable transaction volumes",
                "Process transactions within seconds",
                "Scale to zero during quiet hours",
                "Pay only for actual usage"
            ],
            "solution": "Serverless Inference",
            "solution_details": """
            Serverless inference endpoints process transactions without requiring provisioned 
            capacity. The system automatically scales up during high-volume periods (like 
            holidays) and scales to zero during quiet hours, ensuring optimal cost efficiency.
            """,
            "inference_type": "serverless",
            "model_type": "Gradient Boosting Fraud Detector",
            "metrics": {
                "latency": "210ms (p95)",
                "cold start": "890ms first request",
                "cost_savings": "76% compared to always-on endpoints"
            },
            "traffic_pattern": "Highly variable with 20x difference between peak and quiet periods",
            "architecture_notes": """
            - Memory configuration: 4GB
            - Concurrency: Up to 50 transactions in parallel
            - API Gateway with usage plans
            - Amazon EventBridge for monitoring suspicious activities
            """
        },
        "manufacturing": {
            "title": "Industrial Equipment Defect Analysis",
            "description": """
            A manufacturing company needs to process high-resolution images of equipment 
            components to identify defects. The process involves large image files and complex 
            analysis that takes 5-10 minutes per component.
            """,
            "requirements": [
                "Handle large image files (1GB+)",
                "Allow long processing times (5-10 minutes)",
                "Process in background without blocking operations",
                "Notify engineers when analysis is complete"
            ],
            "solution": "Asynchronous Inference",
            "solution_details": """
            Asynchronous inference endpoints process the large component images without timeouts. 
            Engineers upload images and receive a job ID, then get notified via SNS when the 
            analysis is complete. This approach prevents timeout issues and optimizes resource usage.
            """,
            "inference_type": "async",
            "model_type": "Deep Learning Defect Detector",
            "metrics": {
                "average_processing_time": "7.5 minutes per component",
                "throughput": "120 components analyzed per hour",
                "defect_detection_accuracy": "99.2%"
            },
            "architecture_notes": """
            - Instance type: ml.g5.2xlarge (NVIDIA A10G GPU)
            - SNS notifications for job completions
            - S3 bucket for image storage and results
            - CloudWatch dashboard for processing statistics
            """
        }
    }
    
    return case_studies.get(case_study_type, {})


def create_inference_pricing_table() -> pd.DataFrame:
    """
    Create a pricing comparison table for different inference options
    
    Returns:
        DataFrame with pricing information
    """
    pricing_data = {
        "Inference Type": ["Real-Time", "Asynchronous", "Batch Transform", "Serverless"],
        "Pricing Model": [
            "Pay for provisioned instances",
            "Pay for provisioned instances",
            "Pay for instance hours used",
            "Pay per inference duration"
        ],
        "Instance Types": [
            "All ML instance types",
            "All ML instance types",
            "All ML instance types", 
            "Memory-based configuration"
        ],
        "Min. Billing": [
            "Continuous (always running)",
            "Continuous (while running)",
            "1-minute minimum",
            "1-second minimum"
        ],
        "Best For": [
            "Low-latency, consistent workloads",
            "Long-running predictions, large inputs",
            "Offline prediction on large datasets",
            "Variable or unpredictable workloads"
        ],
        "Cost Efficiency (Low Traffic)": ["Low", "Medium", "High", "Very High"],
        "Cost Efficiency (High Traffic)": ["Very High", "High", "Medium", "Medium"],
    }
    
    return pd.DataFrame(pricing_data)


def generate_sample_code(inference_type: str) -> str:
    """
    Generate sample code for different inference types
    
    Args:
        inference_type: Type of inference (realtime, async, batch, serverless)
        
    Returns:
        String containing sample Python code
    """
    if inference_type == "realtime":
        return '''
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define model artifacts and container
model_artifacts = "s3://your-bucket/model/model.tar.gz"
container = "123456789012.dkr.ecr.us-east-1.amazonaws.com/your-container:latest"

# Create model
model = Model(
    image_uri=container,
    model_data=model_artifacts,
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy model to a real-time endpoint with auto-scaling
predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.xlarge",
    endpoint_name="realtime-endpoint-example",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# Configure auto-scaling
client = boto3.client('application-autoscaling')
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/realtime-endpoint-example/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=5
)

client.put_scaling_policy(
    PolicyName='InvocationsScalingPolicy',
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/realtime-endpoint-example/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleOutCooldown': 60,
        'ScaleInCooldown': 300
    }
)

# Make a prediction
response = predictor.predict({
    "features": [0.5, 0.2, 0.1, 0.7]
})
print(response)

# Delete the endpoint when done
predictor.delete_endpoint()
        '''
    elif inference_type == "async":
        return '''
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define model artifacts and container
model_artifacts = "s3://your-bucket/model/model.tar.gz"
container = "123456789012.dkr.ecr.us-east-1.amazonaws.com/your-container:latest"

# Create model
model = Model(
    image_uri=container,
    model_data=model_artifacts,
    role=role,
    sagemaker_session=sagemaker_session
)

# Configure S3 locations and notifications for async inference
s3_output_path = "s3://your-bucket/async-inference-results/"
sns_topic_arn = "arn:aws:sns:us-east-1:123456789012:async-inference-notifications"

async_config = AsyncInferenceConfig(
    output_path=s3_output_path,
    notification_config={
        "SuccessTopic": sns_topic_arn,
        "ErrorTopic": sns_topic_arn
    },
    max_concurrent_invocations_per_instance=4
)

# Deploy model as async endpoint
async_predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="async-endpoint-example",
    async_inference_config=async_config
)

# Submit async inference request
input_data = "s3://your-bucket/input/large-payload.json"
response = async_predictor.predict_async(
    input_data,
    inference_id="unique-inference-id-123"
)

print(f"Inference ID: {response['InferenceId']}")
print(f"Output Path: {response['OutputPath']}")

# Check inference status
runtime_sm_client = boto3.client('sagemaker-runtime')
response = runtime_sm_client.describe_inference_id(
    EndpointName="async-endpoint-example",
    InferenceId="unique-inference-id-123"
)
print(f"Status: {response['Status']}")

# Delete the endpoint when done
async_predictor.delete_endpoint()
        '''
    elif inference_type == "batch":
        return '''
import sagemaker
from sagemaker.transformer import Transformer

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define model artifacts and container
model_artifacts = "s3://your-bucket/model/model.tar.gz"
container = "123456789012.dkr.ecr.us-east-1.amazonaws.com/your-container:latest"

# Specify input and output S3 paths
input_data_path = "s3://your-bucket/batch-input/"
output_data_path = "s3://your-bucket/batch-output/"

# Create transformer for batch processing
transformer = Transformer(
    model_name="batch-transform-model",
    instance_count=4,           # Use multiple instances for parallelization
    instance_type="ml.c5.2xlarge",
    output_path=output_data_path,
    sagemaker_session=sagemaker_session,
    strategy="MultiRecord",     # Process multiple records in each request
    max_payload=6,              # Max size in MB for each request
    max_concurrent_transforms=8,# Max number of parallel requests per instance
    assemble_with="Line",       # How to combine results
    accept="application/json"   # Output format
)

# Start batch transform job
transformer.transform(
    data=input_data_path,
    data_type="S3Prefix",
    content_type="text/csv",
    split_type="Line",
    job_name="example-batch-job",
    experiment_config={
        "ExperimentName": "my-experiment",
        "TrialName": "my-trial",
        "TrialComponentDisplayName": "my-batch-job"
    }
)

# Wait for the batch job to complete
transformer.wait()

print(f"Batch transform job completed. Results stored at: {output_data_path}")

# Access transform job metrics
transform_job_name = transformer.latest_transform_job.job_name
transform_job_desc = sagemaker_session.sagemaker_client.describe_transform_job(
    TransformJobName=transform_job_name
)
print(f"Job Status: {transform_job_desc['TransformJobStatus']}")
        '''
    elif inference_type == "serverless":
        return '''
import sagemaker
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define model artifacts and container
model_artifacts = "s3://your-bucket/model/model.tar.gz"
container = "123456789012.dkr.ecr.us-east-1.amazonaws.com/your-container:latest"

# Create model
model = Model(
    image_uri=container,
    model_data=model_artifacts,
    role=role,
    sagemaker_session=sagemaker_session
)

# Configure serverless endpoint
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=3072,  # Memory in MB (1024, 2048, 3072, 4096, 5120, or 6144)
    max_concurrency=10       # Maximum concurrent invocations
)

# Deploy model to serverless endpoint
predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="serverless-endpoint-example",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# Make a prediction
response = predictor.predict({
    "features": [0.5, 0.2, 0.1, 0.7]
})
print(response)

# Delete the endpoint when done
predictor.delete_endpoint()
        '''
    else:
        return "# No sample code available for this inference type"


def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'realtime_metrics' not in st.session_state:
        st.session_state.realtime_metrics = generate_random_inference_metrics("realtime")
        
    if 'async_metrics' not in st.session_state:
        st.session_state.async_metrics = generate_random_inference_metrics("async")
        
    if 'batch_metrics' not in st.session_state:
        st.session_state.batch_metrics = generate_random_inference_metrics("batch")
        
    if 'serverless_metrics' not in st.session_state:
        st.session_state.serverless_metrics = generate_random_inference_metrics("serverless")
    
    if 'latency_data' not in st.session_state:
        st.session_state.latency_data = {
            'realtime': generate_random_latency_data(mean=100, std=20, n_samples=100),
            'async': generate_random_latency_data(mean=60000, std=10000, n_samples=100),
            'batch': generate_random_latency_data(mean=300000, std=50000, n_samples=100),
            'serverless': generate_random_latency_data(mean=150, std=50, n_samples=100)
        }


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
        page_title="SageMaker Inference Options",
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
        .case-study-box {
            border: 1px solid #E9ECEF;
            border-left: 5px solid #FF9900;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
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
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #232F3E;
        }
        .metric-label {
            font-size: 0.9em;
            color: #545B64;
            margin-top: 5px;
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
            various inference options. Explore each tab to understand the different 
            deployment strategies and when to use each one.
            """)
        
            # AWS learning resources
            st.subheader("Additional Resources")
            st.markdown("""
                - [SageMaker Inference Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
                - [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
                - [AWS Training and Certification](https://aws.amazon.com/training/)
            """)
    
    # Main app header
    st.title("Amazon SageMaker Inference Options")
    st.markdown("""
    Learn about the different deployment options available in Amazon SageMaker 
    for serving machine learning models in production.
    """)
    
    # # Animation for the header
    # lottie_url = "https://assets1.lottiefiles.com/packages/lf20_kuhijlvx.json"
    # lottie_json = load_lottieurl(lottie_url)
    # if lottie_json:
    #     st_lottie(lottie_json, height=200, key="header_animation")
    
    # Tab-based navigation with emoji
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Overview", 
        "⚡ Real-Time Inference", 
        "🕒 Asynchronous Inference",
        "📦 Batch Transform",
        "☁️ Serverless Inference"
    ])
    
    # OVERVIEW TAB
    with tab1:
        st.header("SageMaker Inference Options Overview")
        
        st.markdown("""
        Amazon SageMaker offers multiple deployment options to serve your machine learning 
        models, each optimized for different use cases and requirements. Understanding these 
        options helps you choose the right deployment strategy for your specific needs.
        """)
        
        # Comparison table
        st.subheader("Inference Options Comparison")
        
        comparison_data = {
            "Feature": ["Response Type", "Max Request Timeout", "Max Payload Size", "Auto-scaling", "Use Case"],
            "Real-Time": ["Synchronous", "60 seconds", "6 MB", "Yes", "Low-latency, interactive applications"],
            "Asynchronous": ["Asynchronous", "Hours/days", "1 GB", "Yes", "Large payloads, long processing times"],
            "Batch Transform": ["Offline", "N/A", "100 MB per record", "N/A", "Bulk processing of datasets"],
            "Serverless": ["Synchronous", "60 seconds", "4 MB", "Automatic", "Variable workloads, cost optimization"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Decision tree visualization
        st.subheader("Choosing the Right Inference Option")
        
        decision_tree_fig = go.Figure()
        
        # Create nodes for the decision tree
        decision_tree_fig.add_trace(go.Scatter(
            x=[5], y=[5],
            mode="markers+text",
            marker=dict(size=30, color=AWS_COLORS["blue"]),
            text=["Start"],
            textposition="middle center",
            textfont=dict(color="white", size=12),
            hoverinfo="skip"
        ))
        
        # Add decision points
        decision_tree_fig.add_trace(go.Scatter(
            x=[5, 2, 8, 2, 8],
            y=[4, 3, 3, 2, 2],
            mode="markers+text",
            marker=dict(size=[25, 25, 25, 25, 25], 
                       color=[AWS_COLORS["teal"], AWS_COLORS["teal"], 
                             AWS_COLORS["teal"], AWS_COLORS["teal"], 
                             AWS_COLORS["teal"]]),
            text=["Need immediate<br>response?", 
                 "Low latency<br>requirement?", 
                 "Processing<br>entire dataset?",
                 "Consistent<br>workload?",
                 "Large payload or<br>long processing?"],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            hoverinfo="skip"
        ))
        
        # Add endpoints/solutions
        decision_tree_fig.add_trace(go.Scatter(
            x=[1, 3, 7, 9],
            y=[1, 1, 1, 1],
            mode="markers+text",
            marker=dict(size=[30, 30, 30, 30], 
                       color=[AWS_COLORS["orange"], AWS_COLORS["green"], 
                             AWS_COLORS["orange"], AWS_COLORS["green"]]),
            text=["Real-Time<br>Inference", 
                 "Serverless<br>Inference", 
                 "Batch<br>Transform",
                 "Asynchronous<br>Inference"],
            textposition="middle center",
            textfont=dict(color="white", size=11),
            hoverinfo="skip"
        ))
        
        # Draw connecting lines
        decision_tree_fig.add_shape(type="line", x0=5, y0=5, x1=5, y1=4, line=dict(color="gray", width=2))
        
        # From "Need immediate response?" node
        decision_tree_fig.add_shape(type="line", x0=5, y0=4, x1=2, y1=3, line=dict(color="gray", width=2))
        decision_tree_fig.add_shape(type="line", x0=5, y0=4, x1=8, y1=3, line=dict(color="gray", width=2))
        
        # From "Low latency requirement?" node
        decision_tree_fig.add_shape(type="line", x0=2, y0=3, x1=2, y1=2, line=dict(color="gray", width=2))
        
        # From "Processing entire dataset?" node
        decision_tree_fig.add_shape(type="line", x0=8, y0=3, x1=8, y1=2, line=dict(color="gray", width=2))
        
        # From "Consistent workload?" node
        decision_tree_fig.add_shape(type="line", x0=2, y0=2, x1=1, y1=1, line=dict(color="gray", width=2))
        decision_tree_fig.add_shape(type="line", x0=2, y0=2, x1=3, y1=1, line=dict(color="gray", width=2))
        
        # From "Large payload or long processing?" node
        decision_tree_fig.add_shape(type="line", x0=8, y0=2, x1=7, y1=1, line=dict(color="gray", width=2))
        decision_tree_fig.add_shape(type="line", x0=8, y0=2, x1=9, y1=1, line=dict(color="gray", width=2))
        
        # Add Yes/No labels
        text_annotations = [
            dict(x=3.2, y=3.7, text="Yes", showarrow=False, font=dict(color=AWS_COLORS["green"])),
            dict(x=6.8, y=3.7, text="No", showarrow=False, font=dict(color=AWS_COLORS["red"])),
            dict(x=1.3, y=2.5, text="Yes", showarrow=False, font=dict(color=AWS_COLORS["green"])),
            dict(x=7.3, y=2.5, text="Yes", showarrow=False, font=dict(color=AWS_COLORS["green"])),
            dict(x=1.3, y=1.5, text="Yes", showarrow=False, font=dict(color=AWS_COLORS["green"])),
            dict(x=2.7, y=1.5, text="No", showarrow=False, font=dict(color=AWS_COLORS["red"])),
            dict(x=7.3, y=1.5, text="No", showarrow=False, font=dict(color=AWS_COLORS["red"])),
            dict(x=8.7, y=1.5, text="Yes", showarrow=False, font=dict(color=AWS_COLORS["green"])),
        ]
        for annotation in text_annotations:
            decision_tree_fig.add_annotation(**annotation)
        
        # Set up layout
        decision_tree_fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            width=800,
            height=500,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(decision_tree_fig, use_container_width=True)
        
        # Key metrics comparison
        st.subheader("Performance Comparison")
        
        # Create dictionary of metrics dataframes
        all_metrics = {
            "realtime": st.session_state.realtime_metrics,
            "async": st.session_state.async_metrics,
            "batch": st.session_state.batch_metrics,
            "serverless": st.session_state.serverless_metrics
        }
        
        # Colors for different inference types
        inference_colors = {
            "realtime": AWS_COLORS["orange"],
            "async": AWS_COLORS["teal"],
            "batch": AWS_COLORS["green"],
            "serverless": AWS_COLORS["blue"]
        }
        
        # Create three columns for different metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latency_chart = create_comparison_chart(all_metrics, "latency", inference_colors)
            st.altair_chart(latency_chart, use_container_width=True)
            
        with col2:
            throughput_chart = create_comparison_chart(all_metrics, "throughput", inference_colors)
            st.altair_chart(throughput_chart, use_container_width=True)
            
        with col3:
            cost_chart = create_comparison_chart(all_metrics, "cost", inference_colors)
            st.altair_chart(cost_chart, use_container_width=True)
            
        # Pricing information
        st.subheader("Pricing Models")
        
        pricing_df = create_inference_pricing_table()
        st.dataframe(pricing_df, use_container_width=True)
        
        # Case studies section
        st.subheader("Use Case Examples")
        
        # Create columns for case studies
        case_study_col1, case_study_col2 = st.columns(2)
        
        with case_study_col1:
            # E-commerce case study
            ecommerce_case = create_case_study_data("e-commerce")
            
            st.markdown(f"""
            <div class="case-study-box">
                <h4>{ecommerce_case["title"]}</h4>
                <p><em>{ecommerce_case["description"]}</em></p>
                <p><strong>Solution:</strong> {ecommerce_case["solution"]}</p>
                <p><strong>Key Metric:</strong> {ecommerce_case["metrics"]["latency"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Healthcare case study
            healthcare_case = create_case_study_data("healthcare")
            
            st.markdown(f"""
            <div class="case-study-box">
                <h4>{healthcare_case["title"]}</h4>
                <p><em>{healthcare_case["description"]}</em></p>
                <p><strong>Solution:</strong> {healthcare_case["solution"]}</p>
                <p><strong>Key Metric:</strong> {healthcare_case["metrics"]["processing_time"]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with case_study_col2:
            # Financial case study
            financial_case = create_case_study_data("financial")
            
            st.markdown(f"""
            <div class="case-study-box">
                <h4>{financial_case["title"]}</h4>
                <p><em>{financial_case["description"]}</em></p>
                <p><strong>Solution:</strong> {financial_case["solution"]}</p>
                <p><strong>Key Metric:</strong> {financial_case["metrics"]["cost_savings"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Manufacturing case study
            manufacturing_case = create_case_study_data("manufacturing")
            
            st.markdown(f"""
            <div class="case-study-box">
                <h4>{manufacturing_case["title"]}</h4>
                <p><em>{manufacturing_case["description"]}</em></p>
                <p><strong>Solution:</strong> {manufacturing_case["solution"]}</p>
                <p><strong>Key Metric:</strong> {manufacturing_case["metrics"]["average_processing_time"]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # REAL-TIME INFERENCE TAB
    with tab2:
        st.header("⚡ Real-Time Inference")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Real-Time Inference is ideal for applications that need immediate, low-latency responses.
            When you deploy a model to a SageMaker real-time endpoint, it remains running, waiting to 
            serve inference requests on demand.
            
            **Key features:**
            - **Synchronous responses** within milliseconds
            - **Auto-scaling** to handle varying traffic 
            - **High availability** with multi-AZ support
            - **Continuous monitoring** of deployed models
            """)
        
        with col2:
            st.image("images/realtime_inference.png", 
                     caption="SageMaker Real-Time Inference", use_container_width=True)
        
        # Real-time inference diagram
        st.subheader("How Real-Time Inference Works")
        
        realtime_fig = create_inference_flow_diagram("realtime")
        st.pyplot(realtime_fig)
        
        # Latency distribution
        st.subheader("Latency Distribution")
        
        latency_data = st.session_state.latency_data['realtime']
        latency_chart = create_latency_distribution_chart(
            latency_data, 
            title="Real-Time Inference Latency Distribution",
            color=AWS_COLORS["orange"]
        )
        st.altair_chart(latency_chart, use_container_width=True)
        
        # Real-time metrics section
        st.subheader("Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Latency (P50)", f"{np.percentile(latency_data, 50):.1f} ms")
            st.metric("Throughput", "100-1000 requests/sec per instance")
        
        with col2:
            st.metric("P95 Latency", f"{np.percentile(latency_data, 95):.1f} ms")
            st.metric("Maximum Payload Size", "6 MB")
        
        with col3:
            st.metric("P99 Latency", f"{np.percentile(latency_data, 99):.1f} ms")
            st.metric("Maximum Request Timeout", "60 seconds")
        
        # Auto-scaling visualization
        st.subheader("Auto-scaling Capabilities")
        
        # Create auto-scaling visualization with Plotly
        auto_scaling_fig = go.Figure()
        
        # Generate some auto-scaling data
        hours = list(range(24))
        traffic = [20, 18, 15, 10, 8, 10, 20, 40, 60, 80, 90, 95, 90, 85, 80, 75, 80, 85, 90, 80, 60, 50, 35, 25]
        instances = [1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1]
        
        # Add traffic line
        auto_scaling_fig.add_trace(go.Scatter(
            x=hours,
            y=traffic,
            mode='lines',
            name='Traffic (requests/sec)',
            line=dict(color=AWS_COLORS["blue"], width=3)
        ))
        
        # Add instance count
        auto_scaling_fig.add_trace(go.Scatter(
            x=hours,
            y=[i * 30 for i in instances],  # Scale for visualization
            mode='lines+markers',
            name='Instance Count',
            line=dict(color=AWS_COLORS["orange"], width=3),
            marker=dict(size=8)
        ))
        
        # Create secondary y-axis for instance count
        auto_scaling_fig.update_layout(
            title="Real-Time Endpoint Auto-scaling Example",
            xaxis_title="Hour of Day",
            yaxis=dict(
                title="Traffic (requests/sec)",
                # titlefont=dict(color=AWS_COLORS["blue"]),
                tickfont=dict(color=AWS_COLORS["blue"])
            ),
            yaxis2=dict(
                title="Instance Count",
                # titlefont=dict(color=AWS_COLORS["orange"]),
                tickfont=dict(color=AWS_COLORS["orange"]),
                anchor="x",
                overlaying="y",
                side="right",
                range=[0, 5]
            )
        )
        
        # Add instances annotation
        for i, count in enumerate(instances):
            if i > 0 and instances[i] != instances[i-1]:
                direction = "↑" if instances[i] > instances[i-1] else "↓"
                auto_scaling_fig.add_annotation(
                    x=i,
                    y=instances[i] * 30,
                    text=f"{direction} {count}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=AWS_COLORS["orange"],
                    ax=0,
                    ay=-30
                )
        
        st.plotly_chart(auto_scaling_fig, use_container_width=True)
        
        # When to use section
        st.subheader("When to Use Real-Time Inference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Ideal for:**
            
            - Interactive web and mobile applications
            - Real-time recommendation systems
            - Fraud detection systems
            - Interactive chatbots and AI assistants
            - Online gaming features
            - Real-time image and video analysis
            """)
        
        with col2:
            st.markdown("""
            **Less suitable for:**
            
            - Processing very large inputs (>6MB)
            - Long-running inference jobs (>60s)
            - Batch processing of accumulated data
            - Highly variable workloads with quiet periods
            - Cost-sensitive applications with unpredictable traffic
            """)
        
        # Example code
        st.subheader("Sample Code: Deploying a Real-Time Endpoint")
        
        st.code(generate_sample_code("realtime"), language="python")
        
        # Best practices
        st.subheader("Best Practices")
        
        st.markdown("""
        ### Optimize Real-Time Inference Performance
        
        - **Choose the right instance type** based on your model's needs (CPU, GPU, memory)
        - **Enable auto-scaling** to handle traffic variations efficiently
        - **Optimize your model** for inference (pruning, quantization, compilation)
        - **Use multi-model endpoints** for serving multiple models efficiently
        - **Set up monitoring** with CloudWatch for latency, errors, and invocations
        - **Use model cache** in your container to speed up frequent inferences
        - **Consider multi-AZ deployment** for high availability
        """)

    # ASYNCHRONOUS INFERENCE TAB
    with tab3:
        st.header("🕒 Asynchronous Inference")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Asynchronous Inference is designed for workloads that don't require immediate responses,
            involve large payloads, or have long processing times. Instead of waiting for results 
            synchronously, you receive a job ID and check for completion later.
            
            **Key features:**
            - **Process large payloads** up to 1GB
            - **Long-running inferences** without timeouts
            - **Queued processing** for efficient resource utilization
            - **Optional notifications** when processing completes
            """)
        
        with col2:
            st.image("images/async_inference.png", 
                     caption="SageMaker Asynchronous Inference", use_container_width=True)
        
        # Async inference diagram
        st.subheader("How Asynchronous Inference Works")
        
        async_fig = create_inference_flow_diagram("async")
        st.pyplot(async_fig)
        
        # Async inference flow animation
        st.subheader("Asynchronous Inference Process")
        
        # Create a step animation with Plotly
        steps = ["Submit Request", "Queue Request", "Process Request", "Store Result", "Retrieve Result"]
        step_descriptions = [
            "Client submits inference request with payload to the async endpoint",
            "Request is queued for processing when resources are available",
            "Model processes the request without timeout constraints",
            "Results are stored in the specified S3 bucket",
            "Client retrieves results from S3 or receives notification"
        ]
        
        # Create radio buttons for steps
        selected_step = st.radio("Select a step to see details:", steps, horizontal=True)
        step_index = steps.index(selected_step)
        
        # Display step diagram
        step_fig = go.Figure()
        
        # Common elements for all steps
        step_fig.add_trace(go.Scatter(
            x=[1, 3, 5, 7, 9],
            y=[2, 2, 2, 2, 2],
            mode="markers+text",
            marker=dict(
                size=30,
                color=[
                    AWS_COLORS["orange"] if i <= step_index else AWS_COLORS["gray"] 
                    for i in range(5)
                ]
            ),
            text=steps,
            textposition="bottom center"
        ))
        
        # Add connecting lines
        for i in range(4):
            step_fig.add_shape(
                type="line",
                x0=1 + i*2,
                y0=2,
                x1=3 + i*2,
                y1=2,
                line=dict(
                    color=AWS_COLORS["orange"] if i < step_index else AWS_COLORS["gray"],
                    width=3
                )
            )
        
        # Add current step description
        step_fig.add_annotation(
            x=5,
            y=3,
            xref="x",
            yref="y",
            text=step_descriptions[step_index],
            showarrow=False,
            bgcolor=AWS_COLORS["light_gray"],
            bordercolor=AWS_COLORS["teal"],
            borderwidth=2,
            borderpad=4,
            font=dict(size=14)
        )
        
        # Set layout
        step_fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            height=250,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 4])
        )
        
        st.plotly_chart(step_fig, use_container_width=True)
        
        # Processing time distribution
        st.subheader("Processing Time Distribution")
        
        async_latency_data = st.session_state.latency_data['async']
        # Convert to seconds for better display
        async_latency_data_seconds = [l / 1000 for l in async_latency_data]
        
        # Adjust chart to show seconds instead of ms
        async_latency_df = pd.DataFrame({"latency_seconds": async_latency_data_seconds})
        
        async_latency_chart = alt.Chart(async_latency_df).mark_bar().encode(
            alt.X('latency_seconds:Q', bin=alt.Bin(maxbins=20), title='Processing Time (seconds)'),
            alt.Y('count()', title='Count'),
            color=alt.value(AWS_COLORS["teal"])
        ).properties(
            title="Asynchronous Inference Processing Time Distribution",
            width='container',
            height=300
        )
        
        st.altair_chart(async_latency_chart, use_container_width=True)
        
        # Async performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Processing Time", f"{np.mean(async_latency_data_seconds):.1f} sec")
            st.metric("Maximum Payload Size", "1 GB")
        
        with col2:
            st.metric("P95 Processing Time", f"{np.percentile(async_latency_data_seconds, 95):.1f} sec")
            st.metric("Queue Capacity", "Up to thousands of requests")
        
        with col3:
            st.metric("Max Concurrent Invocations", "Limited by instance capacity")
            st.metric("Processing Timeout", "Hours/days")
        
        # Payload size comparison
        st.subheader("Payload Size Comparison")
        
        # Create a payload size comparison chart with Plotly
        payload_fig = go.Figure()
        
        # Define payload size limits
        payload_sizes = {
            "Real-Time": 6,
            "Asynchronous": 1024,
            "Batch Transform": 100,
            "Serverless": 4
        }
        
        # Add bars
        payload_fig.add_trace(go.Bar(
            x=list(payload_sizes.keys()),
            y=list(payload_sizes.values()),
            marker_color=[AWS_COLORS["orange"], AWS_COLORS["teal"], AWS_COLORS["green"], AWS_COLORS["blue"]],
            text=[f"{size} MB" for size in payload_sizes.values()],
            textposition="auto"
        ))
        
        # Customize layout
        payload_fig.update_layout(
            title="Maximum Payload Size Comparison (MB)",
            xaxis_title="Inference Type",
            yaxis_title="Size (MB)",
            yaxis_type="log",  # Use log scale due to large difference
            yaxis=dict(
                tickvals=[1, 10, 100, 1000],
                ticktext=["1 MB", "10 MB", "100 MB", "1 GB"]
            )
        )
        
        st.plotly_chart(payload_fig, use_container_width=True)
        
        # When to use section
        st.subheader("When to Use Asynchronous Inference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Ideal for:**
            
            - Processing large input payloads (images, videos)
            - Models with long inference times (>30 sec)
            - NLP tasks with large documents
            - Video processing applications
            - Complex ML pipelines with multiple steps
            - Genomic sequence analysis
            """)
        
        with col2:
            st.markdown("""
            **Less suitable for:**
            
            - Interactive applications requiring immediate responses
            - Simple models with fast inference times
            - High-frequency trading or real-time bidding systems
            - Applications where users wait for results
            - Processing large datasets in bulk (use Batch Transform)
            """)
        
        # Example code
        st.subheader("Sample Code: Deploying an Asynchronous Endpoint")
        
        st.code(generate_sample_code("async"), language="python")
        
        # Best practices
        st.subheader("Best Practices")
        
        st.markdown("""
        ### Optimize Asynchronous Inference
        
        - **Configure S3 buckets** for input and output storage
        - **Set up SNS notifications** to alert when processing completes
        - **Choose appropriate instance types** based on your model's computational requirements
        - **Configure maximum concurrent invocations** to control resource allocation
        - **Implement error handling** for failed inference jobs
        - **Consider compression** for very large payloads
        - **Set appropriate queue size** based on expected traffic
        """)

    # BATCH TRANSFORM TAB
    with tab4:
        st.header("📦 Batch Transform")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Batch Transform is designed for offline processing of large datasets. Instead of 
            responding to real-time requests, it processes an entire dataset at once, making 
            it ideal for periodic or scheduled inference tasks.
            
            **Key features:**
            - **Process entire datasets** stored in S3
            - **Distributed processing** across multiple instances
            - **Cost-effective** for large-scale inference
            - **Flexible data splitting** options for efficiency
            """)
        
        with col2:
            st.image("images/batch_transform.png", 
                     caption="SageMaker Batch Transform", use_container_width=True)
        
        # Batch transform diagram
        st.subheader("How Batch Transform Works")
        
        batch_fig = create_inference_flow_diagram("batch")
        st.pyplot(batch_fig)
        
        # Batch transform process animation
        st.subheader("Batch Transform Process")
        
        # Create batch processing visualization
        batch_process_fig = go.Figure()
        
        # Add S3 input bucket
        batch_process_fig.add_trace(go.Scatter(
            x=[1],
            y=[3],
            mode="markers+text",
            marker=dict(size=40, color=AWS_COLORS["teal"]),
            text=["S3 Input"],
            textposition="bottom center"
        ))
        
        # Add batch transform job
        batch_process_fig.add_trace(go.Scatter(
            x=[3],
            y=[3],
            mode="markers+text",
            marker=dict(size=50, color=AWS_COLORS["orange"]),
            text=["Batch Transform Job"],
            textposition="bottom center"
        ))
        
        # Add worker nodes
        for i in range(3):
            batch_process_fig.add_trace(go.Scatter(
                x=[5],
                y=[4 - i],
                mode="markers+text",
                marker=dict(size=30, color=AWS_COLORS["green"]),
                text=[f"Worker {i+1}"],
                textposition="bottom right"
            ))
        
        # Add S3 output bucket
        batch_process_fig.add_trace(go.Scatter(
            x=[7],
            y=[3],
            mode="markers+text",
            marker=dict(size=40, color=AWS_COLORS["teal"]),
            text=["S3 Output"],
            textposition="bottom center"
        ))
        
        # Add connecting arrows
        batch_process_fig.add_shape(
            type="line", x0=1.5, y0=3, x1=2.5, y1=3,
            line=dict(color="gray", width=2),
            xref="x", yref="y"
        )
        batch_process_fig.add_annotation(
            x=2, y=3.1, text="1. Read Data",
            showarrow=False, xref="x", yref="y"
        )
        
        # Add arrows from transform job to workers
        for i in range(3):
            batch_process_fig.add_shape(
                type="line", x0=3.5, y0=3, x1=4.5, y1=4-i,
                line=dict(color="gray", width=2),
                xref="x", yref="y"
            )
        batch_process_fig.add_annotation(
            x=4, y=3.8, text="2. Distribute Work",
            showarrow=False, xref="x", yref="y"
        )
        
        # Add arrows from workers to output
        for i in range(3):
            batch_process_fig.add_shape(
                type="line", x0=5.5, y0=4-i, x1=6.5, y1=3,
                line=dict(color="gray", width=2),
                xref="x", yref="y"
            )
        batch_process_fig.add_annotation(
            x=6, y=3.8, text="3. Write Results",
            showarrow=False, xref="x", yref="y"
        )
        
        # Set layout
        batch_process_fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 8]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[1, 5])
        )
        
        st.plotly_chart(batch_process_fig, use_container_width=True)
        
        # Batch transform strategies
        st.subheader("Data Splitting Strategies")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Line Strategy
            
            Splits input by newline character. Ideal for CSV or JSON Lines formats.
            
            ```json
            {"id": 1, "text": "example"}
            {"id": 2, "text": "another example"}
            ```
            
            Each line is sent as a separate request.
            """)
        
        with col2:
            st.markdown("""
            ### RecordIO Strategy
            
            Uses RecordIO-encoded data for efficient binary serialization.
            
            Ideal for:
            - Large datasets
            - Binary data
            - SageMaker built-in algorithms
            
            Optimized for high-throughput processing.
            """)
        
        with col3:
            st.markdown("""
            ### None Strategy
            
            Each file is processed as a single record.
            
            Ideal for:
            - Image files
            - Audio files
            - Individual documents
            
            Best when each file represents a complete input.
            """)
        
        # Batch performance metrics
        st.subheader("Performance Metrics")
        
        # Create a performance metrics comparison between instance counts
        instances = [1, 2, 4, 8, 16]
        processing_times = [100, 52, 28, 16, 10]  # Processing time in minutes
        throughput = [100, 192, 357, 625, 1000]  # Records per minute
        
        perf_fig = go.Figure()
        
        # Add processing time trace
        perf_fig.add_trace(go.Scatter(
            x=instances,
            y=processing_times,
            mode='lines+markers',
            name='Processing Time (min)',
            line=dict(color=AWS_COLORS["orange"], width=3)
        ))
        
        # Add throughput trace on secondary axis
        perf_fig.add_trace(go.Scatter(
            x=instances,
            y=throughput,
            mode='lines+markers',
            name='Throughput (records/min)',
            line=dict(color=AWS_COLORS["teal"], width=3),
            yaxis='y2'
        ))
        
        # Set up layout with secondary y-axis
        perf_fig.update_layout(
            title="Batch Transform Performance Scaling",
            xaxis_title="Number of Instances",
            yaxis=dict(
                title="Processing Time (min)",
                # titlefont=dict(color=AWS_COLORS["orange"]),
                tickfont=dict(color=AWS_COLORS["orange"])
            ),
            yaxis2=dict(
                title="Throughput (records/min)",
                # titlefont=dict(color=AWS_COLORS["teal"]),
                tickfont=dict(color=AWS_COLORS["teal"]),
                overlaying="y",
                side="right"
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )
        
        st.plotly_chart(perf_fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Records Per Instance", "10,000 records/hour/instance")
            st.metric("Maximum Record Size", "100 MB")
        
        with col2:
            batch_latency_data = st.session_state.latency_data['batch']
            batch_latency_data_minutes = [l / (1000 * 60) for l in batch_latency_data]
            st.metric("Average Job Duration", f"{np.mean(batch_latency_data_minutes):.1f} minutes")
            st.metric("Scaling Efficiency", "Near-linear up to 10+ instances")
        
        with col3:
            st.metric("Max Concurrent Transforms", "Limited by account quota")
            st.metric("Cost Saving vs. Real-Time", "Up to 90% for large datasets")
        
        # When to use section
        st.subheader("When to Use Batch Transform")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Ideal for:**
            
            - Processing accumulated data periodically
            - Overnight ETL workflows
            - Dataset labeling and enrichment
            - Generating pre-computed predictions
            - Processing archive data
            - Analyzing entire customer datasets
            """)
        
        with col2:
            st.markdown("""
            **Less suitable for:**
            
            - Real-time responses for user interactions
            - Single-record inference requests
            - Streaming data that requires immediate processing
            - Applications requiring interactive feedback
            - Simple models with fast inference
            """)
        
        # Example code
        st.subheader("Sample Code: Running a Batch Transform Job")
        
        st.code(generate_sample_code("batch"), language="python")
        
        # Best practices
        st.subheader("Best Practices")
        
        st.markdown("""
        ### Optimize Batch Transform Performance
        
        - **Choose the right instance count** to balance speed and cost
        - **Select appropriate batch size** to maximize throughput
        - **Use data splitting** to optimize resource utilization
        - **Schedule jobs during off-hours** to reduce costs
        - **Use Multi-Record batching** for small inputs
        - **Monitor job progress** with CloudWatch metrics
        - **Consider spot instances** for cost optimization on non-urgent jobs
        """)

    # SERVERLESS INFERENCE TAB
    with tab5:
        st.header("☁️ Serverless Inference")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Serverless Inference provides on-demand ML inference without having to configure or 
            manage the underlying infrastructure. It automatically scales from zero to handle 
            your traffic and charges only for what you use.
            
            **Key features:**
            - **Auto-scaling** to zero when not in use
            - **Pay-per-use** pricing model
            - **Low operational overhead** (no capacity planning)
            - **Built-in high availability** across multiple AZs
            """)
        
        with col2:
            st.image("images/serverless_inference.png", 
                     caption="Serverless Architecture Pattern", use_container_width=True)
        
        # Serverless inference diagram
        st.subheader("How Serverless Inference Works")
        
        serverless_fig = create_inference_flow_diagram("serverless")
        st.pyplot(serverless_fig)
        
        # Serverless auto-scaling visualization
        st.subheader("Serverless Scaling Behavior")
        
        # Create serverless scaling visualization with Plotly
        serverless_scaling_fig = go.Figure()
        
        # Generate sample traffic and capacity data
        hours = list(range(24))
        traffic = [5, 3, 1, 0, 0, 2, 10, 30, 60, 65, 55, 40, 45, 50, 45, 40, 50, 70, 60, 30, 20, 15, 10, 5]
        
        # Calculate capacity that follows traffic with a slight delay
        capacity = [0]
        for i in range(1, len(traffic)):
            if traffic[i] > capacity[i-1]:
                # Scale up quickly
                capacity.append(min(traffic[i], capacity[i-1] + max(10, traffic[i] - capacity[i-1])))
            else:
                # Scale down more slowly
                capacity.append(max(0, capacity[i-1] - 5))
        
        # Add traffic line
        serverless_scaling_fig.add_trace(go.Scatter(
            x=hours,
            y=traffic,
            mode='lines',
            name='Requests per minute',
            line=dict(color=AWS_COLORS["blue"], width=3)
        ))
        
        # Add capacity line
        serverless_scaling_fig.add_trace(go.Scatter(
            x=hours,
            y=capacity,
            mode='lines',
            name='Provisioned capacity',
            line=dict(color=AWS_COLORS["green"], width=3)
        ))
        
        # Add zero line with shading for "scale to zero" concept
        zero_periods = []
        in_zero = False
        start_zero = None
        
        for i, cap in enumerate(capacity):
            if cap == 0 and not in_zero:
                in_zero = True
                start_zero = i
            elif cap > 0 and in_zero:
                in_zero = False
                zero_periods.append((start_zero, i))
                
        if in_zero:
            zero_periods.append((start_zero, len(capacity) - 1))
        
        # Shade zero capacity regions
        for start, end in zero_periods:
            serverless_scaling_fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                y0=0,
                y1=max(traffic) * 0.1,
                fillcolor="rgba(200, 200, 200, 0.3)",
                line=dict(width=0),
                layer="below"
            )
            serverless_scaling_fig.add_annotation(
                x=(start + end) / 2,
                y=max(traffic) * 0.05,
                text="Scaled to Zero - No Charges",
                showarrow=False,
                font=dict(size=10, color=AWS_COLORS["dark_gray"])
            )
        
        # Update layout
        serverless_scaling_fig.update_layout(
            title="Serverless Inference Auto-scaling (Scale to Zero)",
            xaxis_title="Hour of Day",
            yaxis_title="Traffic and Capacity",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(serverless_scaling_fig, use_container_width=True)
        
        # Serverless cost model visualization
        st.subheader("Serverless Cost Model vs. Traditional Endpoints")
        
        # Create interactive slider for monthly request volume
        monthly_requests = st.slider("Monthly Request Volume", 
                            min_value=1000, 
                            max_value=10000000, 
                            value=1000000, 
                            step=1000,
                            format="%d")
        
        # Create cost comparison based on request volume
        # Cost model: These are example pricing, not actual AWS pricing
        request_sizes = ["1MB", "2MB", "4MB"]
        request_size = st.selectbox("Average Request Size", request_sizes, index=0)
        
        # Define cost parameters based on request size
        if request_size == "1MB":
            serverless_cost_per_request = 0.00000125  # $ per request
            serverless_cost_per_ms = 0.0000000088  # $ per GB-second, assuming 100ms average
            realtime_hourly_cost = 0.52  # $ per hour for ml.c5.large
        elif request_size == "2MB":
            serverless_cost_per_request = 0.0000025  # $ per request
            serverless_cost_per_ms = 0.0000000176  # $ per GB-second, assuming 100ms average
            realtime_hourly_cost = 0.52  # $ per hour for ml.c5.large
        else:  # 4MB
            serverless_cost_per_request = 0.000005  # $ per request
            serverless_cost_per_ms = 0.0000000352  # $ per GB-second, assuming 100ms average
            realtime_hourly_cost = 1.04  # $ per hour for ml.c5.xlarge
        
        # Calculate average duration
        avg_duration = st.slider("Average Inference Duration (ms)", 
                              min_value=10, 
                              max_value=1000, 
                              value=100, 
                              step=10)
        
        # Calculate monthly costs
        serverless_compute_cost = monthly_requests * serverless_cost_per_ms * avg_duration
        serverless_request_cost = monthly_requests * serverless_cost_per_request
        total_serverless_cost = serverless_compute_cost + serverless_request_cost
        
        # Calculate real-time endpoint cost based on instance hours
        # For capacity to handle peak load, assume peak is 5x average and instance can handle 10 req/sec
        peak_rps = (monthly_requests / (30 * 24 * 60 * 60)) * 5  # 5x average load
        min_instances = max(1, int(peak_rps / 10))  # Each instance handles ~10 req/sec
        realtime_monthly_cost = min_instances * realtime_hourly_cost * 24 * 30
        
        # Create cost comparison visualization
        cost_comparison_fig = go.Figure()
        
        cost_comparison_fig.add_trace(go.Bar(
            x=["Serverless Inference", "Real-Time Endpoint"],
            y=[total_serverless_cost, realtime_monthly_cost],
            marker_color=[AWS_COLORS["orange"], AWS_COLORS["blue"]],
            text=[f"${total_serverless_cost:.2f}", f"${realtime_monthly_cost:.2f}"],
            textposition="auto"
        ))
        
        cost_comparison_fig.update_layout(
            title=f"Monthly Cost Comparison ({monthly_requests:,} requests/month)",
            yaxis_title="Monthly Cost ($)",
            height=400
        )
        
        st.plotly_chart(cost_comparison_fig, use_container_width=True)
        
        # Cost breakdown
        st.subheader("Serverless Cost Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Request Cost:**
            - {monthly_requests:,} requests
            - ${serverless_request_cost:.2f} total request cost
            
            **Compute Cost:**
            - {avg_duration} ms average duration
            - ${serverless_compute_cost:.2f} total compute cost
            """)
        
        with col2:
            # Create a pie chart for cost breakdown
            cost_pie = go.Figure(data=[go.Pie(
                labels=['Request Cost', 'Compute Cost'],
                values=[serverless_request_cost, serverless_compute_cost],
                hole=.4,
                marker_colors=[AWS_COLORS["teal"], AWS_COLORS["orange"]]
            )])
            
            cost_pie.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(cost_pie, use_container_width=True)
        
        # Serverless performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            serverless_latency_data = st.session_state.latency_data['serverless']
            st.metric("Average Latency", f"{np.mean(serverless_latency_data):.1f} ms")
            st.metric("Cold Start Time", "300-500 ms")
        
        with col2:
            st.metric("P95 Latency", f"{np.percentile(serverless_latency_data, 95):.1f} ms")
            st.metric("Maximum Concurrency", "Up to account limit")
        
        with col3:
            st.metric("Maximum Memory Size", "6144 MB")
            st.metric("Request Timeout", "60 seconds")
        
        # Serverless memory configuration
        st.subheader("Memory Configuration Options")
        
        # Memory configuration table
        memory_config = {
            "Memory Size (MB)": [1024, 2048, 3072, 4096, 5120, 6144],
            "vCPU": ["0.25", "0.5", "1", "1", "1", "1"],
            "Relative Performance": ["1x", "2x", "4x", "4x", "4x", "4x"],
            "Relative Cost": ["1x", "2x", "3x", "4x", "5x", "6x"],
            "Best For": [
                "Simple ML tasks, low compute", 
                "Basic inferencing, most common choice", 
                "Medium complexity models",
                "Complex models or NLP transformers",
                "Large computer vision models",
                "Very large models with high memory needs"
            ]
        }
        
        memory_df = pd.DataFrame(memory_config)
        st.dataframe(memory_df, hide_index=True, use_container_width=True)
        
        # When to use section
        st.subheader("When to Use Serverless Inference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Ideal for:**
            
            - Applications with infrequent or variable traffic
            - Cost-sensitive deployments
            - Dev/test environments
            - Serverless application integrations
            - Applications with unpredictable load patterns
            - Simple APIs with moderate latency requirements
            """)
        
        with col2:
            st.markdown("""
            **Less suitable for:**
            
            - Very large models (>6GB memory requirements)
            - Applications with consistent, high-volume traffic
            - Ultra-low latency requirements (<20ms)
            - Processing large inputs (>4MB)
            - Complex inference pipelines
            - Models requiring specialized hardware (GPUs)
            """)
        
        # Example code
        st.subheader("Sample Code: Deploying a Serverless Endpoint")
        
        st.code(generate_sample_code("serverless"), language="python")
        
        # Best practices
        st.subheader("Best Practices")
        
        st.markdown("""
        ### Optimize Serverless Inference
        
        - **Choose appropriate memory** configuration based on model requirements
        - **Optimize your model size** to reduce cold starts (quantization, pruning)
        - **Keep container image size small** for faster startup
        - **Test cold start performance** for your specific use case
        - **Use concurrency settings** to balance cost and availability
        - **Monitor invocation metrics** to detect patterns and optimize
        - **Consider pre-warming** strategies for predictable traffic spikes
        """)

    # Add footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    main()
