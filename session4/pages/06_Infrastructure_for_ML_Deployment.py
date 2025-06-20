
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import random
from datetime import datetime, timedelta
import base64
from PIL import Image
import io
import requests
import boto3
import uuid
from typing import Dict, List, Any, Union

# Initialize session state
def init_session_state():
    if 'initialized_infra' not in st.session_state:
        st.session_state.quiz_scores = {
            "autoscaling": 0,
            "container": 0,
            "iac": 0
        }
        st.session_state.quiz_submitted = {
            "autoscaling": False,
            "container": False,
            "iac": False
        }
        st.session_state.initialized_infra = True
        st.session_state.user_id = str(uuid.uuid4())

def load_css():
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
        .model-version-card {
            border: 1px solid #E9ECEF;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            background-color: #FFFFFF;
            transition: transform 0.2s;
        }
        .model-version-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .approved {
            border-left: 5px solid #59BA47;
        }
        .rejected {
            border-left: 5px solid #D13212;
        }
        .pending {
            border-left: 5px solid #FF9900;
        }
        .metrics-table th {
            font-weight: normal;
            color: #545B64;
        }
        .metrics-table td {
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

def render_footer():
    """Render the footer."""
    st.markdown(
        """
        <div class="footer">
            © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

def reset_session():
    """Reset the session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.initialized = False
    st.rerun()

def show_home_page():
    """Display the home page content."""
    
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to AWS ML Deployment Infrastructure</h3>
    <p>This interactive course teaches you how to deploy machine learning models at scale on AWS. 
    Explore different infrastructure options designed to help you optimize performance, manage costs, 
    and create reliable ML workloads.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### What You'll Learn")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>⚖️ SageMaker Autoscaling</h3>
        <p>Learn how to automatically adjust endpoint capacity to handle varying inference loads.</p>
        <ul>
            <li>Target Tracking Scaling</li>
            <li>Step Scaling</li>
            <li>Scheduled Scaling</li>
            <li>On-Demand Scaling</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>🐳 Container Services</h3>
        <p>Deploy ML models using AWS container services.</p>
        <ul>
            <li>Orchestration (ECS, EKS)</li>
            <li>Compute Platform (EC2, Fargate)</li>
            <li>Image Registry (ECR)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
        <h3>🏗️ Infrastructure as Code</h3>
        <p>Deploy ML infrastructure using code-based templates.</p>
        <ul>
            <li>AWS CloudFormation</li>
            <li>AWS Cloud Development Kit (CDK)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### How to Use This Course")
    
    st.markdown("""
    <div class="card">
    <p>Navigate through the tabs above to explore each topic in depth. Each section includes:</p>
    <ul>
        <li><strong>Conceptual Overviews</strong> - Learn the key concepts</li>
        <li><strong>Interactive Examples</strong> - See how the technology works</li>
        <li><strong>Implementation Code</strong> - Ready-to-use code examples</li>
    </ul>
    <p>Complete your learning with the Knowledge Check tab to test your understanding.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("https://d1.awsstatic.com/sagemaker/sagemaker-technical-overview.2dc4c5bf8883e2dc408ab8511661bf0ecc8a1cbe.png", 
             caption="AWS ML Infrastructure Overview")

def generate_sagemaker_endpoint_metrics(days=7):
    """Generate synthetic metrics for SageMaker endpoint visualization."""
    np.random.seed(42)
    date_range = pd.date_range(end=datetime.now(), periods=days*24, freq='h')
    
    # Generate base load with daily patterns
    base_load = np.sin(np.linspace(0, days*2*np.pi, days*24)) * 10 + 20
    
    # Add weekday/weekend pattern
    weekday_boost = np.array([3 if d.weekday() < 5 else -2 for d in date_range])
    
    # Add random noise
    noise = np.random.normal(0, 2, days*24)
    
    # Create the final invocations data
    invocations = base_load + weekday_boost + noise
    invocations = np.maximum(invocations, 1)  # Ensure no negative values
    
    # Generate latency based on invocations (higher load = slightly higher latency)
    base_latency = 100  # base latency in ms
    latency = base_latency + invocations * 0.5 + np.random.normal(0, 10, days*24)
    
    # Calculate instance count based on autoscaling rules
    instance_count = np.ceil(invocations / 20)
    instance_count = np.maximum(1, instance_count)  # Minimum 1 instance
    
    # Add delay to instance scaling to simulate real-world behavior
    for i in range(len(instance_count)-1):
        if instance_count[i+1] > instance_count[i]:
            # Scale up quickly
            instance_count[i+1] = min(instance_count[i+1], instance_count[i] + 2)
        elif instance_count[i+1] < instance_count[i]:
            # Scale down slowly
            instance_count[i+1] = max(instance_count[i+1], instance_count[i] - 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'invocations': invocations,
        'latency_ms': latency,
        'instance_count': instance_count
    })
    
    return df

def show_autoscaling_simulation(scaling_type):
    """Simulate and visualize different autoscaling behaviors."""
    # Generate synthetic data
    data = generate_sagemaker_endpoint_metrics(days=7)
    
    # Filter to last 24 hours for better visualization
    data_24h = data.iloc[-24:].copy()
    
    # Modify instance count based on scaling type
    if scaling_type == "Target Tracking":
        # Simulate target tracking based on invocation count
        data_24h['instance_count'] = np.ceil(data_24h['invocations'] / 15)
        title = "Target Tracking Scaling (Target: 15 invocations/instance)"
        
    elif scaling_type == "Step Scaling":
        # Simulate step scaling with predefined thresholds
        steps = [(10, 1), (20, 2), (30, 3), (50, 4), (80, 5)]
        
        def get_instance_count(invocations):
            for threshold, count in steps:
                if invocations <= threshold:
                    return count
            return 6  # Maximum instances
        
        data_24h['instance_count'] = data_24h['invocations'].apply(get_instance_count)
        title = "Step Scaling (Predefined thresholds)"
        
    elif scaling_type == "Scheduled Scaling":
        # Simulate scheduled scaling with higher capacity during business hours
        def get_scheduled_instances(timestamp):
            hour = timestamp.hour
            # More instances during business hours (8 AM - 6 PM)
            if 8 <= hour <= 18:
                return 3
            # Fewer instances during non-business hours
            return 1
        
        data_24h['instance_count'] = data_24h['timestamp'].apply(get_scheduled_instances)
        title = "Scheduled Scaling (Business hours: 8 AM - 6 PM)"
        
    elif scaling_type == "On-Demand Scaling":
        # Simulate on-demand scaling with immediate response to traffic
        # But add some randomness to show the volatility
        data_24h['instance_count'] = np.maximum(1, np.ceil(data_24h['invocations'] / 10) + np.random.randint(-1, 2, size=len(data_24h)))
        title = "On-Demand Scaling (Immediate response to traffic)"
    
    # Plot the results
    fig = go.Figure()
    
    # Add invocations line
    fig.add_trace(go.Scatter(
        x=data_24h['timestamp'],
        y=data_24h['invocations'],
        mode='lines',
        name='Invocations/min',
        line=dict(color='#FF9900', width=2)
    ))
    
    # Add instance count line with different y-axis
    fig.add_trace(go.Scatter(
        x=data_24h['timestamp'],
        y=data_24h['instance_count'],
        mode='lines',
        name='Instance Count',
        line=dict(color='#1E88E5', width=3),
        yaxis='y2'
    ))
    
    # Layout with two y-axes
    fig.update_layout(
        title=title,
        xaxis=dict(title='Time'),
        yaxis=dict(
            title='Invocations per minute',
            # titlefont=dict(color='#FF9900'),
            tickfont=dict(color='#FF9900')
        ),
        yaxis2=dict(
            title='Instance Count',
            # titlefont=dict(color='#1E88E5'),
            tickfont=dict(color='#1E88E5'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(l=40, r=40, t=60, b=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show cost analysis
    st.markdown("### 💰 Cost Analysis")
    instance_hours = data_24h['instance_count'].sum()
    hourly_rate = 0.348  # ml.m5.large hourly cost in dollars
    total_cost = instance_hours * hourly_rate
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Instance Hours", f"{instance_hours:.1f}")
    with col2:
        st.metric("Average Instances", f"{data_24h['instance_count'].mean():.2f}")
    with col3:
        st.metric("Estimated Cost (USD)", f"${total_cost:.2f}")
    
    # Summary
    if scaling_type == "Target Tracking":
        st.info("📌 **Target Tracking Scaling**: Adjusts capacity based on a target metric value. Great for workloads with predictable scaling patterns.")
    elif scaling_type == "Step Scaling":
        st.info("📌 **Step Scaling**: Uses CloudWatch alarms to trigger scaling policies with predefined steps. Good for fine-grained control.")
    elif scaling_type == "Scheduled Scaling":
        st.info("📌 **Scheduled Scaling**: Adjusts capacity based on predictable schedule. Ideal for workloads with known time-based patterns.")
    else:
        st.info("📌 **On-Demand Scaling**: Rapidly responds to traffic changes. Best for unpredictable workloads but may lead to higher costs.")

def show_sagemaker_autoscaling():
    """Display the SageMaker autoscaling content."""
    st.markdown("## ⚖️ Amazon SageMaker Endpoint Autoscaling")
    
    st.markdown("""
    <div class="info-box">
    <h3>Why Autoscale SageMaker Endpoints?</h3>
    <p>SageMaker endpoints need to handle varying inference loads while maintaining cost efficiency. 
    Autoscaling dynamically adjusts the number of instances backing your endpoint based on traffic patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SageMaker Autoscaling Architecture
    st.markdown("### SageMaker Autoscaling Architecture")
    
    st.image("https://d1.awsstatic.com/re19/Diagram_SageMaker-Training-and-Deployment_How-it-Works.c7f95757a12c97eff12edef1a5dc25b82abc186e.png", 
             caption="SageMaker Endpoint Architecture with Autoscaling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Key Components:
        
        - **Endpoint**: The HTTPS endpoint that applications use for inference
        - **Production Variants**: Different model versions deployed behind the endpoint
        - **Auto Scaling Group**: Manages the fleet of instances for your endpoint
        - **CloudWatch**: Monitors endpoint metrics and triggers scaling actions
        - **Scaling Policy**: Defines when and how to scale the endpoint
        """)
    
    with col2:
        st.markdown("""
        #### Available Metrics for Scaling:
        
        - **InvocationsPerInstance**: Number of requests per instance
        - **CPUUtilization**: Percentage of CPU utilization
        - **MemoryUtilization**: Percentage of memory utilization
        - **GPUUtilization**: Percentage of GPU utilization (for GPU instances)
        - **GPUMemoryUtilization**: Percentage of GPU memory utilization
        """)
    
    # Autoscaling Options and Simulation
    st.markdown("### 📊 Autoscaling Options")
    
    scaling_option = st.radio(
        "Select a scaling option to visualize:",
        ["Target Tracking", "Step Scaling", "Scheduled Scaling", "On-Demand Scaling"],
        horizontal=True
    )
    
    # Show simulation for selected scaling option
    show_autoscaling_simulation(scaling_option)
    
    # Comparison of scaling options
    st.markdown("### Comparing Autoscaling Options")
    
    comparison_data = {
        'Feature': [
            'Best Use Case', 
            'Scaling Trigger', 
            'Configuration Complexity', 
            'Cost Efficiency',
            'Response Time',
            'Proactive vs Reactive'
        ],
        'Target Tracking': [
            'Predictable workloads with consistent patterns', 
            'Metric threshold', 
            'Low', 
            'High',
            'Medium',
            'Reactive'
        ],
        'Step Scaling': [
            'Workloads requiring fine-grained control', 
            'CloudWatch alarms', 
            'Medium', 
            'Medium',
            'Medium',
            'Reactive'
        ],
        'Scheduled Scaling': [
            'Workloads with known time-based patterns', 
            'Time schedule', 
            'Low', 
            'Very high',
            'Immediate',
            'Proactive'
        ],
        'On-Demand Scaling': [
            'Unpredictable, highly variable workloads', 
            'Real-time metrics', 
            'Low', 
            'Low',
            'Fast',
            'Reactive'
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    # Implementation code
    st.markdown("### 💻 Implementation Code")
    
    code_tabs = st.tabs(["Python SDK", "AWS CLI", "AWS Console"])
    
    with code_tabs[0]:
        st.code("""
        import boto3
        
        client = boto3.client('application-autoscaling')
        
        # Register a scalable target
        client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId='endpoint/my-endpoint/variant/my-variant',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=1,
            MaxCapacity=4
        )
        
        # Configure target tracking scaling policy
        client.put_scaling_policy(
            PolicyName='MyScalingPolicy',
            ServiceNamespace='sagemaker',
            ResourceId='endpoint/my-endpoint/variant/my-variant',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': 15.0,  # Target 15 invocations per instance
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleOutCooldown': 60,  # Wait 60 seconds before scaling out again
                'ScaleInCooldown': 300    # Wait 300 seconds before scaling in again
            }
        )
        
        """, language="python")
    
    with code_tabs[1]:
        st.code("""
        # Register a scalable target
        aws application-autoscaling register-scalable-target \\
            --service-namespace sagemaker \\
            --resource-id endpoint/my-endpoint/variant/my-variant \\
            --scalable-dimension sagemaker:variant:DesiredInstanceCount \\
            --min-capacity 1 \\
            --max-capacity 4
            
        # Configure target tracking scaling policy
        aws application-autoscaling put-scaling-policy \\
            --policy-name my-scaling-policy \\
            --service-namespace sagemaker \\
            --resource-id endpoint/my-endpoint/variant/my-variant \\
            --scalable-dimension sagemaker:variant:DesiredInstanceCount \\
            --policy-type TargetTrackingScaling \\
            --target-tracking-scaling-policy-configuration file://config.json
            
        # Contents of config.json:
        # {
        #   "TargetValue": 15.0,
        #   "PredefinedMetricSpecification": {
        #     "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        #   },
        #   "ScaleOutCooldown": 60,
        #   "ScaleInCooldown": 300
        # }
        """, language="bash")
    
    with code_tabs[2]:
        st.image("https://d1.awsstatic.com/sagemaker/autoscaling-config.png", 
                 caption="SageMaker Console - Autoscaling Configuration")
    
    # Best practices
    st.markdown("""
    <div class="card">
    <h3>📌 Best Practices for SageMaker Autoscaling</h3>
    <ul>
        <li><strong>Set appropriate cooldown periods</strong> - Prevent rapid scaling oscillations</li>
        <li><strong>Choose the right metrics</strong> - Base scaling on the resource that limits your model performance</li>
        <li><strong>Test scaling behavior</strong> - Validate scaling policies with load testing before production</li>
        <li><strong>Consider initial scaling settings</strong> - Start conservative and adjust based on observations</li>
        <li><strong>Monitor costs</strong> - Autoscaling affects your billing; set up AWS Cost Explorer alerts</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_container_deployment_architecture(service_type):
    """Display the architecture diagram for different container deployment services."""
    if service_type == "ECS":
        st.markdown("""
        <div class="card">
        <h3>ECS Architecture</h3>
        <ul>
            <li>Tasks run on EC2 instances or Fargate</li>
            <li>Application Load Balancer routes traffic</li>
            <li>ECS Service maintains desired task count</li>
            <li>Cluster manages resource allocation</li>
            <li>Task Definitions specify container configurations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://d1.awsstatic.com/diagrams/product-page-diagrams/product-page-diagram_ECS_1.86ebd8c223ec8b55aa1903c423fbe4e672f3daf7.png",
                caption="Amazon ECS Architecture")
        
    elif service_type == "EKS":
        st.markdown("""
        <div class="card">
        <h3>EKS Architecture</h3>
        <ul>
            <li>Managed Kubernetes control plane</li>
            <li>Worker nodes run as EC2 instances or Fargate</li>
            <li>Deployments manage pod replica sets</li>
            <li>Services expose pods to traffic</li>
            <li>Ingress controllers route external traffic</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://d1.awsstatic.com/product-page-diagram_Amazon-EKS%402x.0d872eb6fb782ddc733a27d2bb9db795fed71185.png",
                caption="Amazon EKS Architecture")
        
    elif service_type == "ECR":
        st.markdown("""
        <div class="card">
        <h3>ECR Architecture</h3>
        <ul>
            <li>Managed container registry service</li>
            <li>Stores, manages, and deploys Docker container images</li>
            <li>Integrated with ECS and EKS</li>
            <li>Supports private repositories</li>
            <li>Integrates with IAM for access control</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://d1.awsstatic.com/diagrams/product-page-diagrams/Product-Page-Diagram_Amazon-ECR.bf2e7a03447ed3aba97a70e5f4aead46a5e04547.png",
                caption="Amazon ECR Architecture")
        
    elif service_type == "Compute":
        st.markdown("""
        <div class="card">
        <h3>EC2 vs Fargate</h3>
        <ul>
            <li><strong>EC2</strong>: You manage EC2 instances in your cluster</li>
            <li><strong>Fargate</strong>: Serverless compute for containers</li>
            <li><strong>EC2</strong>: More control over infrastructure</li>
            <li><strong>Fargate</strong>: No instance management needed</li>
            <li><strong>EC2</strong>: Better for cost optimization at scale</li>
            <li><strong>Fargate</strong>: Better for variable workloads</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://d1.awsstatic.com/legal/AmazonWebServicesLogo/AWS_logo_RGB.jpg",
                    caption="Amazon EC2", width=300)
            
        with col2:
            st.image("https://d1.awsstatic.com/products/fargate/product-page-diagram_Fargate%402x.a20d701dcd1cf898e19c274dfd250cdb5e094b14.png",
                    caption="AWS Fargate", width=300)

def show_service_comparison_matrix():
    """Display a comparison matrix of container services."""
    comparison_data = {
        'Feature': [
            'Container Orchestration',
            'Kubernetes Support', 
            'Compute Options', 
            'Scaling Capabilities',
            'ML Model Size',
            'Management Overhead',
            'Cost Efficiency',
            'Integration with AWS ML'
        ],
        'ECS': [
            'AWS Managed', 
            'No', 
            'EC2 & Fargate', 
            'Service Auto Scaling',
            'Any size',
            'Low-Medium',
            'Medium',
            'Good'
        ],
        'EKS': [
            'Kubernetes', 
            'Yes', 
            'EC2 & Fargate', 
            'Horizontal Pod Autoscaler',
            'Any size',
            'Medium-High',
            'Low',
            'Good'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Apply color coding based on values
    def color_code(val):
        if val in ['Yes', 'Any size', 'Very Good']:
            return 'background-color: #D6F5DB'  # Light green
        elif val in ['Good', 'EC2 & Fargate']:
            return 'background-color: #FFF2CC'  # Light yellow
        elif val in ['No', 'Medium-High']:
            return 'background-color: #FADBD8'  # Light red
        return ''
    
    styled_df = comparison_df.style.map(color_code, subset=pd.IndexSlice[:, ['ECS', 'EKS']])
    
    st.dataframe(styled_df, use_container_width=True)

def show_aws_container_services():
    """Display the AWS Container Services content."""
    st.markdown("## 🐳 AWS Container Services for ML Model Deployment")
    
    st.markdown("""
    <div class="info-box">
    <h3>Why Use Containers for ML Model Deployment?</h3>
    <p>Containers provide a consistent environment for your ML models, ensuring they run the same way in development and production. 
    With AWS container services, you can deploy models at scale with advanced orchestration capabilities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview of container components
    st.markdown("### AWS Container Service Components")
    
    st.image("https://d1.awsstatic.com/diagrams/product-page-diagrams/product-page-diagram_ECS-EC2-DevOps_2x.0e5251c0c32a936fb19c9842bb030ce46492bd38.png",
             caption="AWS Container Services Overview")
    
    # Service selection
    st.markdown("### Container Service Options")
    
    service_tabs = st.tabs([
        "🚢 Orchestration (ECS/EKS)", 
        "💻 Compute Platform (EC2/Fargate)", 
        "📦 Image Registry (ECR)"
    ])
    
    with service_tabs[0]:
        st.markdown("### Container Orchestration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Amazon ECS
            
            Amazon Elastic Container Service (ECS) is a fully managed container orchestration service.
            
            #### Key Features for ML Workloads:
            - Support for GPU and Inferentia instances
            - Integration with AutoScaling
            - Easy deployment of large ML models
            - Integration with AWS services like SageMaker
            """)
        
        with col2:
            st.markdown("""
            ### Amazon EKS
            
            Amazon Elastic Kubernetes Service (EKS) is a managed Kubernetes service.
            
            #### Key Features for ML Workloads:
            - Native Kubernetes capabilities for advanced orchestration
            - Support for custom ML operators and frameworks
            - Integration with Kubernetes autoscaling
            - Support for GPU, Inferentia, and multi-node training
            """)
        
        orchestration_option = st.radio(
            "Select an orchestration service to view:",
            ["ECS", "EKS"],
            horizontal=True
        )
        
        show_container_deployment_architecture(orchestration_option)
        
        if orchestration_option == "ECS":
            st.markdown("#### Example ECS Task Definition for ML Inference")
            
            st.code("""
{
  "family": "ml-inference-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "inference-container",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-model:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "hostPort": 8080,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ml-inference",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/opt/ml/models/model.joblib"
        },
        {
          "name": "WORKERS",
          "value": "2"
        }
      ]
    }
  ]
}
            """, language="json")
        else:
            st.markdown("#### Example Kubernetes Deployment for ML Inference")
            
            st.code("""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      containers:
      - name: model-server
        image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/my-model:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
            nvidia.com/gpu: 1
          requests:
            cpu: "1"
            memory: "2Gi"
        env:
        - name: MODEL_PATH
          value: "/opt/ml/models/model.joblib"
        - name: WORKERS
          value: "2"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-inference-service
spec:
  selector:
    app: ml-inference
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
            """, language="yaml")
    
    with service_tabs[1]:
        st.markdown("### Compute Platform Options")
        st.markdown("""
        When deploying containers for ML workloads, you have two primary compute platform options:
        
        1. **Amazon EC2** - Traditional virtual machines that you manage
        2. **AWS Fargate** - Serverless compute platform for containers
        """)
        
        # Show comparison of EC2 and Fargate
        show_container_deployment_architecture("Compute")
        
        # Compute comparison table
        compute_comparison = {
            'Feature': [
                'Instance Control', 
                'Management Overhead', 
                'Scaling Control',
                'GPU Support',
                'Cost Model',
                'Best For'
            ],
            'EC2': [
                'Full control', 
                'High', 
                'Fine-grained',
                'Yes',
                'Pay for provisioned resources',
                'Large-scale ML deployments, GPU workloads'
            ],
            'Fargate': [
                'Limited control', 
                'Low', 
                'Automatic',
                'No',
                'Pay per task',
                'Variable workloads, simpler deployment needs'
            ]
        }
        
        st.dataframe(pd.DataFrame(compute_comparison), use_container_width=True)
        
        st.markdown("#### Example: Configuring GPU for ML Workloads on EC2")
        st.code("""
# ECS Task Definition with GPU requirements
{
  "family": "ml-inference-gpu",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "placementConstraints": [{
    "type": "memberOf",
    "expression": "attribute:instance-type =~ g4dn.*"
  }],
  "containerDefinitions": [
    {
      "name": "ml-gpu-container",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-model:gpu",
      "essential": true,
      "resourceRequirements": [{
        "type": "GPU",
        "value": "1"
      }],
      "portMappings": [
        {
          "containerPort": 8080,
          "hostPort": 8080
        }
      ],
      "environment": [
        {
          "name": "NVIDIA_VISIBLE_DEVICES",
          "value": "all"
        },
        {
          "name": "MODEL_PATH",
          "value": "/opt/ml/model"
        }
      ]
    }
  ]
}
        """, language="json")
        
    with service_tabs[2]:
        st.markdown("### Amazon Elastic Container Registry (ECR)")
        st.markdown("""
        Amazon ECR is a fully managed Docker container registry that makes it easy to store, manage, and deploy container images for ML workloads.
        
        #### Key Features:
        - Private repositories with fine-grained access control
        - Highly available and scalable
        - Integration with ECS, EKS, and SageMaker
        - Vulnerability scanning
        - Lifecycle policies to manage images
        """)
        
        show_container_deployment_architecture("ECR")
        
        st.markdown("#### Common ECR Commands for ML Workflows")
        
        st.code("""
# Authenticate Docker to your ECR registry
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Create a repository
aws ecr create-repository --repository-name ml-models/inference --image-scanning-configuration scanOnPush=true

# Build and tag your ML model container
docker build -t ml-models/inference:v1.0 .
docker tag ml-models/inference:v1.0 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-models/inference:v1.0

# Push to ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-models/inference:v1.0

# Pull from ECR (in your ECS/EKS deployment)
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-models/inference:v1.0

# Create lifecycle policy (keep only the latest 5 images)
aws ecr put-lifecycle-policy \
  --repository-name ml-models/inference \
  --lifecycle-policy-text '{
    "rules": [
      {
        "rulePriority": 1,
        "description": "Keep only the latest 5 images",
        "selection": {
          "tagStatus": "any",
          "countType": "imageCountMoreThan",
          "countNumber": 5
        },
        "action": {
          "type": "expire"
        }
      }
    ]
  }'
        """, language="bash")
        
        st.markdown("#### Dockerfile Example for ML Model Container")
        
        st.code("""
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /opt/ml

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install ML libraries
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy model files and server code
COPY model/ /opt/ml/model/
COPY code/ /opt/ml/code/

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV MODEL_PATH=/opt/ml/model
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the server
CMD ["python3", "/opt/ml/code/server.py"]
        """, language="dockerfile")
    
    # Service comparison chart
    st.markdown("### 📊 Container Service Comparison")
    show_service_comparison_matrix()
    
    # ML model size vs. service recommendation chart
    st.markdown("### 📈 ML Model Size vs. Container Environment")
    
    # Generate sample data for the chart
    sizes = np.array([10, 100, 500, 1000, 2000, 5000, 10000])  # MB
    ecs_ec2_scores = [5, 6, 7, 8, 9, 10, 10]
    ecs_fargate_scores = [7, 8, 9, 8, 6, 4, 3]
    eks_ec2_scores = [4, 5, 6, 7, 9, 10, 10]
    eks_fargate_scores = [6, 7, 8, 7, 5, 3, 2]
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each service
    fig.add_trace(go.Scatter(
        x=sizes, 
        y=ecs_ec2_scores, 
        mode='lines+markers', 
        name='ECS on EC2',
        line=dict(color='#E16400', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=sizes, 
        y=ecs_fargate_scores, 
        mode='lines+markers', 
        name='ECS on Fargate',
        line=dict(color='#E16400', width=3, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=sizes, 
        y=eks_ec2_scores, 
        mode='lines+markers', 
        name='EKS on EC2',
        line=dict(color='#147EBA', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=sizes, 
        y=eks_fargate_scores, 
        mode='lines+markers', 
        name='EKS on Fargate',
        line=dict(color='#147EBA', width=3, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='ML Model Size vs. Recommended Container Environment',
        xaxis=dict(
            title='Model Size (MB)',
            type='log',
            tickvals=sizes,
            ticktext=[f"{s} MB" for s in sizes]
        ),
        yaxis=dict(
            title='Suitability Score',
            range=[0, 10]
        ),
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ML deployment workflow
    st.markdown("### 🔄 Container-based ML Deployment Workflow")
    
    st.markdown("""
    <div class="card">
    <h3>End-to-End Workflow</h3>
    <ol>
        <li><strong>Build ML Model</strong> - Train and validate your model in SageMaker or your development environment</li>
        <li><strong>Containerize Model</strong> - Create a Docker container with your model and inference code</li>
        <li><strong>Push to ECR</strong> - Upload your containerized model to Amazon ECR</li>
        <li><strong>Deploy Container</strong> - Deploy to ECS or EKS with appropriate compute resources</li>
        <li><strong>Configure Scaling</strong> - Set up auto-scaling rules for your container service</li>
        <li><strong>Monitor and Optimize</strong> - Use CloudWatch to monitor performance and costs</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Best practices
    st.markdown("### 📌 Best Practices for Container-based ML Deployments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Design Practices
        - Design containers to be stateless
        - Separate model artifacts from container code
        - Use multi-stage builds to minimize container size
        - Implement health checks for your model servers
        - Cache common datasets or embeddings when possible
        """)
    
    with col2:
        st.markdown("""
        #### Operational Practices
        - Implement CI/CD pipelines for model updates
        - Use Infrastructure as Code for deployments
        - Monitor both container and model metrics
        - Implement A/B testing capabilities
        - Plan for model version rollbacks
        """)

def show_iac_for_ml():
    """Display the IaC for ML content."""
    st.markdown("## 🏗️ Infrastructure as Code for ML Deployments")
    
    st.markdown("""
    <div class="info-box">
    <h3>Why Use Infrastructure as Code for ML?</h3>
    <p>Infrastructure as Code (IaC) enables you to define your ML infrastructure in a programmatic and version-controlled way. 
    This approach ensures consistency, repeatability, and traceability for your ML deployments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # IaC Tools Overview
    st.markdown("### AWS Infrastructure as Code Tools")
    
    tool_tabs = st.tabs([
        "☁️ AWS CloudFormation", 
        "🌍 AWS CDK"
    ])
    
    with tool_tabs[0]:
        st.markdown("### AWS CloudFormation")
        
        st.markdown("""
        AWS CloudFormation is AWS's native IaC service that allows you to define your AWS resources using JSON or YAML templates.
        
        #### Key Features for ML Infrastructure:
        - Native AWS service with deep integration
        - Support for all ML-related AWS services
        - Change sets to preview infrastructure changes
        - Stack sets for multi-account/multi-region deployments
        """)
        
        st.image("https://d1.awsstatic.com/Products/product-name/diagrams/cloudformation-how-it-works-diagram.040048a9ba773d3a8b6dd9a90dbdcfaea2be45ef.png",
                caption="AWS CloudFormation Workflow")
        
        st.markdown("#### Example CloudFormation Template for SageMaker Endpoint")
        
        st.code("""
AWSTemplateFormatVersion: '2010-09-09'
Description: 'CloudFormation template for SageMaker model and endpoint'

Parameters:
  ModelName:
    Type: String
    Default: xgboost-model
    Description: Name of the SageMaker model
  
  ModelDataUrl:
    Type: String
    Description: S3 path to the model artifacts
  
  InstanceType:
    Type: String
    Default: ml.m5.large
    Description: Instance type for the SageMaker endpoint

  AutoScalingMinCapacity:
    Type: Number
    Default: 1
    Description: Minimum number of instances for autoscaling
  
  AutoScalingMaxCapacity:
    Type: Number
    Default: 4
    Description: Maximum number of instances for autoscaling

Resources:
  ExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

  SageMakerModel:
    Type: AWS::SageMaker::Model
    Properties:
      ModelName: !Ref ModelName
      ExecutionRoleArn: !GetAtt ExecutionRole.Arn
      PrimaryContainer:
        Image: !Sub '{{resolve:ssm:/aws/service/sagemaker-xgboost/latest/image/us-east-1}}'
        ModelDataUrl: !Ref ModelDataUrl

  SageMakerEndpointConfig:
    Type: AWS::SageMaker::EndpointConfig
    Properties:
      EndpointConfigName: !Sub ${ModelName}-config
      ProductionVariants:
        - VariantName: default
          ModelName: !Ref SageMakerModel
          InitialInstanceCount: 1
          InstanceType: !Ref InstanceType

  SageMakerEndpoint:
    Type: AWS::SageMaker::Endpoint
    Properties:
      EndpointName: !Sub ${ModelName}-endpoint
      EndpointConfigName: !GetAtt SageMakerEndpointConfig.EndpointConfigName

  ScalableTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: !Ref AutoScalingMaxCapacity
      MinCapacity: !Ref AutoScalingMinCapacity
      ResourceId: !Sub endpoint/${SageMakerEndpoint}/variant/default
      RoleARN: !Sub arn:aws:iam::${AWS::AccountId}:role/aws-service-role/sagemaker.application-autoscaling.amazonaws.com/AWSServiceRoleForApplicationAutoScaling_SageMakerEndpoint
      ScalableDimension: sagemaker:variant:DesiredInstanceCount
      ServiceNamespace: sagemaker

  ScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: !Sub ${ModelName}-scaling-policy
      PolicyType: TargetTrackingScaling
      ScalingTargetId: !Ref ScalableTarget
      TargetTrackingScalingPolicyConfiguration:
        TargetValue: 70.0
        PredefinedMetricSpecification:
          PredefinedMetricType: SageMakerVariantCPUUtilization

Outputs:
  EndpointName:
    Description: Name of the SageMaker endpoint
    Value: !GetAtt SageMakerEndpoint.EndpointName
        """, language="yaml")
        
        # CloudFormation Visualizer
        st.markdown("#### CloudFormation Stack Visualization")
        
        # Display a sample stack visualization
        st.image("https://d2908q01vomqb2.cloudfront.net/fc074d501302eb2b93e2554793fcaf50b3bf7291/2019/03/22/Visualizer-5.png", 
                caption="CloudFormation Stack Visualization Example")
    
    with tool_tabs[1]:
        st.markdown("### AWS Cloud Development Kit (CDK)")
        
        st.markdown("""
        AWS CDK allows you to define cloud infrastructure using familiar programming languages like Python, TypeScript, and Java.
        
        #### Key Features for ML Infrastructure:
        - Use programming languages instead of JSON/YAML
        - Object-oriented approach to infrastructure
        - Reusable components with constructs
        - Integration with AWS best practices
        """)
        
        st.image("https://d2908q01vomqb2.cloudfront.net/7719a1c782a1ba91c031a682a0a2f8658209adbf/2020/11/04/Mapping-the-path-image-1.png",
                caption="AWS CDK Architecture Workflow")
        
        st.markdown("#### Example AWS CDK Code for SageMaker Endpoint (Python)")
        
        st.code("""
import aws_cdk as cdk
from aws_cdk import (
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    aws_applicationautoscaling as appscaling,
)
from constructs import Construct

class SageMakerEndpointStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # Create IAM Role for SageMaker
        sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess"),
            ]
        )
        
        # Create SageMaker Model
        model = sagemaker.CfnModel(
            self, "XGBoostModel",
            execution_role_arn=sagemaker_role.role_arn,
            model_name="xgboost-model",
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.3-1",
                model_data_url="s3://my-bucket/model/xgboost-model.tar.gz"
            )
        )
        
        # Create Endpoint Configuration
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "XGBoostEndpointConfig",
            endpoint_config_name="xgboost-endpoint-config",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    initial_instance_count=1,
                    instance_type="ml.m5.large",
                    model_name=model.model_name,
                    variant_name="default"
                )
            ]
        )
        endpoint_config.add_dependency(model)
        
        # Create Endpoint
        endpoint = sagemaker.CfnEndpoint(
            self, "XGBoostEndpoint",
            endpoint_name="xgboost-endpoint",
            endpoint_config_name=endpoint_config.endpoint_config_name
        )
        endpoint.add_dependency(endpoint_config)
        
        # Setup autoscaling
        scalable_target = appscaling.CfnScalableTarget(
            self, "SageMakerScalableTarget",
            max_capacity=4,
            min_capacity=1,
            resource_id=f"endpoint/{endpoint.endpoint_name}/variant/default",
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
            service_namespace="sagemaker"
        )
        scalable_target.add_dependency(endpoint)
        
        scaling_policy = appscaling.CfnScalingPolicy(
            self, "SageMakerScalingPolicy",
            policy_name="xgboost-scaling-policy",
            policy_type="TargetTrackingScaling",
            resource_id=scalable_target.resource_id,
            scalable_dimension=scalable_target.scalable_dimension,
            service_namespace=scalable_target.service_namespace,
            target_tracking_scaling_policy_configuration=appscaling.CfnScalingPolicy.TargetTrackingScalingPolicyConfigurationProperty(
                predefined_metric_specification=appscaling.CfnScalingPolicy.PredefinedMetricSpecificationProperty(
                    predefined_metric_type="SageMakerVariantCPUUtilization"
                ),
                target_value=70.0
            )
        )
        scaling_policy.add_dependency(scalable_target)
        
        # Output the endpoint name
        cdk.CfnOutput(
            self, "EndpointName",
            value=endpoint.endpoint_name
        )

app = cdk.App()
SageMakerEndpointStack(app, "SageMakerEndpointStack")
app.synth()
        """, language="python")
        
        # CDK Comparison with CloudFormation
        st.markdown("#### CloudFormation vs. CDK")
        
        comparison_data = {
            'Feature': [
                'Definition Language', 
                'Learning Curve', 
                'Reusability', 
                'Type Safety',
                'Abstraction Level',
                'Debugging',
                'Ecosystem'
            ],
            'CloudFormation': [
                'YAML/JSON', 
                'Medium', 
                'Limited (via Nested Stacks)', 
                'None (templates are text)',
                'Low-level',
                'Harder to debug',
                'Mature'
            ],
            'CDK': [
                'Python, TypeScript, Java, C#', 
                'Medium-High', 
                'High (via Constructs)', 
                'Strong (based on language)',
                'High-level',
                'IDE support, better debugging',
                'Growing rapidly'
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    # IaC for ML use cases
    st.markdown("### 📈 IaC for ML Deployment Scenarios")
    
    scenario_tabs = st.tabs([
        "SageMaker Studio Environment", 
        "ML Pipeline", 
        "Multi-Account ML Platform"
    ])
    
    with scenario_tabs[0]:
        st.markdown("### SageMaker Studio Environment")
        st.markdown("""
        Deploy a complete SageMaker Studio environment for your data science team with user profiles,
        domains, and appropriate IAM permissions.
        """)
        
        st.code("""
# CDK Code to set up a SageMaker Studio environment

import aws_cdk as cdk
from aws_cdk import (
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    aws_ec2 as ec2,
)
from constructs import Construct

class SageMakerStudioStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # Create VPC for SageMaker Studio
        vpc = ec2.Vpc(
            self, "SageMakerStudioVPC",
            max_azs=2,
            nat_gateways=1
        )
        
        # Create SageMaker Studio Execution Role
        studio_role = iam.Role(
            self, "SageMakerStudioExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
            ]
        )
        
        # Create SageMaker Domain
        domain = sagemaker.CfnDomain(
            self, "SageMakerDomain",
            auth_mode="IAM",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=studio_role.role_arn
            ),
            domain_name="ml-team-domain",
            subnet_ids=vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT).subnet_ids,
            vpc_id=vpc.vpc_id
        )
        
        # Create User Profiles
        data_scientist_profile = sagemaker.CfnUserProfile(
            self, "DataScientistProfile",
            domain_id=domain.attr_domain_id,
            user_profile_name="data-scientist",
            user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                execution_role=studio_role.role_arn
            )
        )
        
        ml_engineer_profile = sagemaker.CfnUserProfile(
            self, "MLEngineerProfile",
            domain_id=domain.attr_domain_id,
            user_profile_name="ml-engineer",
            user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                execution_role=studio_role.role_arn
            )
        )
        
        # Outputs
        cdk.CfnOutput(
            self, "DomainId",
            value=domain.attr_domain_id
        )
        
        cdk.CfnOutput(
            self, "DataScientistProfileName",
            value=data_scientist_profile.user_profile_name
        )
        
        cdk.CfnOutput(
            self, "MLEngineerProfileName",
            value=ml_engineer_profile.user_profile_name
        )

app = cdk.App()
SageMakerStudioStack(app, "SageMakerStudioStack")
app.synth()
        """, language="python")
        
    with scenario_tabs[1]:
        st.markdown("### ML Pipeline Infrastructure")
        st.markdown("""
        Deploy a complete ML pipeline infrastructure including data processing, training, evaluation, and deployment.
        """)
        
        st.image("https://d1.awsstatic.com/Products/product-name/diagrams/product-page-diagram_Amazon-SageMaker-Pipelines.73b37cb5f108ae7a99ca12ad0e0fbc6de7c71ade.png",
                caption="ML Pipeline Architecture")
        
        st.code("""
# CloudFormation Template for a SageMaker Pipeline

AWSTemplateFormatVersion: '2010-09-09'
Description: 'CloudFormation template for a complete ML pipeline'

Resources:
  # IAM Role for SageMaker
  SageMakerExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        - arn:aws:iam::aws:policy/AmazonS3FullAccess

  # S3 buckets for data and model artifacts
  DataBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub ml-pipeline-data-${AWS::AccountId}
      VersioningConfiguration:
        Status: Enabled

  ModelBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub ml-pipeline-models-${AWS::AccountId}
      VersioningConfiguration:
        Status: Enabled

  # SageMaker Pipeline
  MLPipeline:
    Type: AWS::SageMaker::Pipeline
    Properties:
      PipelineName: ml-training-pipeline
      PipelineDescription: ML training and deployment pipeline
      RoleArn: !GetAtt SageMakerExecutionRole.Arn
      PipelineDefinition:
        PipelineDefinitionBody: !Sub |
          {
            "Version": "2020-12-01",
            "Parameters": [
              {
                "Name": "InputDataPath",
                "Type": "String",
                "DefaultValue": "s3://${DataBucket}/raw-data/"
              },
              {
                "Name": "ProcessedDataPath",
                "Type": "String",
                "DefaultValue": "s3://${DataBucket}/processed-data/"
              },
              {
                "Name": "ModelPath",
                "Type": "String", 
                "DefaultValue": "s3://${ModelBucket}/model/"
              },
              {
                "Name": "InstanceType",
                "Type": "String",
                "DefaultValue": "ml.m5.large"
              }
            ],
            "Steps": [
              {
                "Name": "DataPreprocessingStep",
                "Type": "Processing",
                "Arguments": {
                  "ProcessingResources": {
                    "ClusterConfig": {
                      "InstanceCount": 1,
                      "InstanceType": {
                        "Type": "ParameterString",
                        "Name": "InstanceType"
                      },
                      "VolumeSizeInGB": 30
                    }
                  },
                  "AppSpecification": {
                    "ImageUri": "{{resolve:ssm:/aws/service/sagemaker-scikit-learn/latest/image/us-east-1}}"
                  },
                  "ProcessingInputs": [
                    {
                      "InputName": "input-data",
                      "S3Input": {
                        "S3Uri": {
                          "Type": "ParameterString",
                          "Name": "InputDataPath"
                        },
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                      }
                    }
                  ],
                  "ProcessingOutputs": [
                    {
                      "OutputName": "processed-data",
                      "S3Output": {
                        "S3Uri": {
                          "Type": "ParameterString",
                          "Name": "ProcessedDataPath"
                        },
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob"
                      }
                    }
                  ]
                }
              },
              {
                "Name": "ModelTrainingStep",
                "Type": "Training",
                "DependsOn": ["DataPreprocessingStep"],
                "Arguments": {
                  "AlgorithmSpecification": {
                    "TrainingImage": "{{resolve:ssm:/aws/service/sagemaker-xgboost/latest/image/us-east-1}}",
                    "TrainingInputMode": "File"
                  },
                  "OutputDataConfig": {
                    "S3OutputPath": {
                      "Type": "ParameterString",
                      "Name": "ModelPath"
                    }
                  },
                  "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                  },
                  "ResourceConfig": {
                    "InstanceCount": 1,
                    "InstanceType": {
                      "Type": "ParameterString",
                      "Name": "InstanceType"
                    },
                    "VolumeSizeInGB": 50
                  },
                  "RoleArn": "${SageMakerExecutionRole.Arn}",
                  "InputDataConfig": [
                    {
                      "ChannelName": "train",
                      "DataSource": {
                        "S3DataSource": {
                          "S3Uri": {
                            "Type": "ParameterString",
                            "Name": "ProcessedDataPath"
                          },
                          "S3DataType": "S3Prefix",
                          "S3DataDistributionType": "FullyReplicated"
                        }
                      }
                    }
                  ]
                }
              },
              {
                "Name": "ModelEvaluationStep",
                "Type": "Processing",
                "DependsOn": ["ModelTrainingStep"],
                "Arguments": {
                  "ProcessingResources": {
                    "ClusterConfig": {
                      "InstanceCount": 1,
                      "InstanceType": {
                        "Type": "ParameterString",
                        "Name": "InstanceType"
                      },
                      "VolumeSizeInGB": 30
                    }
                  },
                  "AppSpecification": {
                    "ImageUri": "{{resolve:ssm:/aws/service/sagemaker-scikit-learn/latest/image/us-east-1}}"
                  },
                  "ProcessingInputs": [
                    {
                      "InputName": "model",
                      "S3Input": {
                        "S3Uri": {
                          "Type": "Join",
                          "On": "",
                          "Values": [
                            {
                              "Type": "ParameterString",
                              "Name": "ModelPath"
                            },
                            {
                              "Type": "Step",
                              "Name": "ModelTrainingStep"
                            },
                            "/output/model.tar.gz"
                          ]
                        },
                        "LocalPath": "/opt/ml/processing/model",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                      }
                    },
                    {
                      "InputName": "test-data",
                      "S3Input": {
                        "S3Uri": {
                          "Type": "ParameterString",
                          "Name": "ProcessedDataPath"
                        },
                        "LocalPath": "/opt/ml/processing/test",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                      }
                    }
                  ],
                  "ProcessingOutputs": [
                    {
                      "OutputName": "evaluation",
                      "S3Output": {
                        "S3Uri": {
                          "Type": "Join",
                          "On": "",
                          "Values": [
                            {
                              "Type": "ParameterString",
                              "Name": "ModelPath"
                            },
                            "evaluation/"
                          ]
                        },
                        "LocalPath": "/opt/ml/processing/evaluation",
                        "S3UploadMode": "EndOfJob"
                      }
                    }
                  ]
                }
              }
            ]
          }

Outputs:
  PipelineName:
    Description: The name of the SageMaker pipeline
    Value: !GetAtt MLPipeline.PipelineName
  
  DataBucketName:
    Description: The name of the data bucket
    Value: !Ref DataBucket
  
  ModelBucketName:
    Description: The name of the model bucket
    Value: !Ref ModelBucket
        """, language="yaml")
    
    with scenario_tabs[2]:
        st.markdown("### Multi-Account ML Platform")
        st.markdown("""
        Infrastructure for a multi-account ML platform with separate development, testing, and production environments.
        """)
        
        st.image("https://d1.awsstatic.com/Solutions/aws-solutions/Secure-Environment-Accelerator/cid-arch.aaed8984ad4503f9306046eec6abc71520965187.png",
                caption="Multi-Account Architecture")
        
        st.markdown("""
        #### CloudFormation StackSets for Multi-Account Deployment
        
        CloudFormation StackSets enable you to deploy a single infrastructure template across multiple AWS accounts and regions,
        which is perfect for enterprise ML platforms that span development, testing, and production environments.
        """)
        
        st.code("""
# CloudFormation StackSet deployment using AWS CLI

# First, create the IAM roles needed for StackSet operations
aws cloudformation create-stack \
    --stack-name StackSetAdministrationRole \
    --template-url https://s3.amazonaws.com/cloudformation-stackset-sample-templates-us-east-1/AWSCloudFormationStackSetAdministrationRole.yml \
    --capabilities CAPABILITY_NAMED_IAM

# In each target account, create the IAM role for StackSet execution
aws cloudformation create-stack \
    --stack-name StackSetExecutionRole \
    --template-url https://s3.amazonaws.com/cloudformation-stackset-sample-templates-us-east-1/AWSCloudFormationStackSetExecutionRole.yml \
    --parameters ParameterKey=AdministratorAccountId,ParameterValue=ADMIN_ACCOUNT_ID \
    --capabilities CAPABILITY_NAMED_IAM

# Now create the StackSet for ML infrastructure
aws cloudformation create-stack-set \
    --stack-set-name MLInfrastructure \
    --template-body file://ml-infrastructure.yaml \
    --capabilities CAPABILITY_NAMED_IAM \
    --permission-model SERVICE_MANAGED \
    --auto-deployment Enabled=true,RetainStacks=true

# Create stack instances in each environment (dev, test, prod)
aws cloudformation create-stack-instances \
    --stack-set-name MLInfrastructure \
    --accounts DEVELOPMENT_ACCOUNT_ID TEST_ACCOUNT_ID PRODUCTION_ACCOUNT_ID \
    --regions us-east-1 \
    --parameter-overrides \
        ParameterKey=Environment,ParameterValue=development \
        ParameterKey=InstanceType,ParameterValue=ml.m5.large \
        ParameterKey=AutoScalingMaxCapacity,ParameterValue=2
        """, language="bash")
    
    # IaC Best Practices
    st.markdown("### 📌 Best Practices for ML Infrastructure as Code")
    
    st.markdown("""
    <div class="card">
    <ol>
        <li><strong>Parameterize your templates</strong> - Create reusable templates for different environments (dev, staging, production)</li>
        <li><strong>Use CI/CD pipelines</strong> - Automate infrastructure deployment with CI/CD pipelines</li>
        <li><strong>Version control</strong> - Keep infrastructure code in version control alongside application code</li>
        <li><strong>Infrastructure testing</strong> - Test infrastructure changes before deploying to production</li>
        <li><strong>Security as code</strong> - Include security policies and permissions in your IaC</li>
        <li><strong>Cost optimization</strong> - Use IaC to implement cost-saving features like auto-scaling and scheduled scaling</li>
        <li><strong>Documentation</strong> - Document your infrastructure decisions and architecture</li>
        <li><strong>Modularize</strong> - Break down complex infrastructure into reusable modules</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced topics
    st.markdown("### 🚀 Advanced IaC for ML Infrastructure")
    
    advanced_topics = st.expander("Explore Advanced Topics")
    
    with advanced_topics:
        st.markdown("""
        #### 1. GitOps for ML Infrastructure
        
        Implement GitOps workflows for your ML infrastructure changes:
        - Store IaC templates in Git repositories
        - Use pull requests for infrastructure changes
        - Automated testing and validation
        - CI/CD pipelines for deployment
        
        #### 2. Dynamic Infrastructure with AWS Step Functions
        
        Use AWS Step Functions to create dynamic ML infrastructure:
        - Create infrastructure on-demand
        - Tear down resources when not needed
        - Implement ML lifecycle management
        - Optimize costs with ephemeral resources
        
        #### 3. Custom Resource Providers
        
        Extend CloudFormation or CDK with custom resources:
        - Create custom ML-specific resources
        - Integrate with third-party ML tools
        - Create high-level abstractions for ML workflows
        - Implement custom validation logic
        """)
        
        st.code("""
# AWS CDK Custom Resource for ML Model Registration

import aws_cdk as cdk
from aws_cdk import (
    aws_lambda as lambda_,
    custom_resources as cr,
    aws_iam as iam
)
from constructs import Construct

class MLModelRegistrationStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # Lambda function to register a model in the ML registry
        registration_lambda = lambda_.Function(
            self, "ModelRegistrationLambda",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda"),
            timeout=cdk.Duration.minutes(5),
            environment={
                "MODEL_REGISTRY_TABLE": "my-model-registry-table"
            }
        )
        
        # Grant necessary permissions
        registration_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateModelPackage",
                    "sagemaker:CreateModelPackageGroup",
                    "sagemaker:ListModelPackages",
                    "dynamodb:PutItem",
                    "dynamodb:GetItem"
                ],
                resources=["*"]
            )
        )
        
        # Create a custom resource provider
        model_registration_provider = cr.Provider(
            self, "ModelRegistrationProvider",
            on_event_handler=registration_lambda
        )
        
        # Create a custom resource for model registration
        model_registration = cdk.CustomResource(
            self, "ModelRegistration",
            service_token=model_registration_provider.service_token,
            properties={
                "ModelName": "my-xgboost-model",
                "ModelVersion": "1.0",
                "ModelArtifactLocation": "s3://my-bucket/model.tar.gz",
                "Framework": "XGBoost",
                "FrameworkVersion": "1.3-1"
            }
        )
        
        # Output the model package ARN
        cdk.CfnOutput(
            self, "ModelPackageArn",
            value=model_registration.get_att_string("ModelPackageArn")
        )

app = cdk.App()
MLModelRegistrationStack(app, "MLModelRegistrationStack")
app.synth()
        """, language="python")

def show_knowledge_check():
    """Display the Knowledge Check content."""
    st.markdown("## ✅ Knowledge Check")
    
    st.markdown("""
    <div class="info-box">
    <h3>Test Your Knowledge</h3>
    <p>Complete these quizzes to test your understanding of AWS ML Infrastructure concepts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for each quiz category
    quiz_tabs = st.tabs([
        "Autoscaling Quiz", 
        "Container Services Quiz", 
        "Infrastructure as Code Quiz"
    ])
    
    # Autoscaling Quiz
    with quiz_tabs[0]:
        st.markdown("### SageMaker Autoscaling Quiz")
        
        if not st.session_state.quiz_submitted["autoscaling"]:
            with st.form("autoscaling_quiz"):
                st.markdown("**Question 1**: Which scaling policy adjusts capacity based on a target value for a specific metric?")
                q1 = st.radio(
                    "Select the correct answer:",
                    ["Step Scaling", "Target Tracking Scaling", "Scheduled Scaling", "On-Demand Scaling"],
                    key="autoscale_q1", index=None
                )
                
                st.markdown("**Question 2**: What is the best metric to use for autoscaling when your ML model is CPU-bound?")
                q2 = st.radio(
                    "Select the correct answer:",
                    ["InvocationsPerInstance", "CPUUtilization", "MemoryUtilization", "GPUUtilization"],
                    key="autoscale_q2", index=None
                )
                
                st.markdown("**Question 3**: What parameter prevents your endpoint from scaling in too quickly after a traffic spike?")
                q3 = st.radio(
                    "Select the correct answer:",
                    ["ScaleOutCooldown", "ScaleInCooldown", "TargetValue", "MinCapacity"],
                    key="autoscale_q3", index=None
                )
                
                submitted = st.form_submit_button("Submit")
                
                if submitted:
                    score = 0
                    if q1 == "Target Tracking Scaling":
                        score += 1
                    if q2 == "CPUUtilization":
                        score += 1
                    if q3 == "ScaleInCooldown":
                        score += 1
                    
                    st.session_state.quiz_scores["autoscaling"] = score
                    st.session_state.quiz_submitted["autoscaling"] = True
                    st.rerun()
        else:
            score = st.session_state.quiz_scores["autoscaling"]
            st.success(f"You scored {score}/3 on the SageMaker Autoscaling quiz!")
            
            # Show correct answers
            st.markdown("#### Correct Answers:")
            st.markdown("1. Target Tracking Scaling")
            st.markdown("2. CPUUtilization")
            st.markdown("3. ScaleInCooldown")
            
            if st.button("Retake Quiz", key="retake_autoscaling"):
                st.session_state.quiz_submitted["autoscaling"] = False
                st.rerun()
    
    # Container Services Quiz
    with quiz_tabs[1]:
        st.markdown("### Container Services Quiz")
        
        if not st.session_state.quiz_submitted["container"]:
            with st.form("container_quiz"):
                st.markdown("**Question 1**: Which AWS container orchestration service is based on Kubernetes?")
                q1 = st.radio(
                    "Select the correct answer:",
                    ["Amazon ECS", "Amazon EKS", "AWS App Runner", "Amazon ECR"],
                    key="container_q1", index=None
                )
                
                st.markdown("**Question 2**: Which compute platform option requires you to manage the EC2 instances in your container cluster?")
                q2 = st.radio(
                    "Select the correct answer:",
                    ["AWS Fargate", "EC2", "Lambda", "SageMaker"],
                    key="container_q2", index=None
                )
                
                st.markdown("**Question 3**: What AWS service allows you to store, manage, and deploy Docker container images?")
                q3 = st.radio(
                    "Select the correct answer:",
                    ["Amazon ECS", "Amazon EKS", "Amazon ECR", "AWS Fargate"],
                    key="container_q3", index=None
                )
                
                submitted = st.form_submit_button("Submit")
                
                if submitted:
                    score = 0
                    if q1 == "Amazon EKS":
                        score += 1
                    if q2 == "EC2":
                        score += 1
                    if q3 == "Amazon ECR":
                        score += 1
                    
                    st.session_state.quiz_scores["container"] = score
                    st.session_state.quiz_submitted["container"] = True
                    st.rerun()
        else:
            score = st.session_state.quiz_scores["container"]
            st.success(f"You scored {score}/3 on the Container Services quiz!")
            
            # Show correct answers
            st.markdown("#### Correct Answers:")
            st.markdown("1. Amazon EKS")
            st.markdown("2. EC2")
            st.markdown("3. Amazon ECR")
            
            if st.button("Retake Quiz", key="retake_container"):
                st.session_state.quiz_submitted["container"] = False
                st.rerun()
    
    # Infrastructure as Code Quiz
    with quiz_tabs[2]:
        st.markdown("### Infrastructure as Code Quiz")
        
        if not st.session_state.quiz_submitted["iac"]:
            with st.form("iac_quiz"):
                st.markdown("**Question 1**: Which IaC tool allows you to define infrastructure using programming languages like Python and TypeScript?")
                q1 = st.radio(
                    "Select the correct answer:",
                    ["AWS CloudFormation", "AWS CDK", "AWS SAM", "AWS CodeDeploy"],
                    key="iac_q1", index=None
                )
                
                st.markdown("**Question 2**: Which statement about CloudFormation is correct?")
                q2 = st.radio(
                    "Select the correct answer:",
                    ["It can only deploy resources to AWS", 
                     "It uses HashiCorp Configuration Language (HCL)", 
                     "It requires a separate state file management system",
                     "It has limited integration with AWS services"],
                    key="iac_q2", index=None
                )
                
                st.markdown("**Question 3**: What is the primary advantage of using Infrastructure as Code for ML deployments?")
                q3 = st.radio(
                    "Select the correct answer:",
                    ["It automatically optimizes ML model performance",
                     "It enables reproducible and consistent infrastructure deployment",
                     "It eliminates the need for cloud resources",
                     "It provides automatic hyperparameter tuning"],
                    key="iac_q3", index=None
                )
                
                submitted = st.form_submit_button("Submit")
                
                if submitted:
                    score = 0
                    if q1 == "AWS CDK":
                        score += 1
                    if q2 == "It can only deploy resources to AWS":
                        score += 1
                    if q3 == "It enables reproducible and consistent infrastructure deployment":
                        score += 1
                    
                    st.session_state.quiz_scores["iac"] = score
                    st.session_state.quiz_submitted["iac"] = True
                    st.rerun()
        else:
            score = st.session_state.quiz_scores["iac"]
            st.success(f"You scored {score}/3 on the Infrastructure as Code quiz!")
            
            # Show correct answers
            st.markdown("#### Correct Answers:")
            st.markdown("1. AWS CDK")
            st.markdown("2. It can only deploy resources to AWS")
            st.markdown("3. It enables reproducible and consistent infrastructure deployment")
            
            if st.button("Retake Quiz", key="retake_iac"):
                st.session_state.quiz_submitted["iac"] = False
                st.rerun()
    
    # Overall score and progress
    st.markdown("### Your Progress")
    
    # Calculate total score
    total_score = sum(st.session_state.quiz_scores.values())
    total_possible = 9  # 3 quizzes with 3 questions each
    
    # Display progress bar
    progress = total_score / total_possible
    st.progress(progress)
    
    # Display score summary
    st.markdown(f"**Total Score**: {total_score}/{total_possible}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Autoscaling", f"{st.session_state.quiz_scores['autoscaling']}/3")
    with col2:
        st.metric("Container Services", f"{st.session_state.quiz_scores['container']}/3")
    with col3:
        st.metric("Infrastructure as Code", f"{st.session_state.quiz_scores['iac']}/3")
    
    # Reset all quiz scores
    if st.button("Reset All Quizzes"):
        st.session_state.quiz_scores = {
            "autoscaling": 0,
            "container": 0,
            "iac": 0
        }
        st.session_state.quiz_submitted = {
            "autoscaling": False,
            "container": False,
            "iac": False
        }
        st.rerun()

def main():
    """Main function to render the Streamlit application."""
    st.set_page_config(
        page_title="AWS ML Deployment Infrastructure",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Load CSS
    load_css()

    st.markdown("## 🚀 AWS ML Infrastructure Deployment")
    
    st.markdown("""
    <p>Learn about the different infrastructure options designed to help you optimize performance, manage costs, 
    and create reliable ML workloads.</p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:        
        st.markdown("### Session Management")
        st.info(f"User ID: {st.session_state.user_id}")
        if st.button("🔄 Reset Session"):
            reset_session()
        
        st.sidebar.divider()
        
        with st.expander("📚 About This App", expanded=False):
          st.markdown("### Resources")
          st.markdown("""
          - [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
          - [AWS Container Services](https://aws.amazon.com/containers/)
          - [Infrastructure as Code](https://aws.amazon.com/devops/infrastructure-as-code/)
          """)
    
    # Main content area - tabs
    tabs = st.tabs([
        "🏠 Home", 
        "⚖️ SageMaker Autoscaling", 
        "🐳 Container Services",
        "🏗️ Infrastructure as Code",
        "✅ Knowledge Check"
    ])
    
    with tabs[0]:
        show_home_page()
    
    with tabs[1]:
        show_sagemaker_autoscaling()
    
    with tabs[2]:
        show_aws_container_services()
    
    with tabs[3]:
        show_iac_for_ml()
        
    with tabs[4]:
        show_knowledge_check()
    
    render_footer()

if __name__ == "__main__":
    main()
