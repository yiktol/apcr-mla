
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


# Custom utility functions
def load_lottieurl(url: str):
    """
    Load a Lottie animation from a URL
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def generate_random_model_metrics():
    """
    Generate random metrics for model demonstration purposes
    """
    return {
        'accuracy': round(random.uniform(0.85, 0.99), 4),
        'precision': round(random.uniform(0.82, 0.98), 4),
        'recall': round(random.uniform(0.80, 0.96), 4),
        'f1_score': round(random.uniform(0.83, 0.97), 4),
        'auc': round(random.uniform(0.85, 0.99), 4),
        'latency_ms': round(random.uniform(5, 50), 1)
    }


def generate_model_lineage_graph():
    """
    Generate a model lineage graph for visualization
    """
    G = nx.DiGraph()
    
    # Add nodes for datasets
    G.add_node("Raw Data", type="data", color="#3F88C5")
    G.add_node("Processed Data", type="data", color="#3F88C5")
    G.add_node("Training Dataset", type="data", color="#3F88C5")
    G.add_node("Validation Dataset", type="data", color="#3F88C5")
    
    # Add nodes for processing steps
    G.add_node("Data Processing", type="process", color="#FF9900")
    G.add_node("Feature Engineering", type="process", color="#FF9900")
    G.add_node("Training Job", type="process", color="#FF9900")
    G.add_node("Model Evaluation", type="process", color="#FF9900")
    
    # Add model nodes
    G.add_node("Model v1", type="model", color="#16DB93")
    G.add_node("Model v2", type="model", color="#16DB93")
    G.add_node("Approved Model", type="model", color="#16DB93")
    
    # Add deployment nodes
    G.add_node("Production Endpoint", type="endpoint", color="#FFC914")
    
    # Add edges to show flow
    G.add_edge("Raw Data", "Data Processing")
    G.add_edge("Data Processing", "Processed Data")
    G.add_edge("Processed Data", "Feature Engineering")
    G.add_edge("Feature Engineering", "Training Dataset")
    G.add_edge("Feature Engineering", "Validation Dataset")
    G.add_edge("Training Dataset", "Training Job")
    G.add_edge("Training Job", "Model v1")
    G.add_edge("Training Job", "Model v2")
    G.add_edge("Model v1", "Model Evaluation")
    G.add_edge("Model v2", "Model Evaluation")
    G.add_edge("Validation Dataset", "Model Evaluation")
    G.add_edge("Model v2", "Approved Model")
    G.add_edge("Approved Model", "Production Endpoint")
    
    return G


def draw_model_lineage(G, highlight_path=None):
    """
    Draw a model lineage graph using NetworkX and Matplotlib
    
    Args:
        G (NetworkX.DiGraph): The graph to draw
        highlight_path (list): Optional list of node names to highlight
    """
    plt.figure(figsize=(12, 8))
    
    # Define node positions using a hierarchical layout
    pos = {
        "Raw Data": (0, 5),
        "Data Processing": (2, 5),
        "Processed Data": (4, 5),
        "Feature Engineering": (6, 5),
        "Training Dataset": (8, 6),
        "Validation Dataset": (8, 4),
        "Training Job": (10, 6),
        "Model v1": (12, 7),
        "Model v2": (12, 5),
        "Model Evaluation": (14, 6),
        "Approved Model": (16, 5),
        "Production Endpoint": (18, 5)
    }
    
    # Get node colors based on type
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    # Define node shapes based on type
    node_shapes = {}
    for node in G.nodes:
        if G.nodes[node]['type'] == 'data':
            node_shapes[node] = 's'  # square
        elif G.nodes[node]['type'] == 'process':
            node_shapes[node] = 'o'  # circle
        elif G.nodes[node]['type'] == 'model':
            node_shapes[node] = 'd'  # diamond
        else:
            node_shapes[node] = 'h'  # hexagon
    
    # Draw edges
    edge_color = '#aaaaaa'
    highlight_color = '#FF9900'
    
    for edge in G.edges():
        if highlight_path and edge[0] in highlight_path and edge[1] in highlight_path:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=highlight_color, 
                                  width=2.0, arrowsize=20)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_color, 
                                  width=1.0, arrowsize=15, alpha=0.7)
    
    # Draw nodes
    for node_type in ['data', 'process', 'model', 'endpoint']:
        node_list = [n for n in G.nodes() if G.nodes[n]['type'] == node_type]
        
        if not node_list:
            continue
            
        node_color = [G.nodes[n]['color'] for n in node_list]
        node_shape = node_shapes[node_list[0]]
        
        highlighted = False
        
        if highlight_path:
            # Check if any nodes of this type are in the highlight path
            highlighted_nodes = [n for n in node_list if n in highlight_path]
            regular_nodes = [n for n in node_list if n not in highlight_path]
            
            if highlighted_nodes:
                # Draw highlighted nodes with darker border
                nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes, 
                                      node_color=[G.nodes[n]['color'] for n in highlighted_nodes],
                                      node_shape=node_shape, node_size=1500, edgecolors='black', linewidths=2)
                highlighted = True
                
            if regular_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, 
                                      node_color=[G.nodes[n]['color'] for n in regular_nodes],
                                      node_shape=node_shape, node_size=1500, edgecolors='black', linewidths=1, alpha=0.7)
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                                  node_color=node_color,
                                  node_shape=node_shape, node_size=1500, edgecolors='black', linewidths=1)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()


def generate_model_versions():
    """
    Generate sample model versions data
    """
    versions = []
    base_date = datetime.now() - timedelta(days=30)
    
    model_packages = ["customer-churn-prediction", "fraud-detection", "product-recommendation"]
    model_package = random.choice(model_packages)
    
    frameworks = ["XGBoost", "PyTorch", "TensorFlow", "Scikit-learn"]
    statuses = ["Approved", "Rejected", "Pending"]
    users = ["data-scientist-1", "ml-engineer-2", "model-builder-3"]
    
    # Generate 5-10 versions
    for i in range(1, random.randint(5, 10)):
        version_date = base_date + timedelta(days=i*3)
        
        versions.append({
            "Version": i,
            "ModelPackageGroupName": model_package,
            "ModelPackageArn": f"arn:aws:sagemaker:us-west-2:123456789012:model-package/{model_package}/{i}",
            "CreationTime": version_date,
            "ModelApprovalStatus": random.choice(statuses) if i < 5 else "Approved",
            "Framework": random.choice(frameworks),
            "CreatedBy": random.choice(users),
            "Metrics": generate_random_model_metrics(),
        })
    
    return versions


def generate_registry_groups():
    """
    Generate sample model package groups
    """
    model_groups = [
        {
            "ModelPackageGroupName": "customer-churn-prediction",
            "ModelPackageGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/customer-churn-prediction",
            "Description": "Models for predicting customer churn",
            "CreationTime": datetime.now() - timedelta(days=60),
            "ModelCount": 8,
            "LatestApprovedVersion": 7,
            "Domain": "Marketing",
        },
        {
            "ModelPackageGroupName": "fraud-detection",
            "ModelPackageGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/fraud-detection",
            "Description": "Models for detecting fraudulent transactions",
            "CreationTime": datetime.now() - timedelta(days=45),
            "ModelCount": 5,
            "LatestApprovedVersion": 5,
            "Domain": "Risk",
        },
        {
            "ModelPackageGroupName": "product-recommendation",
            "ModelPackageGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/product-recommendation",
            "Description": "Models for recommending products to customers",
            "CreationTime": datetime.now() - timedelta(days=30),
            "ModelCount": 6,
            "LatestApprovedVersion": 4,
            "Domain": "Marketing",
        }
    ]
    
    return model_groups


def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'registry_groups' not in st.session_state:
        st.session_state.registry_groups = generate_registry_groups()
        
    if 'selected_group' not in st.session_state:
        st.session_state.selected_group = st.session_state.registry_groups[0]
        
    if 'model_versions' not in st.session_state:
        st.session_state.model_versions = generate_model_versions()
        
    if 'lineage_graph' not in st.session_state:
        st.session_state.lineage_graph = generate_model_lineage_graph()
        
    if 'highlight_path' not in st.session_state:
        st.session_state.highlight_path = None


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
        page_title="SageMaker Model Registry Explorer",
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
                This interactive learning application demonstrates the capabilities 
                of Amazon SageMaker Model Registry. Explore how to organize, track, and 
                manage your machine learning models throughout their lifecycle.
                """)
            
            # AWS learning resources
            st.subheader("Additional Resources")
            st.markdown("""
                - [SageMaker Model Registry Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
                - [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
                - [AWS Training and Certification](https://aws.amazon.com/training/)
            """)
    
    # Main app header
    st.title("Amazon SageMaker Model Registry")
    st.markdown("Learn how to organize, catalog, and manage models throughout their lifecycle.")
    
    # # Animation for the header
    # lottie_url = "https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json"
    # lottie_json = load_lottieurl(lottie_url)
    # if lottie_json:
    #     st_lottie(lottie_json, height=200, key="header_animation")
    
    # Tab-based navigation with emoji
    tab1, tab2 = st.tabs([
        "📋 Model Registry Overview", 
        "🏗️ Model Registry Structure",
    ])
    
    # TAB 1: MODEL REGISTRY OVERVIEW
    with tab1:
        st.header("Amazon SageMaker Model Registry Overview")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            SageMaker Model Registry provides a centralized repository for organizing and cataloging your machine learning models. 
            It helps you track model versions, manage model approval states, and deploy approved models to production.
            
            **Key benefits:**
            - **Organize models** into logical groups
            - **Track and version** model iterations
            - **Manage approval workflows** for model governance
            - **Deploy approved models** to production
            - **Trace model lineage** for audibility and compliance
            """)
        
        with col2:
            st.image("images/register_model.png", width=30,
                     caption="SageMaker Model Registry Process Flow", use_container_width=True)
        
        st.subheader("Model Registry Workflow")
        
        # Create a workflow diagram using Plotly
        fig = go.Figure()
        
        # Define stages
        stages = ["Register Model", "Version Model", "Approve/Reject", "Deploy to Production", "Monitor Performance"]
        stage_descriptions = [
            "Register a model and create a model package group",
            "Create versions for each model iteration",
            "Review and approve models for production use",
            "Deploy approved model versions to endpoints",
            "Track and monitor model performance in production"
        ]
        
        # Add nodes
        x_positions = [0, 1, 2, 3, 4]
        y_positions = [0, 0, 0, 0, 0]
        
        colors = [AWS_COLORS["orange"], AWS_COLORS["orange"], AWS_COLORS["teal"], 
                 AWS_COLORS["green"], AWS_COLORS["blue"]]
        
        # Draw nodes
        for i, stage in enumerate(stages):
            fig.add_trace(go.Scatter(
                x=[x_positions[i]],
                y=[y_positions[i]],
                mode="markers+text",
                marker=dict(size=40, color=colors[i]),
                text=[stage],
                textposition="bottom center",
                hoverinfo="text",
                hovertext=[stage_descriptions[i]],
                name=stage
            ))
        
        # Draw arrows between nodes
        for i in range(len(stages) - 1):
            fig.add_annotation(
                x=x_positions[i] + 0.5,
                y=y_positions[i],
                ax=x_positions[i] + 0.1,
                ay=y_positions[i],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="#545B64"
            )
        
        # Configure layout
        fig.update_layout(
            showlegend=False,
            height=250,
            margin=dict(l=20, r=20, t=20, b=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>How Model Registry Works</h3>
        <p>The SageMaker Model Registry provides a central repository for organizing, cataloging, and managing machine learning models:</p>
        <ol>
            <li><b>Register Models:</b> Group related models in Model Package Groups</li>
            <li><b>Version Control:</b> Track iterations with automatic versioning</li>
            <li><b>Review and Approve:</b> Implement model governance with approval states</li>
            <li><b>Deploy:</b> Easily deploy approved models to production</li>
            <li><b>Monitor:</b> Track model performance and lineage</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Package Group demonstration
        st.subheader("Interactive Demo: Model Package Groups")
        
        # Create columns for selection and display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Model package group selection
            st.markdown("### Available Model Package Groups")
            
            # Create radio buttons for selecting model groups
            selected_group_name = st.radio(
                "Select a Model Package Group",
                options=[group["ModelPackageGroupName"] for group in st.session_state.registry_groups],
                index=0,
                key="model_group_selection"
            )
            
            # Find the selected group
            selected_group = next((g for g in st.session_state.registry_groups 
                                  if g["ModelPackageGroupName"] == selected_group_name), None)
            
            if selected_group:
                st.session_state.selected_group = selected_group
                
                st.markdown(f"**Description:** {selected_group['Description']}")
                st.markdown(f"**Domain:** {selected_group['Domain']}")
                st.markdown(f"**Created:** {selected_group['CreationTime'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Models:** {selected_group['ModelCount']}")
                
                if st.button("View Model Versions"):
                    # Generate model versions for the selected group
                    st.session_state.model_versions = generate_model_versions()
                    for version in st.session_state.model_versions:
                        version["ModelPackageGroupName"] = selected_group_name
                    
                    st.success(f"Loaded model versions for {selected_group_name}")
        
        with col2:
            st.markdown("### Model Package Group Details")
            
            if 'model_versions' in st.session_state and st.session_state.model_versions:
                versions = st.session_state.model_versions
                
                # Create a header with model count
                st.markdown(f"#### {len(versions)} Model Versions")
                
                # Create a version timeline chart
                version_data = {
                    "Version": [v["Version"] for v in versions],
                    "CreationTime": [v["CreationTime"] for v in versions],
                    "Status": [v["ModelApprovalStatus"] for v in versions],
                    "Framework": [v["Framework"] for v in versions]
                }
                
                version_df = pd.DataFrame(version_data)
                
                # Create a timeline chart
                status_colors = {
                    "Approved": AWS_COLORS["green"],
                    "Rejected": AWS_COLORS["red"],
                    "Pending": AWS_COLORS["orange"]
                }
                
                fig = px.scatter(
                    version_df,
                    x="CreationTime",
                    y="Version",
                    color="Status",
                    color_discrete_map=status_colors,
                    size=[10] * len(version_df),
                    hover_data=["Framework"],
                    labels={"CreationTime": "Creation Date", "Version": "Model Version"}
                )
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Creation Date",
                    yaxis_title="Version"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show model version details
                st.markdown("#### Model Version Details")
                
                selected_version = st.slider(
                    "Select model version to view details",
                    min_value=1,
                    max_value=len(versions),
                    value=len(versions),
                    step=1
                )
                
                # Get the selected model version
                model = next((v for v in versions if v["Version"] == selected_version), None)
                
                if model:
                    # Create a styled card for the model version
                    status_class = model["ModelApprovalStatus"].lower()
                    
                    st.markdown(f"""
                    <div class="model-version-card {status_class}">
                        <h4>Model Version {model['Version']} - {model["ModelApprovalStatus"]}</h4>
                        <p><strong>Framework:</strong> {model["Framework"]}</p>
                        <p><strong>Created:</strong> {model["CreationTime"].strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>Created By:</strong> {model["CreatedBy"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show model metrics
                    st.markdown("#### Model Metrics")
                    
                    metrics = model["Metrics"]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                    
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                        st.metric("AUC", f"{metrics['auc']:.4f}")
                    
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                        st.metric("Latency", f"{metrics['latency_ms']} ms")
                    
                    # Add an approval/rejection interface
                    st.markdown("#### Model Approval")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if model["ModelApprovalStatus"] == "Pending":
                            if st.button("Approve Model"):
                                # Update model status
                                for v in st.session_state.model_versions:
                                    if v["Version"] == model["Version"]:
                                        v["ModelApprovalStatus"] = "Approved"
                                st.success(f"Model version {model['Version']} approved!")
                                time.sleep(1)
                                st.rerun()
                    
                    with col2:
                        if model["ModelApprovalStatus"] == "Pending":
                            if st.button("Reject Model"):
                                # Update model status
                                for v in st.session_state.model_versions:
                                    if v["Version"] == model["Version"]:
                                        v["ModelApprovalStatus"] = "Rejected"
                                st.error(f"Model version {model['Version']} rejected!")
                                time.sleep(1)
                                st.rerun()
            else:
                st.info("Select a Model Package Group and click 'View Model Versions' to see model details")
                
                # Show a placeholder diagram
                st.image("https://docs.aws.amazon.com/sagemaker/latest/dg/images/model_registry/model_registry_diagram.png", 
                        caption="SageMaker Model Registry Structure", use_container_width=True)
        
        # Code examples
        st.markdown("### Implementing Model Registry")
        st.markdown("Here's how to register and manage models using SageMaker Model Registry API:")
        
        # Create model package group
        st.markdown("#### Create a Model Package Group")
        st.code('''
import boto3
import time

sagemaker_client = boto3.client('sagemaker')

# Create a model package group
model_package_group_name = "customer-churn-prediction"
model_package_group_input_dict = {
    "ModelPackageGroupName": model_package_group_name,
    "ModelPackageGroupDescription": "Models for predicting customer churn"
}

create_model_package_group_response = sagemaker_client.create_model_package_group(
    **model_package_group_input_dict
)

print(f"ModelPackageGroup Arn: {create_model_package_group_response['ModelPackageGroupArn']}")
        ''')
        
        # Register a model version
        st.markdown("#### Register a Model Version")
        st.code('''
# Training details
training_job_name = "customer-churn-training-job"

# Get training job info
training_job = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)

# Create model package
model_package_input_dict = {
    # Specify the model source
    "ModelPackageGroupName": "customer-churn-prediction",
    "ModelPackageDescription": "XGBoost customer churn prediction model",
    
    # Specify the model approval status
    "ModelApprovalStatus": "PendingManualApproval",
    
    # Specify inference specification
    "InferenceSpecification": {
        "Containers": [
            {
                "Image": training_job["AlgorithmSpecification"]["TrainingImage"],
                "ModelDataUrl": training_job["ModelArtifacts"]["S3ModelArtifacts"]
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    },
    
    # Add model metrics
    "ModelMetrics": {
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": "s3://bucket-name/metrics/evaluation.json"
            }
        }
    }
}

# Create the model package
create_model_package_response = sagemaker_client.create_model_package(
    **model_package_input_dict
)

model_package_arn = create_model_package_response["ModelPackageArn"]
print(f"ModelPackage Version ARN: {model_package_arn}")
        ''')
        
        # List and approve model versions
        st.markdown("#### List and Approve Model Versions")
        st.code('''
# List model package versions
list_model_packages_response = sagemaker_client.list_model_packages(
    ModelPackageGroupName="customer-churn-prediction",
    MaxResults=10,
    SortBy="CreationTime",
    SortOrder="Descending"
)

for model_package in list_model_packages_response["ModelPackageSummaryList"]:
    print(f"ModelPackageVersion: {model_package['ModelPackageVersion']}")
    print(f"ModelPackageArn: {model_package['ModelPackageArn']}")
    print(f"ModelApprovalStatus: {model_package['ModelApprovalStatus']}")
    print("---")

# Approve a model version
model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/customer-churn-prediction/3"

sagemaker_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved"
)
print(f"Model {model_package_arn} approved")
        ''')
        
        # Deploy approved model
        st.markdown("#### Deploy an Approved Model")
        st.code('''
from sagemaker import ModelPackage
import sagemaker

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Get the latest approved model
model_package_list = sagemaker_client.list_model_packages(
    ModelPackageGroupName="customer-churn-prediction",
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)

approved_package = model_package_list["ModelPackageSummaryList"][0]["ModelPackageArn"]

# Create model from the approved model package
model = ModelPackage(
    model_package_arn=approved_package,
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy the model to an endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="customer-churn-predictor"
)

print(f"Deployed model to endpoint: {predictor.endpoint_name}")
        ''')
        
    # TAB 2: MODEL REGISTRY STRUCTURE
    with tab2:
        st.header("Model Registry Structure")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            SageMaker Model Registry organizes models in a hierarchical structure that facilitates version tracking, governance, and deployment.
            
            **Key Components:**
            - **Model Package Groups:** Collections of related model versions
            - **Model Packages:** Individual model versions with metadata and artifacts
            - **Model Approval Status:** Indicates if a model is approved for production use
            - **Model Lineage:** Tracks relationships between datasets, algorithms, and models
            """)
        
        with col2:
            st.image("images/model_registry_components.png", 
                     caption="SageMaker Model Registry Components")
        
        # Interactive diagram
        st.subheader("Registry Component Hierarchy")
        
        # Create interactive diagram with Plotly
        fig = go.Figure()
        
        # Add nodes
        nodes = [
            {"name": "Model Registry", "x": 0, "y": 0, "size": 30, "color": AWS_COLORS["blue"]},
            {"name": "Model Package Group 1", "x": -2, "y": -2, "size": 25, "color": AWS_COLORS["teal"]},
            {"name": "Model Package Group 2", "x": 2, "y": -2, "size": 25, "color": AWS_COLORS["teal"]},
            {"name": "Model v1", "x": -3, "y": -4, "size": 20, "color": AWS_COLORS["orange"]},
            {"name": "Model v2", "x": -1, "y": -4, "size": 20, "color": AWS_COLORS["orange"]},
            {"name": "Model v1", "x": 1, "y": -4, "size": 20, "color": AWS_COLORS["orange"]},
            {"name": "Model v2", "x": 3, "y": -4, "size": 20, "color": AWS_COLORS["orange"]}
        ]
        
        # Add nodes to figure
        for node in nodes:
            fig.add_trace(go.Scatter(
                x=[node["x"]],
                y=[node["y"]],
                mode="markers+text",
                marker=dict(size=node["size"], color=node["color"]),
                text=[node["name"]],
                textposition="bottom center",
                name=node["name"]
            ))
        
        # Add connecting lines
        # Registry to package groups
        fig.add_shape(
            type="line", x0=0, y0=0, x1=-2, y1=-2,
            line=dict(color="gray", width=2)
        )
        
        fig.add_shape(
            type="line", x0=0, y0=0, x1=2, y1=-2,
            line=dict(color="gray", width=2)
        )
        
        # Package groups to models
        fig.add_shape(
            type="line", x0=-2, y0=-2, x1=-3, y1=-4,
            line=dict(color="gray", width=2)
        )
        
        fig.add_shape(
            type="line", x0=-2, y0=-2, x1=-1, y1=-4,
            line=dict(color="gray", width=2)
        )
        
        fig.add_shape(
            type="line", x0=2, y0=-2, x1=1, y1=-4,
            line=dict(color="gray", width=2)
        )
        
        fig.add_shape(
            type="line", x0=2, y0=-2, x1=3, y1=-4,
            line=dict(color="gray", width=2)
        )
        
        # Configure layout
        fig.update_layout(
            height=500,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, 4]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 1]),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Lineage section
        st.subheader("Model Lineage Tracking")
        
        st.markdown("""
        Model lineage tracking helps you understand the complete lifecycle of a model,
        from the datasets used for training to the deployed endpoints. This is crucial for
        reproducibility, auditing, and compliance.
        """)
        
        # Interactive lineage visualization
        lineage_graph = st.session_state.lineage_graph
        
        # Path highlighting options
        st.markdown("#### Explore Model Lineage Paths")
        
        path_options = {
            "None": None,
            "Data Processing Path": ["Raw Data", "Data Processing", "Processed Data"],
            "Model Training Path": ["Training Dataset", "Training Job", "Model v2"],
            "Model Evaluation Path": ["Validation Dataset", "Model Evaluation", "Approved Model"],
            "Complete Deployment Path": ["Raw Data", "Data Processing", "Processed Data", 
                                       "Feature Engineering", "Training Dataset", "Training Job", 
                                       "Model v2", "Model Evaluation", "Approved Model", "Production Endpoint"]
        }
        
        selected_path = st.selectbox("Highlight a lineage path:", list(path_options.keys()))
        
        # Update highlighted path in session state
        st.session_state.highlight_path = path_options[selected_path]
        
        # Draw the lineage graph
        lineage_fig = draw_model_lineage(lineage_graph, st.session_state.highlight_path)
        st.pyplot(lineage_fig)
        
        # Model metadata components
        st.subheader("Model Package Metadata")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            Each model package contains metadata that describes the model:
            
            - **Basic Information**
              - Model name and version
              - Description
              - Framework and framework version
              - Creation time and creator
            
            - **Approval Information**
              - Approval status (Pending/Approved/Rejected)
              - Approver
              - Approval time
              - Approval comments
            """)
        
        with col2:
            st.markdown("""
            - **Performance Metrics**
              - Accuracy, precision, recall, F1-score
              - Latency and throughput
              - Custom metrics
            
            - **Artifacts**
              - Model artifacts S3 location
              - Training data reference
              - Validation data reference
              - Inference container image
            """)
        
        # Create a sample model package metadata JSON
        sample_metadata = {
            "ModelPackageName": "customer-churn-prediction",
            "ModelPackageVersion": 3,
            "ModelPackageArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/customer-churn-prediction/3",
            "ModelPackageDescription": "XGBoost model for customer churn prediction",
            "ModelPackageStatus": "Completed",
            "ModelApprovalStatus": "Approved",
            "CreationTime": "2023-06-15T10:30:00Z",
            "CreatedBy": {
                "UserProfileArn": "arn:aws:sagemaker:us-west-2:123456789012:user-profile/domain-id/user-profile-name",
                "UserProfileName": "data-scientist-1",
                "DomainId": "domain-id"
            },
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest",
                        "ModelDataUrl": "s3://sagemaker-us-west-2-123456789012/models/customer-churn/model.tar.gz"
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"]
            },
            "ValidationSpecification": {
                "ValidationRole": "arn:aws:iam::123456789012:role/SageMakerValidationRole",
                "ValidationProfiles": [
                    {
                        "ProfileName": "validation-profile-1",
                        "TransformJobDefinition": {
                            "MaxConcurrentTransforms": 1,
                            "MaxPayloadInMB": 6,
                            "TransformInput": {
                                "DataSource": {
                                    "S3DataSource": {
                                        "S3DataType": "S3Prefix",
                                        "S3Uri": "s3://sagemaker-us-west-2-123456789012/validation/input"
                                    }
                                },
                                "ContentType": "text/csv"
                            },
                            "TransformOutput": {
                                "S3OutputPath": "s3://sagemaker-us-west-2-123456789012/validation/output",
                                "Accept": "text/csv",
                                "AssembleWith": "Line"
                            },
                            "TransformResources": {
                                "InstanceType": "ml.m5.xlarge",
                                "InstanceCount": 1
                            }
                        }
                    }
                ]
            },
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": "s3://sagemaker-us-west-2-123456789012/metrics/model-quality.json"
                    },
                    "Constraints": {
                        "ContentType": "application/json",
                        "S3Uri": "s3://sagemaker-us-west-2-123456789012/metrics/model-constraints.json"
                    }
                }
            },
            "LastModifiedTime": "2023-06-17T14:45:00Z",
            "LastModifiedBy": {
                "UserProfileArn": "arn:aws:sagemaker:us-west-2:123456789012:user-profile/domain-id/user-profile-name",
                "UserProfileName": "ml-engineer-2",
                "DomainId": "domain-id"
            },
            "ApprovalDescription": "Model meets all performance requirements for production deployment",
            "Domain": "Marketing",
            "Task": "Classification",
            "FrameworkVersion": "1.5-1",
            "Framework": "XGBoost",
            "ModelPackageGroupName": "customer-churn-prediction"
        }
        
        # Display the JSON with syntax highlighting
        st.markdown("#### Sample Model Package Metadata")
        st.json(sample_metadata)
        
        # Model Registry API examples
        st.subheader("Model Registry API Reference")
        
        # Create tabs for different API operations
        api_tab1, api_tab2, api_tab3 = st.tabs(["List Operations", "Create Operations", "Update Operations"])
        
        with api_tab1:
            st.markdown("#### List Model Registry Resources")
            st.code('''
# List model package groups
response = sagemaker_client.list_model_package_groups(
    NameContains="churn",  # Optional filter
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=10
)

# List model packages in a group
response = sagemaker_client.list_model_packages(
    ModelPackageGroupName="customer-churn-prediction",
    ModelApprovalStatus="Approved",  # Optional filter
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=10
)

# Search for model packages using custom filters
search_response = sagemaker_client.search(
    Resource="ModelPackage",
    SearchExpression={
        "Filters": [
            {
                "Name": "ModelPackageGroupName",
                "Value": "customer-churn-prediction"
            },
            {
                "Name": "ModelApprovalStatus",
                "Value": "Approved"
            }
        ]
    }
)

# Get detailed information about a model package
model_package_details = sagemaker_client.describe_model_package(
    ModelPackageName="arn:aws:sagemaker:us-west-2:123456789012:model-package/customer-churn-prediction/3"
)
            ''')
        
        with api_tab2:
            st.markdown("#### Create Model Registry Resources")
            st.code('''
# Create a model package group
response = sagemaker_client.create_model_package_group(
    ModelPackageGroupName="customer-churn-prediction",
    ModelPackageGroupDescription="Models for predicting customer churn",
    Tags=[
        {
            'Key': 'Project',
            'Value': 'CustomerRetention'
        },
        {
            'Key': 'Department',
            'Value': 'Marketing'
        }
    ]
)

# Create a model package
response = sagemaker_client.create_model_package(
    ModelPackageGroupName="customer-churn-prediction",
    ModelPackageDescription="XGBoost model for customer churn prediction",
    ModelApprovalStatus="PendingManualApproval",
    InferenceSpecification={
        'Containers': [
            {
                'Image': '123456789012.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
                'ModelDataUrl': 's3://sagemaker-us-west-2-123456789012/models/customer-churn/model.tar.gz'
            }
        ],
        'SupportedContentTypes': ['text/csv'],
        'SupportedResponseMIMETypes': ['text/csv']
    },
    ModelMetrics={
        'ModelQuality': {
            'Statistics': {
                'ContentType': 'application/json',
                'S3Uri': 's3://sagemaker-us-west-2-123456789012/metrics/model-quality.json'
            }
        }
    },
    SourceAlgorithmSpecification={
        'SourceAlgorithms': [
            {
                'AlgorithmName': 'xgboost',
                'ModelDataSource': {
                    'S3DataSource': {
                        'S3Uri': 's3://sagemaker-us-west-2-123456789012/models/customer-churn/model.tar.gz',
                        'S3DataType': 'S3Object'
                    }
                }
            }
        ]
    },
    ValidationSpecification={
        'ValidationRole': 'arn:aws:iam::123456789012:role/SageMakerValidationRole',
        'ValidationProfiles': [
            {
                'ProfileName': 'ValidationProfile1',
                'TransformJobDefinition': {
                    'MaxConcurrentTransforms': 1,
                    'MaxPayloadInMB': 6,
                    'TransformInput': {
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': 's3://sagemaker-us-west-2-123456789012/validation-data/'
                            }
                        },
                        'ContentType': 'text/csv',
                        'SplitType': 'Line'
                    },
                    'TransformOutput': {
                        'S3OutputPath': 's3://sagemaker-us-west-2-123456789012/validation-output/',
                        'Accept': 'text/csv',
                        'AssembleWith': 'Line'
                    },
                    'TransformResources': {
                        'InstanceType': 'ml.m5.xlarge',
                        'InstanceCount': 1
                    }
                }
            }
        ]
    },
    Tags=[
        {
            'Key': 'ModelVersion',
            'Value': '3.0'
        }
    ]
)
            ''')
        
        with api_tab3:
            st.markdown("#### Update Model Registry Resources")
            st.code('''
# Update model package group
response = sagemaker_client.update_model_package_group(
    ModelPackageGroupName="customer-churn-prediction",
    ModelPackageGroupDescription="Updated description for customer churn prediction models"
)

# Update model package approval status
response = sagemaker_client.update_model_package(
    ModelPackageName="arn:aws:sagemaker:us-west-2:123456789012:model-package/customer-churn-prediction/3",
    ModelApprovalStatus="Approved",
    ApprovalDescription="Model meets performance requirements for production use"
)

# Delete model package group (will fail if it contains model packages)
response = sagemaker_client.delete_model_package_group(
    ModelPackageGroupName="customer-churn-prediction"
)

# Delete model package
response = sagemaker_client.delete_model_package(
    ModelPackageName="arn:aws:sagemaker:us-west-2:123456789012:model-package/customer-churn-prediction/3"
)
            ''')
        
        # Best practices
        st.subheader("Best Practices for Model Registry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Organization Best Practices
            
            - **Group related models** into logical model package groups
            - **Use consistent naming conventions** for models and groups
            - **Add meaningful descriptions** to all registry resources
            - **Use tags** to categorize models by project, team, or purpose
            - **Define approval workflows** with clear criteria
            """)
        
        with col2:
            st.markdown("""
            #### Governance Best Practices
            
            - **Document model cards** with intended use cases and limitations
            - **Record comprehensive metrics** for each model version
            - **Implement approval stages** (dev → staging → production)
            - **Track model lineage** for reproducibility and auditing
            - **Integrate with CI/CD pipelines** for automated testing
            """)
        
        # Governance workflow
        st.subheader("Model Governance Workflow Example")
        
        # Create a timeline visualization
        timeline_data = [
            {"stage": "Development", "description": "Model is built and undergoes initial testing", "status": "PendingManualApproval"},
            {"stage": "Testing", "description": "Model undergoes validation with test datasets", "status": "PendingManualApproval"},
            {"stage": "Review", "description": "ML team reviews model metrics and documentation", "status": "PendingManualApproval"},
            {"stage": "Approval", "description": "Business stakeholders approve model for production", "status": "Approved"},
            {"stage": "Deployment", "description": "Model is deployed to production environment", "status": "Approved"},
            {"stage": "Monitoring", "description": "Model performance is continuously monitored", "status": "Approved"}
        ]
        
        # Create timeline visualization with Plotly
        fig = go.Figure()
        
        # Add timeline nodes
        y_pos = 0
        status_colors = {
            "PendingManualApproval": AWS_COLORS["orange"],
            "Approved": AWS_COLORS["green"],
            "Rejected": AWS_COLORS["red"]
        }
        
        for i, stage in enumerate(timeline_data):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[y_pos],
                mode="markers+text",
                marker=dict(size=30, color=status_colors[stage["status"]]),
                text=[stage["stage"]],
                textposition="bottom center",
                hoverinfo="text",
                hovertext=[stage["description"]],
                name=stage["stage"]
            ))
        
        # Add connecting lines
        for i in range(len(timeline_data) - 1):
            fig.add_shape(
                type="line",
                x0=i,
                y0=y_pos,
                x1=i + 1,
                y1=y_pos,
                line=dict(color="gray", width=2, dash="dot")
            )
        
        # Configure layout
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add descriptions below timeline
        for i, stage in enumerate(timeline_data):
            st.markdown(f"**{stage['stage']}:** {stage['description']}")
            
        # Integration with other SageMaker services
        st.subheader("Integration with SageMaker MLOps")
        
        st.markdown("""
        SageMaker Model Registry integrates with other SageMaker services to create a comprehensive MLOps solution:
        
        - **SageMaker Pipelines:** Automate model building and registration workflows
        - **SageMaker Projects:** Create end-to-end ML solutions with CI/CD templates
        - **SageMaker Model Monitor:** Monitor deployed models for quality and drift
        - **SageMaker Lineage Tracking:** Track model artifacts and relationships
        - **SageMaker Endpoints:** Deploy approved models to production endpoints
        """)
        
        # Final diagram showing the complete MLOps cycle
        st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2020/12/01/ML-4319-1.jpg",
                caption="SageMaker MLOps with Model Registry", use_container_width=True)

    # Add footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
