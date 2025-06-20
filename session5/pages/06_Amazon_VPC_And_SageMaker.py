import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import random
import json
from io import BytesIO
import base64
import uuid
import time
import re
import os
import requests
from streamlit_lottie import st_lottie

# Initialize session state variables
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "knowledge_check_answers" not in st.session_state:
        st.session_state.knowledge_check_answers = [None] * 5
    if "knowledge_check_score" not in st.session_state:
        st.session_state.knowledge_check_score = 0
    if "knowledge_check_submitted" not in st.session_state:
        st.session_state.knowledge_check_submitted = False
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = 0

# Function to reset session state
def reset_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.knowledge_check_answers = [None] * 5
    st.session_state.knowledge_check_score = 0
    st.session_state.knowledge_check_submitted = False
    st.success("Session has been reset!")

# Function to load Lottie animations from URL or file
def load_lottie(url_or_path):
    if url_or_path.startswith('http'):
        try:
            r = requests.get(url_or_path)
            if r.status_code != 200:
                return None
            return r.json()
        except:
            return None
    else:
        try:
            with open(url_or_path, "r") as file:
                return json.load(file)
        except:
            return None

# Function to generate a sample VPC configuration code
def generate_vpc_config_code():
    return '''
import boto3

def create_sagemaker_vpc_endpoint():
    """Creates VPC endpoints for SageMaker API and Runtime"""
    
    ec2_client = boto3.client('ec2', region_name='us-east-1')
    
    # Create SageMaker API endpoint
    sm_api_response = ec2_client.create_vpc_endpoint(
        VpcEndpointType='Interface',
        VpcId='vpc-12345678',
        ServiceName='com.amazonaws.us-east-1.sagemaker.api',
        SubnetIds=['subnet-abcdef12', 'subnet-34567890'],
        SecurityGroupIds=['sg-11223344'],
        PrivateDnsEnabled=True
    )
    
    # Create SageMaker Runtime endpoint
    sm_runtime_response = ec2_client.create_vpc_endpoint(
        VpcEndpointType='Interface',
        VpcId='vpc-12345678',
        ServiceName='com.amazonaws.us-east-1.sagemaker.runtime',
        SubnetIds=['subnet-abcdef12', 'subnet-34567890'],
        SecurityGroupIds=['sg-11223344'],
        PrivateDnsEnabled=True
    )
    
    print(f"Created SageMaker API endpoint: {sm_api_response['VpcEndpoint']['VpcEndpointId']}")
    print(f"Created SageMaker Runtime endpoint: {sm_runtime_response['VpcEndpoint']['VpcEndpointId']}")

if __name__ == "__main__":
    create_sagemaker_vpc_endpoint()
    '''

# Function to generate SageMaker endpoint configuration code
def generate_endpoint_config_code():
    return '''
import boto3

def deploy_model_to_vpc_endpoint():
    """Deploy a SageMaker model using VPC configuration"""
    
    sm_client = boto3.client('sagemaker')
    
    # Create model with VPC config
    model_response = sm_client.create_model(
        ModelName='my-vpc-isolated-model',
        ExecutionRoleArn='arn:aws:iam::123456789012:role/SageMakerExecutionRole',
        PrimaryContainer={
            'Image': '123456789012.dkr.ecr.us-east-1.amazonaws.com/my-model-image:latest',
            'ModelDataUrl': 's3://my-bucket/model.tar.gz'
        },
        VpcConfig={
            'SecurityGroupIds': ['sg-11223344'],
            'Subnets': ['subnet-abcdef12', 'subnet-34567890']
        }
    )
    
    # Create endpoint config
    endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName='my-vpc-endpoint-config',
        ProductionVariants=[{
            'VariantName': 'PrimaryVariant',
            'ModelName': 'my-vpc-isolated-model',
            'InstanceType': 'ml.m5.xlarge',
            'InitialInstanceCount': 1
        }]
    )
    
    # Create endpoint
    endpoint_response = sm_client.create_endpoint(
        EndpointName='my-vpc-isolated-endpoint',
        EndpointConfigName='my-vpc-endpoint-config'
    )
    
    print(f"Endpoint creation initiated: {endpoint_response['EndpointArn']}")
    print("The endpoint will be accessible only within the VPC")

if __name__ == "__main__":
    deploy_model_to_vpc_endpoint()
    '''

# Function to generate SageMaker Studio VPC only mode code
def generate_studio_vpc_code():
    return '''
import boto3

def create_sagemaker_studio_domain_vpc_only():
    """Create a SageMaker Studio domain with VPC only mode"""
    
    sm_client = boto3.client('sagemaker')
    
    # Create domain with VPC only access
    domain_response = sm_client.create_domain(
        DomainName='vpc-only-studio-domain',
        AuthMode='IAM',
        DefaultUserSettings={
            'ExecutionRole': 'arn:aws:iam::123456789012:role/SageMakerStudioExecutionRole',
            'SecurityGroups': ['sg-11223344']
        },
        SubnetIds=['subnet-abcdef12', 'subnet-34567890'],
        VpcId='vpc-12345678',
        AppNetworkAccessType='VpcOnly',  # This is the key setting for VPC only mode
        DefaultSpaceSettings={
            'ExecutionRole': 'arn:aws:iam::123456789012:role/SageMakerStudioExecutionRole',
            'SecurityGroups': ['sg-11223344']
        }
    )
    
    print(f"Created VPC-only Studio domain: {domain_response['DomainId']}")
    
    # Create user profile
    user_response = sm_client.create_user_profile(
        DomainId=domain_response['DomainId'],
        UserProfileName='vpc-only-user',
        UserSettings={
            'ExecutionRole': 'arn:aws:iam::123456789012:role/SageMakerStudioExecutionRole'
        }
    )
    
    print(f"Created user profile: {user_response['UserProfileArn']}")

if __name__ == "__main__":
    create_sagemaker_studio_domain_vpc_only()
    '''

# Create diagrams for visualization
def generate_vpc_diagram():
    # Create a Plotly diagram showing VPC interaction with SageMaker
    nodes = [
        dict(name="Your VPC", x=0, y=0),
        dict(name="Private Subnet", x=1, y=0),
        dict(name="SageMaker API", x=2, y=0.5),
        dict(name="SageMaker Runtime", x=2, y=-0.5),
        dict(name="Interface Endpoint", x=1, y=0.5),
        dict(name="EC2 Instance", x=0.5, y=-0.5)
    ]
    
    edges = [
        dict(source=0, target=1, value=1),
        dict(source=1, target=4, value=1),
        dict(source=4, target=2, value=1),
        dict(source=1, target=5, value=1),
        dict(source=5, target=4, value=1),
        dict(source=4, target=3, value=1)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add nodes
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node["x"]], 
            y=[node["y"]],
            mode="markers+text",
            marker=dict(size=30, color="#FF9900", line=dict(width=2, color="#232F3E")),
            text=[node["name"]],
            textposition="bottom center",
            name=node["name"]
        ))
    
    # Add edges
    for edge in edges:
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        fig.add_trace(go.Scatter(
            x=[source["x"], target["x"]],
            y=[source["y"], target["y"]],
            mode="lines",
            line=dict(width=2, color="#232F3E"),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="SageMaker VPC Network Connection",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        height=500,
        plot_bgcolor='#F8F9FA'
    )
    
    return fig

# Create SageMaker endpoint diagram
def generate_endpoint_diagram():
    # Create a flow diagram showing VPC endpoint connections
    nodes = [
        dict(name="Your Application", x=0, y=0),
        dict(name="VPC Boundary", x=1, y=0),
        dict(name="Interface\nEndpoint", x=2, y=0),
        dict(name="AWS\nPrivateLink", x=3, y=0),
        dict(name="SageMaker\nEndpoint", x=4, y=0)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add nodes with different shapes/colors
    colors = ["#FF9900", "#232F3E", "#00A1C9", "#1D8102", "#C94700"]
    
    for i, node in enumerate(nodes):
        fig.add_trace(go.Scatter(
            x=[node["x"]], 
            y=[node["y"]],
            mode="markers+text",
            marker=dict(size=40, color=colors[i], symbol="circle"),
            text=[node["name"]],
            textposition="bottom center",
            name=node["name"]
        ))
    
    # Add arrows between nodes
    for i in range(len(nodes)-1):
        fig.add_trace(go.Scatter(
            x=[nodes[i]["x"], nodes[i+1]["x"]],
            y=[nodes[i]["y"], nodes[i+1]["y"]],
            mode="lines+markers",
            marker=dict(size=10, color="#232F3E", symbol="arrow-right"),
            line=dict(width=2, color="#232F3E"),
            showlegend=False
        ))
    
    # Add annotations for security features
    annotations = [
        dict(x=1.5, y=0.3, text="Secure Connection", showarrow=False,
             font=dict(color="#1D8102", size=12)),
        dict(x=3.5, y=0.3, text="AWS Network Only", showarrow=False,
             font=dict(color="#1D8102", size=12)),
        dict(x=2.5, y=-0.3, text="No Internet Exposure", showarrow=False,
             font=dict(color="#1D8102", size=12))
    ]
    
    # Update layout
    fig.update_layout(
        title="SageMaker VPC Interface Endpoint Architecture",
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=400,
        plot_bgcolor='#F8F9FA'
    )
    
    return fig

# Create SageMaker Studio VPC diagram
def generate_studio_vpc_diagram():
    # Create a diagram showing Studio in private VPC
    
    # Simplified architecture for visualization
    fig = go.Figure()
    
    # Define the components
    components = [
        {"name": "User", "x": 0, "y": 0, "color": "#232F3E"},
        {"name": "VPC Boundary", "x": 2, "y": 0, "color": "#232F3E", "width": 4, "height": 4},
        {"name": "Private Subnet", "x": 2.5, "y": 0, "color": "#FF9900", "width": 3, "height": 3},
        {"name": "SageMaker Studio", "x": 2.5, "y": 0, "color": "#00A1C9"},
        {"name": "Amazon EFS", "x": 4, "y": 1, "color": "#C94700"},
        {"name": "VPC Endpoint (S3)", "x": 1.5, "y": 1.5, "color": "#1D8102"},
        {"name": "VPC Endpoint (API)", "x": 1.5, "y": 0.5, "color": "#1D8102"},
        {"name": "VPC Endpoint (Runtime)", "x": 1.5, "y": -0.5, "color": "#1D8102"}
    ]
    
    # Draw boxes for VPC and subnet
    fig.add_shape(
        type="rect",
        x0=0.5, y0=-2, x1=4.5, y1=2,
        line=dict(color="#232F3E", width=2),
        fillcolor="rgba(35, 47, 62, 0.1)",
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        x0=1, y0=-1.5, x1=4, y1=1.5,
        line=dict(color="#FF9900", width=2),
        fillcolor="rgba(255, 153, 0, 0.1)",
        layer="below"
    )
    
    # Add components as nodes
    for comp in components:
        if comp["name"] in ["VPC Boundary", "Private Subnet"]:
            continue
            
        fig.add_trace(go.Scatter(
            x=[comp["x"]], 
            y=[comp["y"]],
            mode="markers+text",
            marker=dict(size=30, color=comp["color"]),
            text=[comp["name"]],
            textposition="bottom center",
            name=comp["name"]
        ))
    
    # Add connections
    connections = [
        (0, 6), (6, 3), (0, 5), (5, 3), (0, 7), (7, 3), (3, 4)
    ]
    
    for source, target in connections:
        fig.add_trace(go.Scatter(
            x=[components[source]["x"], components[target]["x"]],
            y=[components[source]["y"], components[target]["y"]],
            mode="lines",
            line=dict(width=1.5, color="#232F3E", dash="dot"),
            showlegend=False
        ))
    
    # Add labels
    fig.add_annotation(
        x=0.75, y=2.1,
        text="Customer VPC",
        showarrow=False,
        font=dict(size=14, color="#232F3E")
    )
    
    fig.add_annotation(
        x=1.25, y=1.6,
        text="Private Subnet",
        showarrow=False,
        font=dict(size=12, color="#FF9900")
    )
    
    # Update layout
    fig.update_layout(
        title="SageMaker Studio in Private VPC Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
        showlegend=False,
        height=600,
        plot_bgcolor='#F8F9FA'
    )
    
    return fig

# Questions for the knowledge check
def get_knowledge_check_questions():
    return [
        {
            "question": "What AWS service allows you to connect directly to SageMaker API and Runtime from within your VPC?",
            "options": [
                "AWS Direct Connect", 
                "AWS VPN", 
                "VPC Interface Endpoints (AWS PrivateLink)", 
                "Internet Gateway"
            ],
            "correct": 2
        },
        {
            "question": "When configuring SageMaker Studio for VPC only access, which setting must you specify?",
            "options": [
                "NetworkAccessType='Private'", 
                "AppNetworkAccessType='VpcOnly'", 
                "VPCOnlyMode=True", 
                "NetworkIsolation=True"
            ],
            "correct": 1
        },
        {
            "question": "What is the benefit of using VPC Interface Endpoints with SageMaker?",
            "options": [
                "It makes SageMaker endpoints faster", 
                "It reduces the cost of SageMaker", 
                "It keeps traffic within AWS network without internet exposure", 
                "It provides more compute resources for your models"
            ],
            "correct": 2
        },
        {
            "question": "Which AWS service powers VPC Interface Endpoints?",
            "options": [
                "AWS PrivateLink", 
                "AWS Direct Connect", 
                "AWS Global Accelerator", 
                "AWS Transit Gateway"
            ],
            "correct": 0
        },
        {
            "question": "What resources do you need to define when setting up a SageMaker Studio domain with VPC only mode?",
            "options": [
                "Only subnet IDs", 
                "VPC ID, subnet IDs, and security groups", 
                "Only security groups", 
                "Internet Gateway ID"
            ],
            "correct": 1
        }
    ]

# Function to render the home page
def render_home():
       
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## What is Amazon VPC?
        
        Amazon Virtual Private Cloud (Amazon VPC) is a service that lets you launch AWS resources in a logically 
        isolated virtual network that you define. It gives you complete control over your virtual networking 
        environment, including selection of your own IP address ranges, creation of subnets, and configuration 
        of route tables and network gateways.
        
        ## Integration with SageMaker
        
        When working with Amazon SageMaker, you can use Amazon VPC to add an additional layer of security and isolation to your machine learning workflows.
        This integration enables you to:
        
        - Securely access resources within your VPC
        - Protect your ML models from unauthorized access
        - Control network traffic to and from your SageMaker resources
        - Comply with strict security and regulatory requirements
        """)
        
        with st.expander("Why use SageMaker with VPC?"):
            st.markdown("""
            ### Key benefits of using SageMaker with VPC:
            
            1. **Enhanced Security**: Keep your data and models private by restricting internet access
            2. **Network Isolation**: Control which resources can communicate with your SageMaker endpoints
            3. **Compliance**: Meet regulatory requirements for data privacy and network security
            4. **Data Protection**: Prevent sensitive data from being exposed outside your VPC
            5. **Secure Access**: Access SageMaker API and Runtime without going through the public internet
            """)
    
    with col2:
        # Add a lottie animation for cloud security 
        security_animation = load_lottie("https://assets1.lottiefiles.com/packages/lf20_hg7lougv.json")
        if security_animation:
            st_lottie(security_animation, height=300, key="security_animation")
        
        st.markdown("""
        ### Default vs. VPC Only Mode
        
        **Default Mode**:
        - Internet access enabled
        - SageMaker manages network configuration
        - Simpler to set up
        
        **VPC Only Mode**:
        - No direct internet access
        - You control all network traffic
        - Additional security
        - Requires proper VPC configuration
        """)
    
    st.markdown("### Amazon VPC and SageMaker Architecture")
    
    # Display a simplified VPC diagram
    vpc_diagram = generate_vpc_diagram()
    st.plotly_chart(vpc_diagram, use_container_width=True)
    
    st.markdown("""
    ### Setting up VPC for SageMaker - Basic Configuration
    
    Here's a simple example of how to configure a VPC for use with SageMaker:
    """)
    
    with st.expander("Show VPC Configuration Code"):
        st.code(generate_vpc_config_code(), language="python")
    
    # Add information about key resources
    st.markdown("### Key Resources Required")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **VPC Components**
        - VPC with private CIDR block
        - At least two private subnets
        - Security groups
        """)
    
    with col2:
        st.markdown("""
        **VPC Endpoints Required**
        - SageMaker API
        - SageMaker Runtime
        - Amazon S3
        - Amazon CloudWatch (optional)
        """)
    
    with col3:
        st.markdown("""
        **Network Requirements**
        - NAT Gateway (for outbound traffic)
        - Route tables
        - Network ACLs (optional)
        """)

# Function to render the endpoints page
def render_endpoints():
    st.title("SageMaker Endpoints within VPC Network")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## What are VPC Interface Endpoints?
        
        VPC Interface Endpoints powered by AWS PrivateLink allow you to connect to AWS services using private IP addresses. 
        This means traffic between your VPC and SageMaker doesn't leave the Amazon network.
        
        ### Benefits:
        
        - No internet gateway, NAT device, or VPN connection required
        - Increased security as traffic doesn't traverse the public internet
        - Simplified network architecture
        - Better reliability and lower latency
        """)
    
    with col2:
        # Display a box highlighting the key PrivateLink features
        st.markdown("""
        <div class="info-box">
        <h3>AWS PrivateLink Features</h3>
        <ul>
        <li>Private connectivity between VPCs and services</li>
        <li>No exposure to the public internet</li>
        <li>Service traffic stays within AWS network</li>
        <li>Highly available and scalable</li>
        <li>Support for DNS resolution</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display the endpoint connectivity diagram
    st.subheader("SageMaker VPC Endpoint Architecture")
    endpoint_diagram = generate_endpoint_diagram()
    st.plotly_chart(endpoint_diagram, use_container_width=True)
    
    # Create columns for required endpoints
    st.subheader("Required SageMaker VPC Endpoints")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>SageMaker API</h3>
        <p>Endpoint: <code>com.amazonaws.region.sagemaker.api</code></p>
        <p>Used for all SageMaker API operations (create model, create endpoint, etc.)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>SageMaker Runtime</h3>
        <p>Endpoint: <code>com.amazonaws.region.sagemaker.runtime</code></p>
        <p>Used for invoking endpoints for real-time inference</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
        <h3>Amazon S3</h3>
        <p>Endpoint: <code>com.amazonaws.region.s3</code></p>
        <p>Used for model artifacts, data storage and access</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add code example for deploying a model to VPC
    st.subheader("Deploying a SageMaker Model with VPC Configuration")
    
    with st.expander("Show Endpoint VPC Configuration Code"):
        st.code(generate_endpoint_config_code(), language="python")
    
    # Add interactive example showing network flow
    st.subheader("Interactive VPC Endpoint Network Flow")
    
    # Create tabs for showing different scenarios
    network_tabs = st.tabs(["Default Setup (Internet)", "VPC Endpoint Setup (Private)"])
    
    with network_tabs[0]:
        st.markdown("""
        ### Default SageMaker Endpoint (Internet Access)
        
        In the default setup, traffic flows through the public internet:
        
        1. Client application sends a request to SageMaker endpoint
        2. Request travels through the internet
        3. Potential security vulnerabilities and higher latency
        4. Response returns via the internet
        
        <div class="warning-box">
        <strong>Security Concerns:</strong><br>
        - Traffic exposed to the public internet<br>
        - Requires internet gateway and public subnets<br>
        - May not meet strict regulatory requirements
        </div>
        """, unsafe_allow_html=True)
        
        # Simplified diagram for internet-based setup
        st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2019/01/18/sagemaker-endpoint-1.gif", 
                 caption="Default SageMaker Endpoint Network Flow")
    
    with network_tabs[1]:
        st.markdown("""
        ### VPC Interface Endpoint Setup (Private)
        
        With VPC endpoints, traffic stays within the AWS network:
        
        1. Client application in your VPC sends request to the interface endpoint
        2. Traffic stays within the AWS network using private IP addresses
        3. No internet exposure, enhanced security, lower latency
        4. Response returns via the same private network path
        
        <div class="info-box">
        <strong>Security Benefits:</strong><br>
        - Traffic never leaves the AWS network<br>
        - No internet gateway required<br>
        - Uses private IP addresses<br>
        - Meets strict security and compliance requirements
        </div>
        """, unsafe_allow_html=True)
        
        # Simplified diagram for VPC endpoint setup
        st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2019/01/18/sagemaker-endpoint-2.gif", 
                 caption="VPC Endpoint-Based SageMaker Network Flow")
    
    # Add best practices
    st.subheader("Best Practices for VPC Interface Endpoints")
    
    best_practices = [
        "Create VPC endpoints in multiple Availability Zones for high availability",
        "Use security groups to control access to your VPC endpoints",
        "Enable private DNS for a seamless experience without code changes",
        "Configure endpoint policies to further restrict access to specific API actions",
        "Use CloudTrail to audit all API calls through the VPC endpoints"
    ]
    
    for i, practice in enumerate(best_practices):
        st.markdown(f"**{i+1}. {practice}**")

# Function to render the studio page
def render_studio():
    st.title("SageMaker Studio in a Private VPC")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## SageMaker Studio VPC Only Mode
        
        SageMaker Studio can be configured to operate entirely within your VPC without internet access.
        This provides a secure environment for your machine learning development workflow.
        
        ### Key Components:
        
        - SageMaker Studio domain with VPC only configuration
        - Private subnets in your VPC
        - VPC endpoints for required AWS services
        - Security groups to control access
        - Amazon EFS filesystem for storage
        """)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>VPC Only Mode vs Direct Internet Access</h3>
        <p><strong>Direct Internet Access:</strong> Default mode, simpler setup but less secure</p>
        <p><strong>VPC Only Mode:</strong> Enhanced security, no internet access, requires proper VPC endpoint setup</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display architecture diagram
    st.subheader("Architecture: SageMaker Studio in Private VPC")
    studio_vpc_diagram = generate_studio_vpc_diagram()
    st.plotly_chart(studio_vpc_diagram, use_container_width=True)
    
    # Required VPC endpoints for Studio
    st.subheader("Required VPC Endpoints for SageMaker Studio")
    
    endpoints = {
        "SageMaker API": "com.amazonaws.region.sagemaker.api",
        "SageMaker Runtime": "com.amazonaws.region.sagemaker.runtime",
        "Amazon S3": "com.amazonaws.region.s3",
        "CloudWatch": "com.amazonaws.region.logs",
        "CloudWatch Events": "com.amazonaws.region.events",
        "AWS STS": "com.amazonaws.region.sts",
        "ECR API": "com.amazonaws.region.ecr.api",
        "ECR DKR": "com.amazonaws.region.ecr.dkr"
    }
    
    col1, col2 = st.columns(2)
    
    for i, (name, endpoint) in enumerate(endpoints.items()):
        with col1 if i < len(endpoints) / 2 else col2:
            st.markdown(f"""
            <div class="status-card status-1">
            <strong>{name}</strong><br>
            <code>{endpoint}</code>
            </div>
            """, unsafe_allow_html=True)
    
    # Add code example
    st.subheader("Creating a SageMaker Studio Domain with VPC Only Mode")
    
    with st.expander("Show Studio VPC Configuration Code"):
        st.code(generate_studio_vpc_code(), language="python")
    
    # Add interactive example
    st.subheader("Network Access Configuration")
    
    option = st.selectbox(
        "Select Network Access Type",
        ["Direct Internet Access", "VPC Only"]
    )
    
    if option == "Direct Internet Access":
        st.markdown("""
        <div class="card">
        <h3>Direct Internet Access Configuration</h3>
        <p>With this configuration:</p>
        <ul>
        <li>SageMaker Studio can access the internet directly</li>
        <li>Easier to set up and use</li>
        <li>No need to configure VPC endpoints</li>
        <li>Data and code can be downloaded from public sites</li>
        <li>Less secure as internet access is allowed</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.code('''
# Direct Internet Access configuration
{
    "DomainName": "example-studio-domain",
    "AuthMode": "IAM",
    "DefaultUserSettings": {
        "ExecutionRole": "arn:aws:iam::123456789012:role/SageMakerStudioExecutionRole"
    },
    "SubnetIds": ["subnet-abcdef12", "subnet-34567890"],
    "VpcId": "vpc-12345678",
    "AppNetworkAccessType": "PublicInternetOnly"  # This allows internet access
}
        ''', language="json")
    else:
        st.markdown("""
        <div class="card">
        <h3>VPC Only Configuration</h3>
        <p>With this configuration:</p>
        <ul>
        <li>SageMaker Studio cannot access the internet</li>
        <li>Requires proper VPC endpoint setup</li>
        <li>Higher security as there's no internet exposure</li>
        <li>Data and code must come from within the VPC or via VPC endpoints</li>
        <li>More complex to set up but better for secure environments</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.code('''
# VPC Only configuration
{
    "DomainName": "example-studio-domain",
    "AuthMode": "IAM",
    "DefaultUserSettings": {
        "ExecutionRole": "arn:aws:iam::123456789012:role/SageMakerStudioExecutionRole",
        "SecurityGroups": ["sg-11223344"]
    },
    "SubnetIds": ["subnet-abcdef12", "subnet-34567890"],
    "VpcId": "vpc-12345678",
    "AppNetworkAccessType": "VpcOnly",  # This restricts to VPC only
    "DefaultSpaceSettings": {
        "ExecutionRole": "arn:aws:iam::123456789012:role/SageMakerStudioExecutionRole",
        "SecurityGroups": ["sg-11223344"]
    }
}
        ''', language="json")
    
    # Add checklist for VPC Only setup
    if option == "VPC Only":
        st.subheader("VPC Only Mode Checklist")
        
        checklist_items = [
            "Create a VPC with private subnets",
            "Configure necessary VPC endpoints",
            "Set up proper security groups",
            "Configure route tables for VPC endpoints",
            "Enable private DNS for endpoints",
            "Test connectivity to required AWS services",
            "Configure NAT Gateway if outbound internet is needed for specific use cases"
        ]
        
        for i, item in enumerate(checklist_items):
            st.checkbox(item, key=f"checklist_{i}")
    
    # Best practices
    st.subheader("Best Practices for SageMaker Studio in VPC Only Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Security
        
        - Use the principle of least privilege for IAM roles
        - Regularly audit security groups and network ACLs
        - Use VPC Flow Logs to monitor network traffic
        - Enable AWS CloudTrail for API auditing
        - Create separate subnets for different environments
        """)
    
    with col2:
        st.markdown("""
        ### Performance & Cost
        
        - Place VPC endpoints in the same AZ as Studio
        - Use interface endpoints for frequently accessed services
        - Consider Gateway endpoints (S3, DynamoDB) for cost savings
        - Monitor endpoint usage for optimization
        - Use cached repositories for common packages
        """)

# Function to render the knowledge check page
def render_knowledge_check():
    st.title("Knowledge Check: SageMaker VPC Network Isolation")
    
    questions = get_knowledge_check_questions()
    
    if st.session_state.knowledge_check_submitted:
        # Show results
        st.header(f"Your Score: {st.session_state.knowledge_check_score}/{len(questions)}")
        
        for i, question in enumerate(questions):
            st.subheader(f"Question {i+1}: {question['question']}")
            user_answer = st.session_state.knowledge_check_answers[i]
            correct_answer = question['correct']
            
            if user_answer == correct_answer:
                st.success(f"✓ Your answer: {question['options'][user_answer]} (Correct)")
            else:
                st.error(f"✗ Your answer: {question['options'][user_answer] if user_answer is not None else 'Not answered'}")
                st.info(f"Correct answer: {question['options'][correct_answer]}")
        
        if st.button("Retake Quiz"):
            st.session_state.knowledge_check_answers = [None] * len(questions)
            st.session_state.knowledge_check_score = 0
            st.session_state.knowledge_check_submitted = False
            st.experimental_rerun()
    else:
        # Show questions
        for i, question in enumerate(questions):
            st.subheader(f"Question {i+1}: {question['question']}")
            st.session_state.knowledge_check_answers[i] = st.radio(
                f"Select an answer for question {i+1}:",
                options=range(len(question['options'])),
                format_func=lambda x: question['options'][x],
                index=None,
                key=f"question_{i}"
            )
        
        if st.button("Submit Answers"):
            # Calculate score
            score = 0
            for i, question in enumerate(questions):
                if st.session_state.knowledge_check_answers[i] == question['correct']:
                    score += 1
            
            st.session_state.knowledge_check_score = score
            st.session_state.knowledge_check_submitted = True
            st.experimental_rerun()

# Main function
def main():


    # Initialize session state
    init_session_state()
    
    # Set page config and title
    st.set_page_config(
        page_title="SageMaker VPC Network Isolation",
        page_icon="🛡️",
        layout="wide"
    )
    
        
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
    </style>
    
    """, unsafe_allow_html=True)    
    
    
    st.title("Amazon VPC and SageMaker")
    
    # Sidebar
    # Session management
    st.sidebar.subheader("Session Management")
    st.sidebar.info(f"User ID: {st.session_state.session_id}")
    if st.sidebar.button("Reset Session"):
        reset_session()
    
    st.sidebar.divider()
    # Resources section
    with st.sidebar.expander("About this application", expanded=False):
        st.subheader("Additional Resources")
        st.markdown("""
        - [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
        - [Amazon VPC Documentation](https://docs.aws.amazon.com/vpc/)
        - [AWS PrivateLink Documentation](https://docs.aws.amazon.com/vpc/latest/privatelink/)
        - [SageMaker Studio Security](https://docs.aws.amazon.com/sagemaker/latest/dg/security.html)
        """)
    
    # Create tabs
    tabs = st.tabs([
        "🏠 Home", 
        "🔌 SageMaker Endpoints within VPC Network", 
        "🛠️ SageMaker Studio in a Private VPC", 
        "🧠 Knowledge Check"
    ])
    
    with tabs[0]:
        render_home()
    
    with tabs[1]:
        render_endpoints()
    
    with tabs[2]:
        render_studio()
    
    with tabs[3]:
        render_knowledge_check()
    
    # Footer
    st.markdown(
        """
        <div class="footer">
        © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
