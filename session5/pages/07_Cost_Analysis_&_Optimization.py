import streamlit as st
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64
import io
import json
import math
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="SageMaker Performance & Optimization",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    
    if "quiz_attempted" not in st.session_state:
        st.session_state["quiz_attempted"] = False
    
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = []

# Initialize session state
init_session_state()

# Apply AWS Style
def apply_aws_style():
    st.markdown("""
    <style>
    .main {background-color: #F8F8F8;}
    h1 {color: #232F3E;}
    h2 {color: #FF9900;}
    h3 {color: #232F3E;}
    .stButton>button {background-color: #FF9900; color: white;}
    .stTextInput>div>div>input {border-color: #FF9900;}
    .css-1aumxhk {background-color: #232F3E;}
    
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
    </style>
    """, unsafe_allow_html=True)


apply_aws_style()





# Sidebar
with st.sidebar:

    st.subheader("Session Management")
    st.info(f"User ID: {st.session_state.session_id}")
        
    # Reset session
    if st.button("Reset Session"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.rerun()

    st.divider()
    # About this App (collapsible)
    with st.expander("About this App", expanded=False):
        st.write("""
        This interactive e-learning application focuses on AWS SageMaker Performance & Optimization.
        
        Learn about:
        - Cost analysis tools
        - Cost optimization strategies
        
        Complete knowledge checks to test your understanding!
        """)
    


# Main content area
st.title("🚀 SageMaker Performance & Optimization")
st.markdown("---")

# Create tabs for navigation
tabs = st.tabs([
    "💰 Cost Analysis Tools", 
    "⚙️ Optimization Strategies",
    "✅ Knowledge Check"
])

# Tab 1: Cost Analysis Tools
with tabs[0]:
    st.header("Cost Analysis Tools")
    
    st.markdown("""
    AWS provides several tools to help you monitor, analyze, and optimize your SageMaker costs.
    These tools give you visibility into your spending patterns and help you identify opportunities
    for cost optimization.
    """)
    
    # Create subtabs for cost analysis tools
    cost_tools_tabs = st.tabs([
        "AWS Cost Explorer", 
        "AWS Budgets", 
        "AWS Cost & Usage Report", 
        "AWS Trusted Advisor"
    ])
    
    # AWS Cost Explorer tab
    with cost_tools_tabs[0]:
        st.subheader("AWS Cost Explorer")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            AWS Cost Explorer provides a visual interface to understand and analyze your AWS costs and usage over time.
            
            **Key Features:**
            - **Visualize costs** - View costs by service, region, tag, and more
            - **Filter and group** - Analyze SageMaker costs by endpoint, training job, or notebook instances
            - **Forecasting** - Predict future costs based on historical spending
            - **Savings recommendations** - Get suggestions for potential cost savings
            
            **Best for:**
            - Historical cost analysis
            - Identifying cost trends
            - Service-level cost breakdowns
            """)
            
            # Sample code for accessing Cost Explorer API
            st.code('''
# Using the AWS SDK for Python (Boto3) to access Cost Explorer
import boto3

ce_client = boto3.client('ce')

# Get SageMaker costs for the last month
response = ce_client.get_cost_and_usage(
    TimePeriod={
        'Start': '2023-01-01',
        'End': '2023-01-31'
    },
    Granularity='DAILY',
    Filter={
        'Dimensions': {
            'Key': 'SERVICE',
            'Values': ['Amazon SageMaker']
        }
    },
    Metrics=['UnblendedCost'],
    GroupBy=[
        {
            'Type': 'DIMENSION',
            'Key': 'USAGE_TYPE'
        }
    ]
)

# Process and analyze the results
for result in response['ResultsByTime']:
    date = result['TimePeriod']['Start']
    for group in result['Groups']:
        usage_type = group['Keys'][0]
        cost = group['Metrics']['UnblendedCost']['Amount']
        print(f"{date}: {usage_type} - ${cost}")
            ''', language='python')
        
        with col2:
            # Example Cost Explorer visualization
            st.image("https://d1.awsstatic.com/products/costmanagement/ce-sample-dash.cc87ee2076b1b631f8a207804204bc728ae691b0.png", 
                    caption="AWS Cost Explorer Dashboard", 
                    use_container_width=True)
    
    # AWS Budgets tab
    with cost_tools_tabs[1]:
        st.subheader("AWS Budgets")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            AWS Budgets allows you to set custom cost and usage budgets with notifications when you exceed your thresholds.
            
            **Key Features:**
            - **Cost budgets** - Set alerts for total SageMaker spending
            - **Usage budgets** - Monitor specific resource usage like instance hours
            - **Alert thresholds** - Configure notifications at different percentage thresholds
            - **Actions** - Automatically trigger responses when thresholds are crossed
            
            **Best for:**
            - Proactive cost management
            - Budget enforcement
            - Cost anomaly detection
            """)
            
            # Sample code for creating a budget
            st.code('''
# Using the AWS SDK for Python (Boto3) to create a budget
import boto3

budgets_client = boto3.client('budgets')

# Create a budget for SageMaker with monthly limit of $1000
response = budgets_client.create_budget(
    AccountId='123456789012',
    Budget={
        'BudgetName': 'SageMaker-Monthly-Budget',
        'BudgetLimit': {
            'Amount': '1000',
            'Unit': 'USD'
        },
        'CostFilters': {
            'Service': ['Amazon SageMaker']
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST'
    },
    NotificationsWithSubscribers=[
        {
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 80.0,
                'ThresholdType': 'PERCENTAGE'
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'email@example.com'
                }
            ]
        }
    ]
)
            ''', language='python')
        
        with col2:
            # Example Budgets visualization
            st.image("https://docs.aws.amazon.com/images/cost-management/latest/userguide/images/AWSBudgets_Dashboard.png", 
                    caption="AWS Budgets Dashboard", 
                    use_container_width=True)
    
    # AWS Cost & Usage Report tab            
    with cost_tools_tabs[2]:
        st.subheader("AWS Cost & Usage Report")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            AWS Cost & Usage Report provides the most comprehensive set of cost and usage data available for AWS services.
            
            **Key Features:**
            - **Granular data** - Track costs down to hourly level with resource tags
            - **Resource-level insights** - See costs for each SageMaker component (endpoints, training jobs, etc.)
            - **Integration** - Load data into Amazon Athena, Amazon Redshift, or AWS QuickSight for analysis
            - **Customization** - Configure report content, partitioning, and update frequency
            
            **Best for:**
            - Detailed cost analysis
            - Building custom cost dashboards
            - Advanced cost allocation and chargeback
            """)
            
            # Example of analyzing CUR data with Athena
            st.code('''
# Example SQL query for analyzing SageMaker costs in Athena
SELECT
    line_item_usage_start_date AS date,
    product_instance_type,
    line_item_usage_type,
    SUM(line_item_unblended_cost) AS cost
FROM
    cost_and_usage_report
WHERE
    product_product_name = 'Amazon SageMaker'
    AND line_item_usage_start_date BETWEEN '2023-01-01' AND '2023-01-31'
GROUP BY
    line_item_usage_start_date,
    product_instance_type,
    line_item_usage_type
ORDER BY
    date,
    cost DESC
            ''', language='sql')
        
        with col2:
            # Example CUR visualization
            st.image("https://d2908q01vomqb2.cloudfront.net/77de68daecd823babbb58edb1c8e14d7106e83bb/2018/05/02/LakeFormationCosts-2_1.gif", 
                    caption="AWS Cost & Usage Report Analysis", 
                    use_container_width=True)
    
    # AWS Trusted Advisor tab            
    with cost_tools_tabs[3]:
        st.subheader("AWS Trusted Advisor")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            AWS Trusted Advisor provides recommendations to help you optimize costs, improve performance, and enhance security.
            
            **Key Features:**
            - **Cost Optimization** - Identify idle resources, oversized instances, and opportunities for Reserved Instances
            - **Performance** - Get recommendations to improve service performance
            - **Security** - Identify security vulnerabilities and compliance issues
            - **Fault Tolerance** - Assess resilience of your architecture
            
            **Best for:**
            - Quick cost optimization opportunities
            - Real-time recommendations
            - Cross-service optimizations
            """)
            
            # Example of accessing Trusted Advisor recommendations
            st.code('''
# Using the AWS SDK for Python (Boto3) to access Trusted Advisor recommendations
import boto3

support_client = boto3.client('support')

# Get cost optimization check results
response = support_client.describe_trusted_advisor_check_result(
    checkId='Qch7DwouX1',  # Cost Optimization check ID
    language='en'
)

# Process the recommendations
for resource in response['result']['flaggedResources']:
    resource_id = resource['resourceId']
    estimated_savings = resource['metadata'][3]
    region = resource['region']
    print(f"Resource: {resource_id} in {region} - Potential savings: ${estimated_savings}")
            ''', language='python')
        
        with col2:
            # Example Trusted Advisor dashboard
            st.image("https://d1.awsstatic.com/product-marketing/AWS%20Support/Trusted-Advisor-screenshot.321d0a3a729b9d4468f24a005d53a0142998c952.png", 
                    caption="AWS Trusted Advisor Dashboard", 
                    use_container_width=True)
    
    # Cost monitoring best practices
    st.subheader("Cost Monitoring Best Practices")
    
    st.markdown("""
    1. **Implement tagging** - Use consistent resource tagging for cost allocation
    2. **Set up budgets** - Create budget alerts to avoid unexpected spending
    3. **Regular reviews** - Schedule weekly or monthly cost reviews
    4. **Track per-model costs** - Allocate costs to specific ML models and projects
    5. **Analyze trends** - Look for cost patterns and optimization opportunities
    """)
    
    # Cost metrics to monitor
    st.subheader("Key Cost Metrics to Monitor")
    
    metrics_data = {
        'Metric': [
            'Cost per training hour', 
            'Cost per inference', 
            'GPU utilization (%)', 
            'Endpoint uptime cost',
            'Data storage cost'
        ],
        'Target Range': [
            '$1-5/hr', 
            '$0.0001-0.001', 
            '70-90%', 
            'Varies by SLA',
            'Optimize for access pattern'
        ],
        'Optimization Tip': [
            'Use spot instances for non-critical training',
            'Batch predictions when possible',
            'Right-size instances based on model needs',
            'Use auto-scaling for variable traffic',
            'Lifecycle policies for older artifacts'
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

# Tab 2: Optimization Strategies
with tabs[1]:
    st.header("SageMaker Cost Optimization Strategies")
    
    st.markdown("""
    Optimizing costs for machine learning workloads requires a strategic approach to 
    how models are deployed and managed. SageMaker offers several specialized deployment 
    options that can significantly reduce costs while maintaining performance.
    """)
    
    # Create subtabs for optimization strategies
    optimization_tabs = st.tabs([
        "Multi-Model Endpoints", 
        "Multi-Container Endpoints", 
        "Asynchronous/Serverless Inference", 
        "AWS Inferentia", 
        "SageMaker Neo"
    ])
    
    # Multi-Model Endpoints tab
    with optimization_tabs[0]:
        st.subheader("Multi-Model Endpoints")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            **Multi-Model Endpoints** allow you to deploy multiple models to a single endpoint, sharing 
            compute resources and reducing costs.
            
            **How it works:**
            - Multiple models share the same container and instance
            - Models are loaded into memory on-demand
            - Inactive models are unloaded to free resources
            
            **Ideal for:**
            - Serving many models with low to moderate traffic
            - Models that share the same framework (PyTorch, TensorFlow, etc.)
            - Personalization use cases with per-user models
            
            **Cost savings:** Up to 90% compared to deploying individual endpoints for each model
            """)
            
            # Sample code for Multi-Model Endpoints
            st.code('''
# Deploy multiple models to a single endpoint
from sagemaker.multidel_model import MultiModelPredictor

# Create the multi-model endpoint
mme = MultiModelPredictor(
    endpoint_name='my-multi-model-endpoint',
    model_path='s3://my-bucket/models/',  # Path to models
    image_uri='<framework-container-uri>',
    instance_type='ml.c5.xlarge',
    instance_count=1,
    role=role
)

# Invoke a specific model
response = mme.predict(
    target_model='customer-model-1.tar.gz',
    data=my_payload
)
            ''', language='python')
        
        with col2:
            # Visualization of Multi-Model Endpoints
            st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2019/11/07/sagemaker-multimodel-2.gif", 
                    caption="Multi-Model Endpoint Architecture", 
                    use_container_width=True)
            
            # Cost comparison widget
            st.subheader("Cost Comparison")
            
            # Create interactive slider for number of models
            num_models = st.slider("Number of models", min_value=2, max_value=50, value=10)
            avg_tps = st.slider("Average transactions per second", min_value=1, max_value=100, value=10)
            
            # Calculate costs
            single_endpoint_cost = num_models * 0.298 * 24 * 30  # ml.c5.xlarge at $0.298/hour
            mme_cost = 0.298 * 24 * 30 * math.ceil(num_models / 10)
            
            # Display savings
            savings = single_endpoint_cost - mme_cost
            savings_percent = (savings / single_endpoint_cost) * 100
            
            st.metric(
                "Monthly cost savings with MME", 
                f"${savings:.2f}", 
                f"{savings_percent:.0f}%"
            )
    
    # Multi-Container Endpoints tab
    with optimization_tabs[1]:
        st.subheader("Multi-Container Endpoints")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            **Multi-Container Endpoints** allow you to deploy up to 15 different containers on the same endpoint,
            each potentially running a different framework or model.
            
            **How it works:**
            - Multiple containers run simultaneously on the same instance
            - Each container can use a different ML framework
            - No cold start unlike Multi-Model Endpoints
            
            **Ideal for:**
            - Deploying models with different frameworks
            - Preprocessing and inference pipelines
            - Ensemble models that require different frameworks
            
            **Cost savings:** Up to 85% compared to deploying separate endpoints
            """)
            
            # Sample code for Multi-Container Endpoints
            st.code('''
# Deploy multiple containers to a single endpoint
import sagemaker
from sagemaker.multidel_container import MultiContainerModel

# Define the containers
containers = [
    {
        'Image': '<tensorflow-container-uri>',
        'ModelDataUrl': 's3://my-bucket/model1.tar.gz'
    },
    {
        'Image': '<pytorch-container-uri>',
        'ModelDataUrl': 's3://my-bucket/model2.tar.gz'
    }
]

# Create multi-container model
multi_model = MultiContainerModel(
    name='multi-container-model',
    role=role,
    containers=containers
)

# Deploy to endpoint
predictor = multi_model.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.2xlarge'
)

# Make a prediction with container selection
response = predictor.predict(
    data=my_payload,
    target_container=0  # Index of the container to use
)
            ''', language='python')
        
        with col2:
            # Visualization of Multi-Container Endpoints
            st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/10/13/Picture4-2.png", 
                    caption="Multi-Container Endpoint Architecture", 
                    use_container_width=True)
            
            # Key differences vs Multi-Model Endpoints
            st.subheader("Comparing with Multi-Model Endpoints")
            
            comparison = {
                'Feature': ['Max containers/models', 'Cold start', 'Different frameworks', 'Memory sharing', 'Model loading'],
                'Multi-Container': ['15 containers', 'No', 'Yes', 'Limited', 'Always loaded'],
                'Multi-Model': ['Thousands', 'Yes', 'No', 'Yes', 'On-demand']
            }
            st.table(pd.DataFrame(comparison))
    
    # Asynchronous/Serverless Inference tab
    with optimization_tabs[2]:
        st.subheader("Asynchronous & Serverless Inference")
        
        # Create nested tabs for Asynchronous and Serverless Inference
        async_serverless_tabs = st.tabs(["Asynchronous Inference", "Serverless Inference"])
        
        # Asynchronous Inference subtab
        with async_serverless_tabs[0]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                **Asynchronous Inference** enables processing requests with large payloads or long processing times
                without maintaining a persistent connection.
                
                **How it works:**
                - Requests are queued for processing
                - Results are stored in S3
                - No timeout constraints for long-running inference
                
                **Ideal for:**
                - Large input payloads (up to 1GB)
                - Long processing times (minutes or hours)
                - Batch-like workloads that don't need immediate results
                
                **Cost savings:** Pay only for compute time used for processing requests
                """)
                
                # Sample code for Asynchronous Inference
                st.code('''
# Deploy model for asynchronous inference
from sagemaker.model import Model

model = Model(
    image_uri=image_uri,
    model_data=model_data,
    role=role
)

# Deploy as asynchronous endpoint
async_predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='my-async-endpoint',
    async_inference_config={
        "OutputConfig": {
            "S3OutputPath": "s3://my-bucket/async-results/"
        },
        "ClientConfig": {
            "MaxConcurrentInvocationsPerInstance": 4
        }
    }
)

# Invoke asynchronously
response = async_predictor.predict_async(
    data=my_payload,
    input_path='s3://my-bucket/input/my-data.csv',
    wait=False
)

# Get result when ready
output_path = response["OutputPath"]
                ''', language='python')
            
            with col2:
                # Visualization of Asynchronous Inference
                st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/01/05/1-Architecture.jpg", 
                        caption="Asynchronous Inference Architecture", 
                        use_container_width=True)
                
                # Benefits list
                st.subheader("Key Benefits")
                st.markdown("""
                - **Cost efficiency** - No need to provision for peak capacity
                - **Handle large payloads** - Process inputs up to 1GB
                - **Queue management** - Built-in request queuing and scaling
                - **No timeout constraints** - Process long-running inferences
                """)
        
        # Serverless Inference subtab
        with async_serverless_tabs[1]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                **Serverless Inference** provides on-demand ML inference without having to configure or manage the 
                underlying infrastructure.
                
                **How it works:**
                - Auto-scales from zero to thousands of instances
                - Pay only for compute used during inference
                - No instance configuration required
                
                **Ideal for:**
                - Unpredictable or variable workloads
                - Applications with idle periods
                - Development and testing environments
                
                **Cost savings:** Up to 70% for variable workloads compared to provisioned endpoints
                """)
                
                # Sample code for Serverless Inference
                st.code('''
# Deploy model for serverless inference
from sagemaker.model import Model

model = Model(
    image_uri=image_uri,
    model_data=model_data,
    role=role
)

# Deploy as serverless endpoint
serverless_predictor = model.deploy(
    endpoint_name='my-serverless-endpoint',
    serverless_inference_config={
        "MemorySizeInMB": 2048,
        "MaxConcurrency": 5
    }
)

# Invoke the endpoint
response = serverless_predictor.predict(
    data=my_payload
)
                ''', language='python')
            
            with col2:
                # Visualization of Serverless Inference
                st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/12/08/ML-4337-image001.png", 
                        caption="Serverless Inference Architecture", 
                        use_container_width=True)
                
                # Cost comparison widget
                st.subheader("Cost Comparison")
                
                # Create interactive slider for usage pattern
                usage_hours = st.slider("Hours of active usage per day", min_value=1, max_value=24, value=8)
                
                # Calculate costs
                provisioned_cost = 0.298 * 24 * 30  # ml.c5.xlarge at $0.298/hour
                serverless_cost = 0.0000021 * 3600 * usage_hours * 30  # $0.0000021 per second of compute time
                
                # Display savings
                savings = provisioned_cost - serverless_cost
                savings_percent = (savings / provisioned_cost) * 100
                
                st.metric(
                    "Monthly cost savings with Serverless", 
                    f"${savings:.2f}", 
                    f"{savings_percent:.0f}%"
                )
    
    # AWS Inferentia tab
    with optimization_tabs[3]:
        st.subheader("AWS Inferentia")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            **AWS Inferentia** is a custom silicon chip designed by AWS specifically for machine learning inference workloads.
            
            **How it works:**
            - Purpose-built for ML inference
            - Optimized for TensorFlow, PyTorch, and MXNet
            - Uses AWS Neuron SDK for model compilation
            
            **Ideal for:**
            - High-throughput inference workloads
            - Cost-sensitive production deployments
            - Models that can be compiled with Neuron SDK
            
            **Performance:** Up to 2.3x higher throughput and up to 70% lower cost per inference than comparable EC2 instances
            """)
            
            # Sample code for AWS Inferentia
            st.code('''
# Compile model for Inferentia
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

# First compile the model for Inferentia
sagemaker_session = sagemaker.Session()

# Define the compilation job
compile_job_name = "compile-inferentia-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_path = f"s3://{sagemaker_session.default_bucket()}/{compile_job_name}/output"

# Create the compilation job
sagemaker_session.compile_model(
    target_instance_family="ml_inf1",
    input_model_s3_uri="s3://my-bucket/model.tar.gz",
    output_model_s3_uri=output_path,
    role=role,
    framework="TENSORFLOW",
    framework_version="2.5.0",
    job_name=compile_job_name
)

# Deploy the compiled model
model = TensorFlowModel(
    model_data=output_path,
    role=role
)

# Deploy to Inferentia instance
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.inf1.xlarge"
)
            ''', language='python')
        
        with col2:
            # Visualization of AWS Inferentia
            st.image("https://d2908q01vomqb2.cloudfront.net/77de68daecd823babbb58edb1c8e14d7106e83bb/2019/12/03/Untitled-5.png", 
                    caption="AWS Inferentia Workflow", 
                    use_container_width=True)
            
            # Performance comparison
            st.subheader("Performance Comparison")
            
            perf_data = {
                'Metric': ['Throughput (images/sec)', 'Latency (ms)', 'Cost per 1M inferences ($)'],
                'ml.c5.xlarge': [100, 125, 10.00],
                'ml.g4dn.xlarge': [170, 75, 15.00],
                'ml.inf1.xlarge': [220, 60, 6.00]
            }
            
            perf_df = pd.DataFrame(perf_data)
            st.table(perf_df)
            
            # Highlight requirements
            st.subheader("Requirements")
            st.markdown("""
            - Model must be compilable with AWS Neuron SDK
            - Supported frameworks: TensorFlow, PyTorch, MXNet
            - Model must use supported operators
            - Additional compilation step required
            """)
    
    # SageMaker Neo tab
    with optimization_tabs[4]:
        st.subheader("SageMaker Neo")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            **SageMaker Neo** automatically optimizes machine learning models for inference on SageMaker,
            edge devices, and a variety of hardware platforms.
            
            **How it works:**
            - Analyzes models and optimizes for target hardware
            - Creates optimized binaries that use less compute resources
            - Supports multiple frameworks and hardware targets
            
            **Ideal for:**
            - Optimizing models for any deployment target
            - Edge deployments with resource constraints
            - Maximizing performance across different hardware
            
            **Performance:** Up to 25% improved performance at no additional cost
            """)
            
            # Sample code for SageMaker Neo
            st.code('''
# Compile a model with SageMaker Neo
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

# Create a compilation job
sagemaker_session = sagemaker.Session()

# Define the compilation job
compile_job_name = "neo-optimized-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_path = f"s3://{sagemaker_session.default_bucket()}/{compile_job_name}/output"

# Compile the model with Neo
sagemaker_session.compile_model(
    target_instance_family="ml_c5",  # Target hardware 
    input_model_s3_uri="s3://my-bucket/model.tar.gz",
    output_model_s3_uri=output_path,
    framework="TENSORFLOW",
    framework_version="2.5.0",
    data_shape={"inputs": [1, 224, 224, 3]},  # Input shape
    role=role
)

# Deploy the Neo-optimized model
model = TensorFlowModel(
    model_data=output_path,
    framework_version="2.5.0",
    role=role
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.c5.xlarge"
)
            ''', language='python')
        
        with col2:
            # Visualization of SageMaker Neo
            st.image("https://d1.awsstatic.com/diagrams/product-page-diagrams_Amazon-Sagemaker-Neo_How-it-Works.4a484c26d8712b2a25a7fe1c17f02e01197c8dd9.png", 
                    caption="SageMaker Neo Workflow", 
                    use_container_width=True)
            
            # Supported frameworks and targets
            st.subheader("Supported Platforms")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Frameworks:**")
                st.markdown("""
                - TensorFlow
                - PyTorch
                - MXNet
                - XGBoost
                - ONNX
                - Darknet
                """)
            
            with col_b:
                st.markdown("**Hardware Targets:**")
                st.markdown("""
                - EC2 CPU/GPU instances
                - AWS Inferentia
                - NVIDIA Jetson
                - Raspberry Pi
                - Intel devices
                - Qualcomm devices
                """)
    
    # Overall cost optimization best practices
    st.subheader("Overall Cost Optimization Best Practices")
    
    st.markdown("""
    1. **Right-size your instances** - Select the smallest instance type that meets your performance requirements. Run load tests to determine the optimal configuration.
    
    2. **Use auto-scaling** - Configure auto-scaling to dynamically adjust resources based on traffic patterns. This prevents over-provisioning during low-traffic periods.
    
    3. **Implement batch transformations** - For non-real-time needs, use batch transform jobs instead of maintaining always-on endpoints.
    
    4. **Optimize input data** - Reduce payload sizes and optimize preprocessing to reduce compute requirements and inference time.
    
    5. **Monitor and analyze costs** - Regularly review usage patterns and costs to identify optimization opportunities.
    """)
    
    # Decision tree for choosing optimization strategy
    st.subheader("Choosing the Right Optimization Strategy")
    
    st.markdown("""
    Use this decision flow to determine the best deployment strategy for your model:
    
    1. **Do you have many similar models with low individual traffic?**
       - Yes → Consider Multi-Model Endpoints
       - No → Continue
    
    2. **Do you need to deploy models with different frameworks together?**
       - Yes → Consider Multi-Container Endpoints
       - No → Continue
    
    3. **Is your workload variable or unpredictable?**
       - Yes → Consider Serverless Inference
       - No → Continue
    
    4. **Do you have large payloads or long processing times?**
       - Yes → Consider Asynchronous Inference
       - No → Continue
    
    5. **Are you looking for maximum cost-performance ratio?**
       - Yes → Consider AWS Inferentia or SageMaker Neo optimization
       - No → Use standard SageMaker endpoints with right-sized instances
    """)

# Tab 3: Knowledge Check
with tabs[2]:
    st.markdown("""
    <style>
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
    
    custom_header("Test Your Knowledge")
    
    st.markdown("""
    This quiz will test your understanding of the key concepts covered in SageMaker Performance & Optimization.
    Answer the following questions to evaluate your knowledge of cost analysis tools and optimization strategies.
    """)
    
    # Define quiz questions
    questions = [
        {
            "question": "Which AWS tool provides a visual interface to understand and analyze your AWS costs and usage over time?",
            "options": ["AWS Cost Explorer", "AWS CloudFormation", "AWS Budgets", "AWS Trusted Advisor"],
            "correct": "AWS Cost Explorer",
            "explanation": "AWS Cost Explorer provides a visual interface to understand and analyze your AWS costs and usage over time, allowing you to visualize costs by service, region, tag, and more."
        },
        {
            "question": "Which SageMaker deployment option allows you to deploy multiple models to a single endpoint, sharing compute resources?",
            "options": ["Multi-Model Endpoints", "Multi-Container Endpoints", "Serverless Inference", "Asynchronous Inference"],
            "correct": "Multi-Model Endpoints",
            "explanation": "Multi-Model Endpoints allow you to deploy multiple models to a single endpoint, sharing compute resources and reducing costs. Models are loaded into memory on-demand and inactive models are unloaded to free resources."
        },
        {
            "question": "What is the main advantage of using Serverless Inference?",
            "options": ["Handles large payloads up to 1GB", "Auto-scales from zero with no instance management", "Supports different ML frameworks on the same endpoint", "Optimizes model execution for specific hardware"],
            "correct": "Auto-scales from zero with no instance management",
            "explanation": "Serverless Inference provides on-demand ML inference without having to configure or manage the underlying infrastructure. It auto-scales from zero to thousands of instances and you only pay for compute used during inference."
        },
        {
            "question": "Which AWS service allows you to set custom cost and usage budgets with notifications when you exceed your thresholds?",
            "options": ["AWS Cost Explorer", "AWS Budgets", "AWS Cost & Usage Report", "AWS CloudWatch"],
            "correct": "AWS Budgets",
            "explanation": "AWS Budgets allows you to set custom cost and usage budgets with notifications when you exceed your thresholds. You can configure cost budgets, usage budgets, and alert thresholds at different percentage levels."
        },
        {
            "question": "What is the maximum number of different containers that can be deployed on a SageMaker Multi-Container Endpoint?",
            "options": ["5", "10", "15", "Unlimited"],
            "correct": "15",
            "explanation": "SageMaker Multi-Container Endpoints support up to 15 distinct containers on a single endpoint. Each container can use a different ML framework."
        },
        {
            "question": "Which specialized AWS hardware is custom-designed for machine learning inference workloads?",
            "options": ["AWS Graviton", "AWS Inferentia", "AWS Trainium", "AWS Nitro"],
            "correct": "AWS Inferentia",
            "explanation": "AWS Inferentia is a custom silicon chip designed by AWS specifically for machine learning inference workloads. It's optimized for TensorFlow, PyTorch, and MXNet and offers up to 2.3x higher throughput and 70% lower cost per inference than comparable instances."
        },
        {
            "question": "Which SageMaker feature automatically optimizes machine learning models for inference on various hardware platforms?",
            "options": ["SageMaker Pipelines", "SageMaker Neo", "SageMaker Clarify", "SageMaker Feature Store"],
            "correct": "SageMaker Neo",
            "explanation": "SageMaker Neo automatically optimizes machine learning models for inference on SageMaker, edge devices, and a variety of hardware platforms. It analyzes models and optimizes them for target hardware, creating optimized binaries that use less compute resources."
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
            st.success("🎉 Perfect score! You've mastered the concepts of SageMaker Performance & Optimization")
        elif score >= 3:
            st.success("👍 Good job! You have a solid understanding of the concepts.")
        else:
            st.warning("📚 You might want to review the content again to strengthen your understanding.")
        
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
