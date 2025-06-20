
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io

# Set page config
st.set_page_config(
    page_title="ML Engineer - Associate Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'quiz_score' not in st.session_state:
    st.session_state['quiz_score'] = 0
if 'quiz_attempted' not in st.session_state:
    st.session_state['quiz_attempted'] = False
if 'name' not in st.session_state:
    st.session_state['name'] = ""
if 'visited_Model_Registry' not in st.session_state:
    st.session_state['visited_Model_Registry'] = False
if 'visited_Inference_Options' not in st.session_state:
    st.session_state['visited_Inference_Options'] = False
if 'visited_Infrastructure' not in st.session_state:
    st.session_state['visited_Infrastructure'] = False
if 'visited_Pipelines' not in st.session_state:
    st.session_state['visited_Pipelines'] = False
if 'visited_MLOps' not in st.session_state:
    st.session_state['visited_MLOps'] = False

# Custom CSS for styling - same as Domain 1
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
    .info-box {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #D1FAE5;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .tip-box {
        background-color: #E0F2FE;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #0EA5E9;
    }
    .step-box {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
        transition: transform 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .aws-orange {
        color: #FF9900;
    }
    .aws-blue {
        color: #232F3E;
    }
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    /* Make the tab content container take full height */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8F9FA;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 16px;
        padding-right: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF9900 !important;
        color: white !important;
    }
    .definition {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 10px 15px;
        margin: 15px 0;
        border-radius: 0 5px 5px 0;
    }
    .code-box {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        margin: 15px 0;
        border: 1px solid #E5E7EB;
    }
    .model-diagram {
        text-align: center;
        margin: 20px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        border-left: 4px solid #FF9900;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
    }
    .comparison-table th {
        background-color: #FF9900;
        color: white;
        padding: 10px;
        text-align: left;
    }
    .comparison-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    .comparison-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .feature-list {
        list-style-type: none;
        padding-left: 0;
    }
    .feature-list li {
        margin-bottom: 8px;
        padding-left: 25px;
        position: relative;
    }
    .feature-list li:before {
        content: "✓";
        color: #FF9900;
        position: absolute;
        left: 0;
        font-weight: bold;
    }
    .deployment-option {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
    }
    .deployment-option h4 {
        color: #FF9900;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to display custom header - same as Domain 1
def custom_header(text, level="main"):
    if level == "main":
        st.markdown(f'<div class="main-header">{text}</div>', unsafe_allow_html=True)
    elif level == "sub":
        st.markdown(f'<div class="sub-header">{text}</div>', unsafe_allow_html=True)
    elif level == "section":
        st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

# Function to create custom info box - same as Domain 1
def info_box(text, box_type="info"):
    if box_type == "info":
        st.markdown(f"""
            <div class="info-box">
                <div markdown="1">
                    {text}
            """, unsafe_allow_html=True)
    elif box_type == "success":
        st.markdown(f"""
            <div class="success-box">
                <div markdown="1">
                    {text}
            """, unsafe_allow_html=True)
    elif box_type == "warning":
        st.markdown(f"""
            <div class="warning-box">
                <div markdown="1">
                    {text}
            """, unsafe_allow_html=True)
    elif box_type == "tip":
        st.markdown(f"""
            <div class="tip-box">
                <div markdown="1">
                    {text}
            """, unsafe_allow_html=True)

# Function for definition box - same as Domain 1
def definition_box(term, definition):
    st.markdown(f"""
    <div class="definition">
        <strong>{term}:</strong> {definition}
    </div>
    """, unsafe_allow_html=True)

# Function to display deployment option
def deployment_option(title, description, use_cases, limitations=None):
    st.markdown(f"""
    <div class="deployment-option">
        <h4>{title}</h4>
        <p>{description}</p>
        <strong>Example Use Cases:</strong>
        <ul>
            {"".join([f"<li>{case}</li>" for case in use_cases])}
        </ul>
        {f"<strong>Limitations:</strong><ul>{''.join([f'<li>{limit}</li>' for limit in limitations])}</ul>" if limitations else ""}
    </div>
    """, unsafe_allow_html=True)

# Function to reset session - same as Domain 1
def reset_session():
    st.session_state['quiz_score'] = 0
    st.session_state['quiz_attempted'] = False
    st.session_state['name'] = ""
    st.session_state['visited_Model_Registry'] = False
    st.session_state['visited_Inference_Options'] = False
    st.session_state['visited_Infrastructure'] = False
    st.session_state['visited_Pipelines'] = False
    st.session_state['visited_MLOps'] = False
    st.rerun()

# Sidebar for session management - similar to Domain 1
with st.sidebar:
    st.image("images/mla_badge.png", width=150)
    st.markdown("### ML Engineer - Associate")
    st.markdown("#### Domain 3: Deployment and Orchestration")
    
    # If user has provided their name, greet them
    if st.session_state['name']:
        st.success(f"Welcome, {st.session_state['name']}! 👋")
    else:
        name = st.text_input("Enter your name:")
        if name:
            st.session_state['name'] = name
            st.rerun()
    
    # Reset button
    if st.button("🔄 Reset Session"):
        reset_session()
    
    # Progress tracking
    if st.session_state['name']:
        st.markdown("---")
        st.markdown("### Your Progress")
        
        # Track visited pages
        visited_pages = [page for page in ["Model_Registry", "Inference_Options", "Infrastructure", "Pipelines", "MLOps"] 
                         if st.session_state.get(f"visited_{page}", False)]
        
        progress = len(visited_pages) / 5
        st.progress(progress)
        st.markdown(f"**{len(visited_pages)}/5 sections completed**")
        
        # Track quiz score if attempted
        if st.session_state['quiz_attempted']:
            st.markdown(f"**Quiz Score: {st.session_state['quiz_score']}/5**")
        
        # Learning outcomes reminder
        st.markdown("---")
        st.markdown("### Learning Outcomes")
        st.markdown("""
        - Understand SageMaker Model Registry
        - Compare different inference options
        - Create infrastructure using IaC
        - Implement MLOps practices with CI/CD
        - Design automated ML pipelines
        """)

# Main content with tabs
tabs = st.tabs([
    "🏠 Home", 
    "📦 SageMaker Model Registry", 
    "🚀 Inference Options", 
    "🏗️ Infrastructure", 
    "🔄 SageMaker Pipelines", 
    "⚙️ MLOps",
    "❓ Quiz", 
    "📚 Resources"
])

# Home tab
with tabs[0]:
    custom_header("AWS Partner Certification Readiness")
    st.markdown("## Machine Learning Engineer - Associate")
    
    st.markdown("### Domain 3: Deployment and Orchestration of ML Workflows")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        info_box("""
        This interactive e-learning application covers Domain 3 from the AWS Machine Learning Engineer - Associate certification.
        
        Domain 3 focuses on **Deployment and Orchestration of ML Workflows**, covering how to deploy and automate machine learning models in production.
        
        Navigate through the content using the tabs above to learn about:
        - SageMaker Model Registry
        - Model Inference Options
        - Infrastructure as Code
        - SageMaker Pipelines
        - MLOps Practices
        
        Test your knowledge with the quiz when you're ready!
        """, "info")
        
        st.markdown("### Learning Outcomes")
        st.markdown("""
        By the end of this module, you will be able to:
        - Use SageMaker Model Registry to organize and deploy models
        - Select appropriate model deployment infrastructure based on requirements
        - Create and script infrastructure using CloudFormation and AWS CDK
        - Use SageMaker Pipelines for automating ML workflows
        - Implement MLOps practices and CI/CD pipelines
        """)
    
    with col2:
        st.image("images/mla_badge_big.png", width=250)
        
        if st.session_state['quiz_attempted']:
            st.success(f"Current Quiz Score: {st.session_state['quiz_score']}/5")
        
        st.info("Use the tabs above to navigate through different sections!")
        
    st.markdown("---")
    
    st.markdown("### Machine Learning Lifecycle")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("#### Setup")
        st.markdown("""
        - Environment preparation
        - Permission configuration
        - Resource allocation
        """)
    
    with col2:
        st.markdown("#### Data Processing")
        st.markdown("""
        - Data collection
        - Data preparation
        - Feature engineering
        """)
    
    with col3:
        st.markdown("#### Model Development")
        st.markdown("""
        - Algorithm selection
        - Model training
        - Model evaluation
        """)
    
    with col4:
        st.markdown("#### Deployment & Inference")
        st.markdown("""
        - **Model registry** ← You are here
        - **Deployment strategies**
        - **Infrastructure setup**
        """)
        
    with col5:
        st.markdown("#### Monitoring")
        st.markdown("""
        - Data drift detection
        - Performance monitoring
        - Retraining pipelines
        """)
    
    st.markdown("---")
    
    st.markdown("### Domain 3 Task Statements")
    
    task_col1, task_col2, task_col3 = st.columns(3)
    
    with task_col1:
        st.markdown("#### Task 3.1: Select Deployment Infrastructure")
        st.markdown("""
        - Evaluate infrastructure options
        - Choose appropriate deployment methods
        - Determine scaling requirements
        - Optimize for performance and cost
        """)
    
    with task_col2:
        st.markdown("#### Task 3.2: Create and Script Infrastructure")
        st.markdown("""
        - Use infrastructure as code
        - Implement CloudFormation templates
        - Utilize AWS CDK
        - Configure auto-scaling
        """)
    
    with task_col3:
        st.markdown("#### Task 3.3: Set Up CI/CD Pipelines")
        st.markdown("""
        - Implement MLOps practices
        - Create automated workflows
        - Use SageMaker Pipelines
        - Configure model deployment automations
        """)

# Model Registry tab
with tabs[1]:
    # Mark as visited
    st.session_state['visited_Model_Registry'] = True
    
    custom_header("SageMaker Model Registry")
    
    st.markdown("""
    SageMaker Model Registry provides a central repository for organizing and managing your machine learning models.
    It helps track model versions, approval status, and deployment information, making it easier to collaborate and
    manage models throughout their lifecycle.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("images/model_registry.png", caption="SageMaker Model Registry")
        st.markdown("""
        Amazon SageMaker Model Registry is a feature that helps you:
        
        - Catalog models for production
        - Manage model versions
        - Track model approval status
        - Associate metadata (metrics, artifacts) with models
        - Deploy approved models to production
        - Create model lineage
        """)
    
    with col2:
        info_box("""
        **Key Benefits of Model Registry:**
        
        - **Centralized repository** for trained models
        - **Version control** for model artifacts
        - **Approval workflow** for model governance
        - **Metadata tracking** for model evaluation metrics
        - **Simplified deployment** from registry to production
        - **Cross-account model sharing** capabilities
        - **Integration** with SageMaker Pipelines and MLOps workflows
        """, "info")
    
    custom_header("Model Registry Structure", "sub")
    
    st.markdown("""
    The Model Registry is organized into model groups and model packages:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Groups")
        st.markdown("""
        A model group is a collection of model versions trained to solve a particular problem.
        
        **Example Model Groups:**
        - Customer Churn Prediction
        - Product Recommendation
        - Fraud Detection
        - Image Classification
        
        Each model group contains multiple model versions (packages) that represent iterations
        of models addressing the same problem.
        """)
    
    with col2:
        st.markdown("### Model Packages")
        st.markdown("""
        A model package is a versioned entity in a model group representing a trained model.
        
        **Model Package Components:**
        - Model artifacts
        - Inference code
        - Training details
        - Performance metrics
        - Approval status
        
        Model packages are versioned automatically, starting from version 1 and incrementing
        with each new registration.
        """)
    
    st.image("images/model_registry_structure.png", caption="Model Registry Structure with Model Groups and Model Packages",width=800)
    
    custom_header("Using Model Registry", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Registering Models")
        st.markdown("""
        Models can be registered in the Model Registry:
        
        1. **From SageMaker Studio**
           - Through the UI after training
           - Via model evaluation visualizations
        
        2. **Programmatically**
           - Using SageMaker Python SDK
           - Using AWS SDK (boto3)
           - Within SageMaker Pipelines
        
        3. **From Different Accounts**
           - Cross-account resource policies required
        """)
    
    with col2:
        st.markdown("### Model Approval Workflow")
        st.markdown("""
        The Model Registry supports a model approval workflow:
        
        - **PendingManualApproval**: Initial state when model is registered
        - **Approved**: Model is approved for deployment to production
        - **Rejected**: Model is rejected and should not be deployed
        
        This workflow enables governance processes where data scientists register models,
        and ML engineers or other stakeholders review and approve them before deployment.
        """)
    
    with st.expander("Code Example: Registering a Model"):
        st.code("""
        # Using SageMaker Python SDK
        from sagemaker.model import Model
        from sagemaker.model_metrics import MetricsSource, ModelMetrics
        
        # Define model metrics to store with the model
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f"s3://{bucket}/metrics/model-statistics.json",
                content_type="application/json"
            ),
            bias=MetricsSource(
                s3_uri=f"s3://{bucket}/metrics/bias-metrics.json",
                content_type="application/json"
            ),
            explainability=MetricsSource(
                s3_uri=f"s3://{bucket}/metrics/explainability-metrics.json",
                content_type="application/json"
            )
        )
        
        # Create the model
        model = Model(
            image_uri=image_uri,
            model_data=model_data_url,
            role=role,
            predictor_cls=predictor_cls
        )
        
        # Register the model in the registry
        model_package = model.register(
            model_package_group_name="CustomerChurnModels",
            model_metrics=model_metrics,
            approval_status="PendingManualApproval",
            content_types=["text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
            transform_instances=["ml.m5.xlarge"],
            description="Customer Churn Prediction Model"
        )
        """, language="python")
    
    with st.expander("Code Example: Approving and Deploying a Model"):
        st.code("""
        # Using AWS SDK (boto3) to approve a model package
        import boto3
        
        sm_client = boto3.client('sagemaker')
        
        # Update model package approval status
        sm_client.update_model_package(
            ModelPackageArn='arn:aws:sagemaker:us-west-2:012345678901:model-package/customerchurnmodels/1',
            ModelApprovalStatus='Approved'
        )
        
        # Deploy the approved model
        from sagemaker import ModelPackage
        
        model_package_arn = 'arn:aws:sagemaker:us-west-2:012345678901:model-package/customerchurnmodels/1'
        model = ModelPackage(model_package_arn=model_package_arn, role=role)
        
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge',
            endpoint_name='churn-prediction-endpoint'
        )
        """, language="python")
    
    custom_header("Model Registry in MLOps", "section")
    
    st.markdown("""
    The Model Registry plays a crucial role in MLOps workflows, bridging the gap between model development and deployment:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Integration with SageMaker Pipelines")
        st.markdown("""
        - RegisterModel step automatically registers models
        - Store training metrics with model packages
        - Track lineage from data preparation to deployment
        - Trigger deployments based on approval status
        
        With SageMaker Pipelines, you can automate the entire process from training to registration
        and deployment, creating end-to-end MLOps workflows.
        """)
    
    with col2:
        st.markdown("### Event-Driven Model Deployment")
        st.markdown("""
        - Amazon EventBridge integration
        - Events triggered on approval status changes
        - Automatic deployment of approved models
        - Notifications for model registration events
        
        This event-driven approach enables automating the deployment process
        while maintaining governance through the approval workflow.
        """)
    
    # st.image("images/model_registry_mlops.png", caption="Model Registry in MLOps Workflow")
    
    info_box("""
    **Best Practices for Model Registry:**
    
    1. **Create meaningful model group names** that reflect business problems
    2. **Store comprehensive metrics** with each model package
    3. **Include model artifacts** like explainability reports and validation data
    4. **Use approval workflows** consistently for governance
    5. **Integrate with CI/CD pipelines** for automated deployments
    6. **Document model versions** with detailed descriptions
    7. **Standardize model tags** for better searchability
    """, "tip")

# Inference Options tab
with tabs[2]:
    # Mark as visited
    st.session_state['visited_Inference_Options'] = True
    
    custom_header("SageMaker Model Inference Options")
    
    st.markdown("""
    Amazon SageMaker offers multiple deployment options for machine learning models, each designed to meet different
    inference requirements. Understanding these options helps you choose the right deployment strategy based on 
    latency, throughput, cost, and other considerations.
    """)
    
    st.markdown("### Comparing Inference Options")
    
    inference_options_df = pd.DataFrame({
        "Feature": ["Latency", "Payload Size", "Processing Time", "Persistence", "Auto-scaling", "Use Case"],
        "Real-Time Inference": ["Low (milliseconds)", "Up to 6MB", "Up to 60 seconds", "Persistent endpoint", "Yes", "Interactive applications"],
        "Asynchronous Inference": ["Near real-time", "Up to 1GB", "Up to 1 hour", "Persistent with scale-to-zero", "Yes", "Large payloads"],
        "Batch Transform": ["N/A (offline)", "GB scale", "Up to days", "Transient jobs", "N/A", "Offline prediction"],
        "Serverless Inference": ["Milliseconds (+ cold start)", "Up to 4MB", "Up to 60 seconds", "Serverless (scale-to-zero)", "Automatic", "Intermittent traffic"]
    })
    
    st.table(inference_options_df)
    
    st.markdown("")
    
    custom_header("Real-Time Inference", "sub")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/realtime_inference.png", caption="Real-Time Inference Architecture", width=600)
        st.markdown("""
        Real-time inference provides a persistent HTTPS endpoint for making predictions with low latency.
        It's ideal for applications that need immediate responses, such as user-facing applications or
        systems requiring real-time decision making.
        """)
        
        st.markdown("### Key Features:")
        st.markdown("""
        - Low-latency responses (milliseconds)
        - Persistent endpoint with HTTPS access
        - Auto-scaling capabilities
        - Multi-model/multi-container endpoint options
        - A/B testing support
        - CPU and GPU support
        - Secure with HTTPS and VPC options
        """)
    
    with col2:
        st.markdown("""
        ### Use Cases:
        - Fraud detection
        - Real-time recommendations
        - Interactive web applications
        - Mobile app backends
        - Chatbots
        
        ### Limitations:
        - Payload size up to 6MB
        - Processing time up to 60 seconds
        - Always-running instances (cost)
        """)
        
        code = """
        # Deploy a model to a real-time endpoint
        from sagemaker.sklearn import SKLearnModel
        
        model = SKLearnModel(
            model_data='s3://my-bucket/model.tar.gz',
            role=role,
            entry_point='inference.py',
            framework_version='0.23-1'
        )
        
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge'
        )
        
        # Make a prediction
        result = predictor.predict(data)
        """
        
        st.code(code, language="python")
    
    custom_header("Asynchronous Inference", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/async_inference.png", caption="Asynchronous Inference Architecture", width=600)
        st.markdown("""
        Asynchronous inference is ideal when you have large payload sizes or long processing times but don't need
        immediate responses. It queues incoming requests and processes them asynchronously, returning results
        via Amazon S3.
        """)
        
        st.markdown("### Key Features:")
        st.markdown("""
        - Support for large payloads (up to 1GB)
        - Long processing times (up to 1 hour)
        - Internal request queuing
        - Scale-to-zero capability
        - SNS notifications for job completion
        - Pay only when processing requests
        """)
    
    with col2:
        st.markdown("""
        ### Use Cases:
        - Computer vision with large images/videos
        - Credit risk prediction with complex models
        - Medical image analysis
        - Document processing
        
        ### Limitations:
        - Not suitable for real-time interactive applications
        - Doesn't support multi-model endpoints
        - No marketplace containers
        - No Inferentia instances
        """)
        
        code = """
        # Create an asynchronous inference endpoint
        from sagemaker.model import Model
        
        model = Model(
            image_uri=image_uri,
            model_data='s3://my-bucket/model.tar.gz',
            role=role
        )
        
        async_predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge',
            endpoint_name='async-endpoint',
            async_inference_config={
                "OutputConfig": {
                    "S3OutputPath": "s3://my-bucket/async-results/"
                },
                "NotificationConfig": {
                    "SuccessTopic": "arn:aws:sns:region:account:success-topic",
                    "ErrorTopic": "arn:aws:sns:region:account:error-topic"
                }
            }
        )
        
        # Invoke async endpoint
        response = async_predictor.predict_async(
            data='s3://my-bucket/input/data.csv',
            input_content_type='text/csv'
        )
        """
        
        st.code(code, language="python")
    
    custom_header("Batch Transform", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/batch_transform.png", caption="Batch Transform Architecture", width=600)
        st.markdown("""
        Batch transform is used for offline processing of large datasets when you don't need a
        persistent endpoint. It processes complete datasets and stores results in Amazon S3.
        """)
        
        st.markdown("### Key Features:")
        st.markdown("""
        - Process entire datasets at once
        - Support for very large datasets (GB scale)
        - Long-running jobs (hours or days)
        - No persistent endpoint
        - Cost-effective for infrequent processing
        - Automatic resource provisioning and cleanup
        """)
    
    with col2:
        st.markdown("""
        ### Use Cases:
        - Weekly inventory predictions
        - Periodic churn analysis
        - Bulk document processing
        - Predictive maintenance
        - Data pre-processing
        
        ### Benefits:
        - No need to maintain persistent endpoints
        - Cost-effective for infrequent batch jobs
        - Suitable for very large datasets
        - Simple API for processing entire datasets
        """)
        
        code = """
        # Create a batch transform job
        from sagemaker.sklearn import SKLearnModel
        
        model = SKLearnModel(
            model_data='s3://my-bucket/model.tar.gz',
            role=role,
            entry_point='inference.py',
            framework_version='0.23-1'
        )
        
        transformer = model.transformer(
            instance_count=1,
            instance_type='ml.m5.xlarge',
            output_path='s3://my-bucket/batch-results/'
        )
        
        transformer.transform(
            data='s3://my-bucket/batch-data/',
            content_type='text/csv',
            split_type='Line'
        )
        """
        
        st.code(code, language="python")
    
    custom_header("Serverless Inference", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/serverless_inference.png", caption="Serverless Inference Architecture", width=600)
        st.markdown("""
        Serverless inference automatically provisions and scales compute resources as needed. It's ideal for
        workloads with intermittent or unpredictable traffic patterns, as you only pay for the compute resources
        used during inference.
        """)
        
        st.markdown("### Key Features:")
        st.markdown("""
        - No infrastructure management
        - Automatic scaling (including scale-to-zero)
        - Pay-per-use pricing
        - Handles intermittent traffic
        - Configurable memory sizes
        - Built-in high availability
        """)
    
    with col2:
        st.markdown("""
        ### Use Cases:
        - Chatbots
        - Document processing applications
        - Infrequently used models
        - Dev/test environments
        - Applications with unpredictable traffic
        
        ### Limitations:
        - Cold start latency
        - Payload size up to 4MB
        - Processing time up to 60 seconds
        - CPU only (no GPU)
        - Maximum 6GB memory
        """)
        
        code = """
        # Deploy a model to a serverless endpoint
        from sagemaker import Model
        
        model = Model(
            image_uri=image_uri,
            model_data='s3://my-bucket/model.tar.gz',
            role=role
        )
        
        serverless_predictor = model.deploy(
            serverless_inference_config={
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 20
            }
        )
        
        # Make a prediction
        result = serverless_predictor.predict(data)
        """
        
        st.code(code, language="python")
    
    custom_header("Decision Flow for Choosing Inference Options", "sub")
    
    st.image("images/inference_decision_flow.png", caption="Decision Flow for Selecting the Right Inference Option", width=600)
    
    st.markdown("""
    When deciding which inference option to use, consider these key questions:
    
    1. **Does your workload need to return an inference for each request?**
       - If no, consider Batch Transform.
       
    2. **Would it be helpful to queue requests due to longer processing times or larger payloads?**
       - If yes, consider Asynchronous Inference.
       
    3. **Does your workload have intermittent traffic patterns or periods of no traffic?**
       - If yes, consider Serverless Inference.
       
    4. **Does your workload have sustained traffic and need lower and consistent latency?**
       - If yes, use Real-Time Inference.
    """)
    
    custom_header("Advanced Hosting Options", "section")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Multi-Model Endpoints")
        st.image("images/multi_model.png", width=400)
        st.markdown("""
        Host multiple models behind a single endpoint:
        
        - **Cost-efficient**: Share compute resources among models
        - **Dynamic loading**: Models loaded on-demand
        - **Memory management**: Automatic caching and unloading
        - **Best for**: Similar models in size and latency
        - **Example**: Separate models for different store locations
        """)
    
    with col2:
        st.markdown("### Multi-Container Endpoints")
        st.image("images/multi_container.png", width=400)
        st.markdown("""
        Deploy multiple containers on a single endpoint:
        
        - **Multiple frameworks**: Mix different ML frameworks
        - **Direct invocation**: Choose which container handles the request
        - **Always running**: All containers run simultaneously
        - **No cold starts**: Unlike multi-model endpoints
        - **Example**: Hosting PyTorch, TensorFlow, and HuggingFace models
        """)
    
    with col3:
        st.markdown("### Inference Pipelines")
        st.image("images/inference_pipeline.png", width=400)
        st.markdown("""
        Chain multiple containers for sequential processing:
        
        - **Pre/post-processing**: Include data processing steps
        - **Linear sequence**: 2-15 containers in a sequence
        - **End-to-end**: From raw data to final prediction
        - **Fully managed**: No need to manage inter-container communication
        - **Example**: Data preprocessing + prediction + explanation
        """)
    
    with st.expander("Code Example: Multi-Model Endpoint"):
        st.code("""
        # Deploy a multi-model endpoint
        from sagemaker.multidatamodel import MultiDataModel
        
        # Create model
        model = MultiDataModel(
            name='multi-model-example',
            model_data_prefix='s3://my-bucket/models/',
            image_uri='<ecr-image-uri-with-multi-model-support>',
            role=role
        )
        
        # Deploy to endpoint
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge'
        )
        
        # Invoke specific model
        response = predictor.predict(
            target_model='model1.tar.gz',
            data=payload
        )
        """, language="python")
    
    with st.expander("Code Example: Multi-Container Endpoint"):
        st.code("""
        # Deploy a multi-container endpoint
        from sagemaker.multidatamodel import MultiDataModel
        from sagemaker.sklearn import SKLearnModel
        from sagemaker.xgboost import XGBoostModel
        
        # Create first model
        sklearn_model = SKLearnModel(
            model_data='s3://my-bucket/sklearn-model.tar.gz',
            role=role,
            entry_point='sklearn_inference.py',
            framework_version='0.23-1'
        )
        
        # Create second model
        xgboost_model = XGBoostModel(
            model_data='s3://my-bucket/xgboost-model.tar.gz',
            role=role,
            entry_point='xgboost_inference.py',
            framework_version='1.2-1'
        )
        
        # Create a pipeline model with both containers
        from sagemaker.model import Model
        
        model = Model(
            containers=[
                {'Image': sklearn_model.image_uri, 'ModelDataUrl': sklearn_model.model_data},
                {'Image': xgboost_model.image_uri, 'ModelDataUrl': xgboost_model.model_data}
            ],
            role=role
        )
        
        # Deploy model
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge'
        )
        
        # Invoke specific container
        response = predictor.predict(
            data=payload,
            target_container='1'  # Target the second container (0-indexed)
        )
        """, language="python")
    
    with st.expander("Code Example: Inference Pipeline"):
        st.code("""
        # Deploy an inference pipeline
        from sagemaker.sklearn import SKLearnModel
        from sagemaker.xgboost import XGBoostModel
        
        # Create preprocessor model
        sklearn_preprocessor = SKLearnModel(
            model_data='s3://my-bucket/preprocessor.tar.gz',
            role=role,
            entry_point='preprocessor.py',
            framework_version='0.23-1'
        )
        
        # Create prediction model
        xgboost_model = XGBoostModel(
            model_data='s3://my-bucket/xgboost-model.tar.gz',
            role=role,
            entry_point='predictor.py',
            framework_version='1.2-1'
        )
        
        # Create a pipeline with both models
        from sagemaker.pipeline import PipelineModel
        
        model = PipelineModel(
            name="inference-pipeline",
            models=[sklearn_preprocessor, xgboost_model],
            role=role
        )
        
        # Deploy pipeline
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge'
        )
        
        # Inference request is processed through the entire pipeline
        response = predictor.predict(data)
        """, language="python")

# Infrastructure tab
with tabs[3]:
    # Mark as visited
    st.session_state['visited_Infrastructure'] = True
    
    custom_header("Infrastructure for ML Deployment")
    
    st.markdown("""
    Creating and managing infrastructure is a critical part of deploying machine learning models to production.
    AWS provides tools to automate infrastructure provisioning and management, ensuring consistent, reproducible
    deployments with proper scaling capabilities.
    """)
    
    custom_header("Autoscaling Options", "sub")
    
    st.markdown("""
    Amazon SageMaker endpoints support autoscaling, which automatically adjusts the number of instances to handle
    varying traffic loads. This helps optimize cost and performance by ensuring you have the right capacity
    at the right time.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Scaling Options")
        st.markdown("""
        1. **Target Tracking Scaling**
           - Scale based on a specific CloudWatch metric
           - Set a target value for the metric
           - Automatically adjust instance count
           - Recommended for most use cases
           
        2. **Step Scaling**
           - Define multiple scaling steps
           - Adjust scale based on alarm breach size
           - More complex but offers fine-tuned control
           
        3. **Scheduled Scaling**
           - Scale based on predictable patterns
           - Configure for specific dates/times
           - Supports one-time or recurring schedules
           - Uses cron expressions for recurring schedules
           
        4. **On-demand (Manual) Scaling**
           - Manually adjust the number of instances
           - Useful for testing or predictable workloads
        """)
    
    with col2:
        # Create a chart showing autoscaling over time
        time_points = list(range(24))
        # Create a step function for demand
        demand = [10] * 6 + [20] * 3 + [35] * 4 + [25] * 3 + [40] * 2 + [30] * 3 + [15] * 3
        # Create capacity lines for different autoscaling strategies
        fixed_capacity = [30] * 24
        target_tracking = [max(10, min(d, 50)) for d in demand]
        scheduled = [15] * 6 + [30] * 12 + [15] * 6
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_points, demand, 'r-', label='Traffic Demand')
        ax.plot(time_points, fixed_capacity, 'b--', label='Fixed Capacity')
        ax.plot(time_points, target_tracking, 'g-', label='Target Tracking')
        ax.plot(time_points, scheduled, 'c-.', label='Scheduled Scaling')
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Capacity / Traffic')
        ax.set_title('Autoscaling Strategies Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        info_box("""
        **Autoscaling Best Practices:**
        
        - Use target tracking scaling for most scenarios
        - Set appropriate cooldown periods to prevent thrashing
        - Monitor scaling activities to fine-tune policies
        - Consider pre-warming for predictable traffic spikes
        - Use scheduled scaling for known traffic patterns
        """, "tip")
    
    with st.expander("Code Example: Setting Up Target Tracking Scaling"):
        st.code("""
        # Set up target tracking scaling policy for a SageMaker endpoint
        import boto3
        
        # Create application autoscaling client
        client = boto3.client('application-autoscaling')
        
        # Register a scalable target
        client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId='endpoint/my-endpoint/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=1,
            MaxCapacity=4
        )
        
        # Create a target tracking scaling policy
        client.put_scaling_policy(
            PolicyName='SageMakerEndpointTargetTracking',
            ServiceNamespace='sagemaker',
            ResourceId='endpoint/my-endpoint/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': 70.0,  # Target 70% utilization
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleOutCooldown': 300,  # 5 minutes
                'ScaleInCooldown': 300    # 5 minutes
            }
        )
        """, language="python")
    
    custom_header("AWS Container Services", "section")
    
    st.markdown("""
    AWS offers a comprehensive suite of container services for deploying machine learning models.
    These services provide flexible options for container-based deployments based on your requirements.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Image Registry")
        st.image("images/ecr.png", width=100)
        st.markdown("""
        **Amazon Elastic Container Registry (ECR)**
        
        - Fully-managed Docker container registry
        - Store, manage, and deploy container images
        - Integrated with ECS, EKS, and SageMaker
        - Secure with encryption and IAM integration
        - Supports vulnerability scanning
        - Lifecycle policies for repository management
        """)
    
    with col2:
        st.markdown("### Orchestration")
        st.image("images/ecs.png", width=100)
        st.markdown("""
        **Amazon Elastic Container Service (ECS)**
        
        - Fully managed container orchestration service
        - Simple way to run containerized applications
        - Integrates with AWS services
        - Supports both EC2 and Fargate launch types
        """)
        
        st.image("images/eks.png", width=100)
        st.markdown("""
        **Amazon Elastic Kubernetes Service (EKS)**
        
        - Managed Kubernetes service
        - Run K8s without managing control plane
        - Compatible with standard Kubernetes tools
        - Support for EC2 and Fargate
        """)
    
    with col3:
        st.markdown("### Compute")
        st.image("images/ec2.png", width=100)
        st.markdown("""
        **Amazon EC2**
        
        - Manage your own virtual servers
        - Full control over instances
        - Broad selection of instance types
        - Support for custom AMIs
        """)
        st.image("images/fargate.png", width=100)
        st.markdown("""
        **AWS Fargate**
        
        - Serverless compute for containers
        - Run containers without managing servers
        - Pay only for resources used by containers
        - Automatic scaling and security patching
        """)
    
    custom_header("Infrastructure as Code (IaC)", "sub")
    
    st.markdown("""
    Infrastructure as Code (IaC) enables you to define and deploy infrastructure resources through code rather than
    manual processes. AWS offers multiple tools for implementing IaC for ML deployments.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AWS CloudFormation")
        st.image("images/cloudformation.png", width=100)
        st.markdown("""
        **Key Features:**
        
        - Declarative templates in JSON or YAML
        - Provision and manage AWS resources as a single unit
        - Track changes and roll back if needed
        - Reuse templates across regions and accounts
        - Support for parameters, conditions, and mappings
        
        **SageMaker Resources:**
        
        - AWS::SageMaker::Model
        - AWS::SageMaker::Endpoint
        - AWS::SageMaker::EndpointConfig
        - AWS::SageMaker::Pipeline
        """)
        
        with st.expander("CloudFormation Example: SageMaker Endpoint"):
            st.code("""
            AWSTemplateFormatVersion: '2010-09-09'
            Resources:
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
            
              MLModel:
                Type: AWS::SageMaker::Model
                Properties:
                  ExecutionRoleArn: !GetAtt SageMakerExecutionRole.Arn
                  PrimaryContainer:
                    Image: '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-ml-model:latest'
                    ModelDataUrl: 's3://my-bucket/model.tar.gz'
            
              EndpointConfig:
                Type: AWS::SageMaker::EndpointConfig
                Properties:
                  ProductionVariants:
                    - InitialInstanceCount: 1
                      InitialVariantWeight: 1.0
                      InstanceType: ml.m5.large
                      ModelName: !GetAtt MLModel.ModelName
                      VariantName: AllTraffic
            
              Endpoint:
                Type: AWS::SageMaker::Endpoint
                Properties:
                  EndpointConfigName: !GetAtt EndpointConfig.EndpointConfigName
            
            Outputs:
              EndpointId:
                Value: !Ref Endpoint
              EndpointName:
                Value: !GetAtt Endpoint.EndpointName
            """, language="yaml")
    
    with col2:
        st.markdown("### AWS Cloud Development Kit (CDK)")
        st.image("images/cdk.png", width=100)
        st.markdown("""
        **Key Features:**
        
        - Define infrastructure using programming languages
        - Support for TypeScript, Python, Java, C#, etc.
        - Reusable components (constructs)
        - Benefit from IDE features like auto-completion
        - Generate CloudFormation templates
        - Integration with CI/CD pipelines
        
        **Benefits for ML Workflows:**
        
        - Define conditional infrastructure based on ML parameters
        - Create reusable ML deployment patterns
        - Integration with testing and validation
        - Manage complex ML infrastructures programmatically
        """)
        
        with st.expander("AWS CDK Example: SageMaker Endpoint (Python)"):
            st.code("""
            import aws_cdk as cdk
            from aws_cdk import (
                aws_sagemaker as sagemaker,
                aws_iam as iam,
                Stack
            )
            from constructs import Construct
            
            class MLModelStack(Stack):
                def __init__(self, scope: Construct, id: str, **kwargs) -> None:
                    super().__init__(scope, id, **kwargs)
            
                    # Create IAM Role for SageMaker
                    role = iam.Role(
                        self, "SageMakerExecutionRole",
                        assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
                    )
                    role.add_managed_policy(
                        iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
                    )
                    role.add_managed_policy(
                        iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
                    )
            
                    # Create SageMaker Model
                    model = sagemaker.CfnModel(
                        self, "MLModel",
                        execution_role_arn=role.role_arn,
                        primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                            image="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-ml-model:latest",
                            model_data_url="s3://my-bucket/model.tar.gz"
                        )
                    )
            
                    # Create Endpoint Config
                    endpoint_config = sagemaker.CfnEndpointConfig(
                        self, "EndpointConfig",
                        production_variants=[
                            sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                                initial_instance_count=1,
                                initial_variant_weight=1.0,
                                instance_type="ml.m5.large",
                                model_name=model.attr_model_name,
                                variant_name="AllTraffic"
                            )
                        ]
                    )
            
                    # Create Endpoint
                    endpoint = sagemaker.CfnEndpoint(
                        self, "Endpoint",
                        endpoint_config_name=endpoint_config.attr_endpoint_config_name
                    )
            
                    # Output Endpoint Name
                    cdk.CfnOutput(
                        self, "EndpointName",
                        value=endpoint.attr_endpoint_name
                    )
            
            app = cdk.App()
            MLModelStack(app, "MLModelStack")
            app.synth()
            """, language="python")
    
    info_box("""
    **Benefits of Infrastructure as Code for ML:**
    
    - **Reproducibility**: Consistently deploy the same infrastructure for models
    - **Version control**: Track changes and roll back when needed
    - **Automation**: Integrate with CI/CD pipelines for automated deployments
    - **Documentation**: Infrastructure defined as code serves as documentation
    - **Collaboration**: Enable teams to work together on infrastructure
    - **Testing**: Test infrastructure changes before applying them
    - **Multi-environment support**: Easily deploy to dev, test, and production
    """, "success")

# Pipelines tab
with tabs[4]:
    # Mark as visited
    st.session_state['visited_Pipelines'] = True
    
    custom_header("Amazon SageMaker Pipelines")
    
    st.markdown("""
    Amazon SageMaker Pipelines is a purpose-built CI/CD service for machine learning. It helps you
    automate and manage ML workflows, from data preparation and model training to deployment and monitoring.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/sagemaker_pipelines.png", caption="SageMaker Pipelines Overview", width=400)
        st.markdown("""
        SageMaker Pipelines enables you to:
        
        - Create end-to-end ML workflows
        - Automate steps from data preparation to model deployment
        - Track lineage of models and datasets
        - Orchestrate training and deployment jobs
        - Reuse pipeline definitions across projects
        - Visualize pipeline execution in SageMaker Studio
        """)
    
    with col2:
        info_box("""
        **Key Benefits of SageMaker Pipelines:**
        
        - **Automation**: Reduce manual steps in the ML workflow
        - **Reproducibility**: Ensure consistent results
        - **Collaboration**: Share pipelines across teams
        - **Governance**: Track lineage and audit history
        - **Integration**: Works with SageMaker features
        - **Visualization**: Monitor progress in Studio UI
        - **Parameterization**: Customize pipeline executions
        """, "info")
    
    custom_header("Pipeline Structure", "sub")
    
    st.markdown("""
    A SageMaker pipeline is defined as a directed acyclic graph (DAG) of steps, where each step represents
    a specific operation in the ML workflow. Steps can have dependencies, allowing data to flow from one
    step to another.
    """)
    
    st.image("images/pipeline_example.png", caption="Example ML Pipeline Workflow", width=800)
    
    st.markdown("### Pipeline Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Processing Steps:**
        
        - **Processing Step**: Run data processing jobs
        - **Transform Step**: Run batch transform jobs
        - **Tuning Step**: Run hyperparameter tuning jobs
        - **Clarify Check Step**: Check data bias and model explainability
        
        **Model Building Steps:**
        
        - **Training Step**: Train machine learning models
        - **Create Model Step**: Create model artifacts for deployment
        - **Register Model Step**: Register models in the registry
        """)
    
    with col2:
        st.markdown("""
        **Flow Control Steps:**
        
        - **Condition Step**: Add branching logic based on conditions
        - **Fail Step**: Fail the pipeline execution explicitly
        - **Callback Step**: Integrate with external systems
        - **Lambda Step**: Run AWS Lambda functions
        
        **Custom Steps:**
        
        - **Parameter**: Define pipeline parameters
        - **Properties**: Access step properties
        - **JsonGet**: Extract information from JSON
        """)
    
    custom_header("Model Build Workflow Example", "section")
    
    st.markdown("""
    Let's examine a typical model build workflow using SageMaker Pipelines that includes:
    - Data processing
    - Model training
    - Model evaluation
    - Conditional model registration based on accuracy
    - Bias and explainability analysis
    """)
    
    with st.expander("Code Example: SageMaker Pipeline Definition"):
        st.code("""
        import boto3
        import sagemaker
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ClarifyCheckStep
        from sagemaker.workflow.step_collections import RegisterModel
        from sagemaker.workflow.conditions import ConditionGreaterThan
        from sagemaker.workflow.condition_step import ConditionStep
        from sagemaker.workflow.parameters import ParameterString, ParameterFloat
        from sagemaker.sklearn.processing import SKLearnProcessor
        from sagemaker.sklearn.estimator import SKLearn
        from sagemaker.clarify.processor import SageMakerClarifyProcessor
        
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        
        # Define pipeline parameters
        input_data = ParameterString(
            name="InputData",
            default_value=f"s3://{sagemaker_session.default_bucket()}/data/raw"
        )
        model_approval_status = ParameterString(
            name="ModelApprovalStatus",
            default_value="PendingManualApproval"
        )
        min_accuracy = ParameterFloat(
            name="MinAccuracy",
            default_value=0.80
        )
        
        # Define processing step for data preparation
        sklearn_processor = SKLearnProcessor(
            framework_version="0.23-1",
            role=role,
            instance_type="ml.m5.xlarge",
            instance_count=1
        )
        
        processing_step = ProcessingStep(
            name="PreprocessData",
            processor=sklearn_processor,
            inputs=[
                sagemaker.processing.ProcessingInput(
                    source=input_data,
                    destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/output/train",
                    destination=f"s3://{sagemaker_session.default_bucket()}/pipeline/train"
                ),
                sagemaker.processing.ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/output/test",
                    destination=f"s3://{sagemaker_session.default_bucket()}/pipeline/test"
                )
            ],
            code="preprocessing.py"
        )
        
        # Define training step
        sklearn_estimator = SKLearn(
            entry_point="train.py",
            framework_version="0.23-1",
            instance_type="ml.m5.xlarge",
            role=role,
            output_path=f"s3://{sagemaker_session.default_bucket()}/pipeline/model"
        )
        
        training_step = TrainingStep(
            name="TrainModel",
            estimator=sklearn_estimator,
            inputs={
                "train": sagemaker.inputs.TrainingInput(
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
                )
            }
        )
        
        # Define evaluation step
        evaluation_processor = SKLearnProcessor(
            framework_version="0.23-1",
            role=role,
            instance_type="ml.m5.xlarge",
            instance_count=1
        )
        
        evaluation_step = ProcessingStep(
            name="EvaluateModel",
            processor=evaluation_processor,
            inputs=[
                sagemaker.processing.ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"
                ),
                sagemaker.processing.ProcessingInput(
                    source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=f"s3://{sagemaker_session.default_bucket()}/pipeline/evaluation"
                )
            ],
            code="evaluate.py"
        )
        
        # Define model registration step
        register_model_step = RegisterModel(
            name="RegisterModel",
            estimator=sklearn_estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name="ChurnPredictionModels",
            approval_status=model_approval_status
        )
        
        # Define conditional step for model registration
        condition_step = ConditionStep(
            name="CheckAccuracy",
            conditions=[
                ConditionGreaterThan(
                    left=evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    right=min_accuracy
                )
            ],
            if_steps=[register_model_step],
            else_steps=[]
        )
        
        # Define clarify check steps
        clarify_processor = SageMakerClarifyProcessor(
            role=role,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=sagemaker_session
        )
        
        bias_step = ClarifyCheckStep(
            name="CheckBias",
            clarify_check_config=bias_config,
            check_job_config=check_job_config,
            model_package_group_name="ChurnPredictionModels",
            register_new_baseline=True,
            skip_check=False
        )
        
        # Define the pipeline
        pipeline = Pipeline(
            name="ChurnPredictionPipeline",
            parameters=[input_data, model_approval_status, min_accuracy],
            steps=[processing_step, training_step, evaluation_step, condition_step, bias_step]
        )
        
        # Create the pipeline in SageMaker
        pipeline.upsert(role_arn=role)
        
        # Start the pipeline execution
        execution = pipeline.start()
        """, language="python")
    
    custom_header("Pipeline Execution and Monitoring", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # st.image("images/pipelines_studio.png", caption="SageMaker Pipelines in Studio UI", width=600)
        st.markdown("""
        SageMaker Studio provides an intuitive interface for:
        
        - Viewing pipeline execution status and history
        - Inspecting individual step details and logs
        - Visualizing the pipeline DAG (directed acyclic graph)
        - Navigating to associated artifacts and models
        - Comparing different pipeline executions
        - Tracking lineage between data, training, and models
        """)
    
    with col2:
        st.markdown("### Integrating with EventBridge")
        st.image("images/eventbridge.png", width=400)
        st.markdown("""
        Amazon EventBridge can be used to trigger automated actions based on pipeline events:
        
        - Start pipeline execution on schedule
        - Trigger deployments when models are approved
        - Send notifications on pipeline completion
        - React to training failures or success
        - Integrate with other AWS services
        """)
        
        st.code("""
        {
          "source": ["aws.sagemaker"],
          "detail-type": ["SageMaker Model Package State Change"],
          "detail": {
            "ModelPackageGroupName": ["ChurnPredictionModels"],
            "ModelApprovalStatus": ["Approved"]
          }
        }
        """, language="json")
        
        st.markdown("""
        This EventBridge rule pattern triggers when a model in the
        "ChurnPredictionModels" group is approved, allowing automated
        deployment of newly approved models.
        """)
    
    custom_header("Pipeline Best Practices", "section")
    
    st.markdown("""
    Follow these best practices when working with SageMaker Pipelines:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        info_box("""
        **Pipeline Design Practices:**
        
        - **Parameterize pipelines** for flexibility
        - **Create reusable components** for common patterns
        - **Use proper error handling** with try/except
        - **Implement conditional steps** for robustness
        - **Include validation checks** at critical points
        - **Document pipeline logic** for team knowledge
        """, "success")
    
    with col2:
        info_box("""
        **Operational Practices:**
        
        - **Version control** your pipeline definitions
        - **Test pipelines** in dev environment before production
        - **Monitor pipeline executions** for failures
        - **Set up alerts** for pipeline failures
        - **Implement retry mechanisms** for transient errors
        - **Use caching** to speed up iterative development
        """, "success")

# MLOps tab
with tabs[5]:
    # Mark as visited
    st.session_state['visited_MLOps'] = True
    
    custom_header("MLOps and CI/CD for ML")
    
    st.markdown("""
    MLOps (Machine Learning Operations) combines DevOps practices with machine learning workflows to streamline
    the development, deployment, and maintenance of ML systems. CI/CD (Continuous Integration/Continuous Delivery)
    pipelines automate the process of building, testing, and deploying ML models.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/mlops_workflow.png", caption="End-to-end MLOps Workflow")
        st.markdown("""
        MLOps addresses challenges unique to ML systems:
        
        - Managing experimental ML workflows
        - Tracking model versions and datasets
        - Ensuring reproducibility of results
        - Automating training and retraining processes
        - Monitoring model performance in production
        - Handling model drift and retraining cycles
        - Implementing governance and compliance
        """)
    
    with col2:
        info_box("""
        **Benefits of MLOps:**
        
        - **Faster Time to Market**: Automate repetitive tasks
        - **Enhanced Collaboration**: Common tools for data scientists and engineers
        - **Higher Quality Models**: Consistent testing and validation
        - **Reduced Risk**: Systematic governance and approval
        - **Better Resource Utilization**: Optimize infrastructure usage
        - **Improved Auditability**: Track model lineage and changes
        - **Simplified Compliance**: Standardized approval workflows
        """, "success")
    
    custom_header("MLOps Practices and ML Workflow", "sub")
    
    st.markdown("""
    MLOps involves implementing end-to-end automation across the entire ML workflow, from data preparation
    to model monitoring.
    """)
    
    st.image("images/mlops_practices.png", caption="MLOps Practices and ML Workflow", width=800)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Data Pipeline")
        st.markdown("""
        - Automate data collection and ingestion
        - Version control input datasets
        - Track data transformations
        - Validate data quality and schema
        - Maintain feature consistency
        - Log data lineage for auditing
        """)
    
    with col2:
        st.markdown("### Building and Testing Pipeline")
        st.markdown("""
        - Version control model code
        - Automate model training and evaluation
        - Track experiments and hyperparameters
        - Run unit and integration tests
        - Validate model performance
        - Register validated models
        """)
    
    with col3:
        st.markdown("### Deployment Pipeline")
        st.markdown("""
        - Automate model deployment
        - Implement deployment strategies (blue-green, canary)
        - Run inference tests before production
        - Configure monitoring and alarms
        - Implement rollback capabilities
        - Update documentation automatically
        """)
    
    custom_header("MLOps-Ready Features in SageMaker", "section")
    
    st.markdown("""
    Amazon SageMaker provides several built-in features to support MLOps practices across the ML workflow:
    """)
    
    st.image("images/sagemaker_mlops_features.png", caption="MLOps-Ready Features in SageMaker", width=800)
    
    with st.expander("Data Preparation Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Amazon SageMaker Data Wrangler")
            st.markdown("""
            - Visual interface for data preparation
            - 300+ built-in transforms
            - Data quality checks
            - Export to pipeline code
            - Data flow tracking
            """)
        
        with col2:
            st.markdown("### Amazon SageMaker Feature Store")
            st.markdown("""
            - Centralized repository for features
            - Online and offline storage
            - Feature versioning and lineage
            - Point-in-time retrievals
            - Feature sharing across teams
            - Built-in monitoring capabilities
            """)
    
    with st.expander("Model Build Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Amazon SageMaker Experiments")
            st.markdown("""
            - Track experiments and trials
            - Log parameters and metrics
            - Compare training runs
            - Visualize performance metrics
            - Search and filter experiments
            - Share results with team members
            """)
        
        with col2:
            st.markdown("### Amazon SageMaker Pipelines")
            st.markdown("""
            - Define ML workflows as code
            - Orchestrate end-to-end processes
            - Automate training and deployment
            - Track metadata and lineage
            - Integrate with SageMaker features
            - Visualize pipeline execution
            """)
    
    with st.expander("Deployment Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Amazon SageMaker Model Registry")
            st.markdown("""
            - Centralized model repository
            - Model versioning
            - Approval workflow
            - Model metadata tracking
            - Deployment integration
            - Cross-account support
            """)
        
        with col2:
            st.markdown("### SageMaker Deployment Guardrails")
            st.markdown("""
            - Blue/green deployments
            - Traffic shifting strategies
            - Canary deployments
            - Auto-rollback capabilities
            - Baking period configuration
            - Production monitoring
            """)
    
    with st.expander("Monitoring Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Amazon SageMaker Model Monitor")
            st.markdown("""
            - Data quality monitoring
            - Model quality monitoring
            - Bias drift detection
            - Feature attribution drift monitoring
            - Automatic alerts for drift
            - Integration with CloudWatch
            """)
        
        with col2:
            st.markdown("### Amazon SageMaker Clarify")
            st.markdown("""
            - Bias detection in data and models
            - Model explainability
            - Feature importance analysis
            - Visualize explanations
            - Compliance documentation
            - Monitor bias drift over time
            """)
    
    custom_header("CI/CD Architecture for MLOps", "sub")
    
    st.markdown("""
    Implementing proper CI/CD pipelines for ML requires thoughtful architecture design. Below we'll explore
    two approaches: a small MLOps architecture for simpler use cases and a medium-sized architecture for
    more complex requirements.
    """)
    
    with st.expander("Small MLOps Architecture"):
        st.image("images/small_mlops_architecture.png", caption="Small MLOps Architecture", width=800)
        
        st.markdown("""
        This architecture is suitable for small teams or early ML projects:
        
        **Key Components:**
        
        1. **SageMaker Studio** - Central environment for data scientists
        2. **Version Control** - CodeCommit for code, ECR for environments, S3 for models
        3. **Automated Pipelines** - SageMaker Pipelines for end-to-end workflow
        4. **Deployment Automation** - Triggered by model approval
        5. **Auto Scaling** - For SageMaker endpoints
        6. **Canary Deployment** - For safe model updates
        7. **EventBridge** - For triggering pipeline executions
        
        **Benefits:**
        - Simple to set up and manage
        - Suitable for single-account deployments
        - Good starting point for MLOps adoption
        - Covers essential automation needs
        """)
    
    with st.expander("Medium MLOps Architecture"):
        st.image("images/medium_mlops_architecture.png", caption="Medium MLOps Architecture", width=800)
        
        st.markdown("""
        This architecture is designed for teams with more complex requirements:
        
        **Key Components:**
        
        1. **Multiple AWS Accounts** - Separate development, staging, and production
        2. **Automated Cross-Account Deployment** - Using CodePipeline and CloudFormation StackSets
        3. **Model Registry** - Central repository for model management
        4. **Data Replication** - Consistent datasets across environments
        5. **Model Monitoring** - SageMaker Model Monitor with automated retraining
        6. **Drift Detection** - Triggers automatic retraining pipelines
        7. **Testing Environment** - Dedicated staging for model validation
        
        **Benefits:**
        - Better isolation between environments
        - Enhanced security and access control
        - More robust testing and validation
        - Automated governance workflows
        - Systematic monitoring and drift handling
        """)
    
    custom_header("Deployment Strategies with SageMaker", "section")
    
    st.markdown("""
    SageMaker supports several deployment strategies that help you safely update models in production environments.
    These strategies minimize risk when deploying new models or making infrastructure changes.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Blue/Green Deployments")
        st.image("images/blue_green.png", width=350)
        st.markdown("""
        - Creates new fleet (green) alongside existing fleet (blue)
        - Allows testing of new deployment before shifting traffic
        - Enables quick rollback by routing back to blue fleet
        - Eliminates downtime during deployment
        - Different traffic shifting options available
        """)
    
    with col2:
        st.markdown("### Traffic Shifting Modes")
        st.image("images/traffic_shifting.png", width=350)
        st.markdown("""
        **All-at-once:**
        - 100% of traffic shifts at once to new fleet
        - Fastest deployment option
        - Higher risk if issues occur
        
        **Canary:**
        - Small portion of traffic (canary) sent to new fleet first
        - Validates performance before full deployment
        - Minimizes impact of potential issues
        
        **Linear:**
        - Traffic shifts gradually over multiple steps
        - More controlled rollout
        - Specified percentage shifts at each step
        """)
    
    with st.expander("Code Example: Blue/Green Deployment with Canary Traffic Shifting"):
        st.code("""
        import boto3
        
        client = boto3.client('sagemaker')
        
        response = client.update_endpoint(
            EndpointName='my-production-endpoint',
            RetainAllVariantProperties=True,
            DeploymentConfig={
                'BlueGreenUpdatePolicy': {
                    'TrafficRoutingConfiguration': {
                        'Type': 'CANARY',
                        'CanarySize': {
                            'Type': 'PERCENT',
                            'Value': 10
                        },
                        'WaitIntervalInSeconds': 300
                    },
                    'TerminationWaitInSeconds': 600
                },
                'AutoRollbackConfiguration': {
                    'Alarms': [
                        {
                            'AlarmName': 'HighErrorRate'
                        },
                        {
                            'AlarmName': 'HighLatency'
                        }
                    ]
                }
            }
        )
        """, language="python")
    
    info_box("""
    **Best Practices for Safe Deployments:**
    
    1. **Use blue/green deployments** to avoid downtime
    2. **Implement canary testing** for critical models
    3. **Configure CloudWatch alarms** to catch issues early
    4. **Set up automatic rollback** when alarms trigger
    5. **Test model behavior** before shifting all traffic
    6. **Monitor performance metrics** during deployment
    7. **Gradually increase traffic** to new models for important use cases
    """, "tip")

# Quiz tab
with tabs[6]:
    custom_header("Test Your Knowledge")
    
    st.markdown("""
    This quiz will test your understanding of the key concepts covered in Domain 3: Deployment and Orchestration of ML Workflows.
    Answer the following questions to evaluate your knowledge of model deployment and orchestration.
    """)
    
    # Define quiz questions
    questions = [
        {
            "question": "Which SageMaker inference option allows you to queue requests and is ideal for processing large payloads up to 1GB?",
            "options": [
                "Real-time Inference", 
                "Asynchronous Inference", 
                "Batch Transform", 
                "Serverless Inference"
            ],
            "correct": "Asynchronous Inference",
            "explanation": "Asynchronous Inference queues incoming requests and processes them asynchronously. It's ideal for large payload sizes (up to 1GB) and processing times up to 1 hour, making it suitable for workloads that don't require immediate responses."
        },
        {
            "question": "What is the main benefit of using SageMaker Model Registry in an MLOps workflow?",
            "options": [
                "It automatically optimizes model performance", 
                "It provides a central repository for tracking model versions and approval status", 
                "It automatically deploys models to production", 
                "It creates visualization dashboards for model metrics"
            ],
            "correct": "It provides a central repository for tracking model versions and approval status",
            "explanation": "SageMaker Model Registry serves as a central repository for organizing and managing your machine learning models. It helps track model versions, approval status, and deployment information, making it easier to collaborate and implement governance processes."
        },
        {
            "question": "In a SageMaker Pipeline, what does a ConditionStep allow you to do?",
            "options": [
                "Compare model performance between different runs", 
                "Add branching logic based on conditions to control pipeline flow", 
                "Monitor pipeline resource utilization", 
                "Automatically retrain models when performance degrades"
            ],
            "correct": "Add branching logic based on conditions to control pipeline flow",
            "explanation": "A ConditionStep in SageMaker Pipelines allows you to add branching logic based on conditions. This enables dynamic workflows where different steps are executed based on specific conditions, such as model performance metrics meeting a threshold."
        },
        {
            "question": "What is the purpose of AWS CloudFormation in an ML deployment workflow?",
            "options": [
                "To monitor model performance in production", 
                "To define and provision infrastructure resources as code", 
                "To automatically scale machine learning models", 
                "To create machine learning algorithms"
            ],
            "correct": "To define and provision infrastructure resources as code",
            "explanation": "AWS CloudFormation is an Infrastructure as Code (IaC) service that allows you to define and provision AWS infrastructure resources using declarative templates. In ML workflows, it helps automate the deployment of consistent, reproducible infrastructure for ML models."
        },
        {
            "question": "What is a key advantage of using a multi-model endpoint in SageMaker?",
            "options": [
                "It allows a single model to handle multiple types of requests", 
                "It improves model accuracy through ensemble techniques", 
                "It reduces costs by allowing multiple models to share compute resources", 
                "It enables automatic selection of the best model for each request"
            ],
            "correct": "It reduces costs by allowing multiple models to share compute resources",
            "explanation": "Multi-model endpoints allow multiple models to be hosted behind a single endpoint, sharing compute resources. This improves cost efficiency compared to deploying separate endpoints for each model, especially when the models are similar in size and latency requirements."
        },
        {
            "question": "Which deployment strategy in SageMaker gradually shifts a small percentage of traffic to the new model version before full deployment?",
            "options": [
                "Shadow deployment", 
                "Canary deployment", 
                "Rolling deployment", 
                "A/B testing"
            ],
            "correct": "Canary deployment",
            "explanation": "Canary deployment is a strategy where a small portion of traffic (the 'canary') is first sent to the new model version. This allows for validation of the new version's performance before shifting all traffic, minimizing the impact of potential issues."
        },
        {
            "question": "What AWS service can be used to trigger automated actions based on SageMaker pipeline events?",
            "options": [
                "AWS Lambda", 
                "Amazon EventBridge", 
                "Amazon SNS", 
                "AWS Step Functions"
            ],
            "correct": "Amazon EventBridge",
            "explanation": "Amazon EventBridge can be used to trigger automated actions based on events from SageMaker pipelines. It can start pipeline executions on a schedule, trigger deployments when models are approved, or send notifications when pipelines complete."
        },
        {
            "question": "Which scaling option in SageMaker is recommended for most use cases and scales based on a specific CloudWatch metric?",
            "options": [
                "Step scaling", 
                "Target tracking scaling", 
                "Scheduled scaling", 
                "Manual scaling"
            ],
            "correct": "Target tracking scaling",
            "explanation": "Target tracking scaling is recommended for most use cases. It automatically scales based on a specific CloudWatch metric, such as CPU utilization or request count, maintaining the metric close to a target value you specify."
        },
        {
            "question": "What is the main advantage of using AWS CDK over CloudFormation templates directly for ML infrastructure?",
            "options": [
                "It automatically optimizes ML models", 
                "It allows infrastructure to be defined using programming languages instead of JSON/YAML", 
                "It provides built-in ML algorithms", 
                "It eliminates the need for AWS resources"
            ],
            "correct": "It allows infrastructure to be defined using programming languages instead of JSON/YAML",
            "explanation": "AWS CDK (Cloud Development Kit) allows you to define infrastructure using programming languages like Python, TypeScript, or Java, rather than JSON or YAML templates. This enables you to use familiar programming constructs, leverage IDE features, and create reusable components."
        },
        {
            "question": "In SageMaker Pipelines, what is the purpose of the RegisterModel step?",
            "options": [
                "To train a new model version", 
                "To register a model in the SageMaker Model Registry", 
                "To validate model performance", 
                "To deploy a model to production"
            ],
            "correct": "To register a model in the SageMaker Model Registry",
            "explanation": "The RegisterModel step in SageMaker Pipelines is used to register a trained model in the SageMaker Model Registry. This step creates a new model package in the specified model group, allowing for version tracking, approval workflows, and deployment management."
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
            st.success("🎉 Perfect score! You've mastered the concepts of ML Deployment and Orchestration!")
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

# Resources tab
with tabs[7]:
    custom_header("Additional Resources")
    
    st.markdown("""
    Explore these resources to deepen your understanding of ML Deployment and Orchestration.
    These materials provide additional context and practical guidance for implementing the concepts covered in this module.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AWS Documentation")
        st.markdown("""
        - [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
        - [SageMaker Deployment Options](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model-options.html)
        - [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
        - [SageMaker Multi-Model Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)
        - [SageMaker Asynchronous Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)
        - [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
        - [AWS CloudFormation for SageMaker](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/AWS_SageMaker.html)
        - [SageMaker Deployment Guardrails](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green.html)
        - [SageMaker MLOps Projects](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-why.html)
        """)
        
        st.markdown("### AWS Blog Posts")
        st.markdown("""
        - [Implementing MLOps on AWS](https://aws.amazon.com/blogs/machine-learning/implementing-mlops-practices-with-amazon-sagemaker/)
        - [CI/CD for ML using SageMaker Pipelines](https://aws.amazon.com/blogs/machine-learning/build-a-ci-cd-pipeline-for-deploying-custom-machine-learning-models-using-amazon-sagemaker-pipelines/)
        - [Using Asynchronous Inference for Large Videos](https://aws.amazon.com/blogs/machine-learning/run-computer-vision-inference-on-large-videos-with-amazon-sagemaker-asynchronous-endpoints/)
        - [Model Deployment Best Practices](https://aws.amazon.com/blogs/machine-learning/best-practices-for-deploying-machine-learning-models-in-sagemaker/)
        - [Infrastructure as Code for ML](https://aws.amazon.com/blogs/machine-learning/infrastructure-as-code-for-amazon-sagemaker-use-cases/)
        - [MLOps with AWS CDK](https://aws.amazon.com/blogs/devops/building-mlops-pipelines-with-aws-cdk/)
        """)
    
    with col2:
        st.markdown("### Training Courses")
        st.markdown("""
        - [AWS Cloud Quest: Machine Learning](https://aws.amazon.com/training/learn-about/cloud-quest/)
        - [Getting Started with AWS CloudFormation](https://explore.skillbuilder.aws/learn/course/external/view/elearning/11354/getting-started-with-aws-cloudformation)
        - [Getting Started with Amazon ECS](https://explore.skillbuilder.aws/learn/course/external/view/elearning/2751/getting-started-with-amazon-ecs)
        - [Getting Started with DevOps on AWS](https://explore.skillbuilder.aws/learn/course/external/view/elearning/2000/getting-started-with-devops-on-aws)
        - [SageMaker MLOps Technical Talk](https://aws.amazon.com/partners/training/partner-cast/)
        """)
        
        st.markdown("### Tools and Services")
        st.markdown("""
        - [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/)
        - [AWS CloudFormation](https://aws.amazon.com/cloudformation/)
        - [AWS Cloud Development Kit](https://aws.amazon.com/cdk/)
        - [Amazon EventBridge](https://aws.amazon.com/eventbridge/)
        - [AWS CodePipeline](https://aws.amazon.com/codepipeline/)
        - [Amazon Elastic Container Registry](https://aws.amazon.com/ecr/)
        - [Amazon Elastic Container Service](https://aws.amazon.com/ecs/)
        - [Amazon Elastic Kubernetes Service](https://aws.amazon.com/eks/)
        """)
    
    custom_header("Practical Guides", "sub")
    
    st.markdown("""
    ### Step-by-Step Tutorials
    
    - [Building an MLOps Pipeline on AWS](https://aws.amazon.com/getting-started/hands-on/build-machine-learning-pipeline-sagemaker-model-registry/)
    - [Deploying Models with SageMaker Pipelines](https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.ipynb)
    - [Setting Up Blue/Green Deployments](https://aws.amazon.com/blogs/machine-learning/deploy-ml-models-safely-with-new-deployment-guardrails-blue-green-deployment-on-amazon-sagemaker/)
    - [Creating Multi-model Endpoints](https://aws.amazon.com/blogs/machine-learning/save-on-inference-costs-by-using-amazon-sagemaker-multi-model-endpoints/)
    
    ### Sample Projects
    
    - [SageMaker MLOps Project Templates](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-templates.html)
    - [SageMaker Pipelines Examples on GitHub](https://github.com/aws/amazon-sagemaker-examples/tree/master/sagemaker-pipelines)
    - [MLOps Workshop](https://catalog.workshops.aws/mlops/en-US)
    - [AWS CDK Examples for ML](https://github.com/aws-samples/aws-cdk-examples)
    """)
    
    custom_header("Community Resources", "sub")
    
    st.markdown("""
    ### AWS Community Engagement
    
    - [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
    - [AWS re:Post for SageMaker](https://repost.aws/tags/TAVUGXsbkQ9kYLSfprnR7XZQ/amazon-sage-maker)
    - [SageMaker Examples GitHub Repository](https://github.com/aws/amazon-sagemaker-examples)
    - [AWS ML Community](https://aws.amazon.com/machine-learning/community/)
    - [SageMaker Studio Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/63069e26-921c-4ce1-9cc7-dd882ff62575/en-US)
    """)

# Footer
st.markdown("---")
col1, col2 = st.columns([1, 5])
with col1:
    st.image("images/aws_logo.png", width=70)
with col2:
    st.markdown("**AWS Machine Learning Engineer - Associate | Domain 3: Deployment and Orchestration**")
    st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")
# ```

# To use this application, you would need to create the following directory structure:

# ```
# - app.py (the main file with the code above)
# - images/
#   - mla_badge.png
#   - mla_badge_big.png
#   - aws_logo.png
#   - model_registry.png
#   - model_registry_structure.png
#   - model_registry_mlops.png
#   - realtime_inference.png
#   - async_inference.png
#   - batch_transform.png
#   - serverless_inference.png
#   - inference_decision_flow.png
#   - multi_model.png
#   - multi_container.png
#   - inference_pipeline.png
#   - cloudformation.png
#   - cdk.png
#   - ecr.png
#   - ecs_eks.png
#   - ec2_fargate.png
#   - sagemaker_pipelines.png
#   - pipeline_example.png
#   - pipelines_studio.png
#   - eventbridge.png
#   - mlops_workflow.png
#   - mlops_practices.png
#   - sagemaker_mlops_features.png
#   - small_mlops_architecture.png
#   - medium_mlops_architecture.png
#   - blue_green.png
#   - traffic_shifting.png
# ```

# The application follows the same UI/UX styling as the Domain 1 code, with consistent components like custom headers, info boxes, definition boxes, and expandable sections. It organizes the content into tabs covering SageMaker Model Registry, Inference Options, Infrastructure, SageMaker Pipelines, MLOps, plus a quiz to test knowledge and a resources section for further learning.