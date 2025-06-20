
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
if 'visited_Drift_Detection' not in st.session_state:
    st.session_state['visited_Drift_Detection'] = False
if 'visited_Model_Testing' not in st.session_state:
    st.session_state['visited_Model_Testing'] = False
if 'visited_Cost_Optimization' not in st.session_state:
    st.session_state['visited_Cost_Optimization'] = False
if 'visited_Security_VPC' not in st.session_state:
    st.session_state['visited_Security_VPC'] = False
if 'visited_Security_Governance' not in st.session_state:
    st.session_state['visited_Security_Governance'] = False

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
    .security-card {
        border: 1px solid #ddd;
        border-left: 4px solid #FF9900;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
        transition: transform 0.2s;
    }
    .security-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .drift-type {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
    }
    .drift-type h4 {
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

# Function to create security card
def security_card(title, description, link=None):
    link_html = f'<a href="{link}" target="_blank">Learn more</a>' if link else ''
    st.markdown(f"""
    <div class="security-card">
        <h4>{title}</h4>
        <p>{description}</p>
        {link_html}
    </div>
    """, unsafe_allow_html=True)

# Function to create drift type card
def drift_type_card(title, description):
    st.markdown(f"""<div class="drift-type">
        <h4>{title}</h4>
        <p>{description}</p>
    </div>""", unsafe_allow_html=True)

# Function to reset session - same as Domain 1
def reset_session():
    st.session_state['quiz_score'] = 0
    st.session_state['quiz_attempted'] = False
    st.session_state['name'] = ""
    st.session_state['visited_Drift_Detection'] = False
    st.session_state['visited_Model_Testing'] = False
    st.session_state['visited_Cost_Optimization'] = False
    st.session_state['visited_Security_VPC'] = False
    st.session_state['visited_Security_Governance'] = False
    st.rerun()

# Sidebar for session management - similar to Domain 1
with st.sidebar:
    st.image("images/mla_badge.png", width=150)
    st.markdown("### ML Engineer - Associate")
    st.markdown("#### Domain 4: ML Solution Monitoring, Maintenance, and Security")
    
    # If user has provided their name, greet them
    if st.session_state['name']:
        st.success(f"Welcome, {st.session_state['name']}! 👋")
    else:
        name = st.text_input("Enter your name:")
        if name:
            st.session_state['name'] = name
            st.rerun()
    
    # Reset button
    if st.button("Reset Session 🔄"):
        reset_session()
    
    # Progress tracking
    if st.session_state['name']:
        st.markdown("---")
        st.markdown("### Your Progress")
        
        # Track visited pages
        visited_pages = [page for page in ["Drift_Detection", "Model_Testing", "Cost_Optimization", "Security_VPC", "Security_Governance"] 
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
        - Understand model monitoring and drift detection
        - Implement model testing strategies
        - Optimize ML infrastructure and costs
        - Secure ML resources in a VPC
        - Implement security governance for ML
        """)

# Main content with tabs
tabs = st.tabs([
    "🏠 Home", 
    "📊 Drift Detection", 
    "🧪 Model Testing", 
    "💰 Cost Optimization", 
    "🔒 Security & VPC", 
    "📋 Security Governance",
    "❓ Quiz", 
    "📚 Resources"
])

# Home tab
with tabs[0]:
    custom_header("AWS Partner Certification Readiness")
    st.markdown("## Machine Learning Engineer - Associate")
    
    st.markdown("### Domain 4: ML Solution Monitoring, Maintenance, and Security")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        info_box("""
        This interactive e-learning application covers Domain 4 from the AWS Machine Learning Engineer - Associate certification.
        
        Domain 4 focuses on **ML Solution Monitoring, Maintenance, and Security**, covering how to monitor model performance, optimize costs, and secure ML resources.
        
        Navigate through the content using the tabs above to learn about:
        - Drift Detection and Model Monitoring
        - Model Testing Strategies
        - Cost Optimization Techniques
        - Security with VPC Integration
        - ML Governance and Compliance
        
        Test your knowledge with the quiz when you're ready!
        """, "info")
        
        st.markdown("### Learning Outcomes")
        st.markdown("""
        By the end of this module, you will be able to:
        - Detect and address different types of drift in ML models
        - Implement testing strategies for model validation
        - Monitor and optimize ML infrastructure for cost efficiency
        - Secure AWS resources with proper network isolation
        - Implement ML governance with SageMaker Role Manager and Model Cards
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
        - Model registry
        - Deployment strategies
        - Infrastructure setup
        """)
        
    with col5:
        st.markdown("#### Monitoring")
        st.markdown("""
        - **Model monitoring** ← You are here
        - **Cost optimization**
        - **Security**
        """)
    
    st.markdown("---")
    
    st.markdown("### Domain 4 Task Statements")
    
    task_col1, task_col2, task_col3 = st.columns(3)
    
    with task_col1:
        st.markdown("#### Task 4.1: Monitor Model Inference")
        st.markdown("""
        - Set up monitoring for model predictions
        - Detect and address model drift
        - Implement model testing strategies
        - Compare model versions and performance
        """)
    
    with task_col2:
        st.markdown("#### Task 4.2: Monitor and Optimize Infrastructure and Costs")
        st.markdown("""
        - Select optimal instance types
        - Implement cost optimization techniques
        - Use SageMaker Inference Recommender
        - Monitor resource utilization
        """)
    
    with task_col3:
        st.markdown("#### Task 4.3: Secure AWS Resources")
        st.markdown("""
        - Implement network isolation with VPC
        - Set up authentication and authorization
        - Protect data with encryption
        - Implement governance and compliance
        """)

# Drift Detection tab
with tabs[1]:
    # Mark as visited
    st.session_state['visited_Drift_Detection'] = True
    
    custom_header("Detecting and Addressing Drift")
    
    st.markdown("""
    Model monitoring is a critical aspect of ML operations that involves tracking the performance of 
    deployed models to ensure they continue to make accurate predictions over time. One of the most 
    important aspects of monitoring is detecting various types of drift that can degrade model performance.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### What is Model Drift?
        
        Model drift occurs when the statistical properties of the target variable, which the model is trying 
        to predict, change over time. This causes a degradation in model performance because the relationships 
        between input and output variables that the model learned during training no longer hold.
        
        Different types of drift can affect your model's performance, and it's important to monitor for 
        all of them to maintain the quality and reliability of your ML solutions.
        """)
        
        # st.image("images/model_drift.png", caption="Model Drift Over Time", width=600)
    
    with col2:
        info_box("""
        **Why Monitor for Drift?**
        
        - **Maintain prediction quality**: Ensure models continue to make accurate predictions
        - **Detect performance degradation** early before it impacts business
        - **Determine when retraining** is needed
        - **Identify data quality issues** in production data feeds
        - **Ensure regulatory compliance** with model governance requirements
        - **Build trust** in ML systems with stakeholders
        """, "tip")
    
    custom_header("Types of Drift", "sub")
    
    st.markdown("""
    There are several types of drift that can affect machine learning models. SageMaker Model Monitor can detect and alert on all of these drift types.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        drift_type_card("Data Quality Drift", """
        Changes in the distribution of input data that can degrade model accuracy. This occurs when production data differs significantly from training data, such as:
        <br>
        <ul>
            <li>Missing values appear in previously complete features</li>
            <li>New categories emerge in categorical features</li>
            <li>Numerical features shift in range or distribution</li>
            <li>Data formats or types change</li>
        </ul>
        """)
        
        drift_type_card("Model Quality Drift", """
        Changes in the relationship between input features and target predictions. This happens when:
        <br>
        <ul>
            <li>Model predictions deviate from ground truth labels</li>
            <li>Performance metrics like accuracy, precision, or recall decline</li>
            <li>Error patterns change over time</li>
            <li>The underlying real-world relationship the model learned has changed</li>
        </ul>
        """)
    
    with col2:
        drift_type_card("Bias Drift", """
        Occurs when bias in model predictions increases over time, affecting specific segments:
                                <br>
        <ul>
            <li>The fairness of predictions across different demographic groups changes</li>
            <li>Disparate impact or treatment emerges for protected classes</li>
            <li>Model performance differs significantly across demographic groups</li>
            <li>Previously balanced predictions become unbalanced</li>
        </ul>
        """)
        
        drift_type_card("Feature Attribution Drift", """
        Changes in how much individual features contribute to model predictions:
                <br>
        <ul>
            <li>Features that were important become less important</li>
            <li>Previously unimportant features gain influence</li>
            <li>The relationship between features shifts</li>
            <li>The model uses different patterns to make predictions</li>
                        </ul>
        """)
    
    custom_header("Amazon SageMaker Model Monitor", "section")
    
    st.markdown("""
    Amazon SageMaker Model Monitor automatically monitors machine learning models in production and notifies
    you when data quality issues arise. This helps detect concept drift early so you can take corrective actions,
    such as retraining models, auditing upstream systems, or fixing data quality issues.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/sagemaker_model_monitor.png", caption="SageMaker Model Monitor Workflow", width=900)
        
        st.markdown("""
        SageMaker Model Monitor works by establishing a baseline from training data, then comparing
        production data against this baseline to identify deviations. The workflow includes:
        
        1. **Capturing data**: Collecting incoming requests, predictions, and ground truth
        2. **Establishing baselines**: Creating statistical profiles of training data
        3. **Defining constraints**: Setting acceptable thresholds for deviation
        4. **Scheduling monitoring**: Running regular monitoring jobs
        5. **Analyzing results**: Detecting violations of constraints
        6. **Taking action**: Generating alerts and automating responses
        """)
    
    with col2:
        st.markdown("### Key Capabilities")
        st.markdown("""
        **Model Monitor provides four types of monitoring:**
        
        - **Data Quality Monitoring**
          - Track input feature distributions
          - Detect missing values, type mismatches
          - Compare against baseline statistics

        - **Model Quality Monitoring**
          - Evaluate predictions against ground truth
          - Track metrics like accuracy, precision, recall
          - Alert when performance declines

        - **Bias Drift Monitoring**
          - Track fairness metrics across segments
          - Monitor for emerging bias patterns
          - Ensure consistent treatment across groups

        - **Feature Attribution Drift Monitoring**
          - Track feature importance over time
          - Detect shifts in feature contributions
          - Understand changing prediction patterns
        """)
        
        info_box("""
        **Integration with AWS Services:**
        
        - **Amazon CloudWatch**: Monitoring metrics and alarms
        - **Amazon S3**: Storage for captured data and analysis results
        - **Amazon SNS**: Notifications for drift detection
        - **AWS Lambda**: Automated responses to drift
        """, "info")
    
    with st.expander("Code Example: Setting Up Model Monitor for Data Quality"):
        st.code("""
        import boto3
        import sagemaker
        from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
        from sagemaker.model_monitor.dataset_format import DatasetFormat

        # Initialize SageMaker session
        session = sagemaker.Session()
        role = sagemaker.get_execution_role()

        # Deploy model with data capture enabled
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,
            destination_s3_uri=f"s3://{session.default_bucket()}/monitor/captured-data"
        )

        # Deploy the model with data capture enabled
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.xlarge",
            data_capture_config=data_capture_config
        )

        # Create a baseline from training data
        my_monitor = DefaultModelMonitor(
            role=role,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600
        )

        # Generate statistics and suggested constraints from training data
        baseline_job = my_monitor.suggest_baseline(
            baseline_dataset=f"s3://{bucket}/path/to/baseline/dataset",
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=f"s3://{bucket}/path/to/baseline/output",
            wait=True
        )

        # Set up scheduled monitoring
        monitoring_schedule_name = "my-monitoring-schedule"
        
        my_monitor.create_monitoring_schedule(
            monitor_schedule_name=monitoring_schedule_name,
            endpoint_input=predictor.endpoint_name,
            statistics=my_monitor.baseline_statistics(),
            constraints=my_monitor.suggested_constraints(),
            schedule_cron_expression="cron(0 * ? * * *)"  # Hourly
        )
        """, language="python")
    
    custom_header("Monitoring Workflow and Best Practices", "section")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Monitoring Workflow
        
        An effective model monitoring strategy follows these steps:
        
        1. **Define monitoring objectives**
           - Determine which metrics are most relevant
           - Set appropriate thresholds for alerts
           - Establish monitoring frequency
        
        2. **Establish baselines**
           - Create statistical profiles from training data
           - Define acceptable ranges for each metric
           - Document expected behavior
        
        3. **Implement monitoring**
           - Configure data capture for endpoints
           - Set up scheduled monitoring jobs
           - Integrate with notification systems
        
        4. **Analyze and respond**
           - Review monitoring results regularly
           - Investigate alerts and violations
           - Implement corrective actions when needed
           - Retrain models when necessary
        
        5. **Continuous improvement**
           - Refine monitoring thresholds based on experience
           - Add new metrics as needed
           - Automate response workflows
        """)
    
    with col2:
        info_box("""
        **Best Practices for Model Monitoring:**
        
        - Monitor all production models, not just critical ones
        - Set appropriate alert thresholds to minimize false alarms
        - Establish clear ownership and response procedures
        - Automate routine monitoring and response tasks
        - Include monitoring metrics in MLOps dashboards
        - Document baseline assumptions for future reference
        - Plan for model updates and retraining cycles
        - Test monitoring systems themselves for reliability
        """, "success")
        
        st.markdown("### Common Monitoring Metrics")
        st.markdown("""
        **Performance Metrics:**
        - Accuracy, Precision, Recall, F1-score
        - AUC-ROC, log-loss
        - Mean squared error, RMSE
        
        **Operational Metrics:**
        - Prediction latency
        - Throughput (requests per minute)
        - Error rates and failures
        
        **Data Metrics:**
        - Distribution statistics (mean, variance)
        - Missing values percentage
        - Categorical value distributions
        - Feature correlations
        """)

# Model Testing tab
with tabs[2]:
    # Mark as visited
    st.session_state['visited_Model_Testing'] = True
    
    custom_header("Model Testing Strategies")
    
    st.markdown("""
    Testing machine learning models in production is essential to validate their performance, compare different 
    versions, and ensure they meet business requirements before full deployment. There are several strategies 
    for testing models in real-world scenarios.
    """)
    
    custom_header("Challenger/Shadow Testing", "sub")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/shadow_testing.png", caption="Shadow Testing Architecture", width=600)
        
        st.markdown("""
        In shadow testing, a challenger model is compared with the current production model without affecting 
        user experience. Production requests are captured and sent in parallel to the challenger, allowing 
        comparison of performance without risk.
        
        **How it works:**
        
        1. **Capture production traffic** - Record real user requests to the production model
        2. **Process with challenger** - Send the same requests to the challenger model
        3. **Compare results** - Analyze how the challenger would have performed
        4. **Make decisions** - Determine if the challenger is ready for production
        
        **Challenger methods:**
        - **Shadow method**: Process requests in parallel in real-time
        - **Replay method**: Record requests and process them later
        """)
    
    with col2:
        st.markdown("### Benefits of Shadow Testing")
        st.markdown("""
        - **No user impact** - Testing happens without affecting user experience
        - **Real-world data** - Uses actual production traffic
        - **Safe evaluation** - Minimizes risk during model validation
        - **Performance comparison** - Direct comparison of models on identical inputs
        - **Operational validation** - Tests both model accuracy and system performance
        """)
        
        info_box("""
        **When to Use Shadow Testing:**
        
        - When testing new model versions with significant changes
        - For high-stakes applications where errors are costly
        - When you need to validate performance on real data
        - Before making major algorithmic changes
        - To build confidence in a new model approach
        """, "tip")
    
    with st.expander("Code Example: Setting Up Shadow Testing"):
        st.code("""
        import boto3
        import json
        
        # Lambda function to implement shadow deployment
        def lambda_handler(event, context):
            # Get the input payload
            payload = json.loads(event['body'])
            
            # Initialize the SageMaker runtime client
            sagemaker_runtime = boto3.client('sagemaker-runtime')
            
            # Invoke the production endpoint
            prod_response = sagemaker_runtime.invoke_endpoint(
                EndpointName='production-endpoint',
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # Parse the production model response
            prod_result = json.loads(prod_response['Body'].read().decode())
            
            try:
                # Invoke the shadow endpoint (doesn't block the response)
                sagemaker_runtime.invoke_endpoint_async(
                    EndpointName='shadow-endpoint',
                    ContentType='application/json',
                    Body=json.dumps({
                        'payload': payload,
                        'production_result': prod_result
                    }),
                    InvocationTimeoutSeconds=60
                )
            except Exception as e:
                # Log error but don't fail the request
                print(f"Error invoking shadow endpoint: {e}")
                
            # Return the production model's response to the user
            return {
                'statusCode': 200,
                'body': json.dumps(prod_result)
            }
        """, language="python")
    
    custom_header("A/B Testing", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/ab_testing.png", caption="A/B Testing with Production Variants", width=600)
        
        st.markdown("""
        A/B testing involves deploying multiple variants of a model and routing a portion of traffic to each one. 
        This allows direct comparison of performance with real users and gradual rollout of new models.
        
        **How it works:**
        
        1. **Deploy multiple variants** - Set up multiple model versions on the same endpoint
        2. **Configure traffic distribution** - Specify how much traffic goes to each variant
        3. **Monitor performance** - Track metrics for each variant
        4. **Analyze results** - Compare performance to determine the best model
        5. **Adjust distribution** - Gradually shift more traffic to the better performing model
        """)
    
    with col2:
        st.markdown("### SageMaker Production Variants")
        st.markdown("""
        Amazon SageMaker makes A/B testing easy with production variants:
        
        - **Variant weights** - Control percentage of traffic to each model
        - **Target variants** - Route specific requests to specific models
        - **CloudWatch integration** - Monitor performance by variant
        - **Automatic scaling** - Each variant can scale independently
        - **Gradual deployment** - Safely roll out new models
        """)
        
        info_box("""
        **Benefits of A/B Testing:**
        
        - Direct comparison using live traffic
        - Gradual rollout minimizes risk
        - Statistical validation of improvements
        - Observable impact on business metrics
        - Helps build confidence in new models
        """, "info")
    
    st.markdown("### Test Models by Specifying Traffic Distribution")
    
    st.image("images/traffic_distribution.png", caption="SageMaker Endpoint with Traffic Distribution", width=800)
    
    st.markdown("""
    In the example above:
    
    - **ProductionVariant1** receives 70% of traffic
    - **ProductionVariant2** receives 20% of traffic
    - **ProductionVariant3** receives 10% of traffic
    
    Each variant can use different model versions, instance types, and instance counts, allowing for comprehensive testing of different configurations.
    """)
    
    with st.expander("Code Example: A/B Testing with SageMaker"):
        st.code("""
        import boto3
        import sagemaker
        from sagemaker.model import Model
        
        # Initialize SageMaker session
        session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        
        # Create models
        model_a = Model(
            image_uri="<ecr-image-a>",
            model_data="s3://bucket/model-a/model.tar.gz",
            role=role
        )
        
        model_b = Model(
            image_uri="<ecr-image-b>",
            model_data="s3://bucket/model-b/model.tar.gz",
            role=role
        )
        
        # Deploy both models to the same endpoint with traffic splitting
        model_a.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.xlarge",
            endpoint_name="ab-test-endpoint",
            wait=True,
            variant_name="ModelA",
            initial_weight=80
        )
        
        model_b.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.xlarge",
            endpoint_name="ab-test-endpoint",
            wait=True,
            variant_name="ModelB",
            initial_weight=20
        )
        
        # To invoke a specific variant (optional)
        sagemaker_runtime = boto3.client('runtime.sagemaker')
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName="ab-test-endpoint",
            ContentType="application/json",
            Body="...",
            TargetVariant="ModelB"  # Explicitly target ModelB
        )
        """, language="python")
    
    custom_header("Advanced Testing Strategies", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Multi-armed Bandits")
        st.markdown("""
        A more dynamic method than traditional A/B testing that uses reinforcement learning to optimize traffic allocation:
        
        - **Automatic optimization** - Dynamically shifts traffic to better performing models
        - **Exploration vs. exploitation** - Balances testing new models and using proven ones
        - **Faster convergence** - Finds winning variants more quickly than static allocation
        - **Continuous learning** - Adapts as model performance changes
        - **Reduced opportunity cost** - Minimizes impact of underperforming models
        """)
    
    with col2:
        st.markdown("### Blue/Green Deployments")
        st.markdown("""
        A deployment strategy that minimizes downtime and risk by running two identical environments:
        
        - **Two environments** - Blue (current) and Green (new)
        - **Complete testing** - Fully test new version before switching
        - **Instant rollback** - Easy to revert to previous version
        - **Zero downtime** - Users can always access one environment
        - **Validation period** - Green environment is validated before becoming primary
        """)
    
    info_box("""
    **Testing Best Practices:**
    
    1. **Start small** - Begin with a small percentage of traffic
    2. **Monitor closely** - Watch for unexpected behavior or degradation
    3. **Use relevant metrics** - Choose metrics that align with business goals
    4. **Test duration** - Run tests long enough to achieve statistical significance
    5. **Document everything** - Keep records of all tests and results
    6. **Automate testing** - Set up automated testing pipelines
    7. **Consider seasonality** - Account for time-based variations in traffic
    """, "success")

# Cost Optimization tab
with tabs[3]:
    # Mark as visited
    st.session_state['visited_Cost_Optimization'] = True
    
    custom_header("Monitoring and Optimizing Costs")
    
    st.markdown("""
    Managing and optimizing the costs of machine learning workloads is essential for sustainable AI/ML operations.
    AWS provides various tools and strategies to monitor, analyze, and optimize your ML infrastructure costs.
    """)
    
    custom_header("ML Instance Options", "sub")
    
    st.markdown("""
    Selecting the right compute instances for your machine learning workloads is critical for balancing performance and cost.
    AWS offers various instance types optimized for different ML scenarios.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### CPU Instances")
        st.image("images/cpu_instances.png", width=200)
        st.markdown("""
        **Best for:**
        - Low throughput workloads
        - Cost-sensitive applications
        - Preprocessing and lightweight inference
        - Models without GPU optimization
        
        **Examples:** C5, M5, R5 families
        
        **Characteristics:**
        - Most flexible and widely available
        - Lower cost than GPU instances
        - Good for most ML serving workloads
        """)
    
    with col2:
        st.markdown("### GPU Instances")
        st.image("images/gpu_instances.png", width=200)
        st.markdown("""
        **Best for:**
        - Deep learning models
        - High throughput requirements
        - Complex computer vision or NLP tasks
        - Batch processing of large datasets
        
        **Examples:** P3, P4, G4 families
        
        **Characteristics:**
        - Higher cost but greater computational power
        - CUDA acceleration for deep learning
        - Best for large, complex models
        - Good for training and high-performance inference
        """)
    
    with col3:
        st.markdown("### Custom Chip Instances")
        st.image("images/inferentia.png", width=200)
        st.markdown("""
        **Best for:**
        - Cost-optimized inference
        - High-throughput production deployments
        - Models optimized for specific hardware
        
        **Examples:** Inf1 (AWS Inferentia)
        
        **Characteristics:**
        - Purpose-built for ML inference
        - Higher throughput than CPU
        - Lower cost than equivalent GPU performance
        - Custom optimizations required for best results
        """)
    
    custom_header("SageMaker Inference Recommender", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/inference_recommender.png", caption="SageMaker Inference Recommender Workflow", width=600)
        
        st.markdown("""
        Amazon SageMaker Inference Recommender automates the process of selecting the best instance types and configurations
        for deploying machine learning models, helping optimize for performance and cost.
        
        **How it works:**
        
        1. **Run comprehensive load tests** across multiple instance types
        2. **Analyze performance metrics** like throughput, latency, and cost
        3. **Recommend optimal instance types** based on your requirements
        4. **Fine-tune model servers and containers** for best performance
        5. **Integrate with the model registry** for streamlined deployment
        """)
    
    with col2:
        st.markdown("### Inference Recommender Job Types")
        
        st.markdown("""
        #### Default Job (Instance Recommendations)
        
        - Runs quick load tests (completes in ~45 minutes)
        - Tests recommended instance types
        - Only requires model package ARN
        - Provides high-level recommendations
        
        #### Advanced Job (Endpoint Recommendations)
        
        - Based on custom load test parameters
        - Select specific instance types to test
        - Define custom traffic patterns
        - Specify latency and throughput requirements
        - More detailed performance analysis
        - Takes ~2 hours depending on configuration
        """)
        
        info_box("""
        **Benefits of Using Inference Recommender:**
        
        - Eliminates guesswork in instance selection
        - Reduces time spent on load testing
        - Helps find the optimal cost-performance balance
        - Provides data-driven deployment decisions
        - Integrates with SageMaker workflows
        """, "tip")
    
    with st.expander("Code Example: SageMaker Inference Recommender"):
        st.code("""
        import boto3
        import sagemaker
        from sagemaker.model import Model
        
        # Initialize SageMaker session
        session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        
        # Create an Inference Recommender client
        client = boto3.client('sagemaker')
        
        # Start a default (instance recommendation) job
        instance_recommendation_response = client.create_inference_recommendations_job(
            JobName='my-instance-recommendation-job',
            JobType='Default',
            RoleArn=role,
            InputConfig={
                'ModelPackageVersionArn': 'arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1'
            }
        )
        
        # Start an advanced (endpoint recommendation) job
        endpoint_recommendation_response = client.create_inference_recommendations_job(
            JobName='my-advanced-recommendation-job',
            JobType='Advanced',
            RoleArn=role,
            InputConfig={
                'ModelPackageVersionArn': 'arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1',
                'JobDurationInSeconds': 7200,  # 2 hours
                'TrafficPattern': {
                    'TrafficType': 'PHASES',
                    'Phases': [
                        {
                            'InitialNumberOfUsers': 1,
                            'SpawnRate': 1,
                            'DurationInSeconds': 120
                        },
                        {
                            'InitialNumberOfUsers': 5,
                            'SpawnRate': 1,
                            'DurationInSeconds': 600
                        }
                    ]
                },
                'ResourceLimit': {
                    'MaxNumberOfTests': 10
                },
                'EndpointConfigurations': [
                    {
                        'InstanceType': 'ml.c5.large'
                    },
                    {
                        'InstanceType': 'ml.c5.xlarge'
                    },
                    {
                        'InstanceType': 'ml.m5.large'
                    }
                ]
            }
        )
        
        # Describe job results
        response = client.describe_inference_recommendations_job(
            JobName='my-instance-recommendation-job'
        )
        
        # Get recommendations
        recommendations = response['InferenceRecommendations']
        for rec in recommendations:
            print(f"Instance: {rec['InstanceType']}")
            print(f"Cost: {rec['MonthlyCost']} USD")
            print(f"Throughput: {rec['Throughput']}")
            print(f"Latency P99: {rec['LatencyP99']}")
        """, language="python")
    
    custom_header("Cost Analysis Tools", "section")
    
    st.markdown("""
    AWS provides several tools to monitor, analyze, and optimize your machine learning costs.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### AWS Cost Explorer")
        st.image("images/cost_explorer.png", width=200)
        st.markdown("""
        - Visualize and analyze AWS costs over time
        - Filter and group by services, tags, or regions
        - Identify cost drivers and trends
        - Create custom reports
        - Access cost forecasting
        """)
    
    with col2:
        st.markdown("### AWS Budgets")
        st.image("images/aws_budgets.png", width=200)
        st.markdown("""
        - Set custom cost and usage budgets
        - Create alerts for budget thresholds
        - Track costs by services or tags
        - Monitor utilization targets
        - Receive notifications for potential overruns
        """)
    
    with col3:
        st.markdown("### AWS Cost & Usage Report")
        st.image("images/cost_usage_report.png", width=200)
        st.markdown("""
        - Comprehensive cost and usage data
        - Most detailed cost data available
        - Track usage at hourly or daily levels
        - Identify resource-specific costs
        - Integrate with data analytics tools
        """)
    
    with col4:
        st.markdown("### AWS Trusted Advisor")
        st.image("images/trusted_advisor.png", width=200)
        st.markdown("""
        - Recommendations for cost optimization
        - Identify idle or underutilized resources
        - Suggestions for right-sizing instances
        - Reserved instance purchase recommendations
        - Best practice guidance
        """)
    
    custom_header("SageMaker Cost Optimization Strategies", "section")
    
    st.markdown("""
    Amazon SageMaker offers several strategies to optimize costs while maintaining performance for machine learning workloads.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Endpoint Optimization Strategies")
        
        st.markdown("""
        #### Multi-Model Endpoints
        - Host multiple models in one container
        - Share compute resources across models
        - Significantly reduce hosting costs
        - Best for models using the same framework
        
        #### Multi-Container Endpoints
        - Host up to 15 distinct containers
        - No cold start unlike multi-model endpoints
        - Direct invocation of specific containers
        - Share infrastructure across teams
        
        #### Asynchronous or Serverless Inference
        - Asynchronous for large payloads and batch processing
        - Serverless for variable workloads and automatic scaling
        - Pay only for compute time used
        - Scale to zero when not in use
        """)
    
    with col2:
        st.markdown("### Hardware and Optimization Strategies")
        
        st.markdown("""
        #### AWS Inferentia
        - Purpose-built inference chips
        - Up to 2.3x higher throughput than EC2
        - Up to 70% lower cost per inference
        - Optimized for deep learning models
        
        #### SageMaker Neo
        - Automatically optimizes models for target hardware
        - Improves performance without manual tuning
        - Supports multiple frameworks
        - Reduces compute requirements
        
        #### Auto Scaling
        - Scale based on demand
        - Optimize for cost during low-traffic periods
        - Support various scaling policies
        - Maintain performance during traffic spikes
        """)
    
    info_box("""
    **Cost Optimization Best Practices:**
    
    1. **Right-size your instances** - Use Inference Recommender to find optimal instance types
    2. **Use appropriate endpoint types** - Match endpoint type to your workload patterns
    3. **Implement auto-scaling** - Scale resources based on demand
    4. **Leverage spot instances** for training - Use managed spot training for non-time-critical workloads
    5. **Monitor and analyze costs** - Regularly review costs and identify optimization opportunities
    6. **Use tagging** - Implement cost allocation tags to track spending by projects or teams
    7. **Clean up unused resources** - Remove endpoints, notebooks, and other resources when not needed
    """, "success")

# Security & VPC tab
with tabs[4]:
    # Mark as visited
    st.session_state['visited_Security_VPC'] = True
    
    custom_header("Security and VPC Integration")
    
    st.markdown("""
    Securing machine learning resources is critical to protect sensitive data, maintain compliance, and ensure 
    the integrity of your ML systems. AWS provides comprehensive security features for SageMaker, including 
    network isolation, authentication, and data protection.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # st.image("images/sagemaker_security.png", caption="SageMaker Security Features", width=600)
        
        st.markdown("""
        Amazon SageMaker offers a comprehensive set of security features to help you build secure ML workflows:
        
        - **Authentication and Authorization**: Control access with AWS IAM
        - **Data Protection**: Encrypt data at rest and in transit
        - **Infrastructure and Network Isolation**: Integrate with VPC for network security
        - **Auditability and Monitoring**: Track API calls and data access
        - **Compliance Certifications**: Meet industry standards and regulations
        """)
    
    with col2:
        info_box("""
        **Security Benefits in SageMaker:**
        
        - Built-in security from day one
        - Comprehensive access control
        - End-to-end encryption
        - Network isolation capabilities
        - Extensive logging and monitoring
        - Compliance with major standards
        - Integration with AWS security services
        """, "info")
    
    custom_header("Infrastructure and Network Isolation", "sub")
    
    st.markdown("""
    Amazon SageMaker can be integrated with your Virtual Private Cloud (VPC) to provide enhanced network security 
    and isolation for your machine learning workloads.
    """)
    
    st.markdown("### Amazon VPC and SageMaker")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Default Network Configuration:**
        
        By default, a SageMaker domain uses two Amazon VPCs:
        
        1. A **SageMaker-managed VPC** that provides direct internet access
        2. A **customer-specified VPC** that handles encrypted traffic to Amazon EFS
        
        This configuration allows SageMaker to access public internet resources while 
        securely storing your data.
        """)
        
        st.markdown("""
        **VPC-Only Mode:**
        
        For enhanced security, you can configure SageMaker to send all traffic over your specified VPC by:
        
        1. Setting the network access type to **VPC only**
        2. Providing necessary subnets and security groups
        3. Creating interface endpoints for SageMaker API, runtime, and other AWS services
        4. Configuring proper routing for all required services
        """)
    
    with col2:
        st.image("images/vpc_config.png", caption="SageMaker VPC Configuration", width=400)
        
        info_box("""
        **VPC-Only Mode Benefits:**
        
        - Enhanced security through network isolation
        - Controlled internet access for ML workloads
        - Protection of sensitive training data
        - Fine-grained network access controls
        - Compliance with strict security requirements
        """, "tip")
    
    custom_header("SageMaker Endpoints within VPC Network", "section")
    
    st.image("images/endpoint_vpc.png", caption="SageMaker Endpoints within VPC", width=800)
    
    st.markdown("""
    When using VPC interface endpoints with SageMaker:
    
    - Communication occurs entirely within AWS network via AWS PrivateLink
    - No need for internet gateway, NAT device, VPN, or Direct Connect
    - Secure access to SageMaker API and runtime from within your VPC
    - Enhanced security for ML model endpoints
    - Simplified network architecture
    """)
    
    with st.expander("Code Example: Configuring SageMaker with VPC Endpoints"):
        st.code("""
        import boto3
        import sagemaker
        
        # Create VPC endpoints for SageMaker API and Runtime
        ec2 = boto3.client('ec2')
        
        # Create VPC endpoint for SageMaker API
        sm_api_response = ec2.create_vpc_endpoint(
            VpcEndpointType='Interface',
            VpcId='vpc-12345678',
            ServiceName='com.amazonaws.us-west-2.sagemaker.api',
            SubnetIds=[
                'subnet-12345678',
                'subnet-87654321'
            ],
            SecurityGroupIds=[
                'sg-12345678'
            ],
            PrivateDnsEnabled=True
        )
        
        # Create VPC endpoint for SageMaker Runtime
        sm_runtime_response = ec2.create_vpc_endpoint(
            VpcEndpointType='Interface',
            VpcId='vpc-12345678',
            ServiceName='com.amazonaws.us-west-2.sagemaker.runtime',
            SubnetIds=[
                'subnet-12345678',
                'subnet-87654321'
            ],
            SecurityGroupIds=[
                'sg-12345678'
            ],
            PrivateDnsEnabled=True
        )
        
        # Create a SageMaker model with VPC configuration
        sagemaker_client = boto3.client('sagemaker')
        
        response = sagemaker_client.create_model(
            ModelName='vpc-isolated-model',
            PrimaryContainer={
                'Image': '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-model:latest',
                'ModelDataUrl': 's3://my-bucket/model.tar.gz',
            },
            ExecutionRoleArn='arn:aws:iam::123456789012:role/SageMakerExecutionRole',
            VpcConfig={
                'SecurityGroupIds': ['sg-12345678'],
                'Subnets': ['subnet-12345678', 'subnet-87654321']
            }
        )
        """, language="python")
    
    custom_header("SageMaker Studio in a Private VPC", "section")
    
    # st.image("images/sagemaker_studio_vpc.png", caption="SageMaker Studio in Private VPC", width=800)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        When configuring SageMaker Studio in a private VPC:
        
        1. **Studio domain** is created within your VPC
        2. **Amazon EFS** stores home directories in a private subnet
        3. **Elastic network interfaces** connect Studio to your VPC
        4. **VPC endpoints** provide access to required AWS services:
           - Amazon S3
           - SageMaker API
           - SageMaker Runtime
           - Amazon CloudWatch
        5. **Security groups** control inbound and outbound traffic
        """)
    
    with col2:
        st.markdown("### Benefits of Private VPC Setup")
        st.markdown("""
        - Complete network isolation for sensitive ML workloads
        - Fine-grained network access controls
        - Secure access to internal resources
        - Compliance with strict security policies
        - Data never traverses the public internet
        - Integration with existing security infrastructure
        """)
        
        info_box("""
        **Required VPC Endpoints for SageMaker Studio:**
        
        - com.amazonaws.region.sagemaker.api
        - com.amazonaws.region.sagemaker.runtime
        - com.amazonaws.region.notebook
        - com.amazonaws.region.s3
        - com.amazonaws.region.cloudwatch
        - com.amazonaws.region.logs
        """, "tip")

# Security Governance tab
with tabs[5]:
    # Mark as visited
    st.session_state['visited_Security_Governance'] = True
    
    custom_header("Security Governance for ML")
    
    st.markdown("""
    Implementing strong security governance for machine learning is essential to protect sensitive data, 
    ensure compliance, and maintain the integrity and trustworthiness of ML systems. AWS provides multiple 
    services and tools to help establish comprehensive security governance for your ML workflows.
    """)
    
    custom_header("Authentication and Authorization", "sub")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # st.image("images/iam_policies.png", caption="Identity-Based Policies in AWS", width=600)
        
        st.markdown("""
        AWS Identity and Access Management (IAM) enables you to securely control access to AWS services and resources.
        
        **Identity-based policies** are attached to IAM users, groups, or roles and define what actions those identities 
        can perform. These policies come in several forms:
        
        - **AWS managed policies**: Created and managed by AWS, recommended for new users
        - **Customer managed policies**: Created and managed by you for precise control
        - **Inline policies**: Embedded directly into a single user, group, or role
        
        For ML workloads, well-defined IAM policies ensure that users and services have only the permissions 
        they need to perform their tasks, following the principle of least privilege.
        """)
    
    with col2:
        st.markdown("### Identity vs. Resource-based Policies")
        
        st.markdown("""
        **Identity-based policies** specify what an identity (user/role) can do:
        - Attached to IAM identities
        - Define permissions for the identity
        - Control access across multiple resources
        
        **Resource-based policies** specify who can access a resource:
        - Attached directly to resources
        - Define which principals can perform actions
        - Useful for cross-account access
        """)
        
        # st.image("images/permissions_evaluation.png", caption="Permissions Evaluation Logic", width=400)
        
        st.markdown("""
        AWS evaluates all permissions granted by policies for at least one Allow within the same account. 
        An explicit Deny in any policy overrides an Allow.
        """)
    
    with st.expander("Example IAM Policy for ML Developers"):
        st.code("""
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:CreateTrainingJob",
                        "sagemaker:CreateModel",
                        "sagemaker:CreateEndpointConfig",
                        "sagemaker:CreateEndpoint",
                        "sagemaker:InvokeEndpoint"
                    ],
                    "Resource": "arn:aws:sagemaker:*:*:*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        "arn:aws:s3:::my-ml-bucket",
                        "arn:aws:s3:::my-ml-bucket/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "cloudwatch:PutMetricData",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "*"
                }
            ]
        }
        """, language="json")
    
    custom_header("Data Protection", "section")
    
    st.markdown("""
    Protecting sensitive data throughout the ML lifecycle is crucial for security and compliance.
    AWS provides several services to help secure your data.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        security_card("AWS Key Management Service (KMS)", """
        AWS KMS helps you create and manage cryptographic keys for data encryption. Use KMS to encrypt data in SageMaker notebooks, training jobs, models, and endpoints.
        """, "https://docs.aws.amazon.com/kms/latest/developerguide/overview.html")
    
    with col2:
        security_card("AWS Secrets Manager", """
        Securely store and manage sensitive information like API keys, database credentials, and other secrets needed for ML workflows.
        """, "https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html")
    
    with col3:
        security_card("Amazon Macie", """
        Uses machine learning to automatically discover, classify, and protect sensitive data like personally identifiable information (PII) in S3 buckets.
        """, "https://docs.aws.amazon.com/macie/latest/user/what-is-macie.html")
    
    st.markdown("### Data Encryption Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Encryption at Rest:**
        - Enable encryption for S3 buckets containing training data
        - Use KMS-managed keys for SageMaker notebook storage
        - Encrypt EBS volumes for training instances
        - Store encrypted model artifacts
        - Encrypt EFS file systems for SageMaker Studio
        """)
    
    with col2:
        st.markdown("""
        **Encryption in Transit:**
        - Use HTTPS/TLS for all API communications
        - Enable encryption for inter-node communication during distributed training
        - Secure endpoint communications with TLS
        - Use private VPC endpoints for enhanced security
        - Implement VPC flow logs to monitor network traffic
        """)
    
    custom_header("ML Governance with Amazon SageMaker", "section")
    
    st.markdown("""
    Amazon SageMaker provides several tools to help establish and maintain governance over your ML workflows, 
    including role management, model documentation, and monitoring capabilities.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### SageMaker Role Manager")
        # st.image("images/role_manager.png", width=300)
        st.markdown("""
        Amazon SageMaker Role Manager helps define least-privileged user permissions quickly:
        
        - **User Role**: Role assumed via federation to access AWS Console
        - **Execution Role**: Role assigned to Studio domain or user profile
        - Create properly scoped roles in minutes
        - Predefined personas with appropriate permissions
        - Reduce security risks from overprivileged access
        - Simplify complex IAM policy creation
        """)
    
    with col2:
        st.markdown("### SageMaker Model Cards")
        # st.image("images/model_cards.png", width=300)
        st.markdown("""
        Easily document, retrieve, and share model information:
        
        - Capture model purpose and intended uses
        - Document training data and methodology
        - Auto-capture training metrics and results
        - Visualize evaluation results
        - Track model versions and history
        - Export documentation for audits
        - Support model approval workflows
        """)
    
    with col3:
        st.markdown("### SageMaker Model Dashboard")
        # st.image("images/model_dashboard.png", width=300)
        st.markdown("""
        Unified view across all your models for auditing:
        
        - Single pane of glass for all models
        - Monitor for bias and drift
        - Get notifications on violations
        - View model lineage and history
        - Track performance metrics
        - Support governance and compliance
        - Simplify model management
        """)
    
    custom_header("Model Cards Workflow", "section")
    
    # st.image("images/model_cards_workflow.png", caption="SageMaker Model Cards Workflow", width=800)
    
    st.markdown("""
    Amazon SageMaker Model Cards provide a structured way to document ML models throughout their lifecycle:
    
    1. **Create baseline model information** at the problem framing stage
    2. **Add data preparation and feature engineering details** as you prepare your data
    3. **Include model training information** automatically populated from training jobs
    4. **Attach evaluation results** from model evaluation processes
    5. **Document deployment information** when the model goes to production
    6. **Attach external documents** for additional context and information
    
    The resulting model card provides comprehensive documentation that supports governance, 
    compliance, and collaboration requirements.
    """)
    
    custom_header("Model Dashboard Benefits", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key Features
        
        SageMaker Model Dashboard provides:
        
        - **Single view for all models**, endpoints, and batch transform jobs
        - **Performance insights** for deployed models
        - **Monitoring support** for model quality, data quality, bias drift, and feature attribution drift
        - **Alerting** for violations and issues
        - **Missing monitor detection** to ensure comprehensive monitoring
        - **Model lineage** for tracking data and training parameters
        """)
    
    with col2:
        info_box("""
        **Governance Best Practices:**
        
        1. **Document everything** - Use Model Cards for comprehensive documentation
        2. **Implement least privilege** - Use Role Manager to limit permissions
        3. **Monitor continuously** - Set up Model Monitor for all production models
        4. **Centralize visibility** - Use Model Dashboard for oversight
        5. **Establish approval workflows** - Create formal processes for model approval
        6. **Audit regularly** - Conduct periodic reviews of models and permissions
        7. **Automate compliance** - Use tools to enforce governance policies
        """, "success")

# Quiz tab
with tabs[6]:
    custom_header("Test Your Knowledge")
    
    st.markdown("""
    This quiz will test your understanding of the key concepts covered in Domain 4: ML Solution Monitoring, Maintenance, and Security.
    Answer the following questions to evaluate your knowledge of monitoring, security, and cost optimization for ML solutions.
    """)
    
    # Define quiz questions
    questions = [
        {
            "question": "Which type of drift occurs when the contribution of individual features to model predictions differs from what was observed during training?",
            "options": ["Data Quality Drift", "Model Quality Drift", "Bias Drift", "Feature Attribution Drift"],
            "correct": "Feature Attribution Drift",
            "explanation": "Feature Attribution Drift occurs when the contribution of individual features to model predictions differs from what was observed during training. This can happen when the relationship between features shifts or the model begins using different patterns to make predictions."
        },
        {
            "question": "Which testing strategy allows you to compare a new model against the production model without affecting user experience?",
            "options": ["A/B Testing", "Shadow Testing", "Multi-armed Bandit", "Blue/Green Deployment"],
            "correct": "Shadow Testing",
            "explanation": "Shadow Testing (also called Challenger testing) allows you to compare a new model with the current production model without affecting user experience. Production requests are captured and processed by both models, but only the production model's responses are returned to users."
        },
        {
            "question": "What is the main benefit of using Amazon SageMaker Inference Recommender?",
            "options": [
                "It automatically improves model accuracy", 
                "It helps select optimal instance types based on performance and cost", 
                "It reduces the need for model monitoring", 
                "It automatically handles model updates"
            ],
            "correct": "It helps select optimal instance types based on performance and cost",
            "explanation": "Amazon SageMaker Inference Recommender helps select optimal instance types by running load tests across multiple instance types and analyzing performance metrics like throughput, latency, and cost. This eliminates guesswork in selecting the right infrastructure for model deployment."
        },
        {
            "question": "In the context of SageMaker security, what does 'VPC-only mode' refer to?",
            "options": [
                "Running SageMaker only within a Virtual Private Cloud with no public internet access", 
                "Encrypting all VPC traffic with a custom encryption method", 
                "Using only VPC endpoints for SageMaker access", 
                "Creating a dedicated VPC for each SageMaker model"
            ],
            "correct": "Running SageMaker only within a Virtual Private Cloud with no public internet access",
            "explanation": "VPC-only mode refers to configuring SageMaker to send all traffic over your specified VPC with no public internet access. This enhances security by isolating SageMaker resources within your private network and requiring you to set up the necessary VPC endpoints for communication with AWS services."
        },
        {
            "question": "Which SageMaker feature helps you define least-privileged user permissions in minutes?",
            "options": ["SageMaker Model Dashboard", "SageMaker Role Manager", "SageMaker Model Cards", "SageMaker Security Groups"],
            "correct": "SageMaker Role Manager",
            "explanation": "SageMaker Role Manager helps define least-privileged user permissions in minutes by providing pre-configured role personas and predefined permissions for common ML activities. It simplifies the complex task of creating proper IAM policies for SageMaker users."
        },
        {
            "question": "Which SageMaker Model Monitor capability is used to detect when model predictions deviate from ground truth labels?",
            "options": ["Data Quality Monitoring", "Model Quality Monitoring", "Bias Drift Monitoring", "Feature Attribution Drift Monitoring"],
            "correct": "Model Quality Monitoring",
            "explanation": "Model Quality Monitoring is used to detect when model predictions deviate from ground truth labels. It evaluates the model's predictions against actual outcomes over time and alerts when performance metrics like accuracy, precision, or recall decline."
        },
        {
            "question": "Which of the following is a cost optimization strategy for SageMaker endpoints?",
            "options": [
                "Using only GPU instances for all models", 
                "Deploying each model to its own endpoint", 
                "Using Multi-Model Endpoints to host multiple models on shared resources", 
                "Always using the largest instance type available"
            ],
            "correct": "Using Multi-Model Endpoints to host multiple models on shared resources",
            "explanation": "Using Multi-Model Endpoints is a cost optimization strategy that allows multiple models to share compute resources. This improves resource utilization and can significantly reduce hosting costs compared to deploying each model to its own endpoint, especially for models that don't receive constant traffic."
        },
        {
            "question": "When testing a new model variant using A/B testing in SageMaker, what parameter allows you to control the percentage of traffic each variant receives?",
            "options": ["InitialWeight", "TrafficDistribution", "ProductionPercentage", "VariantSplit"],
            "correct": "InitialWeight",
            "explanation": "In SageMaker's A/B testing implementation, the InitialWeight parameter controls the percentage of traffic each production variant receives. For example, setting variants with weights of 80 and 20 would direct approximately 80% of traffic to the first variant and 20% to the second."
        },
        {
            "question": "Which AWS service would you use to automatically discover, classify, and protect sensitive data like personally identifiable information (PII) in your ML training datasets?",
            "options": ["AWS Shield", "Amazon Inspector", "Amazon Macie", "AWS Config"],
            "correct": "Amazon Macie",
            "explanation": "Amazon Macie uses machine learning to automatically discover, classify, and protect sensitive data like personally identifiable information (PII), protected health information (PHI), and financial data. It's particularly useful for scanning S3 buckets that contain training data before using them to train ML models."
        },
        {
            "question": "What is the main purpose of Amazon SageMaker Model Cards?",
            "options": [
                "To optimize model inference performance", 
                "To document model information throughout the ML lifecycle", 
                "To automatically detect model drift", 
                "To secure model endpoints in a VPC"
            ],
            "correct": "To document model information throughout the ML lifecycle",
            "explanation": "Amazon SageMaker Model Cards provide a structured way to document ML models throughout their lifecycle, including problem framing, data preparation, model training, evaluation, and deployment. This documentation supports governance, compliance, and collaboration requirements."
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
            st.success("🎉 Perfect score! You've mastered the concepts of ML Solution Monitoring, Maintenance, and Security!")
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
    Explore these resources to deepen your understanding of ML Solution Monitoring, Maintenance, and Security.
    These materials provide additional context and practical guidance for implementing the concepts covered in this module.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AWS Documentation")
        st.markdown("""
        - [SageMaker Security](https://docs.aws.amazon.com/sagemaker/latest/dg/security.html)
        - [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
        - [SageMaker Inference Recommender](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-recommender.html)
        - [A/B Testing Models](https://docs.aws.amazon.com/sagemaker/latest/dg/model-ab-testing.html)
        - [SageMaker VPC Integration](https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html)
        - [SageMaker Studio in VPC](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-vpc.html)
        - [SageMaker Identity-Based Policies](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html)
        - [SageMaker Role Manager](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)
        - [SageMaker Model Cards](https://docs.aws.amazon.com/sagemaker/latest/dg/model-cards.html)
        - [SageMaker Model Dashboard](https://docs.aws.amazon.com/sagemaker/latest/dg/model-dashboard-faqs.html)
        """)
        
        st.markdown("### AWS Blog Posts")
        st.markdown("""
        - [Detecting Data Drift using Amazon SageMaker](https://aws.amazon.com/blogs/architecture/detecting-data-drift-using-amazon-sagemaker)
        - [ML Cost Optimization Best Practices](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlcost-20.html)
        - [ML Governance with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/governance.html)
        - [Validating ML Models in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/model-validation.html)
        - [Securing SageMaker Studio Connectivity using a Private VPC](https://aws.amazon.com/jp/blogs/machine-learning/securing-amazon-sagemaker-studio-connectivity-using-a-private-vpc/)
        - [MLOps with SageMaker, CloudFormation, and CloudWatch](https://aws.amazon.com/blogs/machine-learning/the-weather-company-enhances-mlops-with-amazon-sagemaker-aws-cloudformation-and-amazon-cloudwatch/)
        """)
    
    with col2:
        st.markdown("### Training Courses")
        st.markdown("""
        - [Security, Compliance, and Governance for AI Solutions](https://explore.skillbuilder.aws/learn/course/external/view/elearning/17395/security-compliance-and-governance-for-ai-solutions)
        - [Responsible Artificial Intelligence Practices](https://explore.skillbuilder.aws/learn/course/external/view/elearning/13164/responsible-artificial-intelligence-rai-practices)
        - [AWS Identity and Access Management - Architecture and Terminology](https://explore.skillbuilder.aws/learn/course/external/view/elearning/90/aws-identity-and-access-management-iam-architecture-and-terminology)
        - [AWS Well-Architected Machine Learning Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html)
        """)
        
        st.markdown("### Tools and Services")
        st.markdown("""
        - [Amazon SageMaker Model Monitor](https://aws.amazon.com/sagemaker/model-monitor/)
        - [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/)
        - [AWS Budgets](https://aws.amazon.com/aws-cost-management/aws-budgets/)
        - [AWS Cost & Usage Report](https://aws.amazon.com/aws-cost-management/aws-cost-and-usage-reporting/)
        - [AWS Trusted Advisor](https://aws.amazon.com/premiumsupport/technology/trusted-advisor/)
        - [Amazon VPC](https://aws.amazon.com/vpc/)
        - [AWS Private Link](https://aws.amazon.com/privatelink/)
        - [AWS Key Management Service](https://aws.amazon.com/kms/)
        """)
    
    custom_header("Practical Guides", "sub")
    
    st.markdown("""
    ### Step-by-Step Tutorials
    
    - [Monitoring Models for Data and Quality Drift](https://aws.amazon.com/getting-started/hands-on/amazon-sagemaker-model-monitor/)
    - [Setting Up A/B Testing with SageMaker](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/multi_model_bring_your_own)
    - [Implementing SageMaker in a Private VPC](https://github.com/aws-samples/amazon-sagemaker-studio-vpc-networkfirewall)
    - [Cost Optimization for SageMaker Training and Inference](https://aws.amazon.com/blogs/machine-learning/optimizing-costs-for-machine-learning-with-amazon-sagemaker/)
    - [Using SageMaker Role Manager](https://docs.aws.amazon.com/sagemaker/latest/dg/role-manager.html)
    
    ### Sample Code and Examples
    
    - [SageMaker Model Monitor Examples on GitHub](https://github.com/aws/amazon-sagemaker-examples/tree/master/sagemaker_model_monitor)
    - [A/B Testing Models in SageMaker](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/ab_testing)
    - [Inference Recommender Examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/sagemaker-inference-recommender)
    - [SageMaker Security Examples](https://github.com/aws-samples/amazon-sagemaker-security-workshop)
    """)
    
    custom_header("Community Resources", "sub")
    
    st.markdown("""
    ### AWS Community Engagement
    
    - [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
    - [AWS re:Post for SageMaker](https://repost.aws/tags/TAVUGXsbkQ9kYLSfprnR7XZQ/amazon-sage-maker)
    - [SageMaker Examples GitHub Repository](https://github.com/aws/amazon-sagemaker-examples)
    - [AWS ML Community](https://aws.amazon.com/machine-learning/community/)
    - [AWS Security Blog](https://aws.amazon.com/blogs/security/)
    """)

# Footer
st.markdown("---")
col1, col2 = st.columns([1, 5])
with col1:
    st.image("images/aws_logo.png", width=70)
with col2:
    st.markdown("**AWS Machine Learning Engineer - Associate | Domain 4: ML Solution Monitoring, Maintenance, and Security**")
    st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")
# ```

# To use this application, you would need to create the following directory structure:

# ```
# - app.py (the main file with the code above)
# - images/
#   - mla_badge.png
#   - mla_badge_big.png
#   - aws_logo.png
#   - model_drift.png
#   - sagemaker_model_monitor.png
#   - shadow_testing.png
#   - ab_testing.png
#   - traffic_distribution.png
#   - inference_recommender.png
#   - cpu_instances.png
#   - gpu_instances.png
#   - inferentia.png
#   - cost_explorer.png
#   - aws_budgets.png
#   - cost_usage_report.png
#   - trusted_advisor.png
#   - sagemaker_security.png
#   - vpc_config.png
#   - endpoint_vpc.png
#   - sagemaker_studio_vpc.png
#   - iam_policies.png
#   - permissions_evaluation.png
#   - role_manager.png
#   - model_cards.png
#   - model_dashboard.png
#   - model_cards_workflow.png
# ```

# This application follows the same UI/UX styling as the Domain 1 code, with consistent components like custom headers, info boxes, definition boxes, and expandable sections. I've organized the content into tabs covering Drift Detection, Model Testing, Cost Optimization, Security & VPC, Security Governance, plus a quiz to test knowledge and a resources section for further learning.