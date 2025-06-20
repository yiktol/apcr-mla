
import streamlit as st
import pandas as pd
import numpy as np
import uuid
import time
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import json

# Set page configuration
st.set_page_config(
    page_title="Model Testing Strategies",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS Color Scheme
AWS_COLORS = {
    "orange": "#FF9900",
    "dark_blue": "#232F3E",
    "light_blue": "#1A73E8",
    "teal": "#00A1C9",
    "light_gray": "#F2F3F3",
    "medium_gray": "#D5DBDB",
    "dark_gray": "#545B64",
    "green": "#3B7A57",
    "red": "#D13212"
}

# Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F2F3F3;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF9900 !important;
        color: white !important;
    }
    h1, h2, h3, h4 {
        color: #232F3E;
    }
    .stButton>button {
        background-color: #FF9900;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #EC7211;
    }
    .highlight {
        background-color: #F2F3F3;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #FF9900;
    }
    .aws-card {
        border: 1px solid #D5DBDB;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: rgba(59, 122, 87, 0.1);
        border-left: 5px solid #3B7A57;
        padding: 10px;
        border-radius: 5px;
    }
    .error-box {
        background-color: rgba(209, 50, 18, 0.1);
        border-left: 5px solid #D13212;
        padding: 10px;
        border-radius: 5px;
    }
    .code-box {
        background-color: #232F3E;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        overflow-x: auto;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'questions_answered' not in st.session_state:
        st.session_state.questions_answered = [False] * 5
        
    if 'show_answers' not in st.session_state:
        st.session_state.show_answers = False

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session()
    st.experimental_rerun()

# Initialize session
initialize_session()

# Sidebar
with st.sidebar:
    st.image("https://d1.awsstatic.com/training-and-certification/certification-badges/AWS-Certified-Machine-Learning-Specialty_badge.69c72eb6587c05b613fe7ef896f79b3ce2918f90.png", width=100)
    st.title("ML Testing Strategies")
    
    st.markdown(f"**Session ID**: {st.session_state.session_id[:8]}...")
    
    if st.button("Reset Session"):
        reset_session()
        
    with st.expander("About this App", expanded=False):
        st.markdown("""
        This interactive learning application teaches model testing strategies in machine learning, 
        focusing on AWS SageMaker implementations. Explore challenger/shadow deployment models and A/B testing 
        with practical examples and visualizations. Test your knowledge with the built-in assessment.
        
        Created for AWS Partner Certification Readiness training.
        """)

# Create tabs for navigation
tabs = st.tabs([
    "🏠 Introduction", 
    "🥊 Challenger/Shadow", 
    "🔄 A/B Testing", 
    "💻 Implementation", 
    "📊 Visualization", 
    "✅ Knowledge Check"
])

# Introduction tab
with tabs[0]:
    st.title("Model Testing Strategies")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        ## Why Test Machine Learning Models?
        
        Testing machine learning models in production is critical for ensuring:
        
        - **Reliability**: Do models perform consistently in real-world scenarios?
        - **Performance**: Are predictions accurate and timely?
        - **Safety**: Do models behave as expected with edge cases?
        - **ROI**: Does the model provide business value compared to alternatives?
        
        This learning module focuses on two key testing strategies:
        """)
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            st.markdown("""
            ### 🥊 Challenger/Shadow Testing
            
            Compare a new model against current production model by shadowing incoming requests
            """)
        
        with col1b:
            st.markdown("""
            ### 🔄 A/B Testing
            
            Direct part of your traffic to different model variants to compare performance
            """)
    
    with col2:
        testing_flow = """
        graph TD
            A[Data Collection] --> B[Model Training]
            B --> C[Initial Testing]
            C --> D[Production Deployment]
            D --> E[Live Testing]
            E --> F{Performance?}
            F -->|Good| G[Full Rollout]
            F -->|Poor| H[Retrain/Adjust]
            H --> B
        """
        
        st.markdown(f"""
        ```mermaid
        {testing_flow}
        ```
        """)
        
        st.markdown("""
        ### Testing Flow
        
        The diagram shows how model testing fits into the ML workflow, with continuous evaluation and improvement.
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Key Benefits of Production Testing
    
    - **Risk Mitigation**: Safely introduce new models without disrupting service
    - **Continuous Improvement**: Compare and validate model iterations
    - **Real-world Validation**: Ensure models perform in production as they did in development
    - **Informed Decisions**: Gather data to make evidence-based deployment choices
    """)
    
    # Interactive element
    st.info("Navigate through the tabs above to learn about each testing strategy in detail!")

# Challenger/Shadow Model tab
with tabs[1]:
    st.title("Challenger/Shadow Testing")
    
    col1, col2 = st.columns([3,2])
    
    with col1:
        st.markdown("""
        ## What is Challenger/Shadow Testing?
        
        In a challenger or shadow deployment scenario, a new model (the challenger) is compared with the current 
        production model. Production requests are captured, and performance metrics are collected. The requests are:
        
        - **Shadow method**: Processed in parallel with production requests
        - **Challenger method**: Recorded and replayed later
        
        This allows you to compare how the new model would have performed without affecting the production environment.
        """)
        
        with st.expander("Benefits and Drawbacks"):
            st.markdown("""
            ### Benefits
            - Zero risk to production traffic
            - Direct performance comparison with identical input
            - No user impact during testing
            - Ideal for critical applications where errors are costly
            
            ### Drawbacks
            - Requires additional infrastructure
            - Higher computational cost (running two models)
            - Delayed feedback (especially for challenger replay)
            - Cannot measure user interaction effects
            """)
    
    with col2:
        shadow_diagram = """
        graph TD
            A[User Request] --> B[Production Model]
            A --> |Copy| C[Shadow Model]
            B --> D[Response to User]
            C --> E[Log Results]
            F[Analysis] --> |Compare| B
            F --> |Compare| C
            E --> F
        """
        
        st.markdown(f"""
        ```mermaid
        {shadow_diagram}
        ```
        """)
        
        st.markdown("""
        ### Shadow Model Flow
        
        User requests are processed by both the production model and the shadow model in parallel, 
        but only the production model's responses are returned to users.
        """)
    
    st.markdown("---")
    
    st.subheader("Interactive Example")
    
    # Simple interactive example
    st.markdown("""
    ### Simulate Shadow Testing
    
    This interactive example shows how a production model and shadow model would process the same inputs.
    Adjust the parameters to see how both models perform with different data.
    """)
    
    # Create simple interactive model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Parameters")
        num_samples = st.slider("Number of samples", 10, 100, 50)
        noise_level = st.slider("Noise level", 0.0, 2.0, 0.5)
        
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, num_samples)
    true_y = 3 + 0.5 * x
    
    # Production model (simple linear)
    production_y = 3 + 0.48 * x + np.random.normal(0, noise_level, num_samples)
    
    # Shadow model (slightly better)
    shadow_y = 3 + 0.51 * x + np.random.normal(0, noise_level * 0.8, num_samples)
    
    # Calculate metrics
    prod_mse = np.mean((true_y - production_y)**2)
    shadow_mse = np.mean((true_y - shadow_y)**2)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(x, true_y, alpha=0.6, label='Ground Truth', color='black')
    ax.plot(x, production_y, label='Production Model', color=AWS_COLORS['orange'], linewidth=2)
    ax.plot(x, shadow_y, label='Shadow Model', color=AWS_COLORS['teal'], linewidth=2)
    
    ax.set_xlabel('Input Feature')
    ax.set_ylabel('Prediction')
    ax.set_title('Production vs Shadow Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Performance Metrics")
        st.metric("Production Model MSE", round(prod_mse, 4))
        st.metric("Shadow Model MSE", round(shadow_mse, 4), 
                 delta=round(prod_mse - shadow_mse, 4), 
                 delta_color="normal" if shadow_mse < prod_mse else "inverse")
        
        if shadow_mse < prod_mse:
            st.success("The Shadow Model outperforms the Production Model!")
        else:
            st.error("The Shadow Model underperforms compared to the Production Model.")
    
    st.markdown("---")
    
    st.markdown("""
    ## Implementation Example in AWS SageMaker
    
    SageMaker makes it easy to implement shadow testing by:
    
    1. Deploying the production model to an endpoint
    2. Logging inference requests
    3. Creating a second endpoint for the shadow model
    4. Replaying logged requests to the shadow endpoint
    5. Comparing performance metrics between models
    """)
    
    with st.expander("Sample Python Code for Shadow Testing"):
        st.code("""
import boto3
import json
import numpy as np

# Set up clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
cloudwatch = boto3.client('cloudwatch')

# Production endpoint
production_endpoint = "production-model-endpoint"

# Shadow endpoint
shadow_endpoint = "shadow-model-endpoint"

# Function to process requests
def process_request(input_data):
    # Send request to production model
    prod_response = sagemaker_runtime.invoke_endpoint(
        EndpointName=production_endpoint,
        ContentType='application/json',
        Body=json.dumps(input_data)
    )
    prod_result = json.loads(prod_response['Body'].read().decode())
    
    # Also send to shadow model
    try:
        shadow_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=shadow_endpoint,
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        shadow_result = json.loads(shadow_response['Body'].read().decode())
        
        # Log differences without affecting production output
        log_model_comparison(input_data, prod_result, shadow_result)
    except Exception as e:
        print(f"Shadow invocation failed: {e}")
    
    # Return only production model result to users
    return prod_result

# Function to log comparison metrics
def log_model_comparison(input_data, prod_result, shadow_result):
    # Calculate some comparison metric
    if isinstance(prod_result, (int, float)) and isinstance(shadow_result, (int, float)):
        difference = abs(prod_result - shadow_result)
        
        # Log to CloudWatch
        cloudwatch.put_metric_data(
            Namespace='ModelComparison',
            MetricData=[
                {
                    'MetricName': 'PredictionDifference',
                    'Value': difference,
                    'Unit': 'None'
                }
            ]
        )
        """, language="python")

# A/B Testing tab
with tabs[2]:
    st.title("A/B Testing")
    
    col1, col2 = st.columns([3,2])
    
    with col1:
        st.markdown("""
        ## What is A/B Testing?
        
        A/B testing involves directing a portion of your live traffic to different model variants to compare their 
        performance in real-world conditions. Each model is called a "production variant" and traffic is split 
        between them according to defined weights.
        
        This approach allows you to:
        - Test multiple models on the same endpoint
        - Gradually roll out new models
        - Compare variants with real users and requests
        - Make data-driven decisions about which model to fully deploy
        """)
        
        with st.expander("Benefits and Drawbacks"):
            st.markdown("""
            ### Benefits
            - Real user feedback and engagement metrics
            - Gradual, controlled rollout of new models
            - Direct business impact measurement
            - Multiple variants can be tested simultaneously
            
            ### Drawbacks
            - Some users experience the potentially inferior model
            - Changes in traffic patterns can affect results
            - Requires proper statistical analysis
            - More complex to set up than shadow testing
            """)
    
    with col2:
        ab_diagram = """
        graph TD
            A[User Request Pool] --> B{Traffic Split}
            B -->|70%| C[Model A]
            B -->|30%| D[Model B]
            C --> E[Response from A]
            D --> F[Response from B]
            E --> G[Performance Metrics]
            F --> G
            G --> H[Statistical Analysis]
            H --> I[Deployment Decision]
        """
        
        st.markdown(f"""
        ```mermaid
        {ab_diagram}
        ```
        """)
        
        st.markdown("""
        ### A/B Testing Flow
        
        Traffic is split between variants, with each model handling real requests from users.
        Performance metrics are collected for comparison.
        """)
    
    st.markdown("---")
    
    st.subheader("Interactive A/B Testing Simulation")
    
    # Create interactive example
    col1, col2, col3 = st.columns([1,1,2])
    
    with col1:
        model_a_weight = st.slider("Model A Traffic %", 10, 90, 70)
        model_a_conv = st.slider("Model A Conversion Rate %", 1.0, 10.0, 5.0, 0.1)
    
    with col2:
        model_b_weight = 100 - model_a_weight
        st.metric("Model B Traffic %", model_b_weight)
        model_b_conv = st.slider("Model B Conversion Rate %", 1.0, 10.0, 6.2, 0.1)
    
    # Simulate traffic and conversions
    num_days = 14
    daily_traffic = 1000
    
    np.random.seed(42)
    days = list(range(1, num_days + 1))
    
    # Model A data
    model_a_traffic = [(model_a_weight/100) * daily_traffic for _ in range(num_days)]
    model_a_conversions = [np.random.binomial(int(traffic), model_a_conv/100) for traffic in model_a_traffic]
    model_a_rates = [conv/traffic*100 for conv, traffic in zip(model_a_conversions, model_a_traffic)]
    
    # Model B data
    model_b_traffic = [(model_b_weight/100) * daily_traffic for _ in range(num_days)]
    model_b_conversions = [np.random.binomial(int(traffic), model_b_conv/100) for traffic in model_b_traffic]
    model_b_rates = [conv/traffic*100 for conv, traffic in zip(model_b_conversions, model_b_traffic)]
    
    # Create data for plotting
    df = pd.DataFrame({
        'Day': days + days,
        'Conversion Rate': model_a_rates + model_b_rates,
        'Model': ['Model A'] * num_days + ['Model B'] * num_days,
        'Traffic': model_a_traffic + model_b_traffic
    })
    
    # Create interactive visualization
    with col3:
        chart = alt.Chart(df).mark_line(point=True).encode(
            x='Day:O',
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate (%)'),
            color=alt.Color('Model:N', scale=alt.Scale(
                domain=['Model A', 'Model B'],
                range=[AWS_COLORS['orange'], AWS_COLORS['teal']]
            )),
            strokeWidth=alt.value(3),
            tooltip=['Day', 'Model', 'Conversion Rate', 'Traffic']
        ).properties(
            title='Daily Conversion Rates by Model',
            height=300
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    
    # Show aggregate statistics
    col1, col2, col3 = st.columns(3)
    
    # Calculate totals and averages
    total_a_traffic = sum(model_a_traffic)
    total_a_conv = sum(model_a_conversions)
    avg_a_rate = total_a_conv / total_a_traffic * 100
    
    total_b_traffic = sum(model_b_traffic)
    total_b_conv = sum(model_b_conversions)
    avg_b_rate = total_b_conv / total_b_traffic * 100
    
    # Confidence calculation (simplified)
    p_value = 0.03 if abs(avg_a_rate - avg_b_rate) > 0.8 else 0.25
    
    with col1:
        st.metric("Model A Avg. Conversion", f"{avg_a_rate:.2f}%")
        st.metric("Total Model A Conversions", int(total_a_conv))
    
    with col2:
        st.metric("Model B Avg. Conversion", f"{avg_b_rate:.2f}%", 
                 delta=f"{avg_b_rate - avg_a_rate:.2f}%",
                 delta_color="normal" if avg_b_rate > avg_a_rate else "inverse")
        st.metric("Total Model B Conversions", int(total_b_conv))
    
    with col3:
        st.markdown("### Statistical Significance")
        
        if p_value < 0.05:
            st.success(f"The difference is statistically significant (p={p_value:.3f})")
            if avg_b_rate > avg_a_rate:
                st.markdown("**Recommendation**: Deploy Model B to 100% of traffic")
            else:
                st.markdown("**Recommendation**: Keep Model A as primary model")
        else:
            st.warning(f"The difference is NOT statistically significant (p={p_value:.3f})")
            st.markdown("**Recommendation**: Continue testing to gather more data")
            
    st.markdown("---")
    
    st.markdown("""
    ## Key Components of A/B Testing
    
    1. **Traffic Distribution**: Specify the percentage of traffic each model receives
    2. **Endpoint Configuration**: Multiple model variants hosted behind a single endpoint
    3. **Metrics Collection**: Monitor key performance indicators for each variant
    4. **Statistical Analysis**: Determine if differences are significant
    5. **Gradual Rollout**: Increase traffic to the better-performing model over time
    """)
    
    with st.expander("A/B Testing vs Shadow Testing: When to Use Each"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Use A/B Testing When:")
            st.markdown("""
            - You need user interaction metrics
            - The risk of incorrect predictions is low
            - You want to measure business impact directly
            - You have high traffic volume for statistical power
            - You need to test multiple variants simultaneously
            """)
            
        with col2:
            st.markdown("### Use Shadow Testing When:")
            st.markdown("""
            - You can't risk exposing users to an unproven model
            - You have a critical application where errors are costly
            - You need to validate technical performance first
            - You want to test on 100% of traffic without user impact
            - You need to debug model differences
            """)

# Implementation tab
with tabs[3]:
    st.title("Implementation in AWS SageMaker")
    
    st.markdown("""
    ## Setting Up Model Testing in AWS SageMaker
    
    Amazon SageMaker provides built-in capabilities for both shadow testing and A/B testing, making it easy to 
    implement these strategies in your ML workflow.
    """)
    
    testing_tabs = st.tabs(["A/B Testing Setup", "Shadow Testing Setup", "Code Examples"])
    
    # A/B Testing Setup
    with testing_tabs[0]:
        st.markdown("""
        ### A/B Testing Setup in SageMaker
        
        To set up A/B testing in SageMaker, you need to:
        
        1. Create multiple model variants
        2. Configure production variants with traffic allocation
        3. Deploy to a single endpoint
        4. Monitor metrics with CloudWatch
        """)
        
        st.markdown("""
        #### Endpoint Configuration with Multiple Production Variants
        """)
        
        st.image("https://d1.awsstatic.com/product-marketing/SageMaker/SageMaker_InferenceOverviewDiagram.24b2958ac18e88566f1e053d0497b6b578389c63.png", caption="SageMaker Inference Overview")
        
        st.code("""
import boto3
import json

# Create SageMaker client
sm_client = boto3.client('sagemaker')

# Create endpoint configuration with multiple variants
response = sm_client.create_endpoint_config(
    EndpointConfigName='ab-test-endpoint-config',
    ProductionVariants=[
        {
            'VariantName': 'ProductionVariant1',
            'ModelName': 'model-a',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge',
            'InitialVariantWeight': 0.7  # 70% of traffic
        },
        {
            'VariantName': 'ProductionVariant2',
            'ModelName': 'model-b',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge',
            'InitialVariantWeight': 0.3  # 30% of traffic
        }
    ]
)

# Create the endpoint
response = sm_client.create_endpoint(
    EndpointName='ab-test-endpoint',
    EndpointConfigName='ab-test-endpoint-config'
)

# Wait for endpoint creation to complete
sm_client.get_waiter('endpoint_in_service').wait(EndpointName='ab-test-endpoint')
        """, language="python")
        
        st.markdown("""
        #### Invoking the Endpoint
        
        When invoking the endpoint, requests are automatically routed based on the specified weights:
        """)
        
        st.code("""
# Invoke endpoint with automatic traffic splitting
runtime_client = boto3.client('sagemaker-runtime')

for i in range(100):  # 100 sample requests
    response = runtime_client.invoke_endpoint(
        EndpointName='ab-test-endpoint',
        ContentType='application/json',
        Body=json.dumps({'input': [1.0, 2.0, 3.0]})
    )
    result = json.loads(response['Body'].read().decode())
    # Process result...
        """, language="python")
        
        st.markdown("""
        #### Target-Specific Invocation
        
        You can also target a specific variant if needed:
        """)
        
        st.code("""
# Force invocation to specific variant
response = runtime_client.invoke_endpoint(
    EndpointName='ab-test-endpoint',
    ContentType='application/json',
    Body=json.dumps({'input': [1.0, 2.0, 3.0]}),
    TargetVariant='ProductionVariant2'  # Force to model B
)
        """, language="python")
        
    # Shadow Testing Setup
    with testing_tabs[1]:
        st.markdown("""
        ### Shadow Testing Setup in SageMaker
        
        For shadow testing, the approach involves:
        
        1. Deploy the production model to an endpoint
        2. Capture and log inference requests
        3. Deploy the shadow model to a separate endpoint
        4. Replay captured requests to the shadow endpoint
        5. Compare performance metrics
        """)
        
        st.markdown("""
        #### Capturing Inference Data
        """)
        
        st.code("""
# Enable data capture for the production endpoint
sm_client.update_endpoint_config(
    EndpointConfigName='production-endpoint-config',
    DataCaptureConfig={
        'EnableCapture': True,
        'InitialSamplingPercentage': 100,
        'DestinationS3Uri': 's3://my-bucket/captured-data/',
        'CaptureOptions': [
            {
                'CaptureMode': 'Input'
            },
            {
                'CaptureMode': 'Output'
            }
        ],
        'CaptureContentTypeHeader': {
            'CsvContentTypes': ['text/csv'],
            'JsonContentTypes': ['application/json']
        }
    }
)
        """, language="python")
        
        st.markdown("""
        #### Replaying Captured Data to Shadow Model
        """)
        
        st.code("""
import boto3
import json
import glob
import os

# Set up clients
s3_client = boto3.client('s3')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Shadow endpoint
shadow_endpoint = "shadow-model-endpoint"

# Download captured data from S3
s3_client.download_file('my-bucket', 'captured-data/requests.json', 'requests.json')

# Read captured requests
with open('requests.json', 'r') as f:
    captured_requests = json.load(f)

# Replay requests to shadow endpoint
shadow_results = []
for request in captured_requests:
    shadow_response = sagemaker_runtime.invoke_endpoint(
        EndpointName=shadow_endpoint,
        ContentType='application/json',
        Body=json.dumps(request['input'])
    )
    shadow_result = json.loads(shadow_response['Body'].read().decode())
    
    # Compare with original production output
    production_result = request['output']
    
    # Store for analysis
    shadow_results.append({
        'input': request['input'],
        'production_output': production_result,
        'shadow_output': shadow_result
    })

# Save comparison results
with open('shadow_comparison.json', 'w') as f:
    json.dump(shadow_results, f)
        """, language="python")
    
    # Code Examples
    with testing_tabs[2]:
        st.markdown("""
        ### Complete SageMaker Model Testing Example
        
        This example demonstrates a comprehensive workflow for model testing in SageMaker:
        """)
        
        st.code("""
import boto3
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime

class ModelTester:
    def __init__(self):
        self.sm_client = boto3.client('sagemaker')
        self.runtime_client = boto3.client('sagemaker-runtime')
        self.cloudwatch = boto3.client('cloudwatch')
        self.s3 = boto3.client('s3')
        
    def setup_ab_test(self, model_a_name, model_b_name, 
                     model_a_weight=0.9, model_b_weight=0.1,
                     endpoint_name='ab-test-endpoint'):
        '''
        Set up an A/B test between two models with specified traffic weights
        '''
        # Create endpoint configuration
        response = self.sm_client.create_endpoint_config(
            EndpointConfigName=f'{endpoint_name}-config',
            ProductionVariants=[
                {
                    'VariantName': 'ModelA',
                    'ModelName': model_a_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.xlarge',
                    'InitialVariantWeight': model_a_weight
                },
                {
                    'VariantName': 'ModelB',
                    'ModelName': model_b_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.xlarge',
                    'InitialVariantWeight': model_b_weight
                }
            ]
        )
        
        # Create or update endpoint
        try:
            self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            # Endpoint exists, update it
            response = self.sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=f'{endpoint_name}-config'
            )
        except self.sm_client.exceptions.ClientError:
            # Endpoint doesn't exist, create it
            response = self.sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=f'{endpoint_name}-config'
            )
        
        print(f"Waiting for endpoint {endpoint_name} to be ready...")
        self.sm_client.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} is ready!")
        
        return endpoint_name
        
    def setup_shadow_test(self, production_model_name, shadow_model_name,
                         bucket_name, capture_path,
                         production_endpoint='production-endpoint',
                         shadow_endpoint='shadow-endpoint'):
        '''
        Set up shadow testing with data capture for the production model
        '''
        # Create production endpoint with data capture
        prod_config_name = f'{production_endpoint}-config'
        self.sm_client.create_endpoint_config(
            EndpointConfigName=prod_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'Production',
                    'ModelName': production_model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.xlarge'
                }
            ],
            DataCaptureConfig={
                'EnableCapture': True,
                'InitialSamplingPercentage': 100,
                'DestinationS3Uri': f's3://{bucket_name}/{capture_path}',
                'CaptureOptions': [
                    {'CaptureMode': 'Input'},
                    {'CaptureMode': 'Output'}
                ],
                'CaptureContentTypeHeader': {
                    'JsonContentTypes': ['application/json']
                }
            }
        )
        
        # Create shadow endpoint
        shadow_config_name = f'{shadow_endpoint}-config'
        self.sm_client.create_endpoint_config(
            EndpointConfigName=shadow_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'Shadow',
                    'ModelName': shadow_model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.xlarge'
                }
            ]
        )
        
        # Create or update production endpoint
        try:
            self.sm_client.describe_endpoint(EndpointName=production_endpoint)
            self.sm_client.update_endpoint(
                EndpointName=production_endpoint,
                EndpointConfigName=prod_config_name
            )
        except:
            self.sm_client.create_endpoint(
                EndpointName=production_endpoint,
                EndpointConfigName=prod_config_name
            )
            
        # Create or update shadow endpoint
        try:
            self.sm_client.describe_endpoint(EndpointName=shadow_endpoint)
            self.sm_client.update_endpoint(
                EndpointName=shadow_endpoint,
                EndpointConfigName=shadow_config_name
            )
        except:
            self.sm_client.create_endpoint(
                EndpointName=shadow_endpoint,
                EndpointConfigName=shadow_config_name
            )
        
        # Wait for both endpoints
        self.sm_client.get_waiter('endpoint_in_service').wait(EndpointName=production_endpoint)
        self.sm_client.get_waiter('endpoint_in_service').wait(EndpointName=shadow_endpoint)
        
        return production_endpoint, shadow_endpoint
        
    def invoke_ab_test_endpoint(self, endpoint_name, data, target_variant=None):
        '''
        Invoke endpoint for A/B testing, optionally targeting a specific variant
        '''
        kwargs = {
            'EndpointName': endpoint_name,
            'ContentType': 'application/json',
            'Body': json.dumps(data)
        }
        
        if target_variant:
            kwargs['TargetVariant'] = target_variant
            
        response = self.runtime_client.invoke_endpoint(**kwargs)
        return json.loads(response['Body'].read().decode())
    
    def invoke_with_shadow(self, production_endpoint, shadow_endpoint, data):
        '''
        Invoke both production and shadow endpoints with same data
        '''
        # Production invocation (this is what user sees)
        prod_response = self.runtime_client.invoke_endpoint(
            EndpointName=production_endpoint,
            ContentType='application/json',
            Body=json.dumps(data)
        )
        prod_result = json.loads(prod_response['Body'].read().decode())
        
        # Shadow invocation (for comparison only)
        shadow_response = self.runtime_client.invoke_endpoint(
            EndpointName=shadow_endpoint,
            ContentType='application/json',
            Body=json.dumps(data)
        )
        shadow_result = json.loads(shadow_response['Body'].read().decode())
        
        # Log metrics to CloudWatch
        self._log_comparison_metrics(prod_result, shadow_result)
        
        # Return only the production result to the user
        return prod_result
        
    def _log_comparison_metrics(self, prod_result, shadow_result):
        '''
        Log comparison metrics to CloudWatch
        '''
        # Simple example for numeric outputs
        if isinstance(prod_result, (int, float)) and isinstance(shadow_result, (int, float)):
            difference = abs(prod_result - shadow_result)
            
            self.cloudwatch.put_metric_data(
                Namespace='ModelComparison',
                MetricData=[{
                    'MetricName': 'PredictionDifference',
                    'Value': difference,
                    'Unit': 'None',
                    'Timestamp': datetime.now()
                }]
            )

# Example usage
if __name__ == "__main__":
    tester = ModelTester()
    
    # Setup A/B test
    ab_endpoint = tester.setup_ab_test(
        model_a_name='linear-model', 
        model_b_name='xgboost-model',
        model_a_weight=0.8,
        model_b_weight=0.2
    )
    
    # Setup shadow test
    prod_endpoint, shadow_endpoint = tester.setup_shadow_test(
        production_model_name='current-model',
        shadow_model_name='new-model',
        bucket_name='my-ml-bucket',
        capture_path='inference-data'
    )
    
    # Example inference requests
    test_data = {'features': [1.0, 2.5, 3.2, 0.7]}
    
    # A/B test inference
    for i in range(10):
        result = tester.invoke_ab_test_endpoint(ab_endpoint, test_data)
        print(f"A/B test result: {result}")
    
    # Shadow test inference
    for i in range(10):
        result = tester.invoke_with_shadow(prod_endpoint, shadow_endpoint, test_data)
        print(f"Production result: {result}")
        """, language="python")

# Visualization tab
with tabs[4]:
    st.title("Visualizing Model Performance")
    
    st.markdown("""
    ## Performance Visualization for Model Testing
    
    Proper visualization is critical for interpreting the results of model testing. Different metrics and 
    visualization techniques can help you understand how your models are performing and make data-driven decisions.
    """)
    
    viz_tabs = st.tabs(["Metrics Dashboard", "Conversion Funnel", "Time Series Analysis", "Statistical Significance"])
    
    # Metrics Dashboard
    with viz_tabs[0]:
        st.markdown("""
        ### Metrics Dashboard Example
        
        A comprehensive dashboard can help you monitor key metrics for your model variants.
        """)
        
        # Create sample data
        np.random.seed(42)
        
        # Models and metrics
        models = ['Model A', 'Model B', 'Model C']
        metrics = {
            'Accuracy': {m: round(0.8 + np.random.normal(0, 0.05), 3) for m in models},
            'Latency (ms)': {m: round(np.random.normal(120, 20), 1) for m in models},
            'Error Rate (%)': {m: round(np.random.normal(3, 1), 2) for m in models},
            'Coverage (%)': {m: round(np.random.normal(95, 3), 1) for m in models}
        }
        
        # Create metrics cards
        cols = st.columns(len(models))
        
        for i, model in enumerate(models):
            with cols[i]:
                st.markdown(f"### {model}")
                for metric, values in metrics.items():
                    if metric == 'Latency (ms)' or metric == 'Error Rate (%)':
                        delta_color = "inverse"
                    else:
                        delta_color = "normal"
                    
                    # Compare to best model for this metric
                    if metric == 'Latency (ms)' or metric == 'Error Rate (%)':
                        best = min(values.values())
                        delta = values[model] - best
                    else:
                        best = max(values.values())
                        delta = values[model] - best
                    
                    # Only show delta if not the best
                    if abs(delta) > 0.001:
                        st.metric(metric, values[model], delta=round(delta, 3), delta_color=delta_color)
                    else:
                        st.metric(metric, values[model])
        
        # Create comparison chart
        st.markdown("### Comparative Performance")
        
        # Reshape data for plotting
        plot_data = []
        for metric, values in metrics.items():
            for model, value in values.items():
                plot_data.append({
                    'Metric': metric,
                    'Model': model,
                    'Value': value
                })
        
        df_metrics = pd.DataFrame(plot_data)
        
        # Create normalized version for radar chart
        df_norm = df_metrics.copy()
        for metric in metrics.keys():
            metric_values = df_norm[df_norm['Metric'] == metric]['Value']
            min_val = metric_values.min()
            max_val = metric_values.max()
            
            if metric in ['Latency (ms)', 'Error Rate (%)']:
                # Lower is better, invert normalization
                df_norm.loc[df_norm['Metric'] == metric, 'Normalized'] = 1 - ((df_norm[df_norm['Metric'] == metric]['Value'] - min_val) / (max_val - min_val) if max_val > min_val else 0)
            else:
                # Higher is better
                df_norm.loc[df_norm['Metric'] == metric, 'Normalized'] = (df_norm[df_norm['Metric'] == metric]['Value'] - min_val) / (max_val - min_val) if max_val > min_val else 0
        
        # Set up radar chart data
        categories = list(metrics.keys())
        fig = go.Figure()
        
        # Add traces for each model
        colors = [AWS_COLORS['orange'], AWS_COLORS['teal'], AWS_COLORS['green']]
        
        for i, model in enumerate(models):
            model_df = df_norm[df_norm['Model'] == model]
            values = model_df.sort_values('Metric')['Normalized'].tolist()
            # Close the loop
            values.append(values[0])
            categories_plot = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_plot,
                fill='toself',
                name=model,
                line_color=colors[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Performance Comparison (Normalized)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        This radar chart shows normalized performance across multiple metrics, making it easy to identify 
        which model excels in which areas. Normalization ensures that metrics with different scales can be 
        compared visually.
        """)
        
    # Conversion Funnel
    with viz_tabs[1]:
        st.markdown("""
        ### Conversion Funnel Comparison
        
        For user-facing ML models, tracking how each model variant impacts the user journey and conversion 
        funnel is crucial.
        """)
        
        # Create sample funnel data
        stages = ['Impressions', 'Clicks', 'Add to Cart', 'Purchase']
        
        model_a_funnel = [10000, 1500, 320, 95]
        model_b_funnel = [10000, 1650, 380, 120]
        
        # Calculate conversion rates
        model_a_rates = [100]
        model_b_rates = [100]
        
        for i in range(1, len(model_a_funnel)):
            model_a_rates.append(round(model_a_funnel[i] / model_a_funnel[i-1] * 100, 2))
            model_b_rates.append(round(model_b_funnel[i] / model_b_funnel[i-1] * 100, 2))
        
        # Create funnel visualization
        fig = go.Figure()
        
        fig.add_trace(go.Funnel(
            name='Model A',
            y=stages,
            x=model_a_funnel,
            textinfo="value+percent initial",
            marker={"color": AWS_COLORS['orange']}
        ))
        
        fig.add_trace(go.Funnel(
            name='Model B',
            y=stages,
            x=model_b_funnel,
            textinfo="value+percent initial",
            marker={"color": AWS_COLORS['teal']}
        ))
        
        fig.update_layout(
            title="Conversion Funnel Comparison",
            funnelmode="stack",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show conversion rates
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model A Conversion Rates")
            for i in range(1, len(stages)):
                st.metric(
                    f"{stages[i-1]} → {stages[i]}", 
                    f"{model_a_rates[i]}%"
                )
                
        with col2:
            st.markdown("### Model B Conversion Rates")
            for i in range(1, len(stages)):
                delta = model_b_rates[i] - model_a_rates[i]
                st.metric(
                    f"{stages[i-1]} → {stages[i]}", 
                    f"{model_b_rates[i]}%",
                    delta=f"{delta:.2f}%",
                    delta_color="normal" if delta > 0 else "inverse"
                )
        
        # Show overall conversion
        st.markdown("### Overall Conversion (Impressions → Purchase)")
        
        overall_a = round(model_a_funnel[-1] / model_a_funnel[0] * 100, 2)
        overall_b = round(model_b_funnel[-1] / model_b_funnel[0] * 100, 2)
        delta_overall = overall_b - overall_a
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model A", f"{overall_a}%")
            
        with col2:
            st.metric(
                "Model B", 
                f"{overall_b}%", 
                delta=f"{delta_overall:.2f}%",
                delta_color="normal" if delta_overall > 0 else "inverse"
            )
        
        if overall_b > overall_a:
            st.success(f"Model B shows a {delta_overall:.2f}% higher overall conversion rate, which could lead to increased revenue.")
        else:
            st.error(f"Model B shows a {abs(delta_overall):.2f}% lower overall conversion rate, indicating potential issues.")
            
    # Time Series Analysis
    with viz_tabs[2]:
        st.markdown("""
        ### Time Series Performance Analysis
        
        Monitoring model performance over time helps identify trends, seasonality effects, and potential degradation.
        """)
        
        # Generate time series data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Model A data - stable but less accurate
        model_a_accuracy = [0.82 + np.random.normal(0, 0.02) for _ in range(len(dates))]
        
        # Model B data - improves over time
        base_trend = np.linspace(0.79, 0.88, len(dates))
        model_b_accuracy = [base + np.random.normal(0, 0.025) for base in base_trend]
        
        # Create DataFrame
        df_time = pd.DataFrame({
            'Date': dates.repeat(2),
            'Model': ['Model A'] * len(dates) + ['Model B'] * len(dates),
            'Accuracy': model_a_accuracy + model_b_accuracy
        })
        
        # Create time series chart
        chart = alt.Chart(df_time).mark_line(point=True).encode(
            x='Date:T',
            y=alt.Y('Accuracy:Q', scale=alt.Scale(domain=[0.75, 0.95])),
            color=alt.Color('Model:N', scale=alt.Scale(
                domain=['Model A', 'Model B'],
                range=[AWS_COLORS['orange'], AWS_COLORS['teal']]
            )),
            strokeWidth=alt.value(3),
            tooltip=['Date', 'Model', 'Accuracy']
        ).properties(
            title='Model Accuracy Over Time',
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # Add rolling average
        df_time['Rolling_Avg'] = df_time.groupby('Model')['Accuracy'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        
        chart_rolling = alt.Chart(df_time).mark_line().encode(
            x='Date:T',
            y='Rolling_Avg:Q',
            color=alt.Color('Model:N', scale=alt.Scale(
                domain=['Model A', 'Model B'],
                range=[AWS_COLORS['orange'], AWS_COLORS['teal']]
            )),
            strokeWidth=alt.value(4),
            strokeDash=alt.value([5, 5]),
            tooltip=['Date', 'Model', 'Rolling_Avg']
        ).properties(
            title='7-Day Rolling Average Accuracy',
            height=400
        ).interactive()
        
        st.altair_chart(chart_rolling, use_container_width=True)
        
        # Show insights
        st.markdown("### Time Series Analysis Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model A")
            avg_a = np.mean(model_a_accuracy)
            std_a = np.std(model_a_accuracy)
            trend_a = model_a_accuracy[-1] - model_a_accuracy[0]
            
            st.metric("Average Accuracy", f"{avg_a:.3f}")
            st.metric("Stability (Std Dev)", f"{std_a:.3f}")
            st.metric("Overall Trend", f"{trend_a:.3f}", delta_color="normal" if trend_a > 0 else "inverse")
            
        with col2:
            st.markdown("#### Model B")
            avg_b = np.mean(model_b_accuracy)
            std_b = np.std(model_b_accuracy)
            trend_b = model_b_accuracy[-1] - model_b_accuracy[0]
            
            st.metric("Average Accuracy", f"{avg_b:.3f}", delta=f"{avg_b - avg_a:.3f}")
            st.metric("Stability (Std Dev)", f"{std_b:.3f}")
            st.metric("Overall Trend", f"{trend_b:.3f}", delta_color="normal" if trend_b > 0 else "inverse")
        
        st.markdown("### Recommendation")
        if avg_b > avg_a and trend_b > 0:
            st.success("Model B shows better overall performance and a positive trend. Consider increasing its traffic allocation.")
        elif avg_a > avg_b and std_a < std_b:
            st.info("Model A has better average performance and stability. Continue with Model A as the primary model.")
        else:
            st.warning("Both models show trade-offs. Continue testing with the current traffic allocation.")
    
    # Statistical Significance
    with viz_tabs[3]:
        st.markdown("""
        ### Statistical Significance Testing
        
        When comparing model variants, it's crucial to determine if the observed differences are statistically 
        significant or just due to random chance.
        """)
        
        # Create interactive significance testing tool
        st.markdown("""
        #### Significance Calculator
        
        Use this tool to calculate statistical significance between two model variants:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model A")
            sample_size_a = st.number_input("Sample Size A", min_value=10, value=1000, step=100)
            conversion_a = st.number_input("Conversion Rate A (%)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
            conversions_a = int(sample_size_a * (conversion_a / 100))
            st.metric("Total Conversions A", conversions_a)
            
        with col2:
            st.markdown("#### Model B")
            sample_size_b = st.number_input("Sample Size B", min_value=10, value=1000, step=100)
            conversion_b = st.number_input("Conversion Rate B (%)", min_value=0.1, max_value=100.0, value=5.8, step=0.1)
            conversions_b = int(sample_size_b * (conversion_b / 100))
            st.metric("Total Conversions B", conversions_b)
        
        # Calculate significance (simplified z-test)
        # This is a simplified calculation for educational purposes
        p_a = conversion_a / 100
        p_b = conversion_b / 100
        
        p_pooled = (conversions_a + conversions_b) / (sample_size_a + sample_size_b)
        se_pooled = (p_pooled * (1 - p_pooled) * (1/sample_size_a + 1/sample_size_b)) ** 0.5
        
        if se_pooled == 0:
            z_score = 0
        else:
            z_score = (p_b - p_a) / se_pooled
        
        # Calculate p-value (approximation)
        from scipy.stats import norm
        p_value = (1 - norm.cdf(abs(z_score))) * 2
        
        # Calculate confidence intervals
        ci_a_lower = max(0, p_a - 1.96 * ((p_a * (1 - p_a)) / sample_size_a) ** 0.5)
        ci_a_upper = min(1, p_a + 1.96 * ((p_a * (1 - p_a)) / sample_size_a) ** 0.5)
        
        ci_b_lower = max(0, p_b - 1.96 * ((p_b * (1 - p_b)) / sample_size_b) ** 0.5)
        ci_b_upper = min(1, p_b + 1.96 * ((p_b * (1 - p_b)) / sample_size_b) ** 0.5)
        
        # Display results
        st.markdown("### Statistical Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Z-Score", round(z_score, 3))
            
        with col2:
            st.metric("P-Value", round(p_value, 4))
            
        with col3:
            significant = p_value < 0.05
            st.metric("Significant at 95%?", "YES" if significant else "NO", 
                     delta=None, 
                     delta_color="normal" if significant else "inverse")
        
        # Show visual representation of confidence intervals
        st.markdown("### 95% Confidence Intervals")
        
        # Create confidence interval visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.errorbar([0], [p_a * 100], yerr=[(p_a - ci_a_lower) * 100], fmt='o', color=AWS_COLORS['orange'], 
                   capsize=10, markersize=10, label='Model A')
        ax.errorbar([1], [p_b * 100], yerr=[(p_b - ci_b_lower) * 100], fmt='o', color=AWS_COLORS['teal'], 
                   capsize=10, markersize=10, label='Model B')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Model A', 'Model B'])
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('95% Confidence Intervals')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Shade the area between confidence intervals
        if ci_a_upper < ci_b_lower or ci_b_upper < ci_a_lower:
            overlap = False
        else:
            overlap = True
        
        st.pyplot(fig)
        
        # Show conclusion
        st.markdown("### Conclusion")
        
        if significant:
            if p_b > p_a:
                st.success(f"""
                The difference between Model A ({conversion_a}%) and Model B ({conversion_b}%) is statistically significant (p={p_value:.4f}).
                
                Model B performs better with 95% confidence. You can safely conclude that Model B is superior.
                """)
            else:
                st.success(f"""
                The difference between Model A ({conversion_a}%) and Model B ({conversion_b}%) is statistically significant (p={p_value:.4f}).
                
                Model A performs better with 95% confidence. You should continue using Model A.
                """)
        else:
            st.warning(f"""
            The difference between Model A ({conversion_a}%) and Model B ({conversion_b}%) is NOT statistically significant (p={p_value:.4f}).
            
            You need more data to make a confident decision. Consider:
            - Continuing the test for longer to collect more data
            - Increasing the traffic allocation to the test
            - Making more substantial changes to Model B to create a bigger difference
            """)
            
        st.info("""
        **Note**: This is a simplified statistical calculation for educational purposes. In practice, 
        you might want to use more robust statistical methods like Bayesian analysis or sequential testing.
        """)

# Knowledge Check tab
with tabs[5]:
    st.title("Knowledge Check")
    st.markdown("Test your understanding of model testing strategies with these questions:")
    
    # Track completion
    all_answered = True
    
    # Question 1
    st.markdown("### Question 1")
    st.markdown("What is the key difference between challenger/shadow testing and A/B testing?")
    
    q1_options = [
        "A/B testing is only used for classification models, while shadow testing works for all model types",
        "Shadow testing processes requests but doesn't return results to users, while A/B testing serves results to users",
        "A/B testing requires more computational resources than shadow testing",
        "Shadow testing can only be used in development environments, not production"
    ]
    
    q1_answer = st.radio(
        "Select the correct answer:",
        q1_options,
        key="q1",
        index=None
    )
    
    if q1_answer is None:
        all_answered = False
    
    # Question 2
    st.markdown("### Question 2")
    st.markdown("In Amazon SageMaker, how do you configure traffic distribution for A/B testing?")
    
    q2_options = [
        "By using multiple endpoints and a load balancer",
        "By setting up routing rules in API Gateway",
        "By defining production variants with InitialVariantWeight in the endpoint configuration",
        "By implementing custom routing logic in your application code"
    ]
    
    q2_answer = st.radio(
        "Select the correct answer:",
        q2_options,
        key="q2",
        index=None
    )
    
    if q2_answer is None:
        all_answered = False
    
    # Question 3
    st.markdown("### Question 3")
    st.markdown("Which of the following are advantages of shadow testing over A/B testing? (Select all that apply)")
    
    q3_options = [
        "Zero risk to production traffic",
        "Lower computational cost",
        "Ability to test with 100% of traffic without user impact",
        "Can measure user interaction metrics",
        "Allows for faster model deployment"
    ]
    
    q3_answer = st.multiselect(
        "Select all that apply:",
        q3_options,
        key="q3"
    )
    
    if not q3_answer:
        all_answered = False
    
    # Question 4
    st.markdown("### Question 4")
    st.markdown("When invoking an endpoint with multiple production variants in SageMaker, what happens if you don't specify a TargetVariant parameter?")
    
    q4_options = [
        "The request fails with an error",
        "Traffic is automatically routed based on the variant weights",
        "The request is always sent to the first variant defined",
        "The request is duplicated and sent to all variants"
    ]
    
    q4_answer = st.radio(
        "Select the correct answer:",
        q4_options,
        key="q4",
        index=None
    )
    
    if q4_answer is None:
        all_answered = False
    
    # Question 5
    st.markdown("### Question 5")
    st.markdown("What does statistical significance tell you in the context of A/B testing?")
    
    q5_options = [
        "It guarantees that the better-performing model will continue to perform better in the future",
        "It ensures that your sample size is large enough for meaningful testing",
        "It indicates that the observed difference between variants is unlikely to be due to random chance",
        "It confirms that your model will be profitable when fully deployed"
    ]
    
    q5_answer = st.radio(
        "Select the correct answer:",
        q5_options,
        key="q5",
        index=None
    )
    
    if q5_answer is None:
        all_answered = False
    
    # Submit button
    if st.button("Submit Answers", disabled=not all_answered):
        st.session_state.show_answers = True
    
    # Show answer explanations if submitted
    if st.session_state.show_answers:
        st.markdown("## Your Results")
        
        score = 0
        
        # Q1 explanation
        st.markdown("### Question 1")
        if q1_answer == q1_options[1]:
            st.success("Correct! Shadow testing allows you to process the same requests without impacting users, while A/B testing serves different models' results to different users.")
            score += 1
        else:
            st.error(f"Incorrect. The correct answer is: {q1_options[1]}")
            st.markdown("Shadow testing processes requests but doesn't return results to users, making it zero-risk. A/B testing actually serves results from different models to users.")
        
        # Q2 explanation
        st.markdown("### Question 2")
        if q2_answer == q2_options[2]:
            st.success("Correct! In SageMaker, you define production variants with weights in the endpoint configuration to control traffic distribution.")
            score += 1
        else:
            st.error(f"Incorrect. The correct answer is: {q2_options[2]}")
            st.markdown("SageMaker uses the InitialVariantWeight parameter in the endpoint configuration to specify the percentage of traffic each model variant receives.")
        
        # Q3 explanation
        st.markdown("### Question 3")
        correct_q3 = [q3_options[0], q3_options[2]]
        if set(q3_answer) == set(correct_q3):
            st.success("Correct! Shadow testing has zero risk to production traffic and allows testing with 100% of traffic without user impact.")
            score += 1
        else:
            st.error(f"Incorrect. The correct answers are: {', '.join(correct_q3)}")
            st.markdown("Shadow testing provides zero risk to production traffic since users don't see the shadow model's predictions. It also allows you to test with 100% of traffic without impact to users. However, it has higher computational costs (running two models), can't measure user interactions, and typically extends the testing phase.")
        
        # Q4 explanation
        st.markdown("### Question 4")
        if q4_answer == q4_options[1]:
            st.success("Correct! When no TargetVariant is specified, traffic is routed based on the weights defined for each variant.")
            score += 1
        else:
            st.error(f"Incorrect. The correct answer is: {q4_options[1]}")
            st.markdown("Without a TargetVariant parameter, SageMaker automatically routes traffic according to the weights specified in the endpoint configuration.")
        
        # Q5 explanation
        st.markdown("### Question 5")
        if q5_answer == q5_options[2]:
            st.success("Correct! Statistical significance means the observed difference between variants is unlikely to be due to random chance.")
            score += 1
        else:
            st.error(f"Incorrect. The correct answer is: {q5_options[2]}")
            st.markdown("Statistical significance indicates that the observed difference between model variants is unlikely to be due to random chance. It helps you make confident decisions about which model performs better.")
        
        # Show final score
        st.markdown(f"## Final Score: {score}/5")
        if score == 5:
            st.balloons()
            st.success("Perfect score! You have an excellent understanding of model testing strategies.")
        elif score >= 3:
            st.success("Good job! You understand the key concepts of model testing strategies.")
        else:
            st.warning("You might want to review the material on model testing strategies.")

# Apply custom CSS
apply_custom_css()

# ```

# ## Requirements.txt

# ```
# streamlit==1.28.0
# pandas==2.1.1
# numpy==1.25.2
# matplotlib==3.8.0
# altair==5.1.2
# seaborn==0.13.0
# plotly==5.17.0
# pillow==10.0.1
# scipy==1.11.3
# uuid==1.30
# ```

# This enhanced Streamlit application provides a comprehensive interactive learning experience for model testing strategies. It includes:

# 1. **Introduction Tab**: Explains the importance of model testing strategies
# 2. **Challenger/Shadow Tab**: Detailed explanation with interactive visualization 
# 3. **A/B Testing Tab**: Simulation with dynamic traffic allocation
# 4. **Implementation Tab**: Code samples and integration details with AWS services
# 5. **Visualization Tab**: Advanced analytics for comparing model performance
# 6. **Knowledge Check Tab**: Quiz questions to test understanding

# The application follows modern UI/UX design principles with AWS color schemes and has a responsive layout. It includes session management, interactive visualizations, and comprehensive explanations of model testing strategies.