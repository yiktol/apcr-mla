
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import json
import boto3
import time
from datetime import datetime
import uuid
import io
from PIL import Image
import base64
import altair as alt
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="SageMaker MLOps",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if "initialized_mlops" not in st.session_state:
        st.session_state.initialized_mlops = True
        st.session_state.current_page = "Home"
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.start_time = datetime.now()
        st.session_state.interactions = 0

# Call initialization
init_session_state()

# Define AWS color scheme
AWS_COLORS = {
    "orange": "#FF9900",
    "dark_blue": "#232F3E",
    "light_blue": "#1166BB",
    "teal": "#00A1C9",
    "red": "#D13212",
    "green": "#7AA116",
    "purple": "#8C4799"
}

# CSS for styling
def load_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #232F3E;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #1166BB;
            font-weight: bold;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .card {
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .info-box {
            background-color: #e6f7ff;
            border-left: 5px solid #00A1C9;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .code-box {
            background-color: #f0f2f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.8rem;
            padding: 20px 0;
            border-top: 1px solid #ddd;
        }
        .highlight {
            background-color: #FFF2CC;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .aws-button {
            background-color: #FF9900;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        .step-box {
            border-left: 3px solid #FF9900;
            padding-left: 15px;
            margin-bottom: 15px;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #00A1C9;
            cursor: help;
        }
        .tooltip .tooltip-text {
            visibility: hidden;
            background-color: #232F3E;
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
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
    </style>
    """, unsafe_allow_html=True)

load_css()

# Function to load Lottie animations
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Sidebar for navigation and session management
def render_sidebar():
    with st.sidebar:
        
        st.markdown("### Session Management")

        st.info(f"User ID: {st.session_state.user_id}")
        
        if st.button("🔄 Reset Session", key="reset_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    

        st.divider()

        with st.expander("📚 About This App", expanded=False):
           st.markdown("""
                This interactive learning application demonstrates MLOps-ready features of Amazon SageMaker, allowing you to explore the complete machine learning lifecycle from data preparation to monitoring.
            """)
# Function to render the home page
def home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:

        
        st.header("What is MLOps?")
        st.markdown("""
        MLOps (Machine Learning Operations) combines Machine Learning, DevOps, and Data Engineering to streamline 
        the end-to-end machine learning lifecycle. It focuses on the deployment, monitoring, and management of 
        machine learning models in production environments.
        """)
        
        st.markdown('<div class="info-box">MLOps helps organizations deliver ML-enabled software with increased velocity, quality, and consistency.</div>', unsafe_allow_html=True)
    
    with col2:
        lottie_ml = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")
        st_lottie(lottie_ml, height=300, key="ml_animation")
    
    st.markdown('<div class="sub-header">MLOps Workflow Stages</div>', unsafe_allow_html=True)
    
    # Create a timeline of the MLOps stages
    timeline_data = pd.DataFrame({
        "Stage": ["Data Preparation", "Model Build", "Model Evaluation", "Model Selection", "Deployment", "Monitoring"],
        "Description": [
            "Prepare and transform data for training",
            "Build and train machine learning models",
            "Evaluate model performance with metrics",
            "Select the best-performing model",
            "Deploy models to production environments",
            "Monitor model performance and data drift"
        ],
        "Order": [1, 2, 3, 4, 5, 6]
    })
    
    # Create horizontal timeline with Altair
    timeline_chart = alt.Chart(timeline_data).mark_circle(size=300).encode(
        x=alt.X('Order:O', axis=alt.Axis(title=None, labelAngle=0, grid=False),
               scale=alt.Scale(domain=list(range(1, 7)))),
        y=alt.Y('Stage:N', axis=None),
        color=alt.Color('Stage:N', scale=alt.Scale(
            domain=timeline_data['Stage'].tolist(),
            range=[AWS_COLORS['orange'], AWS_COLORS['light_blue'], AWS_COLORS['teal'], 
                  AWS_COLORS['green'], AWS_COLORS['purple'], AWS_COLORS['red']]
        ), legend=None),
        tooltip=['Stage', 'Description']
    ).properties(height=100)
    
    # Add text labels
    text_chart = alt.Chart(timeline_data).mark_text(dy=20, fontSize=14).encode(
        x=alt.X('Order:O'),
        y=alt.Y('Stage:N'),
        text='Stage:N'
    )
    
    # Combine the charts
    complete_chart = (timeline_chart + text_chart).properties(
        title='MLOps Workflow Stages'
    ).configure_view(
        strokeWidth=0
    )
    
    st.altair_chart(complete_chart, use_container_width=True)
    
    # MLOps benefits section
    st.markdown('<div class="sub-header">Benefits of MLOps with Amazon SageMaker</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Accelerate Innovation")
        st.markdown("Automate the ML lifecycle to deploy models faster and more frequently")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Ensure Reliability")
        st.markdown("Implement CI/CD practices for ML to ensure consistent, reliable deployments")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Data-Driven Decisions")
        st.markdown("Monitor models and make data-driven decisions for retraining and improvements")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Call to action
    st.markdown("""
    <div style="text-align: center; margin: 40px 0;">
        <p style="font-size: 1.2rem;">Ready to dive into MLOps with Amazon SageMaker?</p>
        <p>Start by exploring the Data Preparation stage using the navigation menu on the left.</p>
    </div>
    """, unsafe_allow_html=True)

# Function to render the data preparation page
def data_preparation_page():
    st.markdown('<div class="main-header">Data Preparation with Amazon SageMaker</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Data preparation is the first critical step in the MLOps workflow. Amazon SageMaker provides powerful tools to help you:
        
        - Clean and preprocess your data
        - Transform features into optimal format for ML models
        - Store and version your features
        - Create reproducible data pipelines
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/SageMaker/sagemaker-components.39fa1212c238f72187485e3961a9cb20af6415b6.png", 
                caption="SageMaker Data Processing Components")
    
   
    tab1, tab2, tab3 = st.tabs(["Data Wrangler", "Processing Jobs", "Feature Store"])

    with tab1:
        st.markdown('<div class="sub-header">Amazon SageMaker Data Wrangler</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Data Wrangler provides a visual interface to prepare data for machine learning:
        
        - Connect to various data sources (S3, Athena, Redshift, etc.)
        - Apply 300+ built-in data transformations
        - Generate data visualizations to understand your data
        - Export transformation pipelines to SageMaker Processing Jobs
        """)
        
        # Interactive example of data wrangler
        st.markdown('<div class="info-box">Interactive Example: Data Analysis with Data Wrangler</div>', unsafe_allow_html=True)
        
        # Sample dataset
        sample_data = pd.DataFrame({
            'age': np.random.randint(18, 65, 100),
            'income': np.random.randint(30000, 120000, 100),
            'education_years': np.random.randint(8, 22, 100),
            'has_debt': np.random.choice([0, 1], 100),
            'credit_score': np.random.randint(300, 850, 100)
        })
        
        # Add some null values for demonstration
        sample_data.loc[np.random.choice(sample_data.index, 10), 'income'] = None
        sample_data.loc[np.random.choice(sample_data.index, 5), 'education_years'] = None
        
        st.dataframe(sample_data.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data analysis
            st.markdown("### Data Analysis")
            
            if st.button("Run Data Quality Analysis"):
                # Show data quality visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Calculate missing values
                missing_values = sample_data.isnull().sum()
                total = sample_data.isnull().count()
                percent_missing = (missing_values / total) * 100
                
                # Create DataFrame for plotting
                missing_stats = pd.DataFrame({
                    'Column': missing_values.index,
                    'Missing Values': missing_values.values,
                    'Percent Missing': percent_missing.values
                })
                
                # Sort by percentage
                missing_stats = missing_stats.sort_values('Percent Missing', ascending=False)
                
                # Plot
                sns.barplot(x='Percent Missing', y='Column', data=missing_stats, palette='viridis', ax=ax)
                ax.set_title("Missing Values Analysis")
                ax.set_xlim(0, 100)
                
                for i, v in enumerate(missing_stats['Percent Missing']):
                    ax.text(v + 0.5, i, f"{v:.1f}%", va='center')
                
                st.pyplot(fig)
                
                # Data statistics
                st.markdown("### Data Statistics")
                st.dataframe(sample_data.describe())
        
        with col2:
            # Feature correlation
            st.markdown("### Feature Correlation")
            
            if st.button("Generate Correlation Heatmap"):
                # Create correlation matrix
                corr = sample_data.corr()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                mask = np.triu(corr)
                sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', ax=ax)
                ax.set_title('Feature Correlation Matrix')
                
                st.pyplot(fig)
        
        # Data Wrangler transformation example
        st.markdown("### Apply Data Transformations")
        
        transform_options = st.multiselect(
            "Select transformations to apply:",
            ["Fill missing values", "Normalize numeric features", "Create age groups", "Drop unnecessary columns"]
        )
        
        if transform_options and st.button("Apply Transformations"):
            transformed_data = sample_data.copy()
            
            if "Fill missing values" in transform_options:
                # Fill missing income with mean
                transformed_data['income'] = transformed_data['income'].fillna(transformed_data['income'].mean())
                # Fill missing education years with median
                transformed_data['education_years'] = transformed_data['education_years'].fillna(transformed_data['education_years'].median())
                
            if "Normalize numeric features" in transform_options:
                # Min-max scaling for numeric columns
                for col in ['age', 'income', 'credit_score']:
                    transformed_data[col] = (transformed_data[col] - transformed_data[col].min()) / (transformed_data[col].max() - transformed_data[col].min())
            
            if "Create age groups" in transform_options:
                # Create age groups
                bins = [0, 25, 35, 50, 100]
                labels = ['Young', 'Adult', 'Middle-aged', 'Senior']
                transformed_data['age_group'] = pd.cut(transformed_data['age'], bins=bins, labels=labels)
            
            if "Drop unnecessary columns" in transform_options:
                # Just for demonstration - not actually dropping
                st.info("In a real scenario, you would identify and drop unnecessary columns here.")
            
            st.success("Transformations applied successfully!")
            st.dataframe(transformed_data.head(10))
            
            # Show the Python code that would generate these transformations
            st.markdown('<div class="sub-header">Generated Python Code</div>', unsafe_allow_html=True)
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            code = """
import pandas as pd
import numpy as np

def transform_data(df):
    # Create a copy of the dataframe to avoid modifying the original
    transformed_df = df.copy()
    
    # Fill missing values
    transformed_df['income'] = transformed_df['income'].fillna(transformed_df['income'].mean())
    transformed_df['education_years'] = transformed_df['education_years'].fillna(transformed_df['education_years'].median())
    
    # Normalize numeric features
    for col in ['age', 'income', 'credit_score']:
        transformed_df[col] = (transformed_df[col] - transformed_df[col].min()) / (transformed_df[col].max() - transformed_df[col].min())
    
    # Create age groups
    bins = [0, 25, 35, 50, 100]
    labels = ['Young', 'Adult', 'Middle-aged', 'Senior']
    transformed_df['age_group'] = pd.cut(transformed_df['age'], bins=bins, labels=labels)
    
    return transformed_df

# Apply transformations
transformed_data = transform_data(input_data)
"""
            st.code(code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="sub-header">SageMaker Processing Jobs</div>', unsafe_allow_html=True)
        
        st.markdown("""
        SageMaker Processing Jobs handle data processing workloads for ML:
        
        - Run data preprocessing, feature engineering, and model evaluation
        - Use pre-built containers or your own custom containers
        - Process data at scale with distributed computing
        - Schedule recurring jobs for automated pipelines
        """)
        
        # Processing job example
        st.markdown('<div class="info-box">Example: Creating a SageMaker Processing Job</div>', unsafe_allow_html=True)
        
        # Processing job architecture diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a simple diagram
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Draw components
        ax.add_patch(plt.Rectangle((1, 7), 2, 2, fill=True, alpha=0.3, color=AWS_COLORS["orange"], linewidth=2))
        ax.text(2, 8, "S3\nInput Data", ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.add_patch(plt.Rectangle((4, 4), 3, 3, fill=True, alpha=0.3, color=AWS_COLORS["light_blue"], linewidth=2))
        ax.text(5.5, 5.5, "SageMaker\nProcessing Job", ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.add_patch(plt.Rectangle((7, 7), 2, 2, fill=True, alpha=0.3, color=AWS_COLORS["green"], linewidth=2))
        ax.text(8, 8, "S3\nOutput Data", ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw arrows
        ax.arrow(3, 7.5, 1, -1, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
        ax.arrow(6.5, 5.5, 1, 1.5, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
        
        st.pyplot(fig)
        
        st.markdown("### Processing Job Code Example")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        code = """
import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define container URI
container_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-preprocessing:latest"

# Create processor
processor = Processor(
    role=role,
    image_uri=container_uri,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=sagemaker_session
)

# Run the processing job
processor.run(
    inputs=[
        ProcessingInput(
            source="s3://my-bucket/input-data/",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/train",
            destination="s3://my-bucket/output/train"
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/validation",
            destination="s3://my-bucket/output/validation"
        )
    ],
    arguments=["--train-test-split-ratio", "0.2"]
)
"""
        st.code(code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show preprocessing script example
        st.markdown("### Sample Preprocessing Script")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        preproc_code = """
#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Paths for input and output data
    input_data_path = '/opt/ml/processing/input/'
    train_path = '/opt/ml/processing/output/train/'
    validation_path = '/opt/ml/processing/output/validation/'
    
    # Create output directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    
    # Read input data
    print('Reading input data...')
    input_files = [file for file in os.listdir(input_data_path) if file.endswith('.csv')]
    df_list = []
    for file in input_files:
        df_list.append(pd.read_csv(os.path.join(input_data_path, file)))
    
    data = pd.concat(df_list)
    
    # Preprocessing
    print('Preprocessing data...')
    
    # Drop missing values or impute
    data = data.dropna()
    
    # Split features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Normalize numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.train_test_split_ratio, random_state=42
    )
    
    # Save processed data
    print('Saving processed data...')
    train_data = pd.concat([X_train, y_train], axis=1)
    validation_data = pd.concat([X_val, y_val], axis=1)
    
    train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
    validation_data.to_csv(os.path.join(validation_path, 'validation.csv'), index=False)
    
    print('Data preprocessing completed!')
"""
        st.code(preproc_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab3:
        st.markdown('<div class="sub-header">Amazon SageMaker Feature Store</div>', unsafe_allow_html=True)
        
        st.markdown("""
        SageMaker Feature Store is a fully managed repository for machine learning features:
        
        - Create, share, and reuse features across teams and ML projects
        - Store feature values in both online and offline stores
        - Maintain feature consistency between training and inference
        - Track feature lineage and metadata
        """)
        
        # Feature store architecture diagram
        st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/feature-store-architecture.png",
                caption="SageMaker Feature Store Architecture")
        
        st.markdown("### Creating a Feature Group")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        feature_store_code = """
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
import pandas as pd
from datetime import datetime

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create sample data with customer features
customer_data = pd.DataFrame({
    'customer_id': ['C1', 'C2', 'C3', 'C4', 'C5'],
    'age': [34, 45, 29, 58, 22],
    'income': [75000, 120000, 55000, 92000, 38000],
    'credit_score': [720, 680, 750, 830, 640],
    'num_accounts': [3, 5, 2, 4, 1],
    'years_customer': [5, 12, 2, 20, 1],
    'is_active': [1, 1, 0, 1, 1]
})

# Add event time for feature store
current_time = datetime.now()
customer_data['event_time'] = current_time

# Define feature group
feature_group_name = 'customer-features'

customer_features = FeatureGroup(
    name=feature_group_name,
    sagemaker_session=sagemaker_session
)

# Create feature definitions
customer_features.load_feature_definitions(data_frame=customer_data)

# Create the feature group
customer_features.create(
    s3_uri=f"s3://my-bucket/feature-store/{feature_group_name}",
    record_identifier_name='customer_id',
    event_time_feature_name='event_time',
    role_arn=role,
    enable_online_store=True
)

# Ingest data into feature store
customer_features.ingest(data_frame=customer_data, max_processes=4)
"""
        st.code(feature_store_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Retrieving Features for Training")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        retrieve_code = """
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# Reference the feature group
feature_group_name = 'customer-features'
customer_features = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

# Get latest data for model training from offline store
query = customer_features.athena_query()
query_string = f'''
SELECT customer_id, age, income, credit_score, num_accounts, years_customer, is_active
FROM "{feature_group_name}"
WHERE event_time = (
    SELECT MAX(event_time)
    FROM "{feature_group_name}"
    GROUP BY customer_id
)
'''

# Execute the query
query.run(query_string=query_string, output_location=f's3://my-bucket/query-results/')

# Wait for the query to complete
query.wait()

# Load the query results to a dataframe
df = query.as_dataframe()

print(df.head())
"""
        st.code(retrieve_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Using Features for Real-time Inference")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        inference_code = """
import boto3

# Create feature store runtime client
featurestore_runtime = boto3.client(service_name='sagemaker-featurestore-runtime')

# Get online feature values for real-time inference
response = featurestore_runtime.get_record(
    FeatureGroupName='customer-features',
    RecordIdentifierValueAsString='C1'
)

# Extract feature values from response
features = response['Record']
feature_dict = {feature['FeatureName']: feature['ValueAsString'] for feature in features}

print(feature_dict)
"""
        st.code(inference_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Benefits of Feature Store
        st.markdown('<div class="sub-header">Benefits of Using Feature Store</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔄 Consistency")
            st.markdown("""
            - Maintain consistent features between training and inference
            - Ensure all models use the same feature definitions
            - Version control for features
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ⚡ Efficiency")
            st.markdown("""
            - Reuse features across multiple ML projects
            - Reduce duplicate feature engineering work
            - Dedicated online store for low-latency retrieval
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# Function to render the model build page
def model_build_page():
    st.markdown('<div class="main-header">Model Building with Amazon SageMaker</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Model building is a crucial step in the MLOps workflow. Amazon SageMaker provides robust tools for:
    
    - Training machine learning models with built-in algorithms
    - Using custom algorithms with your own code
    - Scaling training jobs across multiple instances
    - Tracking and managing experiments
    """)
    

    tab1, tab2, tab3 = st.tabs(["Training Jobs", "Built-in Algorithms", "Experiments"])


    with tab1:
        st.markdown('<div class="sub-header">SageMaker Training Jobs</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            SageMaker Training Jobs provide a fully managed environment to train ML models:
            
            - Simple API to start training jobs
            - Automatic provisioning and scaling of resources
            - Built-in metrics and logs
            - Integration with SageMaker experiments for tracking
            - Support for distributed training
            """)
        
        with col2:
            # Training job workflow diagram
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Create a simple diagram
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Draw components
            ax.add_patch(plt.Rectangle((1, 7), 2, 2, fill=True, alpha=0.3, color=AWS_COLORS["orange"], linewidth=2))
            ax.text(2, 8, "Input Data\nin S3", ha='center', va='center', fontsize=10, fontweight='bold')
            
            ax.add_patch(plt.Rectangle((1, 4), 2, 2, fill=True, alpha=0.3, color=AWS_COLORS["purple"], linewidth=2))
            ax.text(2, 5, "Training\nCode", ha='center', va='center', fontsize=10, fontweight='bold')
            
            ax.add_patch(plt.Rectangle((4, 5.5), 3, 2, fill=True, alpha=0.3, color=AWS_COLORS["light_blue"], linewidth=2))
            ax.text(5.5, 6.5, "SageMaker\nTraining Job", ha='center', va='center', fontsize=10, fontweight='bold')
            
            ax.add_patch(plt.Rectangle((8, 5.5), 1.5, 2, fill=True, alpha=0.3, color=AWS_COLORS["green"], linewidth=2))
            ax.text(8.75, 6.5, "Model\nArtifacts", ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Draw arrows
            ax.arrow(3, 8, 1.5, -1, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            ax.arrow(3, 5, 1, 1, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            ax.arrow(7, 6.5, 0.9, 0, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            
            st.pyplot(fig)
        
        # Training job example
        st.markdown('<div class="info-box">Example: Creating a SageMaker Training Job</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        training_code = """
import boto3
import sagemaker
from sagemaker.estimator import Estimator

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define the training container
container = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region="us-east-1",
    version="1.3-1"
)

# Define the estimator
xgb_estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://my-bucket/models/",
    sagemaker_session=sagemaker_session
)

# Set hyperparameters
xgb_estimator.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    objective="binary:logistic",
    num_round=100
)

# Define data channels
train_data = "s3://my-bucket/data/train/"
validation_data = "s3://my-bucket/data/validation/"

# Start the training job
xgb_estimator.fit({
    'train': train_data,
    'validation': validation_data
})

# Get the model artifact
model_artifact = xgb_estimator.model_data
print(f"Model artifact stored at: {model_artifact}")
"""
        st.code(training_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Distributed Training")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            SageMaker supports distributed training to scale your workloads:
            
            - **Data Parallelism**: Each GPU has a complete copy of the model but trains on a subset of the data
            - **Model Parallelism**: Model is split across multiple GPUs when it's too large for a single GPU
            - **SageMaker Distributed**: Optimized libraries for distributed training
            """)
        
        with col2:
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            distributed_code = """
from sagemaker.tensorflow import TensorFlow

# Configure a distributed TensorFlow training job
tensorflow_estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_count=4,  # Use multiple instances
    instance_type='ml.p3.8xlarge',  # Multi-GPU instances
    framework_version='2.6.0',
    py_version='py38',
    distribution={
        'parameter_server': {
            'enabled': True
        }
    }
)

# Start distributed training
tensorflow_estimator.fit('s3://my-bucket/training-data/')
"""
            st.code(distributed_code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Training Script Entry Point")
        
        st.markdown("A sample training script that SageMaker would execute:")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        train_script = """
# train.py - Script that SageMaker will run for training

import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import json

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--num_round', type=int, default=100)
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    return parser.parse_args()

def load_data(data_path):
    # The SageMaker XGBoost container expects data in CSV or LibSVM format
    input_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('csv')]
    
    if len(input_files) == 0:
        raise ValueError('No CSV files found')
    
    # Use only the first file for this example
    raw_data = pd.read_csv(input_files[0])
    
    # Split into features and labels
    y = raw_data.iloc[:, 0].values  # First column is target
    X = raw_data.iloc[:, 1:].values  # All other columns are features
    
    return X, y

if __name__ == '__main__':
    args = parse_args()
    
    # Load training and validation data
    X_train, y_train = load_data(args.train)
    X_val, y_val = load_data(args.validation)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set hyperparameters
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    # Define watchlist to track progress
    watchlist = [(dtrain, 'train'), (dval, 'validation')]
    
    # Train model
    print("Training model...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=watchlist
    )
    
    # Save model to disk
    print(f"Saving model to {args.model_dir}")
    model.save_model(os.path.join(args.model_dir, 'model.json'))
    
    # Save feature importance
    feature_importance = model.get_score(importance_type='weight')
    
    # Save feature importance as part of model artifacts
    with open(os.path.join(args.model_dir, 'feature_importance.json'), 'w') as f:
        json.dump(feature_importance, f)
    
    print("Training complete!")
"""
        st.code(train_script, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="sub-header">SageMaker Built-in Algorithms</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Amazon SageMaker provides many built-in algorithms that are optimized for scale and performance:
        
        - No need to build and manage your own algorithm containers
        - Optimized for AWS infrastructure
        - Thoroughly tested and benchmarked
        - Support for distributed training
        """)
        
        # Algorithm categories
        algo_categories = {
            "Supervised Learning": [
                "XGBoost", "Linear Learner", "KNN", "Factorization Machines"
            ],
            "Unsupervised Learning": [
                "K-Means", "Principal Component Analysis (PCA)", "Random Cut Forest"
            ],
            "Computer Vision": [
                "Image Classification", "Object Detection", "Semantic Segmentation"
            ],
            "Natural Language Processing": [
                "BlazingText (Word2Vec)", "Sequence-to-Sequence", "LDA (Topic Modeling)"
            ],
            "Time Series": [
                "DeepAR", "Prophet", "CNN-QR"
            ],
            "Reinforcement Learning": [
                "Markov Decision Process (MDP)"
            ]
        }
        
        selected_category = st.selectbox("Select Algorithm Category", list(algo_categories.keys()))
        
        # Display algorithms in the selected category
        st.markdown(f"### {selected_category} Algorithms")
        
        col1, col2 = st.columns(2)
        
        for i, algo in enumerate(algo_categories[selected_category]):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"#### {algo}")
                with st.expander("Learn more"):
                    if algo == "XGBoost":
                        st.markdown("""
                        **XGBoost** is an implementation of the gradient boosted trees algorithm that is highly efficient and flexible.
                        
                        **Use cases:**
                        - Classification problems
                        - Regression problems
                        - Ranking problems
                        
                        **Benefits:**
                        - High accuracy
                        - Fast training and inference
                        - Handles missing data automatically
                        """)
                        
                        st.markdown('<div class="code-box">', unsafe_allow_html=True)
                        xgb_code = """
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.estimator import Estimator

# Get container, model, and script URIs
container_uri = image_uris.retrieve(
    framework="xgboost",
    region="us-east-1",
    version="1.5-1"
)

# Create an estimator
xgb_estimator = Estimator(
    image_uri=container_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://my-bucket/models/xgboost",
)

# Set hyperparameters
xgb_estimator.set_hyperparameters(
    max_depth=6,
    eta=0.3,
    objective="binary:logistic",
    num_round=100,
    subsample=0.7
)

# Fit the model
xgb_estimator.fit({
    "train": "s3://my-bucket/data/train",
    "validation": "s3://my-bucket/data/validation"
})
"""
                        st.code(xgb_code, language='python')
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    elif algo == "K-Means":
                        st.markdown("""
                        **K-Means** is an unsupervised learning algorithm that attempts to find discrete groupings within data.
                        
                        **Use cases:**
                        - Customer segmentation
                        - Document clustering
                        - Image compression
                        
                        **Benefits:**
                        - Fast implementation optimized for AWS
                        - Scales to large datasets
                        - Online clustering mode available
                        """)
                        
                        st.markdown('<div class="code-box">', unsafe_allow_html=True)
                        kmeans_code = """
from sagemaker import image_uris
from sagemaker.estimator import Estimator

# Get container URI
container_uri = image_uris.retrieve(
    framework="kmeans",
    region="us-east-1"
)

# Create an estimator
kmeans_estimator = Estimator(
    image_uri=container_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://my-bucket/models/kmeans",
)

# Set hyperparameters
kmeans_estimator.set_hyperparameters(
    k=10,  # number of clusters
    feature_dim=784,  # for MNIST images (28x28)
    mini_batch_size=500
)

# Fit the model
kmeans_estimator.fit({"train": "s3://my-bucket/data/train"})
"""
                        st.code(kmeans_code, language='python')
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    elif algo == "Image Classification":
                        st.markdown("""
                        **Image Classification** algorithm classifies images into categories.
                        
                        **Use cases:**
                        - Product categorization
                        - Medical image analysis
                        - Quality control in manufacturing
                        
                        **Benefits:**
                        - Transfer learning from pre-trained models
                        - Automatic data augmentation
                        - Multi-GPU training support
                        """)
                        
                        st.markdown('<div class="code-box">', unsafe_allow_html=True)
                        ic_code = """
from sagemaker import image_uris
from sagemaker.estimator import Estimator

# Get container URI
container_uri = image_uris.retrieve(
    framework="image-classification",
    region="us-east-1"
)

# Create an estimator
ic_estimator = Estimator(
    image_uri=container_uri,
    role=role,
    instance_count=1,
    instance_type="ml.p3.2xlarge",  # GPU instance for image tasks
    output_path="s3://my-bucket/models/image-classification",
)

# Set hyperparameters
ic_estimator.set_hyperparameters(
    num_classes=10,  # number of classes
    num_training_samples=50000,  # number of training samples
    epochs=30,
    learning_rate=0.01,
    use_pretrained_model=True
)

# Fit the model
ic_estimator.fit({
    "train": "s3://my-bucket/data/train",
    "validation": "s3://my-bucket/data/validation"
})
"""
                        st.code(ic_code, language='python')
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    elif algo == "DeepAR":
                        st.markdown("""
                        **DeepAR** is a forecasting algorithm based on recurrent neural networks.
                        
                        **Use cases:**
                        - Retail demand forecasting
                        - Server capacity planning
                        - Financial time series prediction
                        
                        **Benefits:**
                        - Works well with seasonal data
                        - Can train on multiple related time series
                        - Probabilistic forecasts (quantiles)
                        """)
                        
                        st.markdown('<div class="code-box">', unsafe_allow_html=True)
                        deepar_code = """
from sagemaker import image_uris
from sagemaker.estimator import Estimator

# Get container URI
container_uri = image_uris.retrieve(
    framework="forecasting-deepar",
    region="us-east-1"
)

# Create an estimator
deepar_estimator = Estimator(
    image_uri=container_uri,
    role=role,
    instance_count=1,
    instance_type="ml.c5.2xlarge",
    output_path="s3://my-bucket/models/deepar",
)

# Set hyperparameters
deepar_estimator.set_hyperparameters(
    time_freq="D",  # daily data
    context_length=30,  # use 30 days of context
    prediction_length=7,  # predict 7 days ahead
    epochs=100,
    num_cells=40,
    num_layers=3,
    dropout_rate=0.1,
    learning_rate=0.001,
)

# Fit the model
deepar_estimator.fit({"train": "s3://my-bucket/data/train", 
                     "test": "s3://my-bucket/data/test"})
"""
                        st.code(deepar_code, language='python')
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    else:
                        st.markdown(f"Detailed information about {algo} will be added soon.")
        
        # Algorithm selection guide
        st.markdown('<div class="sub-header">Algorithm Selection Guide</div>', unsafe_allow_html=True)
        
        st.markdown("""
        When selecting a SageMaker built-in algorithm, consider:
        
        1. **Problem type**: Classification, regression, clustering, etc.
        2. **Data type**: Tabular, image, text, time series, etc.
        3. **Dataset size**: Some algorithms handle large datasets better
        4. **Training time**: Some algorithms train faster than others
        5. **Interpretability**: Some models are more explainable than others
        """)
        
        # Algorithm decision tree visualization
        st.markdown("### Algorithm Decision Flow")
        
        decision_graph = """
        digraph G {
            node [shape=box, style="filled", color="lightblue", fontname="Arial"];
            
            start [label="What type of problem?"];
            
            supervised [label="Supervised Learning"];
            unsupervised [label="Unsupervised Learning"];
            
            tabular [label="Tabular Data"];
            image [label="Image Data"];
            text [label="Text Data"];
            timeseries [label="Time Series"];
            
            classification [label="Classification"];
            regression [label="Regression"];
            
            xgboost [label="XGBoost", color="#FF9900"];
            linear [label="Linear Learner", color="#FF9900"];
            factorization [label="Factorization Machines", color="#FF9900"];
            
            kmeans [label="K-Means", color="#00A1C9"];
            pca [label="PCA", color="#00A1C9"];
            
            img_class [label="Image Classification", color="#8C4799"];
            object_detect [label="Object Detection", color="#8C4799"];
            
            blazingtext [label="BlazingText", color="#7AA116"];
            seq2seq [label="Seq2Seq", color="#7AA116"];
            
            deepar [label="DeepAR", color="#D13212"];
            
            start -> supervised;
            start -> unsupervised;
            
            supervised -> tabular;
            supervised -> image;
            supervised -> text;
            supervised -> timeseries;
            
            tabular -> classification;
            tabular -> regression;
            
            classification -> xgboost;
            classification -> linear;
            regression -> xgboost;
            regression -> linear;
            
            unsupervised -> kmeans;
            unsupervised -> pca;
            
            image -> img_class;
            image -> object_detect;
            
            text -> blazingtext;
            text -> seq2seq;
            
            timeseries -> deepar;
        }
        """
        
        # Create a Graphviz object
        try:
            import graphviz
            st.graphviz_chart(decision_graph)
        except:
            st.error("Graphviz is required to display the algorithm decision tree.")
        
    with tab3:
        st.markdown('<div class="sub-header">SageMaker Experiments</div>', unsafe_allow_html=True)
        
        st.markdown("""
        SageMaker Experiments help you organize, track, compare, and evaluate machine learning experiments:
        
        - Track training jobs with metrics, parameters, and artifacts
        - Compare different model versions and hyperparameters
        - Visualize experiment results
        - Reproduce past experiments with exact configurations
        """)
        
        st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/experiments/experiments-components.png",
                caption="SageMaker Experiments Structure", width=700)
        
        # Experiment tracking example
        st.markdown('<div class="info-box">Example: Tracking Experiments in SageMaker</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        exp_code = """
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.experiments.run import Run

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create an experiment
experiment_name = "customer-churn-prediction"

# Create different runs for hyperparameter tuning
for max_depth in [3, 5, 7]:
    for learning_rate in [0.1, 0.01]:
        # Create a run for this hyperparameter combination
        with Run(
            experiment_name=experiment_name,
            sagemaker_session=sagemaker_session,
            run_name=f"xgboost-depth-{max_depth}-lr-{learning_rate}"
        ) as run:
            # Log the hyperparameters
            run.log_parameters({
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "objective": "binary:logistic",
                "num_round": 100
            })
            
            # Get container URI
            container = sagemaker.image_uris.retrieve(
                framework="xgboost",
                region="us-east-1",
                version="1.3-1"
            )
            
            # Create an estimator
            xgb_estimator = Estimator(
                image_uri=container,
                role=role,
                instance_count=1,
                instance_type="ml.m5.xlarge",
                output_path="s3://my-bucket/models/",
                sagemaker_session=sagemaker_session
            )
            
            # Set hyperparameters
            xgb_estimator.set_hyperparameters(
                max_depth=max_depth,
                eta=learning_rate,
                objective="binary:logistic",
                num_round=100
            )
            
            # Associate the estimator with the current run
            run.associate_estimator(xgb_estimator)
            
            # Start training
            xgb_estimator.fit({
                "train": "s3://my-bucket/data/train",
                "validation": "s3://my-bucket/data/validation"
            })
            
            # Log metrics from training
            training_metrics = xgb_estimator.training_job_analytics.dataframe()
            for index, row in training_metrics.iterrows():
                run.log_metric(
                    name=row['metric_name'],
                    value=row['value'],
                    iteration=row.get('iteration', 0)
                )
"""
        st.code(exp_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizing and comparing experiments
        st.markdown("### Visualizing and Comparing Experiments")
        
        # Create some simulated experiment results
        max_depths = [3, 5, 7]
        learning_rates = [0.1, 0.01]
        
        # Generate synthetic metrics for each combination
        results = []
        for depth in max_depths:
            for lr in learning_rates:
                # Simplified model: deeper trees + lower learning rate generally better until overfitting
                train_auc = min(0.95, 0.80 + depth * 0.02 + (1/lr) * 0.01)
                # Add some noise
                train_auc += np.random.normal(0, 0.01)
                
                # Validation AUC - peaks at medium depth
                if depth == 5:
                    val_auc = train_auc - 0.02
                elif depth == 3:
                    val_auc = train_auc - 0.04
                else:  # depth 7 - slight overfitting
                    val_auc = train_auc - 0.05
                
                # Lower learning rate has less overfitting
                if lr == 0.01:
                    val_auc += 0.02
                    
                # Training time - deeper trees and smaller learning rates take longer
                training_time = 30 + depth * 5 + (1/lr) * 10
                
                results.append({
                    'max_depth': depth,
                    'learning_rate': lr,
                    'train_auc': train_auc,
                    'validation_auc': val_auc,
                    'training_time': training_time,
                    'model_size_mb': 5 + depth * 2
                })
        
        results_df = pd.DataFrame(results)
        
        # Add experiment names
        results_df['experiment_name'] = results_df.apply(
            lambda row: f"depth={row['max_depth']}, lr={row['learning_rate']}", 
            axis=1
        )
        
        # Create visualization tabs
        viz_tab = st.radio(
            "Choose visualization", 
            ["Performance Comparison", "Hyperparameter Impact", "Training Time vs Performance"]
        )
        
        if viz_tab == "Performance Comparison":
            # Create bar chart to compare validation AUC across experiments
            fig = px.bar(
                results_df,
                x='experiment_name',
                y=['train_auc', 'validation_auc'],
                barmode='group',
                title='Model Performance by Configuration',
                labels={'value': 'AUC', 'experiment_name': 'Experiment Configuration', 'variable': 'Metric'},
                color_discrete_sequence=[AWS_COLORS['orange'], AWS_COLORS['light_blue']]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_tab == "Hyperparameter Impact":
            # Create heatmap of validation AUC by hyperparameters
            pivot_df = results_df.pivot(
                index='max_depth', 
                columns='learning_rate', 
                values='validation_auc'
            )
            
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Learning Rate", y="Max Depth", color="Validation AUC"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale='Viridis',
                title='Impact of Hyperparameters on Validation AUC'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_tab == "Training Time vs Performance":
            # Scatter plot of validation AUC vs training time
            fig = px.scatter(
                results_df,
                x='training_time',
                y='validation_auc',
                size='model_size_mb',
                color='max_depth',
                hover_name='experiment_name',
                labels={
                    'training_time': 'Training Time (seconds)',
                    'validation_auc': 'Validation AUC',
                    'model_size_mb': 'Model Size (MB)',
                    'max_depth': 'Max Depth'
                },
                title='Training Time vs Performance'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Best experiment selection
        st.markdown('<div class="sub-header">Selecting the Best Experiment</div>', unsafe_allow_html=True)
        
        # Get the best experiment by validation AUC
        best_exp = results_df.loc[results_df['validation_auc'].idxmax()]
        
        st.markdown(f"""
        <div class="card">
            <h3>Best Model Configuration</h3>
            <p><strong>Max Depth:</strong> {best_exp['max_depth']}</p>
            <p><strong>Learning Rate:</strong> {best_exp['learning_rate']}</p>
            <p><strong>Validation AUC:</strong> {best_exp['validation_auc']:.4f}</p>
            <p><strong>Training AUC:</strong> {best_exp['train_auc']:.4f}</p>
            <p><strong>Training Time:</strong> {best_exp['training_time']:.1f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Code for loading an experiment
        st.markdown("### Loading and Analyzing Experiment Results")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        analyze_exp_code = """
import pandas as pd
from sagemaker.analytics import ExperimentAnalytics

# Load experiment results into a DataFrame
experiment_name = "customer-churn-prediction"
analytics = ExperimentAnalytics(
    experiment_name=experiment_name,
    sagemaker_session=sagemaker_session
)

# Get all runs as a DataFrame
runs_df = analytics.dataframe()

# Find the best run by validation metric
best_run = runs_df.sort_values(by='validation:auc', ascending=False).iloc[0]
print(f"Best run: {best_run.run_name}")
print(f"Validation AUC: {best_run['validation:auc']}")

# Get all parameters and metrics from the best run
best_run_params = best_run[best_run.index.str.startswith('parameters.')]
best_run_metrics = best_run[best_run.index.str.startswith('metrics.')]

# Load the model from the best run
best_model_uri = best_run['artifact_s3_uri']
print(f"Best model URI: {best_model_uri}")
"""
        st.code(analyze_exp_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)

# Function to render the model evaluation page
def model_evaluation_page():
    st.markdown('<div class="main-header">Model Evaluation with Amazon SageMaker</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Model evaluation is crucial for understanding how well your model will perform on unseen data. SageMaker provides tools to:
    
    - Evaluate model performance using various metrics
    - Compare multiple models side by side
    - Understand model behavior and predictions
    - Assess model fairness and bias
    """)
    
    
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Model Explainability", "Bias Detection"])

    with tab1:
        st.markdown('<div class="sub-header">Evaluating Model Performance</div>', unsafe_allow_html=True)
        
        st.markdown("""
        SageMaker provides built-in functionality to evaluate your models based on standard metrics:
        
        - Classification metrics (accuracy, precision, recall, F1, AUC, etc.)
        - Regression metrics (MSE, RMSE, MAE, R²)
        - Ranking metrics (NDCG, MRR)
        - Custom metrics defined by you
        """)
        
        # Interactive metrics evaluation
        st.markdown('<div class="info-box">Interactive Example: Model Performance Evaluation</div>', unsafe_allow_html=True)
        
        # Sample model evaluation scenario
        st.markdown("### Evaluate a Binary Classification Model")
        
        # Generate synthetic prediction data
        np.random.seed(42)
        n_samples = 1000
        
        # True labels (60/40 class imbalance)
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        
        # Model 1: Good but not perfect model
        y_pred_proba_1 = np.zeros_like(y_true, dtype=float)
        # True positives with noise
        y_pred_proba_1[y_true == 1] = np.clip(np.random.normal(0.7, 0.15, size=sum(y_true == 1)), 0.01, 0.99)
        # True negatives with noise
        y_pred_proba_1[y_true == 0] = np.clip(np.random.normal(0.3, 0.15, size=sum(y_true == 0)), 0.01, 0.99)
        
        # Model 2: Better model
        y_pred_proba_2 = np.zeros_like(y_true, dtype=float)
        # True positives with less noise
        y_pred_proba_2[y_true == 1] = np.clip(np.random.normal(0.8, 0.1, size=sum(y_true == 1)), 0.01, 0.99)
        # True negatives with less noise
        y_pred_proba_2[y_true == 0] = np.clip(np.random.normal(0.2, 0.1, size=sum(y_true == 0)), 0.01, 0.99)
        
        # Create a DataFrame
        eval_df = pd.DataFrame({
            'true_label': y_true,
            'model1_score': y_pred_proba_1,
            'model2_score': y_pred_proba_2
        })
        
        # Let users select threshold for predictions
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        
        # Convert probabilities to binary predictions based on threshold
        y_pred_1 = (y_pred_proba_1 >= threshold).astype(int)
        y_pred_2 = (y_pred_proba_2 >= threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        metrics_model1 = {
            'Accuracy': accuracy_score(y_true, y_pred_1),
            'Precision': precision_score(y_true, y_pred_1),
            'Recall': recall_score(y_true, y_pred_1),
            'F1 Score': f1_score(y_true, y_pred_1),
            'AUC-ROC': roc_auc_score(y_true, y_pred_proba_1)
        }
        
        metrics_model2 = {
            'Accuracy': accuracy_score(y_true, y_pred_2),
            'Precision': precision_score(y_true, y_pred_2),
            'Recall': recall_score(y_true, y_pred_2),
            'F1 Score': f1_score(y_true, y_pred_2),
            'AUC-ROC': roc_auc_score(y_true, y_pred_proba_2)
        }
        
        # Display metrics comparison
        metrics_df = pd.DataFrame({
            'Model 1': metrics_model1,
            'Model 2': metrics_model2
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Performance Metrics Comparison")
            st.dataframe(metrics_df.style.highlight_max(axis=1))
        
        with col2:
            # Create bar chart
            metrics_comparison = pd.DataFrame({
                'Metric': metrics_df.index,
                'Model 1': metrics_df['Model 1'],
                'Model 2': metrics_df['Model 2']
            })
            
            fig = px.bar(
                metrics_comparison.melt(id_vars=['Metric'], var_name='Model', value_name='Value'),
                x='Metric',
                y='Value',
                color='Model',
                barmode='group',
                title='Model Performance Comparison',
                color_discrete_sequence=[AWS_COLORS['orange'], AWS_COLORS['light_blue']]
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig)
        
        # Confusion matrices
        st.markdown("### Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model 1 Confusion Matrix")
            cm1 = confusion_matrix(y_true, y_pred_1)
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')
            ax1.set_title('Model 1 Confusion Matrix')
            st.pyplot(fig1)
            
        with col2:
            st.markdown("#### Model 2 Confusion Matrix")
            cm2 = confusion_matrix(y_true, y_pred_2)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('True Label')
            ax2.set_title('Model 2 Confusion Matrix')
            st.pyplot(fig2)
        
        # ROC curves
        st.markdown("### ROC Curves")
        
        from sklearn.metrics import roc_curve
        
        fpr1, tpr1, _ = roc_curve(y_true, y_pred_proba_1)
        fpr2, tpr2, _ = roc_curve(y_true, y_pred_proba_2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr1, y=tpr1, name='Model 1', 
                                line=dict(color=AWS_COLORS['orange'], width=2)))
        fig.add_trace(go.Scatter(x=fpr2, y=tpr2, name='Model 2', 
                                line=dict(color=AWS_COLORS['light_blue'], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Classifier',
                                line=dict(color='gray', width=2, dash='dash')))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.7, y=0.1),
            width=700,
            height=500
        )
        st.plotly_chart(fig)
        
        # SageMaker Clarify code example
        st.markdown('<div class="sub-header">Using SageMaker Clarify for Evaluation</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        clarify_code = """
import boto3
import sagemaker
from sagemaker import clarify

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create a clarify processor
clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=sagemaker_session
)

# Define data config
data_config = clarify.DataConfig(
    s3_data_input_path="s3://my-bucket/evaluation/test-data.csv", 
    s3_output_path="s3://my-bucket/evaluation/output",
    label="target_column",
    headers=["feature1", "feature2", ..., "target_column"],
    dataset_type="text/csv"
)

# Define model config
model_config = clarify.ModelConfig(
    model_name="my-deployed-model",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    content_type="text/csv",
    accept_type="text/csv"
)

# Define metrics to compute
model_scores = "predicted_label"  # output column name from model
metrics_config = clarify.ModelPredictedLabelConfig(
    probability_threshold=0.5  # for binary classification
)

# Run model evaluation
clarify_processor.run_model_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=metrics_config,
    model_scores=model_scores
)
"""
        st.code(clarify_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="sub-header">Model Explainability with SageMaker Clarify</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Model explainability provides insights into how your models make decisions:
            
            - **Feature Importance**: Which features impact predictions the most?
            - **Local Explanations**: Why did the model make a specific prediction for an individual instance?
            - **Global Explanations**: How does the model behave across the entire dataset?
            - **Partial Dependence Plots**: How does the model's prediction change as a feature value changes?
            
            SageMaker Clarify provides these explanations using techniques like SHAP (SHapley Additive exPlanations).
            """)
        
        
        # SHAP example
        st.markdown('<div class="info-box">Interactive Example: Feature Importance with SHAP</div>', unsafe_allow_html=True)
        
        # Generate synthetic data for loan approval model
        np.random.seed(42)
        n_samples = 100
        
        # Features
        credit_score = np.random.randint(300, 850, n_samples)
        income = np.random.randint(20000, 200000, n_samples)
        debt_to_income = np.random.uniform(0.1, 0.6, n_samples)
        loan_amount = np.random.randint(5000, 50000, n_samples)
        loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)
        employment_years = np.random.uniform(0, 20, n_samples)
        
        # Create a dataframe
        loan_data = pd.DataFrame({
            'credit_score': credit_score,
            'income': income,
            'debt_to_income': debt_to_income,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'employment_years': employment_years
        })
        
        # Create synthetic SHAP values (in a real scenario these would come from the model)
        # Higher credit score and income are positive, higher debt ratio and loan amount are negative
        shap_values = np.zeros((n_samples, 6))
        shap_values[:, 0] = (credit_score - 550) / 300  # Credit score (higher is better)
        shap_values[:, 1] = (income - 50000) / 150000  # Income (higher is better)
        shap_values[:, 2] = -2 * (debt_to_income - 0.35)  # Debt to income (lower is better)
        shap_values[:, 3] = -1 * (loan_amount - 25000) / 25000  # Loan amount (lower is better)
        shap_values[:, 4] = -0.5 * (loan_term - 36) / 24  # Loan term (shorter is better)
        shap_values[:, 5] = 0.8 * np.minimum(employment_years, 5) / 5  # Employment years (more is better up to 5 yrs)
        
        # Add some random noise
        shap_values += np.random.normal(0, 0.1, shap_values.shape)
        
        # Select a sample to explain
        sample_index = st.slider("Select a loan application to explain", 0, n_samples-1, 42)
        
        selected_sample = loan_data.iloc[sample_index]
        selected_shap = shap_values[sample_index]
        
        # Display the sample's features
        st.markdown("### Loan Application Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Credit Score", f"{selected_sample['credit_score']}")
            st.metric("Income", f"${selected_sample['income']:,}")
        with col2:
            st.metric("Debt-to-Income", f"{selected_sample['debt_to_income']:.2f}")
            st.metric("Loan Amount", f"${selected_sample['loan_amount']:,}")
        with col3:
            st.metric("Loan Term", f"{selected_sample['loan_term']} months")
            st.metric("Employment Years", f"{selected_sample['employment_years']:.1f} years")
        
        # Calculate overall prediction
        base_value = 0  # Base SHAP value
        prediction_score = base_value + np.sum(selected_shap)
        approved = prediction_score > 0
        
        # Display the prediction
        st.markdown(f"""
        <div style="background-color:{'#D5F5E3' if approved else '#F5B7B1'}; padding:15px; border-radius:10px; margin-top:20px;">
            <h3>Loan Decision: {"Approved" if approved else "Denied"}</h3>
            <p>Prediction score: {prediction_score:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display SHAP waterfall chart
        st.markdown("### Feature Impact Analysis")
        
        feature_names = loan_data.columns
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': selected_shap
        })
        
        # Sort by absolute SHAP value
        shap_df['abs_shap'] = shap_df['SHAP Value'].abs()
        shap_df = shap_df.sort_values('abs_shap', ascending=False).drop('abs_shap', axis=1)
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["relative"] * len(feature_names),
            x=shap_df['SHAP Value'],
            y=shap_df['Feature'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": AWS_COLORS['green']}},
            decreasing={"marker": {"color": AWS_COLORS['red']}},
            text=[f"{x:.3f}" for x in shap_df['SHAP Value']],
            textposition="outside"
        ))
        
        fig.update_layout(
            title="Feature Impact on Loan Decision",
            xaxis_title="SHAP Value (Impact on Model Output)",
            yaxis={'categoryorder': 'array', 'categoryarray': shap_df['Feature']},
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance across all samples
        st.markdown("### Global Feature Importance")
        
        # Calculate mean absolute SHAP value for each feature
        global_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.mean(np.abs(shap_values), axis=0)
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            global_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Global Feature Importance',
            color_discrete_sequence=[AWS_COLORS['light_blue']]
        )
        
        fig.update_layout(
            xaxis_title="Mean |SHAP Value|",
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # SageMaker Clarify code example
        st.markdown("### Using SageMaker Clarify for Explainability")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        clarify_explain_code = """
import boto3
import sagemaker
from sagemaker import clarify

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create a clarify processor
clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=sagemaker_session
)

# Define data config
data_config = clarify.DataConfig(
    s3_data_input_path="s3://my-bucket/loan/data.csv", 
    s3_output_path="s3://my-bucket/loan/output",
    label="approved",
    features=["credit_score", "income", "debt_to_income", "loan_amount", "loan_term", "employment_years"],
    dataset_type="text/csv"
)

# Define model config
model_config = clarify.ModelConfig(
    model_name="loan-approval-model",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    content_type="text/csv",
    accept_type="application/json"
)

# Define SHAP config
shap_config = clarify.SHAPConfig(
    baseline=[
        [650, 50000, 0.4, 15000, 36, 5]  # baseline values for features
    ],
    num_samples=100,
    agg_method="mean_abs"  # absolute mean for feature importance
)

# Run explainability analysis
clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=shap_config
)
"""
        st.code(clarify_explain_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab3:
        st.markdown('<div class="sub-header">Bias Detection with SageMaker Clarify</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Machine learning models can inadvertently learn biases present in training data. SageMaker Clarify helps detect and measure bias:
        
        - **Pre-training bias metrics**: Identify bias in your training data before model training
        - **Post-training bias metrics**: Measure bias in your trained model's predictions
        - **Sensitive attributes**: Analyze model behavior across protected groups
        """)
        
        # Interactive bias detection example
        st.markdown('<div class="info-box">Interactive Example: Bias Detection in a Loan Approval Model</div>', unsafe_allow_html=True)
        
        # Generate synthetic data for bias analysis in loan approval
        np.random.seed(42)
        n_samples = 1000
        
        # Create sensitive attributes
        age_group = np.random.choice(['18-25', '26-40', '41-60', '60+'], n_samples, p=[0.15, 0.35, 0.35, 0.15])
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])
        
        # Create features correlated with sensitive attributes but with added noise
        credit_score = np.zeros(n_samples)
        income = np.zeros(n_samples)
        
        # Credit score distribution by age
        for i, group in enumerate(['18-25', '26-40', '41-60', '60+']):
            mask = age_group == group
            base_score = [600, 670, 720, 730][i]  # Younger people tend to have lower credit scores
            credit_score[mask] = np.clip(np.random.normal(base_score, 50, sum(mask)), 300, 850)
        
        # Income distribution by age and gender with a gender pay gap
        for a, age in enumerate(['18-25', '26-40', '41-60', '60+']):
            for g, gen in enumerate(['Male', 'Female']):
                mask = (age_group == age) & (gender == gen)
                base_income = [35000, 60000, 85000, 65000][a]  # Income varies by age
                if gen == 'Female':
                    base_income *= 0.82  # Simulate pay gap
                income[mask] = np.clip(np.random.normal(base_income, base_income * 0.2, sum(mask)), 15000, 250000)
        
        # Create loan amount requests
        loan_amount = np.clip(income * np.random.uniform(0.1, 0.5, n_samples), 5000, 100000)
        
        # Create approval decisions with bias
        # Credit score and income are the main factors, but we'll add a subtle bias
        
        # Legitimate factors
        credit_score_norm = (credit_score - 300) / 550  # Normalize to 0-1
        income_norm = np.minimum(income, 150000) / 150000  # Cap at 150k and normalize
        loan_ratio = np.minimum(loan_amount / income, 1)  # Loan amount to income ratio
        
        # Generate approval probability
        approval_prob = 0.7 * credit_score_norm + 0.3 * income_norm - 0.4 * loan_ratio
        
        # Add slight bias against young applicants and subtle gender bias
        for i, group in enumerate(age_group):
            if group == '18-25':
                approval_prob[i] -= 0.1  # Decrease approval chance for young applicants
        
        for i, gen in enumerate(gender):
            if gen == 'Female':
                approval_prob[i] -= 0.05  # Subtle gender bias
        
        # Add random noise
        approval_prob += np.random.normal(0, 0.1, n_samples)
        
        # Convert to binary decisions
        approved = (approval_prob > 0.5).astype(int)
        
        # Create dataframe
        bias_df = pd.DataFrame({
            'age_group': age_group,
            'gender': gender,
            'credit_score': credit_score,
            'income': income,
            'loan_amount': loan_amount,
            'approved': approved
        })
        
        # Let user select a sensitive attribute to analyze
        sensitive_attr = st.selectbox("Select sensitive attribute to analyze:", ['age_group', 'gender'])
        
        # Calculate approval rates by group
        approval_rates = bias_df.groupby(sensitive_attr)['approved'].mean().reset_index()
        approval_rates['approval_rate'] = approval_rates['approved'] * 100
        
        # Plot approval rates
        fig = px.bar(
            approval_rates,
            x=sensitive_attr,
            y='approval_rate',
            title=f'Loan Approval Rate by {sensitive_attr}',
            color=sensitive_attr,
            labels={'approval_rate': 'Approval Rate (%)'},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display bias metrics
        st.markdown("### Bias Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Demographic Disparity (DDP)
            if sensitive_attr == 'age_group':
                # For age, compare youngest vs others
                young_approval = approval_rates[approval_rates[sensitive_attr] == '18-25']['approval_rate'].values[0]
                others_approval = approval_rates[approval_rates[sensitive_attr] != '18-25']['approved'].mean() * 100
                disp_impact = young_approval / others_approval if others_approval > 0 else 0
                
                reference = "18-25"
                comparison = "other age groups"
                
            else:  # gender
                # For gender, compare Female vs Male
                female_approval = approval_rates[approval_rates[sensitive_attr] == 'Female']['approval_rate'].values[0]
                male_approval = approval_rates[approval_rates[sensitive_attr] == 'Male']['approval_rate'].values[0]
                disp_impact = female_approval / male_approval if male_approval > 0 else 0
                
                reference = "Female"
                comparison = "Male"
            
            st.markdown(f"""
            #### Disparate Impact Ratio
            
            Ratio of approval rate for {reference} compared to {comparison}:
            
            **{disp_impact:.2f}**
            
            <div style="background-color:{'#D5F5E3' if 0.8 <= disp_impact <= 1.2 else '#F5B7B1'}; padding:10px; border-radius:5px; margin-top:10px;">
            A value close to 1.0 indicates parity. Values below 0.8 or above 1.2 may indicate bias.
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Difference in approval rates
            if sensitive_attr == 'age_group':
                young_approval = approval_rates[approval_rates[sensitive_attr] == '18-25']['approval_rate'].values[0]
                others_approval = approval_rates[approval_rates[sensitive_attr] != '18-25']['approved'].mean() * 100
                diff = young_approval - others_approval
                
                reference = "18-25"
                comparison = "other age groups"
                
            else:  # gender
                female_approval = approval_rates[approval_rates[sensitive_attr] == 'Female']['approval_rate'].values[0]
                male_approval = approval_rates[approval_rates[sensitive_attr] == 'Male']['approval_rate'].values[0]
                diff = female_approval - male_approval
                
                reference = "Female"
                comparison = "Male"
            
            st.markdown(f"""
            #### Demographic Difference
            
            Difference in approval rate between {reference} and {comparison}:
            
            **{diff:.2f}%**
            
            <div style="background-color:{'#D5F5E3' if abs(diff) <= 5 else '#F5B7B1'}; padding:10px; border-radius:5px; margin-top:10px;">
            A value close to 0% indicates parity. Larger differences may indicate bias.
            </div>
            """, unsafe_allow_html=True)
        
        # Feature correlation with sensitive attribute
        st.markdown("### Feature Correlation with Sensitive Attribute")
        
        # Calculate correlation between features and sensitive attribute
        if sensitive_attr == 'gender':
            # Convert gender to numeric (Female=0, Male=1)
            gender_numeric = (bias_df['gender'] == 'Male').astype(int)
            corr_data = pd.DataFrame({
                'feature': ['credit_score', 'income', 'loan_amount'],
                'correlation': [
                    np.corrcoef(gender_numeric, bias_df['credit_score'])[0, 1],
                    np.corrcoef(gender_numeric, bias_df['income'])[0, 1],
                    np.corrcoef(gender_numeric, bias_df['loan_amount'])[0, 1]
                ]
            })
            
        else:  # age_group
            # Convert age_group to numeric (ordinal)
            age_map = {'18-25': 0, '26-40': 1, '41-60': 2, '60+': 3}
            age_numeric = bias_df['age_group'].map(age_map)
            corr_data = pd.DataFrame({
                'feature': ['credit_score', 'income', 'loan_amount'],
                'correlation': [
                    np.corrcoef(age_numeric, bias_df['credit_score'])[0, 1],
                    np.corrcoef(age_numeric, bias_df['income'])[0, 1],
                    np.corrcoef(age_numeric, bias_df['loan_amount'])[0, 1]
                ]
            })
        
        # Plot correlation
        fig = px.bar(
            corr_data,
            x='feature',
            y='correlation',
            title=f'Correlation between {sensitive_attr} and Model Features',
            color='correlation',
            color_continuous_scale=['red', 'white', 'blue'],
            height=400
        )
        
        fig.update_layout(
            yaxis=dict(range=[-1, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show distribution of key features by sensitive attribute
        st.markdown("### Feature Distributions by Group")
        
        feature_to_show = st.selectbox("Select feature to visualize:", ['credit_score', 'income', 'loan_amount'])
        
        fig = px.violin(
            bias_df,
            y=feature_to_show,
            x=sensitive_attr,
            color=sensitive_attr,
            box=True,
            points="all",
            title=f'Distribution of {feature_to_show} by {sensitive_attr}'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # SageMaker Clarify code example for bias detection
        st.markdown("### Using SageMaker Clarify for Bias Detection")
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        clarify_bias_code = """
import boto3
import sagemaker
from sagemaker import clarify

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create a clarify processor
clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=sagemaker_session
)

# Define data config
data_config = clarify.DataConfig(
    s3_data_input_path="s3://my-bucket/loan/data.csv", 
    s3_output_path="s3://my-bucket/loan/bias_output",
    label="approved",
    features=["age_group", "gender", "credit_score", "income", "loan_amount"],
    dataset_type="text/csv"
)

# Define bias config
bias_config = clarify.BiasConfig(
    label_values_or_threshold=[1],  # positive outcome (approved=1)
    facet_name="gender",  # protected attribute
    facet_values_or_threshold=["Female"]  # protected group
)

# Define the list of bias metrics to compute
bias_metrics = [
    "CI",  # Class Imbalance
    "DPL",  # Difference in Positive Proportions in Labels
    "DCR",  # Difference in Conditional Rejections
    "DCA",  # Difference in Conditional Acceptances
    "DRR",  # Disparate Rejection Rates
    "DAR",  # Disparate Acceptance Rates
    "DI"  # Disparate Impact
]

# Run pre-training bias analysis
clarify_processor.run_pre_training_bias(
    data_config=data_config,
    bias_config=bias_config,
    methods=bias_metrics
)

# For post-training bias, we also need a model config
model_config = clarify.ModelConfig(
    model_name="loan-approval-model",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    content_type="text/csv",
    accept_type="application/json"
)

# Define model predictions config
predictions_config = clarify.ModelPredictedLabelConfig(
    probability_threshold=0.5  # threshold for binary classification
)

# Run post-training bias analysis
clarify_processor.run_post_training_bias(
    data_config=data_config,
    bias_config=bias_config,
    model_config=model_config,
    model_predicted_label_config=predictions_config,
    methods=bias_metrics
)
"""
        st.code(clarify_bias_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)

# Function to render the model selection page
def model_selection_page():
    st.markdown('<div class="main-header">Model Selection with Amazon SageMaker</div>', unsafe_allow_html=True)
    
    st.markdown("""
    After training and evaluating multiple model candidates, you need to select the best model for deployment.
    SageMaker provides tools to help you:
    
    - Compare models based on performance metrics
    - Register models in the SageMaker Model Registry
    - Version and track model lineage
    - Approve models for different deployment stages
    """)
    
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Model Registry", "Model Approval Workflow"])
    
    with tab1:
        st.markdown('<div class="sub-header">Comparing Model Candidates</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Model comparison helps you evaluate multiple models and select the best one based on your criteria:
        
        - Compare performance metrics across models
        - Evaluate tradeoffs between accuracy and other considerations
        - Visualize model performance differences
        - Consider model size, latency, and explainability
        """)
        
        # Interactive model comparison example
        st.markdown('<div class="info-box">Interactive Example: Model Comparison Dashboard</div>', unsafe_allow_html=True)
        
        # Generate synthetic data for multiple models
        model_names = [
            "XGBoost (default)",
            "XGBoost (tuned)",
            "Linear Learner",
            "Random Forest",
            "Neural Network"
        ]
        
        # Performance metrics for each model
        accuracy = [0.82, 0.88, 0.79, 0.85, 0.86]
        precision = [0.80, 0.87, 0.78, 0.84, 0.83]
        recall = [0.76, 0.85, 0.74, 0.79, 0.87]
        f1_score = [0.78, 0.86, 0.76, 0.81, 0.85]
        
        # Other considerations
        training_time = [45, 120, 30, 60, 180]  # seconds
        inference_latency = [12, 15, 5, 40, 50]  # milliseconds
        model_size = [10, 22, 5, 75, 120]  # MB
        
        # Create DataFrame
        model_comparison_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'Training Time (s)': training_time,
            'Inference Latency (ms)': inference_latency,
            'Model Size (MB)': model_size
        })
        
        # Let user select primary metric
        primary_metric = st.selectbox(
            "Select primary performance metric:",
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        )
        
        # Sort models by the selected metric
        model_comparison_df = model_comparison_df.sort_values(by=primary_metric, ascending=False).reset_index(drop=True)
        
        # Performance metrics visualization
        st.markdown("### Performance Metrics Comparison")
        
        # Reshape for plotting
        plot_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        plot_df = model_comparison_df.melt(
            id_vars=['Model'], 
            value_vars=plot_metrics,
            var_name='Metric', 
            value_name='Score'
        )
        
        # Create plot
        fig = px.bar(
            plot_df,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            color_discrete_sequence=[AWS_COLORS['orange'], AWS_COLORS['light_blue'], AWS_COLORS['teal'], AWS_COLORS['green']]
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            legend_title="Metric"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Other considerations
        st.markdown("### Model Efficiency Metrics")
        
        efficiency_metrics = ['Training Time (s)', 'Inference Latency (ms)', 'Model Size (MB)']
        efficiency_df = model_comparison_df.melt(
            id_vars=['Model'], 
            value_vars=efficiency_metrics,
            var_name='Metric', 
            value_name='Value'
        )
        
        # Create visualization for each efficiency metric
        for metric in efficiency_metrics:
            fig = px.bar(
                model_comparison_df,
                x='Model',
                y=metric,
                title=f'Model {metric}',
                color='Model',
                color_discrete_sequence=[AWS_COLORS['orange'], AWS_COLORS['light_blue'], AWS_COLORS['teal'], AWS_COLORS['green'], AWS_COLORS['purple']]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for model comparison
        st.markdown("### Multi-Dimension Model Comparison")
        
        # Normalize metrics for radar chart
        radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Inverse for metrics where lower is better
        radar_df = model_comparison_df.copy()
        radar_df['Training Time (normalized)'] = (max(radar_df['Training Time (s)']) - radar_df['Training Time (s)']) / (max(radar_df['Training Time (s)']) - min(radar_df['Training Time (s)']))
        radar_df['Inference Latency (normalized)'] = (max(radar_df['Inference Latency (ms)']) - radar_df['Inference Latency (ms)']) / (max(radar_df['Inference Latency (ms)']) - min(radar_df['Inference Latency (ms)']))
        radar_df['Model Size (normalized)'] = (max(radar_df['Model Size (MB)']) - radar_df['Model Size (MB)']) / (max(radar_df['Model Size (MB)']) - min(radar_df['Model Size (MB)']))
        
        # Add normalized metrics to radar metrics
        radar_metrics += ['Training Time (normalized)', 'Inference Latency (normalized)', 'Model Size (normalized)']
        
        # Select models to compare
        models_to_compare = st.multiselect(
            "Select models to compare in radar chart:",
            model_names,
            default=model_names[:2]
        )
        
        if models_to_compare:
            # Filter selected models
            filtered_df = radar_df[radar_df['Model'].isin(models_to_compare)]
            
            # Create radar chart
            fig = go.Figure()
            
            for i, model in enumerate(filtered_df['Model']):
                values = filtered_df[filtered_df['Model'] == model][radar_metrics].values.flatten().tolist()
                # Close the polygon
                values.append(values[0])
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=radar_metrics + [radar_metrics[0]],
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Model Comparison (Higher is Better)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one model to compare")
        
        # Decision Support
        st.markdown('<div class="sub-header">Model Selection Decision Support</div>', unsafe_allow_html=True)
        
        # Let user customize importance weights for each metric
        st.markdown("### Customize Metric Importance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight_accuracy = st.slider("Accuracy importance", 0.0, 1.0, 0.8, 0.1)
            weight_precision = st.slider("Precision importance", 0.0, 1.0, 0.6, 0.1)
            weight_recall = st.slider("Recall importance", 0.0, 1.0, 0.7, 0.1)
            
        with col2:
            weight_training = st.slider("Training time importance", 0.0, 1.0, 0.3, 0.1)
            weight_latency = st.slider("Inference latency importance", 0.0, 1.0, 0.5, 0.1)
            weight_size = st.slider("Model size importance", 0.0, 1.0, 0.4, 0.1)
        
        # Calculate weighted score for each model
        normalized_df = radar_df.copy()
        
        # Calculate weighted score
        normalized_df['Weighted Score'] = (
            weight_accuracy * normalized_df['Accuracy'] +
            weight_precision * normalized_df['Precision'] +
            weight_recall * normalized_df['Recall'] +
            weight_training * normalized_df['Training Time (normalized)'] +
            weight_latency * normalized_df['Inference Latency (normalized)'] +
            weight_size * normalized_df['Model Size (normalized)']
        ) / (weight_accuracy + weight_precision + weight_recall + weight_training + weight_latency + weight_size)
        
        # Sort by weighted score
        normalized_df = normalized_df.sort_values(by='Weighted Score', ascending=False)
        
        # Display results
        st.markdown("### Weighted Model Rankings")
        
        fig = px.bar(
            normalized_df,
            x='Model',
            y='Weighted Score',
            title='Models Ranked by Weighted Score',
            color='Model',
            color_discrete_sequence=[AWS_COLORS['orange'], AWS_COLORS['light_blue'], AWS_COLORS['teal'], AWS_COLORS['green'], AWS_COLORS['purple']]
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        best_model = normalized_df.iloc[0]['Model']
        
        st.markdown(f"""
        <div style="background-color:#D5F5E3; padding:20px; border-radius:10px; margin-top:20px;">
            <h3>Recommended Model: {best_model}</h3>
            <p>Based on your metric importance weights, {best_model} has the highest weighted score of {normalized_df.iloc[0]['Weighted Score']:.4f}.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="sub-header">SageMaker Model Registry</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            The SageMaker Model Registry provides a centralized repository for organizing and cataloging your models:
            
            - **Model Groups**: Organize related model versions
            - **Model Versions**: Track different iterations of the same model
            - **Metadata**: Store information about each model version
            - **Approval Status**: Track which models are approved for deployment
            - **Deployment History**: Track where models have been deployed
            """)
        
        with col2:
            # Model registry illustration
            fig, ax = plt.subplots(figsize=(5, 6))
            
            # Create a simple diagram
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Draw components
            ax.add_patch(plt.Rectangle((1, 7), 8, 2, fill=True, alpha=0.3, color=AWS_COLORS["light_blue"], linewidth=2))
            ax.text(5, 8, "Model Group: Customer Churn Prediction", ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Model versions
            versions = [
                {"x": 2, "y": 5, "version": "1", "status": "Rejected", "color": AWS_COLORS["red"]},
                {"x": 5, "y": 5, "version": "2", "status": "Approved", "color": AWS_COLORS["green"]},
                {"x": 8, "y": 5, "version": "3", "status": "Pending", "color": AWS_COLORS["orange"]}
            ]
            
            for v in versions:
                ax.add_patch(plt.Rectangle((v["x"]-1, v["y"]-1), 2, 2, fill=True, alpha=0.3, color=v["color"], linewidth=2))
                ax.text(v["x"], v["y"], f"v{v['version']}\n{v['status']}", ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Draw arrows from group to versions
            for v in versions:
                ax.arrow(5, 7, v["x"]-5, v["y"]-7+0.5, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            
            st.pyplot(fig)
        
        # Model registry example
        st.markdown('<div class="info-box">Example: Working with SageMaker Model Registry</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        model_registry_code = """
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.utils import name_from_base

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
sm_client = boto_session.client('sagemaker')
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create a model group if it doesn't exist
model_group_name = "CustomerChurnPrediction"

try:
    # Check if model group exists
    sm_client.describe_model_package_group(ModelPackageGroupName=model_group_name)
    print(f"Model group {model_group_name} already exists")
except sm_client.exceptions.ResourceNotFound:
    # Create model group
    model_group_response = sm_client.create_model_package_group(
        ModelPackageGroupName=model_group_name,
        ModelPackageGroupDescription="Models for predicting customer churn"
    )
    print(f"Created model group: {model_group_name}")

# Create a model package (version) in the model group
model_version_description = "XGBoost model trained on customer data with hyperparameter tuning"

# Model artifacts from training
model_artifacts = "s3://my-bucket/models/xgboost-churn/model.tar.gz"

# Create model package
model_package_response = sm_client.create_model_package(
    ModelPackageGroupName=model_group_name,
    ModelPackageDescription=model_version_description,
    InferenceSpecification={
        "Containers": [
            {
                "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest",
                "ModelDataUrl": model_artifacts
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    },
    ValidationSpecification={
        "ValidationRole": role,
        "ValidationProfiles": [
            {
                "ProfileName": "validation-profile",
                "TransformJobDefinition": {
                    "MaxConcurrentTransforms": 1,
                    "MaxPayloadInMB": 6,
                    "TransformInput": {
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "s3://my-bucket/validation/data/"
                            }
                        },
                        "ContentType": "text/csv"
                    },
                    "TransformOutput": {
                        "S3OutputPath": "s3://my-bucket/validation/output/"
                    },
                    "TransformResources": {
                        "InstanceType": "ml.m5.xlarge",
                        "InstanceCount": 1
                    }
                }
            }
        ]
    },
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": "s3://my-bucket/metrics/model-metrics.json"
            }
        }
    }
)

model_package_arn = model_package_response["ModelPackageArn"]
print(f"Created model package: {model_package_arn}")

# Update the model approval status
sm_client.update_model_package(
    ModelPackageName=model_package_arn,
    ModelApprovalStatus="Approved"  # Options: Approved, Rejected, PendingManualApproval
)
print(f"Updated model approval status to 'Approved'")

# Create a SageMaker model from the approved model package
model = Model(
    role=role,
    model_package_arn=model_package_arn,
    sagemaker_session=sagemaker_session
)

# Deploy model (optional)
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)
"""
        st.code(model_registry_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive model registry visualization
        st.markdown("### Interactive Model Registry")
        
        # Sample model groups
        model_groups = [
            "CustomerChurnPrediction",
            "FraudDetection",
            "ProductRecommendation",
            "SentimentAnalysis"
        ]
        
        selected_group = st.selectbox("Select Model Group", model_groups)
        
        # Sample model versions for each group
        model_versions = {
            "CustomerChurnPrediction": [
                {"version": "1", "algorithm": "XGBoost", "accuracy": 0.82, "created": "2023-09-10", "status": "Rejected", "deployed": "No"},
                {"version": "2", "algorithm": "XGBoost (tuned)", "accuracy": 0.88, "created": "2023-09-15", "status": "Approved", "deployed": "Production"},
                {"version": "3", "algorithm": "Neural Network", "accuracy": 0.86, "created": "2023-09-20", "status": "PendingApproval", "deployed": "No"},
            ],
            "FraudDetection": [
                {"version": "1", "algorithm": "Isolation Forest", "accuracy": 0.75, "created": "2023-08-05", "status": "Rejected", "deployed": "No"},
                {"version": "2", "algorithm": "Random Forest", "accuracy": 0.85, "created": "2023-08-12", "status": "Approved", "deployed": "Production"},
            ],
            "ProductRecommendation": [
                {"version": "1", "algorithm": "Matrix Factorization", "accuracy": 0.72, "created": "2023-07-15", "status": "Approved", "deployed": "Staging"},
                {"version": "2", "algorithm": "Neural CF", "accuracy": 0.78, "created": "2023-07-28", "status": "Approved", "deployed": "Production"},
            ],
            "SentimentAnalysis": [
                {"version": "1", "algorithm": "BERT", "accuracy": 0.90, "created": "2023-06-10", "status": "Approved", "deployed": "Production"},
            ]
        }
        
        # Display model versions as a table
        if selected_group in model_versions:
            versions_df = pd.DataFrame(model_versions[selected_group])
            
            # Add status color
            def status_color(status):
                if status == "Approved":
                    return f'<span style="color:{AWS_COLORS["green"]}">●</span> {status}'
                elif status == "Rejected":
                    return f'<span style="color:{AWS_COLORS["red"]}">●</span> {status}'
                else:
                    return f'<span style="color:{AWS_COLORS["orange"]}">●</span> {status}'
            
            versions_df["Status"] = versions_df["status"].apply(status_color)
            
            # Display columns in preferred order
            display_df = versions_df[["version", "algorithm", "accuracy", "Status", "deployed", "created"]]
            display_df.columns = ["Version", "Algorithm", "Accuracy", "Status", "Deployment", "Created Date"]
            
            st.markdown(f"### Model Versions for {selected_group}")
            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Select a version to view details
            selected_version = st.selectbox(
                "Select a version to view details",
                [f"Version {v['version']}: {v['algorithm']}" for v in model_versions[selected_group]]
            )
            
            version_idx = int(selected_version.split(":")[0].split()[-1]) - 1
            version_details = model_versions[selected_group][version_idx]
            
            # Create tabs for different aspects of the model
            model_tabs = st.tabs(["Overview", "Metrics", "Lineage", "Artifacts"])
            
            with model_tabs[0]:
                st.markdown(f"### Version {version_details['version']}: {version_details['algorithm']}")
                
                # Display model metadata
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Algorithm:** {version_details['algorithm']}")
                    st.markdown(f"**Created Date:** {version_details['created']}")
                    st.markdown(f"**Approval Status:** {version_details['status']}")
                
                with col2:
                    st.markdown(f"**Accuracy:** {version_details['accuracy']}")
                    st.markdown(f"**Deployment:** {version_details['deployed']}")
                    st.markdown(f"**Artifact Location:** s3://model-registry/{selected_group.lower()}/v{version_details['version']}/")
            
            with model_tabs[1]:
                # Show sample metrics
                st.markdown("### Performance Metrics")
                
                # Generate additional metrics based on algorithm type
                if "XGBoost" in version_details['algorithm']:
                    precision = round(version_details['accuracy'] - 0.02, 2)
                    recall = round(version_details['accuracy'] - 0.04, 2)
                    f1 = round((2 * precision * recall) / (precision + recall), 2)
                    auc = round(version_details['accuracy'] + 0.01, 2)
                elif "Neural" in version_details['algorithm']:
                    precision = round(version_details['accuracy'] - 0.03, 2)
                    recall = round(version_details['accuracy'] + 0.02, 2)
                    f1 = round((2 * precision * recall) / (precision + recall), 2)
                    auc = round(version_details['accuracy'] + 0.02, 2)
                else:
                    precision = round(version_details['accuracy'] - 0.05, 2)
                    recall = round(version_details['accuracy'] - 0.02, 2)
                    f1 = round((2 * precision * recall) / (precision + recall), 2)
                    auc = round(version_details['accuracy'], 2)
                
                metrics_data = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
                    'Value': [version_details['accuracy'], precision, recall, f1, auc]
                })
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        metrics_data,
                        x='Metric',
                        y='Value',
                        title='Model Performance Metrics',
                        color_discrete_sequence=[AWS_COLORS['light_blue']]
                    )
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig)
                    
                with col2:
                    st.dataframe(metrics_data)
                
            with model_tabs[2]:
                # Show model lineage
                st.markdown("### Model Lineage")
                
                # Create a lineage graph
                try:
                    import graphviz
                    
                    g = graphviz.Digraph()
                    g.attr(rankdir='LR')
                    
                    # Add nodes
                    g.node('raw_data', 'Raw Data', shape='cylinder', style='filled', fillcolor='#D5E8D4')
                    g.node('processed_data', 'Processed Data', shape='cylinder', style='filled', fillcolor='#D5E8D4')
                    g.node('train_job', 'Training Job', shape='box', style='filled', fillcolor='#DAE8FC')
                    g.node('model', f'Model v{version_details["version"]}', shape='box', style='filled', fillcolor='#FFE6CC')
                    
                    if version_details['deployed'] != 'No':
                        g.node('endpoint', f'Endpoint ({version_details["deployed"]})', shape='box', style='filled', fillcolor='#F8CECC')
                    
                    # Add edges
                    g.edge('raw_data', 'processed_data', label='Preprocessing')
                    g.edge('processed_data', 'train_job', label='Training')
                    g.edge('train_job', 'model', label='Creates')
                    
                    if version_details['deployed'] != 'No':
                        g.edge('model', 'endpoint', label='Deployed to')
                    
                    st.graphviz_chart(g)
                    
                except:
                    st.error("Graphviz is required to display the lineage graph.")
                    
                    # Display as text instead
                    st.markdown("""
                    **Model Lineage:**
                    1. Raw Data → Preprocessing → Processed Data
                    2. Processed Data → Training Job → Model
                    3. Model → Deployment
                    """)
            
            with model_tabs[3]:
                # Show model artifacts
                st.markdown("### Model Artifacts")
                
                artifact_types = [
                    {"name": "Model Weights", "path": f"s3://model-registry/{selected_group.lower()}/v{version_details['version']}/model.tar.gz", "size": "42.8 MB"},
                    {"name": "Metrics JSON", "path": f"s3://model-registry/{selected_group.lower()}/v{version_details['version']}/metrics.json", "size": "4.2 KB"},
                    {"name": "Training Config", "path": f"s3://model-registry/{selected_group.lower()}/v{version_details['version']}/training_config.json", "size": "1.8 KB"},
                    {"name": "Hyperparameters", "path": f"s3://model-registry/{selected_group.lower()}/v{version_details['version']}/hyperparameters.json", "size": "0.9 KB"}
                ]
                
                artifacts_df = pd.DataFrame(artifact_types)
                st.dataframe(artifacts_df, use_container_width=True)
                
                st.markdown("### Example Hyperparameters")
                
                if "XGBoost" in version_details['algorithm']:
                    hyperparams = """
                    {
                        "max_depth": 6,
                        "eta": 0.3,
                        "gamma": 4,
                        "min_child_weight": 6,
                        "subsample": 0.8,
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "num_round": 100
                    }
                    """
                elif "Neural Network" in version_details['algorithm']:
                    hyperparams = """
                    {
                        "hidden_units": [128, 64, 32],
                        "activation": "relu",
                        "dropout_rate": 0.3,
                        "learning_rate": 0.001,
                        "batch_size": 64,
                        "epochs": 50
                    }
                    """
                else:
                    hyperparams = """
                    {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "min_samples_split": 5,
                        "min_samples_leaf": 2,
                        "bootstrap": true,
                        "random_state": 42
                    }
                    """
                
                st.code(hyperparams, language='json')
        
    with tab3:
        st.markdown('<div class="sub-header">Model Approval Workflow</div>', unsafe_allow_html=True)
        
        st.markdown("""
        The model approval workflow allows teams to:
        
        - Define a structured process for model deployment
        - Enforce governance and compliance requirements
        - Control which models get deployed to production
        - Implement CI/CD pipelines for MLOps
        """)
        
        # Model approval stages illustration
        stages = ["Development", "Testing", "Staging", "Production"]
        
        fig = go.Figure()
        
        # Add flow diagram
        for i, stage in enumerate(stages):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                marker=dict(size=40, color=list(AWS_COLORS.values())[i], symbol="square"),
                text=stage,
                textposition="middle center",
                name=stage
            ))
        
        # Add arrows between stages
        for i in range(len(stages)-1):
            fig.add_shape(
                type="line",
                x0=i + 0.2,
                y0=0,
                x1=i + 0.8,
                y1=0,
                line=dict(color="black", width=2),
                xref="x",
                yref="y"
            )
            # Add arrowhead
            fig.add_annotation(
                x=i + 0.8,
                y=0,
                ax=i + 0.2,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="black"
            )
        
        fig.update_layout(
            title="Model Approval Workflow Stages",
            showlegend=False,
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False, range=[-0.5, len(stages)-0.5]),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False, range=[-0.5, 0.5]),
            height=150,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Model Approval Workflow Components")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 👥 Roles")
            st.markdown("""
            - **Data Scientists**: Create and train models
            - **ML Engineers**: Review model metrics and code
            - **Subject Matter Experts**: Evaluate business impact
            - **Compliance Team**: Ensure models meet regulations
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📋 Conditions")
            st.markdown("""
            - Performance metrics above thresholds
            - Bias metrics within acceptable ranges
            - Successful integration tests
            - Documentation completed
            - Security review passed
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🚀 Actions")
            st.markdown("""
            - Approval/rejection of model versions
            - Promotion to next environment
            - Rollback if issues detected
            - A/B testing in staging
            - Full deployment to production
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # CI/CD pipeline for model deployment
        st.markdown('<div class="sub-header">CI/CD Pipeline for Model Deployment</div>', unsafe_allow_html=True)
        
        # Create tabs for different aspects of the CI/CD pipeline
        cicd_tab = st.radio(
            "CI/CD Pipeline Components", 
            ["Pipeline Overview", "Code Example"]
        )
        
        if cicd_tab == "Pipeline Overview":
            # CI/CD pipeline diagram
            try:
                import graphviz
                
                g = graphviz.Digraph(format='png')
                g.attr(rankdir='LR')
                
                # Define nodes
                g.node('registry', 'Model\nRegistry', shape='cylinder')
                g.node('testing', 'Automated\nTesting', shape='box')
                g.node('approval', 'Approval\nGate', shape='diamond')
                g.node('staging', 'Staging\nDeployment', shape='box')
                g.node('monitor', 'Monitoring\n& Validation', shape='box')
                g.node('prod_approval', 'Production\nApproval', shape='diamond')
                g.node('production', 'Production\nDeployment', shape='box')
                
                # Define edges
                g.edge('registry', 'testing', label='Approved\nModel')
                g.edge('testing', 'approval', label='Test\nResults')
                g.edge('approval', 'staging', label='Approved')
                g.edge('approval', 'registry', label='Rejected', color='red')
                g.edge('staging', 'monitor', label='Deploy')
                g.edge('monitor', 'prod_approval', label='Metrics')
                g.edge('prod_approval', 'production', label='Approved')
                g.edge('prod_approval', 'registry', label='Rejected', color='red')
                
                st.graphviz_chart(g)
                
            except:
                st.error("Graphviz is required to display the CI/CD pipeline diagram.")
                st.image("https://d1.awsstatic.com/diagrams/sagemaker-pipeline-complete-ml-workflow.953f5217c15a59a7c7186e8a38a50fe45e8c9368.png",
                        caption="SageMaker Pipeline Example")
        
        else:  # Code Example
            st.markdown("### SageMaker Pipelines for Model Deployment CI/CD")
            
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            pipeline_code = """
import boto3
import sagemaker
import sagemaker.session
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Step 1: Data processing step
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-processing:latest",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    command=["python3"],
    sagemaker_session=sagemaker_session
)

processing_step = ProcessingStep(
    name="ProcessData",
    processor=processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source="s3://my-bucket/raw-data/",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination="s3://my-bucket/processed-data/train"
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/output/validation",
            destination="s3://my-bucket/processed-data/validation"
        )
    ],
    code="s3://my-bucket/scripts/preprocessing.py"
)

# Step 2: Model training step
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://my-bucket/model-output/",
    hyperparameters={
        "max_depth": 5,
        "objective": "binary:logistic",
        "num_round": 100
    }
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Step 3: Model evaluation step
evaluation_processor = ScriptProcessor(
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-processing:latest",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    command=["python3"],
    sagemaker_session=sagemaker_session
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        sagemaker.processing.ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            destination="/opt/ml/processing/validation"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination="s3://my-bucket/evaluation"
        )
    ],
    code="s3://my-bucket/scripts/evaluate.py"
)

# Define a property file to read evaluation metrics
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)
eval_step.add_property_file(evaluation_report)

# Step 4: Register model step (conditional)
model_metrics = {
    "accuracy": {
        "value": eval_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri + "/accuracy.json",
        "content_type": "application/json"
    }
}

register_model_step = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="CustomerChurnPrediction",
    approval_status="PendingManualApproval",
    model_metrics=model_metrics
)

# Define condition step to register model only if accuracy is above threshold
accuracy_condition = ConditionGreaterThanOrEqualTo(
    left=eval_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri + "/accuracy.json",
    right=0.8  # Minimum accuracy threshold
)

condition_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[accuracy_condition],
    if_steps=[register_model_step],
    else_steps=[]
)

# Create pipeline
pipeline = Pipeline(
    name="CustomerChurnModelPipeline",
    parameters=[],
    steps=[processing_step, training_step, eval_step, condition_step],
    sagemaker_session=sagemaker_session
)

# Submit pipeline execution
pipeline_execution = pipeline.start()
"""
            st.code(pipeline_code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### Using AWS CodePipeline with SageMaker")
            
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            codepipeline_code = """
import boto3
import json

# Initialize clients
codepipeline_client = boto3.client('codepipeline')
sagemaker_client = boto3.client('sagemaker')

def create_model_deployment_pipeline():
    "Create a CodePipeline for model deployment"
    response = codepipeline_client.create_pipeline(
        pipeline={
            'name': 'ModelDeploymentPipeline',
            'roleArn': 'arn:aws:iam::123456789012:role/CodePipelineServiceRole',
            'artifactStore': {
                'type': 'S3',
                'location': 'my-codepipeline-artifacts'
            },
            'stages': [
                # Source stage - Get model from Model Registry
                {
                    'name': 'Source',
                    'actions': [
                        {
                            'name': 'ModelRegistry',
                            'actionTypeId': {
                                'category': 'Source',
                                'owner': 'AWS',
                                'provider': 'CodeStarSourceConnection',
                                'version': '1'
                            },
                            'configuration': {
                                'ConnectionArn': 'arn:aws:codestar-connections:us-east-1:123456789012:connection/my-connection',
                                'FullRepositoryId': 'my-org/model-config-repo',
                                'BranchName': 'main'
                            },
                            'outputArtifacts': [
                                {
                                    'name': 'SourceCode'
                                }
                            ]
                        }
                    ]
                },
                # Build stage - Run tests and validation
                {
                    'name': 'Validate',
                    'actions': [
                        {
                            'name': 'ModelValidation',
                            'actionTypeId': {
                                'category': 'Build',
                                'owner': 'AWS',
                                'provider': 'CodeBuild',
                                'version': '1'
                            },
                            'configuration': {
                                'ProjectName': 'ModelValidationProject'
                            },
                            'inputArtifacts': [
                                {
                                    'name': 'SourceCode'
                                }
                            ],
                            'outputArtifacts': [
                                {
                                    'name': 'ValidationResults'
                                }
                            ]
                        }
                    ]
                },
                # Approval stage - Manual approval to deploy to staging
                {
                    'name': 'ApproveStaging',
                    'actions': [
                        {
                            'name': 'ManualApproval',
                            'actionTypeId': {
                                'category': 'Approval',
                                'owner': 'AWS',
                                'provider': 'Manual',
                                'version': '1'
                            },
                            'configuration': {
                                'CustomData': 'Please review validation results and approve for staging deployment'
                            }
                        }
                    ]
                },
                # Deploy to staging
                {
                    'name': 'DeployToStaging',
                    'actions': [
                        {
                            'name': 'DeployModel',
                            'actionTypeId': {
                                'category': 'Deploy',
                                'owner': 'AWS',
                                'provider': 'ServiceCatalog',
                                'version': '1'
                            },
                            'configuration': {
                                'TemplateFilePath': 'templates/sagemaker-endpoint.yaml',
                                'ProductId': 'prod-abc123',
                                'ProductVersion': 'v1',
                                'DeploymentParameters': 'params/staging-deployment.json'
                            },
                            'inputArtifacts': [
                                {
                                    'name': 'ValidationResults'
                                }
                            ]
                        }
                    ]
                },
                # Monitor staging deployment
                {
                    'name': 'MonitorStaging',
                    'actions': [
                        {
                            'name': 'RunMonitoringTests',
                            'actionTypeId': {
                                'category': 'Test',
                                'owner': 'AWS',
                                'provider': 'CodeBuild',
                                'version': '1'
                            },
                            'configuration': {
                                'ProjectName': 'ModelMonitoringProject'
                            },
                            'inputArtifacts': [
                                {
                                    'name': 'ValidationResults'
                                }
                            ],
                            'outputArtifacts': [
                                {
                                    'name': 'MonitoringResults'
                                }
                            ]
                        }
                    ]
                },
                # Approval for production
                {
                    'name': 'ApproveProduction',
                    'actions': [
                        {
                            'name': 'ManualApproval',
                            'actionTypeId': {
                                'category': 'Approval',
                                'owner': 'AWS',
                                'provider': 'Manual',
                                'version': '1'
                            },
                            'configuration': {
                                'CustomData': 'Please review staging performance and approve for production deployment'
                            }
                        }
                    ]
                },
                # Deploy to production
                {
                    'name': 'DeployToProduction',
                    'actions': [
                        {
                            'name': 'DeployModel',
                            'actionTypeId': {
                                'category': 'Deploy',
                                'owner': 'AWS',
                                'provider': 'ServiceCatalog',
                                'version': '1'
                            },
                            'configuration': {
                                'TemplateFilePath': 'templates/sagemaker-endpoint.yaml',
                                'ProductId': 'prod-abc123',
                                'ProductVersion': 'v1',
                                'DeploymentParameters': 'params/production-deployment.json'
                            },
                            'inputArtifacts': [
                                {
                                    'name': 'MonitoringResults'
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    )
    
    return response

# Execute function to create pipeline
create_model_deployment_pipeline()
"""
            st.code(codepipeline_code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Example SageMaker Model Approval Steps
            st.markdown("### Model Approval WorkFlow Steps")
            
            approval_steps = [
                {"step": "Submit Model", "description": "Data Scientist submits model to the registry for approval"},
                {"step": "Automated Validation", "description": "Tests run automatically to verify model accuracy, bias, explainability"},
                {"step": "Review Results", "description": "ML Engineers review validation results and model documentation"},
                {"step": "Staging Approval", "description": "Team Lead approves deployment to staging environment"},
                {"step": "Staging Monitoring", "description": "Model performance is monitored in staging for 7 days"},
                {"step": "Production Approval", "description": "Business Stakeholders approve production deployment"},
                {"step": "Production Deployment", "description": "Model is deployed to production environment"},
                {"step": "Continuous Monitoring", "description": "Ongoing monitoring for model drift and performance"}
            ]
            
            for i, step in enumerate(approval_steps):
                st.markdown(f"""
                <div class="step-box">
                    <h4>Step {i+1}: {step['step']}</h4>
                    <p>{step['description']}</p>
                </div>
                """, unsafe_allow_html=True)

# Function to render the deployment page
def deployment_page():
    st.markdown('<div class="main-header">Model Deployment with Amazon SageMaker</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Model deployment is the process of putting your trained models into production so they can generate predictions.
    SageMaker provides multiple deployment options to meet various requirements:
    
    - Deploy to real-time endpoints for online predictions
    - Use batch transform for offline predictions
    - Multi-model endpoints for hosting multiple models
    - Serverless inference for cost-effective deployment
    """)
    
    tab1, tab2, tab3 = st.tabs(["Real-time Endpoints","Batch Transform","Advanced Deployment"])

    with tab1:
        st.markdown('<div class="sub-header">SageMaker Real-time Endpoints</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            SageMaker real-time endpoints provide a fully managed HTTPS endpoint to get predictions:
            
            - Low-latency, synchronous predictions
            - Automatic scaling to handle traffic spikes
            - High availability with multi-AZ support
            - Monitoring and logging of invocations
            - A/B testing with endpoint configurations
            """)
        
        with col2:
            # Endpoint architecture diagram
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Create a simple diagram
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Draw components - Application
            ax.add_patch(plt.Rectangle((1, 7), 2, 2, fill=True, alpha=0.3, color=AWS_COLORS["orange"], linewidth=2))
            ax.text(2, 8, "Application", ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Load Balancer
            ax.add_patch(plt.Rectangle((4, 7), 2, 2, fill=True, alpha=0.3, color=AWS_COLORS["light_blue"], linewidth=2))
            ax.text(5, 8, "SageMaker\nEndpoint", ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Model instances
            ax.add_patch(plt.Rectangle((7, 9), 2, 1.5, fill=True, alpha=0.3, color=AWS_COLORS["green"], linewidth=2))
            ax.text(8, 9.75, "Model\nInstance 1", ha='center', va='center', fontsize=10)
            
            ax.add_patch(plt.Rectangle((7, 7), 2, 1.5, fill=True, alpha=0.3, color=AWS_COLORS["green"], linewidth=2))
            ax.text(8, 7.75, "Model\nInstance 2", ha='center', va='center', fontsize=10)
            
            ax.add_patch(plt.Rectangle((7, 5), 2, 1.5, fill=True, alpha=0.3, color=AWS_COLORS["green"], linewidth=2))
            ax.text(8, 5.75, "Model\nInstance N", ha='center', va='center', fontsize=10)
            
            # Draw arrows
            ax.arrow(3, 8, 1, 0, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            ax.arrow(6, 8, 1, 1, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            ax.arrow(6, 8, 1, 0, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            ax.arrow(6, 8, 1, -1, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            
            # Text for scaling
            ax.text(8, 4, "Auto Scaling", ha='center', va='center', fontsize=10, style='italic')
            ax.plot([7.5, 8.5], [4.5, 3.5], color=AWS_COLORS["dark_blue"], linewidth=1)
            ax.arrow(8, 3.75, 0, 0.5, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            
            st.pyplot(fig)
        
        # Real-time endpoint example
        st.markdown('<div class="info-box">Example: Creating a SageMaker Real-time Endpoint</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        endpoint_code = """
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define model artifact and image
model_artifact = "s3://my-bucket/models/xgboost/model.tar.gz"
image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest"

# Create model
model = Model(
    image_uri=image_uri,
    model_data=model_artifact,
    role=role,
    sagemaker_session=sagemaker_session,
    name="xgboost-churn-predictor"
)

# Deploy model to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="customer-churn-predictor",
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

# Get endpoint name
endpoint_name = predictor.endpoint_name
print(f"Created endpoint: {endpoint_name}")

# Example: Invoke the endpoint with sample data
sample_data = "42,25000,5,3,0"  # CSV format: age,income,tenure,products,complaints
response = predictor.predict(sample_data)

print(f"Prediction: {response}")

# Update the endpoint with more capacity
predictor.update_endpoint(
    initial_instance_count=2,
    instance_type="ml.m5.xlarge",
    model_name=model.name
)
"""
        st.code(endpoint_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Endpoint auto-scaling
        st.markdown('<div class="sub-header">Endpoint Auto Scaling</div>', unsafe_allow_html=True)
        
        st.markdown("""
        SageMaker endpoints can automatically scale the number of instances based on workload:
        
        - Define scaling policies based on instance utilization
        - Set minimum and maximum number of instances
        - Use Application Auto Scaling to adjust capacity
        """)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        autoscaling_code = """
import boto3

# Initialize clients
application_autoscaling = boto3.client('application-autoscaling')
endpoint_name = "customer-churn-predictor"
variant_name = "AllTraffic"  # Default variant name

# Register endpoint as a scalable target
application_autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=4
)

# Define scaling policy
application_autoscaling.put_scaling_policy(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyName='CPUUtilizationScalingPolicy',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,  # Target CPU utilization (%)
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,  # 5 minutes
        'ScaleOutCooldown': 60,   # 1 minute
    }
)

print(f"Auto scaling configured for endpoint {endpoint_name}")
"""
        st.code(autoscaling_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Endpoint invoking code
        st.markdown('<div class="sub-header">Invoking the Endpoint</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Once deployed, you can invoke the endpoint from your applications:
        
        - Use the SageMaker runtime client for direct invocations
        - Use the SageMaker Python SDK for a higher-level interface
        - Set up retries and error handling for production use
        """)
        
        # Interactive endpoint invocation example
        st.markdown('<div class="info-box">Interactive Example: Endpoint Invocation</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Customer Churn Prediction Form")
            
            customer_age = st.slider("Customer Age", 18, 90, 42)
            customer_income = st.number_input("Annual Income ($)", 10000, 200000, 75000, 5000)
            tenure_years = st.slider("Customer Tenure (years)", 0, 20, 5)
            product_count = st.slider("Number of Products", 1, 6, 2)
            support_cases = st.slider("Support Cases in Last Year", 0, 10, 1)
            
            if st.button("Predict Churn Risk"):
                # Simulate model prediction
                # In a real app, this would call the SageMaker endpoint
                
                # Simple rule-based simulation of a model
                churn_score = 0
                
                # Age factor (younger customers more likely to churn)
                if customer_age < 30:
                    churn_score += 0.3
                elif customer_age < 40:
                    churn_score += 0.2
                else:
                    churn_score += 0.1
                
                # Income factor (higher income less likely to churn)
                if customer_income < 50000:
                    churn_score += 0.2
                elif customer_income < 100000:
                    churn_score += 0.1
                
                # Tenure factor (longer tenure less likely to churn)
                if tenure_years < 2:
                    churn_score += 0.3
                elif tenure_years < 5:
                    churn_score += 0.1
                
                # Product count (more products less likely to churn)
                churn_score -= (product_count - 1) * 0.05
                
                # Support cases (more cases more likely to churn)
                churn_score += support_cases * 0.05
                
                # Add some randomness
                churn_score += np.random.normal(0, 0.05)
                
                # Clamp score between 0 and 1
                churn_score = max(0, min(1, churn_score))
                
                # Show the prediction
                st.markdown(f"""
                <div style="background-color:{'#F5B7B1' if churn_score > 0.5 else '#D5F5E3'}; padding:20px; border-radius:10px; margin-top:20px;">
                    <h3>Prediction Result</h3>
                    <p><strong>Churn Risk Score:</strong> {churn_score:.2f}</p>
                    <p><strong>Risk Level:</strong> {"High" if churn_score > 0.7 else "Medium" if churn_score > 0.4 else "Low"}</p>
                    <p><strong>Recommendation:</strong> {"Immediate action needed" if churn_score > 0.7 else "Monitor closely" if churn_score > 0.4 else "Maintain service level"}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show code that would make this call
                st.markdown("### Endpoint Invocation Code")
                
                invocation_code = f"""
import boto3

# Initialize runtime client
runtime_client = boto3.client('sagemaker-runtime')

# Prepare input data (CSV format)
input_data = "{customer_age},{customer_income},{tenure_years},{product_count},{support_cases}"

# Invoke endpoint
response = runtime_client.invoke_endpoint(
    EndpointName="customer-churn-predictor",
    ContentType="text/csv",
    Body=input_data
)

# Parse response
result = response['Body'].read().decode('utf-8')
print(f"Churn prediction: {result}")
"""
                
                st.code(invocation_code, language='python')
        
        with col2:
            st.markdown("### AWS SDK Client Invocation")
            
            client_code = """
import boto3
import json

# Initialize SageMaker runtime client
runtime_client = boto3.client('sagemaker-runtime')

def predict_churn(customer_data):
    "
    Invoke SageMaker endpoint to predict customer churn
    
    Args:
        customer_data (str): CSV string with customer features
        
    Returns:
        dict: Prediction results
    "
    try:
        # Call the endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName="customer-churn-predictor",
            ContentType="text/csv",
            Body=customer_data,
            Accept="application/json"
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode('utf-8'))
        return result
        
    except Exception as e:
        print(f"Error invoking endpoint: {e}")
        raise
"""
            st.code(client_code, language='python')
            
            st.markdown("### Python SDK Invocation")
            
            sdk_code = """
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Create predictor object for existing endpoint
predictor = Predictor(
    endpoint_name="customer-churn-predictor",
    sagemaker_session=sagemaker_session,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

# Invoke with single data row
result = predictor.predict("42,75000,5,2,1")
print(f"Single prediction: {result}")

# Batch inference with multiple rows
batch_data = (
    "42,75000,5,2,1\n"
    "35,62000,2,1,3\n"
    "28,48000,1,3,0\n"
)
batch_results = predictor.predict(batch_data)
print(f"Batch predictions: {batch_results}")
"""
            st.code(sdk_code, language='python')
        
        # A/B Testing with endpoint variants
        st.markdown('<div class="sub-header">A/B Testing with Endpoint Variants</div>', unsafe_allow_html=True)
        
        st.markdown("""
        SageMaker endpoints support multiple variants for A/B testing different models:
        
        - Deploy multiple model variants to the same endpoint
        - Control traffic distribution between variants
        - Compare performance for model evaluation
        - Gradually shift traffic to better-performing models
        """)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        ab_testing_code = """
import boto3
import sagemaker
from sagemaker.model import Model

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
sm_client = boto_session.client('sagemaker')
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create two model objects - original and new version
model_original = Model(
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.0",
    model_data="s3://my-bucket/models/xgboost-original/model.tar.gz",
    role=role,
    name="xgboost-churn-original"
)

model_new = Model(
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.2",
    model_data="s3://my-bucket/models/xgboost-improved/model.tar.gz",
    role=role,
    name="xgboost-churn-improved"
)

# Create model objects in SageMaker
model_original.create(instance_type="ml.m5.large")
model_new.create(instance_type="ml.m5.large")

# Create an endpoint configuration with both variants
endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName="churn-abtest-config",
    ProductionVariants=[
        {
            "VariantName": "Original",
            "ModelName": "xgboost-churn-original",
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 0.8  # 80% of traffic
        },
        {
            "VariantName": "Improved",
            "ModelName": "xgboost-churn-improved",
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 0.2  # 20% of traffic
        }
    ]
)

# Create or update the endpoint
try:
    # Check if endpoint exists
    sm_client.describe_endpoint(EndpointName="churn-abtest-endpoint")
    
    # Update existing endpoint
    update_response = sm_client.update_endpoint(
        EndpointName="churn-abtest-endpoint",
        EndpointConfigName="churn-abtest-config"
    )
    print("Updating existing endpoint")
    
except sm_client.exceptions.ClientError:
    # Create new endpoint
    create_response = sm_client.create_endpoint(
        EndpointName="churn-abtest-endpoint",
        EndpointConfigName="churn-abtest-config"
    )
    print("Creating new endpoint")

print("A/B test endpoint deployment initiated")

# Update traffic distribution after evaluation
def update_traffic_distribution(original_weight, improved_weight):
    updated_config = sm_client.create_endpoint_config(
        EndpointConfigName=f"churn-abtest-config-{original_weight}-{improved_weight}",
        ProductionVariants=[
            {
                "VariantName": "Original",
                "ModelName": "xgboost-churn-original",
                "InstanceType": "ml.m5.large",
                "InitialInstanceCount": 1,
                "InitialVariantWeight": original_weight
            },
            {
                "VariantName": "Improved",
                "ModelName": "xgboost-churn-improved",
                "InstanceType": "ml.m5.large",
                "InitialInstanceCount": 1,
                "InitialVariantWeight": improved_weight
            }
        ]
    )
    
    update_response = sm_client.update_endpoint(
        EndpointName="churn-abtest-endpoint",
        EndpointConfigName=f"churn-abtest-config-{original_weight}-{improved_weight}"
    )
    
    return update_response

# Example: After evaluation, shift more traffic to the improved model
# update_traffic_distribution(0.2, 0.8)
"""
        st.code(ab_testing_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="sub-header">SageMaker Batch Transform</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            SageMaker Batch Transform is ideal for offline predictions on large datasets:
            
            - Get predictions for an entire dataset at once
            - No need to maintain a persistent endpoint
            - Cost-effective for large batches
            - Automatically scales to process large datasets
            - Results are stored in S3 for later use
            """)
        
        with col2:
            # Batch transform illustration
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Create a simple diagram
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Draw components - Input Data
            ax.add_patch(plt.Rectangle((1, 7), 2, 2, fill=True, alpha=0.3, color=AWS_COLORS["orange"], linewidth=2))
            ax.text(2, 8, "Input Data\nin S3", ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Batch Transform Job
            ax.add_patch(plt.Rectangle((4, 6), 3, 3, fill=True, alpha=0.3, color=AWS_COLORS["light_blue"], linewidth=2))
            ax.text(5.5, 7.5, "SageMaker\nBatch Transform\nJob", ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Output Results
            ax.add_patch(plt.Rectangle((8, 7), 2, 2, fill=True, alpha=0.3, color=AWS_COLORS["green"], linewidth=2))
            ax.text(9, 8, "Results\nin S3", ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Draw arrows
            ax.arrow(3, 8, 1, 0, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            ax.arrow(7, 8, 1, 0, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
            
            st.pyplot(fig)
        
        # Batch transform example
        st.markdown('<div class="info-box">Example: Running a SageMaker Batch Transform Job</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        batch_transform_code = """
import boto3
import sagemaker
from sagemaker.transformer import Transformer
from sagemaker.model import Model

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define model artifact and image
model_artifact = "s3://my-bucket/models/xgboost/model.tar.gz"
image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest"

# Create model
model = Model(
    image_uri=image_uri,
    model_data=model_artifact,
    role=role,
    sagemaker_session=sagemaker_session,
    name="xgboost-churn-batch"
)

# Create transformer object
transformer = Transformer(
    model_name=model.name,
    instance_count=4,  # Use multiple instances for faster processing
    instance_type="ml.m5.xlarge",
    output_path="s3://my-bucket/batch-predictions/",
    assemble_with="Line",  # How to combine prediction results
    accept="text/csv",  # Output format
    max_payload=6,  # Max size in MB for payload per batch
    strategy="MultiRecord"  # Process multiple records per batch
)

# Run the batch transform job
transformer.transform(
    data="s3://my-bucket/batch-data/customers.csv",
    data_type="S3Prefix",
    content_type="text/csv",
    split_type="Line",
    job_name="customer-churn-batch-predictions"
)

# Wait for the batch job to complete
transformer.wait()

# Get the results location
output_path = transformer.output_path
print(f"Batch transform results available at: {output_path}")

# Check the results using S3 client
s3_client = boto_session.client('s3')
bucket_name = "my-bucket"
prefix = "batch-predictions/"

response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
for obj in response.get('Contents', []):
    print(f"Result file: {obj['Key']}, Size: {obj['Size']} bytes")
"""
        st.code(batch_transform_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Batch transform vs real-time endpoints
        st.markdown('<div class="sub-header">Batch Transform vs. Real-Time Endpoints</div>', unsafe_allow_html=True)
        
        comparison_data = {
            "Feature": [
                "Use Case", 
                "Latency", 
                "Throughput", 
                "Cost Model", 
                "Operational Complexity",
                "Input Data Size",
                "Scaling",
                "Persistence"
            ],
            "Batch Transform": [
                "Large datasets, offline predictions", 
                "Not optimized for low latency", 
                "High throughput for large datasets", 
                "Pay only for the job runtime", 
                "Simple: submit job and retrieve results",
                "Handles very large datasets easily",
                "Automatic scaling for the job duration",
                "Temporary: runs only for job duration"
            ],
            "Real-Time Endpoints": [
                "On-demand, interactive predictions", 
                "Optimized for low latency", 
                "Limited by endpoint capacity", 
                "Pay for continuous endpoint uptime", 
                "Higher: requires monitoring and scaling",
                "Limited by payload size per request",
                "Requires auto-scaling configuration",
                "Persistent: runs continuously"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create tabs for different views of the comparison
        view = st.radio("View options:", ["Table View", "Visual Comparison"])
        
        if view == "Table View":
            st.dataframe(comparison_df, use_container_width=True)
            
        else:  # Visual Comparison
            selected_features = st.multiselect(
                "Select features to compare:",
                comparison_data["Feature"][1:],  # Exclude the first row which is "Use Case"
                default=["Latency", "Cost Model", "Operational Complexity"]
            )
            
            if selected_features:
                # Create a scaled comparison for selected features
                scales = {
                    "Latency": {"Batch Transform": 3, "Real-Time Endpoints": 1},  # Lower is better
                    "Throughput": {"Batch Transform": 5, "Real-Time Endpoints": 3},  # Higher is better
                    "Cost Model": {"Batch Transform": 2, "Real-Time Endpoints": 4},  # Lower is better
                    "Operational Complexity": {"Batch Transform": 2, "Real-Time Endpoints": 4},  # Lower is better
                    "Input Data Size": {"Batch Transform": 5, "Real-Time Endpoints": 2},  # Higher is better
                    "Scaling": {"Batch Transform": 4, "Real-Time Endpoints": 3},  # Higher is better
                    "Persistence": {"Batch Transform": 2, "Real-Time Endpoints": 5}  # Depends on needs
                }
                
                # Create data for radar chart
                radar_data = []
                for feature in selected_features:
                    batch_value = scales[feature]["Batch Transform"]
                    realtime_value = scales[feature]["Real-Time Endpoints"]
                    
                    radar_data.append({
                        "Feature": feature,
                        "Batch Transform": batch_value,
                        "Real-Time Endpoints": realtime_value
                    })
                
                radar_df = pd.DataFrame(radar_data)
                
                # Plot the comparison
                fig = px.line_polar(
                    radar_df.melt(id_vars=["Feature"], var_name="Deployment Type", value_name="Value"),
                    r="Value",
                    theta="Feature",
                    color="Deployment Type",
                    line_close=True,
                    color_discrete_sequence=[AWS_COLORS["orange"], AWS_COLORS["light_blue"]]
                )
                
                fig.update_layout(
                    title="Batch Transform vs. Real-Time Endpoints",
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]
                        )
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("Note: This visualization rates each feature on a scale of 1-5, where higher values generally indicate better performance or capability for that feature.")
            else:
                st.warning("Please select at least one feature to compare.")
        
        # Batch transform interactive example
        st.markdown('<div class="info-box">Interactive Example: Batch Transform Job</div>', unsafe_allow_html=True)
        
        st.markdown("### Configure a Batch Transform Job")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_location = st.text_input("Input Data S3 URI", "s3://my-bucket/input-data/")
            output_location = st.text_input("Output Location S3 URI", "s3://my-bucket/batch-results/")
            instance_type = st.selectbox(
                "Instance Type",
                ["ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.c5.xlarge", "ml.p3.2xlarge"]
            )
            instance_count = st.slider("Instance Count", 1, 10, 2)
        
        with col2:
            max_payload = st.slider("Max Payload Size (MB)", 1, 100, 6)
            batch_strategy = st.selectbox("Batch Strategy", ["SingleRecord", "MultiRecord"])
            split_type = st.selectbox("Split Type", ["None", "Line", "RecordIO", "TFRecord"])
            content_type = st.selectbox("Content Type", ["text/csv", "application/json", "application/x-recordio-protobuf"])
        
        # Calculate estimated duration and cost
        data_size_gb = st.slider("Estimated Input Data Size (GB)", 0.1, 100.0, 5.0, 0.1)
        
        if st.button("Estimate Job Duration and Cost"):
            # Very simplified estimation model
            instance_processing_rates = {
                "ml.m5.large": 0.05,     # GB per minute
                "ml.m5.xlarge": 0.1,     # GB per minute
                "ml.m5.2xlarge": 0.2,    # GB per minute
                "ml.c5.xlarge": 0.15,    # GB per minute
                "ml.p3.2xlarge": 0.5     # GB per minute
            }
            
            # Instance pricing (simplified approximation in USD per hour)
            instance_pricing = {
                "ml.m5.large": 0.134,
                "ml.m5.xlarge": 0.269,
                "ml.m5.2xlarge": 0.538,
                "ml.c5.xlarge": 0.238,
                "ml.p3.2xlarge": 3.825
            }
            
            # Calculate duration based on data size, instance type, and count
            processing_rate = instance_processing_rates[instance_type] * instance_count  # GB per minute
            estimated_duration_minutes = data_size_gb / processing_rate
            
            # Calculate cost
            estimated_cost = (estimated_duration_minutes / 60) * instance_pricing[instance_type] * instance_count
            
            # Add some overhead for startup and processing
            estimated_duration_minutes = estimated_duration_minutes * 1.2  # 20% overhead
            
            st.success(f"""
            ### Estimated Job Details
            - Estimated Duration: {estimated_duration_minutes:.1f} minutes
            - Estimated Cost: ${estimated_cost:.2f}
            - Processing Rate: {processing_rate:.2f} GB/minute
            """)
            
            # Show job submission code
            st.markdown("### Batch Transform Job Code")
            
            generated_code = f"""
import boto3
import sagemaker
from sagemaker.transformer import Transformer

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# Create transformer from existing model
transformer = Transformer(
    model_name="my-model",  # Use your model name
    instance_count={instance_count},
    instance_type="{instance_type}",
    output_path="{output_location}",
    max_payload={max_payload},
    strategy="{batch_strategy}"
)

# Run batch transform job
transformer.transform(
    data="{input_location}",
    content_type="{content_type}",
    split_type="{split_type}",
    job_name="batch-transform-" + sagemaker_session.default_bucket()
)

# Wait for job completion
transformer.wait()
"""
            st.code(generated_code, language='python')
            
            # Progress simulation
            st.markdown("### Job Progress Simulation")
            
            total_duration_sec = int(estimated_duration_minutes * 60)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for percent_complete in range(0, 101):
                # Calculate progress metrics
                elapsed_time = total_duration_sec * (percent_complete / 100)
                remaining_time = total_duration_sec - elapsed_time
                
                # Update progress
                progress_bar.progress(percent_complete)
                status_text.markdown(f"""
                **Progress:** {percent_complete}% complete
                **Status:** {'Completed' if percent_complete == 100 else 'In Progress'}
                **Files Processed:** {int(percent_complete * data_size_gb / 100 * 100)} / {int(data_size_gb * 100)} files
                """)
                
                # Only advance progress for demo purposes
                if percent_complete < 100:
                    break
        
    with tab3:
        st.markdown('<div class="sub-header">Advanced Deployment Options</div>', unsafe_allow_html=True)
        
        st.markdown("""
        SageMaker offers several advanced deployment options to meet specific requirements:
        
        - Multi-model endpoints for hosting many models on a shared endpoint
        - Serverless inference for cost-effective, auto-scaling deployment
        - Inference pipelines for sequentially executing multiple models
        - Asynchronous inference for handling high-throughput requests
        """)
        
        # Tabs for different advanced options
        advanced_tab = st.radio(
            "Advanced deployment options:",
            ["Multi-Model Endpoints", "Serverless Inference", "Inference Pipelines", "Asynchronous Inference"]
        )
        
        if advanced_tab == "Multi-Model Endpoints":
            st.markdown('<div class="sub-header">Multi-Model Endpoints</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                Multi-model endpoints allow you to deploy multiple models behind a single endpoint:
                
                - Host hundreds or thousands of models on a single endpoint
                - Reduce hosting costs by sharing infrastructure
                - Dynamic loading and unloading of models based on usage
                - Ideal for per-user personalization or per-tenant models
                """)
            
            with col2:
                # Multi-model endpoint illustration
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Create a simple diagram
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.axis('off')
                
                # Draw endpoint
                ax.add_patch(plt.Rectangle((2, 4), 7, 4, fill=True, alpha=0.2, color=AWS_COLORS["light_blue"], linewidth=2, edgecolor=AWS_COLORS["light_blue"]))
                ax.text(5.5, 8, "Multi-Model Endpoint", ha='center', va='center', fontsize=14, fontweight='bold')
                
                # Draw models
                model_positions = [(3, 6.5), (5, 6.5), (7, 6.5), (4, 5), (6, 5)]
                for i, pos in enumerate(model_positions):
                    color = list(AWS_COLORS.values())[i % len(AWS_COLORS)]
                    ax.add_patch(plt.Rectangle((pos[0]-0.7, pos[1]-0.5), 1.4, 1, fill=True, alpha=0.6, color=color, linewidth=1))
                    ax.text(pos[0], pos[1], f"Model {i+1}", ha='center', va='center', fontsize=10)
                
                # Add more models with ellipsis
                ax.text(5, 4.2, "...", ha='center', va='center', fontsize=20, fontweight='bold')
                
                # Application invoking the endpoint
                ax.add_patch(plt.Rectangle((0.5, 2), 2, 1, fill=True, alpha=0.3, color=AWS_COLORS["orange"], linewidth=2))
                ax.text(1.5, 2.5, "Application", ha='center', va='center', fontsize=10)
                
                # Draw arrow
                ax.arrow(2.5, 2.5, 1.5, 2, head_width=0.3, head_length=0.3, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
                ax.text(3.2, 3.5, "Request with\nTarget Model", ha='center', va='center', fontsize=8)
                
                st.pyplot(fig)
            
            # Multi-model endpoint example
            st.markdown('<div class="info-box">Example: Creating a Multi-Model Endpoint</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            mme_code = """
import boto3
import sagemaker
from sagemaker import MultiDataModel
from sagemaker import get_execution_role
from sagemaker.multidatamodel import MultiDataModel

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = get_execution_role()

# Define the container that will host multiple models
# For XGBoost, we can use the built-in container
xgb_container = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=boto_session.region_name,
    version="1.3-1"
)

# Create the MultiDataModel
multi_model = MultiDataModel(
    name="customer-models",
    model_data_prefix="s3://my-bucket/models/",
    image_uri=xgb_container,
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy the multi-model endpoint
predictor = multi_model.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.xlarge",
    endpoint_name="customer-multi-model"
)

# Example: Add models to the endpoint (they will be dynamically loaded)
model_paths = {
    "customer_segment_1": "s3://my-bucket/models/segment1/model.tar.gz",
    "customer_segment_2": "s3://my-bucket/models/segment2/model.tar.gz",
    "customer_segment_3": "s3://my-bucket/models/segment3/model.tar.gz",
    # Add more models as needed
}

# Add each model to the endpoint
for model_name, model_path in model_paths.items():
    multi_model.add_model(model_data_source=model_path, model_data_path=model_name)
    print(f"Added model: {model_name}")

# Example: Invoke a specific model
input_data = "41,52000,3,1"  # Example customer data in CSV format
response = predictor.predict(
    input_data,
    target_model="customer_segment_2"  # Specify which model to use
)

print(f"Prediction from segment 2 model: {response}")

# Invoke a different model with the same endpoint
response = predictor.predict(
    input_data,
    target_model="customer_segment_3"  # Different model
)

print(f"Prediction from segment 3 model: {response}")

# List models currently loaded in the endpoint
runtime_client = boto_session.client('sagemaker-runtime')
response = runtime_client.list_models(
    EndpointName="customer-multi-model",
    MaxResults=10
)

print(f"Currently loaded models: {response['Models']}")
"""
            st.code(mme_code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Interactive multi-model demo
            st.markdown("### Interactive Multi-Model Endpoint Demo")
            
            # Sample customer segments
            segments = {
                "High Value": {
                    "description": "Customers with high lifetime value and purchase frequency",
                    "models": ["churn_prediction", "upsell_recommendation", "loyalty_tier_prediction"],
                    "color": AWS_COLORS["green"]
                },
                "New Customers": {
                    "description": "Recently acquired customers with fewer than 3 purchases",
                    "models": ["propensity_to_buy", "product_recommendation", "email_response_prediction"],
                    "color": AWS_COLORS["light_blue"]
                },
                "At Risk": {
                    "description": "Customers showing signs of disengagement",
                    "models": ["churn_prediction", "winback_offer_optimization", "sentiment_analysis"],
                    "color": AWS_COLORS["red"]
                },
                "Seasonal": {
                    "description": "Customers who purchase mainly during specific seasons",
                    "models": ["next_purchase_prediction", "seasonal_recommendation", "discount_sensitivity"],
                    "color": AWS_COLORS["orange"]
                }
            }
            
            # Let user select a segment
            selected_segment = st.selectbox("Select Customer Segment", list(segments.keys()))
            
            # Show segment details
            st.markdown(f"""
            ### {selected_segment} Segment
            
            **Description:** {segments[selected_segment]['description']}
            
            **Models deployed for this segment:**
            """)
            
            # Show models for the selected segment
            for model in segments[selected_segment]['models']:
                st.markdown(f"- **{model}**")
            
            # Simulated invocation
            st.markdown("### Invoke Model for Selected Segment")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Input Customer Data")
                
                customer_id = st.text_input("Customer ID", "CUST-12345")
                
                # Simple form with customer attributes
                feature1 = st.slider("Purchase Frequency (monthly)", 0, 20, 5)
                feature2 = st.slider("Months as Customer", 1, 60, 12)
                feature3 = st.slider("Average Order Value ($)", 10, 500, 85)
                feature4 = st.selectbox("Preferred Channel", ["Web", "Mobile App", "In-Store", "Phone"])
            
            with col2:
                st.markdown("#### Model Invocation")
                
                # Select which model to invoke for this segment
                model_to_invoke = st.selectbox("Select Model to Invoke", segments[selected_segment]['models'])
                
                # Generate prediction when button is clicked
                if st.button("Get Prediction"):
                    # Simulate model call with progress spinner
                    with st.spinner(f"Invoking {model_to_invoke} for segment {selected_segment}..."):
                        time.sleep(1)  # Simulate API call
                        
                        # Generate different responses based on model and segment
                        if "churn_prediction" in model_to_invoke:
                            churn_risk = np.clip(0.7 - (feature1 * 0.05) - (feature3 * 0.001) + (0.01 if feature4 == "Phone" else 0), 0.05, 0.95)
                            st.markdown(f"""
                            #### Churn Prediction Result
                            
                            **Churn Risk Score:** {churn_risk:.2f}
                            
                            **Risk Level:** {"High" if churn_risk > 0.7 else "Medium" if churn_risk > 0.3 else "Low"}
                            
                            **Recommended Action:** {"Immediate intervention" if churn_risk > 0.7 else "Proactive outreach" if churn_risk > 0.3 else "Standard engagement"}
                            """)
                            
                            # Display risk gauge
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = churn_risk * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Churn Risk"},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "green"},
                                        {'range': [30, 70], 'color': "yellow"},
                                        {'range': [70, 100], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': churn_risk * 100
                                    }
                                }
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif "recommendation" in model_to_invoke:
                            # Generate mock product recommendations
                            products = [
                                "Premium Subscription Upgrade",
                                "Complementary Product Bundle",
                                "Extended Warranty",
                                "Loyalty Program Enrollment",
                                "New Product Trial Offer"
                            ]
                            
                            # Select a few based on "features"
                            num_recommendations = min(3, max(1, int(feature1 / 5)))
                            selected_products = np.random.choice(products, num_recommendations, replace=False)
                            
                            st.markdown(f"#### Product Recommendations")
                            
                            for i, product in enumerate(selected_products):
                                confidence = np.random.uniform(0.7, 0.95)
                                st.markdown(f"{i+1}. **{product}** (confidence: {confidence:.2f})")
                        
                        else:
                            # Generic prediction for other model types
                            st.markdown(f"#### {model_to_invoke.title()} Results")
                            
                            score = np.random.uniform(0.5, 0.9)
                            st.markdown(f"**Prediction Score:** {score:.2f}")
                            st.markdown(f"**Confidence Level:** {(score * 100):.1f}%")
                    
                    # Show the invocation code
                    st.markdown("#### API Invocation Code")
                    
                    invoke_code = f"""
# Example SDK code for multi-model endpoint invocation
import boto3

runtime_client = boto3.client('sagemaker-runtime')

# Example customer data
customer_data = "{feature1},{feature2},{feature3},{feature4}"

response = runtime_client.invoke_endpoint_with_target_model(
    EndpointName="customer-multi-model",
    TargetModelName="{selected_segment.lower().replace(' ', '_')}_{model_to_invoke}",
    ContentType="text/csv",
    Body=customer_data
)

# Parse the response
result = response['Body'].read().decode('utf-8')
print(f"Model prediction: {result}")
"""
                    st.code(invoke_code, language='python')
                    
        elif advanced_tab == "Serverless Inference":
            st.markdown('<div class="sub-header">Serverless Inference</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                Serverless Inference is ideal for workloads with intermittent or unpredictable traffic:
                
                - No need to provision or manage server capacity
                - Automatic scaling from zero to peak capacity
                - Pay only for the compute time used during inference
                - Eliminates the need for choosing instance types
                - Simplified operations without managing infrastructure
                """)
            
            with col2:
                # Load serverless animation
                lottie_serverless = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_g7xmvi61.json")
                st_lottie(lottie_serverless, height=200, key="serverless_animation")
            
            # Serverless inference example
            st.markdown('<div class="info-box">Example: Creating a Serverless Endpoint</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            serverless_code = """
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define model artifact and image
model_artifact = "s3://my-bucket/models/xgboost/model.tar.gz"
image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest"

# Create a model
model = Model(
    image_uri=image_uri,
    model_data=model_artifact,
    role=role,
    sagemaker_session=sagemaker_session,
    name="serverless-xgboost"
)

# Configure serverless inference
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,  # Memory in MB (1024, 2048, 3072, 4096, 5120, or 6144)
    max_concurrency=5  # Maximum concurrent invocations
)

# Deploy the model to a serverless endpoint
predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="serverless-churn-predictor"
)

# Example: Invoke the serverless endpoint
sample_data = "42,75000,5,3,0"  # Example customer data
response = predictor.predict(sample_data)

print(f"Prediction from serverless endpoint: {response}")
"""
            st.code(serverless_code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Serverless vs. real-time endpoints comparison
            st.markdown('<div class="sub-header">Serverless vs. Real-Time Endpoints</div>', unsafe_allow_html=True)
            
            # Cost comparison calculator
            st.markdown("### Cost Comparison Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Deployment Parameters")
                
                avg_requests_per_day = st.slider("Average Daily Requests", 100, 100000, 5000)
                avg_duration_ms = st.slider("Average Inference Duration (ms)", 10, 500, 100)
                request_pattern = st.selectbox("Request Pattern", ["Consistent", "Spiky", "Business Hours Only", "Infrequent"])
                
                # Memory/instance selection
                memory_size = st.selectbox("Serverless Memory Size (MB)", [1024, 2048, 3072, 4096, 5120, 6144])
                instance_type = st.selectbox("Real-Time Instance Type", ["ml.t2.medium", "ml.m5.large", "ml.m5.xlarge", "ml.c5.large"])
                instance_count = st.slider("Real-Time Instance Count", 1, 5, 1)
            
            with col2:
                st.markdown("#### Calculate Monthly Cost")
                
                if st.button("Calculate Costs"):
                    # Simplified cost model for illustration
                    # Actual pricing would require the latest AWS pricing API
                    
                    # Pricing constants (approximated)
                    serverless_price_per_gb_s = 0.00000533  # per GB-second
                    serverless_price_per_request = 0.0000002  # per request
                    
                    instance_hourly_rates = {
                        "ml.t2.medium": 0.068,
                        "ml.m5.large": 0.134,
                        "ml.m5.xlarge": 0.269,
                        "ml.c5.large": 0.119
                    }
                    
                    # Calculate serverless cost
                    serverless_gb = memory_size / 1024
                    serverless_seconds_per_request = avg_duration_ms / 1000
                    
                    # Apply concurrency patterns
                    if request_pattern == "Consistent":
                        utilization_factor = 0.75
                    elif request_pattern == "Spiky":
                        utilization_factor = 0.35
                    elif request_pattern == "Business Hours Only":
                        utilization_factor = 0.3
                    else:  # Infrequent
                        utilization_factor = 0.1
                    
                    # Monthly requests
                    monthly_requests = avg_requests_per_day * 30
                    
                    # Serverless costs
                    serverless_compute_cost = (
                        monthly_requests * serverless_seconds_per_request * serverless_price_per_gb_s * serverless_gb
                    )
                    serverless_request_cost = monthly_requests * serverless_price_per_request
                    total_serverless_cost = serverless_compute_cost + serverless_request_cost
                    
                    # Real-time endpoint costs - 24/7 uptime
                    realtime_hours_per_month = 24 * 30
                    realtime_monthly_cost = instance_hourly_rates[instance_type] * instance_count * realtime_hours_per_month
                    
                    # Create comparison visualization
                    cost_data = pd.DataFrame({
                        "Deployment Type": ["Serverless Inference", "Real-Time Endpoint"],
                        "Monthly Cost ($)": [total_serverless_cost, realtime_monthly_cost]
                    })
                    
                    fig = px.bar(
                        cost_data,
                        x="Deployment Type",
                        y="Monthly Cost ($)",
                        color="Deployment Type",
                        text_auto='.2f',
                        title="Monthly Cost Comparison",
                        color_discrete_sequence=[AWS_COLORS["light_blue"], AWS_COLORS["orange"]]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show cost breakdown
                    st.markdown("#### Cost Breakdown")
                    
                    st.markdown("**Serverless Inference:**")
                    st.markdown(f"- Compute cost: ${serverless_compute_cost:.2f}")
                    st.markdown(f"- Request cost: ${serverless_request_cost:.2f}")
                    st.markdown(f"- Total monthly cost: ${total_serverless_cost:.2f}")
                    
                    st.markdown("**Real-Time Endpoint:**")
                    st.markdown(f"- Instance cost ({instance_count} x {instance_type}): ${realtime_monthly_cost:.2f}")
                    st.markdown(f"- Total monthly cost: ${realtime_monthly_cost:.2f}")
                    
                    # Recommendation
                    if total_serverless_cost < realtime_monthly_cost:
                        savings = realtime_monthly_cost - total_serverless_cost
                        savings_percent = (savings / realtime_monthly_cost) * 100
                        
                        st.markdown(f"""
                        <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin-top:15px;">
                            <h4>Recommendation: Use Serverless Inference</h4>
                            <p>Serverless Inference is more cost-effective for your workload.</p>
                            <p>Estimated monthly savings: <b>${savings:.2f}</b> ({savings_percent:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        savings = total_serverless_cost - realtime_monthly_cost
                        savings_percent = (savings / total_serverless_cost) * 100
                        
                        st.markdown(f"""
                        <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin-top:15px;">
                            <h4>Recommendation: Use Real-Time Endpoint</h4>
                            <p>A traditional endpoint is more cost-effective for your workload.</p>
                            <p>Estimated monthly savings: <b>${savings:.2f}</b> ({savings_percent:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Best use cases for serverless inference
            st.markdown('<div class="sub-header">Best Use Cases for Serverless Inference</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📊 Sporadic Traffic")
                st.markdown("""
                - Applications with unpredictable usage patterns
                - Development and testing environments
                - Periodic batch processing jobs
                - Demo applications
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 💰 Cost Optimization")
                st.markdown("""
                - Low to moderate throughput applications
                - Applications where idling costs are significant
                - Startups with limited budgets
                - Pay-as-you-go pricing model preferred
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🚀 Fast Deployment")
                st.markdown("""
                - Models that need quick deployment
                - Proof of concept implementations
                - Temporary or seasonal applications
                - Multiple model variants for A/B testing
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
        elif advanced_tab == "Inference Pipelines":
            st.markdown('<div class="sub-header">Inference Pipelines</div>', unsafe_allow_html=True)
            
            st.markdown("""
            Inference pipelines allow you to chain multiple models and preprocessing/postprocessing steps:
            
            - Create an end-to-end ML pipeline with multiple models
            - Chain preprocessing, inference, and postprocessing steps
            - Pass data automatically between stages
            - Deploy the entire pipeline as a single endpoint
            """)
            
            # Inference pipeline diagram
            try:
                import graphviz
                
                g = graphviz.Digraph()
                g.attr(rankdir='LR')
                
                # Define nodes
                g.node('preprocess', 'Data\nPreprocessing', shape='box', style='filled', fillcolor='#D5E8D4')
                g.node('model1', 'Feature\nExtraction\nModel', shape='box', style='filled', fillcolor='#DAE8FC')
                g.node('model2', 'Classification\nModel', shape='box', style='filled', fillcolor='#FFE6CC')
                g.node('postprocess', 'Response\nFormatting', shape='box', style='filled', fillcolor='#D5E8D4')
                
                # Define edges
                g.edge('preprocess', 'model1')
                g.edge('model1', 'model2')
                g.edge('model2', 'postprocess')
                
                # Add input and output
                g.node('input', 'Raw Input', shape='oval', style='filled', fillcolor='#F8CECC')
                g.node('output', 'Prediction', shape='oval', style='filled', fillcolor='#F8CECC')
                
                g.edge('input', 'preprocess')
                g.edge('postprocess', 'output')
                
                st.graphviz_chart(g)
                
            except:
                st.error("Graphviz is required to display the inference pipeline diagram.")
                # Fallback image
                st.image("https://docs.aws.amazon.com/sagemaker/latest/dg/images/inference-pipeline.png", 
                        caption="Inference Pipeline Architecture")
            
            # Inference pipeline example
            st.markdown('<div class="info-box">Example: Creating an Inference Pipeline</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            pipeline_code = """
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.xgboost.model import XGBoostModel

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Step 1: Define preprocessing model
sklearn_preprocessor = SKLearnModel(
    model_data="s3://my-bucket/models/preprocessor/model.tar.gz",
    role=role,
    entry_point="preprocessing.py",
    framework_version="0.23-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
    name="feature-preprocessor"
)

# Step 2: Define inference model
xgb_model = XGBoostModel(
    model_data="s3://my-bucket/models/xgboost/model.tar.gz",
    role=role,
    entry_point="inference.py",
    framework_version="1.3-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
    name="churn-predictor"
)

# Step 3: Define postprocessing model
sklearn_postprocessor = SKLearnModel(
    model_data="s3://my-bucket/models/postprocessor/model.tar.gz",
    role=role,
    entry_point="postprocessing.py",
    framework_version="0.23-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
    name="response-formatter"
)

# Create the pipeline model
pipeline_model = PipelineModel(
    name="churn-prediction-pipeline",
    models=[sklearn_preprocessor, xgb_model, sklearn_postprocessor],
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy the pipeline as a single endpoint
pipeline_predictor = pipeline_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="churn-inference-pipeline"
)

# Example: Invoke the pipeline with raw data
raw_input = {
    "customer_id": "C123456",
    "demographics": {
        "age": 42,
        "income": "52000",
        "state": "CA"
    },
    "behavior": {
        "login_count": 5,
        "avg_session_minutes": 8.3,
        "last_active_days": 2
    },
    "purchase_history": [
        {"date": "2023-01-15", "amount": 120.50},
        {"date": "2023-02-28", "amount": 85.20}
    ]
}

import json
response = pipeline_predictor.predict(json.dumps(raw_input))
print(f"Pipeline response: {response}")
"""
            st.code(pipeline_code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Example preprocessing script
            st.markdown("### Example Preprocessing Script")
            
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            preproc_script = """
# preprocessing.py - Script for the preprocessing container

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the scaler model when container starts
def model_fn(model_dir):
    "Load the scikit-learn model from the model_dir."
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    return preprocessor

def input_fn(request_body, request_content_type):
    "Parse input data payload"
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        # Extract features from nested JSON
        features = {}
        
        # Extract demographic features
        features['age'] = data['demographics'].get('age', 0)
        features['income'] = float(data['demographics'].get('income', 0))
        
        # Convert state to one-hot encoding for the most common states
        state = data['demographics'].get('state', 'unknown')
        common_states = ['CA', 'NY', 'TX', 'FL', 'IL']
        for s in common_states:
            features[f'state_{s}'] = 1.0 if state == s else 0.0
        
        # Extract behavioral features
        features['login_count'] = data['behavior'].get('login_count', 0)
        features['avg_session_minutes'] = data['behavior'].get('avg_session_minutes', 0)
        features['days_since_active'] = data['behavior'].get('last_active_days', 0)
        
        # Calculate purchase metrics
        purchases = data.get('purchase_history', [])
        features['purchase_count'] = len(purchases)
        features['total_spent'] = sum(p.get('amount', 0) for p in purchases)
        
        if purchases:
            # Calculate days since last purchase
            from datetime import datetime
            latest_purchase = max(datetime.strptime(p.get('date', '2000-01-01'), '%Y-%m-%d') for p in purchases)
            features['days_since_purchase'] = (datetime.now() - latest_purchase).days
        else:
            features['days_since_purchase'] = 365  # Default value
        
        # Convert to DataFrame for processing
        return pd.DataFrame([features])
    else:
        # Handle other content types or raise an error
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    "Preprocess the input data using the model"
    # Apply preprocessing transformations
    preprocessed_data = model.transform(input_data)
    
    # Return preprocessed data as numpy array
    return preprocessed_data

def output_fn(prediction, content_type):
    "Format the prediction output"
    if content_type == 'application/json':
        # Convert numpy array to list for JSON serialization
        return json.dumps(prediction.tolist())
    else:
        return prediction
"""
            st.code(preproc_script, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Use cases
            st.markdown('<div class="sub-header">Inference Pipeline Use Cases</div>', unsafe_allow_html=True)
            
            use_cases = [
                {
                    "title": "Computer Vision Processing",
                    "description": "A pipeline that preprocesses images, extracts features with a CNN, and classifies objects with a secondary model.",
                    "steps": ["Image Preprocessing", "Feature Extraction CNN", "Classification Model"]
                },
                {
                    "title": "NLP Document Analysis",
                    "description": "Process text documents through tokenization, embedding generation, and multi-class classification.",
                    "steps": ["Text Tokenization", "BERT Embedding Generation", "Document Classifier"]
                },
                {
                    "title": "Multi-stage Recommendation",
                    "description": "Generate personalized recommendations through candidate generation, ranking, and diversity filtering.",
                    "steps": ["Candidate Generation Model", "Ranking Model", "Diversity Filter"]
                },
                {
                    "title": "Time Series Forecasting",
                    "description": "Process time series data, detect anomalies, and generate forecasts with confidence intervals.",
                    "steps": ["Signal Processing", "Anomaly Detection", "Forecast Generation", "Confidence Interval Calculation"]
                }
            ]
            
            selected_case = st.selectbox(
                "Select a Pipeline Use Case",
                [case["title"] for case in use_cases]
            )
            
            # Find the selected case
            selected_case_details = next((case for case in use_cases if case["title"] == selected_case), None)
            
            if selected_case_details:
                st.markdown(f"### {selected_case_details['title']}")
                st.markdown(selected_case_details['description'])
                
                # Create a visual pipeline diagram
                try:
                    import graphviz
                    
                    g = graphviz.Digraph()
                    g.attr(rankdir='LR')
                    
                    # Add input node
                    g.node('input', 'Input\nData', shape='oval', style='filled', fillcolor='#F8CECC')
                    
                    # Add pipeline steps
                    prev_node = 'input'
                    for i, step in enumerate(selected_case_details['steps']):
                        node_id = f'step{i}'
                        g.node(node_id, step, shape='box', style='filled', 
                              fillcolor=[AWS_COLORS["orange"], AWS_COLORS["light_blue"], AWS_COLORS["green"], AWS_COLORS["purple"]][i % 4])
                        g.edge(prev_node, node_id)
                        prev_node = node_id
                    
                    # Add output node
                    g.node('output', 'Output\nPrediction', shape='oval', style='filled', fillcolor='#F8CECC')
                    g.edge(prev_node, 'output')
                    
                    st.graphviz_chart(g)
                    
                except:
                    st.error("Graphviz is required to display the pipeline diagram.")
                    
                    # Display as text instead
                    st.markdown("**Pipeline Steps:**")
                    for i, step in enumerate(selected_case_details['steps']):
                        st.markdown(f"{i+1}. {step}")
            
        elif advanced_tab == "Asynchronous Inference":
            st.markdown('<div class="sub-header">Asynchronous Inference</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                Asynchronous Inference is ideal for requests that take a long time to process:
                
                - Process requests that require longer computation time
                - Handle larger payload sizes (up to 1GB)
                - Queue requests for processing instead of timing out
                - Economical for workloads with spike patterns
                - Better throughput for batch-style requests
                """)
            
            with col2:
                # Asynchronous endpoint illustration
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Create a simple diagram
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.axis('off')
                
                # Draw components - Client
                ax.add_patch(plt.Rectangle((1, 8), 2, 1, fill=True, alpha=0.3, color=AWS_COLORS["orange"], linewidth=2))
                ax.text(2, 8.5, "Client", ha='center', va='center', fontsize=10)
                
                # Async Endpoint
                ax.add_patch(plt.Rectangle((4.5, 6), 3, 1.5, fill=True, alpha=0.3, color=AWS_COLORS["light_blue"], linewidth=2))
                ax.text(6, 6.75, "Async Endpoint", ha='center', va='center', fontsize=10)
                
                # Request Queue
                ax.add_patch(plt.Rectangle((4.5, 4), 3, 1, fill=True, alpha=0.3, color=AWS_COLORS["green"], linewidth=2))
                ax.text(6, 4.5, "Request Queue", ha='center', va='center', fontsize=10)
                
                # S3 Bucket
                ax.add_patch(plt.Rectangle((8, 6), 1.5, 1.5, fill=True, alpha=0.3, color=AWS_COLORS["purple"], linewidth=2))
                ax.text(8.75, 6.75, "S3\nBucket", ha='center', va='center', fontsize=10)
                
                # Draw arrows
                ax.arrow(2, 7.9, 2.5, -1.1, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
                ax.text(3, 7.5, "1. Submit Request", ha='center', va='center', fontsize=8)
                
                ax.arrow(6, 6, 0, -1, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
                ax.text(6.5, 5.5, "2. Queue", ha='center', va='center', fontsize=8)
                
                ax.arrow(6, 4, 0, -1, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
                ax.text(6.5, 3.5, "3. Process", ha='center', va='center', fontsize=8)
                
                ax.arrow(7.5, 6.5, 0.5, 0, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"])
                ax.text(8, 6.2, "4. Store\nResults", ha='center', va='center', fontsize=8)
                
                ax.arrow(8, 7.5, -4, 0.5, head_width=0.2, head_length=0.2, fc=AWS_COLORS["dark_blue"], ec=AWS_COLORS["dark_blue"], linestyle='dashed')
                ax.text(5, 8.2, "5. Check Status / Get Results", ha='center', va='center', fontsize=8)
                
                st.pyplot(fig)
            
            # Asynchronous inference example
            st.markdown('<div class="info-box">Example: Using Asynchronous Inference</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="code-box">', unsafe_allow_html=True)
            async_code = """
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
import time
import json

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
s3_client = boto_session.client('s3')
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define S3 bucket for input and output
bucket = sagemaker_session.default_bucket()
input_path = f"s3://{bucket}/async-inference/input"
output_path = f"s3://{bucket}/async-inference/output"

# Create model
model = Model(
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/large-model:latest", 
    model_data="s3://my-bucket/models/large-model/model.tar.gz",
    role=role,
    sagemaker_session=sagemaker_session,
    name="large-prediction-model"
)

# Configure async inference
async_config = AsyncInferenceConfig(
    output_path=output_path,
    max_concurrent_invocations_per_instance=4,
    max_payload_size=1000 * 1024 * 1024  # 1000 MB
)

# Deploy model for async inference
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.2xlarge",
    endpoint_name="large-model-async",
    async_inference_config=async_config
)

# Create a runtime client for invoking the endpoint
runtime_client = boto_session.client('sagemaker-runtime')

# Function to invoke the async endpoint
def invoke_async_endpoint(data, endpoint_name):
    response = runtime_client.invoke_endpoint_async(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(data)
    )
    return response['OutputLocation'], response['InferenceId']

# Function to check the status of an async inference
def check_inference_status(inference_id, endpoint_name):
    response = runtime_client.describe_endpoint_async_inference_config(
        EndpointName=endpoint_name
    )
    return response['Status']

# Function to download results when ready
def get_inference_result(output_location):
    # Parse S3 location
    s3_parts = output_location.replace("s3://", "").split("/")
    bucket_name = s3_parts[0]
    key = "/".join(s3_parts[1:])
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        print(f"Error getting results: {e}")
        return None

# Example: Large data processing request
large_data = {
    "inputs": "This is a large text document with thousands of words...",
    "parameters": {
        "max_length": 1000,
        "do_sample": True,
        "temperature": 0.7
    }
}

# Submit the request
output_location, inference_id = invoke_async_endpoint(
    large_data, 
    "large-model-async"
)

print(f"Request submitted. Inference ID: {inference_id}")
print(f"Results will be available at: {output_location}")

# Poll for completion (in a real application, implement better waiting mechanisms)
max_wait_time = 300  # 5 minutes
wait_time = 0
poll_interval = 10  # seconds

while wait_time < max_wait_time:
    status = check_inference_status(inference_id, "large-model-async")
    if status == "Completed":
        print("Inference completed!")
        results = get_inference_result(output_location)
        print(f"Results: {results}")
        break
    elif status == "Failed":
        print("Inference failed!")
        break
    
    print(f"Waiting for results... Status: {status}")
    time.sleep(poll_interval)
    wait_time += poll_interval

# Cleanup - delete the endpoint when finished
# predictor.delete_endpoint()
"""
            st.code(async_code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Use cases for asynchronous inference
            st.markdown('<div class="sub-header">When to Use Asynchronous Inference</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📊 Long-Running Inferences")
                st.markdown("""
                - Large language model processing
                - Video analysis and processing
                - Complex anomaly detection
                - Deep learning models with long inference times
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 💾 Large Payload Sizes")
                st.markdown("""
                - High-resolution images
                - Large document processing
                - Audio and video file analysis
                - Batch data processing
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### ⚡ Spike Traffic Patterns")
                st.markdown("""
                - End-of-day processing
                - Report generation
                - Periodic batch requests
                - Event-driven workloads
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Comparison between invocation types
            st.markdown('<div class="sub-header">Synchronous vs. Asynchronous Invocation</div>', unsafe_allow_html=True)
            
            comparison_data = {
                "Feature": [
                    "Response Time", 
                    "Maximum Timeout", 
                    "Maximum Payload Size", 
                    "Use Case",
                    "Implementation Complexity",
                    "Cost Efficiency for Spiky Traffic"
                ],
                "Synchronous": [
                    "Immediate response required", 
                    "60 seconds", 
                    "6 MB", 
                    "Interactive applications",
                    "Simpler - direct request/response",
                    "Lower - instances must be provisioned for peak"
                ],
                "Asynchronous": [
                    "Can wait for response", 
                    "Up to 15 minutes", 
                    "1 GB", 
                    "Batch, long-running processing",
                    "More complex - request/poll/fetch pattern",
                    "Higher - requests are queued and processed as resources available"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Interactive diagram showing request flow differences
            st.markdown("### Request Flow Comparison")
            
            tab1, tab2 = st.tabs(["Synchronous Invocation", "Asynchronous Invocation"])
            
            with tab1:
                st.markdown("""
                **Synchronous Invocation Flow:**
                
                1. Client sends request to endpoint
                2. Request is immediately processed by model
                3. Client waits for processing to complete
                4. Response is returned directly to client
                5. If processing exceeds timeout, request fails
                """)
                
                # Visualize synchronous flow
                try:
                    import graphviz
                    
                    g = graphviz.Digraph()
                    g.attr(rankdir='LR')
                    
                    g.node('client', 'Client', shape='box')
                    g.node('endpoint', 'SageMaker Endpoint', shape='box')
                    
                    g.edge('client', 'endpoint', label='1. Request')
                    g.edge('endpoint', 'client', label='4. Response')
                    
                    st.graphviz_chart(g)
                    
                except:
                    st.error("Graphviz is required to display the flow diagram.")
            
            with tab2:
                st.markdown("""
                **Asynchronous Invocation Flow:**
                
                1. Client sends request to async endpoint
                2. Endpoint immediately returns an inference ID
                3. Request is queued for processing
                4. Processing happens in the background
                5. Results are stored in S3 when complete
                6. Client polls for completion or is notified
                7. Client retrieves results from S3
                """)
                
                # Visualize asynchronous flow
                try:
                    import graphviz
                    
                    g = graphviz.Digraph()
                    g.attr(rankdir='LR')
                    
                    g.node('client', 'Client', shape='box')
                    g.node('endpoint', 'Async Endpoint', shape='box')
                    g.node('queue', 'Request Queue', shape='box')
                    g.node('s3', 'S3 Bucket', shape='box')
                    
                    g.edge('client', 'endpoint', label='1. Request')
                    g.edge('endpoint', 'client', label='2. Inference ID')
                    g.edge('endpoint', 'queue', label='3. Queue request')
                    g.edge('queue', 'endpoint', label='4. Process')
                    g.edge('endpoint', 's3', label='5. Store results')
                    g.edge('client', 'endpoint', label='6. Poll status', style='dashed')
                    g.edge('client', 's3', label='7. Get results', style='dashed')
                    
                    st.graphviz_chart(g)
                    
                except:
                    st.error("Graphviz is required to display the flow diagram.")

# Function to render the monitoring page
def monitoring_page():
    st.markdown('<div class="main-header">Model Monitoring with Amazon SageMaker</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Model monitoring is essential to ensure your deployed models continue to perform well over time.
    SageMaker provides tools to:
    
    - Monitor data quality and detect drift
    - Track model quality and performance degradation
    - Set up alerts for anomalies and bias
    - Automate retraining when necessary
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Quality", "Model Quality", "Bias Drift", "Model Explainability"])
    
    with tab1:
        st.markdown('<div class="sub-header">Data Quality Monitoring</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Data quality monitoring helps detect changes in the statistical properties of your input data:
            
            - Monitor for changes in feature distributions
            - Detect missing values, type mismatches, and range violations
            - Identify when new data differs from training data
            - Alert when data drift exceeds thresholds
            """)
        
        with col2:
            # Data drift illustration
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Generate some data
            np.random.seed(42)
            x = np.linspace(-3, 3, 1000)
            
            # Training data distribution (normal)
            train_dist = np.random.normal(0, 1, 1000)
            
            # Production data distribution (shifted)
            prod_dist = np.random.normal(0.5, 1.2, 1000)
            
            # Plot distributions
            sns.kdeplot(train_dist, ax=ax, label='Training Data', color=AWS_COLORS["light_blue"])
            sns.kdeplot(prod_dist, ax=ax, label='Production Data', color=AWS_COLORS["orange"])
            ax.set_title('Example of Data Drift')
            ax.legend()
            
            st.pyplot(fig)
        
        # SageMaker Model Monitor example
        st.markdown('<div class="info-box">Example: Setting up SageMaker Model Monitor for Data Quality</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        data_monitor_code = """
import boto3
import sagemaker
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from datetime import datetime

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Define data capture configuration for the endpoint
data_capture_config = DataCaptureConfig(
    enable_capture=True,  # Turn on data capture
    sampling_percentage=20,  # Capture 20% of requests
    destination_s3_uri=f"s3://my-bucket/monitor/captured-data/",
    capture_options=["RequestResponse"],  # Capture both request and response
    sagemaker_session=sagemaker_session
)

# Create a DefaultModelMonitor
model_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)

# Point to the baseline data
baseline_data = "s3://my-bucket/training-data/baseline.csv"

# Create baseline statistics and constraints
model_monitor.suggest_baseline(
    baseline_dataset=baseline_data,
    dataset_format=DatasetFormat.csv(),
    output_s3_uri=f"s3://my-bucket/monitor/baseline",
    wait=True
)

# Schedule the monitoring job
monitoring_schedule_name = f"data-quality-monitor-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

model_monitor.create_monitoring_schedule(
    endpoint_input="customer-churn-predictor",
    record_preprocessor_script=None,  # Use default preprocessor
    post_analytics_processor_script=None,  # Use default postprocessor
    output_s3_uri=f"s3://my-bucket/monitor/results",
    statistics=model_monitor.baseline_statistics(),
    constraints=model_monitor.suggested_constraints(),
    schedule_cron_expression="cron(0 * ? * * *)",  # Run hourly
    enable_cloudwatch_metrics=True,
    monitoring_schedule_name=monitoring_schedule_name
)

print(f"Data quality monitoring schedule created: {monitoring_schedule_name}")

# List all monitoring schedules
schedules = sagemaker_session.sagemaker_client.list_monitoring_schedules(
    StatusEquals="Scheduled"
)

for schedule in schedules["MonitoringScheduleSummaries"]:
    print(f"Schedule: {schedule['MonitoringScheduleName']}, Status: {schedule['MonitoringScheduleStatus']}")
"""
        st.code(data_monitor_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive data drift visualization
        st.markdown('<div class="info-box">Interactive Example: Visualizing Data Drift</div>', unsafe_allow_html=True)
        
        st.markdown("### Data Drift Simulation")
        
        # Select a feature to monitor
        feature = st.selectbox(
            "Select feature to monitor for drift:",
            ["Age", "Income", "Transaction Amount", "Web Activity Score", "Product Usage"]
        )
        
        # Time period selection
        drift_period = st.slider("Time Period (Days)", 1, 30, 14)
        
        # Drift intensity
        drift_intensity = st.slider("Drift Intensity", 0.0, 1.0, 0.5, 0.1)
        
        if st.button("Simulate Drift"):
            # Generate time series data
            dates = pd.date_range(end=pd.Timestamp.now(), periods=drift_period)
            
            # Generate synthetic data based on feature selected
            if feature == "Age":
                # Age - normal distribution slowly shifting
                baseline_mean = 35
                baseline_std = 8
                
                # Generate data with gradual drift
                values = []
                for i in range(drift_period):
                    drift_factor = (i / drift_period) * drift_intensity
                    day_mean = baseline_mean + (10 * drift_factor)  # Shift mean up to 10 years older
                    day_std = baseline_std + (2 * drift_factor)  # Increase std dev slightly
                    day_values = np.random.normal(day_mean, day_std, 100)
                    values.append(day_values)
                
                baseline_values = np.random.normal(baseline_mean, baseline_std, 100)
                
            elif feature == "Income":
                # Income - log normal distribution
                baseline_mean = 10.9  # log of income ~$54K
                baseline_std = 0.4
                
                # Generate data with gradual drift
                values = []
                for i in range(drift_period):
                    drift_factor = (i / drift_period) * drift_intensity
                    day_mean = baseline_mean + (0.3 * drift_factor)  # Shift mean income up
                    day_std = baseline_std + (0.2 * drift_factor)  # Increase std dev
                    day_values = np.random.lognormal(day_mean, day_std, 100)
                    values.append(day_values)
                
                baseline_values = np.random.lognormal(baseline_mean, baseline_std, 100)
                
            elif feature == "Transaction Amount":
                # Transaction amount - exponential distribution
                baseline_scale = 50
                
                # Generate data with gradual drift
                values = []
                for i in range(drift_period):
                    drift_factor = (i / drift_period) * drift_intensity
                    day_scale = baseline_scale * (1 - 0.4 * drift_factor)  # Decreasing transaction amounts
                    day_values = np.random.exponential(day_scale, 100)
                    values.append(day_values)
                
                baseline_values = np.random.exponential(baseline_scale, 100)
                
            elif feature == "Web Activity Score":
                # Web activity - bimodal distribution drifting to unimodal
                values = []
                for i in range(drift_period):
                    drift_factor = (i / drift_period) * drift_intensity
                    # Mixture of two normal distributions, with one gradually disappearing
                    weight1 = 0.5 - (0.4 * drift_factor)
                    weight2 = 1 - weight1
                    
                    samples1 = np.random.normal(20, 5, int(100 * weight1))
                    samples2 = np.random.normal(60, 10, int(100 * weight2))
                    
                    combined = np.concatenate([samples1, samples2])
                    np.random.shuffle(combined)
                    
                    # Take only 100 samples
                    day_values = combined[:100]
                    values.append(day_values)
                
                # Baseline is clearly bimodal
                baseline1 = np.random.normal(20, 5, 50)
                baseline2 = np.random.normal(60, 10, 50)
                baseline_values = np.concatenate([baseline1, baseline2])
                
            else:  # Product Usage
                # Product Usage - Poisson distribution
                baseline_lambda = 5
                
                # Generate data with gradual drift
                values = []
                for i in range(drift_period):
                    drift_factor = (i / drift_period) * drift_intensity
                    day_lambda = baseline_lambda + (8 * drift_factor)  # Increasing usage
                    day_values = np.random.poisson(day_lambda, 100)
                    values.append(day_values)
                
                baseline_values = np.random.poisson(baseline_lambda, 100)
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribution Over Time")
                
                # Create violin plot showing drift over time
                df_list = []
                
                # Add each day's data to the dataframe
                for i, day_values in enumerate(values):
                    day_df = pd.DataFrame({
                        'Day': i+1,
                        'Value': day_values
                    })
                    df_list.append(day_df)
                
                # Combine all days
                drift_df = pd.concat(df_list)
                
                # Create violin plot
                fig = px.violin(
                    drift_df, 
                    x='Day', 
                    y='Value',
                    box=True,
                    title=f'{feature} Distribution Over Time',
                    color_discrete_sequence=[AWS_COLORS["light_blue"]]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Baseline vs. Current Comparison")
                
                # Create histogram comparing baseline to most recent
                compare_df = pd.DataFrame({
                    'Period': ['Baseline'] * len(baseline_values) + ['Current'] * len(values[-1]),
                    'Value': np.concatenate([baseline_values, values[-1]])
                })
                
                fig = px.histogram(
                    compare_df,
                    x='Value',
                    color='Period',
                    barmode='overlay',
                    opacity=0.7,
                    nbins=30,
                    title=f'{feature}: Baseline vs. Current Distribution',
                    color_discrete_sequence=[AWS_COLORS["green"], AWS_COLORS["orange"]]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate drift metrics
            st.markdown("### Drift Detection Metrics")
            
            # Calculate simple statistics
            baseline_mean = np.mean(baseline_values)
            current_mean = np.mean(values[-1])
            
            mean_diff = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else float('inf')
            
            baseline_std = np.std(baseline_values)
            current_std = np.std(values[-1])
            
            std_diff = ((current_std - baseline_std) / baseline_std) * 100 if baseline_std != 0 else float('inf')
            
            # KL divergence approximation using histograms
            def kl_divergence_from_samples(p_samples, q_samples, bins=20):
                # Compute histogram for p and q
                min_val = min(np.min(p_samples), np.min(q_samples))
                max_val = max(np.max(p_samples), np.max(q_samples))
                
                bin_edges = np.linspace(min_val, max_val, bins+1)
                p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=True)
                q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=True)
                
                # Add small epsilon to avoid division by zero
                p_hist = np.maximum(p_hist, 1e-10)
                q_hist = np.maximum(q_hist, 1e-10)
                
                # Compute KL divergence
                kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
                return kl_div
            
            kl_div = kl_divergence_from_samples(baseline_values, values[-1])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Mean Shift", 
                    f"{current_mean:.2f}", 
                    f"{mean_diff:.1f}%",
                    delta_color="inverse" if abs(mean_diff) > 10 else "normal"
                )
            
            with col2:
                st.metric(
                    "Std Dev Shift", 
                    f"{current_std:.2f}", 
                    f"{std_diff:.1f}%",
                    delta_color="inverse" if abs(std_diff) > 20 else "normal"
                )
                
            with col3:
                st.metric(
                    "KL Divergence", 
                    f"{kl_div:.4f}",
                    None
                )
            
            # Drift status
            drift_threshold = 0.1
            if kl_div > drift_threshold:
                st.markdown(f"""
                <div style="background-color:#F5B7B1; padding:15px; border-radius:5px; margin-top:15px;">
                    <h4>🚨 Data Drift Detected!</h4>
                    <p>The distribution of {feature} has changed significantly from the baseline.</p>
                    <p>Recommended Actions:</p>
                    <ul>
                        <li>Analyze the drift cause</li>
                        <li>Update feature transformations</li>
                        <li>Consider retraining the model</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin-top:15px;">
                    <h4>✅ No Significant Data Drift</h4>
                    <p>The distribution of {feature} is within acceptable limits compared to baseline.</p>
                    <p>Continue monitoring for changes.</p>
                </div>
                """, unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="sub-header">Model Quality Monitoring</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Model quality monitoring tracks how well your deployed model is performing over time:
        
        - Monitor prediction accuracy, F1 score, precision, recall, etc.
        - Compare metrics against baseline performance
        - Detect performance degradation
        - Get alerted when model quality drops below thresholds
        - Useful when ground truth labels become available after predictions
        """)
        
        # Model quality monitoring example
        st.markdown('<div class="info-box">Example: Setting up Model Quality Monitoring</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        quality_monitor_code = """
import boto3
import sagemaker
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from datetime import datetime

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create a ModelQualityMonitor
model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)

# Point to baseline data with predictions and ground truth
baseline_data = "s3://my-bucket/model-quality/baseline.csv"

# Create baseline statistics and constraints
model_quality_monitor.suggest_baseline(
    baseline_dataset=baseline_data,
    dataset_format=DatasetFormat.csv(),
    problem_type='BinaryClassification',  # For churn prediction
    inference_attribute='prediction',      # Column name for model predictions
    probability_attribute='probability',   # Column name for prediction probabilities
    ground_truth_attribute='actual',       # Column name for ground truth labels
    output_s3_uri=f"s3://my-bucket/model-quality/baseline",
    wait=True
)

# Schedule the monitoring job
monitoring_schedule_name = f"model-quality-monitor-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

model_quality_monitor.create_monitoring_schedule(
    endpoint_input="customer-churn-predictor",
    ground_truth_input=f"s3://my-bucket/model-quality/ground-truth",  # Where ground truth will be uploaded
    record_preprocessor_script=None,  # Use default preprocessor
    post_analytics_processor_script=None,  # Use default postprocessor
    output_s3_uri=f"s3://my-bucket/model-quality/results",
    statistics=model_quality_monitor.baseline_statistics(),
    constraints=model_quality_monitor.suggested_constraints(),
    schedule_cron_expression="cron(0 0 * * ? *)",  # Run daily
    enable_cloudwatch_metrics=True,
    monitoring_schedule_name=monitoring_schedule_name
)

print(f"Model quality monitoring schedule created: {monitoring_schedule_name}")

# Example: Upload ground truth data for evaluation
# In a real scenario, this would be done periodically as ground truth becomes available

# Sample ground truth data format:
# {
#     "groundTruthData": {
#         "data": "0,1,0,1,1,0",  # actual churn labels
#         "encoding": "CSV"
#     },
#     "eventMetadata": {
#         "eventId": "aaaaaaaa-bbbb-cccc-dddd-example012345"
#     },
#     "eventVersion": "0",
#     "inferenceId": "111122223333",
#     "captureTimeInMillis": 1647080894991
# }

import json
from datetime import datetime, timezone

def upload_ground_truth(ground_truth_values, s3_ground_truth_path, capture_timestamp=None):
    "Upload ground truth data for model quality monitoring."
    
    s3_client = boto_session.client('s3')
    
    # Use current time if not provided
    if capture_timestamp is None:
        capture_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    # Format ground truth data
    ground_truth_data = {
        "groundTruthData": {
            "data": ",".join(map(str, ground_truth_values)),
            "encoding": "CSV"
        },
        "eventMetadata": {
            "eventId": f"gtruth-{capture_timestamp}"
        },
        "eventVersion": "0",
        "inferenceId": f"infer-{capture_timestamp}",
        "captureTimeInMillis": capture_timestamp
    }
    
    # Convert to JSON
    ground_truth_json = json.dumps(ground_truth_data)
    
    # Parse S3 path
    s3_path_parts = s3_ground_truth_path.replace("s3://", "").split("/")
    bucket = s3_path_parts[0]
    key_prefix = "/".join(s3_path_parts[1:])
    
    # Create full S3 key with timestamp
    timestamp_str = datetime.fromtimestamp(capture_timestamp/1000).strftime('%Y/%m/%d/%H/%M')
    full_key = f"{key_prefix}/{timestamp_str}/groundtruth.json"
    
    # Upload to S3
    s3_client.put_object(
        Body=ground_truth_json,
        Bucket=bucket,
        Key=full_key
    )
    
    return f"s3://{bucket}/{full_key}"

# Example usage:
ground_truth = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # Example ground truth labels
s3_path = "s3://my-bucket/model-quality/ground-truth"

uploaded_path = upload_ground_truth(ground_truth, s3_path)
print(f"Uploaded ground truth to: {uploaded_path}")
"""
        st.code(quality_monitor_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive model quality visualization
        st.markdown('<div class="info-box">Interactive Example: Model Quality Tracking</div>', unsafe_allow_html=True)
        
        # Tabs for different model quality charts
        quality_chart = st.radio(
            "Select metric to visualize:",
            ["Accuracy", "Precision/Recall", "F1 Score", "ROC Curve"]
        )
        
        # Time period selection
        time_period = st.slider("Time Period (Days)", 7, 90, 30)
        
        # Degradation pattern
        degradation_pattern = st.selectbox(
            "Degradation Pattern",
            ["Gradual Decline", "Sudden Drop", "Seasonal Variation", "Stable Performance"]
        )
        
        if st.button("Generate Model Quality Chart"):
            # Generate dates
            dates = pd.date_range(end=pd.Timestamp.now(), periods=time_period)
            
            # Generate base metrics based on selected pattern
            if degradation_pattern == "Gradual Decline":
                # Start high and gradually decline
                base_trend = np.linspace(0.92, 0.78, time_period)
                noise_level = 0.02
            elif degradation_pattern == "Sudden Drop":
                # Stable then sudden drop at 2/3 of the time period
                base_trend = np.ones(time_period) * 0.90
                drop_point = int(time_period * 2/3)
                base_trend[drop_point:] = 0.75
                noise_level = 0.015
            elif degradation_pattern == "Seasonal Variation":
                # Cyclical pattern
                x = np.linspace(0, 3*np.pi, time_period)
                base_trend = 0.85 + 0.05 * np.sin(x)
                noise_level = 0.02
            else:  # Stable Performance
                # Consistently high
                base_trend = np.ones(time_period) * 0.89
                noise_level = 0.01
            
            # Add noise to the trend
            noise = np.random.normal(0, noise_level, time_period)
            accuracy = np.clip(base_trend + noise, 0.5, 0.99)
            
            # Create related metrics
            precision = np.clip(accuracy + np.random.normal(0.01, 0.02, time_period), 0.5, 0.99)
            recall = np.clip(accuracy + np.random.normal(-0.03, 0.02, time_period), 0.5, 0.99)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            # Create dataframe
            metrics_df = pd.DataFrame({
                'Date': dates,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
            
            # Show appropriate chart based on selection
            if quality_chart == "Accuracy":
                fig = px.line(
                    metrics_df, 
                    x='Date', 
                    y='Accuracy',
                    title='Model Accuracy Over Time',
                    markers=True
                )
                
                # Add threshold line
                fig.add_shape(
                    type="line",
                    x0=metrics_df['Date'].min(),
                    x1=metrics_df['Date'].max(),
                    y0=0.8,
                    y1=0.8,
                    line=dict(color="red", dash="dash"),
                )
                
                fig.add_annotation(
                    x=metrics_df['Date'].min(),
                    y=0.8,
                    text="Threshold",
                    showarrow=False,
                    yshift=10
                )
                
                # Set y-axis range
                fig.update_layout(yaxis_range=[0.7, 1.0])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show threshold violations
                violations = metrics_df[metrics_df['Accuracy'] < 0.8]
                if not violations.empty:
                    st.markdown(f"""
                    <div style="background-color:#F5B7B1; padding:15px; border-radius:5px; margin-top:15px;">
                        <h4>🚨 Accuracy Threshold Violations Detected!</h4>
                        <p>The model accuracy fell below the threshold of 0.8 on {len(violations)} days.</p>
                        <p>First violation: {violations['Date'].min().strftime('%Y-%m-%d')}</p>
                        <p>Recommended Actions:</p>
                        <ul>
                            <li>Investigate changes in input data</li>
                            <li>Check for data drift in key features</li>
                            <li>Consider retraining the model with recent data</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin-top:15px;">
                        <h4>✅ Model Accuracy Within Acceptable Range</h4>
                        <p>The model accuracy has remained above the threshold of 0.8 for the entire period.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif quality_chart == "Precision/Recall":
                # Melt dataframe for multiple lines
                plot_df = metrics_df.melt(
                    id_vars=['Date'],
                    value_vars=['Precision', 'Recall'],
                    var_name='Metric',
                    value_name='Value'
                )
                
                fig = px.line(
                    plot_df,
                    x='Date',
                    y='Value',
                    color='Metric',
                    title='Precision and Recall Over Time',
                    markers=True,
                    color_discrete_sequence=[AWS_COLORS["green"], AWS_COLORS["orange"]]
                )
                
                # Add threshold line
                fig.add_shape(
                    type="line",
                    x0=metrics_df['Date'].min(),
                    x1=metrics_df['Date'].max(),
                    y0=0.75,
                    y1=0.75,
                    line=dict(color="red", dash="dash"),
                )
                
                # Set y-axis range
                fig.update_layout(yaxis_range=[0.65, 1.0])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show precision-recall tradeoff
                st.markdown("### Precision-Recall Tradeoff")
                
                fig = px.scatter(
                    metrics_df,
                    x='Recall',
                    y='Precision',
                    title='Precision vs. Recall',
                    color_discrete_sequence=[AWS_COLORS["light_blue"]],
                )
                
                # Add ideal point annotation
                fig.add_annotation(
                    x=0.95,
                    y=0.95,
                    text="Ideal Point",
                    showarrow=True,
                    arrowhead=1
                )
                
                # Set axis ranges
                fig.update_layout(
                    xaxis_range=[0.65, 1.0],
                    yaxis_range=[0.65, 1.0]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif quality_chart == "F1 Score":
                fig = px.line(
                    metrics_df, 
                    x='Date', 
                    y='F1 Score',
                    title='F1 Score Over Time',
                    markers=True,
                    color_discrete_sequence=[AWS_COLORS["purple"]]
                )
                
                # Add threshold line
                fig.add_shape(
                    type="line",
                    x0=metrics_df['Date'].min(),
                    x1=metrics_df['Date'].max(),
                    y0=0.78,
                    y1=0.78,
                    line=dict(color="red", dash="dash"),
                )
                
                # Set y-axis range
                fig.update_layout(yaxis_range=[0.65, 1.0])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate trend
                from scipy import stats
                
                x = np.arange(len(metrics_df))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, metrics_df['F1 Score'])
                
                trend_direction = "improving" if slope > 0.0001 else "declining" if slope < -0.0001 else "stable"
                trend_strength = abs(r_value)
                
                st.markdown(f"""
                ### F1 Score Trend Analysis
                
                - **Trend Direction:** {trend_direction.title()}
                - **Trend Strength:** {trend_strength:.2f} (R-value)
                - **Slope:** {slope:.6f} per day
                
                {
                    "The model performance is declining over time. Consider retraining with recent data." 
                    if trend_direction == "declining" and trend_strength > 0.5
                    else "The model performance is improving. Recent changes may be beneficial."
                    if trend_direction == "improving" and trend_strength > 0.5
                    else "The model performance is relatively stable."
                }
                """)
            
            else:  # ROC Curve
                # Generate synthetic ROC curves for three time periods
                start_idx = 0
                mid_idx = time_period // 3
                end_idx = time_period - 1
                
                # Function to generate ROC curve points
                def generate_roc_curve(auc_value):
                    # Generate ROC curve with specified AUC
                    points = 100
                    x = np.linspace(0, 1, points)
                    
                    # Using a simple parametric function to approximate ROC curve
                    # with desired AUC (this is a simplification)
                    a = -np.log(0.5) / (auc_value * 2 - 1)
                    y = np.exp(-a * x)
                    
                    return x, y
                
                # Get AUC values from the accuracy with small variations
                auc_start = accuracy[start_idx]
                auc_mid = accuracy[mid_idx]
                auc_end = accuracy[end_idx]
                
                # Generate ROC curves
                fpr_start, tpr_start = generate_roc_curve(auc_start)
                fpr_mid, tpr_mid = generate_roc_curve(auc_mid)
                fpr_end, tpr_end = generate_roc_curve(auc_end)
                
                # Create plot
                fig = go.Figure()
                
                # Add ROC curves
                fig.add_trace(go.Scatter(
                    x=fpr_start, y=tpr_start,
                    name=f'Initial (AUC={auc_start:.3f})',
                    line=dict(color=AWS_COLORS["green"], width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=fpr_mid, y=tpr_mid,
                    name=f'Mid-period (AUC={auc_mid:.3f})',
                    line=dict(color=AWS_COLORS["orange"], width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=fpr_end, y=tpr_end,
                    name=f'Current (AUC={auc_end:.3f})',
                    line=dict(color=AWS_COLORS["red"] if auc_end < 0.8 else AWS_COLORS["light_blue"], width=2)
                ))
                
                # Add diagonal reference line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random (AUC=0.5)',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                # Update layout
                fig.update_layout(
                    title='ROC Curve Evolution Over Time',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    xaxis=dict(constrain='domain'),
                    width=700,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance change summary
                performance_change = (auc_end - auc_start) * 100
                
                if performance_change < -5:
                    st.markdown(f"""
                    <div style="background-color:#F5B7B1; padding:15px; border-radius:5px; margin-top:15px;">
                        <h4>🚨 Model Performance Degradation Detected</h4>
                        <p>The model's AUC has declined by {abs(performance_change):.1f}% over the time period.</p>
                        <p>Recommended Actions:</p>
                        <ul>
                            <li>Check for data drift in your feature distributions</li>
                            <li>Analyze misclassified samples to identify patterns</li>
                            <li>Consider retraining the model with more recent data</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif performance_change > 5:
                    st.markdown(f"""
                    <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin-top:15px;">
                        <h4>✅ Model Performance Improvement</h4>
                        <p>The model's AUC has improved by {performance_change:.1f}% over the time period.</p>
                        <p>This could be due to:</p>
                        <ul>
                            <li>Recent retraining or fine-tuning</li>
                            <li>Data quality improvements</li>
                            <li>Changing patterns that better match the model's learned patterns</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin-top:15px;">
                        <h4>✅ Model Performance is Stable</h4>
                        <p>The model's AUC has changed by only {performance_change:.1f}% over the time period.</p>
                        <p>Continue monitoring for any significant changes.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
    with tab3:
        st.markdown('<div class="sub-header">Bias Drift Monitoring</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Bias drift monitoring helps detect if your model is developing unfair biases over time:
        
        - Track fairness metrics across different sensitive groups
        - Compare current bias metrics to the baseline
        - Detect when bias increases beyond acceptable thresholds
        - Ensure deployed models remain fair and ethical
        """)
        
        # Bias drift example
        st.markdown('<div class="info-box">Example: Setting up Bias Drift Monitoring</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        bias_monitor_code = """
import boto3
import sagemaker
from sagemaker.model_monitor import BiasAnalysisConfig, ModelBiasMonitor
from sagemaker.clarify import BiasConfig, DataConfig
from datetime import datetime

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create a ModelBiasMonitor
model_bias_monitor = ModelBiasMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)

# Define the bias config
bias_config = BiasConfig(
    label_values_or_threshold=[1],  # positive outcome (churn=1)
    facet_name="gender",  # protected attribute
    facet_values_or_threshold=["female"]  # protected group
)

# Define data config
data_config = DataConfig(
    s3_data_input_path="s3://my-bucket/bias-monitor/baseline/",
    s3_output_path="s3://my-bucket/bias-monitor/output/",
    label="churn",
    headers=["age", "gender", "income", "balance", "transactions", "credit_score", "churn"],
    dataset_type="text/csv"
)

# Create bias analysis config
bias_analysis_config = BiasAnalysisConfig(
    bias_config=bias_config,
    headers=data_config.headers,
    label=data_config.label
)

# Create baseline for bias metrics
model_bias_monitor.suggest_baseline(
    data_config=data_config,
    bias_config=bias_config,
    headers=data_config.headers,
    label=data_config.label,
    output_s3_uri=f"s3://my-bucket/bias-monitor/baseline-metrics",
    wait=True
)

# Schedule the monitoring job
monitoring_schedule_name = f"bias-drift-monitor-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

model_bias_monitor.create_monitoring_schedule(
    endpoint_input="customer-churn-predictor",
    record_preprocessor_script=None,  # Use default preprocessor
    post_analytics_processor_script=None,  # Use default postprocessor
    output_s3_uri=f"s3://my-bucket/bias-monitor/results",
    statistics=model_bias_monitor.baseline_statistics(),
    constraints=model_bias_monitor.suggested_constraints(),
    schedule_cron_expression="cron(0 0 ? * MON *)",  # Run weekly on Mondays
    enable_cloudwatch_metrics=True,
    monitoring_schedule_name=monitoring_schedule_name,
    analysis_config=bias_analysis_config,
    ground_truth_input=f"s3://my-bucket/bias-monitor/ground-truth"
)

print(f"Bias drift monitoring schedule created: {monitoring_schedule_name}")

# List all monitoring schedules
schedules = sagemaker_session.sagemaker_client.list_monitoring_schedules(
    StatusEquals="Scheduled"
)

for schedule in schedules["MonitoringScheduleSummaries"]:
    print(f"Schedule: {schedule['MonitoringScheduleName']}, Status: {schedule['MonitoringScheduleStatus']}")
"""
        st.code(bias_monitor_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive bias drift visualization
        st.markdown('<div class="info-box">Interactive Example: Bias Drift Visualization</div>', unsafe_allow_html=True)
        
        # Select protected attribute
        protected_attr = st.selectbox(
            "Select protected attribute:",
            ["Gender", "Age Group", "Race", "Income Level"]
        )
        
        # Time period
        bias_period = st.slider("Time Period (Weeks)", 4, 26, 12)
        
        # Bias metric to display
        bias_metric = st.selectbox(
            "Select bias metric to visualize:",
            ["Disparate Impact Ratio", "Demographic Disparity", "Equal Opportunity Difference"]
        )
        
        if st.button("Generate Bias Drift Chart"):
            # Generate dates (weekly data points)
            dates = pd.date_range(end=pd.Timestamp.now(), freq='W', periods=bias_period)
            
            # Create synthetic data based on selected protected attribute
            if protected_attr == "Gender":
                # Slight worsening bias trend for gender
                base_value = 0.88  # Starting below parity (1.0)
                trend = -0.005  # Small negative trend
                noise = 0.03  # Noise level
                
                fair_range = [0.8, 1.2]  # 80%-120% is often considered the "fair" range
                threshold = 0.8
                ideal_value = 1.0
                
                # Protected groups to show
                groups = ["Female", "Male"]
                
            elif protected_attr == "Age Group":
                # More severe bias trend for age
                base_value = 0.92
                trend = -0.015
                noise = 0.04
                
                fair_range = [0.8, 1.2]
                threshold = 0.8
                ideal_value = 1.0
                
                # Protected groups to show
                groups = ["18-25", "26-40", "41-60", "60+"]
                
            elif protected_attr == "Race":
                # Fluctuating bias trend
                base_value = 0.85
                trend = -0.008
                noise = 0.05
                
                fair_range = [0.8, 1.2]
                threshold = 0.8
                ideal_value = 1.0
                
                # Protected groups to show
                groups = ["Group A", "Group B", "Group C", "Group D"]
                
            else:  # Income Level
                # Initially fair, becoming biased
                base_value = 1.05
                trend = -0.02
                noise = 0.03
                
                fair_range = [0.8, 1.2]
                threshold = 0.8
                ideal_value = 1.0
                
                # Protected groups to show
                groups = ["Low Income", "Middle Income", "High Income"]
            
            # Generate values for primary protected group
            primary_values = []
            for i in range(bias_period):
                value = base_value + (trend * i) + np.random.normal(0, noise)
                primary_values.append(value)
            
            # Generate values for secondary groups with different patterns
            secondary_values = {}
            for group in groups[1:]:
                # Create different bias patterns for other groups
                group_base = base_value + np.random.uniform(-0.1, 0.1)
                group_trend = trend * np.random.uniform(0.5, 1.5)
                group_noise = noise * np.random.uniform(0.8, 1.2)
                
                values = []
                for i in range(bias_period):
                    value = group_base + (group_trend * i) + np.random.normal(0, group_noise)
                    values.append(value)
                secondary_values[group] = values
            
            # Create dataframe for the primary group
            bias_df = pd.DataFrame({
                'Date': dates,
                'Value': primary_values,
                'Group': groups[0]
            })
            
            # Add secondary groups
            for group, values in secondary_values.items():
                temp_df = pd.DataFrame({
                    'Date': dates,
                    'Value': values,
                    'Group': group
                })
                bias_df = pd.concat([bias_df, temp_df], ignore_index=True)
            
            # Create line chart
            fig = px.line(
                bias_df,
                x='Date',
                y='Value',
                color='Group',
                title=f'{bias_metric} Over Time for {protected_attr}',
                markers=True
            )
            
            # Add fair range and threshold
            fig.add_shape(
                type="rect",
                x0=dates.min(),
                x1=dates.max(),
                y0=fair_range[0],
                y1=fair_range[1],
                fillcolor="lightgreen",
                opacity=0.2,
                line_width=0,
                layer="below"
            )
            
            fig.add_shape(
                type="line",
                x0=dates.min(),
                x1=dates.max(),
                y0=threshold,
                y1=threshold,
                line=dict(color="red", dash="dash"),
            )
            
            fig.add_shape(
                type="line",
                x0=dates.min(),
                x1=dates.max(),
                y0=ideal_value,
                y1=ideal_value,
                line=dict(color="green", dash="dash"),
            )
            
            # Add annotations
            fig.add_annotation(
                x=dates.min(),
                y=threshold,
                text="Threshold",
                showarrow=False,
                yshift=-15,
                xshift=30
            )
            
            fig.add_annotation(
                x=dates.min(),
                y=ideal_value,
                text="Parity",
                showarrow=False,
                yshift=10,
                xshift=30
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Check if bias is detected
            primary_group_recent = bias_df[(bias_df['Group'] == groups[0]) & (bias_df['Date'] >= dates[-4])]['Value'].values
            bias_detected = any(v < threshold for v in primary_group_recent)
            
            if bias_detected:
                st.markdown(f"""
                <div style="background-color:#F5B7B1; padding:15px; border-radius:5px; margin-top:15px;">
                    <h4>🚨 Bias Drift Detected!</h4>
                    <p>The {bias_metric} for {protected_attr} ({groups[0]}) has fallen below the threshold of {threshold}.</p>
                    <p>This indicates potential unfairness in your model's predictions.</p>
                    <p>Recommended Actions:</p>
                    <ul>
                        <li>Investigate feature correlations with the protected attribute</li>
                        <li>Check for representation issues in your training data</li>
                        <li>Consider retraining with bias mitigation techniques</li>
                        <li>Review the decision threshold for different groups</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin-top:15px;">
                    <h4>✅ No Significant Bias Drift</h4>
                    <p>The {bias_metric} for {protected_attr} remains within acceptable limits.</p>
                    <p>Continue monitoring for changes in model fairness.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Bias metrics explanation
            st.markdown("### Understanding Bias Metrics")
            
            if bias_metric == "Disparate Impact Ratio":
                st.markdown("""
                **Disparate Impact Ratio** measures the ratio of positive outcome rates between the protected group and the reference group. 
                
                - Formula: `Positive Rate (protected group) / Positive Rate (reference group)`
                - Ideal Value: 1.0 (equal rates)
                - Acceptable Range: 0.8 to 1.25 (the "80% rule" from employment law)
                - Interpretation: Values below 0.8 suggest the protected group is disadvantaged
                
                Example: If females have a loan approval rate of 60% and males have a rate of 80%, the ratio is 0.75, indicating potential bias.
                """)
            elif bias_metric == "Demographic Disparity":
                st.markdown("""
                **Demographic Disparity** measures the absolute difference in positive outcome rates between groups.
                
                - Formula: `|Positive Rate (protected group) - Positive Rate (reference group)|`
                - Ideal Value: 0 (no difference)
                - Acceptable Range: Typically < 0.1 or 10% (depends on context)
                - Interpretation: Larger values indicate greater disparity in outcomes
                
                Example: If 75% of one age group receives a positive outcome but only 60% of another age group does, the disparity is 0.15 or 15%.
                """)
            else:  # Equal Opportunity Difference
                st.markdown("""
                **Equal Opportunity Difference** measures the difference in true positive rates between groups (ability to correctly identify positive cases).
                
                - Formula: `TPR (protected group) - TPR (reference group)`
                - Ideal Value: 0 (equal true positive rates)
                - Acceptable Range: Typically within ±0.1 (depends on context)
                - Interpretation: Negative values indicate the protected group has fewer true positives identified
                
                Example: If the model correctly identifies 80% of qualified applicants from one group but only 65% from another group, the difference is -0.15.
                """)
            
    with tab4:
        st.markdown('<div class="sub-header">Model Explainability Monitoring</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Model explainability monitoring tracks how your model's feature importance and explanations change over time:
        
        - Monitor changes in global feature importance
        - Track whether the model relies on different features than before
        - Detect shifts in feature attribution patterns
        - Ensure model decisions remain explainable and consistent
        """)
        
        # Explainability monitoring example
        st.markdown('<div class="info-box">Example: Setting up Explainability Monitoring</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="code-box">', unsafe_allow_html=True)
        explain_monitor_code = """
import boto3
import sagemaker
from sagemaker.model_monitor import ModelExplainabilityMonitor
from sagemaker.clarify import ExplainabilityConfig, DataConfig, SHAPConfig
from datetime import datetime

# Initialize session
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create a ModelExplainabilityMonitor
explainability_monitor = ModelExplainabilityMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)

# Define data config
data_config = DataConfig(
    s3_data_input_path="s3://my-bucket/explainability-monitor/baseline/",
    s3_output_path="s3://my-bucket/explainability-monitor/output/",
    label="churn",
    headers=["age", "gender", "income", "balance", "transactions", "credit_score", "churn"],
    dataset_type="text/csv"
)

# Define SHAP config for explainability
shap_config = SHAPConfig(
    baseline=[
        [35, 0, 50000, 5000, 20, 700, 0]  # Example baseline row
    ],
    num_samples=100,
    agg_method="mean_abs"  # Use absolute mean for feature importance
)

# Create explainability config
explainability_config = ExplainabilityConfig(
    shap_config=shap_config
)

# Create baseline for explainability metrics
explainability_monitor.suggest_baseline(
    data_config=data_config,
    explainability_config=explainability_config,
    model_config=None,  # Will be provided during monitoring
    output_s3_uri=f"s3://my-bucket/explainability-monitor/baseline-metrics",
    wait=True
)

# Schedule the monitoring job
monitoring_schedule_name = f"explainability-monitor-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

explainability_monitor.create_monitoring_schedule(
    endpoint_input="customer-churn-predictor",
    record_preprocessor_script=None,  # Use default preprocessor
    post_analytics_processor_script=None,  # Use default postprocessor
    output_s3_uri=f"s3://my-bucket/explainability-monitor/results",
    statistics=explainability_monitor.baseline_statistics(),
    constraints=explainability_monitor.suggested_constraints(),
    schedule_cron_expression="cron(0 0 ? * SUN *)",  # Run weekly on Sundays
    enable_cloudwatch_metrics=True,
    monitoring_schedule_name=monitoring_schedule_name,
    explainability_config=explainability_config
)

print(f"Explainability monitoring schedule created: {monitoring_schedule_name}")

# List all monitoring schedules
schedules = sagemaker_session.sagemaker_client.list_monitoring_schedules(
    StatusEquals="Scheduled"
)

for schedule in schedules["MonitoringScheduleSummaries"]:
    print(f"Schedule: {schedule['MonitoringScheduleName']}, Status: {schedule['MonitoringScheduleStatus']}")
"""
        st.code(explain_monitor_code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive feature importance drift visualization
        st.markdown('<div class="info-box">Interactive Example: Feature Importance Drift</div>', unsafe_allow_html=True)
        
        # Model selection
        model_type = st.selectbox(
            "Select model type:",
            ["Customer Churn Prediction", "Loan Approval", "Fraud Detection"]
        )
        
        # Time points to compare
        compare_points = st.slider("Time Points to Compare", 2, 4, 3)
        
        if st.button("Generate Feature Importance Comparison"):
            # Define features based on model type
            if model_type == "Customer Churn Prediction":
                features = [
                    "Tenure", "Monthly Charges", "Total Charges", "Contract Length",
                    "Payment Method", "Internet Service", "Tech Support", "Online Security"
                ]
                color_scale = "Blues"
                
            elif model_type == "Loan Approval":
                features = [
                    "Credit Score", "Income", "Loan Amount", "Loan Term", 
                    "Employment Years", "Debt-to-Income", "Prior Defaults", "Property Value"
                ]
                color_scale = "Greens"
                
            else:  # Fraud Detection
                features = [
                    "Transaction Amount", "Time Since Last Transaction", "Merchant Category",
                    "Distance from Home", "Card Present", "International", "IP Risk Score", "Device Match"
                ]
                color_scale = "Reds"
            
            # Generate time points
            time_points = ["Baseline"]
            for i in range(1, compare_points):
                weeks_ago = (compare_points - i) * 4
                time_points.append(f"{weeks_ago} Weeks Ago")
            time_points.append("Current")
            
            # Generate synthetic feature importances with drift
            importances = {}
            
            # Baseline/initial importances (relatively balanced)
            base_importances = np.random.uniform(0.05, 0.2, len(features))
            base_importances = base_importances / base_importances.sum()  # Normalize
            
            importances[time_points[0]] = base_importances.copy()
            
            # Generate drifted importances
            current_importances = base_importances.copy()
            
            # Gradually shift feature importance (some features become more important, others less)
            for i in range(1, len(time_points)):
                drift_factor = i / len(time_points)
                
                # Increase importance of some features, decrease others
                shift = np.zeros_like(current_importances)
                shift[0] = 0.1 * drift_factor  # First feature increases
                shift[1] = 0.08 * drift_factor  # Second feature increases
                shift[-1] = -0.08 * drift_factor  # Last feature decreases
                shift[-2] = -0.08 * drift_factor  # Second-to-last feature decreases
                
                # Add some random noise
                shift += np.random.normal(0, 0.02, len(features))
                
                # Apply shift and ensure no negative values
                current_importances += shift
                current_importances = np.maximum(current_importances, 0.01)
                
                # Re-normalize
                current_importances = current_importances / current_importances.sum()
                
                importances[time_points[i]] = current_importances.copy()
            
            # Create DataFrame
            rows = []
            for time_point, imp in importances.items():
                for i, feature in enumerate(features):
                    rows.append({
                        'Time': time_point,
                        'Feature': feature,
                        'Importance': imp[i]
                    })
            
            importance_df = pd.DataFrame(rows)
            
            # Create heatmap
            fig = px.density_heatmap(
                importance_df,
                x='Time',
                y='Feature',
                z='Importance',
                title=f'Feature Importance Drift for {model_type}',
                color_continuous_scale=color_scale
            )
            
            fig.update_layout(
                yaxis=dict(categoryorder='array', categoryarray=features)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create line chart for top features
            top_features = features[:4]  # Show top 4 features
            
            top_df = importance_df[importance_df['Feature'].isin(top_features)]
            
            fig = px.line(
                top_df,
                x='Time',
                y='Importance',
                color='Feature',
                title=f'Top Feature Importance Trends',
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and visualize feature rank changes
            st.markdown("### Feature Importance Rank Changes")
            
            # For each time point, get feature ranks
            ranks = {}
            for time_point in time_points:
                time_data = importance_df[importance_df['Time'] == time_point]
                ranked = time_data.sort_values('Importance', ascending=False)
                ranks[time_point] = {feature: i+1 for i, feature in enumerate(ranked['Feature'])}
            
            # Compare baseline to current
            baseline_ranks = ranks[time_points[0]]
            current_ranks = ranks[time_points[-1]]
            
            rank_changes = []
            for feature in features:
                baseline = baseline_ranks[feature]
                current = current_ranks[feature]
                change = baseline - current  # Positive means improved rank (moved up)
                
                rank_changes.append({
                    'Feature': feature,
                    'Baseline Rank': baseline,
                    'Current Rank': current,
                    'Rank Change': change
                })
            
            rank_df = pd.DataFrame(rank_changes).sort_values('Rank Change', ascending=False)
            
            # Bar chart of rank changes
            fig = px.bar(
                rank_df,
                x='Feature',
                y='Rank Change',
                color='Rank Change',
                title='Feature Importance Rank Changes (Baseline → Current)',
                color_continuous_scale=['red', 'lightgray', 'green']
            )
            
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(features) - 0.5,
                y0=0,
                y1=0,
                line=dict(color="black", dash="dash"),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Check for significant shifts
            max_abs_change = max(abs(row['Rank Change']) for row in rank_changes)
            
            if max_abs_change > 2:
                st.markdown(f"""
                <div style="background-color:#F5B7B1; padding:15px; border-radius:5px; margin-top:15px;">
                    <h4>⚠️ Significant Feature Importance Shifts Detected!</h4>
                    <p>The model's decision-making process has changed significantly over time.</p>
                    <p>Some features have changed in importance ranking by up to {max_abs_change} positions.</p>
                    <p>Recommended Actions:</p>
                    <ul>
                        <li>Investigate changes in the data distribution</li>
                        <li>Check for changes in feature correlations</li>
                        <li>Review model retraining process if applicable</li>
                        <li>Consider re-evaluating model fairness and bias</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin-top:15px;">
                    <h4>✅ Feature Importance Relatively Stable</h4>
                    <p>No major shifts in how the model makes decisions have been detected.</p>
                    <p>The ranking of features has remained relatively consistent.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Individual prediction explanations
            st.markdown("### Individual Prediction Explanation Comparison")
            
            # Select a sample case to show explanation drift for
            sample_id = "Sample-12345"
            
            # Create sample prediction scenarios for the first and last time points
            baseline_time = time_points[0]
            current_time = time_points[-1]
            
            # Generate synthetic SHAP values
            baseline_explanation = {
                features[i]: importances[baseline_time][i] * np.random.normal(1, 0.3) 
                for i in range(len(features))
            }
            
            current_explanation = {
                features[i]: importances[current_time][i] * np.random.normal(1, 0.3) 
                for i in range(len(features))
            }
            
            # Sort explanations by absolute value
            baseline_explanation = {k: v for k, v in sorted(
                baseline_explanation.items(), 
                key=lambda item: abs(item[1]), 
                reverse=True
            )}
            
            current_explanation = {k: v for k, v in sorted(
                current_explanation.items(), 
                key=lambda item: abs(item[1]), 
                reverse=True
            )}
            
            # Create waterfall charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {baseline_time} Explanation")
                
                baseline_df = pd.DataFrame({
                    'Feature': list(baseline_explanation.keys()),
                    'SHAP Value': list(baseline_explanation.values())
                })
                
                fig = go.Figure(go.Waterfall(
                    name="SHAP",
                    orientation="h",
                    measure=["relative"] * len(baseline_explanation),
                    x=baseline_df['SHAP Value'],
                    y=baseline_df['Feature'],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": AWS_COLORS["green"]}},
                    decreasing={"marker": {"color": AWS_COLORS["red"]}},
                    text=[f"{x:.3f}" for x in baseline_df['SHAP Value']],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title=f"Prediction Explanation for {sample_id}",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"#### {current_time} Explanation")
                
                current_df = pd.DataFrame({
                    'Feature': list(current_explanation.keys()),
                    'SHAP Value': list(current_explanation.values())
                })
                
                fig = go.Figure(go.Waterfall(
                    name="SHAP",
                    orientation="h",
                    measure=["relative"] * len(current_explanation),
                    x=current_df['SHAP Value'],
                    y=current_df['Feature'],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": AWS_COLORS["green"]}},
                    decreasing={"marker": {"color": AWS_COLORS["red"]}},
                    text=[f"{x:.3f}" for x in current_df['SHAP Value']],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title=f"Prediction Explanation for {sample_id}",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Function to render about page
def about_page():
    st.markdown('<div class="main-header">About This Application</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This interactive learning application demonstrates MLOps-ready features of Amazon SageMaker, 
        allowing you to explore the complete machine learning lifecycle from data preparation to monitoring.
        
        ### What is MLOps?
        
        MLOps (Machine Learning Operations) combines Machine Learning, DevOps, and Data Engineering to streamline
        the deployment, monitoring, and management of machine learning models in production environments.
        
        ### What This Application Covers
        
        - **Data Preparation**: Data Wrangler, Processing Jobs, Feature Store
        - **Model Building**: Training Jobs, Built-in Algorithms, Experiments
        - **Model Evaluation**: Performance Metrics, Explainability, Bias Detection
        - **Model Selection**: Model Comparison, Model Registry, Approval Workflow
        - **Deployment**: Real-time Endpoints, Batch Transform, Advanced Options
        - **Monitoring**: Data Quality, Model Quality, Bias Drift, Explainability
        """)
    
    # with col2:
        # lottie_about = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_l4ny0jb8.json")
        # st_lottie(lottie_about, height=300, key="about_animation")
    
    st.markdown('<div class="sub-header">Learning Resources</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
        - [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)
        - [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎓 Tutorials & Workshops")
        st.markdown("""
        - [SageMaker Workshop](https://sagemaker-workshop.com/)
        - [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
        - [SageMaker Studio Lab](https://studiolab.sagemaker.aws/)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Tools & Services")
        st.markdown("""
        - [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/)
        - [SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines/)
        - [SageMaker Projects](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects.html)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
   
# Main application logic
def main():
    # Render the sidebar

    render_sidebar()

    st.markdown('<div class="main-header">SageMaker MLOps</div>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to the interactive learning platform for Amazon SageMaker MLOps! This application will guide you through the entire MLOps workflow using Amazon SageMaker.
    
    Explore each stage of the MLOps lifecycle with interactive examples, visualizations, and code samples.
    """)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏠 Home",
        "📊 Data Preparation",
        "🛠️ Model Build",
        "📈 Model Evaluation",
        "🔍 Model Selection",
        "🚀 Deployment",
        "📡 Monitoring",
        "ℹ️ About"
    ])

    
    
    # Render the appropriate page based on current_page
    with tab1:
        home_page()
    with tab2:
        data_preparation_page()
        # data_wrangler_page()
    with tab3:
        model_build_page()
        # model_build_page()
    with tab4:
        model_evaluation_page()
        # model_evaluation_page()
    with tab5:
        model_selection_page()
        # model_selection_page()
    with tab6:
        deployment_page()
        # deployment_page()
    with tab7:
        monitoring_page()
        # monitoring_page()
    with tab8:
        about_page()

    # Add footer
    st.markdown('<div class="footer">© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
