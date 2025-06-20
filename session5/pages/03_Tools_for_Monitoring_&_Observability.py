
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="AWS ML Monitoring Tools",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'knowledge_check_responses' not in st.session_state:
        st.session_state.knowledge_check_responses = {}
    
    if 'knowledge_check_completed' not in st.session_state:
        st.session_state.knowledge_check_completed = False
    
    if 'knowledge_check_submitted' not in st.session_state:
        st.session_state.knowledge_check_submitted = False
    
    if 'correct_answers' not in st.session_state:
        st.session_state.correct_answers = 0

    if 'quiz_attempted' not in st.session_state:
        st.session_state['quiz_attempted'] = False
    if 'quiz_score' not in st.session_state:
        st.session_state['quiz_score'] = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state['quiz_answers'] = []


# Reset session state
def reset_session():
    st.session_state.knowledge_check_responses = {}
    st.session_state.knowledge_check_completed = False
    st.session_state.knowledge_check_submitted = False
    st.session_state.correct_answers = 0
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

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

# Function to create CloudWatch metric visualization
def create_cloudwatch_metric_chart():
    # Sample data for CloudWatch metric visualization
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    cpu_values = [40, 45, 60, 75, 90, 85, 70, 55, 50, 45]
    memory_values = [60, 65, 70, 75, 85, 80, 70, 65, 60, 55]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=cpu_values, mode='lines+markers', name='CPU Utilization (%)', line=dict(color='#FF9900')))
    fig.add_trace(go.Scatter(x=dates, y=memory_values, mode='lines+markers', name='Memory Utilization (%)', line=dict(color='#232F3E')))
    
    fig.update_layout(
        title="Model Endpoint Performance Metrics",
        xaxis_title="Date",
        yaxis_title="Utilization (%)",
        legend_title="Metrics",
        height=400
    )
    
    return fig

# Function to create X-Ray service map
def create_xray_service_map():
    # Create a diagram representing X-Ray service map
    nodes = [
        {"id": "client", "label": "Client", "x": 0, "y": 0.5},
        {"id": "api_gateway", "label": "API Gateway", "x": 0.2, "y": 0.5},
        {"id": "lambda", "label": "Lambda Function", "x": 0.4, "y": 0.5},
        {"id": "sagemaker", "label": "SageMaker Endpoint", "x": 0.6, "y": 0.5},
        {"id": "dynamodb", "label": "DynamoDB", "x": 0.8, "y": 0.75},
        {"id": "s3", "label": "S3 Bucket", "x": 0.8, "y": 0.25},
    ]
    
    edges = [
        {"source": "client", "target": "api_gateway", "value": 95},
        {"source": "api_gateway", "target": "lambda", "value": 90},
        {"source": "lambda", "target": "sagemaker", "value": 85},
        {"source": "sagemaker", "target": "dynamodb", "value": 40},
        {"source": "sagemaker", "target": "s3", "value": 60},
    ]
    
    # Create a network diagram using plotly
    edge_x = []
    edge_y = []
    for edge in edges:
        source_node = next(node for node in nodes if node["id"] == edge["source"])
        target_node = next(node for node in nodes if node["id"] == edge["target"])
        edge_x.extend([source_node["x"], target_node["x"], None])
        edge_y.extend([source_node["y"], target_node["y"], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [node["x"] for node in nodes]
    node_y = [node["y"] for node in nodes]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node["label"] for node in nodes],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=30,
            colorbar=dict(
                thickness=15,
                title='Request Volume',
                xanchor='left'
                # titleside='right'
            ),
            color=[95, 90, 85, 80, 40, 60],
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title="AWS X-Ray Service Map",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=400
                    ))
    
    return fig

# Function to create CloudTrail logs visualization
def create_cloudtrail_logs():
    # Sample data for CloudTrail logs
    data = {
        'Service': ['SageMaker', 'IAM', 'S3', 'SageMaker', 'CloudWatch', 'Lambda', 'SageMaker', 'EC2'],
        'Action': ['CreateEndpoint', 'AssumeRole', 'GetObject', 'InvokeEndpoint', 'PutMetricData', 'Invoke', 'DeleteEndpoint', 'StartInstance'],
        'User': ['admin', 'system', 'ml-user', 'api-user', 'system', 'api-user', 'admin', 'devops'],
        'Timestamp': ['2023-01-01 10:15', '2023-01-01 10:16', '2023-01-01 10:20', '2023-01-01 10:25', 
                     '2023-01-01 10:30', '2023-01-01 10:35', '2023-01-01 10:40', '2023-01-01 10:45'],
        'Status': ['Success', 'Success', 'Success', 'Failure', 'Success', 'Success', 'Success', 'Success']
    }
    
    df = pd.DataFrame(data)
    
    # Create a bar chart showing actions by service
    service_counts = df['Service'].value_counts().reset_index()
    service_counts.columns = ['Service', 'Count']
    
    fig = px.bar(service_counts, x='Service', y='Count', 
                 title='API Actions by Service',
                 color='Service', 
                 color_discrete_sequence=px.colors.qualitative.Bold)
    
    fig.update_layout(height=400)
    
    return fig, df

# Function to create EventBridge visualization
def create_eventbridge_visualization():
    # Create a simple diagram showing EventBridge routing
    sources = ["SageMaker Model", "CloudWatch Alarm", "Scheduled Event"]
    targets = ["Lambda Function", "SNS Topic", "Step Functions"]
    connections = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)]
    
    fig = go.Figure()
    
    # Create EventBridge in center
    fig.add_shape(
        type="rect",
        x0=0.4, y0=0.4,
        x1=0.6, y1=0.6,
        line=dict(color="#FF9900", width=2),
        fillcolor="#FF9900",
    )
    
    # Add EventBridge text
    fig.add_annotation(
        x=0.5, y=0.5,
        text="EventBridge",
        showarrow=False,
        font=dict(color="white", size=12)
    )
    
    # Add sources on left
    for i, source in enumerate(sources):
        y_pos = 0.2 + i * 0.3
        
        # Source box
        fig.add_shape(
            type="rect",
            x0=0.1, y0=y_pos-0.05,
            x1=0.3, y1=y_pos+0.05,
            line=dict(color="#232F3E", width=2),
            fillcolor="#232F3E",
        )
        
        # Source text
        fig.add_annotation(
            x=0.2, y=y_pos,
            text=source,
            showarrow=False,
            font=dict(color="white", size=10)
        )
        
        # Arrow to EventBridge
        fig.add_shape(
            type="line",
            x0=0.3, y0=y_pos,
            x1=0.4, y1=0.5,
            line=dict(color="#232F3E", width=1, dash="dot"),
        )
    
    # Add targets on right
    for i, target in enumerate(targets):
        y_pos = 0.2 + i * 0.3
        
        # Target box
        fig.add_shape(
            type="rect",
            x0=0.7, y0=y_pos-0.05,
            x1=0.9, y1=y_pos+0.05,
            line=dict(color="#1E8900", width=2),
            fillcolor="#1E8900",
        )
        
        # Target text
        fig.add_annotation(
            x=0.8, y=y_pos,
            text=target,
            showarrow=False,
            font=dict(color="white", size=10)
        )
    
    # Add connections based on the connections list
    for source_idx, target_idx in connections:
        source_y = 0.2 + source_idx * 0.3
        target_y = 0.2 + target_idx * 0.3
        
        fig.add_shape(
            type="line",
            x0=0.6, y0=0.5,
            x1=0.7, y1=target_y,
            line=dict(color="#1E8900", width=1, dash="dot"),
        )
    
    fig.update_layout(
        title="Amazon EventBridge Event Routing",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1])
    )
    
    return fig

# Function to create QuickSight dashboard mockup
def create_quicksight_mockup():
    # Create a mockup of a QuickSight dashboard with multiple visualizations
    fig = go.Figure()
    
    # Add background
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=1, y1=1,
        line=dict(color="#F8F8F8"),
        fillcolor="#F8F8F8",
    )
    
    # Add header
    fig.add_shape(
        type="rect",
        x0=0, y0=0.9,
        x1=1, y1=1,
        line=dict(color="#232F3E"),
        fillcolor="#232F3E",
    )
    
    fig.add_annotation(
        x=0.5, y=0.95,
        text="ML Model Performance Dashboard",
        showarrow=False,
        font=dict(color="white", size=16)
    )
    
    # Add visualizations
    
    # KPI boxes
    kpi_positions = [(0.15, 0.8), (0.5, 0.8), (0.85, 0.8)]
    kpi_titles = ["Avg. Inference Time", "Model Accuracy", "Daily Predictions"]
    kpi_values = ["125ms", "98.2%", "1.5M"]
    
    for (x, y), title, value in zip(kpi_positions, kpi_titles, kpi_values):
        # KPI box
        fig.add_shape(
            type="rect",
            x0=x-0.15, y0=y-0.07,
            x1=x+0.15, y1=y+0.07,
            line=dict(color="white"),
            fillcolor="white",
        )
        
        # KPI title
        fig.add_annotation(
            x=x, y=y+0.04,
            text=title,
            showarrow=False,
            font=dict(color="#232F3E", size=12)
        )
        
        # KPI value
        fig.add_annotation(
            x=x, y=y-0.02,
            text=value,
            showarrow=False,
            font=dict(color="#FF9900", size=16, weight="bold")
        )
    
    # Line chart
    fig.add_shape(
        type="rect",
        x0=0.05, y0=0.45,
        x1=0.45, y1=0.75,
        line=dict(color="white"),
        fillcolor="white",
    )
    
    # Add line chart title
    fig.add_annotation(
        x=0.25, y=0.72,
        text="Inference Latency Over Time",
        showarrow=False,
        font=dict(color="#232F3E", size=12)
    )
    
    # Add mock line chart
    x_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    y_values = [0.55, 0.6, 0.56, 0.65, 0.58, 0.53, 0.6]
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        line=dict(color="#FF9900"),
        showlegend=False
    ))
    
    # Bar chart
    fig.add_shape(
        type="rect",
        x0=0.55, y0=0.45,
        x1=0.95, y1=0.75,
        line=dict(color="white"),
        fillcolor="white",
    )
    
    # Add bar chart title
    fig.add_annotation(
        x=0.75, y=0.72,
        text="Predictions by Model Version",
        showarrow=False,
        font=dict(color="#232F3E", size=12)
    )
    
    # Add mock bar chart
    bar_x = [0.6, 0.7, 0.8, 0.9]
    bar_heights = [0.1, 0.15, 0.08, 0.12]
    
    for x, height in zip(bar_x, bar_heights):
        fig.add_shape(
            type="rect",
            x0=x-0.04, y0=0.5,
            x1=x+0.04, y1=0.5+height,
            line=dict(color="#232F3E"),
            fillcolor="#232F3E",
        )
    
    # Pie chart
    fig.add_shape(
        type="rect",
        x0=0.05, y0=0.1,
        x1=0.45, y1=0.4,
        line=dict(color="white"),
        fillcolor="white",
    )
    
    # Add pie chart title
    fig.add_annotation(
        x=0.25, y=0.37,
        text="Error Types Distribution",
        showarrow=False,
        font=dict(color="#232F3E", size=12)
    )
    
    # Add mock pie chart (just a circle with lines)
    fig.add_shape(
        type="circle",
        x0=0.15, y0=0.15,
        x1=0.35, y1=0.35,
        line=dict(color="#232F3E"),
        fillcolor="#FF9900",
    )
    
    # Table
    fig.add_shape(
        type="rect",
        x0=0.55, y0=0.1,
        x1=0.95, y1=0.4,
        line=dict(color="white"),
        fillcolor="white",
    )
    
    # Add table title
    fig.add_annotation(
        x=0.75, y=0.37,
        text="Recent Drift Detections",
        showarrow=False,
        font=dict(color="#232F3E", size=12)
    )
    
    # Add mock table header
    fig.add_shape(
        type="rect",
        x0=0.58, y0=0.3,
        x1=0.92, y1=0.34,
        line=dict(color="#232F3E"),
        fillcolor="#232F3E",
    )
    
    # Add mock table rows
    y_positions = [0.26, 0.22, 0.18, 0.14]
    for y in y_positions:
        fig.add_shape(
            type="rect",
            x0=0.58, y0=y,
            x1=0.92, y1=y+0.04,
            line=dict(color="#F8F8F8"),
            fillcolor="#F8F8F8",
        )
    
    fig.update_layout(
        showlegend=False,
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1])
    )
    
    return fig

# Function to render knowledge check
def render_knowledge_check():
    # Apply the custom CSS styling
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
    
    # Display main header
    custom_header("Test Your Knowledge")
    
    st.markdown("""
    This quiz will test your understanding of the key concepts covered in AWS Monitoring and Observability Tools.
    Answer the following questions to evaluate your knowledge of monitoring ML systems in AWS.
    """)
    
    # Define quiz questions
    questions = [
        {
            "question": "Which AWS service provides end-to-end tracing for distributed applications?",
            "options": ["AWS CloudWatch", "AWS X-Ray", "AWS CloudTrail", "AWS EventBridge"],
            "correct": "AWS X-Ray",
            "explanation": "AWS X-Ray provides end-to-end tracing for distributed applications, enabling you to analyze and debug production applications."
        },
        {
            "question": "Which tool would you use to monitor for model drift in SageMaker?",
            "options": ["SageMaker Model Monitor", "CloudWatch Events", "AWS X-Ray", "SageMaker Neo"],
            "correct": "SageMaker Model Monitor",
            "explanation": "SageMaker Model Monitor allows you to detect drift in model quality, data quality, bias, and feature attribution."
        },
        {
            "question": "Which service would you use to audit API calls made to SageMaker?",
            "options": ["AWS CloudWatch", "AWS CloudTrail", "AWS Config", "Amazon Inspector"],
            "correct": "AWS CloudTrail",
            "explanation": "AWS CloudTrail logs, monitors, and retains account activity related to actions across your AWS infrastructure, including API calls to SageMaker."
        },
        {
            "question": "Which strategy allows you to compare a new model against your current production model?",
            "options": ["A/B Testing", "Canary Deployment", "Blue/Green Deployment", "Shadow Mode"],
            "correct": "A/B Testing",
            "explanation": "A/B testing allows you to test different variants of your models by distributing traffic between them and comparing their performance."
        },
        {
            "question": "Which type of drift occurs when the contribution of individual features to model predictions differs from what was observed during training?",
            "options": ["Data Quality Drift", "Model Quality Drift", "Bias Drift", "Feature Attribution Drift"],
            "correct": "Feature Attribution Drift",
            "explanation": "Feature Attribution Drift occurs when the contribution of individual features to model predictions differs from what was observed during training. This can happen when the relationship between features shifts or the model begins using different patterns to make predictions."
        }
    ]
    
    # Check if the quiz has been attempted
    if not st.session_state['quiz_attempted']:
        # Create a form for the quiz
        with st.form("quiz_form"):
            st.markdown("### Answer the following questions:")
            
            # Track user answers
            user_answers = []
            
            # Display 5 questions
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
            st.success("🎉 Perfect score! You've mastered the concepts of AWS Monitoring and Observability Tools!")
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

# Function to render CloudWatch tab content
def render_cloudwatch_tab():
    st.markdown("## Amazon CloudWatch")
    
    # CloudWatch description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Overview
        Amazon CloudWatch is a monitoring and observability service that provides data and actionable insights for AWS, hybrid, and on-premises applications and infrastructure resources. With CloudWatch, you can:
        
        * Collect and track metrics
        * Collect and monitor log files
        * Set alarms
        * Automatically react to changes
        * Gain system-wide visibility into resource utilization, application performance, and operational health
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/products/CloudWatch/product-page-diagram-CloudWatch_v4.1f8e6d166d0c53a44bfcb8b827dde4e00.jpg", caption="CloudWatch Overview")
    
    # CloudWatch key features
    st.markdown("""
    ### Key Features for ML Engineers
    
    * **Metrics for SageMaker endpoints** - Track invocation count, latency, errors, and resource utilization
    * **Log analysis** - Collect logs from SageMaker training jobs, processing jobs, and endpoints
    * **Anomaly detection** - Automatically detect unusual behavior in your metrics
    * **Alarms and notifications** - Get notified when metrics exceed thresholds
    * **Dashboards** - Create custom dashboards to visualize key metrics for ML workloads
    """)
    
    # Sample CloudWatch metrics for ML endpoints
    st.markdown("### Sample CloudWatch Metrics for SageMaker Endpoints")
    st.plotly_chart(create_cloudwatch_metric_chart())
    
    # CloudWatch code example
    st.markdown("### Code Example: Monitoring a SageMaker Endpoint with CloudWatch")
    
    code = '''
    import boto3
    import datetime
    
    # Create CloudWatch client
    cloudwatch = boto3.client('cloudwatch')
    
    # Get metrics for a SageMaker endpoint
    response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'invocations',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/SageMaker',
                        'MetricName': 'Invocations',
                        'Dimensions': [
                            {
                                'Name': 'EndpointName',
                                'Value': 'my-ml-endpoint'
                            },
                        ]
                    },
                    'Period': 300,
                    'Stat': 'Sum',
                }
            },
            {
                'Id': 'latency',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/SageMaker',
                        'MetricName': 'OverheadLatency',
                        'Dimensions': [
                            {
                                'Name': 'EndpointName',
                                'Value': 'my-ml-endpoint'
                            },
                        ]
                    },
                    'Period': 300,
                    'Stat': 'Average',
                }
            }
        ],
        StartTime=datetime.datetime.utcnow() - datetime.timedelta(hours=3),
        EndTime=datetime.datetime.utcnow(),
    )
    
    # Create an alarm for high latency
    cloudwatch.put_metric_alarm(
        AlarmName='ML-Endpoint-HighLatency',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=2,
        MetricName='OverheadLatency',
        Namespace='AWS/SageMaker',
        Period=300,
        Statistic='Average',
        Threshold=100.0,
        ActionsEnabled=True,
        AlarmDescription='Alarm when endpoint latency exceeds 100ms',
        AlarmActions=[
            'arn:aws:sns:region:account-id:ML-Alerts'
        ],
        Dimensions=[
            {
                'Name': 'EndpointName',
                'Value': 'my-ml-endpoint'
            },
        ]
    )
    '''
    
    st.code(code, language='python')
    
    # Best practices
    st.markdown("""
    ### Best Practices for ML Monitoring with CloudWatch
    
    1. **Monitor critical ML metrics** - Track invocations, latency, errors, and model accuracy
    2. **Set appropriate alarms** - Configure alarms with appropriate thresholds to detect issues early
    3. **Use anomaly detection** - Enable CloudWatch anomaly detection for ML metrics
    4. **Create custom metrics** - Send custom metrics from your application to track business KPIs
    5. **Retain logs appropriately** - Configure log retention periods based on your requirements
    6. **Create dashboards** - Build dashboards combining different metrics for a holistic view
    """)
    
    # Interactive demo
    st.markdown("### Interactive Demo: Creating CloudWatch Alarms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric = st.selectbox(
            "Select Metric",
            ["Invocations", "ModelLatency", "OverheadLatency", "CPUUtilization", "MemoryUtilization"]
        )
        
        threshold = st.slider(
            "Threshold Value",
            min_value=1,
            max_value=1000,
            value=100
        )
        
        comparison = st.selectbox(
            "Comparison Operator",
            ["GreaterThanThreshold", "GreaterThanOrEqualToThreshold", "LessThanThreshold", "LessThanOrEqualToThreshold"]
        )
    
    with col2:
        evaluation_periods = st.slider(
            "Evaluation Periods",
            min_value=1,
            max_value=10,
            value=3
        )
        
        period = st.select_slider(
            "Period (seconds)",
            options=[60, 300, 900, 1800, 3600]
        )
        
        action = st.selectbox(
            "Alarm Action",
            ["Send notification", "Auto-scale endpoint", "Invoke Lambda function", "No action"]
        )
    
    if st.button("Create Alarm (Demo)"):
        st.success(f"CloudWatch alarm created for metric {metric} with threshold {threshold}!")
        st.json({
            "AlarmName": f"ML-Endpoint-{metric}-Alert",
            "MetricName": metric,
            "Threshold": threshold,
            "ComparisonOperator": comparison,
            "EvaluationPeriods": evaluation_periods,
            "Period": period,
            "Action": action
        })

# Function to render X-Ray tab content
def render_xray_tab():
    st.markdown("## AWS X-Ray")
    
    # X-Ray description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Overview
        AWS X-Ray helps developers analyze and debug production, distributed applications, such as those built using a microservices architecture. With X-Ray, you can:
        
        * Understand how your application and its underlying services are performing
        * Identify and troubleshoot the root cause of performance issues and errors
        * Analyze requests as they travel through your application
        * Create a service map showing connections between services in your application
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/products/xray/product-page-diagram_aws-xray_how-it-works.70b62a58e646ae3c7b97292aa9243c385e43262d.png", caption="X-Ray Overview")
    
    # X-Ray key features
    st.markdown("""
    ### Key Features for ML Engineers
    
    * **End-to-end tracing** - Track requests as they flow through ML inference endpoints, APIs, and other services
    * **Service maps** - Automatically generate service maps showing the flow between components
    * **Performance analysis** - Identify bottlenecks in your ML inference pipeline
    * **Error identification** - Quickly identify where errors occur in your distributed ML application
    * **Integration with other AWS services** - Works with SageMaker, Lambda, API Gateway, and other services
    """)
    
    # X-Ray service map
    st.markdown("### Sample X-Ray Service Map for ML Inference Pipeline")
    st.plotly_chart(create_xray_service_map())
    
    # X-Ray code example
    st.markdown("### Code Example: Instrumenting a Python ML Application with X-Ray")
    
    code = '''
    from aws_xray_sdk.core import xray_recorder
    from aws_xray_sdk.core import patch_all
    import boto3
    import json
    
    # Configure X-Ray
    xray_recorder.configure(service='ml-inference-service')
    patch_all()
    
    # Initialize SageMaker runtime client
    sagemaker = boto3.client('sagemaker-runtime')
    
    # Function to make ML predictions
    @xray_recorder.capture('make_prediction')
    def make_prediction(input_data):
        # Add annotation for input data size
        xray_recorder.current_subsegment.put_annotation('input_size', len(str(input_data)))
        
        try:
            # Start a custom subsegment for preprocessing
            subsegment = xray_recorder.begin_subsegment('preprocess_data')
            # Your preprocessing code here
            processed_data = preprocess(input_data)
            xray_recorder.end_subsegment()
            
            # Call SageMaker endpoint
            with xray_recorder.capture('sagemaker_invoke_endpoint') as subsegment:
                response = sagemaker.invoke_endpoint(
                    EndpointName='my-ml-endpoint',
                    ContentType='application/json',
                    Body=json.dumps(processed_data)
                )
                prediction = json.loads(response['Body'].read().decode())
                subsegment.put_metadata('prediction_confidence', prediction['confidence'])
                
            return prediction
            
        except Exception as e:
            # Record error
            xray_recorder.current_subsegment.add_exception(e)
            raise
    
    # Lambda handler
    def lambda_handler(event, context):
        # X-Ray automatically traces Lambda invocations
        result = make_prediction(event)
        return result
    '''
    
    st.code(code, language='python')
    
    # X-Ray trace analysis
    st.markdown("### Understanding X-Ray Traces")
    
    trace_data = {
        "id": "1-5f89d3f0-6a5e993f55a1b5a0579ee798",
        "duration": 3.352,
        "segments": [
            {
                "id": "api-gateway",
                "name": "API Gateway",
                "start_time": 0,
                "end_time": 0.125,
                "status": "OK"
            },
            {
                "id": "lambda-function",
                "name": "ML-Inference-Lambda",
                "start_time": 0.135,
                "end_time": 0.625,
                "status": "OK",
                "subsegments": [
                    {
                        "name": "preprocess_data",
                        "start_time": 0.145,
                        "end_time": 0.245,
                        "status": "OK"
                    },
                    {
                        "name": "sagemaker_invoke_endpoint",
                        "start_time": 0.255,
                        "end_time": 0.610,
                        "status": "OK"
                    }
                ]
            },
            {
                "id": "sagemaker-endpoint",
                "name": "SageMaker-Endpoint",
                "start_time": 0.645,
                "end_time": 3.225,
                "status": "OK",
                "subsegments": [
                    {
                        "name": "model_loading",
                        "start_time": 0.655,
                        "end_time": 1.255,
                        "status": "OK"
                    },
                    {
                        "name": "inference",
                        "start_time": 1.265,
                        "end_time": 3.215,
                        "status": "OK"
                    }
                ]
            },
            {
                "id": "dynamodb",
                "name": "DynamoDB-Results-Table",
                "start_time": 3.235,
                "end_time": 3.345,
                "status": "OK"
            }
        ]
    }
    
    # Create a simple Gantt chart for the trace
    segments = trace_data["segments"]
    all_segments = []
    
    # Flatten segments and subsegments
    for segment in segments:
        all_segments.append({
            "Task": segment["name"],
            "Start": segment["start_time"],
            "End": segment["end_time"],
            "Type": "Main"
        })
        
        if "subsegments" in segment:
            for subsegment in segment["subsegments"]:
                all_segments.append({
                    "Task": f"  → {subsegment['name']}",
                    "Start": subsegment["start_time"],
                    "End": subsegment["end_time"],
                    "Type": "Sub"
                })
    
    df = pd.DataFrame(all_segments)
    df["Duration"] = df["End"] - df["Start"]
    
    # Create Gantt chart
    fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Type",
                     color_discrete_map={"Main": "#FF9900", "Sub": "#232F3E"},
                     title="X-Ray Trace Timeline Analysis")
    
    fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Service Component", height=400)
    
    st.plotly_chart(fig)
    
    # Best practices
    st.markdown("""
    ### Best Practices for X-Ray with ML Applications
    
    1. **Instrument critical components** - Focus on instrumenting services that are critical to your ML pipeline
    2. **Use sampling appropriately** - Configure sampling rates based on traffic volume
    3. **Add custom annotations and metadata** - Include ML-specific information like model versions and input sizes
    4. **Create custom subsegments** - Track performance of individual components within your ML workflow
    5. **Analyze trace data** - Regularly review trace data to identify performance bottlenecks
    6. **Integrate with CloudWatch** - Use X-Ray with CloudWatch for comprehensive monitoring
    """)
    
    # Interactive demo
    st.markdown("### Interactive Demo: X-Ray Trace Analysis")
    
    # Simulation options
    col1, col2 = st.columns(2)
    
    with col1:
        service = st.selectbox(
            "Service to analyze",
            ["API Gateway", "Lambda Function", "SageMaker Endpoint", "DynamoDB"]
        )
        
        time_range = st.select_slider(
            "Time range",
            options=["Last hour", "Last 3 hours", "Last 12 hours", "Last 24 hours", "Last 7 days"]
        )
    
    with col2:
        filter_option = st.selectbox(
            "Filter traces by",
            ["All traces", "Error traces", "Throttled traces", "Traces taking longer than 1s"]
        )
        
        group_by = st.selectbox(
            "Group by",
            ["Service", "User", "Request URL", "Response code"]
        )
    
    if st.button("Analyze Traces (Demo)"):
        st.success(f"Analysis completed for {service} over {time_range}!")
        
        if service == "SageMaker Endpoint" and filter_option == "Traces taking longer than 1s":
            st.warning("Found 8 traces where SageMaker inference took longer than 1 second!")
            st.markdown("""
            ### Identified Issues:
            1. Model loading time is significantly higher for cold starts
            2. Large input payloads (>1MB) causing increased preprocessing time
            3. GPU utilization dropping during batch inference
            """)
        else:
            st.info("No significant issues found in the trace data.")

# Function to render CloudTrail tab content
def render_cloudtrail_tab():
    st.markdown("## AWS CloudTrail")
    
    # CloudTrail description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Overview
        AWS CloudTrail is a service that enables governance, compliance, operational auditing, and risk auditing of your AWS account. With CloudTrail, you can:
        
        * Log, continuously monitor, and retain account activity related to actions across your AWS infrastructure
        * Identify which users and accounts called AWS APIs
        * Track when the API calls were made and from which source IP address
        * Get detailed information about the resources that were affected
        * Detect unusual API activity that might indicate security issues
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/products/cloudtrail/product-page-diagram_AWS-CloudTrail_how-it-works.d314964834a3ff50b50b29940469b2a1f7fca23d.png", caption="CloudTrail Overview")
    
    # CloudTrail key features
    st.markdown("""
    ### Key Features for ML Engineers
    
    * **API Activity Tracking** - Log all SageMaker API calls, including model creation, training, and deployment
    * **User Activity Monitoring** - Track who accessed or modified ML resources
    * **Security Analysis** - Identify suspicious activities like unauthorized model access
    * **Compliance** - Help meet regulatory requirements with immutable API activity history
    * **Event History** - View, search, and download the past 90 days of activity in your AWS account
    * **Integration with EventBridge** - Create automated workflows for certain events
    """)
    
    # CloudTrail logs visualization
    st.markdown("### Sample CloudTrail Logs for ML Operations")
    fig, df = create_cloudtrail_logs()
    st.plotly_chart(fig)
    
    st.markdown("### Sample CloudTrail Log Entries")
    st.dataframe(df)
    
    # CloudTrail code example
    st.markdown("### Code Example: Querying CloudTrail Logs for SageMaker API Calls")
    
    code = '''
    import boto3
    from datetime import datetime, timedelta
    
    # Create CloudTrail client
    cloudtrail = boto3.client('cloudtrail')
    
    # Define the time range for the query
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Look up SageMaker API events
    response = cloudtrail.lookup_events(
        LookupAttributes=[
            {
                'AttributeKey': 'EventSource',
                'AttributeValue': 'sagemaker.amazonaws.com'
            },
        ],
        StartTime=start_time,
        EndTime=end_time,
        MaxResults=10
    )
    
    # Process and display the events
    for event in response['Events']:
        print(f"Event time: {event['EventTime']}")
        print(f"Event name: {event['EventName']}")
        print(f"Username: {event['Username']}")
        print(f"Resources: {event['Resources']}")
        print("---")
    
    # Analyze specific SageMaker model deployment events
    create_model_events = cloudtrail.lookup_events(
        LookupAttributes=[
            {
                'AttributeKey': 'EventName',
                'AttributeValue': 'CreateModel'
            },
        ],
        StartTime=start_time,
        EndTime=end_time,
    )
    
    # Extract model configurations from the event
    for event in create_model_events['Events']:
        event_details = json.loads(event['CloudTrailEvent'])
        request_parameters = event_details.get('requestParameters', {})
        
        model_name = request_parameters.get('modelName', 'Unknown')
        container_image = request_parameters.get('primaryContainer', {}).get('image', 'Unknown')
        execution_role = request_parameters.get('executionRoleArn', 'Unknown')
        
        print(f"Model created: {model_name}")
        print(f"Container image: {container_image}")
        print(f"Execution role: {execution_role}")
        print(f"Created by: {event['Username']} at {event['EventTime']}")
        print("---")
    '''
    
    st.code(code, language='python')
    
    # Security and compliance use cases
    st.markdown("""
    ### Security and Compliance Use Cases for ML Workflows
    
    1. **Unauthorized Access Detection** - Monitor for unauthorized access attempts to ML models and data
    2. **Compliance Auditing** - Generate reports showing who accessed sensitive data used for ML training
    3. **Model Lineage Tracking** - Use CloudTrail logs to trace model creation and modifications
    4. **API Usage Analysis** - Track which ML APIs are being used and by whom
    5. **Sensitive Operation Monitoring** - Set up alerts for critical operations like model deletion or endpoint configuration changes
    6. **Resource Change Tracking** - Monitor changes to ML infrastructure and configurations
    """)
    
    # Best practices
    st.markdown("""
    ### Best Practices for CloudTrail with ML Applications
    
    1. **Enable CloudTrail in all regions** - Ensure comprehensive coverage across all AWS regions
    2. **Enable log file validation** - Verify that logs haven't been modified after delivery
    3. **Use log file encryption** - Encrypt CloudTrail log files for additional security
    4. **Analyze logs regularly** - Set up automated log analysis for security and operational insights
    5. **Integrate with CloudWatch Logs** - Send CloudTrail events to CloudWatch Logs for monitoring and alerting
    6. **Use least privilege IAM policies** - Restrict access to CloudTrail logs to only those who need it
    """)
    
    # Interactive demo
    st.markdown("### Interactive Demo: CloudTrail Log Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        event_source = st.selectbox(
            "Event Source",
            ["sagemaker.amazonaws.com", "s3.amazonaws.com", "iam.amazonaws.com", "All AWS Services"]
        )
        
        event_name = st.selectbox(
            "Event Name",
            ["CreateEndpoint", "CreateModel", "CreateTrainingJob", "DeleteEndpoint", "InvokeEndpoint", "All Events"]
        )
    
    with col2:
        time_period = st.select_slider(
            "Time Period",
            options=["Last 24 hours", "Last 7 days", "Last 30 days", "Last 90 days"]
        )
        
        user_identity = st.text_input("Filter by User/Role (optional)", "")
    
    if st.button("Search CloudTrail Logs (Demo)"):
        st.success("Log search completed!")
        
        if event_source == "sagemaker.amazonaws.com" and event_name == "CreateEndpoint":
            st.info("Found 5 CreateEndpoint events in the selected time period.")
            
            sample_data = [
                {"EventTime": "2023-01-05 14:23:45", "Username": "john.smith", "EndpointName": "sentiment-analysis-prod", "Status": "Success"},
                {"EventTime": "2023-01-04 09:12:18", "Username": "ml-pipeline-role", "EndpointName": "image-classification-v2", "Status": "Success"},
                {"EventTime": "2023-01-03 18:45:22", "Username": "mary.jones", "EndpointName": "recommender-system-test", "Status": "Success"},
                {"EventTime": "2023-01-02 11:35:07", "Username": "ml-pipeline-role", "EndpointName": "text-generation-beta", "Status": "Success"},
                {"EventTime": "2023-01-01 08:14:55", "Username": "admin", "EndpointName": "forecasting-model-dev", "Status": "Success"}
            ]
            
            st.table(pd.DataFrame(sample_data))
        else:
            st.markdown("No events matching your criteria were found.")

# Function to render EventBridge tab content
def render_eventbridge_tab():
    st.markdown("## Amazon EventBridge")
    
    # EventBridge description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Overview
        Amazon EventBridge is a serverless event bus service that makes it easy to connect your applications with data from various sources. With EventBridge, you can:
        
        * Create event-driven architectures
        * Route events between AWS services, SaaS applications, and your custom applications
        * Set up rules to match events and route them to targets
        * Transform and filter events before processing them
        * Schedule automated actions at regular intervals
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/products/eventbridge/Product-Page-Diagram_Amazon-EventBridge_How-it-Works.67b72c0e5467ce55a82e45ca9b2b318a64e9fcb6.png", caption="EventBridge Overview")
    
    # EventBridge key features
    st.markdown("""
    ### Key Features for ML Engineers
    
    * **Event-driven ML workflows** - Trigger ML pipelines based on events
    * **Integration with SageMaker** - React to SageMaker job status changes automatically
    * **Scheduled training jobs** - Schedule regular model retraining
    * **Automated response** - Respond automatically to model drift alerts
    * **SaaS integration** - Connect ML workflows with third-party SaaS applications
    * **Custom event patterns** - Define precise conditions for event matching
    """)
    
    # EventBridge visualization
    st.markdown("### How EventBridge Works with ML Workflows")
    st.plotly_chart(create_eventbridge_visualization())
    
    # EventBridge code example
    st.markdown("### Code Example: Creating an EventBridge Rule for ML Model Monitoring")
    
    code = '''
    import boto3
    import json
    
    # Create EventBridge client
    events = boto3.client('events')
    
    # Create rule to detect model drift alerts from SageMaker Model Monitor
    response = events.put_rule(
        Name='ModelDriftDetected',
        EventPattern=json.dumps({
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Model Monitor Violation Notification"],
            "detail": {
                "monitoringJobDefinitionName": ["my-data-quality-job-definition"],
                "violationReport": {
                    "violations": [{
                        "featureName": [{"exists": True}]
                    }]
                }
            }
        }),
        State='ENABLED',
        Description='Rule to detect when SageMaker Model Monitor finds data drift'
    )
    
    # Add target to invoke a Lambda function when rule is matched
    response = events.put_targets(
        Rule='ModelDriftDetected',
        Targets=[
            {
                'Id': 'HandleModelDrift',
                'Arn': 'arn:aws:lambda:us-east-1:123456789012:function:handle-model-drift',
                'InputTransformer': {
                    'InputPathsMap': {
                        'model': '$.detail.monitoringScheduleName',
                        'time': '$.time',
                        'violations': '$.detail.violationReport.violations'
                    },
                    'InputTemplate': json.dumps({
                        'model': '<model>',
                        'detectionTime': '<time>',
                        'violations': '<violations>',
                        'action': 'RETRAIN'
                    })
                }
            }
        ]
    )
    
    # Create a scheduled rule for model retraining
    response = events.put_rule(
        Name='WeeklyModelRetraining',
        ScheduleExpression='cron(0 12 ? * SUN *)', # Noon every Sunday
        State='ENABLED',
        Description='Trigger weekly model retraining pipeline'
    )
    
    # Add target to start a Step Functions state machine for model retraining
    response = events.put_targets(
        Rule='WeeklyModelRetraining',
        Targets=[
            {
                'Id': 'StartModelTrainingPipeline',
                'Arn': 'arn:aws:states:us-east-1:123456789012:stateMachine:ModelTrainingPipeline',
                'RoleArn': 'arn:aws:iam::123456789012:role/EventBridgeStepFunctionsRole',
                'Input': json.dumps({
                    'modelId': 'customer-churn-predictor',
                    'datasetS3Uri': 's3://my-ml-datasets/customer-data/latest/',
                    'forceTrain': True
                })
            }
        ]
    )
    '''
    
    st.code(code, language='python')
    
    # EventBridge use cases for ML
    st.markdown("""
    ### EventBridge Use Cases for ML Operations
    
    1. **Automated Model Retraining** - Trigger retraining when data drift is detected or on a schedule
    2. **Alert Notification** - Send notifications when model performance degrades
    3. **Multi-step ML Workflows** - Orchestrate complex ML workflows with multiple services
    4. **Training Job Monitoring** - Automate responses to training job completion or failure
    5. **Cost Optimization** - Scale down resources automatically when not needed
    6. **Model Approval Workflow** - Trigger human approval workflows for model deployment
    """)
    
    # EventBridge patterns for ML
    st.markdown("### Sample EventBridge Patterns for ML Workflows")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Pattern: SageMaker Training Job State Change")
        st.code('''
        {
          "source": ["aws.sagemaker"],
          "detail-type": ["SageMaker Training Job State Change"],
          "detail": {
            "TrainingJobStatus": ["Completed", "Failed", "Stopped"]
          }
        }
        ''', language="json")
        
        st.markdown("#### Pattern: SageMaker Endpoint Deployment Completed")
        st.code('''
        {
          "source": ["aws.sagemaker"],
          "detail-type": ["SageMaker Endpoint State Change"],
          "detail": {
            "EndpointStatus": ["InService"]
          }
        }
        ''', language="json")
    
    with col2:
        st.markdown("#### Pattern: ModelMonitor Violations Detected")
        st.code('''
        {
          "source": ["aws.sagemaker"],
          "detail-type": ["SageMaker Model Monitor Violation Notification"],
          "detail": {
            "monitoringScheduleName": [{"prefix": "data-quality-"}]
          }
        }
        ''', language="json")
        
        st.markdown("#### Pattern: S3 Data Upload Trigger for Inference")
        st.code('''
        {
          "source": ["aws.s3"],
          "detail-type": ["Object Created"],
          "detail": {
            "bucket": {
              "name": ["incoming-data-bucket"]
            },
            "object": {
              "key": [{"prefix": "new-data/"}]
            }
          }
        }
        ''', language="json")
    
    # Best practices
    st.markdown("""
    ### Best Practices for EventBridge with ML Operations
    
    1. **Use specific event patterns** - Create precise patterns to match only relevant events
    2. **Implement error handling** - Ensure targets have proper error handling for failed executions
    3. **Implement dead-letter queues** - Use dead-letter queues for events that can't be processed
    4. **Monitor EventBridge** - Set up monitoring for your event rules and targets
    5. **Use input transformers** - Transform events before sending them to targets
    6. **Test rules thoroughly** - Verify event patterns match expected events
    """)
    
    # Interactive demo
    st.markdown("### Interactive Demo: Creating an EventBridge Rule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        event_source = st.selectbox(
            "Event Source",
            ["aws.sagemaker", "aws.s3", "aws.cloudwatch", "Custom source"]
        )
        
        detail_type = st.selectbox(
            "Detail Type",
            ["SageMaker Training Job State Change", "SageMaker Endpoint State Change", 
             "SageMaker Model Monitor Violation Notification", "Object Created", "Scheduled Event"]
        )
    
    with col2:
        target_type = st.selectbox(
            "Target Type",
            ["Lambda function", "Step Functions state machine", "SNS topic", "SQS queue", "API destination"]
        )
        
        rule_name = st.text_input("Rule Name", "MyMLWorkflowRule")
    
    # Dynamic fields based on selections
    if detail_type == "SageMaker Training Job State Change":
        job_status = st.multiselect(
            "Job Status",
            ["Completed", "Failed", "Stopped", "InProgress"],
            ["Completed", "Failed"]
        )
        
        pattern = {
            "source": [event_source],
            "detail-type": [detail_type],
            "detail": {
                "TrainingJobStatus": job_status
            }
        }
    
    elif detail_type == "SageMaker Endpoint State Change":
        endpoint_status = st.multiselect(
            "Endpoint Status",
            ["Creating", "Updating", "SystemUpdating", "RollingBack", "InService", "OutOfService", "Deleting", "Failed"],
            ["InService", "Failed"]
        )
        
        pattern = {
            "source": [event_source],
            "detail-type": [detail_type],
            "detail": {
                "EndpointStatus": endpoint_status
            }
        }
    
    else:
        pattern = {
            "source": [event_source],
            "detail-type": [detail_type]
        }
    
    st.markdown("### Generated Event Pattern")
    st.code(json.dumps(pattern, indent=2), language="json")
    
    if st.button("Create Rule (Demo)"):
        st.success(f"EventBridge rule '{rule_name}' created successfully!")
        st.info(f"Rule will trigger {target_type} when matching events are detected.")

# Function to render QuickSight tab content
def render_quicksight_tab():
    st.markdown("## Amazon QuickSight")
    
    # QuickSight description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Overview
        Amazon QuickSight is a cloud-native, serverless business intelligence service that makes it easy to deliver insights to everyone in your organization. With QuickSight, you can:
        
        * Create and publish interactive ML-powered dashboards
        * Access dashboards from any device
        * Embed analytics into applications and portals
        * Use natural language to ask questions about your data
        * Receive automatic insights powered by machine learning
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/products/quicksight/product-page-diagram_Amazon-QuickSight-Enterprise-Edition_How-it-Works.b89003855f32dbcf4a124b298b935a9293227ab5.png", caption="QuickSight Overview")
    
    # QuickSight key features
    st.markdown("""
    ### Key Features for ML Engineers
    
    * **ML Performance Dashboards** - Visualize model performance metrics and drift indicators
    * **Data Source Connectivity** - Connect to CloudWatch metrics, SageMaker output, Athena, and other data sources
    * **Auto-refresh** - Automatically refresh dashboards with the latest ML metrics
    * **ML Insights** - Use QuickSight's built-in ML capabilities for anomaly detection
    * **Sharing and Collaboration** - Share dashboards with stakeholders across the organization
    * **Interactive Filtering** - Drill into specific time periods or model versions
    """)
    
    # QuickSight dashboard mockup
    st.markdown("### Sample ML Performance Dashboard in QuickSight")
    st.plotly_chart(create_quicksight_mockup())
    
    # QuickSight setup and integration
    st.markdown("""
    ### Setting Up QuickSight for ML Monitoring
    
    1. **Data Source Setup**
       * Connect QuickSight to CloudWatch metrics for your SageMaker endpoints
       * Create datasets from CloudWatch Logs using CloudWatch Logs Insights
       * Connect to S3 buckets containing model evaluation metrics
       * Set up direct query to Amazon Athena for large-scale analytics
    
    2. **Dashboard Creation**
       * Create KPI visualizations for critical metrics like accuracy and latency
       * Set up time series charts to monitor trends in model performance
       * Create heat maps to identify patterns in model drift
       * Use forecasts to predict future model behavior
    
    3. **Sharing and Access**
       * Share dashboards with stakeholders
       * Set up email reports for regular updates
       * Embed dashboards in internal portals
       * Configure row-level security if needed
    """)
    
    # QuickSight code example
    st.markdown("### Code Example: Embedding QuickSight Dashboard in an Application")
    
    code = '''
    import boto3
    import json
    from flask import Flask, render_template
    
    app = Flask(__name__)
    
    @app.route('/ml-dashboard')
    def ml_dashboard():
        # Get QuickSight dashboard embedding URL
        quicksight = boto3.client('quicksight')
        response = quicksight.generate_embed_url_for_registered_user(
            AwsAccountId='123456789012',
            ExperienceConfiguration={
                'Dashboard': {
                    'InitialDashboardId': 'dashboard-id'
                }
            },
            UserArn='arn:aws:quicksight:us-east-1:123456789012:user/default/user-name',
            AllowedDomains=[
                'https://example.com',
            ],
            SessionLifetimeInMinutes=600
        )
        
        embed_url = response['EmbedUrl']
        
        return render_template(
            'dashboard.html',
            embed_url=embed_url
        )
    
    if __name__ == '__main__':
        app.run(debug=True)
    '''
    
    st.code(code, language='python')
    
    # QuickSight dashboard template
    st.markdown("""
    ```html
    <!-- dashboard.html -->
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Model Performance Dashboard</title>
        <script src="https://unpkg.com/amazon-quicksight-embedding-sdk@2.0.0/dist/quicksight-embedding-js-sdk.min.js"></script>
        <style>
            #dashboardContainer {
                height: 800px;
                width: 100%;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <h1>ML Model Performance Dashboard</h1>
        <div id="dashboardContainer"></div>
        
        <script>
            const embedDashboard = async () => {
                const embeddingContext = {
                    dashboard: {
                        width: '100%',
                        height: '100%'
                    }
                };
                
                const options = {
                    url: '{{ embed_url }}',
                    container: document.getElementById('dashboardContainer'),
                    parameters: {
                        model: 'sentiment-analysis',
                        version: 'v2.1'
                    },
                    scrolling: 'no',
                    height: '800px',
                    width: '100%'
                };
                
                const dashboard = await QuickSightEmbedding.embedDashboard(options);
            };
            
            embedDashboard();
        </script>
    </body>
    </html>
    ```
    """)
    
    # Best practices
    st.markdown("""
    ### Best Practices for QuickSight ML Dashboards
    
    1. **Focus on key metrics** - Prioritize the most important ML performance indicators
    2. **Use appropriate visualizations** - Choose the right chart types for your metrics
    3. **Set up alerts** - Configure anomaly detection and alerts for critical metrics
    4. **Provide context** - Include descriptions and business impact context
    5. **Enable filtering** - Add filters for time periods, model versions, and other dimensions
    6. **Optimize refresh schedules** - Balance freshness of data with cost
    """)
    
    # Interactive demo
    st.markdown("### Interactive Demo: Designing an ML Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dashboard_title = st.text_input("Dashboard Title", "ML Model Performance Overview")
        
        primary_metrics = st.multiselect(
            "Primary Metrics (KPIs)",
            ["Accuracy", "Precision", "Recall", "F1 Score", "Latency", "Throughput", "Error Rate", "Data Drift Score"],
            ["Accuracy", "Latency", "Error Rate"]
        )
        
        chart_types = st.multiselect(
            "Chart Types to Include",
            ["Line Chart", "Bar Chart", "Heat Map", "Scatter Plot", "KPI Indicators", "Table", "Pie Chart"],
            ["Line Chart", "KPI Indicators", "Heat Map"]
        )
    
    with col2:
        data_sources = st.multiselect(
            "Data Sources",
            ["CloudWatch Metrics", "S3 (Model Evaluation Results)", "Athena Query Results", "DynamoDB"],
            ["CloudWatch Metrics", "S3 (Model Evaluation Results)"]
        )
        
        refresh_schedule = st.select_slider(
            "Dashboard Refresh Frequency",
            options=["5 minutes", "15 minutes", "1 hour", "6 hours", "1 day"]
        )
        
        audience = st.multiselect(
            "Dashboard Audience",
            ["ML Engineers", "Data Scientists", "Product Managers", "Executives", "Operations Team"],
            ["ML Engineers", "Data Scientists"]
        )
    
    if st.button("Generate Dashboard (Demo)"):
        st.success(f"Dashboard '{dashboard_title}' created successfully!")
        st.info(f"Dashboard will refresh every {refresh_schedule} and includes {len(primary_metrics)} primary metrics.")
        
        # Show a simple mockup
        st.markdown("### Dashboard Preview")
        
        # Create KPI cards
        kpi_html = ""
        for metric in primary_metrics[:3]:  # Limit to first 3
            if metric == "Accuracy":
                value = "98.2%"
                trend = "↑ 0.3%"
                color = "green"
            elif metric == "Latency":
                value = "125 ms"
                trend = "↓ 5 ms"
                color = "green"
            elif metric == "Error Rate":
                value = "1.8%"
                trend = "↓ 0.3%"
                color = "green"
            elif metric == "Precision":
                value = "97.5%"
                trend = "↑ 0.2%"
                color = "green"
            else:
                value = "85.0"
                trend = "→ 0.0"
                color = "gray"
                
            kpi_html += f"""<div style="background-color: white; border-radius: 5px; padding: 15px; margin: 10px; width: 30%; display: inline-block; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #232F3E;">{metric}</h3>
                <p style="font-size: 24px; font-weight: bold; color: #FF9900; margin: 10px 0;">{value}</p>
                <p style="margin: 0; color: {color};">{trend} from previous period</p>
            </div>
            """
        
        st.markdown(f"""
        <div style="background-color: #F8F8F8; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #232F3E;">{dashboard_title}</h2>
            <div style="display: flex; flex-wrap: wrap; margin-bottom: 20px;">{kpi_html}</div>
            <p>Dashboard includes data from: {", ".join(data_sources)}</p>
            <p>Refresh frequency: {refresh_schedule}</p>
            <p>Shared with: {", ".join(audience)}</p>
        </div>
        """, unsafe_allow_html=True)

# Main function to build the application
def main():
    # Initialize session state
    init_session_state()
    
    # Apply AWS style
    apply_aws_style()
    
    # Sidebar
    with st.sidebar:
        
        st.subheader("Session Management")
        st.info(f"User ID: {st.session_state.session_id}")
            
        if st.button("Reset Session"):
                reset_session()
        st.divider()
        # About this App (collapsible)
        with st.expander("About this App"):
            st.markdown("""
            This interactive e-learning application focuses on AWS monitoring and observability tools for Machine Learning applications. It's designed to help you understand how to monitor, troubleshoot, and optimize your ML workloads on AWS.
            
            The content is aligned with the AWS Machine Learning Engineer - Associate certification, specifically Domain 4 which covers monitoring, maintenance, and security of ML solutions.
            """)

    
    # Main content header
    st.title("Tools for Monitoring and Observability")
    st.markdown("""
    Monitoring and observability are critical aspects of running machine learning systems in production. 
    This module covers the key AWS tools and services that help you monitor, troubleshoot, and optimize 
    your ML workloads.
    """)
    
    # Tabs for different tools
    tabs = st.tabs([
        "📊 CloudWatch", 
        "🔍 X-Ray", 
        "🧾 CloudTrail", 
        "🔄 EventBridge", 
        "📈 QuickSight", 
        "🧪 Knowledge Check"
    ])
    
    with tabs[0]:
        render_cloudwatch_tab()
        
    with tabs[1]:
        render_xray_tab()
        
    with tabs[2]:
        render_cloudtrail_tab()
        
    with tabs[3]:
        render_eventbridge_tab()
        
    with tabs[4]:
        render_quicksight_tab()
        
    with tabs[5]:
        render_knowledge_check()

if __name__ == "__main__":
    main()
# ```

# ## requirements.txt
# ```
# streamlit==1.22.0
# pandas==1.5.3
# numpy==1.24.3
# matplotlib==3.7.1
# seaborn==0.12.2
# plotly==5.14.1
# Pillow==9.5.0
# ```

# This Streamlit application creates a comprehensive e-learning experience focused on AWS monitoring and observability tools for ML Engineers. The application includes:

# 1. **Five main content tabs with interactive components:**
#    - CloudWatch - Real-time monitoring of ML models with metric visualization and alarm creation
#    - X-Ray - End-to-end tracing for distributed ML applications with service maps
#    - CloudTrail - Security auditing and compliance for ML operations
#    - EventBridge - Event-driven architectures for ML workflows
#    - QuickSight - Business intelligence dashboards for ML metrics

# 2. **Knowledge Check tab with five questions:**
#    - Single-answer (radio button) questions
#    - Multi-answer (checkbox) questions
#    - Detailed explanations for both correct and incorrect answers
#    - Score tracking and feedback

# 3. **Interactive elements in each tab:**
#    - Sample code with Python examples
#    - Interactive visualizations using Plotly
#    - Demo interfaces with controls and sample outputs
#    - Best practices and use cases specific to ML workloads

# 4. **Session management:**
#    - Unique session ID generation
#    - Reset functionality
#    - Progress tracking for knowledge check

# 5. **AWS-themed styling:**
#    - AWS color scheme (navy blue #232F3E and orange #FF9900)
#    - Clean and responsive layout
#    - Consistent formatting throughout the application

# The application is designed to help ML engineers understand the importance of monitoring and observability in their ML workflows, with practical examples and interactive components that reinforce learning.