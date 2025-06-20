
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
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
if 'visited_ML_Lifecycle' not in st.session_state:
    st.session_state['visited_ML_Lifecycle'] = False
if 'visited_Modeling_Approaches' not in st.session_state:
    st.session_state['visited_Modeling_Approaches'] = False
if 'visited_Amazon_Bedrock' not in st.session_state:
    st.session_state['visited_Amazon_Bedrock'] = False
if 'visited_Neural_Networks' not in st.session_state:
    st.session_state['visited_Neural_Networks'] = False
if 'visited_Model_Training' not in st.session_state:
    st.session_state['visited_Model_Training'] = False

# Custom CSS for styling
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
    .stButton>button {
        background-color: #FF9900;
        color: white;
    }
    .stButton>button:hover {
        background-color: #FFAC31;
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

# Function to create custom info box
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

# Function for definition box
def definition_box(term, definition):
    st.markdown(f"""
    <div class="definition">
        <strong>{term}:</strong> {definition}
    </div>
    """, unsafe_allow_html=True)

# Function to reset session
def reset_session():
    st.session_state['quiz_score'] = 0
    st.session_state['quiz_attempted'] = False
    st.session_state['name'] = ""
    st.session_state['visited_ML_Lifecycle'] = False
    st.session_state['visited_Modeling_Approaches'] = False
    st.session_state['visited_Amazon_Bedrock'] = False
    st.session_state['visited_Neural_Networks'] = False
    st.session_state['visited_Model_Training'] = False
    st.rerun()

# Sidebar for session management
with st.sidebar:
    st.image("images/mla_badge.png", width=150)
    st.markdown("### ML Engineer - Associate")
    st.markdown("#### Domain 2: ML Model Development")
    
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
        visited_pages = [page for page in ["ML_Lifecycle", "Modeling_Approaches", "Amazon_Bedrock", "Neural_Networks", "Model_Training"] 
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
        - Understand the ML lifecycle
        - Choose appropriate modeling approaches
        - Utilize Amazon Bedrock for generative AI
        - Implement hyperparameter tuning
        - Apply distributed training techniques
        """)
    
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("""
    - [AWS ML Documentation](https://docs.aws.amazon.com/machine-learning)
    - [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker)
    - [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock)
    """)

# Main content with tabs
tabs = st.tabs([
    "🏠 Home", 
    "🔄 ML Lifecycle", 
    "🧠 Modeling Approaches", 
    "🤖 Amazon Bedrock", 
    "🔬 Neural Networks", 
    "🚂 Model Training", 
    "⚙️ Hyperparameters", 
    "❓ Knowledge Check", 
    "📚 Resources"
])

# Home tab
with tabs[0]:
    custom_header("AWS Partner Certification Readiness")
    st.markdown("## Machine Learning Engineer - Associate")
    
    st.markdown("### Domain 2: ML Model Development")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        info_box("""
        This interactive e-learning application covers the main topics of Domain 2 from the AWS Machine Learning Engineer - Associate certification.
        
        Domain 2 focuses on **ML Model Development**, which accounts for a significant portion of the certification exam.
        
        Navigate through the content using the tabs above to learn about:
        - Machine Learning Lifecycle
        - Modeling Approaches
        - Amazon Bedrock
        - Neural Networks
        - Model Training and Hyperparameter Tuning
        
        Test your knowledge with the quiz when you're ready!
        """, "info")
        
        st.markdown("### Learning Outcomes")
        st.markdown("""
        By the end of this module, you will be able to:
        - Understand the ML lifecycle and where model development fits
        - Select appropriate modeling approaches based on requirements
        - Utilize Amazon Bedrock for foundation model applications
        - Implement effective hyperparameter tuning techniques
        - Apply distributed training for large-scale models
        - Choose the right ensemble learning strategy
        """)
    
    with col2:
        st.image("images/mla_badge_big.png", width=250)
        
        if st.session_state['quiz_attempted']:
            st.success(f"Current Quiz Score: {st.session_state['quiz_score']}/5")
        
        st.info("Use the tabs above to navigate through different sections!")
    
    st.markdown("---")
    
    st.markdown("### Session Roadmap")
    
    roadmap_data = {
        'Sessions': ['Sessions 1 & 2', 'Sessions 3 & 4', 'Sessions 5 & 6', 'Sessions 7 & 8', 'Sessions 9 & 10'],
        'Focus': ['Domain 1: Data Preparation for ML', 'Domain 2: ML Model Development', 'Domain 2: ML Model Development',
                 'Domain 3: Deployment and Orchestration', 'Domain 4: Monitoring, Maintenance, and Security']
    }
    
    roadmap_df = pd.DataFrame(roadmap_data)
    
    # Apply custom styling to highlight the current session
    def highlight_current_row(row):
        styles = [''] * len(row)
        if row['Sessions'] == 'Sessions 3 & 4':
            styles = ['background-color: #FFE4B5; font-weight: bold'] * len(row)
        return styles
    
    # Display the roadmap with styled highlight
    st.dataframe(roadmap_df.style.apply(highlight_current_row, axis=1), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Task Statements in Domain 2")
    
    task_col1, task_col2 = st.columns(2)
    
    with task_col1:
        st.markdown("#### Task 2.1: Choose a modeling approach")
        st.markdown("""
        - Understand business requirements
        - Select appropriate machine learning algorithm
        - Determine training approach
        - Choose validation strategy
        - Implement supervised, unsupervised, or custom model development
        """)
    
    with task_col2:
        st.markdown("#### Task 2.2: Train and refine models")
        st.markdown("""
        - Set up distributed training jobs
        - Apply hyperparameter optimization
        - Evaluate model quality
        - Implement ensemble learning strategies
        - Use transfer learning and fine-tuning
        - Improve models iteratively
        """)

# ML Lifecycle tab
with tabs[1]:
    # Mark as visited
    st.session_state['visited_ML_Lifecycle'] = True
    
    custom_header("Machine Learning Lifecycle")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        The machine learning lifecycle consists of several key phases that guide the development and deployment of ML models.
        Understanding this lifecycle helps you approach ML projects systematically.
        """)
        
        # ML Lifecycle diagram
        st.image('images/ml_lifecycle.png')
    
    with col2:
        info_box("""<b>You are here: Model Development</b>
        
In this phase, you'll focus on:
- Training & Hyperparameter Tuning
- Built-in algorithms
- Automated model development
- Distributed training
        """, "tip")
    
    st.markdown("""
    In the **Model Development** phase, your data processing is complete, and your data is typically stored in S3.
    Now, you need to select the appropriate modeling approach and train your model.
    
    Key components of model development include:
    
    1. **Training and Hyperparameter tuning**
       - SageMaker training jobs
       - Built-in algorithms
       - Bring your own script or container
       - SageMaker Experiments and Debugger
       - Automatic Model Tuning
    
    2. **Automated and preconfigured model development**
       - Autopilot - Automated ML
       - SageMaker JumpStart
    
    3. **Distributed training and optimization**
       - Distributed Training frameworks
       - Training Compiler
    """)
    
    # AWS AI/ML Stack
    custom_header("AWS AI/ML Stack", "section")
    
    stack_data = {
        'Layer': ['AI Services', 'ML Services', 'ML Frameworks & Infrastructure'],
        'Description': [
            'Pre-trained AI services for ready-to-use intelligence',
            'Services to build, train, and deploy ML models (SageMaker)',
            'Support for deep learning frameworks, compute options'
        ],
        'Expertise Required': ['No ML expertise needed', 'Some ML knowledge', 'Deep ML expertise'],
        'Examples': [
            'Amazon Rekognition, Amazon Comprehend, Amazon Bedrock',
            'Amazon SageMaker, SageMaker JumpStart, SageMaker Autopilot',
            'TensorFlow, PyTorch, Apache MXNet on EC2/ECS'
        ]
    }
    
    stack_df = pd.DataFrame(stack_data)
    st.table(stack_df)
    
    custom_header("Model Development in the ML Process", "sub")
    
    st.markdown("""
    Model development is a critical phase in the machine learning lifecycle, consisting of several steps:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. Problem Framing
        - Understand the business problem
        - Define the ML objectives clearly
        - Determine success criteria
        - Select appropriate ML approach (supervised, unsupervised, etc.)
        
        ### 2. Algorithm Selection
        - Choose algorithm based on data type and problem
        - Consider built-in algorithms vs. custom models
        - Evaluate tradeoffs between accuracy, interpretability, and speed
        - Consider ensemble methods when appropriate
        """)
    
    with col2:
        st.markdown("""
        ### 3. Model Training
        - Configure training jobs
        - Select appropriate compute resources
        - Set hyperparameters
        - Implement cross-validation strategies
        
        ### 4. Model Evaluation
        - Assess model performance using validation datasets
        - Fine-tune hyperparameters
        - Compare different approaches
        - Select final model for deployment
        """)
    
    custom_header("SageMaker Components for Model Development", "section")
    
    # st.image("images/sagemaker_workflow.png", caption="Amazon SageMaker Workflow", width=800)
    
    st.markdown("""
    SageMaker provides a comprehensive suite of tools and capabilities specifically for the model development phase:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### SageMaker Training")
        st.markdown("""
        - Configure and run training jobs
        - Specify instance types and counts
        - Set hyperparameters
        - Use different data input modes
        - Enable distributed training
        - Support for spot instances
        """)
    
    with col2:
        st.markdown("### SageMaker Built-in Algorithms")
        st.markdown("""
        - Pre-implemented, optimized algorithms
        - Multiple algorithm categories:
          - Linear Learner
          - XGBoost
          - Image Classification
          - Object Detection
          - Seq2Seq
          - DeepAR
          - K-Means
        """)
    
    with col3:
        st.markdown("### SageMaker Model Development")
        st.markdown("""
        - Automated ML with Autopilot
        - Pre-trained models with JumpStart
        - Experiments for tracking trials
        - Debugger for monitoring training
        - Feature Store for feature management
        - Model Registry for versioning
        """)
    
    info_box("""
    **The Iterative Nature of ML Development**
    
    Machine learning is highly iterative. You may need to revisit earlier phases as you discover new insights:
    
    - If model performance is poor, you might need better features
    - Different algorithms may need to be tried
    - Hyperparameter tuning may need multiple iterations
    - Error analysis often leads to data improvements
    
    This is why SageMaker offers tools like Experiments to track multiple trials and versions.
    """, "info")

# Modeling Approaches tab
with tabs[2]:
    # Mark as visited
    st.session_state['visited_Modeling_Approaches'] = True
    
    custom_header("Modeling Approaches")
    
    st.markdown("""
    When developing ML models on AWS, you have several approaches to choose from, depending on your requirements,
    expertise, and the complexity of your problem.
    """)
    
    # Create a comparison of model development methods
    st.image("images/model_development_methods.png", caption="Spectrum of AWS Machine Learning Options", width=1200)
    
    custom_header("SageMaker Model Development Methods", "sub")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Built-in Algorithms")
        st.markdown("""
        - Pre-implemented algorithms
        - No code required
        - Fast development
        - Scalable and optimized
        - Limited customization
        
        **Best for:** Standard problems with established algorithms
        """)
    
    with col2:
        st.markdown("### Bring Your Own Script")
        st.markdown("""
        - Use familiar ML frameworks
        - Customize training logic
        - SageMaker manages infrastructure
        - Supported frameworks:
          - TensorFlow, PyTorch
          - Scikit-learn, XGBoost
          - MXNet, HuggingFace
        
        **Best for:** Custom models using standard frameworks
        """)
    
    with col3:
        st.markdown("### Bring Your Own Container")
        st.markdown("""
        - Maximum flexibility
        - Full control over environment
        - Custom frameworks or packages
        - Build Docker containers
        - Integrate with SageMaker
        
        **Best for:** Highly specialized models or non-standard frameworks
        """)
    
    info_box("""
    <b>Note:</b> All these methods rely on containerization. The container includes the training code,
    dependencies, and runtime environment needed to train your model.
    """, "warning")
    
    # Supervised Learning Algorithms section
    custom_header("Supervised Learning Algorithms", "section")
    
    st.markdown("""
    Supervised learning algorithms learn from labeled training data to make predictions or classifications.
    SageMaker provides built-in implementations of many common supervised learning algorithms.
    """)
    
    # Create visualization of supervised learning algorithms
    st.image('images/supervised_learning.png',width=800)
    
    # Unsupervised learning
    custom_header("Unsupervised Learning Algorithms", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        Unsupervised learning algorithms find patterns in unlabeled data. These algorithms are used for:
        
        - **Clustering**: Group similar items together (K-means)
        - **Dimensionality Reduction**: Simplify data while preserving information (PCA)
        - **Anomaly Detection**: Find unusual patterns (Random Cut Forest)
        - **Topic Modeling**: Discover abstract topics in documents (LDA, NTM)
        """)
        
        st.image('images/unsupervised_learning.png',width=800)

    with col2:
        # Simple representation of unsupervised learning
        cluster_data = np.random.randn(100, 2) * 0.8
        cluster_centers = [(2, 2), (-2, 2), (0, -2)]
        for center_x, center_y in cluster_centers:
            new_points = np.random.randn(30, 2) * 0.3 + np.array([center_x, center_y])
            cluster_data = np.vstack([cluster_data, new_points])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=np.repeat(range(4), [100, 30, 30, 30]), 
                           cmap='viridis', alpha=0.6)
        ax.set_title("Unsupervised Learning: Clustering Example")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    info_box("""<b>Key SageMaker Built-in Algorithms:</b><br><br>
    
<b>For structured data:</b>
- XGBoost: For both classification and regression
- K-Nearest Neighbors (K-NN): Classification and regression
- Linear Learner: Linear and logistic regression

<b>For text data:</b>
- BlazingText: Word embeddings and text classification
- Neural Topic Model (NTM): Topic discovery
- LDA: Document topic discovery

<b>For images:</b>
- Image Classification: Multi-class classification
- Object Detection: Localize and classify objects
- Semantic Segmentation: Pixel-level classification
    """, "info")
    
    # Special data types
    custom_header("Algorithms for Specialized Data Types", "section")
    
    st.markdown("""
    SageMaker offers specialized algorithms for specific data types:
    """)
    
    tab1, tab2, tab3 = st.tabs(["Text/Speech Data", "Image/Video Data", "Time Series Data"])
    
    with tab1:
        st.markdown("""
        ### Text and Speech Algorithms
        
        - **BlazingText**
          - Word2Vec embeddings
          - Text classification
          - Highly optimized implementation
          
        - **Object2Vec**
          - Multi-purpose embeddings
          - Customer-product, document-document relationships
          
        - **Sequence-to-Sequence**
          - Translation, summarization
          - Speech-to-text applications
          
        - **LDA and NTM**
          - Topic modeling
          - Document classification
          - Content recommendation
        """)
    
    with tab2:
        st.markdown("""
        ### Image and Video Algorithms
        
        - **Image Classification**
          - Multi-class/multi-label classification
          - Built on ResNet architecture
          
        - **Object Detection**
          - Locate and classify objects in images
          - Single Shot Detector (SSD)
          - Faster R-CNN implementations
          
        - **Semantic Segmentation**
          - Pixel-level classification
          - Scene understanding
          - Built on FCN, MobileNet architectures
        """)
    
    with tab3:
        st.markdown("""
        ### Time Series Algorithms
        
        - **DeepAR**
          - Time-series forecasting
          - Recurrent neural network (RNN) based
          - Supports multiple related time series
          
        - **Forecasting**
          - Automatic forecasting algorithm selection
          - Statistical and ML approaches
          - Auto-handles seasonality and missing values
        """)

# Amazon Bedrock tab
with tabs[3]:
    # Mark as visited
    st.session_state['visited_Amazon_Bedrock'] = True
    
    custom_header("Amazon Bedrock")
    
    st.markdown("""
    Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) 
    from leading AI companies through a single API, along with a comprehensive set of capabilities to build 
    generative AI applications with security, privacy, and responsible AI.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Key Features of Amazon Bedrock
        
        - **Choice of leading FMs through a single API**
          - Access multiple foundation models from companies like Anthropic, AI21, Cohere, and Amazon
          - Single API simplifies integration and experimentation
        
        - **Model customization**
          - Fine-tune models to better suit specific use cases
          - Optimize performance for your domain
        
        - **Retrieval Augmented Generation (RAG)**
          - Connect foundation models to your data sources
          - Generate more relevant, contextual, and accurate responses
        
        - **Agents that execute multistep tasks**
          - Create AI agents that can orchestrate complex workflows
          - Integrate with business systems and data sources
        
        - **Security, privacy, and safety**
          - Enterprise-grade security features
          - Private deployment options
          - Control model access and usage
        """)
    
    with col2:
        st.image("images/kb.png", caption="Amazon Bedrock Architecture")
        
        info_box("""<b>Foundation models (FMs)</b> are large AI models pre-trained on vast amounts of data that can be adapted to a wide range of tasks.
        
Unlike traditional ML models built for specific tasks, FMs provide a versatile foundation that can be customized for various applications.
        """, "tip")
    
    custom_header("Customizing Foundation Models", "section")
    
    st.image('images/customizing_fm.png')
    
    st.markdown("""
    ### Common Approaches for Customizing Foundation Models
    
    1. **Prompt Engineering**
       - Crafting effective prompts to guide model outputs
       - No model training required
       - Quick and cost-effective, but limited customization
       
    2. **Retrieval Augmented Generation (RAG)**
       - Retrieving relevant knowledge to supplement model responses
       - Connects models to your data sources
       - Balances customization and complexity
       
    3. **Fine-tuning**
       - Adapting models for specific tasks using your data
       - Requires model training but with smaller datasets
       - Better performance on domain-specific tasks
       
    4. **Continued pretraining**
       - Further training the model on large amounts of domain data
       - Most complex and resource-intensive approach
       - Maximum customization for specialized domains
    """)
    
    custom_header("Knowledge Bases for Amazon Bedrock", "section")
    
    st.markdown("""
    Knowledge Bases for Amazon Bedrock enable you to implement Retrieval Augmented Generation (RAG) 
    by securely connecting foundation models to your data sources.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key Benefits
        
        - **Securely connect FMs to your data sources**
          - Integrate with your enterprise data
          - Keep sensitive information private
        
        - **Fully managed RAG workflow**
          - Automatic ingestion of your data
          - Intelligent retrieval of relevant information
          - Seamless prompt augmentation
        
        - **Session context management**
          - Maintain conversation history
          - Support for multi-turn interactions
          
        - **Automatic citations**
          - Track source information
          - Improve result transparency
          - Validate information accuracy
        """)
    
    with col2:
        # Creating a simple RAG workflow diagram
        st.image('images/rag.png')
        
        st.caption("RAG workflow in Amazon Bedrock Knowledge Bases")
    
    info_box("""<b>RAG in Action with Amazon Bedrock:</b><br><br>
    
1. <b>Data ingestion:</b> Your documents are chunked and converted to vector embeddings.<br>
2. <b>User query:</b> When a user asks a question, it's processed into a vector embedding.<br>
3. <b>Semantic search:</b> The system finds the most relevant document chunks.<br>
4. <b>Context enrichment:</b> The original query is augmented with retrieved information.<br>
5. <b>Foundation model processing:</b> The enriched prompt is sent to the foundation model.<br>
6. <b>Response:</b> The model generates an answer based on both its training and the supplied context.
    """, "success")

# Neural Networks tab
with tabs[4]:
    # Mark as visited
    st.session_state['visited_Neural_Networks'] = True
    
    custom_header("Neural Network Architecture")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) 
        organized in layers that process information and learn from data.
        
        ### Key Components of Neural Networks
        
        - **Input Layer**: Where data enters the network
        - **Hidden Layers**: Internal processing layers that transform data
        - **Output Layer**: Produces the final result/prediction
        - **Nodes/Neurons**: Processing units that apply activation functions
        - **Weights**: Connection strengths between nodes that are adjusted during learning
        - **Activation Functions**: Non-linear functions that determine node output (e.g., ReLU, Sigmoid, Tanh)
        """)
    
    with col2:
        # Create a neural network visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        def draw_neural_network(ax, layer_sizes, layer_names=None):
            """Draw a neural network diagram"""
            if layer_names is None:
                layer_names = [f"Layer {i+1}" for i in range(len(layer_sizes))]
            
            # Vertical spacing
            v_spacing = 1
            h_spacing = 3
            
            # Compute positions
            layer_positions = []
            for i, size in enumerate(layer_sizes):
                layer_pos = []
                for j in range(size):
                    layer_pos.append((i*h_spacing, (size-1)/2 - j*v_spacing))
                layer_positions.append(layer_pos)
            
            # Draw nodes
            for i, layer in enumerate(layer_positions):
                for j, pos in enumerate(layer):
                    circle = plt.Circle(pos, 0.5, fill=True, facecolor='#FF9900' if i==0 else ('#232F3E' if i==len(layer_positions)-1 else '#1E88E5'), alpha=0.7)
                    ax.add_patch(circle)
                    
                # Add layer name
                if layer:
                    ax.text(i*h_spacing, -3, layer_names[i], ha='center')
            
            # Draw edges
            for i in range(len(layer_positions) - 1):
                for j, pos_a in enumerate(layer_positions[i]):
                    for k, pos_b in enumerate(layer_positions[i+1]):
                        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], 'k-', alpha=0.3)
            
            # Set limits
            ax.set_aspect('equal')
            ax.set_xlim(-1, (len(layer_sizes) - 1) * h_spacing + 1)
            ax.set_ylim(-3.5, (max(layer_sizes) - 1) * v_spacing / 2 + 1)
            ax.axis('off')
        
        # Define network architecture
        layer_sizes = [4, 5, 5, 3]
        layer_names = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]
        
        draw_neural_network(ax, layer_sizes, layer_names)
        st.pyplot(fig)
    
    st.markdown("""
    ### How Neural Networks Work
    
    1. **Forward Propagation**
       - Input values are fed into the input layer
       - Each node receives inputs, applies weights, sums them, and applies an activation function
       - The output is passed to the next layer
       - This continues until the output layer produces predictions
    
    2. **Loss Calculation**
       - The model's predictions are compared to actual values
       - A loss function quantifies the error (e.g., MSE for regression, cross-entropy for classification)
    
    3. **Backpropagation**
       - The error is propagated backwards through the network
       - Gradients are calculated to determine how weights should be adjusted
       - The goal is to minimize the loss function
    
    4. **Weight Updates**
       - Weights are updated based on gradients and learning rate
       - The process repeats with new training examples
    """)
    
    info_box("""
    <b>Deep Learning</b> refers to neural networks with multiple hidden layers. These deep architectures can learn hierarchical features from data, with early layers learning simple features and deeper layers learning more complex, abstract features.
    """, "info")
    
    custom_header("Types of Neural Networks", "section")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Feedforward Neural Networks")
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*3fA77_mLNiJTSgZFhYnU0Q.png", caption="Feedforward NN")
        st.markdown("""
        - Information flows in one direction
        - Basic architecture for classification/regression
        - Fully connected layers
        - Good for tabular data
        """)
    
    with col2:
        st.markdown("### Convolutional Neural Networks (CNN)")
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg", caption="CNN")
        st.markdown("""
        - Specialized for grid-like data (images)
        - Uses convolutional filters to detect patterns
        - Pooling layers for downsampling
        - Extracts spatial features
        - Used in computer vision tasks
        """)
    
    with col3:
        st.markdown("### Recurrent Neural Networks (RNN)")
        st.image("images/rnn.png", caption="RNN")
        st.markdown("""
        - Processes sequential data
        - Maintains memory of previous inputs
        - Feedback connections
        - LSTM and GRU variants address vanishing gradients
        - Used for time series, NLP
        """)
    
    custom_header("Common Neural Network Architectures", "sub")
    
    st.markdown("""
    Each neural network architecture is designed to handle specific types of problems and data formats. Understanding these architectures helps in selecting the right one for your task.
    """)
    
    with st.expander("Multilayer Perceptron (MLP)"):
        st.markdown("""
        **Multilayer Perceptron (MLP)** is the most basic type of feedforward neural network.
        
        - **Structure**: Input layer, one or more hidden layers, output layer
        - **Characteristics**: Fully connected layers with non-linear activation functions
        - **Use cases**: 
          - Tabular data classification and regression
          - Simple pattern recognition
          - Feature learning
        - **Strengths**:
          - Simple to implement and understand
          - Can approximate any continuous function with enough neurons
          - Works well for structured data
        - **Weaknesses**:
          - Doesn't capture spatial or temporal relationships well
          - May require more parameters compared to specialized architectures
        """)
    
    with st.expander("Convolutional Neural Networks (CNN) - Detailed"):
        st.markdown("""
        **Convolutional Neural Networks (CNNs)** are specialized for processing grid-like data, such as images.
        
        - **Key Components**:
          - **Convolutional layers**: Apply filters to detect features
          - **Pooling layers**: Reduce dimensionality while preserving important information
          - **Fully connected layers**: Final classification/regression based on extracted features
        
        - **Popular CNN Architectures**:
          - **LeNet**: Early pioneering CNN for digit recognition
          - **AlexNet**: Breakthrough architecture for image classification
          - **VGG**: Simple architecture with small filters but many layers
          - **ResNet**: Introduced skip connections to train very deep networks
          - **Inception/GoogleNet**: Uses multi-scale processing
        
        - **Use cases**:
          - Image classification
          - Object detection and localization
          - Image segmentation
          - Face recognition
          - Medical image analysis
        """)
    
    with st.expander("Recurrent Neural Networks (RNN) - Detailed"):
        st.markdown("""
        **Recurrent Neural Networks (RNNs)** are specialized for sequential data, where the order of inputs matters.
        
        - **Key Variants**:
          - **Vanilla RNN**: Basic recurrent structure (rarely used due to vanishing gradients)
          - **LSTM (Long Short-Term Memory)**: Better at capturing long-range dependencies
          - **GRU (Gated Recurrent Unit)**: Simplified LSTM with fewer parameters
          - **Bidirectional RNN**: Processes sequences in both forward and backward directions
        
        - **Use cases**:
          - Natural language processing
          - Speech recognition
          - Time series prediction
          - Machine translation
          - Sentiment analysis
        
        - **Limitations**:
          - Can be difficult to train due to vanishing/exploding gradients
          - Computationally expensive for very long sequences
          - Has been increasingly replaced by transformer architectures for many NLP tasks
        """)
    
    with st.expander("Transformer Architecture"):
        st.markdown("""
        **Transformers** have revolutionized NLP and are increasingly used in other domains.
        
        - **Key Components**:
          - **Self-attention mechanisms**: Allow the model to focus on different parts of the input sequence
          - **Multi-head attention**: Multiple attention mechanisms in parallel
          - **Positional encoding**: Adds position information to inputs
          - **Feed-forward neural networks**: Process the attention outputs
        
        - **Popular Transformer Models**:
          - **BERT**: Bidirectional Encoder Representations from Transformers
          - **GPT (1-4)**: Generative Pre-trained Transformer series
          - **T5**: Text-to-Text Transfer Transformer
          - **BART**: Bidirectional and Auto-Regressive Transformers
        
        - **Use cases**:
          - Language understanding and generation
          - Document summarization
          - Translation
          - Question answering
          - Now expanding to vision, audio, and multimodal tasks
        
        - **Advantages**:
          - Captures long-range dependencies effectively
          - Highly parallelizable (unlike RNNs)
          - Scales well with more data and compute
          - State-of-the-art performance on many tasks
        """)
    
    definition_box("Neural Network", "A computational model inspired by the human brain, consisting of layers of interconnected nodes (neurons) that process and transform data, capable of learning complex patterns from examples.")

# Model Training tab
with tabs[5]:
    # Mark as visited
    st.session_state['visited_Model_Training'] = True
    
    custom_header("Model Training and Hyperparameters")
    
    st.markdown("""
    Model training is the process of teaching a machine learning model to make accurate predictions by showing it examples.
    The model learns patterns from the data and adjusts its internal parameters to minimize errors.
    
    Hyperparameters are external configuration variables that control the behavior of a machine learning algorithm and significantly impact model performance.
    """)
    
    custom_header("Amazon SageMaker Training Jobs", "sub")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Amazon SageMaker provides a fully managed environment for training machine learning models.
        When you create a training job, SageMaker:
        
        1. **Sets up the training environment**
           - Provisions the specified compute instances
           - Installs the necessary software and dependencies
        
        2. **Loads your data**
           - Fetches data from Amazon S3, EFS, or FSx
           - Makes it available to the training code
        
        3. **Executes your training code**
           - Runs your algorithm within a container
           - Handles distributed training if multiple instances are specified
        
        4. **Monitors the training process**
           - Collects logs and metrics
           - Tracks resource utilization
        
        5. **Stores the model artifacts**
           - Saves trained model to S3 when training completes
           - Makes it available for deployment
        """)
    
    with col2:
        st.image("images/sagemaker_training_job.png", caption="SageMaker Training Overview")
        
        info_box("""<b>SageMaker Training Features:</b><br><br>
        
• Automatic model tuning<br>
• Distributed training<br>
• Spot instance support<br>
• Checkpointing<br>
• Training metrics<br>
• Debug and profile training jobs<br>
• Custom algorithms and containers
        """, "tip")
    
    custom_header("Loading Training Data from Amazon S3", "section")
    
    st.markdown("""
    Amazon SageMaker provides multiple options for loading training data into your training job.
    Choosing the right data loading mode can significantly impact training performance.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### File Mode")
        st.image("images/s3_file_mode.png", width=400)
        st.markdown("""
        - Downloads entire dataset before training starts
        - Requires sufficient storage space
        - **Default setting**
        - Best for: Small datasets, simple workflows
        - All data must fit on training instance storage
        """)
    
    with col2:
        st.markdown("### Fast File Mode")
        st.image("images/s3_fast_file_mode.png", width=400)
        st.markdown("""
        - Looks like normal files but streams in background
        - No waiting for downloads, reduced storage needs
        - Training starts immediately
        - Best for: Most modern workflows, random access patterns
        - Works well with large datasets
        """)
    
    with col3:
        st.markdown("### Pipe Mode")
        st.image("images/s3_pipe_mode.png", width=400)
        st.markdown("""
        - Streaming mode that reads data sequentially
        - Direct streaming from S3 to training algorithm
        - Largely replaced by the newer Fast File mode
        - Best for: Algorithms that don't support Fast File mode
        - Higher throughput for sequential access
        """)
    
    info_box("""
    <b>Recommendation:</b> Fast File Mode is generally the best option for most modern workflows, as it combines the benefits of immediately starting training with the convenience of file system access.
    """, "warning")
    
    custom_header("Key Hyperparameters", "sub")
    
    st.markdown("""
    Hyperparameters are external configuration variables that control the behavior of a machine learning algorithm.
    Unlike model parameters (weights and biases) that are learned during training, hyperparameters are set before
    training begins and influence how the model learns.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Common Hyperparameters
        
        - **Learning rate**: Controls step size during optimization
        - **Number of epochs**: How many times the training dataset is processed
        - **Batch size**: Number of samples processed before model update
        - **Network architecture**: Number of layers, nodes per layer
        - **Regularization parameters**: L1/L2 coefficients to prevent overfitting
        - **Tree depth**: Maximum depth for tree-based algorithms
        - **Number of estimators**: Trees in ensemble methods like Random Forest
        """)
    
    with col2:
        info_box("""<b>Why Hyperparameters Matter:</b><br><br>
        
• Directly impact model performance<br>
• Affect training speed and convergence<br>
• Influence model complexity and overfitting<br>
• Can make the difference between a successful and failed model
        """, "info")
    
    custom_header("Learning Rate", "section")
    
    st.markdown("""
    The learning rate is one of the most important hyperparameters. It controls how much the model parameters
    are adjusted during training in response to the estimated error.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Function to create learning rate visualization
        def plot_learning_rate_effect(alpha):
            # Simple function with local minimum
            x = np.linspace(-5, 5, 100)
            y = x**2 + 2*np.sin(x)
            
            # Starting point
            x0 = 4
            y0 = x0**2 + 2*np.sin(x0)
            
            # Gradient at x0
            grad = 2*x0 + 2*np.cos(x0)
            
            # Update
            x1 = x0 - alpha * grad
            y1 = x1**2 + 2*np.sin(x1)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot function
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Plot current point
            ax.scatter(x0, y0, color='red', s=100, zorder=3)
            
            # Plot update
            ax.scatter(x1, y1, color='green', s=100, zorder=3)
            
            # Plot update vector
            ax.arrow(x0, y0, x1-x0, y1-y0, head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=2)
            
            # Annotate points
            ax.annotate("Starting point", (x0, y0), xytext=(x0+0.5, y0+2), arrowprops=dict(facecolor='black', shrink=0.05))
            ax.annotate("Updated position", (x1, y1), xytext=(x1-0.5, y1+2), arrowprops=dict(facecolor='black', shrink=0.05))
            
            # Title and labels
            ax.set_title(f"Effect of Learning Rate = {alpha}")
            ax.set_xlabel("Parameter Value")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            return fig
        
        # Plot with small learning rate
        fig = plot_learning_rate_effect(0.1)
        st.pyplot(fig)
        st.markdown("**Small Learning Rate**: Slow convergence but stable")
    
    with col2:
        # Plot with large learning rate
        fig = plot_learning_rate_effect(0.5)
        st.pyplot(fig)
        st.markdown("**Large Learning Rate**: Fast updates but may overshoot minimum")
    
    custom_header("Hyperparameter Tuning", "sub")
    
    st.markdown("""
    Hyperparameter tuning is the process of finding the optimal hyperparameter values for a machine learning model.
    Since hyperparameters cannot be learned directly from the training data, we need special techniques to find the best configuration.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Grid Search")
        st.image("images/grid_search.jpg", width=250)
        st.markdown("""
        - Systematically searches all combinations
        - Divides each hyperparameter range into equally spaced values
        - Exhaustive but inefficient for high-dimensional spaces
        - Computationally expensive
        - Works well for small search spaces
        """)
    
    with col2:
        st.markdown("### Random Search")
        st.image("images/random_search.jpg", width=250)
        st.markdown("""
        - Randomly samples points from hyperparameter space
        - More efficient than grid search
        - Better coverage with same compute budget
        - Can be more effective at finding optimal values
        - Works well for spaces with few important parameters
        """)
    
    with col3:
        st.markdown("### Bayesian Optimization")
        st.image("images/bayesian.jpg", width=250)
        st.markdown("""
        - Uses results of previous evaluations
        - Builds probabilistic model of the objective
        - Intelligently selects next points to evaluate
        - More efficient for expensive function evaluations
        - **Default strategy in SageMaker**
        """)
    
    custom_header("Automatic Model Tuning in Amazon SageMaker", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Amazon SageMaker provides automatic model tuning (also known as hyperparameter tuning) to find the best version of a model by running many training jobs on your dataset using different hyperparameter combinations.
        
        ### Key Components for Setting Up a Tuning Job
        
        1. **Objective metrics**
           - The metric to optimize (maximize or minimize)
           - Example: accuracy, AUC, F1-score, MSE
        
        2. **Hyperparameter ranges**
           - The search space for each hyperparameter
           - Types: continuous, integer, categorical
        
        3. **Maximum number of jobs**
           - Total number of training jobs to run
        
        4. **Maximum parallel jobs**
           - Number of concurrent training jobs
        """)
        
        st.code("""
# Setup the hyperparameter ranges
hyperparameter_ranges = {
    'eta': ContinuousParameter(0, 1),
    'min_child_weight': ContinuousParameter(1, 10),
    'alpha': ContinuousParameter(0, 2),
    'max_depth': IntegerParameter(1, 10),
    'num_round': IntegerParameter(100, 1000)
}

# Define the target metric and the objective type (max/min)
objective_metric_name = 'validation:auc'
objective_type='Maximize'

# Define the HyperparameterTuner
tuner = HyperparameterTuner(
    estimator = xgb,
    objective_metric_name = objective_metric_name,
    hyperparameter_ranges = hyperparameter_ranges,
    objective_type = objective_type,
    max_jobs=9,
    max_parallel_jobs=3,
    early_stopping_type='Auto'
)

# Start the tuning job
tuner.fit({'training': inputs})
""", language="python")
    
    with col2:
        # Create diagram for hyperparameter tuning
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create boxes and arrows for the workflow
        components = [
            {"name": "Define hyperparameter ranges\nand objective metric", "y": 5},
            {"name": "Create tuning job", "y": 4},
            {"name": "Run multiple training jobs\nwith different configurations", "y": 3},
            {"name": "Evaluate model performance", "y": 2},
            {"name": "Select best hyperparameters", "y": 1}
        ]
        
        for i, comp in enumerate(components):
            # Draw box
            ax.add_patch(plt.Rectangle((1, comp["y"]-0.4), 6, 0.8, fill=True, facecolor='#E1F5FE', alpha=0.7, edgecolor='#1E88E5'))
            ax.text(4, comp["y"], comp["name"], ha='center', va='center', fontsize=10)
            
            # Draw arrow
            if i < len(components) - 1:
                ax.arrow(4, comp["y"]-0.4, 0, -0.2, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # Add AWS icon
        ax.text(7.5, 3, "Amazon\nSageMaker", ha='center', va='center', fontsize=12, 
               bbox=dict(facecolor='#FF9900', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Connect to tuning
        ax.arrow(7, 3, -0.5, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
        
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        st.pyplot(fig)
        
        info_box("""<b>Benefits of SageMaker Automatic Model Tuning:</b><br><br>
        
• Automates the trial-and-error process<br>
• Uses advanced Bayesian optimization<br>
• Scales to thousands of hyperparameter combinations<br>
• Supports early stopping to save on compute costs<br>
• Tracks all experiments automatically
        """, "success")
    
    custom_header("Distributed Training", "section")
    
    st.markdown("""
    **Distributed training** allows machine learning training to be split across multiple machines or GPUs,
    enabling faster training of large models and datasets.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Data Parallelism
        
        - **How it works**:
          - Splitting the dataset across multiple nodes
          - Each node has a copy of the complete model
          - Nodes process different data batches in parallel
          - Gradients are synchronized to update the model
        
        - **Best for**:
          - Large datasets
          - Models that fit in single device memory
          - Batch size that can be divided across devices
        """)
    
    with col2:
        st.markdown("""
        ### Model Parallelism
        
        - **How it works**:
          - Splitting the model across multiple nodes
          - Different parts of the model run on different devices
          - Activations are passed between devices during forward pass
          - Gradients passed between devices during backward pass
        
        - **Best for**:
          - Large models that don't fit in single device memory
          - Models with components that can be efficiently partitioned
          - Giant neural networks with billions of parameters
        """)
    
    custom_header("Early Stopping", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **Early stopping** is a technique that stops training when the model's performance on a validation set stops improving.
        This helps prevent overfitting and saves compute resources.
        
        ### How Early Stopping Works
        
        1. Monitor a validation metric during training
        2. Stop training when the metric stops improving for a specified number of iterations
        3. Use the best model from the training process
        
        ### Benefits of Early Stopping
        
        - **Prevents overfitting**: Stops before model starts memorizing training data
        - **Reduces training time**: Avoids unnecessary additional epochs
        - **Automatic optimal epoch selection**: No need to manually determine the ideal number of epochs
        - **Resource efficiency**: Saves compute resources by not running unnecessary iterations
        """)
    
    with col2:
        # Create visualization for early stopping
        epochs = np.arange(1, 101)
        train_loss = 1 / (1 + 0.1*epochs) + 0.1*np.random.randn(100)
        val_loss = 1 / (1 + 0.1*epochs) + 0.2/(1 + 0.04*epochs) + 0.1*np.random.randn(100)
        
        # Make validation loss start increasing after epoch 50
        val_loss[50:] = val_loss[50:] + 0.005 * (epochs[50:] - 50)
        
        best_epoch = np.argmin(val_loss)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax.axvline(x=best_epoch, color='g', linestyle='--', label=f'Early Stopping (Epoch {best_epoch})')
        
        # Highlight overfitting region
        ax.fill_between(epochs[best_epoch:], val_loss[best_epoch:], train_loss[best_epoch:], 
                       alpha=0.3, color='orange', label='Overfitting Region')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Early Stopping to Prevent Overfitting')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        info_box("""<b>SageMaker Early Stopping:</b><br><br>
        
SageMaker automatically implements early stopping for hyperparameter tuning jobs, saving time and resources by terminating poorly performing training jobs early.
        """, "tip")
    
    custom_header("Ensemble Learning", "section")
    
    st.markdown("""
    Ensemble learning combines multiple machine learning models to produce a more powerful model that has better predictive performance than any individual model in the ensemble.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Key Benefits of Ensemble Methods
        
        - **Improved accuracy**: Combined models often outperform individual models
        - **Reduced overfitting**: Ensembles tend to generalize better to new data
        - **Increased stability**: Less variance in predictions across different datasets
        - **Better handling of complex problems**: Capture different aspects of the data
        
        ### When to Use Ensembles
        
        - When you need the highest possible accuracy
        - When you have computational resources for multiple models
        - When individual models have complementary strengths
        - When you want to reduce the risk of selecting a poor model
        """)
    
    with col2:
        info_box("""<b>Types of Ensemble Methods:</b><br><br>
        
• <b>Stacking</b>: Trains a meta-model on the predictions of base models<br>
• <b>Bagging</b>: Trains models on random subsets of the data (e.g., Random Forest)<br>
• <b>Boosting</b>: Builds models sequentially, each correcting the errors of previous models (e.g., XGBoost)<br>
• <b>Voting</b>: Combines predictions through majority vote (classification) or averaging (regression)
        """, "info")
    
    with st.expander("Stacking (Stacked Generalization)"):
        st.markdown("""
        ### Stacking (Stacked Generalization)
        
        Stacking trains a meta-model to combine the predictions of several base models.
        
        - **How it works**:
          - Train multiple base models on the training data
          - Use predictions from these models as inputs to a meta-model
          - Meta-model learns to combine base predictions optimally
        
        - **Strengths**:
          - Can combine very different types of models
          - Often achieves higher accuracy than any single model
          - Leverages strengths of different algorithms
        
        - **Implementation**:
          - Base layer: diverse models (e.g., random forest, SVM, neural network)
          - Meta-model: simple model to combine predictions (e.g., logistic regression)
        
        **Example in Python:**
        ```python
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Train base models
        rf = RandomForestClassifier()
        gb = GradientBoostingClassifier()
        svm = SVC(probability=True)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
        
        # Get predictions from base models
        rf.fit(X_train, y_train)
        rf_preds = rf.predict_proba(X_val)
        
        gb.fit(X_train, y_train)
        gb_preds = gb.predict_proba(X_val)
        
        svm.fit(X_train, y_train)
        svm_preds = svm.predict_proba(X_val)
        
        # Combine predictions as new features
        stacked_features = np.column_stack([rf_preds, gb_preds, svm_preds])
        
        # Train meta-model
        meta_model = LogisticRegression()
        meta_model.fit(stacked_features, y_val)
        ```
        """)
    
    with st.expander("Bagging (Bootstrap Aggregating)"):
        st.markdown("""
        ### Bagging (Bootstrap Aggregating)
        
        Bagging reduces variance by training the same algorithm on different subsets of the training data.
        
        - **How it works**:
          - Create multiple training sets by sampling with replacement
          - Train the same algorithm on each sample
          - Combine predictions (majority vote or average)
        
        - **Strengths**:
          - Reduces variance and helps avoid overfitting
          - Improves stability and accuracy
          - Works well with high-variance models (e.g., decision trees)
        
        - **Examples**:
          - Random Forest (bagging with decision trees)
          - Bagged SVM
          - Extra Trees (extremely randomized trees)
        
        **Example: Random Forest**
        ```python
        from sklearn.ensemble import RandomForestClassifier
        
        # Random Forest is an implementation of bagging with decision trees
        rf = RandomForestClassifier(n_estimators=100, bootstrap=True)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        ```
        """)
    
    with st.expander("Boosting"):
        st.markdown("""
        ### Boosting
        
        Boosting builds models sequentially, with each model correcting the errors of its predecessors.
        
        - **How it works**:
          - Train models sequentially
          - Each model focuses on examples previous models got wrong
          - Weighted combination of all models
        
        - **Strengths**:
          - Often achieves best performance of ensemble methods
          - Makes weak learners stronger
          - Can capture complex patterns
        
        - **Examples**:
          - AdaBoost
          - Gradient Boosting Machines (GBM)
          - XGBoost
          - LightGBM
          - CatBoost
        
        **Example: XGBoost**
        ```python
        from xgboost import XGBClassifier
        
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        ```
        """)

# Hyperparameters tab
with tabs[6]:
    # Hyperparameters content removed as it's combined with the Model Training tab
    custom_header("Hyperparameter Optimization Techniques")
    
    st.markdown("""
    Beyond basic tuning, there are several advanced techniques for optimizing hyperparameters that can help improve model performance and training efficiency.
    """)
    
    custom_header("Warm Start Tuning", "sub")
    
    st.markdown("""
    Amazon SageMaker supports **Warm Start Tuning**, which lets you start a new hyperparameter tuning job using one or more previous tuning jobs as a starting point.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Benefits of Warm Start
        
        - **Transfer learning for hyperparameters**
          - Use knowledge from previous tuning jobs
          - Focus on promising regions of the hyperparameter space
        
        - **Time and cost efficiency**
          - Reduces the number of jobs needed to find optimal configuration
          - Particularly valuable for expensive models
        
        - **Iterative refinement**
          - Gradually narrow down search space
          - Fine-tune hyperparameters in stages
        """)
    
    with col2:
        # Create a visualization of warm start benefit
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a contour plot representing hyperparameter space
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = 3*(1-X)**2 * np.exp(-X**2 - (Y+1)**2) - 10*(X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2) - 1/3 * np.exp(-(X+1)**2 - Y**2)
        
        # Plot contour
        contour = ax.contourf(X, Y, Z, 15, cmap='viridis', alpha=0.7)
        
        # Add points for cold start (spread across space)
        cold_x = np.random.uniform(0, 10, 20)
        cold_y = np.random.uniform(0, 10, 20)
        ax.scatter(cold_x, cold_y, color='red', marker='x', label='Cold Start Evaluations')
        
        # Add points for warm start (concentrated in promising area)
        warm_x = np.random.normal(7, 1, 20)
        warm_y = np.random.normal(3, 1, 20)
        ax.scatter(warm_x, warm_y, color='white', marker='o', label='Warm Start Evaluations')
        
        # Circle the optimal region
        optimal = plt.Circle((7, 3), 1.5, fill=False, color='yellow', linewidth=2, linestyle='--', label='Optimal Region')
        ax.add_patch(optimal)
        
        ax.set_xlabel('Hyperparameter 1')
        ax.set_ylabel('Hyperparameter 2')
        ax.set_title('Warm Start vs Cold Start Hyperparameter Tuning')
        ax.legend()
        
        st.pyplot(fig)
    
    custom_header("Transfer Learning and Fine-Tuning", "section")
    
    st.markdown("""
    Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a different task.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Benefits of Transfer Learning
        
        - **Reduced training time**
          - Start with pre-learned features
          - Requires less data for your specific task
        
        - **Better performance**
          - Leverage knowledge from related domains
          - Particularly useful for image and text data
        
        - **Lower resource requirements**
          - Doesn't require training large models from scratch
          - More accessible for smaller teams or projects
        """)
    
    with col2:
        st.markdown("""
        ### When to Use Transfer Learning
        
        - **Limited training data**
          - When you have insufficient data for your task
        
        - **Related tasks**
          - Source and target tasks share common features
        
        - **Specialized domains**
          - Medical imaging, satellite imagery, etc.
        
        - **Complex models**
          - Large language models, vision transformers
        """)
    
    st.markdown("""
    ### Transfer Learning with SageMaker JumpStart
    
    SageMaker JumpStart provides pre-trained models that can be fine-tuned for your specific task:
    
    1. **Select a pre-trained model** suitable for your use case
    2. **Fine-tune the model** with your domain-specific data
    3. **Deploy the customized model** for inference
    
    JumpStart includes models for:
    - Computer vision
    - Natural language processing
    - Tabular data analysis
    - Time series forecasting
    """)
    
    st.code("""
    # Import SageMaker and JumpStart modules
    import sagemaker
    from sagemaker.jumpstart.model import JumpStartModel
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Select a pre-trained model
    model_id = "huggingface-sst2-distilbert-base-uncased"
    
    # Create model instance
    model = JumpStartModel(model_id=model_id)
    
    # Fine-tune the model with your data
    model.fit(
        {"train": "s3://bucket/path/to/train/data",
         "validation": "s3://bucket/path/to/validation/data"}
    )
    
    # Deploy the fine-tuned model
    predictor = model.deploy()
    """, language="python")
    
    custom_header("Learning Rate Scheduling", "section")
    
    st.markdown("""
    Learning rate scheduling adjusts the learning rate during training to improve convergence and model performance.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Common Learning Rate Schedules
        
        - **Step Decay**
          - Reduce learning rate by a factor at specific epochs
          - Example: Halve the learning rate every 10 epochs
        
        - **Exponential Decay**
          - Learning rate decreases exponentially
          - LR = initial_lr * e^(-k*t)
        
        - **Cosine Annealing**
          - Learning rate follows a cosine curve
          - Smooth transition from high to low values
        
        - **Cyclical Learning Rates**
          - Learning rate cycles between bounds
          - Helps escape local minima
        
        - **One-Cycle Policy**
          - Learning rate first increases then decreases
          - Helps faster convergence
        """)
    
    with col2:
        # Create learning rate schedule visualization
        epochs = np.arange(0, 100)
        
        # Different learning rate schedules
        step_decay = 0.1 * np.power(0.5, np.floor(epochs / 20))
        exp_decay = 0.1 * np.exp(-0.01 * epochs)
        cosine_decay = 0.1 * (1 + np.cos(np.pi * epochs / 100)) / 2
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, step_decay, label="Step Decay", linewidth=2)
        ax.plot(epochs, exp_decay, label="Exponential Decay", linewidth=2)
        ax.plot(epochs, cosine_decay, label="Cosine Annealing", linewidth=2)
        
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedules")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        st.pyplot(fig)
    
    info_box("""
    <b>Best Practices for Hyperparameter Optimization:</b><br><br>
    
    1. Start with a coarse search to identify promising regions, then refine.<br>
    2. Use logarithmic scales for parameters that span multiple orders of magnitude (e.g., learning rate).<br>
    3. Monitor both training and validation metrics to detect overfitting.<br>
    4. Consider the computational budget when designing the search strategy.<br>
    5. Use warm start tuning when exploring related model architectures.<br>
    6. Save hyperparameters from successful runs for future reference.
    """, "tip")

# Knowledge Check tab
with tabs[7]:
    custom_header("Knowledge Check")
    
    st.markdown("""
    Test your understanding of Domain 2: ML Model Development concepts with this quiz.
    Select the best answer for each question.
    """)
    
    # Define quiz questions
    questions = [
        {
            "question": "Which Amazon SageMaker model development method requires the least amount of code?",
            "options": ["Using built-in algorithms", "Bring your own script", "Bring your own container", "SageMaker Processing"],
            "correct": "Using built-in algorithms"
        },
        {
            "question": "What is the primary benefit of using SageMaker automatic model tuning?",
            "options": ["It automatically selects the best algorithm for your data", "It finds optimal hyperparameter values by running multiple training jobs", "It reduces the cost of model training", "It automatically deploys the model to production"],
            "correct": "It finds optimal hyperparameter values by running multiple training jobs"
        },
        {
            "question": "Which feature of Amazon Bedrock allows foundation models to incorporate information from your data sources?",
            "options": ["Foundation model fine-tuning", "Continued pretraining", "Retrieval Augmented Generation (RAG)", "Model distillation"],
            "correct": "Retrieval Augmented Generation (RAG)"
        },
        {
            "question": "Which hyperparameter tuning strategy builds a probabilistic model of the objective function based on previous evaluations?",
            "options": ["Grid search", "Random search", "Bayesian optimization", "Evolutionary algorithms"],
            "correct": "Bayesian optimization"
        },
        {
            "question": "In distributed training, what is the main difference between data parallelism and model parallelism?",
            "options": ["Data parallelism splits the dataset across devices while model parallelism splits the model", "Data parallelism is for regression while model parallelism is for classification", "Data parallelism uses CPUs while model parallelism uses GPUs", "Data parallelism is synchronous while model parallelism is asynchronous"],
            "correct": "Data parallelism splits the dataset across devices while model parallelism splits the model"
        }
    ]
    
    # Quiz logic
    if not st.session_state['quiz_attempted']:
        # Form to collect answers
        with st.form("quiz_form"):
            st.markdown("### Answer the following questions:")
            
            user_answers = {}
            
            # Display 5 questions
            for i, q in enumerate(questions):
                st.markdown(f"**Question {i+1}:** {q['question']}")
                user_answers[i] = st.radio(
                    f"Select your answer for question {i+1}:",
                    q['options'],
                    index=None,
                    key=f"q{i}"
                )
                st.markdown("---")
            
            # Submit button
            submitted = st.form_submit_button("Submit Quiz")
            
            if submitted:
                score = 0
                for i, q in enumerate(questions):
                    if user_answers[i] == q['correct']:
                        score += 1
                
                st.session_state['quiz_score'] = score
                st.session_state['quiz_attempted'] = True
                st.session_state['answers'] = user_answers
                st.rerun()
    else:
        # Show results
        score = st.session_state['quiz_score']
        user_answers = st.session_state.get('answers', {})
        
        st.markdown(f"### Your Score: {score}/{len(questions)}")
        
        if score == len(questions):
            st.success("🎉 Perfect score! You've mastered Domain 2 concepts!")
        elif score >= len(questions) * 0.8:
            st.success("👍 Great job! You have a strong understanding of the material.")
        elif score >= len(questions) * 0.6:
            st.warning("🔍 Good effort! Review the concepts you missed to strengthen your understanding.")
        else:
            st.error("📚 You might want to revisit the material to reinforce your understanding.")
        
        # Show detailed results
        st.markdown("### Question Review")
        
        for i, q in enumerate(questions):
            is_correct = user_answers[i] == q['correct']
            result_container = st.container()
            
            with result_container:
                st.markdown(f"**Question {i+1}**: {q['question']}")
                st.markdown(f"Your answer: **{user_answers[i]}**")
                
                if is_correct:
                    st.markdown(f"**✅ Correct!**")
                else:
                    st.markdown(f"**❌ Incorrect. The correct answer is: {q['correct']}**")
                
                st.markdown("---")
        
        # Option to retake the quiz
        if st.button("Try Again"):
            st.session_state['quiz_attempted'] = False
            st.rerun()

# Resources tab
with tabs[8]:
    custom_header("Additional Resources")
    
    st.markdown("""
    Explore these resources to deepen your understanding of ML Model Development.
    These materials provide additional context and practical guidance for implementing the concepts covered in this module.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AWS Documentation")
        st.markdown("""
        - [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
        - [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
        - [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html)
        - [SageMaker Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
        - [Distributed Training with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)
        - [SageMaker Built-in Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
        - [Transfer Learning with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/transfer-learning.html)
        """)
        
        st.markdown("### AWS Blog Posts")
        st.markdown("""
        - [Building, training, and deploying ML models with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/building-training-and-deploying-machine-learning-models-with-amazon-sagemaker/)
        - [Building RAG applications with Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/build-rag-applications-with-amazon-bedrock/)
        - [Hyperparameter optimization best practices](https://aws.amazon.com/blogs/machine-learning/hyperparameter-optimization-best-practices/)
        - [Fine-tuning foundation models with Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/fine-tune-foundation-models-with-amazon-bedrock/)
        - [Training ML at scale with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/train-ml-at-scale-with-amazon-sagemaker/)
        """)
    
    with col2:
        st.markdown("### Training Courses")
        st.markdown("""
        - [Introduction to Amazon SageMaker](https://www.aws.training/Details/eLearning?id=47225)
        - [Practical Data Science with Amazon SageMaker](https://www.coursera.org/specializations/practical-data-science)
        - [Getting Started with AWS Machine Learning](https://www.coursera.org/learn/aws-machine-learning)
        - [AWS Machine Learning Foundations](https://www.udacity.com/course/aws-machine-learning-foundations--ud090)
        - [Deep Learning on AWS](https://aws.amazon.com/training/learn-about/deep-learning/)
        """)
        
        st.markdown("### Tools and Services")
        st.markdown("""
        - [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
        - [Amazon Bedrock](https://aws.amazon.com/bedrock/)
        - [Amazon SageMaker JumpStart](https://aws.amazon.com/sagemaker/jumpstart/)
        - [AWS Deep Learning Containers](https://aws.amazon.com/machine-learning/containers/)
        - [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/)
        - [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/)
        """)
    
    custom_header("AWS Machine Learning Resources", "sub")
    
    st.markdown("""
    ### Hands-on Examples and Workshops
    
    - [Amazon SageMaker Examples GitHub Repository](https://github.com/aws/amazon-sagemaker-examples)
    - [AWS ML Workshop](https://github.com/aws-samples/aws-machine-learning-workshop)
    - [SageMaker Workshop](https://sagemaker-workshop.com/)
    - [Bedrock Workshop](https://github.com/aws-samples/amazon-bedrock-workshop)
    
    ### AWS ML Certification Resources
    
    - [AWS Certified Machine Learning - Specialty](https://aws.amazon.com/certification/certified-machine-learning-specialty/)
    - [AWS Machine Learning Exam Guide](https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Exam-Guide.pdf)
    - [AWS Machine Learning Practice Questions](https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Sample-Questions.pdf)
    - [AWS Machine Learning Ramp-Up Guide](https://d1.awsstatic.com/training-and-certification/docs-ml/AWSTrainingCertification_MachineLearning_Ramp-Up_Guide.pdf)
    """)
    
    with st.expander("Amazon SageMaker Example Notebooks"):
        st.markdown("""
        Amazon SageMaker Example Notebooks provide end-to-end examples of various machine learning workflows.
        
        ### Featured Examples:
        
        - **Introduction to SageMaker**
          - [SageMaker Hello World](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone.ipynb)
          - [SageMaker with Built-in Algorithms](https://github.com/aws/amazon-sagemaker-examples/tree/main/introduction_to_amazon_algorithms)
        
        - **Hyperparameter Tuning**
          - [HPO for XGBoost](https://github.com/aws/amazon-sagemaker-examples/blob/main/hyperparameter_tuning/xgboost_direct_marketing/hpo_xgboost_direct_marketing.ipynb)
          - [HPO for Image Classification](https://github.com/aws/amazon-sagemaker-examples/blob/main/hyperparameter_tuning/image_classification_warmstart/hpo_image_classification_warmstart.ipynb)
        
        - **Distributed Training**
          - [Distributed TensorFlow](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_distributed_mnist.ipynb)
          - [Distributed PyTorch](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/pytorch_mnist/pytorch_mnist.ipynb)
        
        - **Transfer Learning**
          - [Transfer Learning with TensorFlow](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/transfer_learning_with_resnet50/transfer_learning_with_resnet50.ipynb)
          - [Transfer Learning with HuggingFace and BERT](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/huggingface_sentiment_ipynb)
        """)

# Footer
st.markdown("---")
col1, col2 = st.columns([1, 5])
with col1:
    st.image("images/aws_logo.png", width=70)
with col2:
    st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")
