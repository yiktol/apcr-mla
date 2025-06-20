
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
if 'visited_Model_Generalization' not in st.session_state:
    st.session_state['visited_Model_Generalization'] = False
if 'visited_Model_Evaluation' not in st.session_state:
    st.session_state['visited_Model_Evaluation'] = False
if 'visited_Classification_Metrics' not in st.session_state:
    st.session_state['visited_Classification_Metrics'] = False
if 'visited_Regression_Metrics' not in st.session_state:
    st.session_state['visited_Regression_Metrics'] = False
if 'visited_Model_Optimization' not in st.session_state:
    st.session_state['visited_Model_Optimization'] = False

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

# Function to create a metric card
def metric_card(title, description):
    st.markdown(f"""
    <div class="metric-card">
        <strong>{title}</strong><br>
        {description}
    </div>
    """, unsafe_allow_html=True)

# Function to reset session - same as Domain 1
def reset_session():
    st.session_state['quiz_score'] = 0
    st.session_state['quiz_attempted'] = False
    st.session_state['name'] = ""
    st.session_state['visited_Model_Generalization'] = False
    st.session_state['visited_Model_Evaluation'] = False
    st.session_state['visited_Classification_Metrics'] = False
    st.session_state['visited_Regression_Metrics'] = False
    st.session_state['visited_Model_Optimization'] = False
    st.rerun()

# Sidebar for session management - similar to Domain 1
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
        visited_pages = [page for page in ["Model_Generalization", "Model_Evaluation", "Classification_Metrics", "Regression_Metrics", "Model_Optimization"] 
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
        - Understand model generalization concepts
        - Learn about regularization techniques
        - Evaluate model performance with metrics
        - Compare classification and regression metrics
        - Apply model optimization techniques
        """)

# Main content with tabs
tabs = st.tabs([
    "🏠 Home", 
    "📊 Model Generalization", 
    "📈 Model Evaluation", 
    "🔄 Classification Metrics", 
    "📉 Regression Metrics", 
    "⚙️ Model Optimization", 
    "❓ Quiz", 
    "📚 Resources"
])

# Home tab
with tabs[0]:
    custom_header("AWS Partner Certification Readiness")
    st.markdown("## Machine Learning Engineer - Associate")
    
    st.markdown("### Domain 2: ML Model Development - Task 2.3: Analyze Model Performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        info_box("""
        This interactive e-learning application covers Task 2.3 from Domain 2 of the AWS Machine Learning Engineer - Associate certification.
        
        Domain 2 focuses on **ML Model Development**, with Task 2.3 specifically covering how to **Analyze Model Performance**.
        
        Navigate through the content using the tabs above to learn about:
        - Model Generalization
        - Model Evaluation Techniques
        - Classification Metrics
        - Regression Metrics
        - Model Optimization Techniques
        
        Test your knowledge with the quiz when you're ready!
        """, "info")
        
        st.markdown("### Learning Outcomes")
        st.markdown("""
        By the end of this module, you will be able to:
        - Identify underfitting and overfitting in ML models
        - Understand and apply regularization techniques
        - Choose appropriate evaluation metrics for classification and regression problems
        - Interpret confusion matrices, ROC curves, and other metrics
        - Implement optimization techniques like gradient descent
        - Use Amazon SageMaker tools for model evaluation and debugging
        """)
    
    with col2:
        st.image("images/mla_badge_big.png", width=250)
        
        if st.session_state['quiz_attempted']:
            st.success(f"Current Quiz Score: {st.session_state['quiz_score']}/5")
        
        st.info("Use the tabs above to navigate through different sections!")
        
    st.markdown("---")
    
    st.markdown("### Machine Learning Lifecycle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Data Processing")
        st.markdown("""
        - Data preparation
        - Feature engineering
        - Transformation
        - Dataset splitting
        """)
    
    with col2:
        st.markdown("#### Model Development")
        st.markdown("""
        - **Model training** 
        - **Model evaluation** ← You are here
        - Hyperparameter tuning
        - Model selection
        """)
    
    with col3:
        st.markdown("#### Model Deployment")
        st.markdown("""
        - Deployment strategies
        - Inference endpoints
        - Monitoring
        - Maintenance
        """)

# Model Generalization tab
with tabs[1]:
    # Mark as visited
    st.session_state['visited_Model_Generalization'] = True
    
    custom_header("Model Generalization")
    
    st.markdown("""
    Model generalization refers to how well a machine learning model applies to new, unseen data. 
    The goal of any machine learning model is to learn patterns from training data that will generalize well to new data.
    """)
    
    st.markdown("### Common Model Generalization Problems")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Underfitting")
        st.image("images/underfitting.png", caption="Underfitting Example")
        st.markdown("""
        - Model is too simple to capture the underlying pattern
        - Poor performance on both training and test data
        - High bias, low variance
        - Symptoms: High error on training data
        """)
        
        st.markdown("##### How to fix:")
        st.markdown("""
        - Increase model complexity
        - Add more features
        - Reduce regularization
        - Train for more epochs or iterations
        """)
    
    with col2:
        st.markdown("#### Overfitting")
        st.image("images/overfitting.png", caption="Overfitting Example")
        st.markdown("""
        - Model learns noise instead of the underlying pattern
        - Good performance on training data, poor on test data
        - Low bias, high variance
        - Symptoms: Large gap between training and validation error
        """)
        
        st.markdown("##### How to fix:")
        st.markdown("""
        - Apply regularization (L1/L2, dropout)
        - Simplify model (fewer features, smaller n-grams)
        - Gather more training data
        - Use early stopping during training
        """)
    
    with col3:
        st.markdown("#### Appropriate Fitting")
        st.image("images/good_fit.png", caption="Good Fit Example")
        st.markdown("""
        - Model captures the underlying pattern without noise
        - Good performance on both training and test data
        - Balance between bias and variance
        - Symptoms: Similar errors on training and validation data
        """)
        
        st.markdown("##### How to achieve:")
        st.markdown("""
        - Careful feature selection
        - Proper hyperparameter tuning
        - Cross-validation
        - Regular evaluation of model performance
        """)
    
    custom_header("Regularization Techniques", "sub")
    
    st.markdown("""
    Regularization is a set of techniques used to prevent overfitting by adding a penalty term to the loss function during training.
    This penalty discourages overly complex models that might fit the training data too closely.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dropout Regularization")
        st.image("images/dropout.png", caption="Dropout Regularization in Neural Networks")
        st.markdown("""
        Dropout is a regularization technique for neural networks:
        
        - **How it works**: Randomly deactivates neurons during training
        - **Effect**: Forces network to be more robust and not rely on any one neuron
        - **Implementation**: Apply dropout layers between dense layers in network
        - **Dropout rate**: Typically 0.2 to 0.5 (20-50% of neurons dropped)
        - **Training vs Inference**: Only active during training, not during inference
        """)
    
    with col2:
        st.markdown("### L1/L2 Regularization")
        st.image("images/l1_l2.png", caption="L1 vs L2 Regularization")
        
        st.markdown("#### L1 Regularization (Lasso)")
        st.markdown("""
        - Adds sum of absolute values of weights to loss function
        - Encourages sparse models (many weights become exactly zero)
        - Performs feature selection implicitly
        - Formula: Loss = Original_Loss + λ * Σ|w|
        - Best when you suspect many features are irrelevant
        """)
        
        st.markdown("#### L2 Regularization (Ridge)")
        st.markdown("""
        - Adds sum of squared weights to loss function
        - Shrinks all weights toward zero but not exactly zero
        - More stable when features are correlated
        - Formula: Loss = Original_Loss + λ * Σw²
        - Best for most general use cases
        """)
    
    st.markdown("### Key Differences Between L1 and L2 Regularization")
    col1, col2 = st.columns(2)
    
    with col1:
        info_box("""
        **L1 (Lasso) Regularization:**
        
        - Creates sparse models
        - Some coefficients become exactly zero
        - Effectively performs feature selection
        - Better for high-dimensional data with irrelevant features
        - Less stable with correlated features
        """, "info")
    
    with col2:
        info_box("""
        **L2 (Ridge) Regularization:**
        
        - Shrinks all coefficients proportionally
        - No coefficients become exactly zero
        - Handles correlated features better
        - More stable solution in general
        - Better when all features are potentially useful
        """, "info")
    
    st.markdown("### Implementation Example")
    
    with st.expander("Code Example: Implementing Regularization in TensorFlow"):
        st.code("""
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.regularizers import l1, l2, l1_l2

        # L1 Regularization (Lasso)
        model_l1 = tf.keras.Sequential([
            Dense(128, activation='relu', kernel_regularizer=l1(0.01), input_shape=(input_dim,)),
            Dense(64, activation='relu', kernel_regularizer=l1(0.01)),
            Dense(1, activation='sigmoid')
        ])

        # L2 Regularization (Ridge)
        model_l2 = tf.keras.Sequential([
            Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_dim,)),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1, activation='sigmoid')
        ])

        # Dropout Regularization
        model_dropout = tf.keras.Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),  # 30% dropout rate
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        # Combined L1 and L2 (Elastic Net)
        model_combined = tf.keras.Sequential([
            Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        """, language="python")
    
    with st.expander("Real-world Application: Regularization in Image Classification"):
        st.markdown("""
        ### Case Study: Image Classification with Convolutional Neural Networks
        
        In a convolutional neural network (CNN) for image classification, regularization becomes crucial to prevent overfitting, especially when working with limited datasets.
        
        **Problem**: 
        A model trained to classify medical images was performing excellently on training data (99% accuracy) but poorly on validation data (75% accuracy).
        
        **Solution**:
        1. **Added Dropout layers** (0.25) after each max pooling layer
        2. **Applied L2 regularization** to convolutional layers
        3. **Implemented data augmentation** (rotation, flipping, zoom) to effectively increase dataset size
        4. **Used early stopping** to halt training when validation accuracy stopped improving
        
        **Result**:
        - Training accuracy: 92%
        - Validation accuracy: 90%
        - Test accuracy: 89%
        
        Despite lower training accuracy, the model generalized much better to new data, which was the primary goal.
        
        This example demonstrates how regularization helps achieve a better balance between bias and variance, leading to models that perform reliably on unseen data.
        """)

# Model Evaluation tab
with tabs[2]:
    # Mark as visited
    st.session_state['visited_Model_Evaluation'] = True
    
    custom_header("Model Evaluation")
    
    st.markdown("""
    Model evaluation is the process of assessing the performance of a machine learning model using various metrics and techniques.
    Selecting appropriate evaluation metrics is crucial as it helps determine if your model is solving the business problem effectively.
    """)
    
    st.markdown("### Importance of Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        Proper model evaluation helps you:
        
        1. **Determine model effectiveness** for the business problem
        2. **Compare different models** to select the best one
        3. **Identify areas for improvement** in your model
        4. **Prevent overfitting** by evaluating on validation/test data
        5. **Set realistic expectations** for stakeholders about model performance
        6. **Justify the cost** and resources invested in developing the model
        """)
    
    with col2:
        info_box("""
        **Best Practices for Model Evaluation:**
        
        - Always split your data into training, validation, and test sets
        - Choose evaluation metrics based on your business problem
        - Consider class imbalance when evaluating classification models
        - Perform cross-validation for more robust performance estimates
        - Evaluate both model performance and computational efficiency
        - Look beyond single metrics - consider the full picture
        """, "tip")
    
    st.markdown("### Types of Evaluation Metrics")
    
    st.markdown("""
    The choice of evaluation metrics depends on the type of machine learning problem you're solving:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Classification Metrics")
        st.markdown("""
        - **Accuracy**: Proportion of correct predictions
        - **Precision**: Proportion of true positives among positive predictions
        - **Recall**: Proportion of true positives captured by model
        - **F1 Score**: Harmonic mean of precision and recall
        - **AUC-ROC**: Area under the Receiver Operating Characteristic curve
        - **Confusion Matrix**: Table showing prediction outcomes
        """)
    
    with col2:
        st.markdown("#### Regression Metrics")
        st.markdown("""
        - **Mean Squared Error (MSE)**: Average squared difference between predictions and actual values
        - **Root Mean Squared Error (RMSE)**: Square root of MSE
        - **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values
        - **R-squared**: Proportion of variance explained by the model
        - **Adjusted R-squared**: R-squared adjusted for number of predictors
        """)
    
    custom_header("Evaluation Techniques", "sub")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cross-Validation")
        st.image("images/cross_validation.png", caption="K-Fold Cross-Validation")
        st.markdown("""
        Cross-validation is a technique that provides a more reliable estimate of model performance by using multiple train-test splits:
        
        - **K-Fold CV**: Splits data into k folds, trains on k-1 folds and validates on the remaining fold
        - **Stratified K-Fold**: Maintains class distribution in each fold (for classification)
        - **Leave-One-Out**: Uses all but one sample for training, validates on the single left-out sample
        - **Time Series CV**: Respects time order for time series data
        
        Cross-validation helps detect overfitting and provides more robust performance estimates.
        """)
    
    with col2:
        st.markdown("### Learning Curves")
        st.image("images/learning_curves.png", caption="Learning Curves Example")
        st.markdown("""
        Learning curves plot model performance against training set size:
        
        - **Training curve**: Performance on training data
        - **Validation curve**: Performance on validation data
        
        Learning curves help diagnose:
        
        - **Underfitting**: Both curves show poor performance
        - **Overfitting**: Large gap between training and validation curve
        - **Appropriate fit**: Converging curves with good performance
        - **Need for more data**: Improving validation curve with more training examples
        """)
    
    st.markdown("### Amazon SageMaker Tools for Evaluation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### SageMaker Clarify")
        # st.image("https://d1.awsstatic.com/SageMaker-Clarify-How-it-works.3e1739db812b5d9ad35ec0f2164769565df7c749.png", width=300)
        st.markdown("""
        SageMaker Clarify provides:
        
        - Data bias detection
        - Model bias detection
        - Feature importance
        - Partial dependence plots
        - Model explainability
        - Automated reports
        """)
    
    with col2:
        st.markdown("#### SageMaker Debugger")
        # st.image("https://d1.awsstatic.com/rt21-mujiba/demo3-debug.2376975bbf5eb4b9e05d6e9f494c0671550ee374.png", width=300)
        st.markdown("""
        SageMaker Debugger provides:
        
        - Real-time training metrics
        - Automatic error detection
        - Visualization of training process
        - Performance profiling
        - Resource utilization tracking
        - Debug hooks for model inspection
        """)
    
    with col3:
        st.markdown("#### SageMaker Experiments")
        # st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/experiments/experiment1.png", width=300)
        st.markdown("""
        SageMaker Experiments helps:
        
        - Organize training iterations
        - Track inputs, parameters, and results
        - Compare multiple experiments
        - Visualize performance metrics
        - Identify best-performing models
        - Reproduce experiments
        """)
    
    with st.expander("SageMaker Experiments Components"):
        st.markdown("""
        ### Key Components of Amazon SageMaker Experiments
        
        SageMaker Experiments consists of several core components that work together to help you track and evaluate model performance:
        
        1. **Experiment**: An ML problem that you want to solve. Each experiment consists of a collection of trials.
        
        2. **Trial**: An iteration of a data science workflow related to an experiment. Each trial consists of several trial components.
        
        3. **Trial Component**: A specific stage in a given trial. For example:
           - Data preprocessing stage
           - Model training stage
           - Data post-processing stage
        
        4. **Tracker**: A mechanism that records metadata about a trial component, including:
           - Parameters
           - Inputs
           - Outputs
           - Artifacts
           - Metrics
        
        5. **Tracking API**: Records metadata for trials and experiments, particularly useful for computer vision and NLP tasks.
        
        SageMaker Experiments supports both:
        
        - **Automated tracking**: Automatically tracks jobs like Amazon SageMaker Autopilot, training, batch transform, and processing jobs.
        
        - **Manual tracking**: Provides tracking APIs for recording workflows running locally on SageMaker Studio notebooks.
        
        By organizing your machine learning work into experiments, trials, and components, you can easily compare different approaches, track progress, and identify the most successful models.
        """)
    
    st.markdown("### Shadow Testing for Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        Amazon SageMaker Shadow Testing allows you to evaluate new model versions against the current production model without impacting users:
        
        - **How it works**: Deploy new model version alongside production model
        - **Traffic routing**: Send a percentage of live traffic to both models
        - **Response handling**: Only production model responses go to users
        - **Metrics comparison**: Compare performance metrics between models
        - **Safe testing**: No impact on end users during evaluation
        """)
    
    with col2:
        info_box("""
        **Benefits of Shadow Testing:**
        
        - Test with real production traffic
        - Identify potential issues before full deployment
        - Compare performance metrics side by side
        - Evaluate both model accuracy and infrastructure performance
        - Minimize risk when updating models
        - Make data-driven decisions for model promotion
        """, "success")
        
        st.markdown("""
        After successful shadow testing, you can promote the shadow variant to become the new production variant with minimal disruption.
        """)

# Classification Metrics tab
with tabs[3]:
    # Mark as visited
    st.session_state['visited_Classification_Metrics'] = True
    
    custom_header("Classification Metrics")
    
    st.markdown("""
    Classification metrics are used to evaluate models that predict categorical outcomes.
    Each metric provides different insights into model performance and is suitable for different scenarios.
    """)
    
    custom_header("Confusion Matrix", "sub")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/confusion_matrix.png", caption="Confusion Matrix", width=500)
        
        st.markdown("""
        A confusion matrix is a table that summarizes the prediction results of a classification model:
        
        - **True Positives (TP)**: Correct positive predictions
        - **False Positives (FP)**: Incorrect positive predictions (Type I error)
        - **True Negatives (TN)**: Correct negative predictions
        - **False Negatives (FN)**: Incorrect negative predictions (Type II error)
        
        The confusion matrix forms the basis for many classification metrics.
        """)
    
    with col2:
        st.markdown("### Example")
        st.markdown("""
        In a cat image classifier:
        
        - **True Positive**: Model correctly identifies a cat image
        - **False Positive**: Model incorrectly identifies a dog as a cat
        - **True Negative**: Model correctly identifies a non-cat image
        - **False Negative**: Model incorrectly identifies a cat as non-cat
        
        The confusion matrix helps visualize where the model makes mistakes and what types of errors it makes.
        """)
        
        info_box("""
        **When to use a confusion matrix:**
        
        - To get a detailed breakdown of model errors
        - When different types of errors have different consequences
        - To calculate other metrics like precision and recall
        - When working with imbalanced datasets
        """, "tip")
    
    custom_header("Accuracy", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Accuracy** is the proportion of correct predictions among the total number of predictions:
        
        **Formula**: 
        ```
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        ```
        
        **When to use**:
        - When classes are balanced
        - When all types of errors are equally important
        
        **Limitations**:
        - Can be misleading for imbalanced datasets
        - Doesn't distinguish between different types of errors
        """)
        
        # Example calculation
        st.markdown("### Example Calculation")
        st.markdown("""
        Suppose we have:
        - TP = 80
        - TN = 90
        - FP = 10
        - FN = 20
        
        Accuracy = (80 + 90) / (80 + 90 + 10 + 20) = 170 / 200 = 0.85 or 85%
        """)
    
    with col2:
        # Create example plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Dummy data for visualization
        labels = ['True', 'False']
        data = [[80, 10], [20, 90]]
        
        sns.heatmap(data, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=['Predicted Positive', 'Predicted Negative'],
                    yticklabels=['Actually Positive', 'Actually Negative'])
        
        ax.set_title('Example Confusion Matrix for Accuracy Calculation')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        st.markdown("""
        In this example:
        - **85%** of all predictions are correct
        - But this doesn't tell us how well the model performs on each class
        """)
    
    custom_header("Precision and Recall", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Precision")
        st.markdown("""
        **Precision** measures the accuracy of positive predictions:
        
        **Formula**: 
        ```
        Precision = TP / (TP + FP)
        ```
        
        **When to use**:
        - When false positives are costly
        - When you want to be confident in positive predictions
        - Example: Spam detection (avoid marking important emails as spam)
        
        **Interpretation**:
        - Higher precision means fewer false positives
        - "When the model says positive, how often is it right?"
        """)
    
    with col2:
        st.markdown("### Recall (Sensitivity)")
        st.markdown("""
        **Recall** measures the ability to find all positive instances:
        
        **Formula**: 
        ```
        Recall = TP / (TP + FN)
        ```
        
        **When to use**:
        - When false negatives are costly
        - When you need to catch all positive cases
        - Example: Disease detection (don't miss any cases)
        
        **Interpretation**:
        - Higher recall means fewer false negatives
        - "Of all actual positives, how many did the model identify?"
        """)
    
    st.markdown("### Precision vs. Recall Trade-off")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create precision-recall trade-off curve
        thresholds = np.linspace(0, 1, 100)
        precision = 1 / (1 + np.exp(-(thresholds*10 - 5)))
        recall = 1 - thresholds**1.5
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, precision, 'b-', label='Precision')
        ax.plot(thresholds, recall, 'r-', label='Recall')
        ax.set_xlabel('Classification Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision-Recall Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        info_box("""
        **Understanding the Trade-off:**
        
        - Increasing the classification threshold typically:
          - Increases precision (fewer false positives)
          - Decreases recall (more false negatives)
        
        - Lowering the threshold typically:
          - Decreases precision (more false positives)
          - Increases recall (fewer false negatives)
        
        The business context determines which is more important:
        
        - **High Precision**: When false positives are expensive or harmful
        - **High Recall**: When false negatives are expensive or harmful
        """, "info")
    
    custom_header("F1 Score", "section")
    
    st.markdown("""
    The **F1 Score** combines precision and recall into a single metric. It is the harmonic mean of precision and recall,
    giving equal weight to both metrics.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Formula**:
        ```
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        ```
        
        **When to use**:
        - When you need a balance between precision and recall
        - With imbalanced datasets where accuracy is misleading
        - When both false positives and false negatives are important
        
        **Interpretation**:
        - F1 Score ranges from 0 (worst) to 1 (best)
        - A high F1 score indicates both good precision and good recall
        - F1 score is lower than the arithmetic mean when precision and recall differ significantly
        """)
    
    with col2:
        # Create example for F1 scores with different precision/recall combinations
        data = {
            'Model': ['Model A', 'Model B', 'Model C', 'Model D'],
            'Precision': [0.9, 0.5, 0.8, 0.95],
            'Recall': [0.1, 0.5, 0.8, 0.4],
        }
        
        df = pd.DataFrame(data)
        df['F1 Score'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df['Model']))
        width = 0.25
        
        ax.bar(x - width, df['Precision'], width, label='Precision')
        ax.bar(x, df['Recall'], width, label='Recall')
        ax.bar(x + width, df['F1 Score'], width, label='F1 Score')
        
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall and F1 Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'])
        ax.legend()
        
        st.pyplot(fig)
        
        st.markdown("""
        Note how Model C with balanced precision and recall has the highest F1 score, 
        while Models A and D with imbalanced precision and recall have lower F1 scores despite having some very high individual metrics.
        """)
    
    custom_header("AUC-ROC Curve", "section")
    
    st.markdown("""
    The **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)** is a popular metric for binary classification problems.
    It measures the model's ability to discriminate between positive and negative classes across various threshold settings.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create ROC curve example
        fpr = np.linspace(0, 1, 100)
        
        # ROC curves for different models
        tpr1 = np.power(fpr, 0.3)  # good model
        tpr2 = fpr  # random model
        tpr3 = np.power(fpr, 0.7)  # intermediate model
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr1, 'b-', label='Good Model (AUC=0.87)')
        ax.plot(fpr, tpr3, 'g-', label='Fair Model (AUC=0.67)')
        ax.plot(fpr, tpr2, 'r--', label='Random Model (AUC=0.50)')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Examples')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        **What ROC curves show:**
        
        - **X-axis**: False Positive Rate (1 - Specificity)
        - **Y-axis**: True Positive Rate (Sensitivity/Recall)
        - **Each point**: Model performance at a specific threshold
        
        **AUC (Area Under Curve):**
        
        - Ranges from 0 to 1
        - 0.5 indicates a random classifier (diagonal line)
        - Higher values indicate better performance
        - 1.0 represents perfect classification
        
        **When to use:**
        
        - When you need to evaluate model performance across all thresholds
        - When you want to compare multiple models
        - When the threshold might change based on business needs
        """)
        
        info_box("""
        **Interpreting AUC-ROC values:**
        
        - **0.9 - 1.0**: Excellent
        - **0.8 - 0.9**: Good
        - **0.7 - 0.8**: Fair
        - **0.6 - 0.7**: Poor
        - **0.5 - 0.6**: Failed
        """, "tip")
    
    st.markdown("### Heat Maps for Multi-class Classification")
    
    st.markdown("""
    Heat maps provide visual representations of model performance across multiple classes, particularly useful for multi-class classification problems.
    """)
    
    # Create a multi-class confusion matrix heat map
    classes = ['Cat', 'Dog', 'Bird', 'Fish', 'Rabbit']
    
    # Create a confusion matrix with some pattern
    cm = np.array([
        [45, 3, 1, 0, 1],
        [4, 40, 2, 1, 3],
        [2, 1, 38, 5, 4],
        [0, 2, 3, 42, 3],
        [2, 4, 3, 1, 40]
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, 
                xticklabels=classes,
                yticklabels=classes)
    
    ax.set_title('Multi-class Confusion Matrix Heat Map')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    info_box("""
    **Benefits of Heat Map Visualizations:**
    
    - Quickly identify which classes are most confused with each other
    - Spot patterns in model errors
    - Understand class-specific performance
    - Provide intuitive representation of complex multi-class results
    - Help focus improvement efforts on problematic classes
    """, "info")

# Regression Metrics tab
with tabs[4]:
    # Mark as visited
    st.session_state['visited_Regression_Metrics'] = True
    
    custom_header("Regression Metrics")
    
    st.markdown("""
    Regression metrics evaluate models that predict continuous values. These metrics measure how close the predicted values are to the actual values.
    Each metric has its strengths and weaknesses, making them suitable for different scenarios.
    """)
    
    custom_header("Mean Squared Error (MSE)", "sub")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Mean Squared Error (MSE)** measures the average squared difference between predicted and actual values.
        
        **Formula**:
        ```
        MSE = (1/n) * Σ(y_true - y_pred)²
        ```
        
        **Characteristics**:
        - Always positive (greater than or equal to 0)
        - Lower values indicate better model performance
        - Heavily penalizes large errors due to squaring
        - Not in the same unit as the target variable
        
        **When to use**:
        - When large errors are particularly undesirable
        - When outliers should have a significant impact
        - As a loss function for optimization in many algorithms
        """)
    
    with col2:
        # Create visualization for MSE
        x = np.linspace(0, 10, 20)
        y_true = 2 * x + 1 + np.random.normal(0, 2, 20)
        y_pred = 1.8 * x + 0.5
        
        errors = y_true - y_pred
        squared_errors = errors**2
        mse = np.mean(squared_errors)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y_true, label='Actual values', alpha=0.7)
        ax.plot(x, y_pred, 'r-', label='Predictions')
        
        # Draw vertical lines for errors
        for i in range(len(x)):
            ax.plot([x[i], x[i]], [y_true[i], y_pred[i]], 'k--', alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'MSE: {mse:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        This plot shows:
        - Actual values (blue dots)
        - Predicted values (red line)
        - Errors (vertical dashed lines)
        
        MSE calculates the average of the squared lengths of these vertical lines.
        """)
    
    custom_header("Root Mean Squared Error (RMSE)", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Root Mean Squared Error (RMSE)** is the square root of the Mean Squared Error.
        
        **Formula**:
        ```
        RMSE = √MSE = √[(1/n) * Σ(y_true - y_pred)²]
        ```
        
        **Characteristics**:
        - In the same unit as the target variable (more interpretable)
        - Still penalizes large errors more than small ones
        - Lower values indicate better model performance
        - Always greater than or equal to MAE (Mean Absolute Error)
        
        **When to use**:
        - When you need an interpretable metric in the same units as the target
        - When you want to penalize large errors more heavily
        - When comparing different models on the same dataset
        """)
    
    with col2:
        # Calculate RMSE from previous example
        rmse = np.sqrt(mse)
        
        # Create bar chart to compare errors, squared errors
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Sort errors for better visualization
        sorted_indices = np.argsort(np.abs(errors))
        sorted_errors = errors[sorted_indices]
        sorted_squared_errors = squared_errors[sorted_indices]
        
        indices = np.arange(len(sorted_errors))
        width = 0.35
        
        ax.bar(indices - width/2, np.abs(sorted_errors), width, label='|Error|')
        ax.bar(indices + width/2, np.sqrt(sorted_squared_errors), width, label='√(Error²)')
        
        ax.axhline(y=np.mean(np.abs(errors)), color='b', linestyle='--', alpha=0.7, label='MAE')
        ax.axhline(y=rmse, color='r', linestyle='--', alpha=0.7, label='RMSE')
        
        ax.set_xlabel('Samples')
        ax.set_ylabel('Error Magnitude')
        ax.set_title('Comparison of Error Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        This chart shows:
        - Absolute errors for each prediction (blue bars)
        - Square root of squared errors (orange bars)
        - Mean Absolute Error (MAE) as blue dashed line
        - RMSE as red dashed line
        
        Note that RMSE is higher than MAE because it gives more weight to larger errors.
        """)
    
    custom_header("R-squared (R²)", "section")
    
    st.markdown("""
    **R-squared (R²)** measures the proportion of variance in the dependent variable that is predictable from the independent variables.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Formula**:
        ```
        R² = 1 - SSR/SST
        ```
        Where:
        - SSR = Sum of squared residuals = Σ(y_true - y_pred)²
        - SST = Total sum of squares = Σ(y_true - y_mean)²
        
        **Characteristics**:
        - Ranges from 0 to 1 (or can be negative for very poor models)
        - Higher values indicate better fit
        - R² = 1 means perfect prediction
        - R² = 0 means model predicts no better than a constant model
        
        **When to use**:
        - When you need an intuitive measure of model fit
        - When comparing models with similar complexity
        - When communicating results to non-technical stakeholders
        """)
    
    with col2:
        # Create visualization for R-squared
        x = np.linspace(0, 10, 50)
        y_true = 2 * x + 1 + np.random.normal(0, 3, 50)
        y_pred = 2.2 * x + 0.8
        y_mean = np.mean(y_true)
        
        SSR = np.sum((y_true - y_pred)**2)
        SST = np.sum((y_true - y_mean)**2)
        r_squared = 1 - (SSR / SST)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x, y_true, label='Data points')
        line1, = ax.plot(x, y_pred, 'r-', label='Model prediction')
        line2, = ax.plot([0, 10], [y_mean, y_mean], 'g--', label='Mean prediction')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'R² = {r_squared:.2f}')
        ax.legend(handles=[scatter, line1, line2])
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        This plot shows:
        - Actual data points (blue dots)
        - Model predictions (red line)
        - Mean prediction/baseline model (green dashed line)
        
        R² measures how much better the model performs compared to simply predicting the mean value.
        """)
    
    custom_header("Adjusted R-squared", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Adjusted R-squared** is a modified version of R-squared that accounts for the number of predictors in the model.
        
        **Formula**:
        ```
        Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k - 1)]
        ```
        Where:
        - n = number of data points
        - k = number of predictors (excluding constant)
        
        **Characteristics**:
        - Penalizes adding unnecessary predictors
        - Can increase only if new predictors add significant value
        - Always less than or equal to R²
        - More reliable for comparing models with different numbers of predictors
        
        **When to use**:
        - When comparing models with different numbers of predictors
        - When deciding if adding a predictor improves the model
        - To prevent overfitting by discouraging unnecessary complexity
        """)
    
    with col2:
        # Create data for adjusted R-squared comparison
        n_samples = 100
        n_features = np.arange(1, 21)
        r2 = 0.8 - 0.4 * np.exp(-0.1 * n_features)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(n_features, r2, 'b-', marker='o', label='R²')
        ax.plot(n_features, adj_r2, 'r-', marker='s', label='Adjusted R²')
        
        ax.set_xlabel('Number of Predictors')
        ax.set_ylabel('Score')
        ax.set_title('R² vs Adjusted R² as Predictors Increase')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        This chart illustrates how:
        - R² always increases (or stays the same) as predictors are added
        - Adjusted R² penalizes unnecessary predictors
        - Adjusted R² only increases when added predictors provide meaningful improvement
        - The gap between R² and Adjusted R² widens as more predictors are added
        """)
        
        info_box("""
        **Key Insight:**
        
        - If R² increases but Adjusted R² decreases when adding a variable, the new variable likely isn't adding meaningful predictive power and is overfitting the data.
        
        - Look for the model with the highest Adjusted R² when comparing models with different numbers of predictors.
        """, "tip")
    
    st.markdown("### Comparing Regression Metrics")
    
    metrics_comparison = {
        "Metric": ["MSE", "RMSE", "R-squared", "Adjusted R-squared"],
        "Range": ["[0, ∞)", "[0, ∞)", "(-∞, 1]", "(-∞, 1]"],
        "Units": ["Target variable squared", "Same as target", "Unitless", "Unitless"],
        "Optimal Value": ["0", "0", "1", "1"],
        "Penalizes Outliers": ["Heavily", "Moderately", "Depends", "Depends"],
        "Interpretability": ["Low", "Medium", "High", "High"],
        "Considers Model Complexity": ["No", "No", "No", "Yes"]
    }
    
    df_metrics = pd.DataFrame(metrics_comparison)
    st.table(df_metrics)
    
    info_box("""
    **Choosing the Right Regression Metric:**
    
    - **RMSE**: When you want a metric in the same unit as your target and want to penalize large errors
    - **R-squared**: When you need an intuitive measure of model fit for stakeholder communication
    - **Adjusted R-squared**: When comparing models with different numbers of features
    
    For critical applications, it's best practice to consider multiple metrics together to get a complete picture of model performance.
    """, "warning")

# Model Optimization tab
with tabs[5]:
    # Mark as visited
    st.session_state['visited_Model_Optimization'] = True
    
    custom_header("Model Optimization Techniques")
    
    st.markdown("""
    Model optimization involves techniques to improve model performance, speed up training, and enhance generalization.
    Key optimization techniques include loss function selection, achieving convergence, and gradient descent algorithms.
    """)
    
    custom_header("Loss Functions", "sub")
    
    st.markdown("""
    A loss function measures how well a model performs by quantifying the difference between predicted values and actual values.
    Optimization algorithms aim to minimize this loss during model training.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Common Loss Functions")
        st.markdown("""
        **For Regression:**
        - **Mean Squared Error (MSE)**: Penalizes larger errors more
        - **Mean Absolute Error (MAE)**: More robust to outliers
        - **Huber Loss**: Combines properties of MSE and MAE
        
        **For Classification:**
        - **Binary Cross-Entropy**: For binary classification problems
        - **Categorical Cross-Entropy**: For multi-class classification
        - **Hinge Loss**: Used in SVMs and margin-based classifiers
        """)
        
        st.markdown("### Loss Landscape")
        st.markdown("""
        The loss landscape represents the loss function value for different model parameter combinations:
        
        - **Global Minimum**: Best possible set of parameters
        - **Local Minima**: Suboptimal parameter sets that appear optimal locally
        - **Saddle Points**: Points where gradient is zero but not a minimum
        
        Optimization algorithms navigate this landscape to find parameter values that minimize the loss.
        """)
    
    with col2:
        # Create a visualization of a loss landscape
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        # Function representing a complex loss landscape with multiple minima
        Z = 0.1 * (X**2 + Y**2) + 2*np.sin(X) + 2*np.cos(Y) + np.sin(X*Y/3)
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_zlabel('Loss')
        ax.set_title('Example Loss Landscape')
        
        # Mark some key points on the loss landscape
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        global_min_x, global_min_y = X[min_idx], Y[min_idx]
        global_min_z = Z[min_idx]
        
        # Add global minimum point
        ax.scatter([global_min_x], [global_min_y], [global_min_z], color='red', s=100, label='Global Minimum')
        
        # Add some local minima
        local_min1 = (2, 2, 0.1 * (2**2 + 2**2) + 2*np.sin(2) + 2*np.cos(2) + np.sin(2*2/3))
        local_min2 = (-3, 1, 0.1 * ((-3)**2 + 1**2) + 2*np.sin(-3) + 2*np.cos(1) + np.sin(-3*1/3))
        
        ax.scatter([local_min1[0], local_min2[0]], [local_min1[1], local_min2[1]], 
                  [local_min1[2], local_min2[2]], color='orange', s=100, label='Local Minima')
        
        ax.legend()
        ax.view_init(elev=30, azim=-60)
        
        st.pyplot(fig)
    
    custom_header("Convergence", "section")
    
    st.markdown("""
    **Convergence** refers to a model's ability to reach an optimal solution during training.
    It's achieved when further training provides negligible improvements in the loss function.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Common Convergence Issues")
        st.markdown("""
        1. **Vanishing Gradients**:
           - Gradients become too small for effective learning
           - Common in deep networks with certain activation functions
           - Solution: Use ReLU activations, residual connections
        
        2. **Exploding Gradients**:
           - Gradients become extremely large, causing unstable updates
           - Solution: Gradient clipping, proper weight initialization
        
        3. **Getting Stuck in Local Minima**:
           - Model converges to suboptimal solution
           - Solution: Use momentum, learning rate schedules
        
        4. **Saddle Points**:
           - Areas where gradient is zero but not a minimum
           - Common in high-dimensional spaces
           - Solution: Add noise, use advanced optimizers
        """)
    
    with col2:
        # Create a visual showing convergence
        iterations = np.arange(1, 101)
        
        # Different convergence scenarios
        good_convergence = 5 / (1 + np.exp(-0.1 * (iterations - 30))) + 0.05 * np.random.randn(len(iterations))
        slow_convergence = 3 / (1 + np.exp(-0.03 * (iterations - 50))) + 0.05 * np.random.randn(len(iterations))
        no_convergence = 2 + 0.5 * np.sin(iterations / 5) + 0.8 * np.random.randn(len(iterations))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(iterations, good_convergence, 'g-', label='Good Convergence')
        ax.plot(iterations, slow_convergence, 'b-', label='Slow Convergence')
        ax.plot(iterations, no_convergence, 'r-', label='No Convergence')
        
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Model Performance')
        ax.set_title('Convergence Patterns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        This chart shows different convergence patterns:
        
        - **Good Convergence**: Performance improves quickly and stabilizes
        - **Slow Convergence**: Performance improves gradually
        - **No Convergence**: Performance fluctuates without clear improvement
        
        Proper convergence is essential for model training efficiency and effectiveness.
        """)
    
    info_box("""
    **Amazon SageMaker Tools for Convergence Issues:**
    
    - **Automatic Model Tuning (AMT)**: Finds optimal hyperparameters to improve convergence
    - **SageMaker Debugger**: Monitors and visualizes training metrics to identify convergence issues
    - **Training Compiler**: Optimizes model training to reduce convergence challenges
    - **Distributed Training**: Enables faster convergence through parallelized training
    
    These tools can significantly reduce the time and effort needed to achieve model convergence.
    """, "tip")
    
    custom_header("Gradient Descent", "section")
    
    st.markdown("""
    **Gradient Descent** is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Types of Gradient Descent")
        st.markdown("""
        1. **Batch Gradient Descent**:
           - Uses entire dataset for each update
           - Accurate but computationally expensive
           - Slow for large datasets
        
        2. **Stochastic Gradient Descent (SGD)**:
           - Uses single example for each update
           - Fast but noisy
           - May require more iterations to converge
        
        3. **Mini-batch Gradient Descent**:
           - Uses small batch of examples for each update
           - Balance between batch and stochastic
           - Most commonly used in practice
        """)
        
        st.markdown("### Advanced Gradient Descent Algorithms")
        st.markdown("""
        1. **Momentum**: Accelerates SGD by accumulating past gradients
        
        2. **RMSprop**: Adapts learning rates based on historical gradients
        
        3. **Adam**: Combines benefits of momentum and RMSprop
        
        4. **AdamW**: Adam with improved weight decay regularization
        """)
    
    with col2:
        # Create visualization of gradient descent
        def loss_function(x, y):
            return x**2 + y**2 * 0.25
        
        # Create a grid of points
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        Z = loss_function(X, Y)
        
        # Create points for gradient descent path
        gd_x = [-3]
        gd_y = [3]
        learning_rate = 0.1
        
        for _ in range(20):
            x_prev, y_prev = gd_x[-1], gd_y[-1]
            # Compute gradients (derivatives of loss function)
            dx = 2 * x_prev
            dy = 0.5 * y_prev
            # Update position
            x_new = x_prev - learning_rate * dx
            y_new = y_prev - learning_rate * dy
            gd_x.append(x_new)
            gd_y.append(y_new)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        contour = ax.contour(X, Y, Z, 20, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        ax.scatter(gd_x[0], gd_y[0], color='red', s=100, label='Start')
        ax.scatter(gd_x[-1], gd_y[-1], color='green', s=100, label='End')
        ax.plot(gd_x, gd_y, 'r-o', alpha=0.7, label='Gradient Descent Path')
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title('Gradient Descent Optimization')
        ax.legend()
        
        st.pyplot(fig)
        
        st.markdown("""
        This visualization shows:
        
        - Contour lines representing the loss function landscape
        - Starting point (red dot)
        - Path taken by gradient descent algorithm (red line)
        - Final position (green dot)
        
        The algorithm iteratively moves down the loss function surface toward a minimum.
        """)
    
    with st.expander("Exploring Advanced Optimizers"):
        st.markdown("""
        ### Comparing Optimization Algorithms
        
        Different optimizers have different convergence properties and are suited for different types of problems:
        """)
        
        # Create a visual comparing different optimizers
        iterations = np.arange(1, 101)
        
        # Loss curves for different optimizers
        sgd_loss = 2 / np.log(iterations + 1) + 0.5 * np.exp(-0.02 * iterations) + 0.1 * np.random.randn(len(iterations))
        momentum_loss = 1.8 / np.log(iterations + 1) + 0.3 * np.exp(-0.03 * iterations) + 0.08 * np.random.randn(len(iterations))
        rmsprop_loss = 1.5 / np.log(iterations + 1) + 0.2 * np.exp(-0.05 * iterations) + 0.05 * np.random.randn(len(iterations))
        adam_loss = 1.2 / np.log(iterations + 1) + 0.1 * np.exp(-0.07 * iterations) + 0.03 * np.random.randn(len(iterations))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, sgd_loss, label='SGD')
        ax.plot(iterations, momentum_loss, label='SGD with Momentum')
        ax.plot(iterations, rmsprop_loss, label='RMSprop')
        ax.plot(iterations, adam_loss, label='Adam')
        
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Loss')
        ax.set_title('Optimization Algorithm Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        st.markdown("""
        **Key Observations:**
        
        - **Standard SGD** converges slowly and exhibits more noise
        - **Momentum** accelerates convergence by accumulating past gradients
        - **RMSprop** adapts learning rates for each parameter, improving convergence
        - **Adam** often provides the fastest convergence by combining adaptive learning rates with momentum
        
        **When to use different optimizers:**
        
        - **SGD**: When computational efficiency is critical or for simple problems
        - **SGD with Momentum**: For smoother convergence and navigating around local minima
        - **RMSprop**: For problems with sparse gradients or noisy data
        - **Adam**: For most deep learning applications as a robust default choice
        
        In Amazon SageMaker, you can specify these different optimizers when configuring your training job.
        """)
    
    custom_header("SageMaker Tools for Optimization", "sub")
    
    st.markdown("""
    Amazon SageMaker provides several tools to help optimize your machine learning models.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### SageMaker Debugger")
        # st.image("https://d1.awsstatic.com/re19/Diagram_Amazon-SageMaker-Debugger_How-it-Works@2x.155304b8582a1a5f9d8d654461e01d9b08712c53.png", width=400)
        st.markdown("""
        **Key capabilities:**
        
        - Automatically captures training data for analysis
        - Provides real-time insights into training process
        - Detects issues like vanishing gradients, exploding tensors
        - Generates alerts for training problems
        - Visualizes model behavior during training
        - Requires no code changes for basic functionality
        """)
    
    with col2:
        st.markdown("### SageMaker Automatic Model Tuning")
        # st.image("https://d1.awsstatic.com/product-marketing/Automatic%20Model%20Tuning/product-page-diagram_SageMaker_Auto-Model-Tuning_HIW%402x.4ec0c8c9733cd8cfbfacb9777389f16b7c960d97.png", width=400)
        st.markdown("""
        **Key capabilities:**
        
        - Automatically finds optimal hyperparameters
        - Supports multiple search strategies:
          - Random search
          - Bayesian optimization
          - Hyperband
        - Distributed training for faster tuning
        - Reuses previous tuning results
        - Integrates with SageMaker training jobs
        """)
    
    st.markdown("### Other SageMaker Optimization Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Training Compiler")
        st.markdown("""
        - Optimizes model training speed
        - Accelerates deep learning models
        - Reduces training time and cost
        - Works with TensorFlow and PyTorch
        - No code changes required
        """)
    
    with col2:
        st.markdown("#### Distributed Training")
        st.markdown("""
        - Scales model training across instances
        - Supports data parallel training
        - Supports model parallel training
        - Optimizes communication patterns
        - Works with popular frameworks
        """)
    
    with col3:
        st.markdown("#### Spot Training")
        st.markdown("""
        - Utilizes EC2 Spot instances
        - Reduces training costs by up to 90%
        - Automatically saves checkpoints
        - Handles instance interruptions
        - Resumes training automatically
        """)

# Quiz tab
with tabs[6]:
    custom_header("Test Your Knowledge")
    
    st.markdown("""
    This quiz will test your understanding of the key concepts covered in Domain 2: ML Model Development - Task 2.3: Analyze Model Performance.
    Answer the following questions to evaluate your knowledge of model evaluation and optimization.
    """)
    
    # Define quiz questions
    questions = [
        {
            "question": "What is the main difference between L1 and L2 regularization?",
            "options": [
                "L1 can reduce feature weights to exactly zero, while L2 only makes them small", 
                "L1 works only for regression, while L2 works only for classification", 
                "L1 requires more computational resources than L2", 
                "L1 always performs better than L2 for all types of models"
            ],
            "correct": "L1 can reduce feature weights to exactly zero, while L2 only makes them small",
            "explanation": "L1 regularization (Lasso) can drive some feature weights to exactly zero, effectively performing feature selection. L2 regularization (Ridge) shrinks all weights proportionally but doesn't typically make them exactly zero."
        },
        {
            "question": "When would precision be a more appropriate metric than recall for a classification model?",
            "options": [
                "When false positives are more costly than false negatives", 
                "When false negatives are more costly than false positives", 
                "When the dataset is highly imbalanced", 
                "When model training time is a primary concern"
            ],
            "correct": "When false positives are more costly than false negatives",
            "explanation": "Precision focuses on minimizing false positives. It's the ratio of true positives to all predicted positives (TP/(TP+FP)). It's more appropriate when the cost of false positives is high, such as in spam detection where incorrectly filtering legitimate emails (false positives) is worse than letting some spam through."
        },
        {
            "question": "Which of the following is an advantage of using Adjusted R-squared over regular R-squared?",
            "options": [
                "It penalizes models with unnecessary predictors", 
                "It always provides higher values than regular R-squared", 
                "It's more computationally efficient", 
                "It works for both classification and regression problems"
            ],
            "correct": "It penalizes models with unnecessary predictors",
            "explanation": "Adjusted R-squared penalizes models for adding unnecessary predictors by adjusting for the number of predictors relative to the sample size. This helps prevent overfitting and encourages more parsimonious models, unlike regular R-squared which always increases (or stays the same) when more predictors are added."
        },
        {
            "question": "What does an AUC-ROC value of 0.5 indicate about a binary classification model?",
            "options": [
                "The model performs no better than random guessing", 
                "The model is perfectly balanced between precision and recall", 
                "The model needs more training data", 
                "The model has 50% accuracy"
            ],
            "correct": "The model performs no better than random guessing",
            "explanation": "An AUC-ROC (Area Under the Receiver Operating Characteristic Curve) value of 0.5 indicates that the model's predictions are no better than random guessing. The ROC curve for such a model would be a diagonal line from the bottom-left to the top-right of the plot."
        },
        {
            "question": "Which Amazon SageMaker feature helps you visualize and debug training issues such as vanishing gradients or exploding tensors?",
            "options": [
                "Amazon SageMaker Debugger", 
                "Amazon SageMaker Clarify", 
                "Amazon SageMaker Feature Store", 
                "Amazon SageMaker Experiments"
            ],
            "correct": "Amazon SageMaker Debugger",
            "explanation": "Amazon SageMaker Debugger automatically captures training data and provides real-time insights into the training process. It can detect issues like vanishing gradients, exploding tensors, and other training problems, and generate alerts when these issues occur."
        },
        {
            "question": "Which optimization algorithm combines the benefits of momentum and adaptive learning rates?",
            "options": [
                "Adam", 
                "Stochastic Gradient Descent", 
                "Batch Gradient Descent", 
                "RMSprop"
            ],
            "correct": "Adam",
            "explanation": "Adam (Adaptive Moment Estimation) combines the benefits of momentum, which accumulates past gradients to accelerate convergence, and adaptive learning rates like those in RMSprop. It's widely used because of its effective performance across many types of neural network architectures."
        },
        {
            "question": "When a model performs well on training data but poorly on validation data, this is most likely due to:",
            "options": [
                "Overfitting", 
                "Underfitting", 
                "Regularization", 
                "Gradient descent failure"
            ],
            "correct": "Overfitting",
            "explanation": "Overfitting occurs when a model learns the training data too well, including its noise and outliers, rather than learning the underlying pattern. This results in good performance on training data but poor performance on unseen validation data. Regularization techniques like L1/L2 and dropout help prevent overfitting."
        },
        {
            "question": "Which metric is most appropriate for evaluating a regression model when outliers are a concern?",
            "options": [
                "Mean Absolute Error (MAE)", 
                "Root Mean Squared Error (RMSE)", 
                "R-squared", 
                "Classification Accuracy"
            ],
            "correct": "Mean Absolute Error (MAE)",
            "explanation": "Mean Absolute Error (MAE) is less sensitive to outliers than RMSE because it uses absolute differences rather than squared differences. This means extreme values have less impact on the overall error metric, making it more robust when outliers are present in the data."
        },
        {
            "question": "What is the main benefit of using Mini-batch Gradient Descent instead of Batch Gradient Descent?",
            "options": [
                "Better balance between computational efficiency and convergence stability", 
                "It always reaches the global minimum", 
                "It requires less memory", 
                "It's simpler to implement"
            ],
            "correct": "Better balance between computational efficiency and convergence stability",
            "explanation": "Mini-batch Gradient Descent strikes a balance between the computational efficiency of Stochastic Gradient Descent (which updates after each sample) and the stability of Batch Gradient Descent (which uses the entire dataset). It updates after processing small batches of data, providing more frequent updates than Batch GD while being less noisy than Stochastic GD."
        },
        {
            "question": "What component of SageMaker Experiments allows you to track inputs, parameters, and metrics for a specific stage in a machine learning workflow?",
            "options": [
                "Trial component", 
                "Experiment", 
                "Dataset", 
                "Artifact"
            ],
            "correct": "Trial component",
            "explanation": "In SageMaker Experiments, a trial component represents a specific stage in a machine learning workflow, such as data preprocessing, model training, or evaluation. Each trial component tracks inputs, parameters, configurations, metrics, and other metadata related to that stage."
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
            st.success("🎉 Perfect score! You've mastered the concepts of Model Evaluation and Optimization!")
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
    Explore these resources to deepen your understanding of Model Evaluation and Optimization.
    These materials provide additional context and practical guidance for implementing the concepts covered in this module.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AWS Documentation")
        st.markdown("""
        - [Amazon SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html)
        - [Amazon SageMaker Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html)
        - [Amazon SageMaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-detect-data-bias.html)
        - [Amazon SageMaker Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
        - [Amazon SageMaker Training Compiler](https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html)
        - [Amazon SageMaker Shadow Testing](https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html)
        - [Amazon SageMaker Distributed Training](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)
        """)
        
        st.markdown("### AWS Blog Posts")
        st.markdown("""
        - [Evaluate ML Models with Amazon SageMaker Clarify](https://aws.amazon.com/blogs/machine-learning/evaluating-and-debugging-generative-ai-models-with-amazon-sagemaker-clarify/)
        - [Debug and Profile Models with Amazon SageMaker Debugger](https://aws.amazon.com/blogs/machine-learning/debug-and-profile-models-with-amazon-sagemaker-debugger/)
        - [Gain Insights from Your ML Experiments with Amazon SageMaker Experiments](https://aws.amazon.com/blogs/machine-learning/gain-insights-from-your-ml-experiments-with-amazon-sagemaker-experiments/)
        - [Hyperparameter Tuning with Amazon SageMaker Automatic Model Tuning](https://aws.amazon.com/blogs/machine-learning/hyperparameter-tuning-with-amazon-sagemaker-automatic-model-tuning/)
        - [Overfitting in Machine Learning: What It Is and How to Prevent It](https://aws.amazon.com/what-is/overfitting/)
        """)
    
    with col2:
        st.markdown("### Training Courses")
        st.markdown("""
        - [AWS Machine Learning University](https://aws.amazon.com/machine-learning/mlu/)
        - [MLU-Explain: Visual Explanations of Core ML Concepts](https://mlu-explain.github.io/)
        - [Developing Machine Learning Solutions](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/191/developing-machine-learning-solutions)
        - [AWS Cloud Quest: Machine Learning](https://aws.amazon.com/training/learn-about/cloud-quest/)
        - [AWS Machine Learning Specialty Certification](https://aws.amazon.com/certification/certified-machine-learning-specialty/)
        - [Practical Data Science with Amazon SageMaker](https://www.coursera.org/specializations/practical-data-science)
        """)
        
        st.markdown("### Tools and Services")
        st.markdown("""
        - [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
        - [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/)
        - [Amazon SageMaker JumpStart](https://aws.amazon.com/sagemaker/jumpstart/)
        - [TensorFlow](https://www.tensorflow.org/)
        - [PyTorch](https://pytorch.org/)
        - [scikit-learn](https://scikit-learn.org/)
        - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
        """)
    
    custom_header("Further Learning Resources", "sub")
    
    st.markdown("""
    ### Interactive Notebooks and Tutorials
    
    - [SageMaker Examples GitHub Repository](https://github.com/aws/amazon-sagemaker-examples)
    - [Model Evaluation with SageMaker Clarify](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-clarify)
    - [AWS ML Blog: Model Evaluation Best Practices](https://aws.amazon.com/blogs/machine-learning/)
    
    ### Research Papers and Technical Deep Dives
    
    - [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)
    - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    - [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
    
    ### Community Resources
    
    - [AWS Machine Learning Community](https://aws.amazon.com/machine-learning/community/)
    - [Stack Overflow: AWS SageMaker](https://stackoverflow.com/questions/tagged/amazon-sagemaker)
    - [AWS re:Post for Machine Learning](https://repost.aws/tags/TAVUGXsbkQ9kYLSfprnR7XZQ/amazon-sage-maker)
    """)
    
    st.markdown("### Recommended Reading")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### Fundamentals
        
        - **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** by Aurélien Géron
        - **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
        - **Pattern Recognition and Machine Learning** by Christopher Bishop
        """)
    
    with col2:
        st.markdown("""
        #### Specialized Topics
        
        - **Evaluating Machine Learning Models** by Alice Zheng
        - **Feature Engineering for Machine Learning** by Alice Zheng and Amanda Casari
        - **Interpretable Machine Learning** by Christoph Molnar
        """)
    
    with col3:
        st.markdown("""
        #### AWS Specific
        
        - **Machine Learning Best Practices for AWS** by Chris Fregly and Antje Barth
        - **Practical Machine Learning on AWS** by Himanshu Sharma
        - **AWS Certified Machine Learning Specialty** by Shreyas Subramanian
        """)

# Footer
st.markdown("---")
col1, col2 = st.columns([1, 5])
with col1:
    st.image("images/aws_logo.png", width=70)
with col2:
    st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")
# ```

# To use this application, you would need to create the following directory structure:

# ```
# - app.py (the main file with the code above)
# - images/
#   - mla_badge.png
#   - mla_badge_big.png
#   - aws_logo.png
#   - underfitting.png
#   - overfitting.png
#   - good_fit.png
#   - dropout.png
#   - l1_l2.png
#   - confusion_matrix.png
#   - cross_validation.png
#   - learning_curves.png
# ```

# The application follows the same UI/UX styling as the Domain 1 code, with consistent components like custom headers, info boxes, definition boxes, and expandable sections. It organizes the content into tabs covering Model Generalization, Model Evaluation, Classification Metrics, Regression Metrics, and Model Optimization Techniques, plus a quiz to test knowledge and a resources section for further learning.