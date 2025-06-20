
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import time
import uuid
import random
import json
from PIL import Image
import io
import base64
from streamlit_lottie import st_lottie
import requests
from datetime import datetime

# Set page configuration for wider layout
st.set_page_config(
    page_title="SageMaker Optimization Tools Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Load animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Initialize session state
if 'init_opt' not in st.session_state:
    st.session_state.init_opt = True
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.debugger_stage = 0
    st.session_state.current_model = "ResNet"
    st.session_state.learning_rate = 0.01
    st.session_state.batch_size = 64
    st.session_state.epochs = 25
    st.session_state.automl_objective = "Accuracy"
    st.session_state.compiler_model = "BERT"
    st.session_state.distributed_training_strategy = "Data Parallel"
    st.session_state.num_instances = 2
    st.session_state.spot_training_enabled = True
    st.session_state.max_wait_time = 1  # hours

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
</style>
""", unsafe_allow_html=True)

# Sidebar for session management
with st.sidebar:
    st.title("Session Management")
    st.info(f"User ID: {st.session_state.user_id}")
    
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()
    
    st.divider()
    
    # Information about the application
    with st.expander(label='About this application' ,expanded=False):
        st.markdown("""
            This interactive learning application demonstrates the powerful optimization 
            tools available in Amazon SageMaker. Explore each tab to learn about different 
            capabilities and see them in action with interactive examples.
        """)
        
        # AWS learning resources
        st.subheader("Additional Resources")
        st.markdown("""
            - [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
            - [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
            - [AWS Training and Certification](https://aws.amazon.com/training/)
        """)

# Main app header
st.title("Amazon SageMaker Tools for Optimization")
st.markdown("Explore the powerful tools that SageMaker offers for optimizing and accelerating your machine learning workflows.")


# Tab-based navigation with emoji
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🐞 SageMaker Debugger", 
    "🎯 Automatic Model Tuning", 
    "🚀 Training Compiler",
    "🔄 Distributed Training",
    "💰 Spot Training"
])

# TAB 1: SAGEMAKER DEBUGGER
with tab1:
    st.header("Amazon SageMaker Debugger")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        SageMaker Debugger provides transparency into the training process by capturing, visualizing, and 
        analyzing data from training runs.
        
        **Key benefits:**
        - Monitor and debug training jobs in real-time
        - Detect and fix model training issues automatically
        - Optimize training performance and resource utilization
        - Understand model convergence behavior
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/re19/Diagrams/product-page-diagram_Amazon-SageMaker-Debugger_How-it-Works.2dc4323179c41a34b6cc41a1bcc43a031ecfcafd.png", 
                 caption="SageMaker Debugger Workflow")
    
    # Interactive debugger demo
    st.subheader("Interactive Demo: Debug a Training Process")
    
    # Simulated model training visualization
    def simulate_training_data(epochs, learning_rate, has_issues=False, issue_type=None):
        np.random.seed(42)
        train_loss = []
        val_loss = []
        gradients = []
        weights = []
        
        # Starting point
        train_val = 2.0
        val_val = 2.2
        grad_val = 0.5
        weight_val = np.random.normal(0, 0.1, 5)
        
        # Generate data with simulated issues
        for i in range(epochs):
            # Decrease with some noise
            decay_factor = np.exp(-learning_rate * i / 20)
            train_val = max(0.1, train_val * decay_factor + np.random.normal(0, 0.05))
            val_val = max(0.15, val_val * decay_factor + np.random.normal(0, 0.07))
            
            # Add issues after certain point if specified
            if has_issues and i > epochs // 2:
                if issue_type == "vanishing_gradients":
                    # Simulate vanishing gradients
                    grad_val = grad_val * 0.7
                elif issue_type == "overfitting":
                    # Simulate overfitting
                    train_val = max(0.05, train_val * 0.9)
                    val_val = min(3.0, val_val * 1.1)
                elif issue_type == "exploding_gradients":
                    # Simulate exploding gradients
                    if i % 5 == 0:  # occasional spikes
                        grad_val = min(10, grad_val * 2.0)
                    else:
                        grad_val = max(0.05, grad_val * 0.95 + np.random.normal(0, 0.02))
                else:
                    grad_val = max(0.05, grad_val * 0.95 + np.random.normal(0, 0.02))
            else:
                grad_val = max(0.05, grad_val * 0.95 + np.random.normal(0, 0.02))
            
            # Update weights with some drift
            weight_val = weight_val - learning_rate * grad_val * np.random.normal(1, 0.1, 5)
            
            train_loss.append(train_val)
            val_loss.append(val_val)
            gradients.append(grad_val)
            weights.append(weight_val.copy())
            
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'gradients': gradients,
            'weights': weights,
            'epochs': list(range(1, epochs + 1))
        }

    # Training parameters    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_options = ["ResNet", "BERT", "XGBoost"]
        selected_model = st.selectbox("Select Model", model_options, index=model_options.index(st.session_state.current_model))
        st.session_state.current_model = selected_model

    with col2:
        epochs = st.slider("Number of Epochs", min_value=10, max_value=100, value=st.session_state.epochs, step=5)
        st.session_state.epochs = epochs
    
    with col3:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5],
            value=st.session_state.learning_rate
        )
        st.session_state.learning_rate = learning_rate
    
    # Issue selection
    issue_type = st.selectbox("Simulate Issue", 
                       ["None", "Vanishing Gradients", "Overfitting", "Exploding Gradients"])
    
    has_issues = issue_type != "None"
    issue_mapping = {
        "None": None,
        "Vanishing Gradients": "vanishing_gradients",
        "Overfitting": "overfitting",
        "Exploding Gradients": "exploding_gradients"
    }
    
    # Run training
    if st.button("Run Training"):
        with st.spinner("Training model and capturing debug information..."):
            training_data = simulate_training_data(epochs, learning_rate, has_issues, issue_mapping[issue_type])
            st.session_state.debugger_training_data = training_data
            st.session_state.debugger_stage = 1
            
    # Display results if available
    if st.session_state.debugger_stage >= 1:
        training_data = st.session_state.debugger_training_data
        
        st.success("Training complete! Analyzing debugging information...")
        
        # Show the loss curves
        st.markdown("### Loss Curves")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(training_data['epochs'], training_data['train_loss'], label='Training Loss', color=AWS_COLORS['orange'])
        ax.plot(training_data['epochs'], training_data['val_loss'], label='Validation Loss', color=AWS_COLORS['teal'])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for issues
        if issue_type != "None":
            problem_epoch = epochs // 2
            if issue_type == "Vanishing Gradients":
                ax.axvline(x=problem_epoch, color=AWS_COLORS['red'], linestyle='--', alpha=0.7)
                ax.text(problem_epoch + 2, max(training_data['train_loss'][:problem_epoch]), 
                        "Vanishing Gradients Start", color=AWS_COLORS['red'])
            elif issue_type == "Overfitting":
                ax.axvline(x=problem_epoch, color=AWS_COLORS['red'], linestyle='--', alpha=0.7)
                ax.text(problem_epoch + 2, max(training_data['val_loss'][:problem_epoch]), 
                        "Overfitting Begins", color=AWS_COLORS['red'])
            elif issue_type == "Exploding Gradients":
                ax.axvline(x=problem_epoch, color=AWS_COLORS['red'], linestyle='--', alpha=0.7)
                ax.text(problem_epoch + 2, max(training_data['train_loss'][:problem_epoch]), 
                        "Exploding Gradients Start", color=AWS_COLORS['red'])
        
        st.pyplot(fig)
        
        # Show gradients
        st.markdown("### Gradient Magnitudes")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(training_data['epochs'], training_data['gradients'], color=AWS_COLORS['green'])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title('Gradient Flow During Training')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add threshold line for vanishing/exploding gradients
        if issue_type == "Vanishing Gradients":
            ax.axhline(y=0.1, color=AWS_COLORS['red'], linestyle='--')
            ax.text(epochs//4, 0.11, "Healthy Gradient Threshold", color=AWS_COLORS['red'])
        elif issue_type == "Exploding Gradients":
            ax.axhline(y=2.0, color=AWS_COLORS['red'], linestyle='--')
            ax.text(epochs//4, 2.1, "Gradient Explosion Threshold", color=AWS_COLORS['red'])
        
        st.pyplot(fig)
        
        # Weight trajectories visualization
        st.markdown("### Parameter Evolution")
        
        weights_array = np.array(training_data['weights'])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(min(5, weights_array.shape[1])):
            ax.plot(training_data['epochs'], weights_array[:, i], 
                   label=f'Weight {i+1}', alpha=0.7)
            
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title('Weight Evolution During Training')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Debugging insights
        st.markdown("### Debugger Insights")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Training Metrics")
            
            final_train = training_data['train_loss'][-1]
            final_val = training_data['val_loss'][-1]
            final_gradient = training_data['gradients'][-1]
            
            metrics_df = pd.DataFrame({
                'Metric': ['Final Training Loss', 'Final Validation Loss', 'Final Gradient Magnitude'],
                'Value': [f"{final_train:.4f}", f"{final_val:.4f}", f"{final_gradient:.4f}"]
            })
            
            st.dataframe(metrics_df, hide_index=True)
            
            # Compute and display generalization gap
            gap = final_val - final_train
            st.metric("Generalization Gap", f"{gap:.4f}", 
                     delta=None if gap < 0.5 else f"+{gap-0.5:.2f} (High)")
        
        with col2:
            st.markdown("#### Automated Analysis")
            
            # Calculate metrics for analysis
            train_trend = (training_data['train_loss'][-1] - 
                           training_data['train_loss'][len(training_data['train_loss'])//2])
            val_trend = (training_data['val_loss'][-1] - 
                         training_data['val_loss'][len(training_data['val_loss'])//2])
            
            grad_final = training_data['gradients'][-1]
            grad_max = max(training_data['gradients'])
            
            # Display findings based on metrics
            problems = []
            
            if train_trend > 0:
                problems.append("❌ Training loss is **increasing** in later epochs")
            
            if val_trend > 0.3:
                problems.append("❌ Validation loss shows **significant increase** (possible overfitting)")
            elif val_trend > 0:
                problems.append("⚠️ Validation loss is slightly increasing")
                
            if final_val / final_train > 1.5:
                problems.append("❌ Large gap between training and validation loss (overfitting)")
            
            if grad_final < 0.1:
                problems.append("❌ Gradients are very small (vanishing gradient problem)")
                
            if grad_max > 5.0:
                problems.append("❌ Gradients are unusually large (exploding gradient problem)")
            
            if not problems:
                st.success("✅ No major issues detected in the training process")
            else:
                for problem in problems:
                    st.warning(problem)
        
        # Debugger recommendations
        if problems:
            st.markdown("### Recommendations")
            
            if "overfitting" in " ".join(problems).lower():
                st.info("🔍 **Overfitting Detected**: Consider adding regularization (L1/L2), increasing dropout rate, or using early stopping.")
            
            if "vanishing gradient" in " ".join(problems).lower():
                st.info("🔍 **Vanishing Gradients Detected**: Try using skip connections, batch normalization, or a different activation function.")
                
            if "exploding gradient" in " ".join(problems).lower():
                st.info("🔍 **Exploding Gradients Detected**: Implement gradient clipping or reduce the learning rate.")
                
            if train_trend > 0:
                st.info("🔍 **Unstable Training**: Consider reducing the learning rate or implementing learning rate scheduling.")
    
    # Code example for SageMaker Debugger
    st.markdown("### Implementing SageMaker Debugger")
    st.markdown("Here's how you can use SageMaker Debugger in your training script:")
    
    st.code('''
# Import SageMaker Debugger
import smdebug.pytorch as smd
from smdebug import modes

def train(model, train_loader, criterion, optimizer, epoch, device, hook):
    model.train()
    
    # Set the SMDebug hook for the training phase
    if hook:
        hook.set_mode(modes.TRAIN)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Save tensors during training phase
        if batch_idx % 100 == 0:
            if hook:
                # Save loss values
                hook.save_scalar("train_loss", loss.item(), sm_metric=True)
                
                # Save gradients for analysis
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        hook.save_tensor(f"gradients/{name}", param.grad)

# Create a hook to save the metrics
hook = smd.Hook(
    out_dir='/opt/ml/output/tensors',  # SageMaker will upload this to S3
    export_tensorboard=True,
    save_config=smd.SaveConfig(save_interval=100)
)

# Register the model with the hook
hook.register_module(model)

# Training loop with the hook
for epoch in range(1, epochs + 1):
    train(model, train_loader, criterion, optimizer, epoch, device, hook)
    validate(model, val_loader, criterion, epoch, device, hook)
    ''')
    
    # Debugger config code
    st.markdown("### Setting Up Debugger in SageMaker Training Job")
    
    st.code('''
from sagemaker.debugger import Rule, CollectionConfig, DebuggerHookConfig, TensorBoardOutputConfig
from sagemaker.debugger import rule_configs

# Configure Debugger to monitor the training job
debugger_hook_config = DebuggerHookConfig(
    s3_output_path=f"s3://{bucket}/{prefix}/debug-output",
    collection_configs=[
        CollectionConfig(name="losses", parameters={"save_interval": "50"}),
        CollectionConfig(name="gradients", parameters={"save_interval": "50"}),
        CollectionConfig(name="weights", parameters={"save_interval": "500"})
    ]
)

# Configure built-in rules for monitoring
rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
    Rule.sagemaker(rule_configs.loss_not_decreasing())
]

# Set up TensorBoard output config
tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=f"s3://{bucket}/{prefix}/tensorboard"
)

# Use these configurations when creating your estimator
estimator = PyTorch(
    # ... other parameters ...
    debugger_hook_config=debugger_hook_config,
    rules=rules,
    tensorboard_output_config=tensorboard_output_config
)

estimator.fit()
    ''')
    
    # Best practices
    st.markdown("### Best Practices for Model Debugging")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Common Issues to Monitor
        
        - **Vanishing/Exploding Gradients**
          - Check gradient magnitudes over time
          - Look for NaN or Inf values
          
        - **Overfitting**
          - Monitor train vs validation loss gap
          - Track model complexity metrics
          
        - **Learning Rate Issues**
          - Watch for oscillating loss values
          - Monitor step sizes in parameter updates
          
        - **Poor Initialization**
          - Check initial loss values
          - Monitor early training dynamics
        """)
    
    with col2:
        st.markdown("""
        #### Debugging Actions to Take
        
        - **Save Checkpoints Frequently**
          - Enable automatic checkpointing in SageMaker
          
        - **Visualize Key Metrics**
          - Use SageMaker Studio or TensorBoard integration
          
        - **Start Simple**
          - Begin with smaller models before scaling
          - Use simplified datasets for initial validation
          
        - **Use Profiling**
          - Monitor CPU/GPU utilization
          - Check for bottlenecks in data loading
        """)

# TAB 2: AUTOMATIC MODEL TUNING
with tab2:
    st.header("Amazon SageMaker Automatic Model Tuning")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        SageMaker Automatic Model Tuning finds the best version of your model by running many training jobs
        on your dataset using a range of hyperparameter values.
        
        **Key benefits:**
        - Automatically find optimal hyperparameters
        - Use Bayesian optimization to efficiently explore the hyperparameter space
        - Run multiple training jobs in parallel
        - Integrate hyperparameter tuning into your ML workflow
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/product-marketing/Sagemaker/Products/auto-model-tuning/product-page-diagram_Amazon-SageMaker-Automatic-Model-Tuning_How-it-Works.bb2882c5609a080d2a5111d4ca39c1f1f7360b49.png",
                 caption="SageMaker Automatic Model Tuning Process")
    
    # Interactive demo of hyperparameter tuning
    st.subheader("Interactive Demo: Hyperparameter Tuning")
    
    # Define algorithm selection and problem type
    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox("Select Algorithm", 
                               ["XGBoost", "Random Forest", "Neural Network"])
    with col2:
        problem_type = st.selectbox("Problem Type", 
                                  ["Classification", "Regression"])

    # Define hyperparameters to tune
    st.markdown("### Hyperparameters to Tune")
    
    hyperparams = {}
    if algorithm == "XGBoost":
        col1, col2, col3 = st.columns(3)
        with col1:
            hyperparams["max_depth"] = st.slider("max_depth", 3, 10, 6)
        with col2:
            hyperparams["learning_rate"] = st.select_slider(
                "learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
        with col3:
            hyperparams["n_estimators"] = st.slider("n_estimators", 50, 1000, 100, 50)
            
        # Additional hyperparameters
        col1, col2 = st.columns(2)
        with col1:
            hyperparams["subsample"] = st.slider("subsample", 0.5, 1.0, 0.8, 0.1)
        with col2:
            hyperparams["min_child_weight"] = st.slider("min_child_weight", 1, 10, 1)
            
    elif algorithm == "Random Forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            hyperparams["n_estimators"] = st.slider("n_estimators", 10, 200, 100, 10)
        with col2:
            hyperparams["max_depth"] = st.slider("max_depth", 5, 30, 10)
        with col3:
            hyperparams["min_samples_split"] = st.slider("min_samples_split", 2, 10, 2)
            
    else:  # Neural Network
        col1, col2, col3 = st.columns(3)
        with col1:
            hyperparams["learning_rate"] = st.select_slider(
                "learning_rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.01)
        with col2:
            hyperparams["batch_size"] = st.select_slider(
                "batch_size", options=[16, 32, 64, 128, 256], value=64)
        with col3:
            hyperparams["neurons"] = st.slider("neurons in hidden layer", 64, 512, 128, 64)

    # Tuning job configuration
    st.markdown("### Tuning Job Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_jobs = st.slider("Maximum Training Jobs", 5, 50, 20, 5)
    with col2:
        parallel_jobs = st.slider("Parallel Jobs", 1, 10, 4, 1)
    with col3:
        objective = st.selectbox("Optimization Objective", 
                               ["Accuracy", "F1 Score", "AUC", "RMSE", "MAE"],
                              index=0 if st.session_state.automl_objective == "Accuracy" else 
                                     ["Accuracy", "F1 Score", "AUC", "RMSE", "MAE"].index(st.session_state.automl_objective))
        st.session_state.automl_objective = objective

    # Simulate tuning job
    if st.button("Start Hyperparameter Tuning"):
        with st.spinner("Running hyperparameter tuning..."):
            time.sleep(3)  # Simulate processing time
            
            # Generate simulated tuning results
            np.random.seed(42)
            
            # Generate job results based on selected hyperparameters and algorithm
            jobs_data = []
            
            # Base performance characteristics by algorithm
            algorithm_base = {
                "XGBoost": {"mean": 0.85, "std": 0.05},
                "Random Forest": {"mean": 0.82, "std": 0.06},
                "Neural Network": {"mean": 0.83, "std": 0.07}
            }
            
            base_perf = algorithm_base[algorithm]
            
            # Generate synthetic job results
            for i in range(max_jobs):
                # For XGBoost, slightly better performance generally comes from:
                # - Moderate max_depth (not too deep to avoid overfitting)
                # - Learning rate around 0.05-0.1
                # - Higher n_estimators
                
                # Similar patterns for other algorithms...
                
                if algorithm == "XGBoost":
                    job_hyperparams = {
                        "max_depth": np.random.randint(3, 11),
                        "learning_rate": np.random.choice([0.01, 0.05, 0.1, 0.2, 0.3]),
                        "n_estimators": np.random.randint(50, 1001, 50),
                        "subsample": round(np.random.uniform(0.5, 1.0), 1),
                        "min_child_weight": np.random.randint(1, 11)
                    }
                    
                    # Adjust score based on how "optimal" these hyperparameters are
                    # (based on general XGBoost best practices)
                    score_adjust = 0
                    
                    # Better with moderate depth
                    if 4 <= job_hyperparams["max_depth"] <= 7:
                        score_adjust += 0.03
                    
                    # Better with moderate learning rate
                    if 0.05 <= job_hyperparams["learning_rate"] <= 0.1:
                        score_adjust += 0.02
                        
                    # Better with more trees (up to a point)
                    if job_hyperparams["n_estimators"].all() >= 200:
                        score_adjust += 0.01
                        
                elif algorithm == "Random Forest":
                    job_hyperparams = {
                        "n_estimators": np.random.randint(10, 201, 10),
                        "max_depth": np.random.randint(5, 31),
                        "min_samples_split": np.random.randint(2, 11)
                    }
                    
                    score_adjust = 0
                    # Random Forest generally improves with more trees
                    if job_hyperparams["n_estimators"] >= 100:
                        score_adjust += 0.02
                        
                    # Moderate depth often works well
                    if 10 <= job_hyperparams["max_depth"] <= 20:
                        score_adjust += 0.01
                        
                else:  # Neural Network
                    job_hyperparams = {
                        "learning_rate": np.random.choice([0.0001, 0.001, 0.01, 0.1]),
                        "batch_size": np.random.choice([16, 32, 64, 128, 256]),
                        "neurons": np.random.randint(64, 513, 64)
                    }
                    
                    score_adjust = 0
                    # Neural networks often perform better with learning rate around 0.001
                    if job_hyperparams["learning_rate"] == 0.001:
                        score_adjust += 0.03
                        
                    # Middle-range batch sizes often work well
                    if job_hyperparams["batch_size"] in [32, 64]:
                        score_adjust += 0.01
                
                # Generate metrics
                if problem_type == "Classification":
                    if objective == "Accuracy" or objective == "F1 Score" or objective == "AUC":
                        # Generate a score between 0-1
                        score = min(0.99, base_perf["mean"] + score_adjust + np.random.normal(0, base_perf["std"]))
                        score = max(0.5, score)  # Ensure score isn't too low
                    else:
                        # Error metrics - lower is better
                        score = max(0.05, 0.2 - score_adjust + np.random.normal(0, 0.05))
                else:  # Regression
                    if objective == "RMSE" or objective == "MAE":
                        # Error metrics - lower is better
                        score = max(0.2, 1.0 - score_adjust + np.random.normal(0, 0.2))
                    else:
                        # R2 or similar - higher is better
                        score = min(0.95, 0.7 + score_adjust + np.random.normal(0, 0.1))
                
                jobs_data.append({
                    "job_id": f"tuning-job-{i+1}",
                    "hyperparameters": job_hyperparams,
                    "objective_value": score,
                    "job_status": "Completed",
                    "training_time": round(np.random.uniform(60, 300), 1)  # seconds
                })
            
            # Sort jobs by objective value (ascending/descending based on objective type)
            if objective in ["RMSE", "MAE"]:
                # Lower is better
                jobs_data.sort(key=lambda x: x["objective_value"])
            else:
                # Higher is better
                jobs_data.sort(key=lambda x: x["objective_value"], reverse=True)
            
            # Store in session state
            st.session_state.tuning_jobs = jobs_data
            
        st.success(f"Hyperparameter tuning completed with {max_jobs} jobs!")
    
    # Display tuning results if available
    if 'tuning_jobs' in st.session_state:
        jobs_data = st.session_state.tuning_jobs
        
        # Best job and hyperparameters
        best_job = jobs_data[0]
        
        st.markdown("### Tuning Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Best Hyperparameters")
            best_params = pd.DataFrame({
                "Hyperparameter": list(best_job["hyperparameters"].keys()),
                "Value": list(best_job["hyperparameters"].values())
            })
            st.dataframe(best_params, hide_index=True)
        
        with col2:
            st.markdown("#### Optimization Results")
            metric_text = f"{objective}" if objective not in ["RMSE", "MAE"] else f"{objective} (lower is better)"
            st.metric(metric_text, f"{best_job['objective_value']:.4f}")
            st.metric("Training Time", f"{best_job['training_time']} seconds")
            st.metric("Jobs Completed", len(jobs_data))
        
        # Objective value history chart
        st.markdown("### Optimization History")
        
        # Prepare data for chart
        history_data = []
        for i, job in enumerate(jobs_data):
            history_data.append({
                "Job Number": i + 1,
                "Objective Value": job["objective_value"]
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Create chart with highlighting for best job
        chart = alt.Chart(history_df).mark_line(point=True).encode(
            x=alt.X('Job Number:Q', title='Job Number'),
            y=alt.Y('Objective Value:Q', title=objective)
        ).properties(
            width=700,
            height=400,
            title=f"{objective} Progression Over Tuning Jobs"
        )
        
        # Add a horizontal line for best value
        best_value_line = alt.Chart(pd.DataFrame({'y': [best_job["objective_value"]]})).mark_rule(
            color=AWS_COLORS["orange"], 
            strokeDash=[4, 4]
        ).encode(
            y='y'
        )
        
        st.altair_chart(chart + best_value_line, use_container_width=True)
        
        # Hyperparameter importance visualization
        st.markdown("### Hyperparameter Importance")
        
        # Generate simulated importance scores
        hyperparameters = list(best_job["hyperparameters"].keys())
        if algorithm == "XGBoost":
            importances = {
                "max_depth": 0.35,
                "learning_rate": 0.30,
                "n_estimators": 0.20,
                "subsample": 0.10,
                "min_child_weight": 0.05
            }
        elif algorithm == "Random Forest":
            importances = {
                "n_estimators": 0.45,
                "max_depth": 0.40,
                "min_samples_split": 0.15
            }
        else:  # Neural Network
            importances = {
                "learning_rate": 0.50,
                "batch_size": 0.30,
                "neurons": 0.20
            }
        
        importance_df = pd.DataFrame({
            "Hyperparameter": list(importances.keys()),
            "Importance": list(importances.values())
        }).sort_values("Importance", ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(importance_df["Hyperparameter"], importance_df["Importance"], 
                      color=AWS_COLORS["teal"])
        
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importance")
        
        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f"{width:.2f}", va="center")
        
        st.pyplot(fig)
        
        # Hyperparameter relationship visualization
        st.markdown("### Hyperparameter Relationships")
        
        # Show parallel coordinates plot for top jobs
        top_k = min(10, len(jobs_data))
        top_jobs = jobs_data[:top_k]
        
        # Prepare data for parallel coordinates
        parallel_data = []
        for job in top_jobs:
            job_data = {"job_id": job["job_id"], "objective": job["objective_value"]}
            job_data.update(job["hyperparameters"])
            parallel_data.append(job_data)
        
        parallel_df = pd.DataFrame(parallel_data)
        
        # Normalize hyperparameter values for better visualization
        columns_to_normalize = list(best_job["hyperparameters"].keys())
        for col in columns_to_normalize:
            # Convert to numeric, setting invalid parsing as NaN
            parallel_df[col] = pd.to_numeric(parallel_df[col], errors='coerce')
            
            min_val = parallel_df[col].min()
            max_val = parallel_df[col].max()
            if min_val != max_val:
                parallel_df[f"{col}_normalized"] = (parallel_df[col] - min_val) / (max_val - min_val)
            else:
                parallel_df[f"{col}_normalized"] = 0.5
        
        # Create parallel coordinates plot
        if len(columns_to_normalize) >= 2:  # Need at least 2 dimensions
            norm_columns = [f"{col}_normalized" for col in columns_to_normalize]
            
            # Create long format data for parallel coordinates
            melted_df = pd.melt(
                parallel_df,
                id_vars=["job_id", "objective"],
                value_vars=norm_columns,
                var_name="parameter",
                value_name="value"
            )
            
            # Clean parameter names for display
            melted_df["parameter"] = melted_df["parameter"].str.replace("_normalized", "")
            
            # Create parallel coordinates plot with coloring by objective value
            if objective in ["RMSE", "MAE"]:
                color_scale = "viridis_r"  # Reversed so dark blue is best (lowest) value
            else:
                color_scale = "viridis"  # Dark blue is best (highest) value
                
            lines = alt.Chart(melted_df).mark_line().encode(
                x=alt.X("parameter:N", title=None),
                y=alt.Y("value:Q", title="Normalized Value", axis=alt.Axis(labels=False)),
                color=alt.Color("objective:Q", scale=alt.Scale(scheme=color_scale),
                              legend=alt.Legend(title=objective)),
                detail="job_id:N",
                strokeWidth=alt.value(2),
                tooltip=["job_id", "objective"]
            ).properties(
                width=700,
                height=400,
                title=f"Parallel Coordinates Plot for Top {top_k} Jobs"
            )
            
            # Add points to make it clearer
            points = alt.Chart(melted_df).mark_circle(size=50).encode(
                x="parameter:N",
                y="value:Q",
                color=alt.Color("objective:Q", scale=alt.Scale(scheme=color_scale)),
                tooltip=["job_id", "objective", "parameter", "value"]
            )
            
            st.altair_chart(lines + points, use_container_width=True)
        else:
            st.warning("Need at least 2 hyperparameters for relationship visualization.")
            
        # Show job results table
        st.markdown("### All Tuning Jobs")
        
        jobs_table = []
        for job in jobs_data:
            job_row = {
                "Job ID": job["job_id"],
                "Status": job["status"] if "status" in job else "Completed",
                f"{objective}": f"{job['objective_value']:.4f}",
                "Training Time (s)": job["training_time"]
            }
            
            # Add hyperparameters
            for param, value in job["hyperparameters"].items():
                job_row[param] = value
                
            jobs_table.append(job_row)
            
        jobs_df = pd.DataFrame(jobs_table)
        st.dataframe(jobs_df, hide_index=True)
    
    # Code example
    st.markdown("### Implementing Automatic Model Tuning")
    st.markdown("Here's how to set up hyperparameter tuning with SageMaker:")
    
    st.code('''
from sagemaker.tuner import HyperparameterTuner
from sagemaker.estimator import Estimator

# Create an estimator
xgb = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/{prefix}/output'
)

# Set static hyperparameters
xgb.set_hyperparameters(
    objective='binary:logistic',
    num_round=100,
    eval_metric='auc',
    verbosity=1
)

# Define hyperparameter ranges to search
hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'eta': ContinuousParameter(0.01, 0.3),
    'min_child_weight': IntegerParameter(1, 10),
    'subsample': ContinuousParameter(0.5, 1.0),
    'colsample_bytree': ContinuousParameter(0.5, 1.0)
}

# Define objective metric to optimize
objective_metric_name = 'validation:auc'
objective_type = 'Maximize'

# Create the hyperparameter tuner
tuner = HyperparameterTuner(
    xgb,
    objective_metric_name,
    hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=4,
    objective_type=objective_type,
    early_stopping_type='Auto'
)

# Start the hyperparameter tuning job
tuner.fit(
    {'train': train_data, 'validation': validation_data},
    include_cls_metadata=False
)

# After tuning is complete, get the best model
best_training_job = tuner.best_training_job()
best_model = tuner.create_model(
    name=best_training_job,
    role=role
)

# Deploy the best model
predictor = best_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)
    ''')
    
    # Advanced tuning strategies
    st.markdown("### Advanced Tuning Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Bayesian Optimization
        
        SageMaker uses Bayesian optimization to efficiently search the hyperparameter space:
        
        1. **Build a surrogate model** of the objective function
        2. **Find hyperparameters** that perform best on surrogate
        3. **Apply these hyperparameters** to the real model
        4. **Update the surrogate model** with new results
        5. **Repeat** until maximum jobs are reached
        
        This approach is much more efficient than grid or random search!
        """)
        
        # Add a visual for Bayesian optimization
        st.image("https://miro.medium.com/max/1400/1*FwMRqHc3RVPkL-J8n0VE-A.png", 
                caption="Bayesian Optimization Process")
    
    with col2:
        st.markdown("""
        #### Tuning Best Practices
        
        - **Choose ranges carefully** - too wide wastes resources
        - **Consider early stopping** to save compute resources
        - **Use warm start** to leverage knowledge from previous tuning jobs
        - **Start with low max_jobs** to validate job configuration
        - **Log additional metrics** beyond just the objective metric
        - **Scale hyperparameters** if they vary by orders of magnitude
        
        #### Sample Hyperparameters by Algorithm
        
        - **XGBoost**: max_depth, learning_rate, min_child_weight
        - **Linear Learner**: learning_rate, l1, wd
        - **Neural Networks**: learning_rate, batch_size, layer_size
        """)
    
# TAB 3: TRAINING COMPILER
with tab3:
    st.header("Amazon SageMaker Training Compiler")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        SageMaker Training Compiler accelerates deep learning training by optimizing 
        model computation graphs and efficient GPU usage.
        
        **Key benefits:**
        - Accelerate training without code changes
        - Optimize memory usage for larger models
        - Support for popular deep learning frameworks
        - Lower training costs through faster completion
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/sagemaker/sagemaker-training-compiler/SageMaker-Training-Compiler-HIW.4fbdd3ef40a86e31cfc5af651c4e15a62c0b9141.png",
                 caption="SageMaker Training Compiler Overview")
    
    # Compiler model selection and demo
    st.subheader("Interactive Demo: Training Compiler Speed Improvements")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        model_options = ["BERT", "ResNet", "GPT-2", "Vision Transformer"]
        compiler_model = st.selectbox("Select model architecture:", model_options, index=model_options.index(st.session_state.compiler_model))
        st.session_state.compiler_model = compiler_model
    
    with col2:
        batch_size = st.select_slider(
            "Batch Size", 
            options=[8, 16, 32, 64, 128],
            value=st.session_state.batch_size
        )
        st.session_state.batch_size = batch_size
        
    with col3:
        precision = st.selectbox("Training Precision", ["float32", "float16", "bfloat16"])
    
    # Function to generate training performance data
    def generate_compiler_performance_data(model_name, batch_size, precision):
        # Base times in seconds per epoch (without compiler)
        base_times = {
            "BERT": 980,
            "ResNet": 650,
            "GPT-2": 1450,
            "Vision Transformer": 1100
        }
        
        # Base memory usage in GB (without compiler)
        base_memory = {
            "BERT": 14.2,
            "ResNet": 9.8,
            "GPT-2": 16.5,
            "Vision Transformer": 12.6
        }
        
        # Speedup factors for compiler
        # Models like BERT and GPT-2 typically see bigger gains from compiler
        compiler_speedup = {
            "BERT": 1.4,
            "ResNet": 1.25,
            "GPT-2": 1.5,
            "Vision Transformer": 1.3
        }
        
        # Memory efficiency improvement
        memory_improvement = {
            "BERT": 0.85,
            "ResNet": 0.9,
            "GPT-2": 0.8,
            "Vision Transformer": 0.85
        }
        
        # Batch size effect (larger batch sizes generally show better relative improvement)
        batch_factor = {
            8: 0.9,
            16: 0.95,
            32: 1.0,
            64: 1.05,
            128: 1.1
        }
        
        # Precision effect (lower precision shows better relative improvement)
        precision_factor = {
            "float32": 1.0,
            "float16": 1.15,
            "bfloat16": 1.12
        }
        
        # Calculate performance metrics
        base_time = base_times[model_name]
        base_mem = base_memory[model_name]
        
        # Apply factors
        compiler_time = base_time / (compiler_speedup[model_name] * batch_factor[batch_size] * precision_factor[precision])
        compiler_mem = base_mem * memory_improvement[model_name]
        
        # Add a small random variation
        np.random.seed(42)
        compiler_time *= (1 + np.random.normal(0, 0.05))
        base_time *= (1 + np.random.normal(0, 0.05))
        
        # Throughput (samples/sec)
        samples_per_epoch = 50000  # Assuming a dataset of this size
        base_throughput = samples_per_epoch / base_time
        compiler_throughput = samples_per_epoch / compiler_time
        
        # Return results
        return {
            "base_time": base_time,
            "compiler_time": compiler_time,
            "base_memory": base_mem,
            "compiler_memory": compiler_mem,
            "base_throughput": base_throughput,
            "compiler_throughput": compiler_throughput,
            "time_reduction_percent": (base_time - compiler_time) / base_time * 100,
            "memory_reduction_percent": (base_mem - compiler_mem) / base_mem * 100,
            "throughput_increase_percent": (compiler_throughput - base_throughput) / base_throughput * 100
        }
    
    # Run compiler analysis button
    if st.button("Run Performance Analysis"):
        with st.spinner("Analyzing performance with and without compiler..."):
            time.sleep(2)  # Simulating analysis
            
            performance_data = generate_compiler_performance_data(compiler_model, batch_size, precision)
            st.session_state.compiler_performance = performance_data
            
        st.success("Analysis complete! See results below.")
        
    # Show performance comparison results
    if 'compiler_performance' in st.session_state:
        performance_data = st.session_state.compiler_performance
        
        st.markdown("### Performance Comparison")
        
        # Display metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Time per Epoch")
            
            # Create comparison chart for training time
            fig, ax = plt.subplots(figsize=(10, 5))
            bar_width = 0.35
            index = np.array([0, 1])
            
            bars1 = ax.bar(index[0], performance_data["base_time"], bar_width, 
                          label='Without Compiler', color=AWS_COLORS["blue"])
            bars2 = ax.bar(index[1], performance_data["compiler_time"], bar_width,
                          label='With Compiler', color=AWS_COLORS["orange"])
            
            # Add some text for labels, title and axes ticks
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'Training Time per Epoch - {compiler_model}')
            ax.set_xticks(index)
            ax.set_xticklabels(['Without Compiler', 'With Compiler'])
            
            # Add value annotations
            for bar in [bars1[0], bars2[0]]:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}s',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            ax.legend()
            
            st.pyplot(fig)
            
            # Show time reduction
            st.metric("Training Time Reduction", 
                      f"{performance_data['time_reduction_percent']:.1f}%", 
                      f"{performance_data['time_reduction_percent']:.1f}%")
        
        with col2:
            st.markdown("#### Training Throughput")
            
            # Create comparison chart for throughput
            fig, ax = plt.subplots(figsize=(10, 5))
            
            bars1 = ax.bar(index[0], performance_data["base_throughput"], bar_width, 
                          label='Without Compiler', color=AWS_COLORS["blue"])
            bars2 = ax.bar(index[1], performance_data["compiler_throughput"], bar_width,
                          label='With Compiler', color=AWS_COLORS["orange"])
            
            ax.set_ylabel('Samples per second')
            ax.set_title(f'Training Throughput - {compiler_model}')
            ax.set_xticks(index)
            ax.set_xticklabels(['Without Compiler', 'With Compiler'])
            
            # Add value annotations
            for bar in [bars1[0], bars2[0]]:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            ax.legend()
            
            st.pyplot(fig)
            
            # Show throughput increase
            st.metric("Throughput Improvement", 
                      f"{performance_data['throughput_increase_percent']:.1f}%", 
                      f"{performance_data['throughput_increase_percent']:.1f}%")
        
        # Memory usage comparison
        st.markdown("### Memory Usage Comparison")
        
        # Create comparison chart for memory
        fig, ax = plt.subplots(figsize=(10, 5))
        
        bars1 = ax.bar(index[0], performance_data["base_memory"], bar_width, 
                      label='Without Compiler', color=AWS_COLORS["blue"])
        bars2 = ax.bar(index[1], performance_data["compiler_memory"], bar_width,
                      label='With Compiler', color=AWS_COLORS["orange"])
        
        ax.set_ylabel('GPU Memory (GB)')
        ax.set_title(f'Memory Usage - {compiler_model}')
        ax.set_xticks(index)
        ax.set_xticklabels(['Without Compiler', 'With Compiler'])
        
        # Add value annotations
        for bar in [bars1[0], bars2[0]]:
            height = bar.get_height()
            ax.annotate(f'{height:.1f} GB',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.legend()
        
        st.pyplot(fig)
        
        # Project to complete training job
        st.markdown("### Projected Full Training Job")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Assume 50 epochs for full training
            total_epochs = 50
            base_total_time = performance_data["base_time"] * total_epochs / 3600  # convert to hours
            compiler_total_time = performance_data["compiler_time"] * total_epochs / 3600  # convert to hours
            
            # Create a comparison chart
            fig, ax = plt.subplots(figsize=(10, 5))
            
            bars1 = ax.bar(index[0], base_total_time, bar_width, 
                          label='Without Compiler', color=AWS_COLORS["blue"])
            bars2 = ax.bar(index[1], compiler_total_time, bar_width,
                          label='With Compiler', color=AWS_COLORS["orange"])
            
            ax.set_ylabel('Time (hours)')
            ax.set_title(f'Total Training Time (50 epochs) - {compiler_model}')
            ax.set_xticks(index)
            ax.set_xticklabels(['Without Compiler', 'With Compiler'])
            
            # Add value annotations
            for bar in [bars1[0], bars2[0]]:
                height = bar.get_height()
                ax.annotate(f'{height:.1f} hrs',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            ax.legend()
            
            st.pyplot(fig)
            
            time_saved = base_total_time - compiler_total_time
            st.metric("Total Training Time Saved", 
                      f"{time_saved:.1f} hours", 
                      f"{time_saved:.1f} hours")
        
        with col2:
            # Cost estimation (assuming a typical GPU instance cost)
            hourly_rate = 3.06  # p3.2xlarge hourly rate
            
            base_cost = base_total_time * hourly_rate
            compiler_cost = compiler_total_time * hourly_rate
            cost_saved = base_cost - compiler_cost
            
            # Create a comparison chart
            fig, ax = plt.subplots(figsize=(10, 5))
            
            bars1 = ax.bar(index[0], base_cost, bar_width, 
                          label='Without Compiler', color=AWS_COLORS["blue"])
            bars2 = ax.bar(index[1], compiler_cost, bar_width,
                          label='With Compiler', color=AWS_COLORS["orange"])
            
            ax.set_ylabel('Cost ($)')
            ax.set_title(f'Estimated Training Cost (p3.2xlarge) - {compiler_model}')
            ax.set_xticks(index)
            ax.set_xticklabels(['Without Compiler', 'With Compiler'])
            
            # Add value annotations
            for bar in [bars1[0], bars2[0]]:
                height = bar.get_height()
                ax.annotate(f'${height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            ax.legend()
            
            st.pyplot(fig)
            
            st.metric("Estimated Cost Savings", 
                      f"${cost_saved:.2f}", 
                      f"{(cost_saved / base_cost * 100):.1f}%")
    
    # Under the hood: How Training Compiler works
    st.markdown("### How Training Compiler Works")
    
    # Visualization of compiler optimization process
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        SageMaker Training Compiler optimizes your deep learning model through several stages:
        
        1. **Analysis**: Examines the computational graph of your model
        2. **Optimization**: Applies multiple techniques to the graph:
            - Operator fusion
            - Memory management optimization
            - Kernel tuning
            - Precision calibration
        3. **Code Generation**: Generates optimized code for target hardware
        4. **Efficient Execution**: Runs the optimized code on GPUs
        
        These optimizations result in faster training times and better resource utilization.
        """)
    
    with col2:
        # Create a sequential diagram showing the optimization process
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Background
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')  # Hide axes
        
        # Define boxes for the flow chart
        boxes = [
            {"name": "Original\nModel", "x": 1, "y": 3, "width": 1.5, "height": 1, "color": AWS_COLORS["blue"]},
            {"name": "Graph\nAnalysis", "x": 3, "y": 3, "width": 1.5, "height": 1, "color": AWS_COLORS["teal"]},
            {"name": "Optimization\nPhase", "x": 5, "y": 3, "width": 1.5, "height": 1, "color": AWS_COLORS["orange"]},
            {"name": "Code\nGeneration", "x": 7, "y": 3, "width": 1.5, "height": 1, "color": AWS_COLORS["green"]},
            {"name": "Optimized\nExecution", "x": 9, "y": 3, "width": 1.5, "height": 1, "color": AWS_COLORS["red"]}
        ]
        
        # Draw boxes and add labels
        for box in boxes:
            rect = plt.Rectangle((box["x"] - box["width"]/2, box["y"] - box["height"]/2), 
                                box["width"], box["height"], 
                                facecolor=box["color"], edgecolor="white", alpha=0.8)
            ax.add_patch(rect)
            ax.text(box["x"], box["y"], box["name"], ha="center", va="center", color="white", fontweight="bold")
        
        # Add arrows connecting the boxes
        for i in range(len(boxes)-1):
            ax.annotate("", 
                      xy=(boxes[i+1]["x"] - boxes[i+1]["width"]/2, boxes[i+1]["y"]),
                      xytext=(boxes[i]["x"] + boxes[i]["width"]/2, boxes[i]["y"]),
                      arrowprops=dict(arrowstyle="->", lw=2, color=AWS_COLORS["dark_gray"]))
        
        # Add optimization details
        opt_details = [
            "Operator Fusion",
            "Memory Planning",
            "Kernel Tuning", 
            "Parallelization"
        ]
        
        for i, detail in enumerate(opt_details):
            y_pos = 1.5 - i * 0.5
            ax.text(5, y_pos, f"• {detail}", ha="center", va="center", fontsize=9)
            # Connect to optimization box with a light line
            ax.plot([5, 5], [y_pos + 0.1, 2.5], color=AWS_COLORS["dark_gray"], linestyle=":", linewidth=1)
        
        # Add a title
        ax.text(5, 5.5, "SageMaker Training Compiler Workflow", ha="center", fontsize=14, fontweight="bold")
        
        st.pyplot(fig)
    
    # Code examples
    st.markdown("### Implementing Training Compiler")
    st.markdown("Here's how to enable Training Compiler in your SageMaker training job:")
    
    # PyTorch example
    st.markdown("**PyTorch Example**")
    st.code('''
from sagemaker.pytorch import PyTorch

# Define the PyTorch estimator with compiler_config
pytorch_estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU instance
    framework_version='1.13.1',
    py_version='py38',
    hyperparameters={
        'batch-size': 64,
        'epochs': 50,
        'learning-rate': 0.001
    },
    # Enable Training Compiler
    compiler_config={
        'enabled': True
    }
)

# Start the training job
pytorch_estimator.fit({'train': train_data_uri, 'validation': validation_data_uri})
    ''')
    
    # Hugging Face example
    st.markdown("**Hugging Face Transformer Example**")
    st.code('''
from sagemaker.huggingface import HuggingFace

# Define the Hugging Face estimator with compiler_config
huggingface_estimator = HuggingFace(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    transformers_version='4.17',
    pytorch_version='1.10',
    hyperparameters={
        'model_name_or_path': 'bert-base-uncased',
        'task_name': 'sst2',
        'per_device_train_batch_size': 32,
        'learning_rate': 3e-5,
        'num_train_epochs': 3
    },
    # Enable Training Compiler
    compiler_config={
        'enabled': True
    }
)

# Start the training job
huggingface_estimator.fit({'train': train_data_uri, 'validation': validation_data_uri})
    ''')
    
    # Framework compatibility
    st.markdown("### Framework and Model Compatibility")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Supported Frameworks")
        
        frameworks = {
            "PyTorch": ["1.10.x", "1.11.x", "1.12.x", "1.13.x"],
            "TensorFlow": ["2.9.x", "2.10.x", "2.11.x"],
            "Hugging Face Transformers": ["4.17.x", "4.21.x", "4.23.x"]
        }
        
        for fw, versions in frameworks.items():
            st.markdown(f"**{fw}**")
            for v in versions:
                st.markdown(f"- {v}")
    
    with col2:
        st.markdown("#### Supported Model Architectures")
        
        models = [
            "BERT",
            "GPT-2",
            "ResNet",
            "Vision Transformer",
            "RoBERTa",
            "DistilBERT",
            "MobileNet",
            "DeiT",
            "T5"
        ]
        
        for model in models:
            st.markdown(f"- {model}")
    
    # Best practices
    st.markdown("### Best Practices for Maximum Speedup")
    
    st.markdown("""
    To get the most from SageMaker Training Compiler:
    
    1. **Use supported instance types**: ml.p3, ml.p4d, ml.g4dn, ml.g5
    2. **Mixed precision training**: Use float16 or bfloat16 precision for larger speedups
    3. **Batch size optimization**: Use larger batch sizes when possible
    4. **Avoid unsupported operators**: Some custom operators may not be optimized
    5. **Benchmark your specific model**: Performance gains vary by model architecture
    6. **Monitor metrics**: Check both training time and accuracy metrics
    """)

# TAB 4: DISTRIBUTED TRAINING

# Function to generate distributed scaling data
def generate_distributed_scaling_data(strategy, model_size_m, num_instances, gpus_per_instance):
    np.random.seed(42)
    total_gpus = num_instances * gpus_per_instance
    
    # Base time in minutes for single-GPU training
    if model_size_m < 200:
        base_time = 120  # 2 hours for small models
    elif model_size_m < 500:
        base_time = 240  # 4 hours for medium models
    else:
        base_time = 480  # 8 hours for large models
    
    # Scaling efficiency depends on strategy and number of GPUs
    if strategy == "Data Parallel":
        # Good scaling for moderate GPU counts, diminishing returns at higher counts
        efficiency = {}
        for i in range(1, total_gpus + 1):
            if i == 1:
                efficiency[i] = 1.0
            elif i <= 8:
                efficiency[i] = 0.9 - (i-1) * 0.01
            elif i <= 16:
                efficiency[i] = 0.82 - (i-8) * 0.015
            else:
                efficiency[i] = 0.7 - (i-16) * 0.02
    
    elif strategy == "Model Parallel":
        # Less efficient at low GPU counts, better for larger models
        efficiency = {}
        for i in range(1, total_gpus + 1):
            if i == 1:
                efficiency[i] = 1.0
            elif i <= 4:
                efficiency[i] = 0.7  # Initial overhead of model parallelism
            elif i <= 8:
                efficiency[i] = 0.7 + (i-4) * 0.02  # Improving as model parts distributed
            elif i <= 16:
                efficiency[i] = 0.78 + (i-8) * 0.01
            else:
                efficiency[i] = 0.86 + (i-16) * 0.005
    
    else:  # Hybrid Parallel
        # Balance between the two approaches
        efficiency = {}
        for i in range(1, total_gpus + 1):
            if i == 1:
                efficiency[i] = 1.0
            elif i <= 8:
                efficiency[i] = 0.85 - (i-1) * 0.005
            elif i <= 16:
                efficiency[i] = 0.815 - (i-8) * 0.01
            else:
                efficiency[i] = 0.735 - (i-16) * 0.015
    
    # Calculate actual speedup and time with the selected number of GPUs
    perfect_speedup = total_gpus
    actual_speedup = total_gpus * efficiency[total_gpus]
    
    # Add some small random variation
    actual_speedup *= (1 + np.random.normal(0, 0.03))
    
    distributed_time = base_time / actual_speedup
    
    # Calculate speedup for various GPU counts
    gpu_counts = list(range(1, min(total_gpus + 5, 41)))
    speedups = []
    times = []
    efficiencies = []
    
    for count in gpu_counts:
        if count <= total_gpus:
            eff = efficiency[count]
        else:
            # Extrapolate efficiency for higher GPU counts (for chart visualization)
            last_eff = efficiency[total_gpus]
            decline_rate = (1.0 - last_eff) / total_gpus
            eff = max(0.5, last_eff - decline_rate * (count - total_gpus))
            
        speedup = count * eff
        time_mins = base_time / speedup
        
        speedups.append(speedup)
        times.append(time_mins)
        efficiencies.append(eff * 100)  # Convert to percentage
    
    # Communication overhead data (as percentage of total time)
    comm_overhead = []
    for count in gpu_counts:
        if count == 1:
            comm_overhead.append(0)  # No communication overhead with 1 GPU
        else:
            if strategy == "Data Parallel":
                # Communication increases with more GPUs
                overhead = 5 + (count - 1) * 2
            elif strategy == "Model Parallel":
                # Higher initial overhead, increases but then stabilizes
                overhead = 15 + min(25, (count - 1) * 1.5)
            else:  # Hybrid
                overhead = 10 + min(20, (count - 1) * 1.2)
            
            comm_overhead.append(min(50, overhead))  # Cap at 50%
    
    return {
        "strategy": strategy,
        "model_size_m": model_size_m,
        "num_instances": num_instances,
        "gpus_per_instance": gpus_per_instance,
        "total_gpus": total_gpus,
        "base_time_mins": base_time,
        "distributed_time_mins": distributed_time,
        "perfect_speedup": perfect_speedup,
        "actual_speedup": actual_speedup,
        "scaling_efficiency": efficiency[total_gpus] * 100,  # as percentage
        "gpu_counts": gpu_counts,
        "speedups": speedups,
        "times": times,
        "efficiencies": efficiencies,
        "comm_overhead": comm_overhead
    }

with tab4:
    st.header("Amazon SageMaker Distributed Training")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        SageMaker Distributed Training libraries help you efficiently train large models
        across multiple GPUs and multiple nodes.
        
        **Key benefits:**
        - Scale training to multiple nodes efficiently
        - Minimize communication overhead
        - Support for data parallelism and model parallelism
        - Reduce training time for large models
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/diagrams/sagemaker/distributed-data-parallelism-how-it-works.46ac084d2074123ae433b77f6e4b94a49e659e6f.png",
                 caption="SageMaker Distributed Training")
    
    # Interactive demo for distributed training
    st.subheader("Interactive Demo: Distributed Training Scaling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training strategy selection
        strategy_options = ["Data Parallel", "Model Parallel", "Hybrid Parallel"]
        distributed_strategy = st.selectbox("Training Strategy", strategy_options,
                                          index=strategy_options.index(st.session_state.distributed_training_strategy)
                                          if st.session_state.distributed_training_strategy in strategy_options else 0)
        st.session_state.distributed_training_strategy = distributed_strategy
        
        # Model selection
        model_options = ["BERT (110M params)", "GPT-2 Small (117M params)", 
                        "GPT-2 Medium (345M params)", "GPT-2 Large (774M params)"]
        distributed_model = st.selectbox("Model Architecture", model_options)
        
        # Extract model parameter count for reference
        model_params = int(distributed_model.split("(")[1].split("M")[0])
        
        # Explain selected strategy
        strategy_explanations = {
            "Data Parallel": """
            **Data Parallelism** distributes batches of training data across multiple GPUs, 
            with each GPU maintaining a complete copy of the model. Gradients are 
            synchronized across devices after each forward and backward pass.
            
            **Best for**: Models that fit in a single GPU's memory.
            """,
            
            "Model Parallel": """
            **Model Parallelism** splits the model itself across multiple GPUs,
            with each GPU holding a portion of the model's layers or parameters.
            This allows training models that are too large for a single GPU.
            
            **Best for**: Very large models that don't fit in a single GPU's memory.
            """,
            
            "Hybrid Parallel": """
            **Hybrid Parallelism** combines both data and model parallelism techniques,
            distributing both the data and the model across GPUs to optimize for both
            memory usage and computation efficiency.
            
            **Best for**: Balancing memory constraints with computational efficiency.
            """
        }
        
        st.markdown(f"### {distributed_strategy}")
        st.markdown(strategy_explanations[distributed_strategy])
    
    with col2:
        # Number of nodes/GPUs
        num_instances = st.slider("Number of Instances", 1, 10, st.session_state.num_instances)
        st.session_state.num_instances = num_instances
        
        num_gpus_per_instance = st.slider("GPUs per Instance", 1, 8, 4)
        total_gpus = num_instances * num_gpus_per_instance
        
        st.metric("Total GPUs", f"{total_gpus}")
        
        # Instance type selection
        instance_options = ["ml.p3.2xlarge (1 GPU)", "ml.p3.8xlarge (4 GPUs)", 
                          "ml.p3.16xlarge (8 GPUs)", "ml.p4d.24xlarge (8 GPUs)"]
        instance_type = st.selectbox("Instance Type", instance_options)
        
        # Run scaling test button
        if st.button("Run Scaling Analysis"):
            with st.spinner("Analyzing distributed training performance..."):
                time.sleep(2)  # Simulate analysis
                
                # Generate scaling data based on selected parameters
                scaling_data = generate_distributed_scaling_data(
                    distributed_strategy, 
                    model_params,
                    num_instances, 
                    num_gpus_per_instance
                )
                
                st.session_state.scaling_data = scaling_data
                
            st.success("Analysis complete! See results below.")
    

    
    # Display scaling results
    if 'scaling_data' in st.session_state:
        scaling_data = st.session_state.scaling_data
        
        st.markdown("### Distributed Training Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Training time comparison
            st.metric(
                "Single GPU Training Time", 
                f"{scaling_data['base_time_mins'] / 60:.1f} hours"
            )
        
        with col2:
            # Distributed time
            st.metric(
                f"{scaling_data['total_gpus']} GPU Training Time", 
                f"{scaling_data['distributed_time_mins'] / 60:.1f} hours",
                f"-{(scaling_data['base_time_mins'] - scaling_data['distributed_time_mins']) / 60:.1f} hours"
            )
        
        with col3:
            # Scaling efficiency
            st.metric(
                "Scaling Efficiency", 
                f"{scaling_data['scaling_efficiency']:.1f}%"
            )
        
        # Speedup visualization
        st.markdown("### Speedup vs. GPU Count")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Perfect scaling line
        ax.plot(scaling_data['gpu_counts'], scaling_data['gpu_counts'], 
               label="Perfect Linear Scaling", 
               linestyle="--", color=AWS_COLORS['blue'], alpha=0.7)
        
        # Actual speedup line
        ax.plot(scaling_data['gpu_counts'], scaling_data['speedups'], 
               label=f"Actual Speedup ({scaling_data['strategy']})", 
               marker='o', color=AWS_COLORS['orange'])
        
        # Highlight current configuration
        current_gpu_index = scaling_data['gpu_counts'].index(scaling_data['total_gpus'])
        ax.scatter([scaling_data['total_gpus']], [scaling_data['speedups'][current_gpu_index]], 
                  s=100, color=AWS_COLORS['red'], 
                  label=f"Current Configuration ({scaling_data['total_gpus']} GPUs)")
        
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Speedup Factor")
        ax.set_title(f"Training Speedup vs. GPU Count - {scaling_data['strategy']}")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        st.pyplot(fig)
        
        # Scaling efficiency chart
        st.markdown("### Scaling Efficiency")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scaling efficiency line
        ax.plot(scaling_data['gpu_counts'], scaling_data['efficiencies'], 
               marker='o', color=AWS_COLORS['green'])
        
        # Highlight current configuration
        ax.scatter([scaling_data['total_gpus']], [scaling_data['efficiencies'][current_gpu_index]], 
                  s=100, color=AWS_COLORS['red'])
        
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Scaling Efficiency (%)")
        ax.set_title(f"Scaling Efficiency vs. GPU Count - {scaling_data['strategy']}")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 105)
        
        st.pyplot(fig)
        
        # Communication overhead
        st.markdown("### Communication Overhead Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Communication overhead line
        ax.bar(scaling_data['gpu_counts'], scaling_data['comm_overhead'], 
              color=AWS_COLORS['teal'], alpha=0.7)
        
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Communication Overhead (%)")
        ax.set_title(f"Communication Overhead vs. GPU Count - {scaling_data['strategy']}")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Cost analysis
        st.markdown("### Cost Analysis")
        
        # Instance hourly costs (approximate)
        instance_costs = {
            "ml.p3.2xlarge": 3.06,
            "ml.p3.8xlarge": 12.24,
            "ml.p3.16xlarge": 24.48,
            "ml.p4d.24xlarge": 32.77
        }
        
        # Extract instance type without GPU count
        selected_instance = instance_type.split(" ")[0]
        hourly_cost = instance_costs.get(selected_instance, 3.06)
        
        # Calculate costs
        single_gpu_cost = (scaling_data['base_time_mins'] / 60) * hourly_cost
        distributed_cost = (scaling_data['distributed_time_mins'] / 60) * hourly_cost * num_instances
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Single GPU Total Cost", 
                f"${single_gpu_cost:.2f}"
            )
        
        with col2:
            cost_diff = single_gpu_cost - distributed_cost
            st.metric(
                f"{scaling_data['total_gpus']} GPU Total Cost", 
                f"${distributed_cost:.2f}",
                f"{cost_diff:.2f}" if cost_diff > 0 else f"-${-cost_diff:.2f}"
            )
        
        # Cost-effectiveness chart
        st.markdown("### Cost-Effectiveness Analysis")
        
        # Calculate cost and time for different GPU counts
        gpu_range = list(range(1, min(16, scaling_data['total_gpus'] + 5)))
        costs = []
        times_hours = []
        
        
        instance_gpu_map = {
            "ml.p3.2xlarge (1 GPU)": 1, 
            "ml.p3.8xlarge (4 GPUs)": 4, 
            "ml.p3.16xlarge (8 GPUs)": 8, 
            "ml.p4d.24xlarge (8 GPUs)": 8
        }

        # Then use it like this:
        gpus_per_instance = instance_gpu_map[instance_type]
        
        for gpu_count in gpu_range:
            # Divide GPUs across instances
            if gpu_count <= gpus_per_instance:
                num_inst = 1
            else:
                num_inst = (gpu_count + gpus_per_instance - 1) // gpus_per_instance  # Ceiling division
                
            # Get index in the efficiency data
            idx = min(gpu_count - 1, len(scaling_data['efficiencies']) - 1)
            
            # Calculate time
            time_mins = scaling_data['times'][idx]
            time_hours = time_mins / 60
            times_hours.append(time_hours)
            
            # Calculate cost
            cost = time_hours * hourly_cost * num_inst
            costs.append(cost)
        
        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Training time
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Training Time (hours)', color=AWS_COLORS['blue'])
        ax1.plot(gpu_range, times_hours, color=AWS_COLORS['blue'], marker='o', label='Training Time')
        ax1.tick_params(axis='y', labelcolor=AWS_COLORS['blue'])
        
        # Create a second y-axis for cost
        ax2 = ax1.twinx()
        ax2.set_ylabel('Total Cost ($)', color=AWS_COLORS['green'])
        ax2.plot(gpu_range, costs, color=AWS_COLORS['green'], marker='s', label='Total Cost')
        ax2.tick_params(axis='y', labelcolor=AWS_COLORS['green'])
        
        # Add a title
        fig.tight_layout()
        plt.title('Training Time vs. Cost by GPU Count')
        
        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        st.pyplot(fig)
    
    # Visualize different distributed strategies
    st.markdown("### Distributed Training Strategies Comparison")
    
    # Create visualization for understanding the different strategies
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Data Parallel visualization
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw GPUs
    gpu_positions = [(2, 8), (2, 5), (2, 2), (8, 8), (8, 5), (8, 2)]
    for i, (x, y) in enumerate(gpu_positions[:4]):  # Use 4 GPUs
        rect = plt.Rectangle((x-1.5, y-1), 3, 2, facecolor=AWS_COLORS["blue"], 
                            edgecolor='white', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, f"GPU {i+1}", ha='center', va='center', color='white', fontweight='bold')
        
        # Show full model on each GPU
        for j in range(3):
            layer = plt.Rectangle((x-1, y-0.5+0.3*j), 2, 0.2, 
                                 facecolor='white', alpha=0.7)
            ax.add_patch(layer)
    
    # Show data split
    ax.text(1, 9.5, "Data Chunk 1", fontsize=8)
    ax.text(1, 9, "Data Chunk 2", fontsize=8)
    ax.text(7, 9.5, "Data Chunk 3", fontsize=8)
    ax.text(7, 9, "Data Chunk 4", fontsize=8)
    
    # Gradient synchronization
    for i, (x, y) in enumerate(gpu_positions[:4]):
        for j, (x2, y2) in enumerate(gpu_positions[:4]):
            if i != j:
                ax.arrow(x, y-0.8, (x2-x)*0.3, (y2-y+0.8)*0.3, 
                        head_width=0.2, head_length=0.3, fc=AWS_COLORS["orange"], ec=AWS_COLORS["orange"],
                        alpha=0.4)
    
    ax.text(5, 0.5, "Synchronized Gradients", fontsize=9, ha='center')
    ax.set_title("Data Parallelism")
    
    # Model Parallel visualization
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw GPUs with different model parts
    for i, (x, y) in enumerate(gpu_positions[:4]):
        rect = plt.Rectangle((x-1.5, y-1), 3, 2, facecolor=AWS_COLORS["green"], 
                            edgecolor='white', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, f"GPU {i+1}", ha='center', va='center', color='white', fontweight='bold')
        
        # Show different model parts on each GPU
        for j in range(1):
            layer = plt.Rectangle((x-1, y-0.5+0.3*j), 2, 0.2, 
                                 facecolor='white', alpha=0.7)
            ax.add_patch(layer)
            ax.text(x, y-0.5+0.3*j, f"Layer {i+1}", ha='center', va='center', fontsize=8)
    
    # Show sequential processing
    ax.arrow(3.5, 8, 3, 0, head_width=0.3, head_length=0.3, 
            fc=AWS_COLORS["orange"], ec=AWS_COLORS["orange"])
    ax.arrow(7, 7, -3, -2, head_width=0.3, head_length=0.3, 
            fc=AWS_COLORS["orange"], ec=AWS_COLORS["orange"])
    ax.arrow(3, 4, 3, -2, head_width=0.3, head_length=0.3, 
            fc=AWS_COLORS["orange"], ec=AWS_COLORS["orange"])
    
    ax.text(5, 0.5, "Sequential Processing", fontsize=9, ha='center')
    ax.set_title("Model Parallelism")
    
    # Hybrid Parallel visualization
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw GPUs with hybrid approach
    colors = [AWS_COLORS["blue"], AWS_COLORS["blue"], AWS_COLORS["green"], AWS_COLORS["green"]]
    
    for i, ((x, y), color) in enumerate(zip(gpu_positions[:4], colors)):
        rect = plt.Rectangle((x-1.5, y-1), 3, 2, facecolor=color, 
                            edgecolor='white', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, f"GPU {i+1}", ha='center', va='center', color='white', fontweight='bold')
        
        # Show model parts
        gpu_group = i // 2
        ax.text(x, y-0.5, f"Layer {gpu_group+1}", ha='center', va='center', fontsize=8)
    
    # Data split annotation
    ax.text(1, 9.5, "Data Chunk 1", fontsize=8)
    ax.text(1, 9, "", fontsize=8)
    ax.text(7, 9.5, "Data Chunk 2", fontsize=8)
    ax.text(7, 9, "", fontsize=8)
    
    # Communication arrows
    ax.arrow(3.5, 8, 3, 0, head_width=0.3, head_length=0.3, 
            fc=AWS_COLORS["orange"], ec=AWS_COLORS["orange"], alpha=0.6)
    ax.arrow(3.5, 5, 3, 0, head_width=0.3, head_length=0.3, 
            fc=AWS_COLORS["orange"], ec=AWS_COLORS["orange"], alpha=0.6)
    
    ax.text(5, 0.5, "Hybrid Approach", fontsize=9, ha='center')
    ax.set_title("Hybrid Parallelism")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Code examples
    st.markdown("### Implementing Distributed Training")
    
    # Data Parallel example
    st.markdown("**Data Parallel Training Example**")
    
    st.code('''
from sagemaker.pytorch import PyTorch

# Configure data parallel distribution
distribution = {
    "torch_distributed": {
        "enabled": True
    }
}

# Set up the PyTorch estimator with distribution configuration
pytorch_estimator = PyTorch(
    entry_point='train_script.py',
    role=role,
    instance_type='ml.p3.8xlarge',  # 4 GPUs per instance
    instance_count=2,               # 2 instances = 8 total GPUs
    framework_version='1.12.0',
    py_version='py38',
    distribution=distribution,
    hyperparameters={
        'epochs': 20,
        'batch-size': 256,
        'learning-rate': 0.001
    }
)

# Start distributed training job
pytorch_estimator.fit({'train': train_data, 'val': val_data})
    ''')
    
    # SageMaker distributed data parallel
    st.markdown("**SageMaker Distributed Data Parallel Example**")
    
    st.code('''
from sagemaker.pytorch import PyTorch

# Configure SageMaker Data Parallel
distribution = {
    "smdistributed": {
        "dataparallel": {
            "enabled": True
        }
    }
}

# Set up the PyTorch estimator with SageMaker distribution
pytorch_estimator = PyTorch(
    entry_point='train_script.py',
    role=role,
    instance_type='ml.p3.16xlarge',  # 8 GPUs per instance
    instance_count=4,                # 4 instances = 32 total GPUs
    framework_version='1.12.0',
    py_version='py38',
    distribution=distribution,
    hyperparameters={
        'epochs': 20,
        'batch-size': 64,
        'learning-rate': 0.001
    }
)

# Start distributed training job
pytorch_estimator.fit({'train': train_data, 'val': val_data})
    ''')
    
    # Training script modifications
    st.markdown("**Training Script Modifications for Data Parallel**")
    
    st.code('''
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# SageMaker data parallel imports
import smdistributed.dataparallel.torch.distributed as smdp

def main():
    # Initialize the SageMaker distributed data parallel process group
    smdp.init_process_group()
    
    # Get local rank and world size
    local_rank = smdp.get_local_rank()
    world_size = smdp.get_world_size()
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Load model
    model = YourModel().to(device)
    
    # Wrap model with DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Set up data loader with distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler
    )
    
    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        train_sampler.set_epoch(epoch)
        
        # Training code...
        # ...
        
        # Synchronize all processes
        dist.barrier()
    ''')
    
    # Model Parallel example
    st.markdown("**SageMaker Model Parallel Example**")
    
    st.code('''
from sagemaker.pytorch import PyTorch

# Configure SageMaker Model Parallel
mpi_options = {
    "enabled": True,
    "processes_per_host": 8  # Using all 8 GPUs on an ml.p3.16xlarge
}

smp_options = {
    "enabled": True,
    "parameters": {
        "partitions": 4,             # Number of model partitions
        "microbatches": 8,           # Number of microbatches
        "placement_strategy": "spread",
        "pipeline": "interleaved",   # Pipeline schedule
        "optimize": "speed",         # Optimize for training speed
        "ddp": True                  # Also use data parallelism
    }
}

distribution = {
    "smdistributed": {
        "modelparallel": smp_options
    },
    "mpi": mpi_options
}

# Set up the PyTorch estimator
pytorch_estimator = PyTorch(
    entry_point='train_script.py',
    role=role,
    instance_type='ml.p3.16xlarge',
    instance_count=2,
    framework_version='1.12.0',
    py_version='py38',
    distribution=distribution,
    hyperparameters={
        'epochs': 20,
        'batch-size': 64,
        'learning-rate': 0.001
    }
)

# Start distributed training job
pytorch_estimator.fit({'train': train_data, 'val': val_data})
    ''')

# TAB 5: SPOT TRAINING

# Function to calculate costs and generate spot training simulation
def calculate_spot_costs_and_simulation(
    instance_type, instance_count, training_hours, 
    use_spot, max_wait, checkpoint_interval, interruption_probability):
    
    # On-demand prices (per hour)
    on_demand_prices = {
        "ml.c5.xlarge": 0.19,
        "ml.c5.2xlarge": 0.38,
        "ml.m5.xlarge": 0.23,
        "ml.m5.2xlarge": 0.46,
        "ml.p3.2xlarge": 3.06,
        "ml.p3.8xlarge": 12.24
    }
    
    # Spot prices (average discount of 70% off on-demand)
    spot_discount = 0.7
    spot_prices = {k: v * (1-spot_discount) for k, v in on_demand_prices.items()}
    
    # Calculate base costs
    on_demand_price = on_demand_prices.get(instance_type, 1.0)
    spot_price = spot_prices.get(instance_type, 0.3)
    
    on_demand_cost = on_demand_price * instance_count * training_hours
    potential_spot_cost = spot_price * instance_count * training_hours
    
    # Simulate spot interruptions
    np.random.seed(42)
    
    # Convert times to minutes for simulation
    training_minutes = training_hours * 60
    max_wait_minutes = max_wait * 60
    
    # Simulate training
    if not use_spot:
        # No interruptions with on-demand
        actual_runtime = training_minutes
        completed = True
        interruptions = 0
        time_lost = 0
        checkpoints = []
        
        # Generate checkpoints every checkpoint_interval minutes
        for t in range(0, int(actual_runtime), checkpoint_interval):
            checkpoints.append({"time": t, "progress": t / training_minutes})
            
        events = []
        
    else:
        # Simulate with potential interruptions
        current_time = 0
        remaining_training = training_minutes
        interruptions = 0
        time_lost = 0
        checkpoints = []
        events = []
        
        while current_time < max_wait_minutes and remaining_training > 0:
            # Add starting event
            if current_time == 0:
                events.append({
                    "time": current_time,
                    "event": "start",
                    "detail": "Training started"
                })
            
            # How long until next event (checkpoint or interruption)
            next_checkpoint = checkpoint_interval
            
            # Probability of interruption in this interval
            p_interrupt = interruption_probability / 100
            
            # Check if interruption happens before next checkpoint
            will_interrupt = np.random.random() < p_interrupt
            
            if will_interrupt:
                # Interruption happens at random time before next checkpoint
                time_to_interrupt = np.random.randint(1, next_checkpoint)
                
                # Training proceeds until interruption
                work_done = time_to_interrupt
                current_time += time_to_interrupt
                remaining_training -= work_done
                
                # Record interruption event
                events.append({
                    "time": current_time,
                    "event": "interruption",
                    "detail": "Spot instance reclaimed"
                })
                
                interruptions += 1
                
                # Time to recover (5-15 minutes)
                recovery_time = np.random.randint(5, 16)
                time_lost += recovery_time
                current_time += recovery_time
                
                # Record recovery event
                events.append({
                    "time": current_time,
                    "event": "recovery",
                    "detail": f"Recovered from last checkpoint (lost {work_done} min of progress)"
                })
                
                # Resume from last checkpoint - add time since last checkpoint to remaining
                if checkpoints:
                    last_checkpoint_time = checkpoints[-1]["time"]
                    time_since_checkpoint = work_done - (current_time - last_checkpoint_time - recovery_time)
                    if time_since_checkpoint > 0:
                        remaining_training += time_since_checkpoint
                        time_lost += time_since_checkpoint
                
            else:
                # No interruption, proceed to checkpoint
                work_done = min(next_checkpoint, remaining_training)
                current_time += work_done
                remaining_training -= work_done
                
                # Create checkpoint
                progress = (training_minutes - remaining_training) / training_minutes
                checkpoints.append({
                    "time": current_time,
                    "progress": progress
                })
                
                # Record checkpoint event
                if remaining_training > 0:  # Don't log checkpoint at the very end
                    events.append({
                        "time": current_time,
                        "event": "checkpoint",
                        "detail": f"Created checkpoint ({progress*100:.0f}% complete)"
                    })
        
        # Check if training completed within max wait time
        completed = remaining_training <= 0
        
        if completed:
            events.append({
                "time": current_time,
                "event": "complete",
                "detail": "Training completed successfully"
            })
            actual_runtime = current_time
        else:
            events.append({
                "time": current_time,
                "event": "timeout",
                "detail": f"Exceeded maximum wait time of {max_wait} hours"
            })
            actual_runtime = current_time
        
        # Calculate actual cost
        actual_spot_time = actual_runtime / 60  # Convert back to hours
        potential_spot_cost = spot_price * instance_count * actual_spot_time
    
    # Final cost comparison
    if use_spot and completed:
        actual_cost = potential_spot_cost
        savings_amount = on_demand_cost - actual_cost
        savings_percentage = (savings_amount / on_demand_cost) * 100
    elif use_spot and not completed:
        # If not completed, fall back to on-demand cost
        actual_cost = on_demand_cost
        savings_amount = 0
        savings_percentage = 0
    else:
        # On-demand training
        actual_cost = on_demand_cost
        savings_amount = 0
        savings_percentage = 0
    
    return {
        "instance_type": instance_type,
        "instance_count": instance_count,
        "training_hours": training_hours,
        "use_spot": use_spot,
        "max_wait_hours": max_wait,
        "checkpoint_interval": checkpoint_interval,
        "on_demand_price": on_demand_price,
        "spot_price": spot_price,
        "on_demand_cost": on_demand_cost,
        "spot_cost": potential_spot_cost,
        "actual_cost": actual_cost,
        "savings_amount": savings_amount,
        "savings_percentage": savings_percentage,
        "completed": completed,
        "interruptions": interruptions,
        "time_lost_minutes": time_lost,
        "actual_runtime_minutes": actual_runtime,
        "checkpoints": checkpoints,
        "events": events
    }

with tab5:
    st.header("Amazon SageMaker Spot Training")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        SageMaker Spot Training allows you to run training jobs on Amazon EC2 Spot instances
        to optimize cost, potentially reducing training costs by up to 90%.
        
        **Key benefits:**
        - Significantly lower training costs
        - Automatic checkpointing to resume interrupted jobs
        - Managed instance provisioning and termination
        - Seamless integration with other SageMaker features
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/re19/Diagrams/product-page-diagram_Amazon-SageMaker-Spot-Training_How-it-Works.c7f15ebfcfbe1331b29a7ff3f17a73611ca3483c.png",
                 caption="SageMaker Spot Training Process")
    
    # Interactive demo for cost savings with spot training
    st.subheader("Interactive Demo: Spot Training Cost Savings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enable/disable spot training
        spot_enabled = st.toggle("Enable Spot Training", value=st.session_state.spot_training_enabled)
        st.session_state.spot_training_enabled = spot_enabled
        
        # Instance selection
        instance_type = st.selectbox(
            "Instance Type",
            ["ml.c5.xlarge", "ml.c5.2xlarge", "ml.m5.xlarge", 
             "ml.m5.2xlarge", "ml.p3.2xlarge", "ml.p3.8xlarge"]
        )
        
        # Training job configuration
        instance_count = st.slider("Number of Instances", 1, 10, 2, key='instance_count')
        training_hours = st.slider("Expected Training Time (hours)", 1, 24, 8, key='training_hours')
    
    with col2:
        # Maximum wait time (for spot)
        max_wait = st.slider("Maximum Wait Time (hours)", 1, 12, 
                           st.session_state.max_wait_time)
        st.session_state.max_wait_time = max_wait
        
        # Checkpointing interval
        checkpoint_interval = st.slider("Checkpointing Interval (minutes)", 5, 60, 15)
        
        # Maximum runtime percentage before interruption (for simulation)
        if spot_enabled:
            interruption_probability = st.slider(
                "Spot Interruption Probability (%)", 
                0, 100, 30
            )
        
        # Calculate spot savings button
        if st.button("Calculate Cost and Savings"):
            with st.spinner("Calculating potential savings..."):
                time.sleep(1)  # Simulate calculation
                
                # Calculate costs and generate spot training simulation
                spot_results = calculate_spot_costs_and_simulation(
                    instance_type, 
                    instance_count,
                    training_hours,
                    spot_enabled,
                    max_wait,
                    checkpoint_interval,
                    interruption_probability if spot_enabled else 0
                )
                
                st.session_state.spot_results = spot_results
                
            st.success("Calculation complete!")
    

    
    # Display spot training results
    if 'spot_results' in st.session_state:
        results = st.session_state.spot_results
        
        st.markdown("### Cost Analysis")
        
        # Cost comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("On-Demand Cost", f"${results['on_demand_cost']:.2f}")
        
        with col2:
            cost_label = "Spot Training Cost" if results['use_spot'] else "Training Cost"
            st.metric(cost_label, f"${results['actual_cost']:.2f}")
        
        with col3:
            if results['savings_amount'] > 0:
                st.metric("Cost Savings", 
                         f"${results['savings_amount']:.2f}", 
                         f"{results['savings_percentage']:.1f}%")
            else:
                st.metric("Cost Savings", "$0.00")
        
        # Training simulation visualization
        st.markdown("### Training Simulation")
        
        if results['use_spot']:
            status_color = "green" if results['completed'] else "red"
            status_text = "Completed Successfully" if results['completed'] else "Exceeded Maximum Wait Time"
            
            # Status indicator
            st.markdown(f"""
            <div style="background-color:{AWS_COLORS[status_color]}; padding:10px; border-radius:5px; margin-bottom:10px;">
                <span style="color:white; font-weight:bold;">Status: {status_text}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Display summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Interruptions", f"{results['interruptions']}")
                st.metric("Actual Runtime", f"{results['actual_runtime_minutes'] / 60:.2f} hours")
                
            with col2:
                st.metric("Time Lost to Interruptions", f"{results['time_lost_minutes']} minutes")
                st.metric("Checkpoints Created", f"{len(results['checkpoints'])}")
            
            # Visualize the training timeline
            st.markdown("#### Training Timeline")
            
            # Convert events to DataFrame for visualization
            events_df = pd.DataFrame(results['events'])
            
            # Create timeline chart
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Plot timeline
            event_types = {
                "start": {"color": AWS_COLORS["green"], "marker": "o", "size": 80},
                "interruption": {"color": AWS_COLORS["red"], "marker": "X", "size": 100},
                "recovery": {"color": AWS_COLORS["orange"], "marker": "s", "size": 80},
                "checkpoint": {"color": AWS_COLORS["teal"], "marker": "D", "size": 60},
                "complete": {"color": AWS_COLORS["green"], "marker": "*", "size": 200},
                "timeout": {"color": AWS_COLORS["red"], "marker": "*", "size": 200}
            }
            
            # Plot each event type
            for event_type, style in event_types.items():
                mask = events_df['event'] == event_type
                if mask.any():
                    ax.scatter(events_df[mask]['time'], [1] * mask.sum(), 
                              color=style["color"], s=style["size"], 
                              marker=style["marker"], label=event_type.capitalize())
            
            # Add progress line
            if results['checkpoints']:
                checkpoint_times = [cp['time'] for cp in results['checkpoints']]
                checkpoint_progress = [cp['progress'] for cp in results['checkpoints']]
                
                # Add a start point at (0,0)
                checkpoint_times.insert(0, 0)
                checkpoint_progress.insert(0, 0)
                
                # Plot progress line on a second y-axis
                ax2 = ax.twinx()
                ax2.plot(checkpoint_times, checkpoint_progress, color=AWS_COLORS["blue"], 
                        linestyle='-', marker='o', markersize=4)
                ax2.set_ylabel('Training Progress')
                ax2.set_ylim(0, 1.05)
                ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            
            # Set up the main axis
            ax.set_yticks([])
            ax.set_xlabel('Time (minutes)')
            ax.set_title('Spot Training Timeline with Interruptions')
            
            # Add a legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            
            # Set x-axis limits
            ax.set_xlim(-5, results['actual_runtime_minutes'] + 5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Event log
            st.markdown("#### Event Log")
            
            # Format events for display
            log_data = []
            for event in results['events']:
                log_data.append({
                    "Time (min)": f"{event['time']}",
                    "Event": event['event'].capitalize(),
                    "Details": event['detail']
                })
            
            log_df = pd.DataFrame(log_data)
            st.dataframe(log_df, hide_index=True)
            
        else:
            # Simple visualization for on-demand training
            st.info("On-demand training simulation (no interruptions)")
            
            # Create a simple timeline
            fig, ax = plt.subplots(figsize=(12, 3))
            
            # Plot start and end points
            ax.scatter(0, 1, color=AWS_COLORS["green"], s=80, marker="o", label="Start")
            ax.scatter(results['training_hours'] * 60, 1, color=AWS_COLORS["green"], 
                      s=200, marker="*", label="Complete")
            
            # Plot checkpoints
            checkpoint_times = [cp['time'] for cp in results['checkpoints']]
            for cp_time in checkpoint_times:
                ax.scatter(cp_time, 1, color=AWS_COLORS["teal"], s=60, marker="D")
            
            # Plot progress line
            progress = [cp['progress'] for cp in results['checkpoints']]
            progress.insert(0, 0)  # Add starting point
            checkpoint_times.insert(0, 0)
            
            # Add progress on second y-axis
            ax2 = ax.twinx()
            ax2.plot(checkpoint_times, progress, color=AWS_COLORS["blue"], 
                    linestyle='-', marker='o', markersize=4)
            ax2.set_ylabel('Training Progress')
            ax2.set_ylim(0, 1.05)
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            
            # Set up the main axis
            ax.set_yticks([])
            ax.set_xlabel('Time (minutes)')
            ax.set_title('On-Demand Training Timeline')
            
            # Add legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            
            # Set x-axis limits
            ax.set_xlim(-5, results['training_hours'] * 60 + 5)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Spot instance savings chart
    st.markdown("### Spot Instance Savings by Instance Type")
    
    # Create comparison chart of on-demand vs spot prices
    instance_types = ["ml.c5.xlarge", "ml.m5.xlarge", "ml.p3.2xlarge", "ml.g4dn.xlarge"]
    on_demand_prices = [0.19, 0.23, 3.06, 0.736]
    spot_prices = [0.057, 0.069, 0.918, 0.221]  # approximate 70% discount
    
    price_df = pd.DataFrame({
        'Instance Type': instance_types,
        'On-Demand Price': on_demand_prices,
        'Spot Price': spot_prices
    })
    
    # Calculate savings
    price_df['Savings (%)'] = (
        (price_df['On-Demand Price'] - price_df['Spot Price']) / 
        price_df['On-Demand Price'] * 100
    ).round(1)
    
    # Create bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(instance_types))
    width = 0.35
    
    # Plot bars
    on_demand_bars = ax.bar(x - width/2, on_demand_prices, width, 
                           label='On-Demand Price', color=AWS_COLORS['blue'])
    spot_bars = ax.bar(x + width/2, spot_prices, width, 
                      label='Spot Price', color=AWS_COLORS['orange'])
    
    # Customize chart
    ax.set_xlabel('Instance Type')
    ax.set_ylabel('Price per Hour ($)')
    ax.set_title('On-Demand vs. Spot Pricing by Instance Type')
    ax.set_xticks(x)
    ax.set_xticklabels(instance_types)
    ax.legend()
    
    # Add savings percentage labels
    for i, instance in enumerate(instance_types):
        savings = price_df[price_df['Instance Type'] == instance]['Savings (%)'].values[0]
        ax.text(i, spot_prices[i] + 0.05, f"{savings}% savings", 
               ha='center', va='bottom', color=AWS_COLORS['green'], fontweight='bold')
    
    # Adjust log scale for better visualization of price differences
    ax.set_yscale('log')
    
    st.pyplot(fig)
    
    # Code examples
    st.markdown("### Implementing Spot Training")
    st.markdown("Here's how to set up spot training with SageMaker:")
    
    st.code('''
from sagemaker.pytorch import PyTorch

# Configure spot training
estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_count=2,
    instance_type='ml.p3.2xlarge',
    framework_version='1.12.0',
    py_version='py38',
    
    # Enable spot training
    use_spot_instances=True,
    
    # Maximum time to wait for spot instances
    max_wait=3600,  # seconds
    
    # Maximum runtime once instances are available
    max_run=36000,  # seconds
    
    # Checkpoint config
    checkpoint_s3_uri='s3://mybucket/checkpoints',
    checkpoint_local_path='/opt/ml/checkpoints'
)

# Start training job
estimator.fit('s3://mybucket/training-data')
    ''')
    
    # Checkpoint code example
    st.markdown("### Implementing Checkpoints in Your Training Script")
    
    st.code('''
import os
import torch
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Add checkpoint arguments
    parser.add_argument('--checkpoint-path', type=str, default='/opt/ml/checkpoints')
    
    args, _ = parser.parse_known_args()
    
    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # Load model and optimizer
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training setup
    start_epoch = 0
    best_accuracy = 0
    
    # Check if checkpoints exist
    checkpoint_files = os.listdir(args.checkpoint_path)
    if checkpoint_files:
        # Find latest checkpoint
        latest_checkpoint = max([os.path.join(args.checkpoint_path, f) for f in checkpoint_files],
                                key=os.path.getctime)
        
        print(f"Loading checkpoint from {latest_checkpoint}")
        
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Training code...
        
        # Save checkpoint periodically
        if epoch % args.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(args.checkpoint_path, f'checkpoint-{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }, checkpoint_path)
            
            print(f"Checkpoint saved at {checkpoint_path}")
    ''')
    
    # Best practices
    st.markdown("### Best Practices for Spot Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Checkpointing Strategy
        
        - **Checkpoint frequently** enough to minimize lost work
        - **Store checkpoints in S3** for durability
        - **Optimize checkpoint size** to reduce storage and loading time
        - **Implement incremental checkpoints** for large models
        - **Test checkpoint restoration** before long training runs
        """)
    
    with col2:
        st.markdown("""
        #### Maximizing Spot Availability
        
        - **Be flexible with instance types** when possible
        - **Set appropriate max wait time** based on your workload
        - **Consider multiple Availability Zones**
        - **Schedule training during off-peak hours** when possible
        - **Monitor spot interruption history** in CloudWatch
        """)
    
    # Spot vs. On-demand decision guide
    st.markdown("### When to Use Spot vs. On-Demand Instances")
    
    decision_data = {
        "Criteria": [
            "Training time sensitivity",
            "Job interruption tolerance",
            "Checkpoint implementation",
            "Cost sensitivity",
            "Training job length",
            "Infrastructure availability requirements"
        ],
        "Spot Instances": [
            "Not time-critical",
            "Can tolerate interruptions",
            "Has checkpointing implemented",
            "High cost sensitivity",
            "Long-running jobs (to maximize savings)",
            "Flexible on availability"
        ],
        "On-Demand Instances": [
            "Time-critical deadlines",
            "Cannot afford interruptions",
            "No checkpointing capability",
            "Lower cost sensitivity",
            "Short critical jobs",
            "Requires guaranteed availability"
        ]
    }
    
    decision_df = pd.DataFrame(decision_data)
    st.table(decision_df)

# Add footer
st.markdown("""
<div class="footer">
© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</div>
""", unsafe_allow_html=True)
