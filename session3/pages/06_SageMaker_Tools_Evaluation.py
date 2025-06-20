
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import json
import time
import uuid
from PIL import Image
import io
import base64
from streamlit_lottie import st_lottie
import requests

# Set page configuration for wider layout
st.set_page_config(
    page_title="SageMaker Tools Explorer",
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
if 'init_eval' not in st.session_state:
    st.session_state.init_eval = True
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.clarify_model_choice = "XGBoost"
    st.session_state.debugger_stage = 0
    st.session_state.experiment_model = "ResNet"
    st.session_state.shadow_test_model = "Model A"
    st.session_state.shadow_test_traffic = 50

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
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
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
            This interactive learning application demonstrates the powerful evaluation 
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
st.title("Amazon SageMaker Tools for Evaluation")
st.markdown("Explore the powerful tools that SageMaker offers for evaluating, debugging, and improving your machine learning models.")

# # Animation for the header
# lottie_url = "https://assets9.lottiefiles.com/packages/lf20_kuhijlvx.json"
# lottie_json = load_lottieurl(lottie_url)
# if lottie_json:
#     st_lottie(lottie_json, height=200, key="header_animation")

# Tab-based navigation with emoji
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 SageMaker Clarify", 
    "🐞 SageMaker Debugger", 
    "🧪 SageMaker Experiments",
    "🔄 Shadow Testing"
])

# TAB 1: SAGEMAKER CLARIFY
with tab1:
    st.header("Amazon SageMaker Clarify")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        SageMaker Clarify helps you understand how machine learning models make predictions by 
        identifying feature importance and potential bias in your data and models.
        
        **Key benefits:**
        - Detect bias in your data and models
        - Explain prediction results
        - Generate model-agnostic feature importance values
        - Integrate bias and explainability reports into your ML workflow
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/re19/Diagrams/product-page-diagram_Amazon-SageMaker-Clarify_How-it-Works.6ca5d9385209c9f805fc40a5cbed00ee77a9195f.png", 
                 caption="SageMaker Clarify Workflow")

    st.subheader("Interactive Demo: Explore Model Bias and Explainability")
    
    # Model selection
    model_options = ["XGBoost", "Random Forest", "Neural Network"]
    st.session_state.clarify_model_choice = st.selectbox(
        "Select a model to analyze:", 
        model_options, 
        index=model_options.index(st.session_state.clarify_model_choice)
    )
    
    # Create sample DataFrame for bias analysis
    @st.cache_data
    def generate_sample_data():
        np.random.seed(42)
        n = 1000
        data = {
            'age': np.random.normal(40, 15, n).astype(int),
            'income': np.random.exponential(50000, n).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n)
        }
        
        # Create biased approval rates based on gender and income
        p_approve = (0.4 + 0.3 * (data['income'] > 50000) + 
                      0.2 * (np.array([g == 'Male' for g in data['gender']])))
        data['approved'] = np.random.binomial(1, p_approve)
        
        return pd.DataFrame(data)
    
    df = generate_sample_data()
    
    # Display dataset
    st.markdown("### Sample Dataset:")
    st.dataframe(df.head())
    
    # Bias analysis section
    st.markdown("### Bias Analysis")
    st.markdown("Let's analyze potential bias in our loan approval dataset with respect to gender:")
    
    # Generate bias metrics based on gender
    def calculate_bias_metrics(df):
        total_approvals = df['approved'].sum()
        total_count = len(df)
        
        bias_data = []
        for gender in ['Male', 'Female']:
            gender_df = df[df['gender'] == gender]
            count = len(gender_df)
            approvals = gender_df['approved'].sum()
            approval_rate = approvals / count if count > 0 else 0
            
            bias_data.append({
                'Gender': gender,
                'Count': count,
                'Approvals': approvals,
                'Approval Rate': f"{approval_rate:.2%}",
                'Proportion': count / total_count,
            })
            
        return pd.DataFrame(bias_data)
            
    bias_metrics = calculate_bias_metrics(df)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Calculate disparate impact
        female_rate = bias_metrics[bias_metrics['Gender'] == 'Female']['Approval Rate'].iloc[0]
        male_rate = bias_metrics[bias_metrics['Gender'] == 'Male']['Approval Rate'].iloc[0]
        
        # Convert percentages to floats for calculation
        female_rate_float = float(female_rate.strip('%')) / 100
        male_rate_float = float(male_rate.strip('%')) / 100
        
        disparity = female_rate_float / male_rate_float if male_rate_float > 0 else 0
        
        st.markdown("#### Bias Metrics")
        st.dataframe(bias_metrics)
        
        st.markdown("#### Disparate Impact Analysis")
        st.markdown(f"Disparate Impact Ratio (Female/Male approval): **{disparity:.2f}**")
        
        if disparity < 0.8:
            st.warning("⚠️ Potential bias detected: Female approval rate is significantly lower than male approval rate")
        elif disparity > 1.2:
            st.warning("⚠️ Potential bias detected: Female approval rate is significantly higher than male approval rate")
        else:
            st.success("✅ No significant gender-based disparity detected in approval rates")

    with col2:
        # Create bar chart for approval rates
        chart_data = pd.DataFrame({
            'Gender': ['Male', 'Female'],
            'Approval Rate': [male_rate_float, female_rate_float]
        })
        
        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Gender', title=None),
            y=alt.Y('Approval Rate', title='Approval Rate', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Gender', scale=alt.Scale(domain=['Male', 'Female'], 
                                                    range=[AWS_COLORS['blue'], AWS_COLORS['teal']]))
        ).properties(
            title='Approval Rate by Gender'
        )
        
        st.altair_chart(bar_chart, use_container_width=True)
    
    # Feature Importance section
    st.markdown("### Feature Importance Analysis")
    st.markdown("Let's examine which features most influence the model's decisions:")
    
    # Generate different feature importances based on model choice
    if st.session_state.clarify_model_choice == "XGBoost":
        feature_imp = {
            'income': 0.45, 
            'age': 0.22, 
            'gender': 0.18, 
            'education': 0.10, 
            'region': 0.05
        }
    elif st.session_state.clarify_model_choice == "Random Forest":
        feature_imp = {
            'income': 0.38, 
            'age': 0.25, 
            'gender': 0.12, 
            'education': 0.15, 
            'region': 0.10
        }
    else:  # Neural Network
        feature_imp = {
            'income': 0.36, 
            'age': 0.28, 
            'gender': 0.14, 
            'education': 0.12, 
            'region': 0.10
        }
    
    # Create feature importance DataFrame
    feature_imp_df = pd.DataFrame({
        'Feature': list(feature_imp.keys()),
        'Importance': list(feature_imp.values())
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Plot feature importances
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], 
                color=AWS_COLORS['orange'])
        
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - {st.session_state.clarify_model_choice}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', va='center')
        
        st.pyplot(fig)
    
    with col2:
        
        model_descriptions = {
    "XGBoost": "The XGBoost model shows high sensitivity to income, with gender having substantial influence on decisions.",
    "Random Forest": "The Random Forest model relies most heavily on income and age, with gender having less impact compared to XGBoost.",
    "Neural Network": "The Neural Network shows more balanced feature usage, still prioritizing income but with more weight on age."
}
        description = model_descriptions.get(st.session_state.clarify_model_choice, "")
        
        st.markdown("#### Key Insights")
        st.markdown(f"""
        - **{feature_imp_df.iloc[0]['Feature'].capitalize()}** is the most important feature
        - **Gender** importance: {feature_imp['gender']:.2f}
        - The model is {'highly' if feature_imp['gender'] > 0.15 else 'moderately' if feature_imp['gender'] > 0.10 else 'minimally'} influenced by gender
        
        **What this means:**
            {description}
        """)
    
    # Code examples for SageMaker Clarify
    st.markdown("### Implementing SageMaker Clarify")
    st.markdown("Here's how you can use SageMaker Clarify in your ML workflow:")
    
    st.code('''
# Import necessary libraries
from sagemaker import clarify

# Define the clarify processor
clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=session
)

# Define configurations for analysis
bias_config = clarify.BiasConfig(
    label_values_or_threshold=[1],  # Target is "approved"
    facet_name="gender",            # Protected attribute
    facet_values_or_threshold=["Female"]  # Value to analyze for bias
)

# Define SHAP explainer configuration
shap_config = clarify.SHAPConfig(
    baseline=[df.drop("approved", axis=1).median().values.tolist()],
    num_samples=100,
    agg_method="mean_abs"
)

# Run the analysis job
clarify_processor.run_bias_analysis(
    data_config=clarify.DataConfig(
        s3_data_input_path=train_uri,
        s3_output_path=clarify_output_path,
        label="approved",
        headers=df.columns.tolist(),
        dataset_type="text/csv"
    ),
    bias_config=bias_config,
    model_config=clarify.ModelConfig(
        model_name=model_name,
        instance_type="ml.m5.xlarge",
        instance_count=1,
    ),
)

# Run explainability analysis
clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=shap_config
)
    ''')
    
    # SHAP values explanation
    st.markdown("### SHAP Values Visualizer")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) values help explain how each feature contributes
    to a specific prediction. Let's examine a sample loan application:
    """)
    
    # Sample loan application
    sample_application = {
        'age': 35,
        'income': 72000,
        'gender': 'Female',
        'education': 'Master',
        'region': 'East'
    }
    
    # Display application details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Loan Application")
        for k, v in sample_application.items():
            st.markdown(f"- **{k.capitalize()}**: {v}")
    
    with col2:
        # Generate SHAP values based on model
        base_value = 0.5
        if st.session_state.clarify_model_choice == "XGBoost":
            shap_values = {'income': 0.18, 'age': 0.05, 'gender': -0.03, 'education': 0.07, 'region': -0.01}
            prediction = 0.76
        elif st.session_state.clarify_model_choice == "Random Forest":
            shap_values = {'income': 0.15, 'age': 0.06, 'gender': -0.02, 'education': 0.08, 'region': 0.01}
            prediction = 0.78
        else:  # Neural Network
            shap_values = {'income': 0.16, 'age': 0.07, 'gender': -0.02, 'education': 0.06, 'region': 0.02}
            prediction = 0.79
        
        # Create SHAP waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(shap_values.keys())
        values = list(shap_values.values())
        
        # Cumulative values for waterfall
        cumulative = [base_value]
        for v in values:
            cumulative.append(cumulative[-1] + v)
        
        # Create bars
        # Base value bar
        ax.barh([len(features)], [base_value], color='#cccccc', alpha=0.8)
        ax.text(0, len(features), f"Base value: {base_value:.2f}", va='center')
        
        # Feature contribution bars
        for i, (feature, value) in enumerate(zip(features, values)):
            pos = len(features) - i - 1
            color = AWS_COLORS['teal'] if value > 0 else AWS_COLORS['red']
            ax.barh([pos], [value], left=cumulative[i], color=color)
            
            # Label
            feature_text = f"{feature}: {value:+.3f}"
            text_x = cumulative[i] + value/2
            ax.text(text_x, pos, feature_text, ha='center', va='center', 
                    color='white' if abs(value) > 0.05 else 'black')
        
        # Final prediction bar
        ax.barh([0], [prediction], color=AWS_COLORS['green'], alpha=0.8)
        ax.text(prediction/2, 0, f"Final prediction: {prediction:.2f}", ha='center', va='center')
        
        # Set y-ticks and limits
        ax.set_yticks(range(len(features) + 2))
        ax.set_yticklabels([])
        
        ax.set_title(f"SHAP Feature Contributions - {st.session_state.clarify_model_choice}")
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
    
    st.markdown("""
    ### Why Explainability Matters
    
    Model explainability is crucial for:
    
    1. **Regulatory compliance** - Especially in sensitive industries like banking and healthcare
    2. **Bias detection** - Identifying and addressing unfair discrimination
    3. **Model improvement** - Understanding what influences your model's predictions helps you refine it
    4. **Trust building** - Stakeholders need to understand how AI systems make decisions
    """)

# TAB 2: SAGEMAKER DEBUGGER
with tab2:
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
    def simulate_training_data(epochs, learning_rate, has_issues=False):
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
                # Simulate vanishing gradients
                grad_val = grad_val * 0.7
                # Simulate overfitting
                train_val = max(0.05, train_val * 0.9)
                val_val = min(3.0, val_val * 1.1)
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
        epochs = st.slider("Number of Epochs", min_value=10, max_value=100, value=50, step=5)
    
    with col2:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5],
            value=0.01
        )
    
    with col3:
        issue_type = st.selectbox("Simulate Issues", 
                           ["None", "Vanishing Gradients", "Overfitting"])
    
    has_issues = issue_type != "None"
    
    # Run training
    if st.button("Run Training"):
        with st.spinner("Training model and capturing debug information..."):
            training_data = simulate_training_data(epochs, learning_rate, has_issues)
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
        
        st.pyplot(fig)
        
        # Show gradients
        st.markdown("### Gradient Magnitudes")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(training_data['epochs'], training_data['gradients'], color=AWS_COLORS['green'])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title('Gradient Flow During Training')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add threshold line for vanishing gradients
        if issue_type == "Vanishing Gradients":
            ax.axhline(y=0.1, color=AWS_COLORS['red'], linestyle='--')
            ax.text(epochs//4, 0.11, "Healthy Gradient Threshold", color=AWS_COLORS['red'])
        
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
    
    # Share timeline of debugging a real issue
    st.markdown("### Case Study: Debugging a Real World Model")
    
    # Timeline visualization of debugging process
    timeline_data = {
        'day': [1, 1, 2, 2, 3, 3, 4, 5],
        'event': [
            'Initial model training showing poor convergence',
            'Set up SageMaker Debugger to capture tensors',
            'Analyzed loss curves showing oscillation',
            'Debugger detected exploding gradients',
            'Modified code to implement gradient clipping',
            'Restarted training with new configuration',
            'Debugger confirmed stable gradients',
            'Successfully completed training with improved accuracy'
        ],
        'type': [
            'issue', 'action', 'analysis', 'detection', 
            'action', 'action', 'confirmation', 'success'
        ]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    
    # Draw the timeline
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Color mapping
    color_map = {
        'issue': AWS_COLORS['red'],
        'action': AWS_COLORS['teal'],
        'analysis': AWS_COLORS['blue'],
        'detection': AWS_COLORS['orange'],
        'confirmation': AWS_COLORS['green'],
        'success': AWS_COLORS['green']
    }
    
    # Plot timeline
    y_positions = range(len(timeline_df))
    ax.scatter(timeline_df['day'], y_positions, c=[color_map[t] for t in timeline_df['type']], 
              s=100, zorder=2)
    
    # Connect points with lines
    for i in range(len(timeline_df) - 1):
        ax.plot([timeline_df['day'][i], timeline_df['day'][i+1]], 
                [y_positions[i], y_positions[i+1]], 
                'k-', alpha=0.3, zorder=1)
    
    # Add event labels
    for i, (day, event, event_type) in enumerate(zip(timeline_df['day'], timeline_df['event'], timeline_df['type'])):
        ax.text(day + 0.1, i, event, va='center', fontsize=10, 
               color='black', fontweight='bold' if event_type in ['issue', 'success'] else 'normal')
    
    # Set up the axes
    ax.set_yticks([])
    ax.set_xlabel('Day')
    ax.set_title('Timeline of Debugging a Production Model')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(0.5, 5.5)
    
    st.pyplot(fig)
    
    # Legend for the timeline
    st.markdown("##### Legend:")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(f"🔴 Issue")
        st.markdown(f"🔵 Analysis")
    with col2:
        st.markdown(f"🟢 Success")
        st.markdown(f"🟠 Detection")
    with col3:
        st.markdown(f"🔷 Action")
        st.markdown(f"🔷 Confirmation")

# TAB 3: SAGEMAKER EXPERIMENTS

# Function to generate experiment results
def generate_experiment_results(config):
    np.random.seed(42)
    
    # Model-specific base performance
    model_performance = {
        "ResNet": {"accuracy": 0.92, "training_time": 120},
        "VGG": {"accuracy": 0.90, "training_time": 180},
        "MobileNet": {"accuracy": 0.88, "training_time": 90},
        "EfficientNet": {"accuracy": 0.93, "training_time": 150}
    }
    
    # Get base performance for the selected model
    base = model_performance[config['model']]
    
    # Adjust based on hyperparameters
    # Learning rate impact
    lr_impact = 0
    if config['learning_rate'] <= 0.0001:
        lr_impact = -0.03  # Too small, slower convergence
    elif config['learning_rate'] >= 0.01:
        lr_impact = -0.05  # Too large, might overshoot
    
    # Optimizer impact
    optimizer_impact = {
        "Adam": 0.01,
        "SGD": -0.01, 
        "RMSprop": 0.005
    }
    
    # Batch size impact
    batch_impact = 0
    if config['batch_size'] <= 32:
        batch_impact = 0.01  # Better generalization but slower
    elif config['batch_size'] >= 128:
        batch_impact = -0.01  # Faster but less precise
        
    # Epochs impact (diminishing returns)
    epoch_factor = min(1.0, config['epochs'] / 30)
    
    # Calculate final metrics with some randomness
    accuracy = (base['accuracy'] + 
                lr_impact + 
                optimizer_impact[config['optimizer']] + 
                batch_impact) * epoch_factor
    
    # Add a small random factor
    accuracy += np.random.normal(0, 0.01)
    accuracy = min(0.99, max(0.7, accuracy))
    
    # Calculate training time
    training_time = (base['training_time'] * 
                    (config['epochs'] / 20) * 
                    (config['batch_size'] / 64))
    
    # Generate learning curves
    epochs_range = list(range(1, config['epochs'] + 1))
    
    # Start from a higher loss and decrease
    train_loss = [1.5 * np.exp(-epoch / (10 / config['learning_rate'])) + 0.2 + np.random.normal(0, 0.05) 
                    for epoch in epochs_range]
    
    val_loss = [loss + 0.1 + np.random.normal(0, 0.07) for loss in train_loss]
    
    # Accuracy increases over time
    max_acc = accuracy
    train_acc = [max_acc * (1 - np.exp(-epoch / (15 / config['learning_rate']))) + np.random.normal(0, 0.01) 
                for epoch in epochs_range]
    
    val_acc = [acc - 0.05 - np.random.normal(0, 0.02) for acc in train_acc]
    val_acc = [max(0.4, min(0.98, acc)) for acc in val_acc]
    
    # Resource utilization
    gpu_util = np.random.uniform(70, 95)
    memory_util = np.random.uniform(60, 90)
    
    return {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'accuracy': accuracy,
        'training_time': training_time,
        'gpu_utilization': gpu_util,
        'memory_utilization': memory_util,
        'epochs_data': {
            'epochs': epochs_range,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
    }


with tab3:
    st.header("Amazon SageMaker Experiments")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        SageMaker Experiments helps you organize, track, compare, and evaluate your 
        machine learning experiments, making it easy to find the best performing models.
        
        **Key benefits:**
        - Track all inputs and outputs of your experiments
        - Compare results across multiple training runs
        - Visualize experiment metrics and parameters
        - Reproduce successful experiments
        """)
    
    with col2:
        st.image("https://d1.awsstatic.com/re19/Diagrams/product-page-diagram_Amazon-SageMaker-Experiments_HIW.045b45afb8b65ceeed6eb0a05fdef8b10a8c2cb1.jpg",
                caption="SageMaker Experiments Workflow")
    
    st.subheader("Interactive Demo: Experiment Tracking and Comparison")
    
    # Model experiment interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configure Experiment")
        
        # Model selection
        model_options = ["ResNet", "VGG", "MobileNet", "EfficientNet"]
        model_selected = st.selectbox("Select model architecture:", model_options, 
                                     index=model_options.index(st.session_state.experiment_model))
        st.session_state.experiment_model = model_selected
        
        # Hyperparameters
        st.markdown("#### Hyperparameters")
        
        batch_size = st.slider("Batch Size", 16, 256, 64, step=16)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
        optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
        epochs = st.slider("Epochs", 5, 50, 20, step=5)
        
        # Run experiment button
        if st.button("Run Experiment"):
            with st.spinner(f"Running experiment with {model_selected}..."):
                time.sleep(2)  # Simulate experiment running
                
                # Store the experiment configuration
                experiment = {
                    'model': model_selected,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'optimizer': optimizer,
                    'epochs': epochs,
                }
                
                # Generate experiment results based on parameters (simulated)
                result = generate_experiment_results(experiment)
                
                # Store in session state
                if 'experiments' not in st.session_state:
                    st.session_state.experiments = []
                    
                experiment_id = f"exp-{len(st.session_state.experiments) + 1}"
                experiment['id'] = experiment_id
                experiment.update(result)
                
                st.session_state.experiments.append(experiment)
                
                st.success(f"Experiment {experiment_id} completed!")
    

    
    # Display experiment results
    with col2:
        if 'experiments' in st.session_state and st.session_state.experiments:
            st.markdown("### Experiment Results")
            
            # Get the latest experiment
            latest_exp = st.session_state.experiments[-1]
            
            # Display metrics in columns
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Accuracy", f"{latest_exp['accuracy']:.4f}")
                st.metric("Training Time", f"{latest_exp['training_time']:.1f}s")
                
            with metrics_col2:
                st.metric("GPU Utilization", f"{latest_exp['gpu_utilization']:.1f}%")
                st.metric("Memory Usage", f"{latest_exp['memory_utilization']:.1f}%")
            
            # Show learning curves
            st.markdown("#### Learning Curves")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            ax1.plot(latest_exp['epochs_data']['epochs'], latest_exp['epochs_data']['train_loss'], 
                    label='Training Loss', color=AWS_COLORS['orange'])
            ax1.plot(latest_exp['epochs_data']['epochs'], latest_exp['epochs_data']['val_loss'], 
                    label='Validation Loss', color=AWS_COLORS['teal'])
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss Curves')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Accuracy plot
            ax2.plot(latest_exp['epochs_data']['epochs'], latest_exp['epochs_data']['train_acc'], 
                    label='Training Accuracy', color=AWS_COLORS['orange'])
            ax2.plot(latest_exp['epochs_data']['epochs'], latest_exp['epochs_data']['val_acc'], 
                    label='Validation Accuracy', color=AWS_COLORS['teal'])
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy Curves')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Experiment comparison section
    if 'experiments' in st.session_state and len(st.session_state.experiments) > 1:
        st.subheader("Experiment Comparison")
        
        # Create dataframe from experiments
        exp_df = pd.DataFrame([
            {
                'Experiment ID': exp['id'],
                'Model': exp['model'],
                'Accuracy': f"{exp['accuracy']:.4f}",
                'Training Time (s)': f"{exp['training_time']:.1f}",
                'Batch Size': exp['batch_size'],
                'Learning Rate': exp['learning_rate'],
                'Optimizer': exp['optimizer']
            } for exp in st.session_state.experiments
        ])
        
        # Display experiment table
        st.dataframe(exp_df, use_container_width=True)
        
        # Prepare data for visualization
        comparison_data = {
            'experiment_id': [exp['id'] for exp in st.session_state.experiments],
            'model': [exp['model'] for exp in st.session_state.experiments],
            'accuracy': [exp['accuracy'] for exp in st.session_state.experiments],
            'training_time': [exp['training_time'] for exp in st.session_state.experiments]
        }
        
        # Create comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            
            bar_positions = np.arange(len(comparison_data['experiment_id']))
            bars = ax.bar(bar_positions, comparison_data['accuracy'], 
                         color=[AWS_COLORS['teal'] if m == 'ResNet' else 
                               (AWS_COLORS['orange'] if m == 'VGG' else 
                                (AWS_COLORS['blue'] if m == 'MobileNet' else AWS_COLORS['green'])) 
                                for m in comparison_data['model']])
            
            # Add model labels
            for bar, model in zip(bars, comparison_data['model']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       model, ha='center', va='bottom', rotation=0)
            
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(comparison_data['experiment_id'])
            ax.set_ylim(0.5, 1.0)
            
            st.pyplot(fig)
        
        with col2:
            # Training time comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            
            bar_positions = np.arange(len(comparison_data['experiment_id']))
            bars = ax.bar(bar_positions, comparison_data['training_time'], 
                         color=[AWS_COLORS['teal'] if m == 'ResNet' else 
                               (AWS_COLORS['orange'] if m == 'VGG' else 
                                (AWS_COLORS['blue'] if m == 'MobileNet' else AWS_COLORS['green'])) 
                                for m in comparison_data['model']])
            
            # Add model labels
            for bar, model in zip(bars, comparison_data['model']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       model, ha='center', va='bottom', rotation=0)
            
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title('Training Time Comparison')
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(comparison_data['experiment_id'])
            
            st.pyplot(fig)
        
        # Parallel coordinates plot for hyperparameter comparison
        st.markdown("### Hyperparameter Impact Analysis")
        
        # Create data for parallel coordinates
        parallel_data = pd.DataFrame({
            'Experiment': [exp['id'] for exp in st.session_state.experiments],
            'Learning Rate': [exp['learning_rate'] for exp in st.session_state.experiments],
            'Batch Size': [exp['batch_size'] for exp in st.session_state.experiments],
            'Epochs': [exp['epochs'] for exp in st.session_state.experiments],
            'Accuracy': [exp['accuracy'] for exp in st.session_state.experiments]
        })
        
        # Create parallel coordinates plot using Altair
        cols_to_use = ['Learning Rate', 'Batch Size', 'Epochs', 'Accuracy']
        
        # Normalize data for parallel coordinates
        for col in cols_to_use:
            min_val = parallel_data[col].min()
            max_val = parallel_data[col].max()
            parallel_data[f"{col}_normalized"] = (parallel_data[col] - min_val) / (max_val - min_val)
        
        # Create long format data
        parallel_long = pd.melt(
            parallel_data, 
            id_vars=['Experiment', 'Accuracy'], 
            value_vars=[f"{col}_normalized" for col in cols_to_use[:-1]],
            var_name='Parameter',
            value_name='Value'
        )
        
        # Fix parameter names (remove _normalized suffix)
        parallel_long['Parameter'] = parallel_long['Parameter'].str.replace('_normalized', '')
        
        # Create Altair parallel coordinates plot
        lines = alt.Chart(parallel_long).mark_line().encode(
            x='Parameter:N',
            y='Value:Q',
            color=alt.Color('Accuracy:Q', scale=alt.Scale(scheme='viridis')),
            detail='Experiment:N',
            tooltip=['Experiment', 'Accuracy']
        ).properties(
            width=800,
            height=400,
            title='Hyperparameter Parallel Coordinates Plot'
        )
        
        # Add points to make it clearer
        points = alt.Chart(parallel_long).mark_circle(size=100).encode(
            x='Parameter:N',
            y='Value:Q',
            color=alt.Color('Accuracy:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['Experiment', 'Accuracy', 'Parameter', alt.Tooltip('Value:Q', format='.4f')]
        )
        
        st.altair_chart(lines + points, use_container_width=True)
        
        # Find best performing experiment
        best_exp = max(st.session_state.experiments, key=lambda x: x['accuracy'])
        
        st.markdown(f"""
        #### Best Performing Configuration:
        
        **Experiment:** {best_exp['id']}  
        **Model:** {best_exp['model']}  
        **Accuracy:** {best_exp['accuracy']:.4f}  
        **Learning Rate:** {best_exp['learning_rate']}  
        **Batch Size:** {best_exp['batch_size']}  
        **Optimizer:** {best_exp['optimizer']}  
        """)
    
    # Code examples
    st.markdown("### Implementing SageMaker Experiments")
    st.markdown("Here's how to track experiments with SageMaker:")
    
    st.code('''
# Import required libraries
import time
import boto3
from sagemaker.session import Session
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker

# Set up experiment
experiment_name = f"image-classification-{int(time.time())}"
experiment = Experiment.create(
    experiment_name=experiment_name,
    description="Image classification model testing",
    tags=[{'Key': 'project', 'Value': 'image-classification'}]
)

# Set up trial (training job)
trial_name = f"resnet-training-{int(time.time())}"
trial = Trial.create(
    trial_name=trial_name,
    experiment_name=experiment_name,
    tags=[{'Key': 'model', 'Value': 'resnet50'}]
)

# Set up trial component (specific run of a training job)
with Tracker.create(display_name="training-job", sagemaker_session=session) as tracker:
    # Log hyperparameters
    tracker.log_parameters({
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
        "optimizer": "Adam"
    })
    
    # Train the model...
    
    # Log metrics during training
    for epoch in range(1, epochs + 1):
        # Run training and validation for this epoch...
        
        # Log metrics
        tracker.log_metric(name="train:loss", value=train_loss, step=epoch)
        tracker.log_metric(name="train:accuracy", value=train_accuracy, step=epoch)
        tracker.log_metric(name="validation:loss", value=val_loss, step=epoch)
        tracker.log_metric(name="validation:accuracy", value=val_accuracy, step=epoch)
    
    # Log final metrics
    tracker.log_metric(name="final_accuracy", value=final_accuracy)
    tracker.log_metric(name="training_time", value=training_time)
    
    # Associate the tracker with the trial
    trial.add_trial_component(tracker.trial_component)
    ''')
    
    # Additional example for comparing experiments
    st.markdown("### Querying and Comparing Experiments")
    st.code('''
# List all experiments
sm = boto3.Session().client('sagemaker')
experiments = sm.list_experiments()

# Get trials for a specific experiment
trials = sm.list_trials(ExperimentName=experiment_name)

# Get all metrics from trials
for trial in trials['TrialSummaries']:
    trial_name = trial['TrialName']
    components = sm.list_trial_components(TrialName=trial_name)
    
    for component in components['TrialComponentSummaries']:
        component_name = component['TrialComponentName']
        details = sm.describe_trial_component(TrialComponentName=component_name)
        
        # Get parameters
        parameters = details.get('Parameters', {})
        print(f"Trial: {trial_name}, Parameters: {parameters}")
        
        # Get metrics
        metrics = details.get('Metrics', {})
        print(f"Trial: {trial_name}, Metrics: {metrics}")
    ''')
    
    # Best practices
    st.markdown("### Best Practices for Experiment Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Organizing Experiments
        
        - **Use descriptive naming**
          - Include model type, dataset version, timestamp
          
        - **Group related trials**
          - Keep variations of the same model in one experiment
          
        - **Use tags consistently**
          - Tag experiments by project, team, purpose
          
        - **Document baseline experiments**
          - Always have a reference point for comparison
        """)
    
    with col2:
        st.markdown("""
        #### Tracking Strategy
        
        - **Track both inputs and outputs**
          - Log hyperparameters, code versions, and metrics
          
        - **Capture resource utilization**
          - Monitor training time, GPU usage, memory consumption
          
        - **Use consistent metrics**
          - Define standard metrics across all experiments
          
        - **Save artifacts**
          - Store model checkpoints and evaluation samples
        """)
        
    # Visualization showing experiment tracking workflow
    st.markdown("### SageMaker Experiments Workflow")
    
    # Define the workflow stages and connections
    workflow_stages = [
        {"id": 1, "name": "Configure\nExperiment", "x": 0.1, "y": 0.5},
        {"id": 2, "name": "Set\nHyperparameters", "x": 0.25, "y": 0.5},
        {"id": 3, "name": "Train\nModel", "x": 0.4, "y": 0.5},
        {"id": 4, "name": "Track\nMetrics", "x": 0.55, "y": 0.5},
        {"id": 5, "name": "Compare\nResults", "x": 0.7, "y": 0.5},
        {"id": 6, "name": "Select Best\nModel", "x": 0.85, "y": 0.5}
    ]
    
    # Create workflow visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Draw connections between stages
    for i in range(len(workflow_stages)-1):
        ax.annotate("", 
                    xy=(workflow_stages[i+1]["x"], workflow_stages[i+1]["y"]), 
                    xytext=(workflow_stages[i]["x"], workflow_stages[i]["y"]),
                    arrowprops=dict(arrowstyle="->", color=AWS_COLORS['dark_gray'], lw=2))
    
    # Draw nodes
    for stage in workflow_stages:
        ax.scatter(stage["x"], stage["y"], s=1500, 
                  color=AWS_COLORS['teal'] if stage["id"] % 2 == 0 else AWS_COLORS['orange'],
                  alpha=0.8, edgecolor='white')
        ax.text(stage["x"], stage["y"], stage["name"], 
               ha='center', va='center', color='white', fontweight='bold')
    
    # Customize appearance
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    st.pyplot(fig)

# TAB 4: SAGEMAKER SHADOW TESTING
with tab4:
    st.header("Amazon SageMaker Shadow Testing")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        SageMaker Shadow Testing allows you to evaluate the performance of new ML models 
        against existing production models without affecting the customer experience.
        
        **Key benefits:**
        - Test new models in production with real traffic and data
        - Compare performance between new and existing models
        - Mitigate risks of deploying underperforming models
        - Validate performance before full deployment
        """)
    
    with col2:
        # Shadow Testing concept illustration
        st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/09/16/ML-4493-image001.jpg",
                caption="SageMaker Shadow Testing Concept")

    # Interactive Shadow Testing Demo
    st.subheader("Interactive Demo: Shadow Testing a New Model")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Test Configuration")
        
        # Model selection
        model_options = ["Model A (Production)", "Model B (Candidate)"]
        st.session_state.shadow_test_model = st.radio("Select model to analyze:", ["Model A", "Model B"])
        
        # Traffic allocation slider
        traffic_allocation = st.slider(
            "Traffic allocation for shadow testing (%)", 
            min_value=10, 
            max_value=100, 
            value=st.session_state.shadow_test_traffic,
            step=10
        )
        st.session_state.shadow_test_traffic = traffic_allocation
        
        # Test duration
        test_duration = st.selectbox("Test duration", ["1 day", "3 days", "7 days", "14 days"])
        
        # Test metrics to monitor
        st.markdown("### Metrics to Monitor")
        metrics = {
            "Latency": st.checkbox("Latency", value=True),
            "Error Rate": st.checkbox("Error Rate", value=True),
            "Accuracy": st.checkbox("Accuracy", value=True),
            "CPU Usage": st.checkbox("CPU Usage", value=False),
            "Memory Usage": st.checkbox("Memory Usage", value=False)
        }
        
        # Start shadow test button
        if st.button("Run Shadow Test"):
            with st.spinner("Running shadow test..."):
                time.sleep(2)  # Simulate running the test
                st.success("Shadow test has been started!")
        
    with col2:
        st.markdown("### Shadow Testing Architecture")
        
        # Create shadow testing architecture diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Draw clients
        client_rect = plt.Rectangle((1, 4.5), 2, 1, fc=AWS_COLORS['light_gray'], ec=AWS_COLORS['dark_gray'])
        ax.add_patch(client_rect)
        ax.text(2, 5, "Clients", ha='center', va='center')
        
        # Draw API Gateway
        api_rect = plt.Rectangle((3.5, 3), 3, 0.8, fc=AWS_COLORS['teal'], ec='white')
        ax.add_patch(api_rect)
        ax.text(5, 3.4, "API Gateway", ha='center', va='center', color='white')
        
        # Draw Lambda
        lambda_rect = plt.Rectangle((4.25, 1.5), 1.5, 0.8, fc=AWS_COLORS['orange'], ec='white')
        ax.add_patch(lambda_rect)
        ax.text(5, 1.9, "Lambda", ha='center', va='center', color='white')
        
        # Draw Production Model
        prod_rect = plt.Rectangle((1.5, 0.2), 2, 0.8, fc=AWS_COLORS['blue'], ec='white')
        ax.add_patch(prod_rect)
        ax.text(2.5, 0.6, "Model A\n(Production)", ha='center', va='center', color='white')
        
        # Draw Shadow Model
        shadow_rect = plt.Rectangle((6.5, 0.2), 2, 0.8, fc=AWS_COLORS['green'], ec='white')
        ax.add_patch(shadow_rect)
        ax.text(7.5, 0.6, "Model B\n(Shadow)", ha='center', va='center', color='white')
        
        # Draw connections
        ax.arrow(2, 4.5, 0, -1, head_width=0.2, head_length=0.2, fc=AWS_COLORS['dark_gray'], ec=AWS_COLORS['dark_gray'])
        ax.arrow(5, 3, 0, -0.7, head_width=0.2, head_length=0.2, fc=AWS_COLORS['dark_gray'], ec=AWS_COLORS['dark_gray'])
        
        # Connections to models
        ax.arrow(4.5, 1.5, -1.5, -0.5, head_width=0.2, head_length=0.2, fc=AWS_COLORS['dark_gray'], ec=AWS_COLORS['dark_gray'])
        
        # Shadow connection - dashed line based on traffic allocation
        if traffic_allocation < 100:
            arrow_style = '--'
            alpha = 0.7
        else:
            arrow_style = '-'
            alpha = 1.0
            
        ax.arrow(5.5, 1.5, 1.5, -0.5, head_width=0.2, head_length=0.2, 
                fc=AWS_COLORS['dark_gray'], ec=AWS_COLORS['dark_gray'], 
                linestyle=arrow_style, alpha=alpha)
        
        # Traffic allocation label
        ax.text(6.2, 1.2, f"{traffic_allocation}% Traffic", ha='center', fontsize=9)
        
        # Data comparison arrow
        ax.arrow(2.5, 0.1, 5, 0, head_width=0.15, head_length=0.2, 
                fc=AWS_COLORS['orange'], ec=AWS_COLORS['orange'], linestyle='--')
        ax.text(5, 0, "Performance Comparison", ha='center', fontsize=9, color=AWS_COLORS['orange'])
        
        # Highlight selected model
        if st.session_state.shadow_test_model == "Model A":
            highlight = plt.Rectangle((1.3, 0), 2.4, 1.2, fill=False, ec='red', lw=2, linestyle='-')
            ax.add_patch(highlight)
        else:
            highlight = plt.Rectangle((6.3, 0), 2.4, 1.2, fill=False, ec='red', lw=2, linestyle='-')
            ax.add_patch(highlight)
        
        st.pyplot(fig)
    
    # Shadow test comparison results
    st.markdown("### Performance Comparison Results")
    
    # Generate model performance data
    def generate_model_comparison():
        # Model A (Production) baseline metrics
        model_a = {
            "accuracy": 0.92,
            "latency_ms": 112,
            "error_rate": 0.03,
            "throughput": 95,
            "precision": 0.90,
            "recall": 0.89,
            "f1_score": 0.895
        }
        
        # Model B varies - either better or worse depending on traffic allocation
        # Higher traffic means more confidence in the better metrics
        if st.session_state.shadow_test_traffic >= 70:
            # More confident results with higher traffic
            model_b = {
                "accuracy": 0.94,
                "latency_ms": 98,
                "error_rate": 0.025,
                "throughput": 105,
                "precision": 0.92, 
                "recall": 0.91,
                "f1_score": 0.915
            }
        else:
            # Less confident results with lower traffic
            model_b = {
                "accuracy": 0.93,
                "latency_ms": 105,
                "error_rate": 0.028,
                "throughput": 100,
                "precision": 0.91,
                "recall": 0.90,
                "f1_score": 0.905
            }
        
        return model_a, model_b
    
    model_a, model_b = generate_model_comparison()
    
    # Calculate percentage differences
    diffs = {}
    for key in model_a:
        if key in ["latency_ms", "error_rate"]:  # Lower is better
            diffs[key] = (model_a[key] - model_b[key]) / model_a[key] * 100
        else:  # Higher is better
            diffs[key] = (model_b[key] - model_a[key]) / model_a[key] * 100
    
    # Display metrics
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("#### Model A (Production)")
        st.markdown(f"**Accuracy:** {model_a['accuracy']:.3f}")
        st.markdown(f"**Latency:** {model_a['latency_ms']} ms")
        st.markdown(f"**Error Rate:** {model_a['error_rate']:.3f}")
        st.markdown(f"**Throughput:** {model_a['throughput']} req/sec")
        st.markdown(f"**F1 Score:** {model_a['f1_score']:.3f}")
    
    with col2:
        st.markdown("#### Model B (Shadow)")
        st.markdown(f"**Accuracy:** {model_b['accuracy']:.3f}")
        st.markdown(f"**Latency:** {model_b['latency_ms']} ms")
        st.markdown(f"**Error Rate:** {model_b['error_rate']:.3f}")
        st.markdown(f"**Throughput:** {model_b['throughput']} req/sec")
        st.markdown(f"**F1 Score:** {model_b['f1_score']:.3f}")
    
    with col3:
        st.markdown("#### Difference")
        st.markdown(f"**{diffs['accuracy']:.1f}%** {'✅' if diffs['accuracy'] > 0 else '❌'}")
        st.markdown(f"**{diffs['latency_ms']:.1f}%** {'✅' if diffs['latency_ms'] > 0 else '❌'}")
        st.markdown(f"**{diffs['error_rate']:.1f}%** {'✅' if diffs['error_rate'] > 0 else '❌'}")
        st.markdown(f"**{diffs['throughput']:.1f}%** {'✅' if diffs['throughput'] > 0 else '❌'}")
        st.markdown(f"**{diffs['f1_score']:.1f}%** {'✅' if diffs['f1_score'] > 0 else '❌'}")
    
    # Detailed performance comparison
    st.markdown("### Detailed Performance Analysis")
    
    # Create tabs for different metric visualizations
    perf_tab1, perf_tab2, perf_tab3 = st.tabs(["Response Time", "Accuracy Metrics", "Resource Utilization"])
    
    with perf_tab1:
        # Generate synthetic latency distribution data
        def generate_latency_distribution(mean, std, n=1000):
            np.random.seed(42)  # For reproducibility
            return np.random.gamma(shape=(mean/std)**2, scale=std**2/(mean))
        
        model_a_latencies = generate_latency_distribution(model_a["latency_ms"], 20)
        model_b_latencies = generate_latency_distribution(model_b["latency_ms"], 18)
        
        # Plot latency distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        bins = np.linspace(40, 200, 30)
        ax.hist(model_a_latencies, bins=bins, alpha=0.5, label='Model A', color=AWS_COLORS['blue'])
        ax.hist(model_b_latencies, bins=bins, alpha=0.5, label='Model B', color=AWS_COLORS['green'])
        
        # Add vertical lines for means
        ax.axvline(x=model_a["latency_ms"], color=AWS_COLORS['blue'], linestyle='--', linewidth=2)
        ax.axvline(x=model_b["latency_ms"], color=AWS_COLORS['green'], linestyle='--', linewidth=2)
        
        # Add text for mean values
        ax.text(model_a["latency_ms"]+5, ax.get_ylim()[1]*0.9, f'Mean: {model_a["latency_ms"]}ms', 
               color=AWS_COLORS['blue'], fontweight='bold')
        ax.text(model_b["latency_ms"]+5, ax.get_ylim()[1]*0.8, f'Mean: {model_b["latency_ms"]}ms', 
               color=AWS_COLORS['green'], fontweight='bold')
        
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Response Time Distribution')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Calculate p95 and p99 latencies
        p95_a = np.percentile(model_a_latencies, 95)
        p95_b = np.percentile(model_b_latencies, 95)
        p99_a = np.percentile(model_a_latencies, 99)
        p99_b = np.percentile(model_b_latencies, 99)
        
        # Show percentile metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("p95 Latency - Model A", f"{p95_a:.1f} ms")
            st.metric("p99 Latency - Model A", f"{p99_a:.1f} ms")
        with col2:
            st.metric("p95 Latency - Model B", f"{p95_b:.1f} ms", delta=f"{(p95_a-p95_b):.1f} ms")
            st.metric("p99 Latency - Model B", f"{p99_b:.1f} ms", delta=f"{(p99_a-p99_b):.1f} ms")
        
    with perf_tab2:
        # Generate confusion matrix data
        def generate_confusion_matrix(precision, recall, samples=1000):
            # For a binary classification problem
            tp = int(samples * precision * recall)
            fp = int(tp * (1 - precision) / precision)
            fn = int(tp * (1 - recall) / recall)
            tn = samples - (tp + fp + fn)
            return np.array([[tp, fp], [fn, tn]])
        
        cm_a = generate_confusion_matrix(model_a["precision"], model_a["recall"])
        cm_b = generate_confusion_matrix(model_b["precision"], model_b["recall"])
        
        # Plot confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                   xticklabels=['Predicted +', 'Predicted -'],
                   yticklabels=['Actual +', 'Actual -'])
        ax1.set_title('Model A Confusion Matrix')
        
        sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=ax2,
                   xticklabels=['Predicted +', 'Predicted -'],
                   yticklabels=['Actual +', 'Actual -'])
        ax2.set_title('Model B Confusion Matrix')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Precision-Recall comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Precision', 'Recall', 'F1 Score']
        model_a_values = [model_a['precision'], model_a['recall'], model_a['f1_score']]
        model_b_values = [model_b['precision'], model_b['recall'], model_b['f1_score']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, model_a_values, width, label='Model A', color=AWS_COLORS['blue'])
        ax.bar(x + width/2, model_b_values, width, label='Model B', color=AWS_COLORS['green'])
        
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall and F1 Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add values on top of bars
        for i, v in enumerate(model_a_values):
            ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
            
        for i, v in enumerate(model_b_values):
            ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
        
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
    with perf_tab3:
        # Generate synthetic resource utilization data
        def generate_resource_data():
            np.random.seed(42)
            # Generate time points (e.g., 24 hours)
            time_points = np.arange(24)
            
            # CPU usage patterns with daily variation
            model_a_cpu = 60 + 15 * np.sin(np.pi * time_points / 12) + np.random.normal(0, 5, 24)
            model_b_cpu = 55 + 12 * np.sin(np.pi * time_points / 12) + np.random.normal(0, 4, 24)
            
            # Memory usage patterns
            model_a_mem = 70 + 5 * np.sin(np.pi * time_points / 8) + np.random.normal(0, 3, 24)
            model_b_mem = 65 + 7 * np.sin(np.pi * time_points / 8) + np.random.normal(0, 3, 24)
            
            # Ensure values are within reasonable ranges
            model_a_cpu = np.clip(model_a_cpu, 0, 100)
            model_b_cpu = np.clip(model_b_cpu, 0, 100)
            model_a_mem = np.clip(model_a_mem, 0, 100)
            model_b_mem = np.clip(model_b_mem, 0, 100)
            
            return time_points, model_a_cpu, model_b_cpu, model_a_mem, model_b_mem
        
        time_points, model_a_cpu, model_b_cpu, model_a_mem, model_b_mem = generate_resource_data()
        
        # Plot CPU usage
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        ax1.plot(time_points, model_a_cpu, 'o-', label='Model A', color=AWS_COLORS['blue'])
        ax1.plot(time_points, model_b_cpu, 'o-', label='Model B', color=AWS_COLORS['green'])
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Utilization Over Time')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 100)
        ax1.fill_between(time_points, model_a_cpu, alpha=0.2, color=AWS_COLORS['blue'])
        ax1.fill_between(time_points, model_b_cpu, alpha=0.2, color=AWS_COLORS['green'])
        
        # Plot Memory usage
        ax2.plot(time_points, model_a_mem, 'o-', label='Model A', color=AWS_COLORS['blue'])
        ax2.plot(time_points, model_b_mem, 'o-', label='Model B', color=AWS_COLORS['green'])
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_title('Memory Utilization Over Time')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 100)
        ax2.fill_between(time_points, model_a_mem, alpha=0.2, color=AWS_COLORS['blue'])
        ax2.fill_between(time_points, model_b_mem, alpha=0.2, color=AWS_COLORS['green'])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Resource summary metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg CPU Usage - Model A", f"{np.mean(model_a_cpu):.1f}%")
            st.metric("Avg Memory Usage - Model A", f"{np.mean(model_a_mem):.1f}%")
            st.metric("Peak CPU Usage - Model A", f"{np.max(model_a_cpu):.1f}%")
        
        with col2:
            cpu_diff = np.mean(model_a_cpu) - np.mean(model_b_cpu)
            mem_diff = np.mean(model_a_mem) - np.mean(model_b_mem)
            
            st.metric("Avg CPU Usage - Model B", f"{np.mean(model_b_cpu):.1f}%", 
                     delta=f"{cpu_diff:.1f}%" if cpu_diff > 0 else f"-{-cpu_diff:.1f}%")
            st.metric("Avg Memory Usage - Model B", f"{np.mean(model_b_mem):.1f}%", 
                     delta=f"{mem_diff:.1f}%" if mem_diff > 0 else f"-{-mem_diff:.1f}%")
            st.metric("Peak CPU Usage - Model B", f"{np.max(model_b_cpu):.1f}%")
    
    # Final recommendation
    st.markdown("### Shadow Test Conclusion")
    
    # Calculate overall recommendation
    if (diffs['accuracy'] > 1 and diffs['latency_ms'] > 5 and 
        diffs['error_rate'] > 5 and diffs['throughput'] > 5):
        recommendation = "DEPLOY"
        confidence = "HIGH"
        color = AWS_COLORS['green']
    elif (diffs['accuracy'] > 0 and diffs['latency_ms'] > 0 and 
          diffs['error_rate'] > 0):
        recommendation = "DEPLOY"
        confidence = "MEDIUM"
        color = AWS_COLORS['teal']
    elif (diffs['accuracy'] < -1 or diffs['latency_ms'] < -5 or 
          diffs['error_rate'] < -5):
        recommendation = "DO NOT DEPLOY"
        confidence = "HIGH"
        color = AWS_COLORS['red']
    else:
        recommendation = "CONTINUE TESTING"
        confidence = "LOW"
        color = AWS_COLORS['orange']
    
    # Display recommendation in a nice box
    st.markdown(f"""
    <div style="background-color:{color}; padding:20px; border-radius:10px; color:white;">
        <h3 style="margin:0;">Recommendation: {recommendation}</h3>
        <p style="margin:5px 0 0 0;">Confidence: {confidence}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display specific reasons
    st.markdown("#### Key Findings:")
    reasons = []
    
    if diffs['accuracy'] > 1:
        reasons.append(f"✅ Model B shows {diffs['accuracy']:.1f}% higher accuracy")
    elif diffs['accuracy'] < 0:
        reasons.append(f"❌ Model B shows {-diffs['accuracy']:.1f}% lower accuracy")
        
    if diffs['latency_ms'] > 5:
        reasons.append(f"✅ Model B is {diffs['latency_ms']:.1f}% faster")
    elif diffs['latency_ms'] < 0:
        reasons.append(f"❌ Model B is {-diffs['latency_ms']:.1f}% slower")
        
    if diffs['error_rate'] > 5:
        reasons.append(f"✅ Model B has {diffs['error_rate']:.1f}% lower error rate")
    elif diffs['error_rate'] < 0:
        reasons.append(f"❌ Model B has {-diffs['error_rate']:.1f}% higher error rate")
        
    if np.mean(model_b_cpu) < np.mean(model_a_cpu):
        cpu_saving = np.mean(model_a_cpu) - np.mean(model_b_cpu)
        reasons.append(f"✅ Model B uses {cpu_saving:.1f}% less CPU on average")
    
    for reason in reasons:
        st.markdown(reason)
        
    if st.session_state.shadow_test_traffic < 70:
        st.warning("⚠️ Consider increasing shadow traffic allocation for higher confidence in results")
    
    # Code example for Shadow Testing
    st.markdown("### Implementing SageMaker Shadow Testing")
    st.code('''
import boto3
import json
from datetime import datetime

# Function to set up shadow testing endpoints
def setup_shadow_testing():
    client = boto3.client('sagemaker')
    
    # 1. Create production endpoint
    production_endpoint_config = client.create_endpoint_config(
        EndpointConfigName='prod-model-config',
        ProductionVariants=[{
            'VariantName': 'ProductionVariant',
            'ModelName': 'prod-model',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge'
        }]
    )
    
    production_endpoint = client.create_endpoint(
        EndpointName='production-endpoint',
        EndpointConfigName='prod-model-config'
    )
    
    # 2. Create shadow endpoint
    shadow_endpoint_config = client.create_endpoint_config(
        EndpointConfigName='shadow-model-config',
        ProductionVariants=[{
            'VariantName': 'ShadowVariant',
            'ModelName': 'shadow-model',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge'
        }]
    )
    
    shadow_endpoint = client.create_endpoint(
        EndpointName='shadow-endpoint',
        EndpointConfigName='shadow-model-config'
    )
    
    return production_endpoint, shadow_endpoint

# Lambda function for shadow testing
def lambda_handler(event, context):
    # Parse the input
    input_data = json.loads(event['body'])
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    # Get current timestamp for metrics
    timestamp = datetime.now().isoformat()
    
    try:
        # Call the production endpoint
        production_response = sagemaker_runtime.invoke_endpoint(
            EndpointName='production-endpoint',
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        
        # Parse production response
        production_result = json.loads(production_response['Body'].read().decode())
        
        # Shadow traffic - call the shadow endpoint asynchronously
        # Note: In a real implementation, you'd want to do this asynchronously
        # to not impact production latency
        shadow_response = sagemaker_runtime.invoke_endpoint(
            EndpointName='shadow-endpoint',
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        
        # Parse shadow response (for logging/comparison)
        shadow_result = json.loads(shadow_response['Body'].read().decode())
        
        # Log metrics for comparison
        cloudwatch = boto3.client('cloudwatch')
        
        # Log production model metrics
        cloudwatch.put_metric_data(
            Namespace='ShadowTest',
            MetricData=[
                {
                    'MetricName': 'ProductionLatency',
                    'Timestamp': timestamp,
                    'Value': production_response['ResponseMetadata']['HTTPHeaders']['x-amzn-sagemaker-inference-time'],
                    'Unit': 'Milliseconds'
                }
            ]
        )
        
        # Log shadow model metrics
        cloudwatch.put_metric_data(
            Namespace='ShadowTest',
            MetricData=[
                {
                    'MetricName': 'ShadowLatency',
                    'Timestamp': timestamp,
                    'Value': shadow_response['ResponseMetadata']['HTTPHeaders']['x-amzn-sagemaker-inference-time'],
                    'Unit': 'Milliseconds'
                }
            ]
        )
        
        # Only return the production result to the client
        return {
            'statusCode': 200,
            'body': json.dumps(production_result)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    ''')
    
    # Shadow testing process
    st.markdown("### Shadow Testing Process Workflow")
    
    # Create a visual timeline of the shadow testing process
    process_steps = [
        {"step": 1, "name": "Deploy Production Model", "description": "Implement production model endpoint"},
        {"step": 2, "name": "Deploy Shadow Model", "description": "Set up shadow variant with new model"},
        {"step": 3, "name": "Duplicate Traffic", "description": "Route % of requests to shadow endpoint"},
        {"step": 4, "name": "Collect Metrics", "description": "Gather performance data from both models"},
        {"step": 5, "name": "Compare & Analyze", "description": "Compare metrics between models"},
        {"step": 6, "name": "Make Decision", "description": "Promote, reject, or continue testing"}
    ]
    
    # Create columns for steps
    cols = st.columns(len(process_steps))
    
    # Display each step
    for i, (col, step) in enumerate(zip(cols, process_steps)):
        with col:
            st.markdown(f"### {step['step']}")
            st.markdown(f"**{step['name']}**")
            st.markdown(step['description'])
            
            # Add connecting arrows between steps
            if i < len(process_steps) - 1:
                st.markdown("➡️")

# Add footer
st.markdown("""
<div class="footer">
© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</div>
""", unsafe_allow_html=True)