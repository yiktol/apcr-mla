
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import time
from PIL import Image
import base64
import io

# Set page configuration
st.set_page_config(
    page_title="Ensemble Learning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS Color Scheme
AWS_COLORS = {
    "orange": "#FF9900",
    "light_orange": "#FFAC31",
    "dark_blue": "#232F3E",
    "light_blue": "#1A73E8",
    "teal": "#00A1C9",
    "red": "#D13212",
    "green": "#7AA116",
    "purple": "#8C4FFF",
    "light_grey": "#F2F3F3",
    "slate": "#687078",
    "white": "#FFFFFF",
    "background": "#F8F8F8"
}

# Custom CSS for AWS styling
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #232F3E;
        color: white;
    }
    h1, h2 {
        color: #232F3E;
    }
    h3, h4, h5 {
        color: #FF9900;
    }
    .stButton>button {
        background-color: #FF9900;
        color: white;
    }
    .stButton>button:hover {
        background-color: #FFAC31;
    }
    .highlight {
        background-color: #FFAC31;
        padding: 10px;
        border-radius: 5px;
    }
    .card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F2F3F3;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #232F3E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF9900 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None 
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = 'classification'
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Sidebar
with st.sidebar:
    
    # Session management
    st.subheader("⚙️ Session Management")
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Data generation options
    st.markdown("---")
    st.subheader("🛠️ Data Configuration")
    task = st.radio("Select Task Type:", ["Classification", "Regression"])
    st.session_state.task_type = task.lower()
    
    dataset_option = st.selectbox(
        "Select Dataset:",
        ["Generated Data", "Breast Cancer (Classification)", "Diabetes (Regression)"]
    )
    
    if st.button("Generate Data", key="generate_data_btn"):
        with st.spinner("Generating dataset..."):
            if dataset_option == "Generated Data":
                if st.session_state.task_type == 'classification':
                    X, y = make_classification(
                        n_samples=1000, n_features=20, n_informative=10,
                        n_redundant=5, random_state=42
                    )
                else:
                    X, y = make_regression(
                        n_samples=1000, n_features=20, n_informative=10,
                        random_state=42, noise=0.1
                    )
            elif dataset_option == "Breast Cancer (Classification)":
                data = load_breast_cancer()
                X, y = data.data, data.target
                st.session_state.task_type = 'classification'
            elif dataset_option == "Diabetes (Regression)":
                data = load_diabetes()
                X, y = data.data, data.target
                st.session_state.task_type = 'regression'
                
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Store in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.dataset = dataset_option
            
            st.success("✅ Data generated successfully!")
    
    st.markdown("---")
    st.markdown("### About This App")
    st.info("""
    This interactive explorer demonstrates various ensemble learning techniques 
    used in machine learning to improve model performance.
    
    Learn about:
    - Bagging
    - Boosting
    - Stacking
    
    Try the interactive examples to see how each technique affects model performance!
    """)

# Main content
st.title("Ensemble Learning")

st.markdown("""
<div class="card">
<p>Ensemble learning is a powerful machine learning paradigm where multiple models (often called "weak learners") 
are trained to solve the same problem and combined to get better results. This approach frequently produces more 
accurate solutions than a single model would.</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs([
    "📊 Overview", 
    "🌲 Bagging", 
    "🚀 Boosting", 
    "🏗️ Stacking"
])

# Tab 1: Overview
with tabs[0]:
    st.header("Understanding Ensemble Learning")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### What is Ensemble Learning? 🤔
        
        Ensemble learning combines multiple models to improve predictions, reduce overfitting,
        and increase model stability. By aggregating the predictions of multiple models, 
        ensemble methods can achieve better performance than any single model.
        
        The key principle behind ensemble learning is that a group of "weak learners" 
        can come together to form a "strong learner".
        
        ### Why Use Ensemble Methods?
        """)
        
        benefits = {
            "Improved Accuracy": "Multiple models often yield better predictions than any single model",
            "Reduced Overfitting": "Ensembles help in reducing model variance and generalization error",
            "Increased Stability": "Less sensitive to peculiarities of the data",
            "Better Insights": "Different models may capture different aspects of the data"
        }
        
        for benefit, description in benefits.items():
            st.markdown(f"""
            <div style="margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: {AWS_COLORS['light_grey']}; border-left: 5px solid {AWS_COLORS['orange']}">
                <strong style="color: {AWS_COLORS['dark_blue']};">{benefit}:</strong> {description}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Show ensemble learning concept image
        st.image("https://miro.medium.com/max/1200/1*4G__SV580CxFj-xlBw_Zvg.png", caption="Ensemble Learning Concept")
        
        st.markdown("""
        <div style="text-align: center; font-style: italic; margin-top: 10px;">
            Visual representation of how ensemble learning combines multiple models
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <h4>Main Categories of Ensemble Methods</h4>
        <ul>
        <li><strong>Bagging:</strong> Train models in parallel on random subsets</li>
        <li><strong>Boosting:</strong> Train models sequentially, each focusing on previous errors</li>
        <li><strong>Stacking:</strong> Combine predictions using another learning algorithm</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Getting Started")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card" style="background-color: #E9F5FF;">
            <h3 style="color: #00A1C9;">🌲 Bagging</h3>
            <p>Bootstrap Aggregating - trains multiple instances of the same model on different random subsets of the training data.</p>
            <p><strong>Examples:</strong> Random Forests, Bagging Classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="background-color: #FFF8F0;">
            <h3 style="color: #FF9900;">🚀 Boosting</h3>
            <p>Trains models sequentially, with each new model focusing on the errors of previous ones.</p>
            <p><strong>Examples:</strong> AdaBoost, Gradient Boosting, XGBoost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card" style="background-color: #F0FFF4;">
            <h3 style="color: #7AA116;">🏗️ Stacking</h3>
            <p>Combines multiple models using another meta-model to optimize the outputs.</p>
            <p><strong>Examples:</strong> Stacking Classifier, Blending</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("How to Use This App")
    st.markdown("""
    <ol>
        <li>Use the sidebar to select a dataset and task type (classification or regression)</li>
        <li>Click "Generate Data" to prepare your dataset</li>
        <li>Navigate through the tabs to explore different ensemble learning techniques</li>
        <li>Experiment with parameters and observe how they affect model performance</li>
    </ol>
    """, unsafe_allow_html=True)
    
    # Show warning if data is not generated yet
    if st.session_state.X_train is None:
        st.warning("👆 Please generate a dataset first using the sidebar options.")

# Tab 2: Bagging
with tabs[1]:
    st.header("Bagging (Bootstrap Aggregating)")
    
    st.markdown("""
    <div class="card">
    <h3>What is Bagging? 🌲</h3>
    <p>Bagging (Bootstrap Aggregating) is an ensemble technique that involves training multiple instances 
    of the same model on different random subsets of the training data and then combining their predictions.</p>
    
    <h4>When to use Bagging:</h4>
    <ul>
    <li>When you want to reduce variance and overfitting</li>
    <li>When your base model is sensitive to variations in the training data</li>
    <li>When you have a complex model with high variance (like decision trees)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### How Bagging Works

        Bagging works by following these steps:

        1. Create multiple random subsets (bootstrap samples) of the training data
        2. Train a base model on each subset
        3. Combine predictions by voting (classification) or averaging (regression)

        ### Code Example
        ```python
        # Using BaggingClassifier with decision trees
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier

        bagging_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            random_state=42
        )
        
        bagging_model.fit(X_train, y_train)
        ```

        Popular implementations include Random Forests, which combine bagging with feature randomization.
        """)
    
    with col2:
        # Show bagging process image
        st.image("https://miro.medium.com/max/1200/1*_NJ9oPKK1BnCYLMxmVgA_w.png", caption="Bagging Process")
        
        st.markdown("""
        <div class="card">
        <h4>Advantages of Bagging:</h4>
        <ul>
        <li>Reduces variance without increasing bias</li>
        <li>Reduces overfitting</li>
        <li>Provides more stable predictions</li>
        <li>Models can be trained in parallel</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Interactive Bagging Demo")
    
    if st.session_state.X_train is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            n_estimators = st.slider(
                "Number of Base Estimators", 
                min_value=1, 
                max_value=100, 
                value=10,
                key="bagging_n_estimators"
            )
            
            max_samples = st.slider(
                "Max Samples (% of training data)", 
                min_value=10, 
                max_value=100, 
                value=80,
                key="bagging_max_samples"
            ) / 100.0
            
            max_features = st.slider(
                "Max Features (% of features used)", 
                min_value=10, 
                max_value=100, 
                value=80,
                key="bagging_max_features"
            ) / 100.0
            
            bootstrap = st.checkbox("Bootstrap", value=True, key="bagging_bootstrap")
            
            if st.button("Train Bagging Model", key="train_bagging"):
                with st.spinner("Training..."):
                    if st.session_state.task_type == 'classification':
                        # Base model
                        base_model = DecisionTreeClassifier(max_depth=3)
                        base_model.fit(st.session_state.X_train, st.session_state.y_train)
                        base_preds = base_model.predict(st.session_state.X_test)
                        base_accuracy = accuracy_score(st.session_state.y_test, base_preds)
                        
                        # Bagging model
                        bagging_model = BaggingClassifier(
                            estimator=DecisionTreeClassifier(max_depth=3),
                            n_estimators=n_estimators,
                            max_samples=max_samples,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            random_state=42
                        )
                        
                        bagging_model.fit(st.session_state.X_train, st.session_state.y_train)
                        bagging_preds = bagging_model.predict(st.session_state.X_test)
                        bagging_accuracy = accuracy_score(st.session_state.y_test, bagging_preds)
                        
                        # Store results
                        st.session_state.model_results["bagging"] = {
                            "base_accuracy": base_accuracy,
                            "bagging_accuracy": bagging_accuracy,
                            "conf_matrix": confusion_matrix(st.session_state.y_test, bagging_preds),
                            "classification_report": classification_report(st.session_state.y_test, bagging_preds, output_dict=True)
                        }
                        
                    else:  # regression
                        # Base model
                        base_model = DecisionTreeRegressor(max_depth=3)
                        base_model.fit(st.session_state.X_train, st.session_state.y_train)
                        base_preds = base_model.predict(st.session_state.X_test)
                        base_mse = mean_squared_error(st.session_state.y_test, base_preds)
                        base_r2 = r2_score(st.session_state.y_test, base_preds)
                        
                        # Bagging model
                        bagging_model = BaggingRegressor(
                            estimator=DecisionTreeRegressor(max_depth=3),
                            n_estimators=n_estimators,
                            max_samples=max_samples,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            random_state=42
                        )
                        
                        bagging_model.fit(st.session_state.X_train, st.session_state.y_train)
                        bagging_preds = bagging_model.predict(st.session_state.X_test)
                        bagging_mse = mean_squared_error(st.session_state.y_test, bagging_preds)
                        bagging_r2 = r2_score(st.session_state.y_test, bagging_preds)
                        
                        # Store results
                        st.session_state.model_results["bagging"] = {
                            "base_mse": base_mse,
                            "base_r2": base_r2,
                            "bagging_mse": bagging_mse,
                            "bagging_r2": bagging_r2
                        }
                
                st.success("✅ Bagging model trained successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if "bagging" in st.session_state.model_results:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                if st.session_state.task_type == 'classification':
                    # Get results
                    results = st.session_state.model_results["bagging"]
                    
                    # Create a comparison bar chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Base Model', 'Bagging Model'],
                        y=[results["base_accuracy"] * 100, results["bagging_accuracy"] * 100],
                        text=[f'{results["base_accuracy"]:.2%}', f'{results["bagging_accuracy"]:.2%}'],
                        textposition='auto',
                        marker_color=[AWS_COLORS["slate"], AWS_COLORS["teal"]]
                    ))
                    fig.update_layout(
                        title='Model Accuracy Comparison',
                        yaxis_title='Accuracy (%)',
                        yaxis=dict(range=[0, 100]),
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    class_report = results["classification_report"]
                    cm = results["conf_matrix"]
                    
                    fig = px.imshow(
                        cm, 
                        text_auto=True,
                        color_continuous_scale=px.colors.sequential.Blues,
                        labels=dict(x="Predicted Label", y="True Label"),
                        x=['Class 0', 'Class 1'],
                        y=['Class 0', 'Class 1']
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display classification report
                    st.subheader("Classification Report")
                    report_df = pd.DataFrame(class_report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0))
                    
                else:  # regression
                    # Get results
                    results = st.session_state.model_results["bagging"]
                    
                    # Create plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # MSE Comparison
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['Base Model', 'Bagging Model'],
                            y=[results["base_mse"], results["bagging_mse"]],
                            text=[f'{results["base_mse"]:.4f}', f'{results["bagging_mse"]:.4f}'],
                            textposition='auto',
                            marker_color=[AWS_COLORS["slate"], AWS_COLORS["teal"]]
                        ))
                        fig.update_layout(
                            title='Mean Squared Error (Lower is Better)',
                            yaxis_title='MSE',
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # R² Comparison
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['Base Model', 'Bagging Model'],
                            y=[results["base_r2"], results["bagging_r2"]],
                            text=[f'{results["base_r2"]:.4f}', f'{results["bagging_r2"]:.4f}'],
                            textposition='auto',
                            marker_color=[AWS_COLORS["slate"], AWS_COLORS["teal"]]
                        ))
                        fig.update_layout(
                            title='R² Score (Higher is Better)',
                            yaxis_title='R²',
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="card">
                    <h3>Interpreting the Results:</h3>
                    <ul>
                        <li>Bagging typically improves performance by reducing variance</li>
                        <li>Increasing the number of estimators generally helps but with diminishing returns</li>
                        <li>The optimal max_samples and max_features values depend on the dataset</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Train the model to see the results!")
    else:
        st.warning("Please generate a dataset first using the sidebar options.")

    st.markdown("---")
    
    st.subheader("Real-World Applications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Random Forests</h3>
            <p>Random Forests are the most popular implementation of bagging. They combine bagging with random feature selection 
            to create diverse decision trees.</p>
            <ul>
            <li>Used in finance for credit risk assessment</li>
            <li>Medical diagnosis and disease prediction</li>
            <li>Customer churn prediction in telecommunications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Pasting</h3>
            <p>A variation of bagging that samples without replacement instead of bootstrap sampling</p>
            <ul>
            <li>Image classification tasks</li>
            <li>Text categorization systems</li>
            <li>Anomaly detection in network security</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Boosting
with tabs[2]:
    st.header("Boosting")
    
    st.markdown("""
    <div class="card">
    <h3>What is Boosting? 🚀</h3>
    <p>Boosting is an ensemble technique that combines multiple weak learners into a strong learner by training models 
    sequentially, with each new model focusing on the errors of the previous ones.</p>
    
    <h4>When to use Boosting:</h4>
    <ul>
    <li>When you want to reduce bias and improve predictive accuracy</li>
    <li>When you have a weak model that performs slightly better than random guessing</li>
    <li>When you need to give more importance to certain observations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### How Boosting Works

        Boosting works by following these steps:

        1. Train a base model on the original data
        2. Identify misclassified instances
        3. Increase the weight of misclassified instances
        4. Train a new model on the weighted data
        5. Continue this process sequentially
        6. Combine models using weighted voting

        ### Code Example
        ```python
        # Using AdaBoost for classification
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        boosting_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=0.1,
            random_state=42
        )
        
        boosting_model.fit(X_train, y_train)
        ```

        Popular boosting algorithms include AdaBoost, Gradient Boosting, XGBoost, and LightGBM.
        """)
    
    with col2:
        # Show boosting process image
        st.image("https://miro.medium.com/max/1200/1*Vc4vmEPwQfPP1qkDzFGnMQ.png", caption="Boosting Process")
        
        st.markdown("""
        <div class="card">
        <h4>Advantages of Boosting:</h4>
        <ul>
        <li>Often provides higher accuracy than single models</li>
        <li>Can create strong predictors from relatively weak learners</li>
        <li>Reduces bias in the learning algorithm</li>
        <li>Works well on a variety of problems</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Boosting Demo
    st.subheader("Interactive Boosting Demo")
    
    if st.session_state.X_train is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            boosting_algorithm = st.selectbox(
                "Boosting Algorithm",
                ["AdaBoost", "Gradient Boosting"],
                key="boosting_algorithm"
            )
            
            n_estimators = st.slider(
                "Number of Estimators", 
                min_value=10, 
                max_value=200, 
                value=50,
                step=10,
                key="boosting_n_estimators"
            )
            
            learning_rate = st.slider(
                "Learning Rate", 
                min_value=0.01, 
                max_value=1.0, 
                value=0.1,
                step=0.05,
                key="boosting_learning_rate"
            )
            
            if boosting_algorithm == "Gradient Boosting":
                max_depth = st.slider(
                    "Max Depth of Base Estimators", 
                    min_value=1, 
                    max_value=10, 
                    value=3,
                    key="boosting_max_depth"
                )
            
            if st.button("Train Boosting Model", key="train_boosting"):
                with st.spinner("Training..."):
                    if st.session_state.task_type == 'classification':
                        # Base model
                        base_model = DecisionTreeClassifier(max_depth=1)
                        base_model.fit(st.session_state.X_train, st.session_state.y_train)
                        base_preds = base_model.predict(st.session_state.X_test)
                        base_accuracy = accuracy_score(st.session_state.y_test, base_preds)
                        
                        # Boosting model
                        if boosting_algorithm == "AdaBoost":
                            boosting_model = AdaBoostClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                random_state=42
                            )
                        else:
                            boosting_model = GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                random_state=42
                            )
                        
                        boosting_model.fit(st.session_state.X_train, st.session_state.y_train)
                        
                        # Track performance across iterations
                        staged_scores = []
                        if boosting_algorithm == "Gradient Boosting":
                            for i, y_pred in enumerate(boosting_model.staged_predict(st.session_state.X_test)):
                                staged_scores.append(accuracy_score(st.session_state.y_test, y_pred))
                        
                        boosting_preds = boosting_model.predict(st.session_state.X_test)
                        boosting_accuracy = accuracy_score(st.session_state.y_test, boosting_preds)
                        
                        # Store results
                        st.session_state.model_results["boosting"] = {
                            "base_accuracy": base_accuracy,
                            "boosting_accuracy": boosting_accuracy,
                            "conf_matrix": confusion_matrix(st.session_state.y_test, boosting_preds),
                            "classification_report": classification_report(st.session_state.y_test, boosting_preds, output_dict=True),
                            "staged_scores": staged_scores,
                            "feature_importance": boosting_model.feature_importances_
                        }
                        
                    else:  # regression
                        # Base model
                        base_model = DecisionTreeRegressor(max_depth=1)
                        base_model.fit(st.session_state.X_train, st.session_state.y_train)
                        base_preds = base_model.predict(st.session_state.X_test)
                        base_mse = mean_squared_error(st.session_state.y_test, base_preds)
                        base_r2 = r2_score(st.session_state.y_test, base_preds)
                        
                        # Boosting model
                        if boosting_algorithm == "AdaBoost":
                            boosting_model = AdaBoostRegressor(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                random_state=42
                            )
                        else:
                            boosting_model = GradientBoostingRegressor(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                random_state=42
                            )
                        
                        boosting_model.fit(st.session_state.X_train, st.session_state.y_train)
                        
                        # Track performance across iterations
                        staged_scores = []
                        if boosting_algorithm == "Gradient Boosting":
                            for i, y_pred in enumerate(boosting_model.staged_predict(st.session_state.X_test)):
                                staged_scores.append(mean_squared_error(st.session_state.y_test, y_pred))
                        
                        boosting_preds = boosting_model.predict(st.session_state.X_test)
                        boosting_mse = mean_squared_error(st.session_state.y_test, boosting_preds)
                        boosting_r2 = r2_score(st.session_state.y_test, boosting_preds)
                        
                        # Store results
                        st.session_state.model_results["boosting"] = {
                            "base_mse": base_mse,
                            "base_r2": base_r2,
                            "boosting_mse": boosting_mse,
                            "boosting_r2": boosting_r2,
                            "staged_scores": staged_scores,
                            "feature_importance": boosting_model.feature_importances_
                        }
                
                st.success(f"✅ {boosting_algorithm} model trained successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if "boosting" in st.session_state.model_results:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                results = st.session_state.model_results["boosting"]
                
                if st.session_state.task_type == 'classification':
                    # Create a comparison bar chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Base Model', f'{boosting_algorithm}'],
                        y=[results["base_accuracy"] * 100, results["boosting_accuracy"] * 100],
                        text=[f'{results["base_accuracy"]:.2%}', f'{results["boosting_accuracy"]:.2%}'],
                        textposition='auto',
                        marker_color=[AWS_COLORS["slate"], AWS_COLORS["orange"]]
                    ))
                    fig.update_layout(
                        title='Model Accuracy Comparison',
                        yaxis_title='Accuracy (%)',
                        yaxis=dict(range=[0, 100]),
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Learning curve
                    if len(results["staged_scores"]) > 0:
                        fig = px.line(
                            x=list(range(1, len(results["staged_scores"]) + 1)),
                            y=results["staged_scores"],
                            labels={'x': 'Number of Trees', 'y': 'Accuracy'},
                            title='Learning Curve: Effect of Adding More Trees'
                        )
                        fig.update_traces(line_color=AWS_COLORS["light_blue"])
                        st.plotly_chart(fig, use_container_width=True)
                    
                else:  # regression
                    # MSE Comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Base Model', f'{boosting_algorithm}'],
                        y=[results["base_mse"], results["boosting_mse"]],
                        text=[f'{results["base_mse"]:.4f}', f'{results["boosting_mse"]:.4f}'],
                        textposition='auto',
                        marker_color=[AWS_COLORS["slate"], AWS_COLORS["orange"]]
                    ))
                    fig.update_layout(
                        title='Mean Squared Error (Lower is Better)',
                        yaxis_title='MSE',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Learning curve
                    if len(results["staged_scores"]) > 0:
                        fig = px.line(
                            x=list(range(1, len(results["staged_scores"]) + 1)),
                            y=results["staged_scores"],
                            labels={'x': 'Number of Trees', 'y': 'MSE'},
                            title='Learning Curve: Effect of Adding More Trees'
                        )
                        fig.update_traces(line_color=AWS_COLORS["light_blue"])
                        st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                feature_imp = results["feature_importance"]
                indices = np.argsort(feature_imp)[-10:]  # Top 10 features
                
                fig = px.bar(
                    x=feature_imp[indices],
                    y=[f"Feature {i}" for i in indices],
                    orientation='h',
                    title='Top 10 Feature Importance',
                    labels={'x': 'Importance', 'y': 'Feature'},
                )
                fig.update_traces(marker_color=AWS_COLORS["orange"])
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Train the model to see the results!")
    else:
        st.warning("Please generate a dataset first using the sidebar options.")

    st.markdown("---")
    
    st.subheader("Boosting Algorithms Compared")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>AdaBoost</h3>
            <p>Adaptive Boosting focuses on misclassified samples by increasing their weights.</p>
            <p><strong>Best for:</strong></p>
            <ul>
            <li>Datasets with moderate dimensions</li>
            <li>Problems where interpretability matters</li>
            <li>Clean datasets with minimal noise</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Gradient Boosting</h3>
            <p>Uses gradient descent optimization to minimize the loss function.</p>
            <p><strong>Best for:</strong></p>
            <ul>
            <li>Regression problems</li>
            <li>Complex relationships in data</li>
            <li>Datasets where you need very high accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>XGBoost / LightGBM</h3>
            <p>Optimized implementations with advanced regularization and efficiency features.</p>
            <p><strong>Best for:</strong></p>
            <ul>
            <li>Production systems that need high performance</li>
            <li>Large-scale machine learning problems</li>
            <li>Kaggle competitions and real-world applications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Tab 4: Stacking
with tabs[3]:
    st.header("Stacking (Stacked Generalization)")
    
    st.markdown("""
    <div class="card">
    <h3>What is Stacking? 🏗️</h3>
    <p>Stacking, or Stacked Generalization, is an ensemble technique that combines multiple classification or 
    regression models via a meta-model. The base models are trained on the original dataset, then 
    a meta-model is trained on the outputs of the base models.</p>
    
    <h4>When to use Stacking:</h4>
    <ul>
    <li>When you have diverse models that perform well on different subsets of data</li>
    <li>When you're looking for the best possible predictive performance</li>
    <li>When you have enough data to train multiple models and a meta-model</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### How Stacking Works

        Stacking works by following these steps:

        1. Split the dataset into training and validation sets
        2. Train multiple base models on the training data
        3. Make predictions on the validation data with each base model
        4. Use these predictions as features for a meta-model
        5. Train the meta-model to optimally combine the base models

        ### Code Example
        ```python
        # Using StackingClassifier
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('svc', SVC(probability=True)),
            ('gb', GradientBoostingClassifier())
        ]

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        stacking_model.fit(X_train, y_train)
        ```
        """)
    
    with col2:
        # Show stacking architecture image
        st.image("https://miro.medium.com/max/1400/1*7WySMBrq9A13Zkq7Q_o3uw.jpeg", caption="Stacking Architecture")
        
        st.markdown("""
        <div class="card">
        <h4>Advantages of Stacking:</h4>
        <ul>
        <li>Leverages strengths of different algorithms</li>
        <li>Often provides better predictions than any single model</li>
        <li>Reduces the risk of selecting the wrong model</li>
        <li>Can capture different aspects of the underlying patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Stacking Demo
    st.subheader("Interactive Stacking Demo")
    
    if st.session_state.X_train is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Model selection
            st.subheader("Select Base Models")
            
            if st.session_state.task_type == 'classification':
                use_rf = st.checkbox("Random Forest", value=True, key="stacking_use_rf")
                use_svm = st.checkbox("Support Vector Machine", value=True, key="stacking_use_svm")
                use_lr = st.checkbox("Logistic Regression", value=True, key="stacking_use_lr")
                use_gb = st.checkbox("Gradient Boosting", value=True, key="stacking_use_gb")
                
                st.subheader("Meta-Model")
                meta_model = st.selectbox(
                    "Choose Meta-Model",
                    ["Logistic Regression", "Random Forest"],
                    key="stacking_meta_model"
                )
            else:
                use_rf = st.checkbox("Random Forest", value=True, key="stacking_use_rf")
                use_svr = st.checkbox("SVR", value=True, key="stacking_use_svr")
                use_lr = st.checkbox("Linear Regression", value=True, key="stacking_use_lr")
                use_gb = st.checkbox("Gradient Boosting", value=True, key="stacking_use_gb")
                
                st.subheader("Meta-Model")
                meta_model = st.selectbox(
                    "Choose Meta-Model",
                    ["Linear Regression", "Random Forest"],
                    key="stacking_meta_model"
                )
            
            cv_folds = st.slider(
                "Cross-Validation Folds", 
                min_value=2, 
                max_value=10, 
                value=5,
                key="stacking_cv_folds"
            )
            
            if st.button("Train Stacking Ensemble", key="train_stacking"):
                with st.spinner("Training stacking ensemble..."):
                    if st.session_state.task_type == 'classification':
                        # Create list of base models
                        estimators = []
                        if use_rf:
                            estimators.append(('rf', RandomForestClassifier(n_estimators=10, random_state=42)))
                        if use_svm:
                            estimators.append(('svm', SVC(probability=True, random_state=42)))
                        if use_lr:
                            estimators.append(('lr', LogisticRegression(max_iter=1000, random_state=42)))
                        if use_gb:
                            estimators.append(('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)))
                        
                        if len(estimators) == 0:
                            st.error("Please select at least one base model!")
                            st.stop()
                        
                        # Create and train individual models for comparison
                        individual_scores = {}
                        for name, model in estimators:
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                            y_pred = model.predict(st.session_state.X_test)
                            score = accuracy_score(st.session_state.y_test, y_pred)
                            individual_scores[name] = score
                        
                        # Create meta-model
                        if meta_model == "Logistic Regression":
                            final_estimator = LogisticRegression(max_iter=1000)
                        else:
                            final_estimator = RandomForestClassifier(n_estimators=100)
                        
                        # Create and train stacking model
                        stacking_model = StackingClassifier(
                            estimators=estimators,
                            final_estimator=final_estimator,
                            cv=cv_folds,
                            stack_method='predict_proba',
                            n_jobs=-1,
                            passthrough=True
                        )
                        
                        stacking_model.fit(st.session_state.X_train, st.session_state.y_train)
                        stacking_pred = stacking_model.predict(st.session_state.X_test)
                        stacking_score = accuracy_score(st.session_state.y_test, stacking_pred)
                        
                        # Store results
                        st.session_state.model_results["stacking"] = {
                            "individual_scores": individual_scores,
                            "stacking_score": stacking_score,
                            "conf_matrix": confusion_matrix(st.session_state.y_test, stacking_pred),
                            "classification_report": classification_report(st.session_state.y_test, stacking_pred, output_dict=True)
                        }
                        
                    else:  # regression
                        # Create list of base models
                        estimators = []
                        if use_rf:
                            estimators.append(('rf', RandomForestRegressor(n_estimators=100, random_state=42)))
                        if use_svr:
                            estimators.append(('svr', SVR()))
                        if use_lr:
                            estimators.append(('lr', LinearRegression()))
                        if use_gb:
                            estimators.append(('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)))
                        
                        if len(estimators) == 0:
                            st.error("Please select at least one base model!")
                            st.stop()
                        
                        # Create and train individual models for comparison
                        individual_scores_mse = {}
                        individual_scores_r2 = {}
                        for name, model in estimators:
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                            y_pred = model.predict(st.session_state.X_test)
                            mse = mean_squared_error(st.session_state.y_test, y_pred)
                            r2 = r2_score(st.session_state.y_test, y_pred)
                            individual_scores_mse[name] = mse
                            individual_scores_r2[name] = r2
                        
                        # Create meta-model
                        if meta_model == "Linear Regression":
                            final_estimator = LinearRegression()
                        else:
                            final_estimator = RandomForestRegressor(n_estimators=100)
                        
                        # Create and train stacking model
                        stacking_model = StackingRegressor(
                            estimators=estimators,
                            final_estimator=final_estimator,
                            cv=cv_folds,
                            n_jobs=-1
                        )
                        
                        stacking_model.fit(st.session_state.X_train, st.session_state.y_train)
                        stacking_pred = stacking_model.predict(st.session_state.X_test)
                        stacking_mse = mean_squared_error(st.session_state.y_test, stacking_pred)
                        stacking_r2 = r2_score(st.session_state.y_test, stacking_pred)
                        
                        # Store results
                        st.session_state.model_results["stacking"] = {
                            "individual_scores_mse": individual_scores_mse,
                            "individual_scores_r2": individual_scores_r2,
                            "stacking_mse": stacking_mse,
                            "stacking_r2": stacking_r2
                        }
                
                st.success("✅ Stacking ensemble trained successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if "stacking" in st.session_state.model_results:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                results = st.session_state.model_results["stacking"]
                
                if st.session_state.task_type == 'classification':
                    # Add stacking to individual scores for comparison
                    all_scores = results["individual_scores"].copy()
                    all_scores["stacking"] = results["stacking_score"]
                    
                    # Create bar chart for model comparison
                    models = list(all_scores.keys())
                    scores = list(all_scores.values())
                    
                    # Highlight stacking model with different color
                    colors = [AWS_COLORS["light_blue"] if model != "stacking" else AWS_COLORS["green"] for model in models]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=models,
                        y=[score * 100 for score in scores],
                        marker_color=colors,
                        text=[f'{score:.2%}' for score in scores],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title='Model Accuracy Comparison',
                        xaxis_title='Model',
                        yaxis_title='Accuracy (%)',
                        yaxis=dict(range=[0, 100]),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix (Stacking Model)")
                    cm = results["conf_matrix"]
                    fig = px.imshow(
                        cm, 
                        text_auto=True,
                        color_continuous_scale=px.colors.sequential.Greens,
                        labels=dict(x="Predicted Label", y="True Label"),
                        x=['Class 0', 'Class 1'],
                        y=['Class 0', 'Class 1']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # regression
                    # Add stacking to individual scores for comparison
                    all_scores_mse = results["individual_scores_mse"].copy()
                    all_scores_mse["stacking"] = results["stacking_mse"]
                    
                    all_scores_r2 = results["individual_scores_r2"].copy()
                    all_scores_r2["stacking"] = results["stacking_r2"]
                    
                    # Create bar charts for model comparison
                    models = list(all_scores_mse.keys())
                    mse_scores = list(all_scores_mse.values())
                    r2_scores = list(all_scores_r2.values())
                    
                    # Highlight stacking model with different color
                    colors = [AWS_COLORS["light_blue"] if model != "stacking" else AWS_COLORS["green"] for model in models]
                    
                    # MSE comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=models,
                        y=mse_scores,
                        marker_color=colors,
                        text=[f'{score:.4f}' for score in mse_scores],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title='Mean Squared Error Comparison (Lower is better)',
                        xaxis_title='Model',
                        yaxis_title='MSE',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # R² comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=models,
                        y=r2_scores,
                        marker_color=colors,
                        text=[f'{score:.4f}' for score in r2_scores],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title='R² Score Comparison (Higher is better)',
                        xaxis_title='Model',
                        yaxis_title='R²',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Insights section
                st.markdown("""
                <div class="card">
                    <h3>Key Insights:</h3>
                    <ul>
                        <li>Stacking often outperforms individual models by leveraging their strengths</li>
                        <li>The meta-model learns which base model performs best in different situations</li>
                        <li>Diverse base models typically lead to better stacking performance</li>
                        <li>Cross-validation prevents overfitting in the stacking process</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Train the ensemble to see the results!")
    else:
        st.warning("Please generate a dataset first using the sidebar options.")

    st.markdown("---")
    
    st.subheader("Stacking in Practice")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>When to Use Stacking</h3>
            <ul>
            <li>When you have models with complementary strengths</li>
            <li>For critical applications where accuracy is paramount</li>
            <li>In competitions like Kaggle, where small improvements matter</li>
            <li>When you have sufficient data to train multiple models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Real-World Applications</h3>
            <ul>
            <li><strong>Healthcare:</strong> Combining multiple diagnostic models for better disease prediction</li>
            <li><strong>Finance:</strong> Credit default prediction with multiple risk assessments</li>
            <li><strong>Natural Language Processing:</strong> Ensemble of different text classifiers</li>
            <li><strong>Computer Vision:</strong> Combining different object detection approaches</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>© 2023 Ensemble Learning Explorer | Created with Streamlit</p>
    <p><small>For educational purposes only</small></p>
</div>
""", unsafe_allow_html=True)
