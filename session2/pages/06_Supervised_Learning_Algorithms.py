
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, r2_score
import altair as alt
import time

# Set page configuration
st.set_page_config(
    page_title="SageMaker Algorithms Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS Color Scheme
AWS_COLORS = {
    'orange': '#FF9900',
    'blue': '#232F3E',
    'light_blue': '#1A73E8',
    'grey': '#545B64',
    'light_grey': '#D5DBDB',
    'white': '#FFFFFF',
    'green': '#008296',
    'red': '#D13212'
}

# Custom CSS
st.markdown("""
<style>
    /* General Styling */
    .main {
        background-color: #FFFFFF;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #D5DBDB;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF9900;
        color: #232F3E;
    }
    
    /* Cards */
    .card {
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #F7F7F7;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #FF9900;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #E88B00;
    }
    
    /* Metrics */
    .metric-container {
        background-color: #232F3E;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #232F3E;
    }
    
    h1 {
        font-weight: bold;
        border-bottom: 2px solid #FF9900;
        padding-bottom: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #FF9900;
    }
</style>
""", unsafe_allow_html=True)


# Helper function for sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize session state
def init_session_state():
    if 'initialized_sup' not in st.session_state:
        st.session_state.initialized_sup = True
        st.session_state.linear_learner_trained = False
        st.session_state.xgboost_trained = False
        st.session_state.knn_trained = False
        st.session_state.fm_trained = False
        
        # Linear Learner
        st.session_state.ll_X_train = None
        st.session_state.ll_X_test = None
        st.session_state.ll_y_train = None
        st.session_state.ll_y_test = None
        st.session_state.ll_predictions = None
        st.session_state.ll_mse = None
        st.session_state.ll_r2 = None
        st.session_state.ll_data_type = 'regression'
        
        # XGBoost
        st.session_state.xgb_X_train = None
        st.session_state.xgb_X_test = None
        st.session_state.xgb_y_train = None
        st.session_state.xgb_y_test = None
        st.session_state.xgb_predictions = None
        st.session_state.xgb_accuracy = None
        st.session_state.xgb_cm = None
        
        # KNN
        st.session_state.knn_X_train = None
        st.session_state.knn_X_test = None
        st.session_state.knn_y_train = None
        st.session_state.knn_y_test = None
        st.session_state.knn_predictions = None
        st.session_state.knn_accuracy = None
        st.session_state.knn_cm = None
        
        # Factorization Machines
        st.session_state.fm_X_train = None
        st.session_state.fm_X_test = None
        st.session_state.fm_y_train = None
        st.session_state.fm_y_test = None
        st.session_state.fm_predictions = None
        st.session_state.fm_accuracy = None

# Initialize session state
init_session_state()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg", width=100)
st.sidebar.title("Session Management")

if st.sidebar.button("Reset Session", key="reset_session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    st.sidebar.success("Session has been reset!")

st.sidebar.markdown("---")
st.sidebar.markdown("""
## About This App
This interactive application demonstrates Amazon SageMaker's built-in algorithms:
- Linear Learner
- XGBoost
- K-Nearest Neighbors
- Factorization Machines

Explore each algorithm with interactive visualizations and examples.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Resources")
st.sidebar.markdown("[SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)")
st.sidebar.markdown("[AWS Machine Learning](https://aws.amazon.com/machine-learning/)")

# Main content
st.title("Amazon SageMaker Algorithms Explorer")
st.markdown("""
This interactive application helps you understand the main built-in algorithms available in Amazon SageMaker.
Choose an algorithm tab below to explore its features, use cases, and see it in action!
""")

# Create tabs with emoji
tabs = st.tabs([
    "📈 Linear Learner", 
    "🌲 XGBoost", 
    "🎯 K-Nearest Neighbors", 
    "🔢 Factorization Machines"
])

# Linear Learner Tab
with tabs[0]:
    st.header("📈 Linear Learner Algorithm")
    
    st.markdown("""
    <div class="card">
    <h3>Overview</h3>
    <p>Linear Learner is a versatile algorithm for both classification and regression problems. It trains many models in parallel and automatically tunes them to find the most accurate model.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Supports both binary/multi-class classification and regression</li>
        <li>Built-in L1 and L2 regularization</li>
        <li>Automatic hyperparameter optimization</li>
        <li>Efficiently scales to large datasets</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>Linear relationships between features and target variable</li>
        <li>Low to medium feature dimensionality problems</li>
        <li>When interpretability is important</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive Demo")
    
    # Data setup for Linear Learner
    ll_col1, ll_col2 = st.columns([1, 1])
    
    with ll_col1:
        ll_problem_type = st.selectbox("Problem Type", ["Regression", "Classification"], key="ll_problem_type")
        ll_n_samples = st.slider("Number of Samples", 100, 1000, 500, 50, key="ll_n_samples")
        ll_n_features = st.slider("Number of Features", 2, 20, 5, 1, key="ll_n_features")
        ll_noise = st.slider("Noise Level", 0.0, 1.0, 0.2, 0.05, key="ll_noise")
        
        if st.button("Generate Data", key="ll_generate"):
            with st.spinner("Generating data..."):
                if ll_problem_type == "Regression":
                    X, y = make_regression(
                        n_samples=ll_n_samples,
                        n_features=ll_n_features,
                        noise=ll_noise,
                        random_state=42
                    )
                    st.session_state.ll_data_type = 'regression'
                else:
                    X, y = make_classification(
                        n_samples=ll_n_samples,
                        n_features=ll_n_features,
                        n_informative=ll_n_features-2,
                        n_redundant=1,
                        random_state=42,
                        class_sep=1.5
                    )
                    st.session_state.ll_data_type = 'classification'
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                st.session_state.ll_X_train = X_train
                st.session_state.ll_X_test = X_test
                st.session_state.ll_y_train = y_train
                st.session_state.ll_y_test = y_test
                st.session_state.linear_learner_trained = False
                st.success("Data generated successfully!")

    with ll_col2:
        if st.session_state.ll_X_train is not None:
            # Plot data distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            
            if ll_problem_type == "Regression":
                ax.scatter(st.session_state.ll_X_train[:, 0], st.session_state.ll_y_train, alpha=0.5)
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Target Value")
                ax.set_title("Sample of Training Data")
            else:
                for i in np.unique(st.session_state.ll_y_train):
                    ix = np.where(st.session_state.ll_y_train == i)
                    ax.scatter(st.session_state.ll_X_train[ix, 0], st.session_state.ll_X_train[ix, 1], label=f"Class {i}", alpha=0.6)
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title("Sample of Training Data")
                ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Training parameters
            st.subheader("Model Configuration")
            ll_learning_rate = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.1, 0.2, 0.5], value=0.1, key="ll_lr")
            ll_l1 = st.select_slider("L1 Regularization", options=[0.0, 0.001, 0.01, 0.1, 0.5], value=0.01, key="ll_l1")
            
            if st.button("Train Model", key="ll_train"):
                with st.spinner("Training Linear Learner model..."):
                    # Simulate training
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    if st.session_state.ll_data_type == 'regression':
                        # Simple linear regression simulation
                        from sklearn.linear_model import ElasticNet
                        model = ElasticNet(alpha=ll_l1, l1_ratio=0.5, random_state=42)
                        model.fit(st.session_state.ll_X_train, st.session_state.ll_y_train)
                        
                        # Make predictions
                        predictions = model.predict(st.session_state.ll_X_test)
                        mse = mean_squared_error(st.session_state.ll_y_test, predictions)
                        r2 = r2_score(st.session_state.ll_y_test, predictions)
                        
                        st.session_state.ll_predictions = predictions
                        st.session_state.ll_mse = mse
                        st.session_state.ll_r2 = r2
                    else:
                        # Simple logistic regression simulation
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(C=1/ll_l1 if ll_l1 > 0 else 1000, random_state=42)
                        model.fit(st.session_state.ll_X_train, st.session_state.ll_y_train)
                        
                        # Make predictions
                        predictions = model.predict(st.session_state.ll_X_test)
                        accuracy = accuracy_score(st.session_state.ll_y_test, predictions)
                        cm = confusion_matrix(st.session_state.ll_y_test, predictions)
                        
                        st.session_state.ll_predictions = predictions
                        st.session_state.ll_accuracy = accuracy
                        st.session_state.ll_cm = cm
                    
                    st.session_state.linear_learner_trained = True
                    st.success("Model trained successfully!")

    if st.session_state.linear_learner_trained:
        st.header("Model Results")
        result_col1, result_col2 = st.columns([1, 1])
        
        if st.session_state.ll_data_type == 'regression':
            with result_col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Mean Squared Error", f"{st.session_state.ll_mse:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("R² Score", f"{st.session_state.ll_r2:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col2:
                # Plot predictions vs actual
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(st.session_state.ll_y_test, st.session_state.ll_predictions, alpha=0.5)
                ax.plot([st.session_state.ll_y_test.min(), st.session_state.ll_y_test.max()], 
                        [st.session_state.ll_y_test.min(), st.session_state.ll_y_test.max()], 
                        'r--', lw=2)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Predictions vs Actual Values")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            with result_col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{st.session_state.ll_accuracy:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col2:
                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(st.session_state.ll_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("""
        <div class="card">
        <h3>SageMaker Implementation</h3>
        <p>In Amazon SageMaker, the Linear Learner algorithm would be implemented as follows:</p>
        <pre>
        from sagemaker.amazon.linear_learner import LinearLearner
        
        # Configure the algorithm
        linear_learner = LinearLearner(
            role='SageMakerRole',
            instance_count=1,
            instance_type='ml.m4.xlarge',
            predictor_type='regressor' if problem_type == 'Regression' else 'binary_classifier',
            learning_rate=0.1,
            l1=0.01,
            mini_batch_size=100
        )
        
        # Train the model
        linear_learner.fit({
            'train': train_data,
            'validation': validation_data
        })
        
        # Deploy the model
        predictor = linear_learner.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge'
        )
        </pre>
        </div>
        """, unsafe_allow_html=True)

# XGBoost Tab
with tabs[1]:
    st.header("🌲 XGBoost Algorithm")
    
    st.markdown("""
    <div class="card">
    <h3>Overview</h3>
    <p>XGBoost is a powerful gradient boosting algorithm that excels in structured/tabular data. It's one of the most popular algorithms for Kaggle competitions and real-world applications.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Gradient boosting framework with decision trees</li>
        <li>Handles missing values automatically</li>
        <li>Built-in regularization to prevent overfitting</li>
        <li>Supports parallel processing</li>
        <li>Handles both classification and regression problems</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>Working with structured/tabular data</li>
        <li>When you need high predictive accuracy</li>
        <li>When you have both categorical and numerical features</li>
        <li>When interpretability is less important than performance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive Demo")
    
    # Data setup for XGBoost
    xgb_col1, xgb_col2 = st.columns([1, 1])
    
    with xgb_col1:
        xgb_n_samples = st.slider("Number of Samples", 100, 1000, 500, 50, key="xgb_n_samples")
        xgb_n_features = st.slider("Number of Features", 2, 20, 10, 1, key="xgb_n_features")
        xgb_n_informative = st.slider("Number of Informative Features", 2, 10, 5, 1, key="xgb_n_informative")
        xgb_n_classes = st.slider("Number of Classes", 2, 4, 2, 1, key="xgb_n_classes_key")
        
        if st.button("Generate Data", key="xgb_generate"):
            with st.spinner("Generating data..."):
                X, y = make_classification(
                    n_samples=xgb_n_samples,
                    n_features=xgb_n_features,
                    n_informative=xgb_n_informative,
                    n_redundant=2,
                    n_classes=xgb_n_classes,
                    random_state=42,
                    class_sep=2.0
                )
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                st.session_state.xgb_X_train = X_train
                st.session_state.xgb_X_test = X_test
                st.session_state.xgb_y_train = y_train
                st.session_state.xgb_y_test = y_test
                st.session_state.xgb_n_classes = xgb_n_classes
                st.session_state.xgboost_trained = False
                st.success("Data generated successfully!")

    with xgb_col2:
        if st.session_state.xgb_X_train is not None:
            # Plot data distribution for first two features
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for i in np.unique(st.session_state.xgb_y_train):
                ix = np.where(st.session_state.xgb_y_train == i)
                ax.scatter(st.session_state.xgb_X_train[ix, 0], st.session_state.xgb_X_train[ix, 1], 
                           label=f"Class {i}", alpha=0.6)
            
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_title("Sample of Training Data (First 2 Features)")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            # Training parameters
            st.subheader("Model Configuration")
            xgb_max_depth = st.slider("Max Tree Depth", 3, 10, 6, 1, key="xgb_max_depth")
            xgb_n_estimators = st.slider("Number of Trees", 50, 300, 100, 10, key="xgb_n_estimators")
            xgb_learning_rate = st.select_slider("Learning Rate", 
                                                options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
                                                value=0.1, key="xgb_lr")
            
            if st.button("Train Model", key="xgb_train"):
                with st.spinner("Training XGBoost model..."):
                    # Simulate training
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.03)
                        progress_bar.progress(i + 1)
                    
                    # Train XGBoost model
                    from sklearn.ensemble import GradientBoostingClassifier
                    
                    model = GradientBoostingClassifier(
                        n_estimators=xgb_n_estimators,
                        learning_rate=xgb_learning_rate,
                        max_depth=xgb_max_depth,
                        random_state=42
                    )
                    
                    model.fit(st.session_state.xgb_X_train, st.session_state.xgb_y_train)
                    
                    # Make predictions
                    predictions = model.predict(st.session_state.xgb_X_test)
                    accuracy = accuracy_score(st.session_state.xgb_y_test, predictions)
                    cm = confusion_matrix(st.session_state.xgb_y_test, predictions)
                    
                    st.session_state.xgb_predictions = predictions
                    st.session_state.xgb_accuracy = accuracy
                    st.session_state.xgb_cm = cm
                    st.session_state.xgb_feature_importance = model.feature_importances_
                    
                    st.session_state.xgboost_trained = True
                    st.success("Model trained successfully!")

    if st.session_state.xgboost_trained:
        st.header("Model Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{st.session_state.xgb_accuracy:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(st.session_state.xgb_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            st.pyplot(fig)
        
        with result_col2:
            # Plot feature importances
            if hasattr(st.session_state, 'xgb_feature_importance'):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get feature importances and sort
                feature_importance = st.session_state.xgb_feature_importance
                indices = np.argsort(feature_importance)[-10:]  # Top 10 features
                
                ax.barh(range(len(indices)), feature_importance[indices], align='center', color=AWS_COLORS['orange'])
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([f"Feature {i}" for i in indices])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top Features by Importance')
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("""
        <div class="card">
        <h3>SageMaker Implementation</h3>
        <p>In Amazon SageMaker, XGBoost would be implemented as follows:</p>
        <pre>
        import sagemaker
        from sagemaker.xgboost.estimator import XGBoost
        
        # Configure the algorithm
        xgboost = XGBoost(
            role='SageMakerRole',
            instance_count=1,
            instance_type='ml.m4.xlarge',
            framework_version='1.5-1',
            max_depth=6,
            eta=0.1,
            objective='multi:softmax',
            num_class=3,
            num_round=100
        )
        
        # Train the model
        xgboost.fit({
            'train': train_data_channel,
            'validation': validation_data_channel
        })
        
        # Deploy the model
        predictor = xgboost.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge'
        )
        </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Decision Boundaries Visualization")
        
        # Visualize decision boundaries for the first two features
        if st.session_state.xgb_X_train is not None and st.session_state.xgb_n_classes <= 4:
            from sklearn.ensemble import GradientBoostingClassifier
            
            # Create a simplified model for visualization (using only first 2 features)
            X_2d_train = st.session_state.xgb_X_train[:, :2]
            X_2d_test = st.session_state.xgb_X_test[:, :2]
            
            model_2d = GradientBoostingClassifier(
                n_estimators=xgb_n_estimators,
                learning_rate=xgb_learning_rate,
                max_depth=xgb_max_depth,
                random_state=42
            )
            
            model_2d.fit(X_2d_train, st.session_state.xgb_y_train)
            
            # Create meshgrid for decision boundary
            x_min, x_max = X_2d_train[:, 0].min() - 1, X_2d_train[:, 0].max() + 1
            y_min, y_max = X_2d_train[:, 1].min() - 1, X_2d_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                  np.arange(y_min, y_max, 0.1))
            
            # Predict class for each point in meshgrid
            Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot decision boundaries
            cmap = plt.cm.tab10
            contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
            
            # Plot training points
            for i in np.unique(st.session_state.xgb_y_train):
                idx = np.where(st.session_state.xgb_y_train == i)
                ax.scatter(X_2d_train[idx, 0], X_2d_train[idx, 1], 
                        label=f"Class {i}", s=20, edgecolors='k', cmap=cmap)
            
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_title("Decision Boundaries (2D Projection)")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

# K-Nearest Neighbors Tab
with tabs[2]:
    st.header("🎯 K-Nearest Neighbors (KNN) Algorithm")
    
    st.markdown("""
    <div class="card">
    <h3>Overview</h3>
    <p>K-Nearest Neighbors is a simple yet effective non-parametric algorithm used for classification and regression. It makes predictions based on the k-closest training examples in the feature space.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Instance-based learning (lazy learning) - no explicit training phase</li>
        <li>Simple and intuitive algorithm</li>
        <li>Works well with small to medium datasets</li>
        <li>Can be used for both classification and regression</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>When the decision boundary is irregular</li>
        <li>When you have a small to medium-sized dataset</li>
        <li>When you need interpretable results</li>
        <li>When you can afford the computational cost at prediction time</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive Demo")
    
    # Data setup for KNN
    knn_col1, knn_col2 = st.columns([1, 1])
    
    with knn_col1:
        knn_n_samples = st.slider("Number of Samples", 100, 1000, 300, 50, key="knn_n_samples")
        knn_n_features = st.slider("Number of Features", 2, 10, 2, 1, key="knn_n_features")
        knn_n_classes = st.slider("Number of Classes", 2, 5, 3, 1, key="knn_n_classes")
        
        if st.button("Generate Data", key="knn_generate"):
            with st.spinner("Generating data..."):
                X, y = make_classification(
                    n_samples=knn_n_samples,
                    n_features=knn_n_features,
                    n_informative=knn_n_features,
                    n_redundant=0,
                    n_classes=knn_n_classes,
                    n_clusters_per_class=1,
                    random_state=42,
                    class_sep=1.5
                )
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                st.session_state.knn_X_train = X_train
                st.session_state.knn_X_test = X_test
                st.session_state.knn_y_train = y_train
                st.session_state.knn_y_test = y_test
                st.session_state.knn_trained = False
                st.success("Data generated successfully!")

    with knn_col2:
        if st.session_state.knn_X_train is not None:
            # Plot data distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            
            if knn_n_features >= 2:
                for i in np.unique(st.session_state.knn_y_train):
                    ix = np.where(st.session_state.knn_y_train == i)
                    ax.scatter(st.session_state.knn_X_train[ix, 0], st.session_state.knn_X_train[ix, 1], 
                               label=f"Class {i}", alpha=0.6)
                
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title("Sample of Training Data (First 2 Features)")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            
            # Training parameters
            st.subheader("Model Configuration")
            knn_k = st.slider("Number of Neighbors (k)", 1, 20, 5, 1, key="knn_k")
            knn_weights = st.selectbox("Weight Function", ["uniform", "distance"], key="knn_weights")
            knn_p = st.selectbox("Distance Metric", [1, 2], 
                               format_func=lambda x: "Manhattan (L1)" if x == 1 else "Euclidean (L2)",
                               key="knn_p")
            
            if st.button("Train Model", key="knn_train"):
                with st.spinner("Training KNN model..."):
                    # Simulate training
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    # Train KNN model
                    from sklearn.neighbors import KNeighborsClassifier
                    
                    model = KNeighborsClassifier(
                        n_neighbors=knn_k,
                        weights=knn_weights,
                        p=knn_p
                    )
                    
                    model.fit(st.session_state.knn_X_train, st.session_state.knn_y_train)
                    
                    # Make predictions
                    predictions = model.predict(st.session_state.knn_X_test)
                    accuracy = accuracy_score(st.session_state.knn_y_test, predictions)
                    cm = confusion_matrix(st.session_state.knn_y_test, predictions)
                    
                    st.session_state.knn_predictions = predictions
                    st.session_state.knn_accuracy = accuracy
                    st.session_state.knn_cm = cm
                    st.session_state.knn_k_value = knn_k
                    st.session_state.knn_trained = True
                    st.success("Model trained successfully!")

    if st.session_state.knn_trained and st.session_state.knn_X_train is not None:
        st.header("Model Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{st.session_state.knn_accuracy:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(st.session_state.knn_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            st.pyplot(fig)
        
        with result_col2:
            # Visualize decision boundaries
            if knn_n_features >= 2:
                from sklearn.neighbors import KNeighborsClassifier
                
                # Create model for visualization
                model = KNeighborsClassifier(
                    n_neighbors=st.session_state.knn_k_value,
                    weights=knn_weights,
                    p=knn_p
                )
                
                model.fit(st.session_state.knn_X_train[:, :2], st.session_state.knn_y_train)
                
                # Create meshgrid for decision boundary
                x_min, x_max = st.session_state.knn_X_train[:, 0].min() - 1, st.session_state.knn_X_train[:, 0].max() + 1
                y_min, y_max = st.session_state.knn_X_train[:, 1].min() - 1, st.session_state.knn_X_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                      np.arange(y_min, y_max, 0.1))
                
                # Predict class for each point in meshgrid
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot decision boundaries
                cmap = plt.cm.tab10
                contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
                
                # Plot training points
                for i in np.unique(st.session_state.knn_y_train):
                    idx = np.where(st.session_state.knn_y_train == i)
                    ax.scatter(st.session_state.knn_X_train[idx, 0], st.session_state.knn_X_train[idx, 1], 
                            label=f"Class {i}", s=20, edgecolors='k')
                
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title(f"KNN Decision Boundaries with k={st.session_state.knn_k_value}")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
        
        st.subheader("K-Value Analysis")
        
        # Effect of k-value on accuracy
        k_range = range(1, 21)
        k_accuracies = []
        
        with st.spinner("Analyzing k-values..."):
            for k in k_range:
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(st.session_state.knn_X_train, st.session_state.knn_y_train)
                score = model.score(st.session_state.knn_X_test, st.session_state.knn_y_test)
                k_accuracies.append(score)
            
            # Plot k vs. accuracy
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_range, k_accuracies, 'o-', color=AWS_COLORS['orange'], linewidth=2, markersize=8)
            ax.set_xlabel('Number of Neighbors (k)')
            ax.set_ylabel('Testing Accuracy')
            ax.set_title('Accuracy vs. k-Value')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        <div class="card">
        <h3>SageMaker Implementation</h3>
        <p>In Amazon SageMaker, the K-Nearest Neighbors algorithm would be implemented as follows:</p>
        <pre>
        from sagemaker import KNN
        
        # Configure the algorithm
        knn = KNN(
            role='SageMakerRole',
            instance_count=1,
            instance_type='ml.m4.xlarge',
            k=5,
            sample_size=500,
            predictor_type='classifier',
            dimension_reduction_type='sign',
            dimension_reduction_target=50
        )
        
        # Train the model
        knn.fit({
            'train': train_data_channel,
            'test': test_data_channel
        })
        
        # Deploy the model
        predictor = knn.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge'
        )
        </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <h3>How KNN Works</h3>
        <p>K-Nearest Neighbors algorithm works as follows:</p>
        <ol>
            <li>Calculate the distance between the query point and all training samples</li>
            <li>Sort the distances and determine the k-nearest neighbors</li>
            <li>For classification: Use majority vote of the k neighbors to assign a class to the query point</li>
            <li>For regression: Take the average value of the k neighbors as the predicted value</li>
        </ol>
        <p>The value of k and the distance metric are important hyperparameters that affect model performance.</p>
        </div>
        """, unsafe_allow_html=True)

# Factorization Machines Tab
with tabs[3]:
    st.header("🔢 Factorization Machines Algorithm")
    
    st.markdown("""
    <div class="card">
    <h3>Overview</h3>
    <p>Factorization Machines (FM) are designed to capture interactions between features in sparse datasets. They are particularly useful for recommendation systems and click prediction tasks.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Efficiently models feature interactions in high-dimensional sparse data</li>
        <li>Combines the advantages of Support Vector Machines and factorization models</li>
        <li>Works well for recommendation systems and click-through rate prediction</li>
        <li>Can handle binary classification and regression tasks</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>For recommendation systems</li>
        <li>For click-through rate prediction</li>
        <li>When dealing with sparse, high-dimensional data</li>
        <li>When feature interactions are important for predictions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive Demo")
    
    # Since factorization machines are best for recommendation systems, let's create a movie recommendation demo
    st.markdown("""
    <div class="card">
    <h3>Movie Recommendation System</h3>
    <p>This demo simulates a movie recommendation system using Factorization Machines. We'll generate synthetic user-movie interaction data and predict user ratings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    fm_col1, fm_col2 = st.columns([1, 1])
    
    with fm_col1:
        fm_n_users = st.slider("Number of Users", 20, 100, 50, 5, key="fm_n_users_key")
        fm_n_movies = st.slider("Number of Movies", 20, 100, 40, 5, key="fm_n_movies_key")
        fm_sparsity = st.slider("Data Sparsity (% of missing ratings)", 30, 90, 70, 5, key="fm_sparsity_key")
        
        if st.button("Generate Data", key="fm_generate"):
            with st.spinner("Generating recommendation data..."):
                # Create user-movie rating matrix with some missing values
                n_ratings = int(fm_n_users * fm_n_movies * (1 - fm_sparsity/100))
                
                # Random user and movie indices
                user_indices = np.random.randint(0, fm_n_users, n_ratings)
                movie_indices = np.random.randint(0, fm_n_movies, n_ratings)
                
                # Generate some user and movie features (for demonstration)
                user_features = np.random.randn(fm_n_users, 5)  # 5 user features
                movie_features = np.random.randn(fm_n_movies, 8)  # 8 movie features
                
                # Generate ratings with some underlying pattern
                ratings = np.zeros(n_ratings)
                for i in range(n_ratings):
                    user_idx = user_indices[i]
                    movie_idx = movie_indices[i]
                    
                    # Base rating is dot product of some features
                    base_rating = np.dot(user_features[user_idx, :2], movie_features[movie_idx, :2])
                    
                    # Normalize to 1-5 rating scale and add noise
                    ratings[i] = 1 + 4 * (sigmoid(base_rating) + 0.2 * np.random.randn())
                    
                # Create feature matrix X: [user_id, movie_id, user_features, movie_features]
                X = np.zeros((n_ratings, 2 + user_features.shape[1] + movie_features.shape[1]))
                X[:, 0] = user_indices
                X[:, 1] = movie_indices
                
                for i in range(n_ratings):
                    X[i, 2:2+user_features.shape[1]] = user_features[user_indices[i]]
                    X[i, 2+user_features.shape[1]:] = movie_features[movie_indices[i]]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, ratings, test_size=0.2, random_state=42)
                
                # Store in session state
                st.session_state.fm_X_train = X_train
                st.session_state.fm_X_test = X_test
                st.session_state.fm_y_train = y_train
                st.session_state.fm_y_test = y_test
                st.session_state.fm_n_users = fm_n_users
                st.session_state.fm_n_movies = fm_n_movies
                st.session_state.fm_user_indices = user_indices
                st.session_state.fm_movie_indices = movie_indices
                st.session_state.fm_ratings = ratings
                st.session_state.fm_trained = False
                st.success("Data generated successfully!")

    with fm_col2:
        if hasattr(st.session_state, 'fm_ratings'):
            # Visualize ratings distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(st.session_state.fm_ratings, bins=10, kde=True, color=AWS_COLORS['orange'], ax=ax)
            ax.set_xlabel('Rating')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Movie Ratings')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create a sparse matrix visualization
            st.subheader("User-Movie Interaction Matrix")
            
            # Create a sparse matrix for visualization
            sparse_matrix = np.zeros((min(20, st.session_state.fm_n_users), min(20, st.session_state.fm_n_movies)))
            sparse_matrix.fill(np.nan)  # Fill with NaN to represent missing ratings
            
            for i in range(len(st.session_state.fm_user_indices)):
                if st.session_state.fm_user_indices[i] < 20 and st.session_state.fm_movie_indices[i] < 20:
                    sparse_matrix[st.session_state.fm_user_indices[i], st.session_state.fm_movie_indices[i]] = st.session_state.fm_ratings[i]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            mask = np.isnan(sparse_matrix)
            sns.heatmap(sparse_matrix, mask=mask, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Rating'})
            ax.set_xlabel('Movies')
            ax.set_ylabel('Users')
            ax.set_title('Sparse User-Movie Rating Matrix (Top 20x20 sample)')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Training parameters
            st.subheader("Model Configuration")
            fm_n_factors = st.slider("Number of Latent Factors", 5, 50, 20, 5, key="fm_n_factors")
            fm_iterations = st.slider("Number of Iterations", 20, 200, 100, 10, key="fm_iterations")
            
            if st.button("Train Model", key="fm_train"):
                with st.spinner("Training Factorization Machine model..."):
                    # Simulate training
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.03)
                        progress_bar.progress(i + 1)
                    
                    # Since FM is not directly available in scikit-learn, we'll simulate results
                    # In real SageMaker, you would use the FM algorithm
                    
                    # Simulate predictions - just for demonstration
                    from sklearn.linear_model import Ridge
                    
                    # Use Ridge as a simple substitute (not exactly FM but for demonstration)
                    model = Ridge(alpha=1.0)
                    model.fit(st.session_state.fm_X_train, st.session_state.fm_y_train)
                    
                    # Make predictions
                    predictions = model.predict(st.session_state.fm_X_test)
                    # Clip predictions to rating range
                    predictions = np.clip(predictions, 1, 5)
                    
                    mse = mean_squared_error(st.session_state.fm_y_test, predictions)
                    rmse = np.sqrt(mse)
                    
                    st.session_state.fm_predictions = predictions
                    st.session_state.fm_mse = mse
                    st.session_state.fm_rmse = rmse
                    st.session_state.fm_trained = True
                    st.success("Model trained successfully!")

    if st.session_state.fm_trained:
        st.header("Model Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Root Mean Squared Error (RMSE)", f"{st.session_state.fm_rmse:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Plot predicted vs actual ratings
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(st.session_state.fm_y_test, st.session_state.fm_predictions, alpha=0.5, color=AWS_COLORS['orange'])
            ax.plot([1, 5], [1, 5], 'r--')
            ax.set_xlabel('Actual Rating')
            ax.set_ylabel('Predicted Rating')
            ax.set_title('Predicted vs Actual Ratings')
            ax.set_xlim(1, 5)
            ax.set_ylim(1, 5)
            plt.tight_layout()
            st.pyplot(fig)
        
        with result_col2:
            # Error distribution
            errors = st.session_state.fm_predictions - st.session_state.fm_y_test
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(errors, kde=True, color=AWS_COLORS['green'], ax=ax)
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Prediction Errors')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display error statistics
            st.markdown("""
            <div class="card">
            <h3>Error Statistics</h3>
            <ul>
                <li>Mean Error: {:.4f}</li>
                <li>Mean Absolute Error: {:.4f}</li>
                <li>Mean Squared Error: {:.4f}</li>
                <li>Root Mean Squared Error: {:.4f}</li>
            </ul>
            </div>
            """.format(
                np.mean(errors),
                np.mean(np.abs(errors)),
                st.session_state.fm_mse,
                st.session_state.fm_rmse
            ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <h3>SageMaker Implementation</h3>
        <p>In Amazon SageMaker, Factorization Machines would be implemented as follows:</p>
        <pre>
        from sagemaker.amazon.factorization_machines import FactorizationMachines
        
        # Configure the algorithm
        fm = FactorizationMachines(
            role='SageMakerRole',
            instance_count=1,
            instance_type='ml.m4.xlarge',
            num_factors=20,
            predictor_type='regressor',
            epochs=100,
            mini_batch_size=1000
        )
        
        # Train the model
        fm.fit({
            'train': train_channel,
            'test': test_channel
        })
        
        # Deploy the model
        predictor = fm.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge'
        )
        </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Recommendation Example")
        
        # Create a simple recommendation demo
        st.markdown("""
        <div class="card">
        <h3>Sample Movie Recommendations</h3>
        <p>Below are sample movie recommendations for selected users:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a sample user-movie rating table
        sample_data = {
            'User ID': list(range(5)),
            'Movie 1': [4.2, 3.5, np.nan, 5.0, 2.1],
            'Movie 2': [3.8, np.nan, 4.7, 3.2, np.nan],
            'Movie 3': [np.nan, 4.1, 3.9, np.nan, 4.5],
            'Movie 4': [2.9, 3.3, np.nan, 4.8, 3.7],
            'Movie 5': [np.nan, np.nan, 2.8, 3.9, 4.2]
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Style the DataFrame for better visual appearance
        def highlight_nan(val):
            color = 'lightgray' if pd.isna(val) else ''
            return f'background-color: {color}'
        
        styled_df = sample_df.style.map(highlight_nan).format("{:.1f}", na_rep="?")
        st.dataframe(styled_df, height=200)
        
        # Simulate recommendations
        st.subheader("Predicted Ratings (After FM Model)")
        
        # Create simulated predicted ratings for missing values
        predicted_data = sample_data.copy()
        for col in predicted_data:
            if col != 'User ID':
                for i in range(len(predicted_data[col])):
                    if pd.isna(predicted_data[col][i]):
                        # Generate a plausible prediction
                        predicted_data[col][i] = round(3.0 + np.random.uniform(-1, 1), 1)
        
        predicted_df = pd.DataFrame(predicted_data)
        
        # Style the predicted DataFrame
        def highlight_predictions(val):
            if not isinstance(val, (int, float)):  # Skip the User ID column
                return ''
            if val in sample_data.values():  # Original values
                return 'background-color: white'
            else:  # Predicted values
                return 'background-color: #FFE4B5; font-weight: bold'  # Light orange
        
        styled_pred_df = predicted_df.style.map(highlight_predictions).format("{:.1f}")
        st.dataframe(styled_pred_df, height=200)


# ```

# This Streamlit application provides an interactive e-learning environment that demonstrates four key Amazon SageMaker algorithms: Linear Learner, XGBoost, K-Nearest Neighbors, and Factorization Machines.

# ### Key Features:

# 1. **Interactive Examples**: Each algorithm has interactive examples where users can:
#    - Generate synthetic data with configurable parameters
#    - Train models with different hyperparameters
#    - Visualize the results with appropriate metrics and plots

# 2. **Modern UI/UX**:
#    - Tab-based navigation with emoji icons
#    - AWS color scheme throughout the application
#    - Card-based layout for content organization
#    - Responsive design elements

# 3. **Educational Content**:
#    - Algorithm overviews and key features
#    - Use case recommendations
#    - Visualizations of model performance
#    - Code snippets for SageMaker implementation

# 4. **Session Management**:
#    - Reset functionality in the sidebar
#    - Session state initialization on load
#    - No preservation of inputs across sessions

# 5. **Rich Visualizations**:
#    - Decision boundary plots
#    - Confusion matrices
#    - Performance metrics
#    - Feature importance charts
#    - Recommendation system examples

# The application is structured to be both educational and practical, giving users hands-on experience with each algorithm while explaining their core concepts and implementation details.