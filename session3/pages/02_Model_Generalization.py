
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression, make_classification
from sklearn.neural_network import MLPRegressor, MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import base64
import time
import random

# Set page config
st.set_page_config(
    page_title="Model Generalization in ML | AWS Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for AWS themed styling
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3 {
        color: #232F3E;
    }
    .st-emotion-cache-16txtl3 a {
        color: #FF9900;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #EAEDED;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF9900 !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #FF9900;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #EC7211;
        color: white;
    }
    footer {
        font-size: 0.8em;
        color: #232F3E;
        text-align: center;
        margin-top: 50px;
    }
    .highlight {
        background-color: #FFECCC;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #FF9900;
    }
    .concept-box {
        background-color: #F2F3F3;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .grid-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }
    .formula-box {
        background-color: #EAEDED;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background-color: #EBF5FB;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1166BB;
        margin-bottom: 10px;
    }
    .warning-box {
        background-color: #FDEDEC;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #D13212;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}

init_session_state()

# Sidebar for session management
st.sidebar.markdown("### Session Management")

# Reset session button
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    st.sidebar.success("Session has been reset!")
    time.sleep(1)
    st.rerun()


st.sidebar.divider()

with st.sidebar.expander(label='About this application' ,expanded=False):
    st.markdown("""

    This application explores model generalization in machine learning, focusing on four key areas:

    - **Underfitting & Overfitting**: Visualize the balance between model simplicity and complexity
    - **Regularization Techniques**: Learn how constraints improve model performance on unseen data
    - **Dropout**: Explore how randomly disabling neurons enhances neural network generalization
    - **L1/L2 Regularization**: Compare different penalty approaches for controlling model complexity

    """)



# Data generation functions
@st.cache_data
def generate_polynomial_data(noise=0.5):
    np.random.seed(42)
    x = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = np.sin(x.ravel()) * 2 + 0.5
    y = y_true + np.random.normal(0, noise, size=y_true.shape)
    return x, y, y_true

@st.cache_data
def generate_complex_data(n_samples=100, noise=0.5):
    np.random.seed(42)
    x = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y_true = np.sin(x.ravel() * 1.5) * 2 + 0.5 * x.ravel()**2
    y = y_true + np.random.normal(0, noise, size=y_true.shape)
    return x, y, y_true

@st.cache_data
def generate_high_dim_data(n_samples=100, n_features=20, n_informative=5, noise=1.0):
    np.random.seed(42)
    X, y, coef = make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative,
        noise=noise, 
        coef=True, 
        random_state=42
    )
    return X, y, coef

@st.cache_data
def generate_moons_data(n_samples=1000, noise=0.3):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y

@st.cache_data
def compute_reg_path(reg_type, alphas, x_train_poly, y_train):
    coefs = []
    for a in alphas:
        if reg_type == "Ridge (L2)":
            model = Ridge(alpha=a)
        else:  # Lasso
            model = Lasso(alpha=a, max_iter=10000)
        
        model.fit(x_train_poly, y_train)
        coefs.append(model.coef_)
    
    return np.array(coefs)

# Main content
# st.title("🧠 Model Generalization in Machine Learning")

# Main content with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Introduction", 
    "📉 Underfitting & Overfitting", 
    "🔄 Regularization", 
    "🎭 Dropout", 
    "⚖️ L1/L2 Regularization",
    "❓ Knowledge Check"
])

# Tab 1: Introduction
with tab1:
    st.title("Model Generalization in Machine Learning")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is Model Generalization?</h3>
        <p>Model generalization refers to how well a machine learning model performs on unseen data after being trained on a training dataset.
        A model that generalizes well can make accurate predictions on new data it hasn't encountered before.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3,2])
    
    with col1:
        st.markdown("""
        ### What You'll Learn
        
        In this module, you'll explore:
        
        - **Underfitting & Overfitting**: Understand what happens when models are too simple or too complex
        - **Regularization Techniques**: Learn methods to prevent overfitting
        - **Dropout**: Discover how randomly dropping neurons improves neural network generalization
        - **L1/L2 Regularization**: Explore different penalty approaches to control model complexity
        
        Each section includes interactive examples and visualizations that allow you to explore these concepts in depth.
        
        ### Why Model Generalization Matters
        
        Generalization is the ultimate goal of machine learning:
        
        - It ensures models perform well on new, unseen data
        - It prevents models from simply memorizing training examples
        - It guides the selection of appropriate model complexity
        - It helps build reliable and robust AI systems
        """)
        
    with col2:
        st.image("https://miro.medium.com/max/1400/1*_7OPgojau8hkiPUiHoGK_w.png", caption="Model Generalization Concept")
        
        st.markdown('<div class="highlight">💡 A well-generalized model strikes the perfect balance between underfitting and overfitting!</div>', unsafe_allow_html=True)
    
    st.subheader("The Bias-Variance Tradeoff")
    
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.image("images/bias-and-variance.png", caption="The Bias-Variance Tradeoff")
    
    st.markdown("""
    The bias-variance tradeoff is a central problem in supervised learning:
    
    - **High Bias (Underfitting)**: Model is too simple, fails to capture important patterns
    - **High Variance (Overfitting)**: Model is too complex, captures noise along with patterns
    - **Optimal Model**: Balances bias and variance for best generalization
    
    Understanding this tradeoff helps guide the selection of model complexity and regularization techniques.
    """)
    
    st.markdown("---")
    
    st.subheader("Preview of Key Concepts")
    
    preview_col1, preview_col2 = st.columns(2)
    
    with preview_col1:
        complexity = np.linspace(1, 10, 100)
        bias = 1 / (complexity + 0.5)
        variance = 0.1 * complexity
        total_error = bias + variance
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=complexity, y=bias, mode='lines', name='Bias', line=dict(color='#232F3E', width=2)))
        fig.add_trace(go.Scatter(x=complexity, y=variance, mode='lines', name='Variance', line=dict(color='#FF9900', width=2)))
        fig.add_trace(go.Scatter(x=complexity, y=total_error, mode='lines', name='Total Error', line=dict(color='#D13212', width=2)))
        
        optimal_idx = np.argmin(total_error)
        fig.add_trace(go.Scatter(x=[complexity[optimal_idx]], y=[total_error[optimal_idx]], mode='markers', 
                                marker=dict(color='#D13212', size=10), name='Optimal Point'))
        
        fig.update_layout(
            title="Bias-Variance Tradeoff",
            xaxis_title="Model Complexity",
            yaxis_title="Error",
            width=600,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with preview_col2:
        st.markdown("""
        ### Finding the Right Balance
        
        The key to good generalization is finding the right model complexity:
        
        - **Too Simple (High Bias)**: The model misses important patterns in the data
        - **Too Complex (High Variance)**: The model learns noise and random fluctuations
        - **Just Right**: The model captures the underlying patterns without fitting to noise
        
        The techniques you'll learn in this module will help you find this balance through:
        
        1. **Proper model selection**
        2. **Appropriate regularization**
        3. **Effective training practices**
        """)
        
        st.markdown('<div class="highlight">The model with the lowest error on a validation set (not used for training) is often the one with the best generalization ability.</div>', unsafe_allow_html=True)

# Tab 2: Underfitting & Overfitting
with tab2:
    st.title("📉 Underfitting & Overfitting")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Understanding the Fitting Spectrum</h3>
        <p>Machine learning models fall along a spectrum from underfitting to overfitting. The goal is to find the sweet spot in between.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="info-box"><strong>Underfitting</strong>: Model is too simple to capture the underlying pattern in the data. It performs poorly on both training and test data.</div>', unsafe_allow_html=True)
        st.image("images/underfit.png", width=300, caption="Underfitting Example")
    
    with col2:
        st.markdown('<div class="info-box"><strong>Overfitting</strong>: Model is too complex and captures noise in the training data. It performs well on training data but poorly on test data.</div>', unsafe_allow_html=True)
        st.image("images/overfit.png", width=300, caption="Overfitting Example")
    
    # Interactive demo of polynomial regression
    st.subheader("Interactive Polynomial Regression Demo")
    
    st.markdown("""
    Adjust the polynomial degree slider to see how model complexity affects fitting. 
    Higher polynomial degrees create more complex models that can lead to overfitting.
    """)
    
    # Generate data for polynomial regression
    x, y, y_true = generate_polynomial_data()
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Controls for the demo
    col1, col2 = st.columns([1, 1])
    
    with col1:
        polynomial_degree = st.slider("Model Complexity (Polynomial Degree)", 1, 15, 1, help="Higher degree = more complex model")
    
    with col2:
        show_true_function = st.checkbox("Show True Function", value=True)
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=polynomial_degree)
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.transform(x_test)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    
    # Make predictions
    x_all = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    x_all_poly = poly_features.transform(x_all)
    y_pred = model.predict(x_all_poly)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, model.predict(x_train_poly))
    test_error = mean_squared_error(y_test, model.predict(x_test_poly))
    
    # Create a plot using plotly
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(go.Scatter(
        x=x_train.flatten(),
        y=y_train,
        mode='markers',
        name='Training Data',
        marker=dict(color='#1166BB', size=8)
    ))
    
    # Add test data
    fig.add_trace(go.Scatter(
        x=x_test.flatten(),
        y=y_test,
        mode='markers',
        name='Test Data',
        marker=dict(color='#D13212', size=8)
    ))
    
    # Add model prediction
    fig.add_trace(go.Scatter(
        x=x_all.flatten(),
        y=y_pred,
        mode='lines',
        name=f'Polynomial Degree {polynomial_degree}',
        line=dict(color='#FF9900', width=3)
    ))
    
    # Add true function if selected
    if show_true_function:
        y_true_all = np.sin(x_all.ravel()) * 2 + 0.5
        fig.add_trace(go.Scatter(
            x=x_all.flatten(),
            y=y_true_all,
            mode='lines',
            name='True Function',
            line=dict(color='green', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'Polynomial Regression (Degree {polynomial_degree})',
        xaxis_title='x',
        yaxis_title='y',
        width=800,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display errors
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Error (MSE)", f"{train_error:.4f}")
    with col2:
        st.metric("Test Error (MSE)", f"{test_error:.4f}")
    
    # Assessment based on errors
    error_ratio = test_error / (train_error + 1e-10)  # Avoid division by zero
    
    if polynomial_degree <= 2 and test_error > 0.3:
        st.markdown('<div class="warning-box">⚠️ <b>Underfitting</b>: The model is too simple and cannot capture the underlying pattern. Both training and test errors are high.</div>', unsafe_allow_html=True)
    elif error_ratio > 2.0:
        st.markdown('<div class="warning-box">⚠️ <b>Overfitting</b>: The model is too complex and is capturing noise in the training data. Test error is much higher than training error.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">✅ <b>Good Fit</b>: The model seems to have a good balance between complexity and generalization.</div>', unsafe_allow_html=True)
    
    # Educational content about bias-variance
    st.subheader("Understanding Model Complexity")
    
    st.markdown("""
    As we adjust the polynomial degree (model complexity), we can observe:
    
    1. **Low Complexity (Degree 1-2)**:
       - High bias, low variance
       - Underfits the data (misses the pattern)
       - Both training and test errors are high
    
    2. **Medium Complexity (Degree 3-5)**:
       - Balanced bias and variance
       - Captures the general pattern without fitting to noise
       - Training and test errors are both relatively low
    
    3. **High Complexity (Degree >6)**:
       - Low bias, high variance
       - Overfits the data (fits to noise)
       - Training error is very low, but test error is high
    
    The key lesson is that **more complex isn't always better**. The goal is to find the model complexity that minimizes the error on unseen data.
    """)
    
    st.markdown('<div class="highlight">An ideal model should be complex enough to learn the underlying pattern but simple enough to ignore the noise.</div>', unsafe_allow_html=True)

# Tab 3: Regularization
with tab3:
    st.title("🔄 Regularization")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is Regularization?</h3>
        <p>Regularization is a set of techniques designed to prevent overfitting by adding additional constraints to the learning process, typically in the form of penalties on model complexity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Key Regularization Techniques:
    
    - **L1 Regularization (Lasso)**: Adds a penalty equal to the absolute value of the magnitude of coefficients
    - **L2 Regularization (Ridge)**: Adds a penalty equal to the square of the magnitude of coefficients
    - **Dropout**: Randomly disables neurons during training (specific to neural networks)
    - **Early Stopping**: Stops training when performance on validation data starts to degrade
    - **Data Augmentation**: Artificially increasing the size of the training dataset
    """)
    
    # Interactive visualization for regularization effects
    st.subheader("Effects of Regularization Strength")
    
    st.markdown("""
    Explore how different types and strengths of regularization affect model fitting:
    """)
    
    # Generate synthetic data
    x, y, y_true = generate_complex_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        poly_degree = st.slider("Polynomial Degree", 1, 12, 8, help="Higher degree can lead to overfitting")
    
    with col2:
        reg_type = st.selectbox("Regularization Type", ["None", "Ridge (L2)", "Lasso (L1)"])
    
    with col3:
        if reg_type != "None":
            alpha = st.slider("Regularization Strength (α)", 0.0001, 10.0, 0.1, format="%.4f", help="Higher values = stronger regularization")
        else:
            alpha = 0.0
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=poly_degree)
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.transform(x_test)
    
    # Fit the model based on regularization type
    if reg_type == "Ridge (L2)":
        model = Ridge(alpha=alpha)
    elif reg_type == "Lasso (L1)":
        model = Lasso(alpha=alpha)
    else:
        model = LinearRegression()
    
    model.fit(x_train_poly, y_train)
    
    # Make predictions
    x_all = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    x_all_poly = poly_features.transform(x_all)
    y_pred = model.predict(x_all_poly)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, model.predict(x_train_poly))
    test_error = mean_squared_error(y_test, model.predict(x_test_poly))
    
    # Create a plot using plotly
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(go.Scatter(
        x=x_train.flatten(),
        y=y_train,
        mode='markers',
        name='Training Data',
        marker=dict(color='#1166BB', size=8)
    ))
    
    # Add test data
    fig.add_trace(go.Scatter(
        x=x_test.flatten(),
        y=y_test,
        mode='markers',
        name='Test Data',
        marker=dict(color='#D13212', size=8)
    ))
    
    # Add model prediction
    fig.add_trace(go.Scatter(
        x=x_all.flatten(),
        y=y_pred,
        mode='lines',
        name=f'{reg_type if reg_type != "None" else "No Regularization"}',
        line=dict(color='#FF9900', width=3)
    ))
    
    # Add true function
    y_true_all = np.sin(x_all.ravel() * 1.5) * 2 + 0.5 * x_all.ravel()**2
    fig.add_trace(go.Scatter(
        x=x_all.flatten(),
        y=y_true_all,
        mode='lines',
        name='True Function',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Polynomial Regression with {reg_type} Regularization',
        xaxis_title='x',
        yaxis_title='y',
        width=800,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display errors
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Training Error (MSE)", f"{train_error:.4f}")
    with col2:
        st.metric("Test Error (MSE)", f"{test_error:.4f}")
    with col3:
        st.metric("Generalization Gap", f"{test_error - train_error:.4f}", delta_color="inverse")
    
    # Display feature coefficients
    if poly_degree <= 10:  # Only show coefficients for reasonable polynomial degrees
        st.subheader("Model Coefficients")
        st.write("See how regularization affects the model coefficients:")
        
        # Get feature names
        feature_names = poly_features.get_feature_names_out()
        
        # Get coefficients (handle the intercept differently for LinearRegression vs Ridge/Lasso)
        if reg_type == "None":
            coefs = np.concatenate(([model.intercept_], model.coef_[1:]))
            feature_names = np.array(['intercept'] + list(feature_names[1:]))
        else:
            coefs = np.concatenate(([model.intercept_], model.coef_[1:]))
            feature_names = np.array(['intercept'] + list(feature_names[1:]))
        
        # Sort by absolute value for better visualization
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        sorted_coefs = coefs[sorted_idx]
        sorted_names = feature_names[sorted_idx]
        
        # Create coefficient plot with plotly
        fig = go.Figure()
        
        colors = ['#FF9900' if c > 0 else '#D13212' for c in sorted_coefs]
        
        fig.add_trace(go.Bar(
            x=sorted_names,
            y=sorted_coefs,
            marker_color=colors,
            name='Coefficients'
        ))
        
        fig.update_layout(
            title=f'Model Coefficients with {reg_type} Regularization (α={alpha if reg_type != "None" else 0})',
            xaxis_title='Feature',
            yaxis_title='Coefficient Value',
            width=800,
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational explanation
    st.subheader("How Regularization Works")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **L1 Regularization (Lasso)** adds a penalty term to the loss function proportional to the absolute sum of coefficient values:
        
        <div class='formula-box'>
        $$Loss = MSE + \\alpha \\sum_{i=1}^{n} |w_i|$$
        </div>
        
        This encourages sparse models by pushing some coefficients exactly to zero, effectively performing feature selection.
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">L1 regularization is useful when you suspect many features are irrelevant.</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        **L2 Regularization (Ridge)** adds a penalty term proportional to the squared sum of coefficient values:
        
        <div class="formula-box">
        $$Loss = MSE + \\alpha \\sum_{i=1}^{n} w_i^2$$
        </div>
        
        This shrinks all coefficients toward zero but rarely makes them exactly zero. It works well when all features contribute to the prediction.
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">L2 regularization helps when dealing with multicollinearity (correlated features).</div>', unsafe_allow_html=True)
    
    # Visualize the regularization paths
    st.subheader("Regularization Path")
    st.write("See how coefficient values change with increasing regularization strength:")
    
    # Generate regularization path
    alphas = np.logspace(-3, 1, 20)
    
    # Compare paths between Ridge and Lasso
    path_type = st.radio("Select Regularization Path to Visualize:", ["Ridge (L2)", "Lasso (L1)"])
    
    paths = compute_reg_path(path_type, alphas, x_train_poly, y_train)
    
    # Create path plot with plotly
    fig = go.Figure()
    
    # Plot paths for each coefficient
    for i in range(min(10, paths.shape[1])):  # Only plot the first 10 coefficients for clarity
        fig.add_trace(go.Scatter(
            x=alphas,
            y=paths[:, i],
            name=f"Coef {i+1}",
            mode='lines',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f'{path_type} Regularization Path',
        xaxis_title='Regularization Strength (α)',
        yaxis_title='Coefficient Value',
        xaxis_type='log',
        width=800,
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if path_type == "Lasso (L1)":
        st.markdown('<div class="info-box">Notice how Lasso regularization can drive coefficients exactly to zero as regularization strength increases, performing <b>feature selection</b>.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Ridge regularization shrinks all coefficients toward zero but rarely makes them exactly zero.</div>', unsafe_allow_html=True)

# Tab 4: Dropout
with tab4:
    st.title("🎭 Dropout")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is Dropout?</h3>
        <p>Dropout is a regularization technique specifically designed for neural networks. It works by randomly 
        "dropping out" (temporarily removing) neurons during training, which helps prevent overfitting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### How Dropout Works:
        
        1. During each training iteration, randomly disable a percentage of neurons
        2. Forward and backward passes use only the remaining neurons
        3. At test time, all neurons are used, but outputs are scaled appropriately
        
        This forces the network to learn redundant representations and prevents co-adaptation of neurons,
        where they become too dependent on each other.
        """)
        
        st.markdown('<div class="highlight">💡 Think of dropout as creating an ensemble of many different networks that share weights!</div>', unsafe_allow_html=True)
    
    with col2:
        st.image("https://miro.medium.com/max/1400/1*iWQzxhVlvadk6VAJjsgXgg.png", caption="Dropout Mechanism")
    
    # Interactive dropout demo
    st.subheader("Interactive Dropout Demonstration")
    
    st.markdown("""
    Experiment with dropout rates to see how they affect neural network performance and generalization:
    """)
    
    # Generate synthetic data for a classification problem
    X, y = generate_moons_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        hidden_layers = st.slider("Number of Hidden Layers", 1, 3, 2)
        neurons_per_layer = st.slider("Neurons per Hidden Layer", 4, 32, 16)
        
    with col2:
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.2, 0.05, help="0.0 = No dropout, 0.5 = Drop half the neurons")
        epochs = st.slider("Training Epochs", 10, 200, 50)
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Build a simple neural network with/without dropout
    def build_model(dropout_rate):
        model = Sequential()
        model.add(Dense(neurons_per_layer, activation='relu', input_shape=(2,)))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        for _ in range(hidden_layers - 1):
            model.add(Dense(neurons_per_layer, activation='relu'))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    # Progress indicator for model training
    with st.spinner("Training model with dropout... This may take a moment"):
        model = build_model(dropout_rate)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
    
    # Create a meshgrid for visualization
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
    Z = Z.reshape(xx.shape)
    
    # Create plotly figure with two subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Decision Boundary", "Training History"])
    
    # Decision boundary contour plot
    contour = go.Contour(
        z=Z,
        x=np.arange(x_min, x_max, h), 
        y=np.arange(y_min, y_max, h),
        colorscale='RdBu',
        opacity=0.8,
        showscale=False
    )
    
    # Training points
    train_points = go.Scatter(
        x=X_train[:, 0], 
        y=X_train[:, 1],
        mode='markers',
        marker=dict(
            color=y_train,
            colorscale='RdBu',
            line=dict(color='black', width=1)
        ),
        name='Training Data'
    )
    
    # Test points
    test_points = go.Scatter(
        x=X_test[:, 0], 
        y=X_test[:, 1],
        mode='markers',
        marker=dict(
            symbol='square',
            color=y_test,
            colorscale='RdBu',
            line=dict(color='black', width=1)
        ),
        name='Test Data'
    )
    
    # Add traces to decision boundary plot
    fig.add_trace(contour, row=1, col=1)
    fig.add_trace(train_points, row=1, col=1)
    fig.add_trace(test_points, row=1, col=1)
    
    # Learning curves
    fig.add_trace(
        go.Scatter(
            x=list(range(len(history.history['accuracy']))),
            y=history.history['accuracy'],
            mode='lines',
            name='Training Accuracy',
            line=dict(color='#232F3E', width=2)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(history.history['val_accuracy']))),
            y=history.history['val_accuracy'],
            mode='lines',
            name='Validation Accuracy',
            line=dict(color='#FF9900', width=2)
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Neural Network with Dropout Rate = {dropout_rate}',
        height=500,
        width=1000
    )
    
    fig.update_xaxes(title_text="Feature 1", row=1, col=1)
    fig.update_yaxes(title_text="Feature 2", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display metrics
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Training Accuracy", f"{train_acc:.4f}")
    with col2:
        st.metric("Test Accuracy", f"{test_acc:.4f}")
    with col3:
        gap = train_acc - test_acc
        st.metric("Train-Test Gap", f"{gap:.4f}", delta=f"{-gap:.4f}" if gap > 0 else None, delta_color="inverse")
    
    # Provide feedback based on results
    if gap > 0.15:
        st.markdown('<div class="warning-box">⚠️ <b>Potential Overfitting</b>: The model performs much better on training data than test data. Try increasing dropout rate.</div>', unsafe_allow_html=True)
    elif test_acc < 0.8:
        st.markdown('<div class="warning-box">⚠️ <b>Potential Underfitting</b>: The model has relatively low accuracy on test data. Try decreasing dropout rate or increasing model capacity.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">✅ <b>Good Generalization</b>: The model shows a good balance between training and test performance.</div>', unsafe_allow_html=True)
    
    # Dropout explanation
    st.subheader("Why Does Dropout Work?")
    
    st.markdown("""
    Dropout works by:
    
    1. **Preventing Co-adaptation**: Neurons can't rely on the presence of specific other neurons, making each one more robust
    2. **Ensemble Effect**: Each training batch effectively trains a different "thinned" network, similar to training an ensemble
    3. **Reducing Overfitting**: By randomly removing neurons, the network can't memorize the training data as easily
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="info-box">🎯 <b>Recommended Dropout Rates</b>:<br>- Input layer: 0.1-0.2<br>- Hidden layers: 0.3-0.5<br>- Adjust based on network size and dataset</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">⚙️ <b>When to Use Dropout</b>:<br>- Large neural networks<br>- Limited training data<br>- When overfitting is observed</div>', unsafe_allow_html=True)

# Tab 5: L1/L2 Regularization
with tab5:
    st.title("⚖️ L1/L2 Regularization")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Understanding L1 and L2 Regularization</h3>
        <p>L1 and L2 regularization are powerful techniques to control the complexity of machine learning models by adding penalties on the weights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("L1 Regularization (Lasso)")
        st.markdown("""
        Adds a penalty equal to the absolute value of coefficients:
        
        <div class="formula-box">
        $$Loss = MSE + \\alpha \\sum_{i=1}^{n} |w_i|$$
        </div>
        
        **Key characteristics:**
        - Produces sparse models (many coefficients become exactly zero)
        - Performs feature selection automatically
        - Useful when you suspect many features are irrelevant
        """, unsafe_allow_html=True)
        
        st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*P3bTxrYuLGhF5-m0WEgomQ.png", caption="L1 Regularization Contours")
    
    with col2:
        st.subheader("L2 Regularization (Ridge)")
        st.markdown("""
        Adds a penalty equal to the square of coefficients:
        
        <div class="formula-box">
        $$Loss = MSE + \\alpha \\sum_{i=1}^{n} w_i^2$$
        </div>
        
        **Key characteristics:**
        - Shrinks coefficients toward zero but rarely makes them exactly zero
        - All features typically remain in the model, just with smaller coefficients
        - Works well when all features contribute somewhat to the prediction
        - Effective for handling multicollinearity
        """, unsafe_allow_html=True)
        
        st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*lFmQebk5GoibKN2ZQYzYpw.png", caption="L2 Regularization Contours")
    
    # Interactive demo comparing L1 and L2 regularization
    st.subheader("Interactive L1 vs L2 Regularization Comparison")
    
    st.markdown("""
    This demonstration shows how L1 and L2 regularization affect model coefficients differently.
    L1 tends to generate sparse solutions (many coefficients = 0) while L2 shrinks all coefficients proportionally.
    """)
    
    # Generate more complex synthetic data with many features
    X, y, true_coef = generate_high_dim_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        alpha_l1 = st.slider("L1 (Lasso) Regularization Strength", 0.0001, 2.0, 0.1, 0.01)
        alpha_l2 = st.slider("L2 (Ridge) Regularization Strength", 0.0001, 2.0, 0.1, 0.01)
    
    # Train models
    lasso = Lasso(alpha=alpha_l1, max_iter=10000)
    ridge = Ridge(alpha=alpha_l2)
    linear = LinearRegression()
    
    lasso.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    linear.fit(X_train, y_train)
    
    # Calculate metrics
    models = {
        "Linear (No Regularization)": linear,
        "L1 (Lasso)": lasso,
        "L2 (Ridge)": ridge
    }
    
    train_scores = {}
    test_scores = {}
    nonzero_coefs = {}
    
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        train_scores[name] = train_mse
        test_scores[name] = test_mse
        
        # Count non-zero coefficients
        if hasattr(model, 'coef_'):
            nonzero_coefs[name] = np.sum(np.abs(model.coef_) > 1e-5)
    
    # Display metrics
    st.subheader("Model Performance")
    
    metrics_df = pd.DataFrame({
        'Model': list(train_scores.keys()),
        'Training MSE': [round(v, 4) for v in train_scores.values()],
        'Test MSE': [round(v, 4) for v in test_scores.values()],
        'Non-zero Coefficients': [nonzero_coefs.get(m, "N/A") for m in train_scores.keys()]
    })
    
    st.table(metrics_df)
    
    # Plot coefficient comparison
    st.subheader("Coefficient Comparison")
    
    # Create coefficient comparison with plotly
    fig = go.Figure()
    
    # Add each model's coefficients
    fig.add_trace(go.Bar(
        x=np.arange(len(true_coef)),
        y=true_coef,
        name='True Coefficients',
        marker_color='#232F3E'
    ))
    
    fig.add_trace(go.Bar(
        x=np.arange(len(linear.coef_)),
        y=linear.coef_,
        name='Linear (No Regularization)',
        marker_color='#FF9900'
    ))
    
    fig.add_trace(go.Bar(
        x=np.arange(len(lasso.coef_)),
        y=lasso.coef_,
        name=f'L1 (Lasso, α={alpha_l1})',
        marker_color='#1166BB'
    ))
    
    fig.add_trace(go.Bar(
        x=np.arange(len(ridge.coef_)),
        y=ridge.coef_,
        name=f'L2 (Ridge, α={alpha_l2})',
        marker_color='#D13212'
    ))
    
    fig.update_layout(
        barmode='group',
        title='Model Coefficients Comparison',
        xaxis_title='Feature Index',
        yaxis_title='Coefficient Value',
        width=900,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the results
    st.subheader("Interpretation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if nonzero_coefs["L1 (Lasso)"] < nonzero_coefs["L2 (Ridge)"]:
            st.markdown('<div class="info-box">✅ <b>L1 (Lasso) Effect</b>: Notice how Lasso has reduced many coefficients to exactly zero, performing feature selection.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ️ <b>L1 (Lasso) Effect</b>: Try increasing the L1 regularization strength to see more pronounced feature selection.</div>', unsafe_allow_html=True)
    
    with col2:
        if test_scores["L2 (Ridge)"] < test_scores["Linear (No Regularization)"]:
            st.markdown('<div class="info-box">✅ <b>L2 (Ridge) Effect</b>: Ridge regularization has improved test performance by reducing overfitting while keeping all features.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ️ <b>L2 (Ridge) Effect</b>: Try adjusting the L2 regularization strength to improve test performance.</div>', unsafe_allow_html=True)
    
    # Visual explanation of L1 vs L2 geometry
    st.subheader("Geometrical Interpretation of L1 vs L2 Regularization")
    
    st.markdown("""
    The geometrical difference between L1 and L2 regularization explains why L1 produces sparse solutions:
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*wB7K1ubmrJsB2_vgQvKDTA.png", caption="L1 vs L2 Regularization Geometry")
    
    with col2:
        st.markdown("""
        The plots show the contours of the error function (ellipses) and the constraint regions for L1 (diamond) and L2 (circle).
        
        - **L1 Regularization**: The diamond shape has corners along the axes. When the error contour meets this shape, it often occurs at one of these corners, which correspond to sparse solutions (some coefficients = 0).
        
        - **L2 Regularization**: The circular shape has no corners, so when the error contour meets this shape, it typically results in all coefficients being non-zero but smaller in magnitude.
        """)
    
    # Elastic Net
    st.subheader("Elastic Net: Combining L1 and L2 Regularization")
    
    st.markdown("""
    Elastic Net combines L1 and L2 regularization, offering the best of both worlds:
    
    <div class="formula-box">
    $$Loss = MSE + \\alpha_1 \\sum_{i=1}^{n} |w_i| + \\alpha_2 \\sum_{i=1}^{n} w_i^2$$
    </div>
    
    This approach:
    - Performs feature selection like Lasso
    - Handles groups of correlated features better than Lasso
    - Reduces coefficient magnitude like Ridge
    
    Elastic Net is particularly useful when you have many correlated features.
    """, unsafe_allow_html=True)
    
    # When to use which regularization
    st.subheader("When to Use Each Regularization Method")
    
    methods_df = pd.DataFrame({
        'Method': ['No Regularization', 'L1 (Lasso)', 'L2 (Ridge)', 'Elastic Net'],
        'Best When': [
            'Few features relative to samples, all features relevant',
            'Many irrelevant features that should be excluded',
            'Many somewhat relevant features, especially with multicollinearity',
            'Many features with groups of correlated variables'
        ],
        'Effect on Coefficients': [
            'No effect - uses full coefficient values',
            'Many coefficients become exactly zero (sparse solution)',
            'All coefficients shrink proportionally, rarely become zero',
            'Some coefficients become zero, others shrink (controllable balance)'
        ]
    })
    
    st.table(methods_df)

# Tab 6: Knowledge Check
with tab6:
    st.title("❓ Knowledge Check")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Test Your Understanding</h3>
        <p>Answer these five questions to check your understanding of model generalization concepts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define the questions and answers
    questions = [
        {
            "question": "Which of the following best describes overfitting?",
            "options": [
                "When a model performs well on training data but poorly on new data",
                "When a model is too simple to capture the underlying patterns in the data",
                "When a model performs equally well on both training and test data",
                "When a model has fewer parameters than the number of training examples"
            ],
            "answer": "When a model performs well on training data but poorly on new data",
            "explanation": "Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor generalization to new data."
        },
        {
            "question": "What is the main purpose of regularization in machine learning?",
            "options": [
                "To increase model complexity",
                "To prevent overfitting by adding constraints to the model",
                "To speed up the training process",
                "To make models more interpretable"
            ],
            "answer": "To prevent overfitting by adding constraints to the model",
            "explanation": "Regularization techniques add penalties or constraints to the model to prevent overfitting and improve generalization to unseen data."
        },
        {
            "question": "How does dropout help prevent overfitting in neural networks?",
            "options": [
                "By removing neurons with the lowest weights",
                "By adding more layers to the network",
                "By randomly disabling neurons during training",
                "By increasing the learning rate"
            ],
            "answer": "By randomly disabling neurons during training",
            "explanation": "Dropout randomly disables neurons during training, which prevents co-adaptation and forces the network to learn redundant representations, improving generalization."
        },
        {
            "question": "Which regularization technique is more likely to produce sparse models with many coefficients exactly equal to zero?",
            "options": [
                "L1 (Lasso) regularization",
                "L2 (Ridge) regularization",
                "Elastic Net with high L2 penalty",
                "No regularization"
            ],
            "answer": "L1 (Lasso) regularization",
            "explanation": "L1 (Lasso) regularization tends to produce sparse models with many coefficients exactly equal to zero, effectively performing feature selection."
        },
        {
            "question": "The bias-variance tradeoff suggests that:",
            "options": [
                "Higher bias always leads to better generalization",
                "Higher variance always leads to better generalization",
                "The optimal model minimizes the sum of bias and variance",
                "Bias and variance are unrelated to model generalization"
            ],
            "answer": "The optimal model minimizes the sum of bias and variance",
            "explanation": "The bias-variance tradeoff states that the optimal model minimizes the total error, which is the sum of bias, variance, and irreducible error."
        }
    ]
    
    # Quiz logic
    if not st.session_state.quiz_submitted:
        st.markdown("### Answer the following questions:")
        
        for i, q in enumerate(questions):
            st.markdown(f"**Question {i+1}**: {q['question']}")
            st.session_state.quiz_answers[i] = st.radio(
                f"Select your answer for question {i+1}:",
                q['options'], index=None,
                key=f"q{i}"
            )
            st.markdown("---")
        
        if st.button("Submit Answers"):
            score = 0
            feedback = {}
            
            for i, q in enumerate(questions):
                if st.session_state.quiz_answers[i] == q['answer']:
                    score += 1
                    feedback[i] = {
                        "correct": True,
                        "message": "Correct! " + q["explanation"]
                    }
                else:
                    feedback[i] = {
                        "correct": False,
                        "message": "Incorrect. " + q["explanation"]
                    }
            
            st.session_state.quiz_score = score
            st.session_state.feedback = feedback
            st.session_state.quiz_submitted = True
            st.rerun()
    
    else:
        # Show results
        st.markdown(f"### Your Score: {st.session_state.quiz_score}/{len(questions)}")
        
        # Create gauge chart for score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state.quiz_score / len(questions) * 100,
            title={'text': "Score Percentage"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#FF9900"},
                'steps': [
                    {'range': [0, 40], 'color': "#FF4136"},
                    {'range': [40, 80], 'color': "#FFDC00"},
                    {'range': [80, 100], 'color': "#2ECC40"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig)
        
        # Display detailed feedback
        st.markdown("### Detailed Results:")
        
        for i, q in enumerate(questions):
            if i in st.session_state.feedback:
                feedback = st.session_state.feedback[i]
                
                if feedback["correct"]:
                    st.markdown(f'<div class="info-box">✅ <b>Question {i+1}:</b> {feedback["message"]}</div>', unsafe_allow_html=True)
                else:
                    correct_option = q["answer"]
                    st.markdown(f'<div class="warning-box">❌ <b>Question {i+1}:</b> {feedback["message"]} Correct answer: {correct_option}</div>', unsafe_allow_html=True)
        
        # Final message based on score
        if st.session_state.quiz_score == len(questions):
            st.balloons()
            st.markdown('<div class="highlight">🎉 Perfect score! You have an excellent understanding of model generalization concepts!</div>', unsafe_allow_html=True)
        elif st.session_state.quiz_score >= len(questions) * 0.8:
            st.markdown('<div class="highlight">👍 Great job! You have a strong understanding of model generalization concepts!</div>', unsafe_allow_html=True)
        elif st.session_state.quiz_score >= len(questions) * 0.6:
            st.markdown('<div class="info-box">👍 Good effort! Review the concepts you missed to strengthen your understanding.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">📚 You may need more practice. Try reviewing the material again.</div>', unsafe_allow_html=True)
        
        if st.button("Retake Quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score = 0
            st.session_state.feedback = {}
            st.rerun()

# Summary section
st.markdown("""
## Summary of Model Generalization Techniques

| Concept | Description | Key Techniques |
|-----------|-------------|-------------------|
| Underfitting & Overfitting | The balance between learning the pattern vs. memorizing the data | Model selection, Validation, Cross-validation |
| Regularization | Adding constraints to prevent overfitting | L1/L2 penalties, Early stopping, Data augmentation |
| Dropout | Randomly disabling neurons during training | Used in neural networks to prevent co-adaptation |
| L1/L2 Regularization | Different penalty approaches for weights | L1 for sparsity, L2 for small weights, Elastic Net for both |
""")

# Footer
st.markdown("""
<footer>
© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</footer>
""", unsafe_allow_html=True)
