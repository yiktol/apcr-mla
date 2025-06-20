
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
from PIL import Image
import base64
import math

# Set page config
st.set_page_config(
    page_title="Model Optimization Techniques | AWS Learning",
    page_icon="🔧",
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
    if 'gd_iterations' not in st.session_state:
        st.session_state.gd_iterations = []
    if 'loss_function_type' not in st.session_state:
        st.session_state.loss_function_type = "MSE"
    if 'convergence_data' not in st.session_state:
        st.session_state.convergence_data = None
    if 'gd_history' not in st.session_state:
        st.session_state.gd_history = None

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
This application explores essential machine learning optimization techniques with hands-on visualizations. The app focuses on three key areas:

- **Loss Functions**: Understand different ways to measure prediction errors and when to use MSE, MAE, Log Loss, and Huber Loss
- **Convergence**: Learn how optimization algorithms approach optimal solutions and the factors affecting their stability
- **Gradient Descent**: Visualize the fundamental optimization algorithm through interactive 2D and 3D demonstrations
    """)



# Custom functions for data generation and visualization
def generate_regression_data(n_samples=100, noise=0.5, random_state=42):
    """Generate synthetic regression data"""
    X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=random_state)
    return X, y

def generate_classification_data(n_samples=100, random_state=42):
    """Generate synthetic classification data"""
    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_redundant=0,
        n_informative=2, random_state=random_state, n_clusters_per_class=1
    )
    return X, y

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss function"""
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    """Mean Absolute Error loss function"""
    return np.mean(np.abs(y_true - y_pred))

def log_loss_binary(y_true, y_pred):
    """Binary cross-entropy loss function"""
    # Clip predictions to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss function"""
    errors = y_true - y_pred
    mask = np.abs(errors) <= delta
    squared_loss = 0.5 * (errors ** 2)
    linear_loss = delta * (np.abs(errors) - 0.5 * delta)
    return np.mean(mask * squared_loss + (~mask) * linear_loss)

def linear_regression_predict(X, coef, intercept):
    """Simple linear regression prediction"""
    return X * coef + intercept

def logistic_regression_predict(X, coef, intercept):
    """Simple logistic regression prediction"""
    z = X @ coef + intercept
    return 1 / (1 + np.exp(-z))

def gradient_descent_linear(X, y, learning_rate=0.01, n_iterations=100, tolerance=1e-6):
    """Implement gradient descent for linear regression"""
    # Initialize parameters
    coef = 0.0
    intercept = 0.0
    
    # History for visualization
    history = {
        'iterations': np.arange(n_iterations),
        'coef': np.zeros(n_iterations),
        'intercept': np.zeros(n_iterations),
        'loss': np.zeros(n_iterations),
        'grad_coef': np.zeros(n_iterations),
        'grad_intercept': np.zeros(n_iterations)
    }
    
    # Flatten X for simpler operations
    X = X.flatten()
    
    # Run gradient descent
    for i in range(n_iterations):
        # Make predictions
        y_pred = linear_regression_predict(X, coef, intercept)
        
        # Calculate loss
        loss = mse_loss(y, y_pred)
        
        # Calculate gradients
        grad_coef = -2 * np.mean(X * (y - y_pred))
        grad_intercept = -2 * np.mean(y - y_pred)
        
        # Store in history
        history['coef'][i] = coef
        history['intercept'][i] = intercept
        history['loss'][i] = loss
        history['grad_coef'][i] = grad_coef
        history['grad_intercept'][i] = grad_intercept
        
        # Check for convergence
        if i > 0 and abs(history['loss'][i] - history['loss'][i-1]) < tolerance:
            # Truncate history to actual iterations
            for key in history.keys():
                if key != 'iterations':
                    history[key] = history[key][:i+1]
            history['iterations'] = np.arange(i+1)
            break
        
        # Update parameters
        coef = coef - learning_rate * grad_coef
        intercept = intercept - learning_rate * grad_intercept
        
    return coef, intercept, history

def plot_loss_function(loss_type, x_range=(-5, 5), num_points=100):
    """Plot a loss function visualization"""
    y_true = 0  # 'correct' value
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    
    if loss_type == "MSE":
        loss_vals = [(x - y_true)**2 for x in x_vals]
        title = "Mean Squared Error (MSE) Loss"
    elif loss_type == "MAE":
        loss_vals = [abs(x - y_true) for x in x_vals]
        title = "Mean Absolute Error (MAE) Loss"
    elif loss_type == "Log Loss":
        # Use sigmoid to constrain values between 0 and 1
        sigmoid_vals = 1 / (1 + np.exp(-x_vals))
        # True value is 1 for positive x_vals, 0 for negative
        y_true_binary = (x_vals > 0).astype(int)
        loss_vals = [-y * np.log(max(y_hat, 1e-15)) - (1-y) * np.log(max(1-y_hat, 1e-15)) 
                     for y, y_hat in zip(y_true_binary, sigmoid_vals)]
        title = "Log Loss (Binary Cross-Entropy)"
    elif loss_type == "Huber":
        delta = 1.0
        loss_vals = [
            0.5 * (x - y_true)**2 if abs(x - y_true) <= delta 
            else delta * (abs(x - y_true) - 0.5 * delta)
            for x in x_vals
        ]
        title = f"Huber Loss (δ={delta})"
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, 
        y=loss_vals,
        mode='lines',
        name=loss_type,
        line=dict(color='#FF9900', width=3)
    ))
    
    # Add point at minimum
    min_idx = np.argmin(loss_vals)
    fig.add_trace(go.Scatter(
        x=[x_vals[min_idx]],
        y=[loss_vals[min_idx]],
        mode='markers',
        marker=dict(size=10, color='#232F3E'),
        name='Minimum'
    ))
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title='Prediction',
        yaxis_title='Loss',
        width=700,
        height=400
    )
    
    return fig

def plot_convergence(iterations, loss_values, learning_rate=None, algorithm=None):
    """Plot convergence visualization"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=loss_values,
        mode='lines+markers',
        line=dict(color='#FF9900', width=3),
        name='Loss'
    ))
    
    # Add a vertical line at the "convergence" point
    # Define as where loss change is very small
    loss_diffs = np.diff(loss_values)
    if len(loss_diffs) > 10:  # Make sure we have enough points
        # Find where consecutive differences are small
        conv_idx = None
        for i in range(len(loss_diffs)-3):
            if abs(loss_diffs[i]) < 0.001 and abs(loss_diffs[i+1]) < 0.001 and abs(loss_diffs[i+2]) < 0.001:
                conv_idx = i
                break
            
        if conv_idx is not None and conv_idx > 5:  # Ensure we don't mark very early iterations
            fig.add_vline(
                x=iterations[conv_idx],
                line_dash="dash",
                line_color="green",
                annotation_text="Approximate Convergence",
                annotation_position="top right"
            )
    
    title = f"Convergence Plot"
    if learning_rate is not None:
        title += f" (Learning Rate: {learning_rate})"
    if algorithm is not None:
        title += f" - {algorithm}"
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title='Iteration',
        yaxis_title='Loss',
        width=700,
        height=400
    )
    
    return fig

def plot_regression_with_line(X, y, coef, intercept, iteration=None):
    """Plot regression data with fitted line"""
    
    # Sort for better visualization
    sort_idx = np.argsort(X.flatten())
    X_sorted = X.flatten()[sort_idx]
    y_sorted = y[sort_idx]
    
    # Generate predictions
    y_pred = linear_regression_predict(X_sorted, coef, intercept)
    
    fig = go.Figure()
    
    # Plot data points
    fig.add_trace(go.Scatter(
        x=X_sorted,
        y=y_sorted,
        mode='markers',
        name='Data Points',
        marker=dict(color='#232F3E')
    ))
    
    # Plot regression line
    fig.add_trace(go.Scatter(
        x=X_sorted,
        y=y_pred,
        mode='lines',
        name='Regression Line',
        line=dict(color='#FF9900', width=3)
    ))
    
    title = "Linear Regression Fit"
    if iteration is not None:
        title += f" (Iteration {iteration})"
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title='X',
        yaxis_title='y',
        width=700,
        height=400
    )
    
    return fig

def visualize_gradient_descent_step(X, y, coef, intercept, learning_rate=0.01, iteration=None):
    """Create visualization of a gradient descent step"""
    
    # Sort for better visualization
    X_flat = X.flatten()
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    y_sorted = y[sort_idx]
    
    # Current predictions
    y_pred = linear_regression_predict(X_sorted, coef, intercept)
    
    # Calculate gradients
    errors = y_sorted - y_pred
    grad_coef = -2 * np.mean(X_sorted * errors)
    grad_intercept = -2 * np.mean(errors)
    
    # New parameters after the update
    new_coef = coef - learning_rate * grad_coef
    new_intercept = intercept - learning_rate * grad_intercept
    
    # New predictions
    new_y_pred = linear_regression_predict(X_sorted, new_coef, new_intercept)
    
    # Calculate loss before and after
    loss_before = mse_loss(y_sorted, y_pred)
    loss_after = mse_loss(y_sorted, new_y_pred)
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Parameter Space", "Data Space")
    )
    
    # Parameter space plot (simplified 2D version - intercept vs coef)
    # Create a grid for contour plot
    coef_range = np.linspace(coef-2, coef+2, 50)
    intercept_range = np.linspace(intercept-2, intercept+2, 50)
    coef_grid, intercept_grid = np.meshgrid(coef_range, intercept_range)
    loss_grid = np.zeros_like(coef_grid)
    
    for i in range(coef_grid.shape[0]):
        for j in range(coef_grid.shape[1]):
            y_pred_grid = linear_regression_predict(X_sorted, coef_grid[i, j], intercept_grid[i, j])
            loss_grid[i, j] = mse_loss(y_sorted, y_pred_grid)
    
    # Contour plot of loss surface
    fig.add_trace(
        go.Contour(
            z=loss_grid,
            x=coef_range,
            y=intercept_range,
            colorscale='YlOrRd',
            showscale=False,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10)
            )
        ),
        row=1, col=1
    )
    
    # Current position
    fig.add_trace(
        go.Scatter(
            x=[coef], 
            y=[intercept],
            mode='markers',
            marker=dict(size=10, color='#232F3E'),
            name='Current Parameters'
        ),
        row=1, col=1
    )
    
    # Vector showing gradient direction
    fig.add_trace(
        go.Scatter(
            x=[coef, coef - learning_rate * grad_coef],
            y=[intercept, intercept - learning_rate * grad_intercept],
            mode='lines+markers',
            line=dict(color='#FF9900', width=2, dash='dot'),
            marker=dict(size=8, color='#FF9900'),
            name='Gradient Step'
        ),
        row=1, col=1
    )
    
    # Data space plot
    # Original data points
    fig.add_trace(
        go.Scatter(
            x=X_sorted,
            y=y_sorted,
            mode='markers',
            name='Data Points',
            marker=dict(color='#232F3E')
        ),
        row=1, col=2
    )
    
    # Current regression line
    fig.add_trace(
        go.Scatter(
            x=X_sorted,
            y=y_pred,
            mode='lines',
            name='Current Line',
            line=dict(color='#232F3E', width=3)
        ),
        row=1, col=2
    )
    
    # New regression line
    fig.add_trace(
        go.Scatter(
            x=X_sorted,
            y=new_y_pred,
            mode='lines',
            name='Updated Line',
            line=dict(color='#FF9900', width=3)
        ),
        row=1, col=2
    )
    
    title = "Gradient Descent Step"
    if iteration is not None:
        title += f" (Iteration {iteration})"
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        width=900,
        height=450,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Coefficient", row=1, col=1)
    fig.update_yaxes(title_text="Intercept", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=2)
    
    fig.add_annotation(
        text=f"Loss: {loss_before:.4f} → {loss_after:.4f}",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=14)
    )
    
    return fig, new_coef, new_intercept, loss_after

def visualize_gradient_descent_steps_3d(X, y, history, step_size=5):
    """Create 3D visualization of gradient descent steps"""
    
    # Prepare parameter grid for loss surface
    coef_min, coef_max = min(history['coef'])-0.5, max(history['coef'])+0.5
    intercept_min, intercept_max = min(history['intercept'])-0.5, max(history['intercept'])+0.5
    
    coef_range = np.linspace(coef_min, coef_max, 30)
    intercept_range = np.linspace(intercept_min, intercept_max, 30)
    coef_grid, intercept_grid = np.meshgrid(coef_range, intercept_range)
    loss_grid = np.zeros_like(coef_grid)
    
    # Flatten X for simpler operations
    X_flat = X.flatten()
    
    # Calculate loss at each point in the grid
    for i in range(coef_grid.shape[0]):
        for j in range(coef_grid.shape[1]):
            y_pred = linear_regression_predict(X_flat, coef_grid[i, j], intercept_grid[i, j])
            loss_grid[i, j] = mse_loss(y, y_pred)
    
    # Create 3D surface plot
    fig = go.Figure()
    
    # Loss surface
    fig.add_trace(go.Surface(
        x=coef_grid,
        y=intercept_grid,
        z=loss_grid,
        colorscale='YlOrRd',
        opacity=0.8,
        showscale=False
    ))
    
    # Gradient descent path
    # Subsample for cleaner visualization
    indices = np.arange(0, len(history['coef']), step_size)
    if indices[-1] != len(history['coef'])-1:  # Add last point if not already included
        indices = np.append(indices, len(history['coef'])-1)
    
    fig.add_trace(go.Scatter3d(
        x=history['coef'][indices],
        y=history['intercept'][indices],
        z=history['loss'][indices],
        mode='lines+markers',
        line=dict(color='#232F3E', width=5),
        marker=dict(size=5, color='#FF9900'),
        name='Gradient Descent Path'
    ))
    
    # Add starting point
    fig.add_trace(go.Scatter3d(
        x=[history['coef'][0]],
        y=[history['intercept'][0]],
        z=[history['loss'][0]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Starting Point'
    ))
    
    # Add final point
    fig.add_trace(go.Scatter3d(
        x=[history['coef'][-1]],
        y=[history['intercept'][-1]],
        z=[history['loss'][-1]],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Final Point'
    ))
    
    fig.update_layout(
        title="Gradient Descent Path in 3D Loss Surface",
        title_x=0.5,
        scene=dict(
            xaxis_title='Coefficient',
            yaxis_title='Intercept',
            zaxis_title='Loss',
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.9)
            )
        ),
        width=700,
        height=600
    )
    
    return fig

def visualize_learning_rate_comparison():
    """Create visualization comparing different learning rates"""
    
    # Generate synthetic data
    np.random.seed(42)
    X, y = generate_regression_data(n_samples=50, noise=0.5)
    
    # Define learning rates to compare
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    max_iterations = 100
    
    # Run gradient descent with each learning rate
    results = {}
    
    for lr in learning_rates:
        coef, intercept, history = gradient_descent_linear(
            X, y, learning_rate=lr, n_iterations=max_iterations
        )
        results[lr] = {
            'coef': coef,
            'intercept': intercept,
            'history': history
        }
    
    # Create plot
    fig = go.Figure()
    
    for lr in learning_rates:
        fig.add_trace(go.Scatter(
            x=results[lr]['history']['iterations'],
            y=results[lr]['history']['loss'],
            mode='lines',
            name=f'LR = {lr}'
        ))
    
    fig.update_layout(
        title="Convergence with Different Learning Rates",
        title_x=0.5,
        xaxis_title='Iteration',
        yaxis_title='Loss',
        width=700,
        height=400
    )
    
    return fig

# Main content with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Introduction", 
    "📉 Loss Functions", 
    "🎯 Convergence", 
    "⬇️ Gradient Descent",
    "❓ Knowledge Check"
])

# Introduction Tab
with tab1:
    st.title("Model Optimization Techniques in Machine Learning")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Welcome to the Interactive Model Optimization Course!</h3>
        <p>In this interactive e-learning module, you'll learn about essential techniques 
        for optimizing machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3,2])
    
    with col1:
        st.markdown("""
        ### What You'll Learn
        
        In this module, you'll explore:
        
        1. **Loss Functions**: How models measure prediction errors
        2. **Convergence**: When to stop optimizing your model
        3. **Gradient Descent**: How models learn from data
        
        Each section includes interactive examples and visualizations that allow you to explore these concepts in depth.
        
        ### Why Model Optimization Matters
        
        Optimization is the heart of machine learning:
        
        - It's how models find patterns in data
        - It determines how well your model will perform
        - It affects training time and computational efficiency
        - It can mean the difference between a model that works and one that fails
        
        Let's dive into the fundamental concepts that make model optimization work!
        """)
        
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*SKHGhoGKnBh_GPNX6Vzqaw.png", caption="Model Optimization Flow")
        
        st.markdown("""
        ### How to Use This Module
        
        1. Navigate through the tabs to explore different optimization concepts
        2. Interact with the visualizations to understand the concepts better
        3. Test your knowledge with the quiz in the last section
        
        <p class="highlight">Tip: Try adjusting parameters in the interactive visualizations to see how optimization behaves in different scenarios!</p>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Optimization in the Machine Learning Workflow
    
    Model optimization is a crucial step in the machine learning development process:
    
    ```mermaid
    graph LR
        A[Data Collection] --> B[Data Preprocessing]
        B --> C[Model Selection]
        C --> D[Model Training & Optimization]
        D --> E[Model Evaluation]
        E --> F[Deployment]
        E --> D
    ```
    
    In this module, we'll focus on the **Model Training & Optimization** phase, which includes:
    
    1. Selecting appropriate loss functions
    2. Understanding convergence criteria
    3. Implementing gradient descent algorithms
    
    Let's get started!
    """)
    
    st.markdown("---")
    
    st.subheader("Preview of Key Concepts")
    
    preview_col1, preview_col2 = st.columns(2)
    
    with preview_col1:
        loss_preview = plot_loss_function("MSE", x_range=(-3, 3))
        st.plotly_chart(loss_preview, use_container_width=True)
    
    with preview_col2:
        # Simple convergence data for preview
        iterations = np.arange(20)
        loss_values = 5 * np.exp(-0.2 * iterations) + 0.5 + 0.2 * np.random.random(20)
        
        convergence_preview = plot_convergence(iterations, loss_values)
        st.plotly_chart(convergence_preview, use_container_width=True)

# Loss Functions Tab
with tab2:
    st.title("Loss Functions")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What are Loss Functions?</h3>
        <p>Loss functions quantify how well a model's predictions match the true values. They are the mathematical 
        functions we seek to minimize during model training.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Understanding Loss Functions
        
        A loss function serves as a "compass" that guides model optimization:
        
        - **Measures prediction error**: Quantifies the difference between predicted and actual values
        - **Provides optimization objective**: The goal is to minimize this function
        - **Influences model behavior**: Different loss functions emphasize different aspects of the predictions
        
        ### Key Properties of Good Loss Functions
        
        1. **Differentiable**: Should have well-defined gradients for optimization
        2. **Convex**: Ideally has a single global minimum to ensure convergence
        3. **Scale-appropriate**: Should match the scale and distribution of your target variable
        4. **Robust**: Should handle outliers appropriately for your use case
        
        ### Common Loss Functions
        
        Select a loss function below to visualize its behavior:
        """)
        
        loss_function_options = ["MSE", "MAE", "Log Loss", "Huber"]
        selected_loss = st.selectbox(
            "Select a loss function:",
            loss_function_options,
            index=loss_function_options.index(st.session_state.loss_function_type)
        )
        
        st.session_state.loss_function_type = selected_loss
        
    with col2:
        loss_viz = plot_loss_function(selected_loss)
        st.plotly_chart(loss_viz)
        
        # Show a data table with loss values at different error levels
        error_points = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
        y_true = 0  # 'correct' value
        
        if selected_loss == "MSE":
            loss_vals = [(x - y_true)**2 for x in error_points]
        elif selected_loss == "MAE":
            loss_vals = [abs(x - y_true) for x in error_points]
        elif selected_loss == "Log Loss":
            # Use sigmoid to constrain values between 0 and 1
            sigmoid_vals = 1 / (1 + np.exp(-np.array(error_points)))
            # True value is 1 for positive error_points, 0 for negative
            y_true_binary = (np.array(error_points) > 0).astype(int)
            loss_vals = [-y * np.log(max(y_hat, 1e-15)) - (1-y) * np.log(max(1-y_hat, 1e-15)) 
                         for y, y_hat in zip(y_true_binary, sigmoid_vals)]
        elif selected_loss == "Huber":
            delta = 1.0
            loss_vals = [
                0.5 * (x - y_true)**2 if abs(x - y_true) <= delta 
                else delta * (abs(x - y_true) - 0.5 * delta)
                for x in error_points
            ]
        
        loss_df = pd.DataFrame({
            'Error': error_points,
            'Loss Value': loss_vals
        })
        
        st.dataframe(loss_df.style.highlight_min(subset=['Loss Value']))
    
    st.markdown("""
    ### Loss Function Comparison
    
    Each loss function has unique properties that make it suitable for different scenarios:
    """)
    
    comparison_data = {
        'Loss Function': ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'Log Loss (Binary Cross-Entropy)', 'Huber Loss'],
        'Formula': [
            '$$\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$',
            '$$\\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$',
            '$$-\\frac{1}{n}\\sum_{i=1}^{n}[y_i\\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i)]$$',
            '$$\\begin{cases} \\frac{1}{2}(y_i - \\hat{y}_i)^2 & \\text{for } |y_i - \\hat{y}_i| \\leq \\delta \\\\ \\delta(|y_i - \\hat{y}_i| - \\frac{1}{2}\\delta) & \\text{otherwise} \\end{cases}$$'
        ],
        'Use Case': [
            'Regression when outliers are rare',
            'Regression when outliers are common',
            'Binary classification',
            'Regression with robustness to outliers'
        ],
        'Pros': [
            'Differentiable, penalizes large errors heavily',
            'Robust to outliers, intuitive interpretation',
            'Proper scoring rule for probabilities',
            'Combines benefits of MSE and MAE'
        ],
        'Cons': [
            'Sensitive to outliers',
            'Not differentiable at zero, can be unstable',
            'Undefined for predictions of exactly 0 or 1',
            'Requires tuning of delta parameter'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.markdown(comparison_df.to_markdown(index=False))
    
    st.markdown("""
    ### Interactive Loss Visualization
    
    See how different loss functions behave with real data:
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Generate synthetic data for visualization
        n_samples = st.slider("Number of data points:", min_value=10, max_value=200, value=50, step=10)
        noise_level = st.slider("Noise level:", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        add_outliers = st.checkbox("Add outliers", value=False)
        
        np.random.seed(42)  # For reproducibility
        X, y = generate_regression_data(n_samples=n_samples, noise=noise_level)
        
        if add_outliers:
            # Add a few outliers
            outlier_idx = np.random.choice(len(y), size=max(1, int(0.05 * len(y))), replace=False)
            y[outlier_idx] += np.random.choice([-1, 1], size=len(outlier_idx)) * np.random.uniform(5, 10, size=len(outlier_idx))
    
    # Create linear regression model for each loss function
    X_flat = X.flatten().reshape(-1, 1)  # Reshape for sklearn
    
    # MSE - Using LinearRegression which minimizes MSE
    lr_mse = LinearRegression().fit(X_flat, y)
    y_pred_mse = lr_mse.predict(X_flat)
    mse_val = mean_squared_error(y, y_pred_mse)
    
    # MAE - Using linear model with different solver
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import SGDRegressor
    
    lr_mae = SGDRegressor(loss='epsilon_insensitive', epsilon=0, 
                           max_iter=1000, tol=1e-3, 
                           random_state=42).fit(X_flat, y)
    y_pred_mae = lr_mae.predict(X_flat)
    mae_val = mae_loss(y, y_pred_mae)
    
    # Huber - Using SGDRegressor with huber loss
    lr_huber = SGDRegressor(loss='huber', epsilon=1.35, 
                             max_iter=1000, tol=1e-3, 
                             random_state=42).fit(X_flat, y)
    y_pred_huber = lr_huber.predict(X_flat)
    huber_val = huber_loss(y, y_pred_huber)
    
    with col2:
        # Sort for visualization
        sort_idx = np.argsort(X_flat.flatten())
        X_sorted = X_flat.flatten()[sort_idx]
        y_sorted = y[sort_idx]
        y_pred_mse_sorted = y_pred_mse[sort_idx]
        y_pred_mae_sorted = y_pred_mae[sort_idx]
        y_pred_huber_sorted = y_pred_huber[sort_idx]
        
        fig = go.Figure()
        
        # Add data points
        fig.add_trace(go.Scatter(
            x=X_sorted,
            y=y_sorted,
            mode='markers',
            name='Data Points',
            marker=dict(color='#232F3E')
        ))
        
        # Add regression lines
        fig.add_trace(go.Scatter(
            x=X_sorted,
            y=y_pred_mse_sorted,
            mode='lines',
            name=f'MSE Fit (loss={mse_val:.3f})',
            line=dict(color='#FF9900', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=X_sorted,
            y=y_pred_mae_sorted,
            mode='lines',
            name=f'MAE Fit (loss={mae_val:.3f})',
            line=dict(color='#1E88E5', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=X_sorted,
            y=y_pred_huber_sorted,
            mode='lines',
            name=f'Huber Fit (loss={huber_val:.3f})',
            line=dict(color='#D81B60', width=2)
        ))
        
        fig.update_layout(
            title="Different Loss Functions' Impact on Linear Regression",
            title_x=0.5,
            xaxis_title='X',
            yaxis_title='y',
            width=700,
            height=500
        )
        
        st.plotly_chart(fig)
    
    st.markdown("""
    ### Key Takeaways
    
    Notice from the interactive visualization:
    
    1. **MSE (Mean Squared Error)**:
       - Most sensitive to outliers
       - Gives larger penalties to larger errors
       - Often preferred for regression problems where outliers are legitimate data that shouldn't be ignored
    
    2. **MAE (Mean Absolute Error)**:
       - More robust to outliers
       - All errors are treated with importance proportional to their magnitude
       - Useful when outliers shouldn't significantly influence the model
    
    3. **Huber Loss**:
       - Hybrid approach that combines MSE and MAE
       - MSE-like for small errors, MAE-like for large errors
       - Provides robustness while maintaining differentiability
    
    4. **Log Loss** (not shown in regression example):
       - Designed for classification problems
       - Severely penalizes confident but wrong predictions
       - Optimal for probabilistic outputs
    """)
    
    st.markdown('<p class="highlight">The choice of loss function should be guided by the characteristics of your data and the specific goals of your model.</p>', unsafe_allow_html=True)

# Convergence Tab
with tab3:
    st.title("Convergence")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is Convergence?</h3>
        <p>Convergence refers to the point at which an optimization algorithm reaches (or approaches) 
        the optimal solution, and further iterations yield diminishing returns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Understanding Convergence
        
        In machine learning, convergence indicates that your model has:
        
        - Found a (local or global) minimum of the loss function
        - Reached a point where parameters change very little with additional training
        - Achieved a stable performance level
        
        ### Why Convergence Matters
        
        Proper convergence is essential for:
        
        1. **Training efficiency**: Avoid unnecessary computation
        2. **Model performance**: Ensure the model is sufficiently optimized
        3. **Preventing overfitting**: Stop training before memorizing noise in the data
        4. **Numerical stability**: Avoid issues from extremely small gradients
        
        ### Common Convergence Criteria
        
        - **Loss threshold**: Stop when loss falls below a predefined value
        - **Loss improvement**: Stop when improvement between iterations is small
        - **Parameter change**: Stop when parameters barely change
        - **Fixed iterations**: Stop after a predefined number of iterations
        - **Validation performance**: Stop when validation performance stops improving
        """)
    
    with col2:
        # Generate convergence visualization data
        if st.session_state.convergence_data is None:
            # Sample convergence patterns for demonstration
            iterations = np.arange(50)
            
            # Fast convergence
            fast_loss = 10 * np.exp(-0.3 * iterations) + 0.5 + 0.05 * np.random.random(50)
            
            # Slow convergence
            slow_loss = 10 * np.exp(-0.05 * iterations) + 1 + 0.1 * np.random.random(50)
            
            # Oscillating convergence
            base_loss = 10 * np.exp(-0.1 * iterations) + 1
            osc_loss = base_loss + 1 * np.sin(iterations/2) * np.exp(-0.05 * iterations)
            
            # Non-convergence/divergence
            div_loss = 1 + 0.1 * iterations + 5 * np.random.random(50)
            
            st.session_state.convergence_data = {
                "Fast Convergence": fast_loss,
                "Slow Convergence": slow_loss,
                "Oscillating Convergence": osc_loss,
                "Non-convergence": div_loss
            }
        
        # Select convergence pattern to visualize
        convergence_pattern = st.selectbox(
            "Select a convergence pattern:",
            list(st.session_state.convergence_data.keys())
        )
        
        selected_data = st.session_state.convergence_data[convergence_pattern]
        
        # Plot the convergence pattern
        conv_fig = plot_convergence(np.arange(len(selected_data)), selected_data, algorithm=convergence_pattern)
        st.plotly_chart(conv_fig)
    
    st.markdown("### Convergence Challenges")
    
    challenges_col1, challenges_col2, challenges_col3 = st.columns(3)
    
    with challenges_col1:
        st.markdown("""
        #### 1. Learning Rate Issues
        
        - **Too high**: Causes overshooting and oscillation
        - **Too low**: Results in slow convergence
        - **Solution**: Learning rate scheduling or adaptive rates
        
        ![Learning Rate](https://miro.medium.com/v2/resize:fit:1400/1*XVFmo9NxLnwDr3SxzKy-rA.gif)
        """)
    
    with challenges_col2:
        st.markdown("""
        #### 2. Saddle Points & Local Minima
        
        - **Local minima**: Algorithm gets stuck in suboptimal solution
        - **Saddle points**: Areas where gradient is zero but not a minimum
        - **Solution**: Momentum, random restarts, or advanced optimizers
        
        ![Saddle Points](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ZC9qItK9wI0F6BwSVYMQGg.png)
        """)
    
    with challenges_col3:
        st.markdown("""
        #### 3. Noisy Gradients
        
        - **Issue**: Batch/stochastic methods introduce variance
        - **Result**: Erratic convergence path
        - **Solution**: Larger batch sizes, gradient clipping, or noise reduction techniques
        
        ![Noisy Gradients](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*e_Iw6AxcCyTwAx2sUKRjig.png)
        """)
    
    st.markdown("### Interactive Convergence Explorer")
    
    st.markdown("""
    Explore how different factors affect convergence in linear regression:
    """)
    
    explorer_col1, explorer_col2 = st.columns([1, 2])
    
    with explorer_col1:
        # Parameters for the exploration
        n_points = st.slider("Number of data points:", 20, 200, 50, 10)
        noise_level = st.slider("Noise level:", 0.1, 5.0, 1.0, 0.1)
        
        learning_rate_options = {
            "Very Small (0.001)": 0.001,
            "Small (0.01)": 0.01, 
            "Medium (0.05)": 0.05,
            "Large (0.1)": 0.1,
            "Very Large (0.5)": 0.5
        }
        
        lr_selection = st.selectbox(
            "Learning rate:",
            options=list(learning_rate_options.keys()),
            index=1
        )
        learning_rate = learning_rate_options[lr_selection]
        
        max_iter = st.slider("Maximum iterations:", 10, 500, 100, 10)
        
        run_button = st.button("Run Optimization")
    
    # Run linear regression with gradient descent
    if run_button or 'convergence_X' not in st.session_state:
        # Generate data
        np.random.seed(42)
        X, y = generate_regression_data(n_samples=n_points, noise=noise_level)
        
        # Run gradient descent
        coef, intercept, history = gradient_descent_linear(
            X, y, learning_rate=learning_rate, n_iterations=max_iter, tolerance=1e-6
        )
        
        # Store results
        st.session_state.convergence_X = X
        st.session_state.convergence_y = y
        st.session_state.convergence_coef = coef
        st.session_state.convergence_intercept = intercept
        st.session_state.convergence_history = history
    
    with explorer_col2:
        # Plot convergence
        conv_fig = plot_convergence(
            st.session_state.convergence_history['iterations'],
            st.session_state.convergence_history['loss'],
            learning_rate=learning_rate,
            algorithm="Gradient Descent"
        )
        st.plotly_chart(conv_fig)
        
        # Statistics about convergence
        n_iter = len(st.session_state.convergence_history['iterations'])
        final_loss = st.session_state.convergence_history['loss'][-1]
        
        if n_iter >= 2:
            # Calculate loss improvements
            loss_improvements = np.diff(st.session_state.convergence_history['loss'])
            avg_improvement = np.mean(np.abs(loss_improvements))
            final_improvement = np.abs(loss_improvements[-1])
        else:
            avg_improvement = "N/A"
            final_improvement = "N/A"
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Iterations", f"{n_iter}")
        
        with stats_col2:
            st.metric("Final Loss", f"{final_loss:.4f}")
        
        with stats_col3:
            if isinstance(final_improvement, str):
                st.metric("Final Improvement", final_improvement)
            else:
                st.metric("Final Improvement", f"{final_improvement:.6f}")
    
    # Show model fit
    st.subheader("Resulting Model Fit")
    
    fit_fig = plot_regression_with_line(
        st.session_state.convergence_X,
        st.session_state.convergence_y,
        st.session_state.convergence_coef,
        st.session_state.convergence_intercept
    )
    st.plotly_chart(fit_fig)
    
    st.markdown("""
    ### Learning Rate Impact on Convergence
    
    The learning rate is perhaps the most critical hyperparameter affecting convergence:
    """)
    
    lr_comparison_fig = visualize_learning_rate_comparison()
    st.plotly_chart(lr_comparison_fig)
    
    st.markdown("""
    ### Key Takeaways
    
    1. **Convergence speed vs stability trade-off**:
       - Faster convergence often comes with stability risks
       - More stable convergence often requires more iterations
    
    2. **Data characteristics matter**:
       - Noisier data typically leads to less smooth convergence
       - More complex relationships require more iterations
    
    3. **Monitor multiple signals for convergence**:
       - Loss value
       - Parameter changes
       - Gradient magnitude
       - Validation performance
    
    4. **Practical approach**:
       - Start with conservative learning rates and increase if needed
       - Use early stopping based on validation performance
       - Consider adaptive learning rate methods for complex problems
    """)
    
    st.markdown('<p class="highlight">Remember: The goal is not perfect convergence, but finding a solution that generalizes well to new data!</p>', unsafe_allow_html=True)

# Gradient Descent Tab
with tab4:
    st.title("Gradient Descent")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is Gradient Descent?</h3>
        <p>Gradient descent is an iterative optimization algorithm for finding the minimum of a function. 
        It works by taking steps proportional to the negative of the gradient at the current point.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Understanding Gradient Descent
        
        Gradient descent is the fundamental optimization algorithm in machine learning:
        
        1. **Start** with initial parameter values
        2. **Calculate the gradient** (direction of steepest ascent)
        3. **Move in the opposite direction** (steepest descent)
        4. **Repeat** until convergence
        
        The update rule is:
        
        <div class="formula-box">
        $$ \\theta_{new} = \\theta_{old} - \\alpha \\nabla J(\\theta) $$
        </div>
        
        Where:
        - $\\theta$ represents model parameters
        - $\\alpha$ is the learning rate
        - $\\nabla J(\\theta)$ is the gradient of the loss function
        
        ### Variants of Gradient Descent
        
        - **Batch Gradient Descent**: Uses entire dataset for each update
        - **Stochastic Gradient Descent (SGD)**: Uses single example per update
        - **Mini-batch Gradient Descent**: Uses a small batch of examples
        
        ### Advanced Optimizers
        
        Modern machine learning uses enhanced versions:
        - **Momentum**: Adds "inertia" to updates
        - **AdaGrad/RMSProp**: Adaptive learning rates per parameter
        - **Adam**: Combines momentum and adaptive learning rates
        """)
    
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*f9a162GhpMbiTVTAua_lLQ.png", caption="Gradient Descent Visualization")
        
        st.markdown("""
        ### Mathematical Intuition
        
        The gradient tells us:
        - **Direction** of steepest increase
        - **Magnitude** of the slope
        
        By moving in the opposite direction:
        - We follow the path of steepest decrease
        - We take larger steps when the slope is steeper
        - We take smaller steps as we approach the minimum
        
        The learning rate controls step size:
        - Too small: slow convergence
        - Too large: overshooting/divergence
        """)
    
    st.markdown("### Interactive Gradient Descent")
    
    st.markdown("""
    Watch gradient descent in action! This visualization shows how parameters update with each step:
    """)
    
    # Interactive gradient descent visualization
    gd_col1, gd_col2 = st.columns([1, 3])
    
    with gd_col1:
        # Generate data if needed
        if 'gd_data_generated' not in st.session_state or not st.session_state.gd_data_generated:
            np.random.seed(42)
            X_gd, y_gd = generate_regression_data(n_samples=30, noise=1.0)
            st.session_state.X_gd = X_gd
            st.session_state.y_gd = y_gd
            st.session_state.gd_data_generated = True
        
        # Parameters for GD
        gd_lr = st.slider(
            "Learning rate:",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.001,
            format="%.3f",
            key="gd_lr_slider"
        )
        
        if 'gd_coef' not in st.session_state or 'gd_intercept' not in st.session_state:
            # Initialize with random parameters
            st.session_state.gd_coef = np.random.randn()
            st.session_state.gd_intercept = np.random.randn()
            st.session_state.gd_iterations = []
            st.session_state.gd_loss = []
        
        # Button to perform a single GD step
        if st.button("Perform Step"):
            # Visualize the step
            gd_viz, new_coef, new_intercept, new_loss = visualize_gradient_descent_step(
                st.session_state.X_gd,
                st.session_state.y_gd,
                st.session_state.gd_coef,
                st.session_state.gd_intercept,
                learning_rate=gd_lr,
                iteration=len(st.session_state.gd_iterations) + 1
            )
            
            # Update parameters
            st.session_state.gd_coef = new_coef
            st.session_state.gd_intercept = new_intercept
            
            # Add to history
            if len(st.session_state.gd_iterations) == 0:
                # Add initial loss
                y_pred_init = linear_regression_predict(st.session_state.X_gd.flatten(), st.session_state.gd_coef, st.session_state.gd_intercept)
                init_loss = mse_loss(st.session_state.y_gd, y_pred_init)
                st.session_state.gd_iterations.append(0)
                st.session_state.gd_loss.append(init_loss)
            
            st.session_state.gd_iterations.append(len(st.session_state.gd_iterations))
            st.session_state.gd_loss.append(new_loss)
            
            # Store the visualization
            st.session_state.gd_step_viz = gd_viz
        
        # Button to reset GD
        if st.button("Reset Optimization"):
            # Reset parameters
            st.session_state.gd_coef = np.random.randn()
            st.session_state.gd_intercept = np.random.randn()
            st.session_state.gd_iterations = []
            st.session_state.gd_loss = []
            if 'gd_step_viz' in st.session_state:
                del st.session_state.gd_step_viz
        
        # Show current parameters
        st.write(f"**Current Parameters:**")
        st.write(f"Coefficient: {st.session_state.gd_coef:.4f}")
        st.write(f"Intercept: {st.session_state.gd_intercept:.4f}")
        
        # Show loss curve if we have iterations
        if len(st.session_state.gd_iterations) > 1:
            loss_fig = plot_convergence(
                st.session_state.gd_iterations,
                st.session_state.gd_loss,
                learning_rate=gd_lr
            )
            st.plotly_chart(loss_fig)
    
    with gd_col2:
        if 'gd_step_viz' in st.session_state:
            # Show the step visualization
            st.plotly_chart(st.session_state.gd_step_viz, use_container_width=True)
        else:
            # Initial visualization
            initial_viz, _, _, _ = visualize_gradient_descent_step(
                st.session_state.X_gd,
                st.session_state.y_gd,
                st.session_state.gd_coef,
                st.session_state.gd_intercept,
                learning_rate=gd_lr,
                iteration=0
            )
            st.plotly_chart(initial_viz, use_container_width=True)
    
    st.markdown("### Full Gradient Descent in 3D")
    
    st.markdown("""
    This visualization shows the complete gradient descent path in the loss surface:
    """)
    
    # Full GD run with 3D visualization
    if st.button("Run Full Gradient Descent"):
        with st.spinner("Running gradient descent..."):
            # Generate new data
            np.random.seed(int(time.time()))
            X_full, y_full = generate_regression_data(n_samples=50, noise=1.0)
            
            # Run full gradient descent
            coef, intercept, history = gradient_descent_linear(
                X_full, y_full, 
                learning_rate=0.05, 
                n_iterations=100
            )
            
            # Store in session state
            st.session_state.gd_history = history
            st.session_state.gd_X_full = X_full
            st.session_state.gd_y_full = y_full
            
    if st.session_state.gd_history is not None:
        # Create 3D visualization
        viz_3d = visualize_gradient_descent_steps_3d(
            st.session_state.gd_X_full, 
            st.session_state.gd_y_full,
            st.session_state.gd_history,
            step_size=5
        )
        st.plotly_chart(viz_3d)
        
        st.markdown("""
        **Interpretation of the 3D visualization:**
        
        - The surface represents the loss landscape
        - The blue line shows the path taken by gradient descent
        - The red point is the starting position
        - The green point is where the algorithm converged
        
        Notice how the algorithm follows the steepest path downhill to find the minimum!
        """)
    
    st.markdown("""
    ### Practical Tips for Gradient Descent
    
    1. **Data preprocessing is crucial**:
       - Scale features to similar ranges
       - Normalize data when possible
    
    2. **Learning rate selection**:
       - Start small and increase if convergence is too slow
       - Decrease if the loss oscillates or diverges
       - Consider learning rate schedules
    
    3. **Initialization matters**:
       - Random initialization helps break symmetry
       - Advanced initialization methods (Xavier/He) help with deep networks
    
    4. **Batch size considerations**:
       - Smaller batches: faster updates but noisier gradients
       - Larger batches: more stable but computationally expensive
    
    5. **Consider advanced optimizers**:
       - RMSProp, Adam, or AdamW often converge faster than basic SGD
       - Different optimizers work better for different problems
    """)
    
    st.markdown('<p class="highlight">Remember: No single optimizer or learning rate works best for all problems. Experimentation is key!</p>', unsafe_allow_html=True)

# Knowledge Check Tab
with tab5:
    st.title("Knowledge Check")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Test Your Understanding</h3>
        <p>Answer these five questions to check your understanding of model optimization techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quiz questions
    quiz = [
        {
            "question": "Which loss function is most sensitive to outliers?",
            "options": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Huber Loss", "Log Loss"],
            "answer": "Mean Squared Error (MSE)"
        },
        {
            "question": "In the context of gradient descent, what does the learning rate control?",
            "options": [
                "The direction in which parameters are updated",
                "The size of each update step",
                "The number of iterations required for convergence",
                "The batch size used for each iteration"
            ],
            "answer": "The size of each update step"
        },
        {
            "question": "Which of the following is NOT a common criterion for convergence?",
            "options": [
                "The loss falls below a predefined threshold",
                "The change in loss between iterations is very small",
                "A fixed number of iterations has been reached",
                "The gradient becomes exactly zero"
            ],
            "answer": "The gradient becomes exactly zero"
        },
        {
            "question": "What happens if the learning rate is set too high in gradient descent?",
            "options": [
                "The algorithm will converge very slowly",
                "The algorithm may oscillate or diverge",
                "The algorithm will get stuck in local minima",
                "The algorithm will require more memory"
            ],
            "answer": "The algorithm may oscillate or diverge"
        },
        {
            "question": "Which of the following optimizers combines the benefits of momentum and adaptive learning rates?",
            "options": [
                "Stochastic Gradient Descent (SGD)",
                "RMSProp",
                "AdaGrad",
                "Adam"
            ],
            "answer": "Adam"
        }
    ]
    
    # Quiz logic
    if not st.session_state.quiz_submitted:
        st.markdown("### Answer the following questions:")
        
        for i, q in enumerate(quiz):
            st.markdown(f"**Question {i+1}**: {q['question']}")
            st.session_state.quiz_answers[i] = st.radio(
                f"Select your answer for question {i+1}:",
                q['options'],index=None,
                key=f"q{i}"
            )
            st.markdown("---")
        
        if st.button("Submit Answers"):
            score = 0
            for i, q in enumerate(quiz):
                if st.session_state.quiz_answers[i] == q['answer']:
                    score += 1
            
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True
            st.rerun()
    
    else:
        # Show results
        st.markdown(f"### Your Score: {st.session_state.quiz_score}/{len(quiz)}")
        
        # Create gauge chart for score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state.quiz_score / len(quiz) * 100,
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
        
        # Show detailed results
        st.markdown("### Detailed Results:")
        
        for i, q in enumerate(quiz):
            user_answer = st.session_state.quiz_answers[i]
            correct_answer = q['answer']
            is_correct = user_answer == correct_answer
            
            if is_correct:
                st.markdown(f"**Question {i+1}**: ✅ Correct!")
            else:
                st.markdown(f"**Question {i+1}**: ❌ Incorrect")
                st.markdown(f"Your answer: {user_answer}")
                st.markdown(f"Correct answer: {correct_answer}")
            
            st.markdown(f"*{q['question']}*")
            st.markdown("---")
        
        if st.session_state.quiz_score == len(quiz):
            st.success("🎉 Perfect score! You've mastered model optimization concepts!")
        elif st.session_state.quiz_score >= len(quiz) * 0.8:
            st.success("🌟 Great job! You have a strong understanding of model optimization!")
        elif st.session_state.quiz_score >= len(quiz) * 0.6:
            st.info("👍 Good effort! Review the concepts you missed to strengthen your understanding.")
        else:
            st.warning("📚 You may need more practice. Try reviewing the material again.")
        
        if st.button("Retake Quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score = 0
            st.rerun()

# Summary section
st.markdown("""
## Summary of Model Optimization Techniques

| Technique | Description | Key Considerations |
|-----------|-------------|-------------------|
| Loss Functions | Mathematical functions that quantify prediction errors | Choose based on problem type, sensitivity to outliers, and scale properties |
| Convergence | When an optimization algorithm reaches its optimal solution | Monitor loss, parameter changes, and use appropriate stopping criteria |
| Gradient Descent | Iterative optimization algorithm that follows the negative gradient | Balance learning rate, batch size, and consider advanced variants |
""")

# Footer
st.markdown("""
<footer>
© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</footer>
""", unsafe_allow_html=True)
