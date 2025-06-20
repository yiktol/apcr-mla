
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from PIL import Image
import io
import base64
import math

# Set page config
st.set_page_config(
    page_title="Regression Model Evaluation | AWS Learning",
    page_icon="📈",
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
    if 'dataset' not in st.session_state:
        st.session_state.dataset = 'california'
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'y_pred' not in st.session_state:
        st.session_state.y_pred = None
    if 'custom_data_generated' not in st.session_state:
        st.session_state.custom_data_generated = False
    if 'custom_X' not in st.session_state:
        st.session_state.custom_X = None
    if 'custom_y' not in st.session_state:
        st.session_state.custom_y = None
    if 'custom_model' not in st.session_state:
        st.session_state.custom_model = None

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

# Dataset selection
st.sidebar.markdown("### Dataset Selection")
dataset_option = st.sidebar.selectbox(
    "Choose a dataset:",
    options=["California Housing", "Synthetic Data"],
    key="dataset_selection"
)


if dataset_option == "California Housing":
    st.session_state.dataset = 'california'
else:
    st.session_state.dataset = 'synthetic'

# Model selection
st.sidebar.markdown("### Model Selection")
model_option = st.sidebar.selectbox(
    "Choose a regression model:",
    options=["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"],
    key="model_selection"
)

st.sidebar.divider()

with st.sidebar.expander(label='About this application' ,expanded=False):
    st.markdown("""
This application teaches regression model evaluation techniques through hands-on visualizations and dynamic examples. The app focuses on four key metrics:

- **Mean Squared Error (MSE)**: Explore how this fundamental metric penalizes prediction errors
- **Root Mean Squared Error (RMSE)**: Understand this interpretable metric that uses the same units as your target variable
- **Coefficient of Determination (R²)**: Visualize how well your model explains variance in the data
- **Adjusted R²**: Learn how this metric penalizes unnecessary model complexity

 """)

# Function to prepare dataset
@st.cache_data
def prepare_dataset(dataset_name, model_name):
    if dataset_name == 'california':
        data = fetch_california_housing()
        X, y = data.data, data.target
        feature_names = data.feature_names
    else:  # synthetic
        X, y = make_regression(
            n_samples=1000, n_features=10, n_informative=6, random_state=42, noise=25
        )
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge Regression":
        model = Ridge(alpha=1.0)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:  # Gradient Boosting
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Feature importances (if applicable)
    feature_importances = None
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances = np.abs(model.coef_)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, y_pred, model, feature_names, feature_importances

# Load dataset based on selection
X_train, X_test, y_train, y_test, y_pred, model, feature_names, feature_importances = prepare_dataset(
    st.session_state.dataset, model_option
)

# Store in session state
st.session_state.X_train = X_train
st.session_state.X_test = X_test
st.session_state.y_train = y_train
st.session_state.y_test = y_test
st.session_state.y_pred = y_pred
st.session_state.model = model

# Functions for metrics calculation
def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_adjusted_r2(y_true, y_pred, n_features):
    r2 = calculate_r2(y_true, y_pred)
    n_samples = len(y_true)
    adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    return adjusted_r2

# Helper functions for visualizations
def plot_predictions_vs_actual(y_true, y_pred, metric_name=None, metric_value=None):
    title_text = "Predicted vs Actual Values"
    if metric_name and metric_value is not None:
        title_text += f"  {metric_name}: {metric_value:.4}"
    
    fig = px.scatter(
        x=y_true, y=y_pred,
        labels={"x": "Actual Values", "y": "Predicted Values"},
        title=title_text
    )
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#FF9900", width=2, dash="dash"),
            name="Perfect Prediction"
        )
    )
    
    fig.update_layout(
        title_x=0.5,
        width=700,
        height=500
    )
    
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "Predicted Values", "y": "Residuals"},
        title="Residual Plot"
    )
    
    # Add horizontal line at y=0
    fig.add_hline(
        y=0, line_dash="dash",
        line_color="#FF9900",
        annotation_text="No Error",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title_x=0.5,
        width=700,
        height=500
    )
    
    return fig

def plot_residuals_histogram(y_true, y_pred):
    residuals = y_true - y_pred
    
    fig = px.histogram(
        x=residuals,
        labels={"x": "Residual Value"},
        title="Residuals Distribution",
        nbins=30
    )
    
    fig.add_vline(
        x=0, line_dash="dash",
        line_color="#FF9900",
        annotation_text="Zero Error",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title_x=0.5,
        width=700,
        height=500
    )
    
    return fig

def plot_feature_importance(importances, feature_names):
    if importances is None:
        return None
    
    # Get indices of features sorted by importance
    indices = np.argsort(importances)[::-1]
    
    # Take top 10 features
    top_n = min(10, len(feature_names))
    top_indices = indices[:top_n]
    
    fig = px.bar(
        x=importances[top_indices],
        y=[feature_names[i] for i in top_indices],
        orientation='h',
        color=importances[top_indices],
        color_continuous_scale='YlOrRd',
        labels={"x": "Importance", "y": "Feature"},
        title="Feature Importance"
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        title_x=0.5,
        coloraxis_showscale=False
    )
    
    return fig

def plot_model_comparison(metrics_dict, metric_name):
    models = list(metrics_dict.keys())
    values = [metrics_dict[model][metric_name] for model in models]
    
    fig = px.bar(
        x=models,
        y=values,
        color=values,
        color_continuous_scale='YlOrRd',
        labels={"x": "Model", "y": metric_name},
        text=[f"{val:.4f}" for val in values],
        title=f"Model Comparison by {metric_name}"
    )
    
    fig.update_layout(
        title_x=0.5,
        coloraxis_showscale=False
    )
    
    return fig

def generate_custom_data(n_samples=100, noise=0.5, polynomial=False, degree=2):
    """Generate custom data for interactive demonstration"""
    # Generate X values
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    
    # Generate y values
    if polynomial:
        y_true = np.sum([X**i for i in range(1, degree+1)], axis=0).flatten()
    else:
        y_true = 3*X.flatten() + 2  # Linear relationship
    
    # Add noise
    y = y_true + np.random.normal(0, noise, size=n_samples)
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    return X, y, y_true, model

# Main content with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Introduction", 
    "📏 Mean Squared Error", 
    "📐 RMSE", 
    "📊 R²", 
    "🔍 Adjusted R²",
    "❓ Knowledge Check"
])

# Introduction Tab
with tab1:
    st.title("Model Evaluation for Regression Problems")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Welcome to the Interactive Regression Model Evaluation Course!</h3>
        <p>In this interactive e-learning module, you'll learn about essential metrics 
        for evaluating regression models in machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3,2])
    
    with col1:
        st.markdown("""
        ### What You'll Learn
        
        In this module, you'll explore:
        
        1. **Mean Squared Error (MSE)**: The average of squared differences between predicted and actual values
        2. **Root Mean Squared Error (RMSE)**: A scale-dependent metric that's in the same units as the target variable
        3. **R² (R-squared)**: How well the model explains the variance in the target variable
        4. **Adjusted R²**: A modified version of R² that adjusts for the number of predictors
        
        Each section includes interactive examples that allow you to explore these concepts using real datasets.
        
        ### How to Use This Module
        
        1. Navigate through the tabs to explore different evaluation metrics
        2. Use the sidebar to select different datasets and models
        3. Interact with the visualizations to understand the concepts better
        4. Test your knowledge with the quiz in the last section
        """)
        
        st.info("👈 Select different datasets and models from the sidebar to see how evaluation metrics change!")
        
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GrLwS0WXiRoGn5-iEQkNOw.png", caption="Regression Model Evaluation Flow")
        
        st.markdown(f"""
        ### Dataset & Model Info
        
        **Dataset:** {dataset_option}  
        **Model:** {model_option}
        
        - Training samples: {len(st.session_state.X_train)}
        - Testing samples: {len(st.session_state.X_test)}
        - Features: {len(feature_names)}
        """)
    
    st.markdown("""
    ### Why Regression Evaluation Metrics Matter
    
    Evaluating regression models is crucial for:
    
    1. **Selecting the best model** for your specific problem
    2. **Tuning hyperparameters** to optimize performance
    3. **Understanding model limitations** and where it falls short
    4. **Communicating results** to stakeholders in meaningful terms
    
    Let's dive into the metrics that help us accomplish these goals!
    """)
    
    # Show comparison of models for this dataset
    st.subheader("Quick Model Comparison")
    
    # Calculate metrics for different models (cached)
    @st.cache_data
    def get_model_metrics(_dataset):
        metrics = {}
        for model_name in ["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"]:
            _, _, _, y_test, y_pred, _, _, _ = prepare_dataset(_dataset, model_name)
            
            mse = calculate_mse(y_test, y_pred)
            rmse = calculate_rmse(y_test, y_pred)
            r2 = calculate_r2(y_test, y_pred)
            adj_r2 = calculate_adjusted_r2(y_test, y_pred, X_test.shape[1])
            
            metrics[model_name] = {
                "MSE": mse,
                "RMSE": rmse,
                "R²": r2,
                "Adjusted R²": adj_r2
            }
        return metrics
    
    model_metrics = get_model_metrics(st.session_state.dataset)
    
    metric_to_display = st.selectbox(
        "Select a metric for comparison:",
        ["RMSE", "MSE", "R²", "Adjusted R²"]
    )
    
    comparison_fig = plot_model_comparison(model_metrics, metric_to_display)
    st.plotly_chart(comparison_fig,key='comparison_fig')
    
    st.markdown("""
    <p class="highlight">👆 Use the dropdown above to compare models across different metrics.</p>
    """, unsafe_allow_html=True)
    
# Mean Squared Error Tab
with tab2:
    st.title("Mean Squared Error (MSE)")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is Mean Squared Error?</h3>
        <p>Mean Squared Error is the average of squared differences between predicted and actual values. 
        It's one of the most common metrics for regression problems.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Understanding MSE
        
        **Definition:** The average of the squared differences between predicted and actual values.
        
        <div class="formula-box">
        $$ MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 $$
        </div>
        
        Where:
        - $n$ is the number of samples
        - $y_i$ is the actual value
        - $\\hat{y}_i$ is the predicted value
        
        ### Key Characteristics
        
        - **Always positive**: MSE is always ≥ 0 (perfect prediction gives MSE = 0)
        - **Heavily penalizes large errors**: Due to the squared term
        - **Unit**: MSE is in squared units of the target variable
        - **Sensitive to outliers**: Outliers heavily influence MSE due to squaring
        
        ### When to Use MSE
        
        - When large errors are particularly undesirable
        - When working with optimization algorithms (differentiable)
        - When the target variable's scale is meaningful
        """)
        
        st.markdown('<p class="highlight">MSE is particularly useful as a loss function for optimization algorithms because its derivative is continuous.</p>', unsafe_allow_html=True)
    
    with col2:
        # Calculate MSE for current model and dataset
        mse = calculate_mse(st.session_state.y_test, st.session_state.y_pred)
        
        st.metric("Current Model MSE", f"{mse:.4f}")
        
        # Plot predictions vs actual
        pred_vs_actual = plot_predictions_vs_actual(st.session_state.y_test, st.session_state.y_pred, "MSE", mse)
        st.plotly_chart(pred_vs_actual, key='pred_vs_actual_1')
    
    st.markdown("### Interactive MSE Explorer")
    
    st.markdown("""
    Adjust the sliders below to see how different factors affect MSE. 
    This simple linear regression example will help you understand how errors contribute to MSE.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        noise_level = st.slider(
            "Noise Level:",
            min_value=0.1,
            max_value=3.0,
            value=0.5,
            step=0.1,
            key="mse_noise"
        )
        
        samples = st.slider(
            "Number of Samples:",
            min_value=20,
            max_value=200,
            value=50,
            step=10,
            key="mse_samples"
        )
        
        has_outliers = st.checkbox("Add Outliers", value=False, key="mse_outliers")
    
    # Generate custom data for MSE demonstration
    X, y, y_true, custom_model = generate_custom_data(n_samples=samples, noise=noise_level)
    
    # Add outliers if requested
    if has_outliers:
        # Add a few outliers
        outlier_idx = np.random.choice(len(y), size=max(1, int(0.05 * len(y))), replace=False)
        y[outlier_idx] += np.random.choice([-1, 1], size=len(outlier_idx)) * np.random.uniform(5, 10, size=len(outlier_idx))
        
        # Refit model with outliers
        custom_model = LinearRegression()
        custom_model.fit(X, y)
    
    # Make predictions
    y_pred = custom_model.predict(X)
    
    # Calculate MSE
    custom_mse = calculate_mse(y, y_pred)
    
    with col2:
        # Create a scatter plot of the data and predictions
        fig = go.Figure()
        
        # Add data points
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y,
            mode='markers',
            name='Data Points',
            marker=dict(color='#232F3E')
        ))
        
        # Add true relationship line
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y_true,
            mode='lines',
            name='True Relationship',
            line=dict(color='green', width=2)
        ))
        
        # Add predicted line
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y_pred,
            mode='lines',
            name='Model Prediction',
            line=dict(color='#FF9900', width=2)
        ))
        
        # Add error visualization for a few points
        n_errors = min(10, len(X))
        error_indices = np.random.choice(len(X), n_errors, replace=False)
        
        for i in error_indices:
            fig.add_shape(
                type="line",
                x0=X[i, 0], y0=y[i],
                x1=X[i, 0], y1=y_pred[i],
                line=dict(color="red", width=1.5, dash="dot"),
            )
            
            # Add squared error as a label
            squared_error = (y[i] - y_pred[i])**2
            fig.add_annotation(
                x=X[i, 0],
                y=(y[i] + y_pred[i])/2,
                text=f"{squared_error:.2f}",
                showarrow=False,
                font=dict(size=8)
            )
        
        fig.update_layout(
            title=f"Regression with Error Visualization | MSE: {custom_mse:.4f}",
            xaxis_title="X",
            yaxis_title="y",
            legend_title="Legend",
            title_x=0.5,
            width=700,
            height=500
        )
        
        st.plotly_chart(fig, key='figx')
        
    st.markdown(f"""
    ### Observations
    
    - The current MSE for this example is **{custom_mse:.4f}**
    - Red dotted lines represent errors between actual and predicted values
    - The numbers along the red lines are squared errors for those points
    - MSE is the average of all squared errors
    
    **Try adjusting the sliders to see how:**
    - Higher noise increases MSE
    - Outliers have a significant impact on MSE due to squaring
    - Sample size affects the stability of the MSE measurement
    """)
    
    # Compare MSE with other models
    st.subheader("MSE Across Different Models")
    
    mse_comparison = plot_model_comparison(model_metrics, "MSE")
    st.plotly_chart(mse_comparison, key='mse_comparison')

# RMSE Tab
with tab3:
    st.title("Root Mean Squared Error (RMSE)")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is Root Mean Squared Error?</h3>
        <p>RMSE is the square root of the MSE. It's easier to interpret because it's in the same units as the target variable.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Understanding RMSE
        
        **Definition:** The square root of the mean squared error.
        
        <div class="formula-box">
        $$ RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2} = \\sqrt{MSE} $$
        </div>
        
        Where:
        - $n$ is the number of samples
        - $y_i$ is the actual value
        - $\\hat{y}_i$ is the predicted value
        
        ### Key Characteristics
        
        - **Same units as target variable**: Making it more interpretable than MSE
        - **Always positive**: RMSE is always ≥ 0 (perfect prediction gives RMSE = 0)
        - **Penalizes large errors**: Still sensitive to outliers, but less so than MSE
        - **Commonly used**: One of the most reported metrics in regression analysis
        
        ### When to Use RMSE
        
        - When you want an error metric in the same units as your target
        - When communicating results to non-technical stakeholders
        - When comparing models predicting the same target variable
        """)
        
        st.markdown('<p class="highlight">RMSE is often preferred over MSE in reporting because it\'s more interpretable - an RMSE of 5 means predictions are, on average, about 5 units away from actual values.</p>', unsafe_allow_html=True)
    
    with col2:
        # Calculate RMSE for current model and dataset
        rmse = calculate_rmse(st.session_state.y_test, st.session_state.y_pred)
        
        st.metric("Current Model RMSE", f"{rmse:.4f}")
        
        # Plot predictions vs actual with RMSE
        pred_vs_actual = plot_predictions_vs_actual(st.session_state.y_test, st.session_state.y_pred, "RMSE", rmse)
        st.plotly_chart(pred_vs_actual, key='pred_vs_actual_2')
        
        # Show residuals histogram
        residuals_hist = plot_residuals_histogram(st.session_state.y_test, st.session_state.y_pred)
        st.plotly_chart(residuals_hist, key='redisual_hist')
    
    st.markdown("### MSE vs. RMSE Comparison")
    
    st.markdown("""
    Let's visualize the relationship between MSE and RMSE. The table below shows hypothetical error values
    and how they translate to MSE and RMSE.
    """)
    
    # Create comparison data
    errors = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    mses = errors ** 2
    rmses = np.sqrt(mses)
    
    comparison_df = pd.DataFrame({
        'Average Error': errors,
        'MSE': mses,
        'RMSE': rmses
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(comparison_df.style.format({
            'Average Error': '{:.1f}',
            'MSE': '{:.2f}',
            'RMSE': '{:.2f}'
        }))
    
    with col2:
        # Create bar chart comparing MSE and RMSE
        fig = go.Figure()
        
        # Add MSE bars
        fig.add_trace(go.Bar(
            x=errors,
            y=mses,
            name='MSE',
            marker_color='#FF9900'
        ))
        
        # Add RMSE bars
        fig.add_trace(go.Bar(
            x=errors,
            y=rmses,
            name='RMSE',
            marker_color='#232F3E'
        ))
        
        fig.update_layout(
            title="MSE vs. RMSE for Different Error Values",
            title_x=0.5,
            xaxis_title="Average Error",
            yaxis_title="Value",
            barmode='group',
            width=700,
            height=400
        )
        
        st.plotly_chart(fig, key='fig2')
    
    st.markdown("""
    ### Key Takeaways
    
    Notice from the chart above:
    
    1. **MSE increases quadratically** with the error magnitude
    2. **RMSE increases linearly** with the error magnitude
    3. For small errors (<1), MSE is smaller than RMSE
    4. For large errors (>1), MSE is larger than RMSE
    
    This is why RMSE is often preferred when communicating model performance - it doesn't exaggerate the penalty for large errors as much as MSE does.
    """)
    
    # RMSE across different models
    st.subheader("RMSE Across Different Models")
    
    rmse_comparison = plot_model_comparison(model_metrics, "RMSE")
    st.plotly_chart(rmse_comparison, key='rmse_comparison')

# R² Tab
with tab4:
    st.title("Coefficient of Determination (R²)")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is R²?</h3>
        <p>R² (R-squared) represents the proportion of variance in the dependent variable that's explained by the independent variables. It ranges from 0 to 1.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Understanding R²
        
        **Definition:** The proportion of the variance in the dependent variable that is predictable from the independent variables.
        
        <div class="formula-box">
        $$ R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum_{i} (y_i - \\hat{y}_i)^2}{\\sum_{i} (y_i - \\bar{y})^2} $$
        </div>
        
        Where:
        - $SS_{res}$ is the sum of squares of residuals
        - $SS_{tot}$ is the total sum of squares
        - $\\bar{y}$ is the mean of the observed data
        
        ### Key Characteristics
        
        - **Range**: R² typically ranges from 0 to 1
        - **Interpretation**: 
          - R² = 1: Model perfectly predicts the target
          - R² = 0: Model doesn't explain any variance (equivalent to predicting the mean)
          - R² < 0: Can happen with poorly fitting models (worse than predicting the mean)
        - **Scale-independent**: Unlike MSE/RMSE, R² is unaffected by the scale of the target variable
        
        ### When to Use R²
        
        - When comparing models across different datasets
        - When you need a scale-independent measure of fit
        - When communicating model quality to stakeholders
        """)
        
        st.markdown('<p class="highlight">R² is particularly useful because it has a clear interpretation: "Our model explains X% of the variance in the target variable."</p>', unsafe_allow_html=True)
    
    with col2:
        # Calculate R² for current model and dataset
        r2 = calculate_r2(st.session_state.y_test, st.session_state.y_pred)
        
        # Create gauge chart for R²
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = r2,
            title = {'text': "R² Score"},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': "#FF9900"},
                'steps': [
                    {'range': [0, 0.3], 'color': "#FF4136"},
                    {'range': [0.3, 0.7], 'color': "#FFDC00"},
                    {'range': [0.7, 1], 'color': "#2ECC40"}
                ]
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig,key='fig3')
        
        # Plot predictions vs actual with R²
        pred_vs_actual = plot_predictions_vs_actual(st.session_state.y_test, st.session_state.y_pred, "R²", r2)
        st.plotly_chart(pred_vs_actual,key='pred_vs_actual_3')
    
    st.markdown("### Interactive R² Explorer")
    
    st.markdown("""
    Adjust the sliders below to see how different factors affect R². 
    This visualization helps you understand how variance explained works.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        r2_noise = st.slider(
            "Noise Level:",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key="r2_noise"
        )
        
        r2_samples = st.slider(
            "Number of Samples:",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
            key="r2_samples"
        )
        
        model_strength = st.slider(
            "Model Strength (complexity):",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            key="r2_model"
        )
    
    # Generate custom data for R² demonstration
    X, y, y_true, custom_model = generate_custom_data(
        n_samples=r2_samples, 
        noise=r2_noise, 
        polynomial=True,
        degree=model_strength
    )
    
    # Make predictions
    y_pred = custom_model.predict(X)
    
    # Calculate R²
    custom_r2 = calculate_r2(y, y_pred)
    
    # Calculate total and explained variance
    y_mean = np.mean(y)
    total_variance = np.sum((y - y_mean)**2)
    explained_variance = np.sum((y_pred - y_mean)**2)
    unexplained_variance = np.sum((y - y_pred)**2)
    
    with col2:
        # Create a scatter plot showing total and explained variance
        fig = go.Figure()
        
        # Add data points
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y,
            mode='markers',
            name='Data Points',
            marker=dict(color='#232F3E')
        ))
        
        # Add predicted line
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y_pred,
            mode='lines',
            name='Model Prediction',
            line=dict(color='#FF9900', width=2)
        ))
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=[X.min(), X.max()],
            y=[y_mean, y_mean],
            mode='lines',
            name='Mean (Baseline)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Visualize variance components for a few points
        n_points = min(5, len(X))
        sample_indices = np.random.choice(len(X), n_points, replace=False)
        
        for i in sample_indices:
            # Unexplained variance (to prediction)
            fig.add_shape(
                type="line",
                x0=X[i, 0], y0=y[i],
                x1=X[i, 0], y1=y_pred[i],
                line=dict(color="red", width=1.5, dash="dot"),
            )
            
            # Explained variance (from mean to prediction)
            fig.add_shape(
                type="line",
                x0=X[i, 0], y0=y_mean,
                x1=X[i, 0], y1=y_pred[i],
                line=dict(color="green", width=1.5, dash="dot"),
            )
        
        fig.update_layout(
            title=f"R² Visualization | Score: {custom_r2:.4f}",
            xaxis_title="X",
            yaxis_title="y",
            legend_title="Legend",
            title_x=0.5,
            width=700,
            height=500
        )
        
        st.plotly_chart(fig,key='fig4')
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Total Variance", f"{total_variance:.2f}")
    with col2:
        st.metric("Explained Variance", f"{explained_variance:.2f}", f"{(explained_variance/total_variance*100):.1f}%")
    with col3:
        st.metric("Unexplained Variance", f"{unexplained_variance:.2f}", f"{(unexplained_variance/total_variance*100):.1f}%")
    
    st.markdown(f"""
    ### Observations
    
    - The current R² for this example is **{custom_r2:.4f}**
    - This means the model explains about **{custom_r2*100:.1f}%** of the variance in the data
    - The red dotted lines represent unexplained variance (errors)
    - The green dotted lines represent explained variance (improvement over the mean)
    
    **Try adjusting the sliders to see how:**
    - Higher noise decreases R² (more unexplained variance)
    - More complex models (higher strength) can capture more variance, but may overfit
    - R² approaches 1.0 when the model fits the data very well
    """)
    
    # Compare R² with other models
    st.subheader("R² Across Different Models")
    
    r2_comparison = plot_model_comparison(model_metrics, "R²")
    st.plotly_chart(r2_comparison, key='r2_comparison')
    
    st.markdown("""
    ### Limitations of R²
    
    Despite its usefulness, R² has some important limitations:
    
    1. **Adding predictors always increases R²**, even if they are irrelevant
    2. **Doesn't indicate whether coefficients are biased**
    3. **Doesn't indicate if the model is adequate** - a high R² doesn't necessarily mean a good model
    4. **Can be misleading with non-linear relationships**
    5. **Doesn't account for the complexity of the model**
    
    This last limitation is addressed by the Adjusted R² metric, which we'll explore in the next tab.
    """)

# Adjusted R² Tab
with tab5:
    st.title("Adjusted R²")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is Adjusted R²?</h3>
        <p>Adjusted R² modifies the R² by accounting for the number of predictors in the model, penalizing complex models with many features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Understanding Adjusted R²
        
        **Definition:** A modified version of R² that adjusts for the number of predictors in a regression model.
        
        <div class="formula-box">
        $$ \\text{Adjusted } R^2 = 1 - \\frac{(1 - R^2)(n - 1)}{n - p - 1} $$
        </div>
        
        Where:
        - $n$ is the number of samples
        - $p$ is the number of predictors (features)
        - $R^2$ is the coefficient of determination
        
        ### Key Characteristics
        
        - **Can decrease with additional predictors** if they don't add enough explanatory power
        - **Always less than or equal to R²**
        - **Can be negative** if the model is very poor
        - **Penalizes model complexity**
        
        ### When to Use Adjusted R²
        
        - When comparing models with different numbers of predictors
        - When performing feature selection
        - When you want to avoid overfitting by adding too many features
        """)
        
        st.markdown('<p class="highlight">Adjusted R² helps protect against overfitting by penalizing models that add predictors without substantially improving explanatory power.</p>', unsafe_allow_html=True)
    
    with col2:
        # Calculate R² and Adjusted R² for current model and dataset
        r2 = calculate_r2(st.session_state.y_test, st.session_state.y_pred)
        adj_r2 = calculate_adjusted_r2(st.session_state.y_test, st.session_state.y_pred, X_test.shape[1])
        
        # Show metric comparison
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R²", f"{r2:.4f}")
        with col2:
            st.metric("Adjusted R²", f"{adj_r2:.4f}", f"{(adj_r2-r2):.4f}")
        
        # Plot predictions vs actual with both metrics
        pred_vs_actual = plot_predictions_vs_actual(
            st.session_state.y_test, 
            st.session_state.y_pred, 
            f"Adj. R² | R²", 
            f"{adj_r2:.4f} | {r2:.4f}"
        )
        st.plotly_chart(pred_vs_actual, key='pred_vs_actual_4')
    
    st.markdown("### R² vs. Adjusted R²")
    
    st.markdown("""
    Let's visualize how R² and Adjusted R² behave as we add more features to a model.
    This is a common scenario in multiple regression.
    """)
    
    # Create simulated data for R² vs Adjusted R²
    np.random.seed(42)
    n_samples = 100
    max_features = 20
    
    # Generate feature effectiveness data (diminishing returns)
    feature_r2 = np.array([0.0] + [0.5 * (1 - 0.9**i) for i in range(1, max_features+1)])
    r2_values = np.cumsum(feature_r2)
    r2_values = np.clip(r2_values, 0, 0.95)  # Cap at 0.95 for realism
    
    # Calculate adjusted R² for each feature count
    adj_r2_values = np.array([
        1 - (1 - r2) * (n_samples - 1) / (n_samples - p - 1) 
        for r2, p in zip(r2_values, range(0, max_features+1))
    ])
    
    # Create data frame for plotting
    feature_df = pd.DataFrame({
        'Number of Features': range(0, max_features+1),
        'R²': r2_values,
        'Adjusted R²': adj_r2_values
    })
    
    # Find optimal feature count (where adjusted R² is highest)
    optimal_features = feature_df['Adjusted R²'].argmax()
    
    # Plot the comparison
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=feature_df['Number of Features'],
        y=feature_df['R²'],
        mode='lines+markers',
        name='R²',
        line=dict(color='#FF9900', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=feature_df['Number of Features'],
        y=feature_df['Adjusted R²'],
        mode='lines+markers',
        name='Adjusted R²',
        line=dict(color='#232F3E', width=2)
    ))
    
    # Add vertical line at optimal feature count
    fig.add_vline(
        x=optimal_features,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Optimal Features: {optimal_features}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="R² vs. Adjusted R² as Features Are Added",
        title_x=0.5,
        xaxis_title="Number of Features",
        yaxis_title="Score",
        width=800,
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, key='r2_vs_adj_r2')
    
    st.markdown(f"""
    ### Observations
    
    - R² **always increases** (or stays the same) as more features are added
    - Adjusted R² **increases only when valuable features** are added
    - Adjusted R² **decreases when less useful features** are added
    - In this example, the optimal model has **{optimal_features} features**
    - After {optimal_features} features, we're likely overfitting the data
    
    This demonstrates why Adjusted R² is valuable for feature selection and model comparison.
    """)
    
    st.markdown("### Interactive Adjusted R² Explorer")
    
    st.markdown("""
    Use the controls below to explore how sample size and number of features affect the relationship 
    between R² and Adjusted R².
    """)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        demo_r2 = st.slider(
            "R² Value:",
            min_value=0.0,
            max_value=0.99,
            value=0.75,
            step=0.01,
            key="adj_r2_demo_r2"
        )
    
    with col2:
        demo_n = st.slider(
            "Sample Size:",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            key="adj_r2_demo_n"
        )
    
    # Calculate adjusted R² for different feature counts
    feature_counts = list(range(1, 21))
    demo_adj_r2_values = [
        1 - (1 - demo_r2) * (demo_n - 1) / (demo_n - p - 1) 
        for p in feature_counts
    ]
    
    with col3:
        # Plot adjusted R² for different feature counts
        fig = px.line(
            x=feature_counts, 
            y=demo_adj_r2_values,
            markers=True,
            labels={'x': 'Number of Features', 'y': 'Adjusted R²'},
            title=f"Adjusted R² with Fixed R² = {demo_r2} and n = {demo_n}"
        )
        
        # Add horizontal line for R²
        fig.add_hline(
            y=demo_r2,
            line_dash="dash",
            line_color="#FF9900",
            annotation_text="R²",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title_x=0.5,
            width=600,
            height=400
        )
        
        st.plotly_chart(fig, key='adj_r2_demo')
    
    st.markdown("""
    ### Key Takeaways
    
    1. **The penalty increases with more features**: As you add more features, the gap between R² and Adjusted R² grows
    
    2. **Sample size matters**: With larger sample sizes, the penalty for additional features is smaller
    
    3. **Balance complexity and fit**: Adjusted R² helps you find the sweet spot between model complexity and goodness of fit
    
    4. **Better than R² for model selection**: When comparing models with different numbers of features, Adjusted R² is more appropriate
    """)
    
    # Compare Adjusted R² with other models
    st.subheader("Adjusted R² Across Different Models")
    
    adj_r2_comparison = plot_model_comparison(model_metrics, "Adjusted R²")
    st.plotly_chart(adj_r2_comparison,  key='adj_r2_comparison')

# Knowledge Check Tab
with tab6:
    st.title("Knowledge Check")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Test Your Understanding</h3>
        <p>Answer these five questions to check your understanding of regression model evaluation metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quiz questions
    quiz = [
        {
            "question": "Which regression evaluation metric is in the same units as the target variable?",
            "options": ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R²", "Adjusted R²"],
            "answer": "Root Mean Squared Error (RMSE)"
        },
        {
            "question": "What does an R² value of 0.75 mean?",
            "options": [
                "The model is 75% accurate",
                "The model explains 75% of the variance in the target variable",
                "The model's predictions are 75% away from the actual values",
                "The model has a 75% chance of making correct predictions"
            ],
            "answer": "The model explains 75% of the variance in the target variable"
        },
        {
            "question": "When comparing models with different numbers of features, which metric is most appropriate?",
            "options": ["MSE", "RMSE", "R²", "Adjusted R²"],
            "answer": "Adjusted R²"
        },
        {
            "question": "What happens to MSE when errors increase?",
            "options": [
                "MSE increases linearly with error size",
                "MSE increases quadratically with error size",
                "MSE decreases inversely to error size",
                "MSE is not affected by error size"
            ],
            "answer": "MSE increases quadratically with error size"
        },
        {
            "question": "Which statement about Adjusted R² is TRUE?",
            "options": [
                "Adjusted R² always increases when more features are added",
                "Adjusted R² is always higher than R²",
                "Adjusted R² penalizes models with many features",
                "Adjusted R² cannot be negative"
            ],
            "answer": "Adjusted R² penalizes models with many features"
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
        st.plotly_chart(fig, key='score_gauge')
        
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
            st.success("🎉 Perfect score! You've mastered regression evaluation concepts!")
        elif st.session_state.quiz_score >= len(quiz) * 0.8:
            st.success("🌟 Great job! You have a strong understanding of regression evaluation!")
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
## Summary of Regression Evaluation Metrics

| Metric | Formula | Range | When to Use |
|--------|---------|-------|------------|
| MSE | $\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$ | [0, ∞) | For optimization algorithms |
| RMSE | $\\sqrt{MSE}$ | [0, ∞) | For interpretable error in target units |
| R² | $1 - \\frac{SS_{res}}{SS_{tot}}$ | (-∞, 1] | For variance explained, scale-independent comparison |
| Adjusted R² | $1 - \\frac{(1-R^2)(n-1)}{n-p-1}$ | (-∞, 1] | For comparing models with different feature counts |
""")

# Footer
st.markdown("""
<footer>
© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</footer>
""", unsafe_allow_html=True)
# ```

# ## Explanation of the Application

# This Streamlit application serves as an interactive e-learning tool for model evaluation in regression problems. Here's how it's structured:

# 1. **Setup and Configuration**
#    - Uses modern Python libraries like Streamlit, scikit-learn, Matplotlib, Seaborn, and Plotly
#    - Sets up AWS-themed styling with custom CSS
#    - Configures session state management for preserving data during navigation

# 2. **Navigation**
#    - Tab-based navigation with emojis for easy access to different concepts
#    - Sidebar for session management, dataset selection, and model selection

# 3. **Content Sections**
#    - **Introduction**: Overview of regression evaluation concepts with model comparison
#    - **Mean Squared Error**: Interactive visualization showing squared errors and impact of noise/outliers
#    - **RMSE**: Comparison with MSE and data distribution visualization
#    - **R²**: Interactive explorer showing variance explained and prediction quality
#    - **Adjusted R²**: Demonstration of feature selection and penalty for model complexity
#    - **Knowledge Check**: Five-question quiz with feedback

# 4. **Interactive Elements**
#    - Dataset selection (California Housing, Synthetic Data)
#    - Model selection (Linear Regression, Ridge, Random Forest, Gradient Boosting)
#    - Interactive visualizations with adjustable parameters
#    - Custom data generators to demonstrate statistical concepts
#    - Quiz with instant feedback

# 5. **Visualizations**
#    - Actual vs. predicted scatter plots
#    - Residual plots and histograms
#    - Feature importance charts
#    - Model comparison bar charts
#    - Interactive simulations of R² and adjusted R² behavior

# 6. **User Experience**
#    - Clean, modern AWS-themed styling
#    - Responsive layout with columns and proper spacing
#    - Clear explanations with highlighted important concepts
#    - Formula boxes for mathematical clarity
#    - Session reset functionality

# The application is designed to be educational, engaging, and interactive, allowing users to explore regression evaluation concepts through hands-on examples and visualizations. Each concept is thoroughly explained with both theoretical information and practical demonstrations that respond to user inputs.