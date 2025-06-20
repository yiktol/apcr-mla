
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import altair as alt
from streamlit_lottie import st_lottie
import json

# Page configuration
st.set_page_config(
    page_title="ML Concepts Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS color palette
AWS_COLORS = {
    'primary': '#232F3E',     # AWS Dark Blue
    'secondary': '#FF9900',   # AWS Orange
    'accent1': '#1E3050',     # Darker blue
    'accent2': '#0073BB',     # AWS Light Blue
    'light': '#FFFFFF',       # White
    'mid': '#EAEDED',         # Light gray
    'dark': '#16191F'         # Very dark blue/black
}

# Custom CSS styles
st.markdown(f"""
<style>
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {AWS_COLORS['mid']};
        border-radius: 4px 4px 0px 0px;
        padding: 15px 20px;
        color: {AWS_COLORS['primary']};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {AWS_COLORS['secondary']};
        color: {AWS_COLORS['light']};
    }}
    .stButton > button {{
        background-color: {AWS_COLORS['secondary']};
        color: {AWS_COLORS['primary']};
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-weight: bold;
    }}
    .highlight {{
        background-color: {AWS_COLORS['mid']};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid {AWS_COLORS['secondary']};
    }}
    .title {{
        color: {AWS_COLORS['primary']};
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 20px;
    }}
    .subtitle {{
        color: {AWS_COLORS['accent2']};
        font-size: 20px;
        font-weight: bold;
        margin: 15px 0;
    }}
    .caption {{
        font-size: 14px;
        color: {AWS_COLORS['primary']};
        font-style: italic;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    if 'hyperparams' not in st.session_state:
        st.session_state.hyperparams = {
            'generated_data': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None,
        }
        
    if 'learning_rate' not in st.session_state:
        st.session_state.learning_rate = {
            'data': None,
            'histories': {},
            'best_lr': None,
        }
        
    if 'early_stopping' not in st.session_state:
        st.session_state.early_stopping = {
            'data': None,
            'history': [],
            'best_epoch': None,
            'final_loss': None,
        }

# Initialize session state at app start
init_session_state()

# Sidebar for session management
with st.sidebar:
    st.title("🔄 Session Management")
    if st.button("Reset All Sessions"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.success("Session reset successfully!")
        
    st.markdown("---")
    st.markdown("""
    <div class='highlight'>
    <p>This application helps you learn key machine learning concepts through interactive examples:</p>
    <ul>
        <li>Hyperparameters</li>
        <li>Learning Rate</li>
        <li>Early Stopping</li>
    </ul>
    <p>Use the tabs above to navigate between concepts.</p>
    </div>
    """, unsafe_allow_html=True)

    # Add AWS-styled logo at the bottom
    st.markdown("---")
    st.markdown("<div style='text-align: center'>Powered by <b>ML Explorer</b></div>", unsafe_allow_html=True)

# Load animations/lottie files
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Title
st.markdown("<div class='title'>Interactive Machine Learning Concepts Explorer</div>", unsafe_allow_html=True)
st.markdown("Explore key machine learning concepts through hands-on interactive examples")

# Main tabs
tabs = st.tabs(["🎛️ Hyperparameters", "📈 Learning Rate", "🛑 Early Stopping"])

# Tab 1: Hyperparameters
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='subtitle'>What are Hyperparameters?</div>", unsafe_allow_html=True)
        st.markdown("""
        Hyperparameters are configuration variables that govern the training process of machine learning models. 
        Unlike regular model parameters that are learned during training, hyperparameters must be set before training begins.
        
        **Examples of hyperparameters include:**
        - Learning rate
        - Number of hidden layers and neurons
        - Regularization strength
        - Batch size
        - Number of trees in a random forest
        """)
    
    with col2:
        # Display a relevant animation
        lottie_hyper = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_khzniaya.json")
        if lottie_hyper:
            st_lottie(lottie_hyper, height=200, key="hyper_animation")
        else:
            st.image("https://miro.medium.com/v2/resize:fit:828/format:webp/1*oLMItzWLblVaH6RFQVDmpg.jpeg", caption="Hyperparameter tuning concept")

    st.markdown("---")
    st.markdown("<div class='subtitle'>Interactive Demo: SVM Classifier Hyperparameter Tuning</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='highlight'>In this demo, you can tune hyperparameters of a Support Vector Machine classifier and observe how they affect model performance.</div>", unsafe_allow_html=True)
        
        # Generate classification dataset if not already generated
        if st.session_state.hyperparams['generated_data'] is None:
            X, y = make_classification(
                n_samples=1000, 
                n_features=2, 
                n_informative=2, 
                n_redundant=0, 
                n_clusters_per_class=1,
                random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            st.session_state.hyperparams['generated_data'] = (X, y)
            st.session_state.hyperparams['X_train'] = X_train
            st.session_state.hyperparams['X_test'] = X_test
            st.session_state.hyperparams['y_train'] = y_train
            st.session_state.hyperparams['y_test'] = y_test
            
        # Hyperparameter selection
        alpha = st.slider("Regularization strength (alpha)", 0.0001, 0.1, 0.01, format="%.4f")
        max_iter = st.slider("Maximum iterations", 100, 2000, 1000, step=100)
        loss = st.selectbox("Loss function", ["hinge", "log_loss", "modified_huber", "perceptron"])
        
        if st.button("Train Model", key="train_hyperparam_model"):
            with st.spinner("Training model..."):
                # Get data from session state
                X_train = st.session_state.hyperparams['X_train']
                X_test = st.session_state.hyperparams['X_test']
                y_train = st.session_state.hyperparams['y_train']
                y_test = st.session_state.hyperparams['y_test']
                
                # Train model with selected hyperparameters
                clf = SGDClassifier(
                    loss=loss,
                    alpha=alpha,
                    max_iter=max_iter,
                    random_state=42
                )
                
                # Train and evaluate
                start_time = time.time()
                clf.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store in session state
                st.session_state.hyperparams['current_model'] = clf
                st.session_state.hyperparams['accuracy'] = accuracy
                st.session_state.hyperparams['training_time'] = training_time
    
    with col2:
        # Display results if model has been trained
        if 'current_model' in st.session_state.hyperparams:
            st.markdown("<div class='subtitle'>Model Performance</div>", unsafe_allow_html=True)
            
            # Create metrics display
            col1, col2 = st.columns(2)
            col1.metric("Test Accuracy", f"{st.session_state.hyperparams['accuracy']:.4f}")
            col2.metric("Training Time", f"{st.session_state.hyperparams['training_time']:.2f}s")
            
            # Get data for visualization
            X, y = st.session_state.hyperparams['generated_data']
            
            # Create a meshgrid to visualize the decision boundaries
            h = 0.02  # step size in the mesh
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Get predictions for the entire meshgrid
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            Z = st.session_state.hyperparams['current_model'].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.contourf(xx, yy, Z, alpha=0.3)
            
            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('SVM Decision Boundary with Current Hyperparameters')
            plt.colorbar(scatter)
            
            st.pyplot(fig)
        else:
            st.info("Train a model to see the results here!")

    st.markdown("---")
    
    # Hyperparameter explanation
    with st.expander("🧐 Why are hyperparameters so important?"):
        st.markdown("""
        Hyperparameters directly influence how a model learns from data and makes predictions. Poor hyperparameter choices can result in:
        
        1. **Underfitting**: The model is too simple to capture the underlying patterns in the data.
        2. **Overfitting**: The model captures noise in the training data rather than the actual patterns.
        3. **Slow Convergence**: Training takes unnecessarily long.
        4. **Resource Waste**: Computing resources are used inefficiently.
        
        Proper hyperparameter tuning is often the difference between a mediocre model and a high-performing one!
        """)
        
    with st.expander("📚 Methods for Hyperparameter Tuning"):
        st.markdown("""
        **Popular methods for hyperparameter tuning include:**
        
        | Method | Description | Pros | Cons |
        |--------|-------------|------|------|
        | **Grid Search** | Exhaustively tries all combinations | Simple, thorough | Computationally expensive |
        | **Random Search** | Samples random combinations | More efficient than grid search | May miss optimal combinations |
        | **Bayesian Optimization** | Uses probabilistic model to select best combinations | Very efficient | More complex to implement |
        | **Genetic Algorithms** | Evolves hyperparameters over generations | Can find novel combinations | Requires careful design |
        
        In practice, a combination of these methods is often used, starting with random search to find promising regions, followed by more focused tuning.
        """)

# Tab 2: Learning Rate
with tabs[1]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='subtitle'>What is Learning Rate?</div>", unsafe_allow_html=True)
        st.markdown("""
        Learning rate is a crucial hyperparameter that determines how much to adjust model weights during training. It controls the size of steps taken during optimization.
        
        **Key characteristics:**
        - **Too high**: Can cause training to diverge or oscillate
        - **Too low**: Can lead to slow convergence or getting stuck in local minima
        - **Just right**: Enables efficient training toward the optimal weights
        """)
    
    with col2:
        # Display a relevant animation
        lottie_lr = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_plytpq35.json")
        if lottie_lr:
            st_lottie(lottie_lr, height=200, key="lr_animation")
        else:
            st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*QzmFkZ3pyoKR5OHA4OX5bg.jpeg", caption="Learning rate concept")

    st.markdown("---")
    st.markdown("<div class='subtitle'>Interactive Demo: Finding the Optimal Learning Rate</div>", unsafe_allow_html=True)
    
    # Learning rate visualization demo
    if st.session_state.learning_rate['data'] is None:
        # Generate regression data
        X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        st.session_state.learning_rate['data'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='highlight'>Experiment with different learning rates to see how they affect model training and convergence.</div>", unsafe_allow_html=True)
        
        # Learning rate selection
        learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
        selected_lr = st.select_slider(
            "Select a learning rate to test",
            options=learning_rates,
            format_func=lambda x: f"{x:.4f}"
        )
        
        max_epochs = st.slider("Number of epochs", 10, 200, 100, step=10)
        
        if st.button("Train with this Learning Rate", key="train_lr_model"):
            with st.spinner(f"Training with learning rate {selected_lr}..."):
                # Get data from session state
                data = st.session_state.learning_rate['data']
                X_train, y_train = data['X_train'], data['y_train']
                X_test, y_test = data['X_test'], data['y_test']
                
                # Train model with selected learning rate
                model = SGDRegressor(learning_rate='constant', eta0=selected_lr, max_iter=1, random_state=42)
                
                train_losses = []
                test_losses = []
                
                for epoch in range(max_epochs):
                    model.partial_fit(X_train, y_train)
                    
                    # Compute training and test loss
                    train_pred = model.predict(X_train)
                    train_loss = mean_squared_error(y_train, train_pred)
                    
                    test_pred = model.predict(X_test)
                    test_loss = mean_squared_error(y_test, test_pred)
                    
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                
                # Store history in session state
                st.session_state.learning_rate['histories'][selected_lr] = {
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'final_train_loss': train_losses[-1],
                    'final_test_loss': test_losses[-1]
                }
                
                # Find best learning rate so far
                # if 'best_lr' not in st.session_state.learning_rate or \
                #    st.session_state.learning_rate['histories'][selected_lr]['final_test_loss'] < \
                #    st.session_state.learning_rate['histories'][st.session_state.learning_rate['best_lr']]['final_test_loss']:
                #     st.session_state.learning_rate['best_lr'] = selected_lr

            # Find best learning rate so far
            if 'best_lr' not in st.session_state.learning_rate or \
            st.session_state.learning_rate['best_lr'] is None:
                st.session_state.learning_rate['best_lr'] = selected_lr
            else:
                current_test_loss = st.session_state.learning_rate['histories'][selected_lr]['final_test_loss']
                best_test_loss = st.session_state.learning_rate['histories'][st.session_state.learning_rate['best_lr']]['final_test_loss']
                if current_test_loss < best_test_loss:
                    st.session_state.learning_rate['best_lr'] = selected_lr



    with col2:
        # Display results if model has been trained
        if len(st.session_state.learning_rate.get('histories', {})) > 0:
            st.markdown("<div class='subtitle'>Learning Rate Comparison</div>", unsafe_allow_html=True)
            
            # Create a plot for the selected learning rate
            if selected_lr in st.session_state.learning_rate['histories']:
                history = st.session_state.learning_rate['histories'][selected_lr]
                
                # Create a DataFrame for plotting
                df = pd.DataFrame({
                    'Epoch': list(range(1, len(history['train_losses']) + 1)),
                    'Training Loss': history['train_losses'],
                    'Validation Loss': history['test_losses']
                })
                
                fig = px.line(
                    df, x='Epoch', y=['Training Loss', 'Validation Loss'],
                    title=f'Loss Curves for Learning Rate = {selected_lr}',
                    template='plotly_white',
                    color_discrete_sequence=[AWS_COLORS['secondary'], AWS_COLORS['accent2']]
                )
                
                # Add horizontal line for the best final loss
                if st.session_state.learning_rate.get('best_lr') is not None:
                    best_loss = st.session_state.learning_rate['histories'][st.session_state.learning_rate['best_lr']]['final_test_loss']
                    fig.add_hline(y=best_loss, line_dash="dash", line_color="green", 
                                 annotation_text=f"Best Loss: {best_loss:.4f}", 
                                 annotation_position="bottom right")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show comparison of all learning rates tested
            lr_comparison = []
            for lr, hist in st.session_state.learning_rate['histories'].items():
                lr_comparison.append({
                    'Learning Rate': lr,
                    'Final Training Loss': hist['final_train_loss'],
                    'Final Validation Loss': hist['final_test_loss']
                })
            
            lr_df = pd.DataFrame(lr_comparison)
            
            # Create comparison chart
            fig = px.bar(
                lr_df, x='Learning Rate', y='Final Validation Loss',
                title='Comparison of Learning Rates',
                template='plotly_white',
                color='Final Validation Loss',
                color_continuous_scale=['green', 'yellow', 'red']
            )
            
            # Highlight the best learning rate
            if st.session_state.learning_rate.get('best_lr') is not None:
                best_lr = st.session_state.learning_rate['best_lr']
                best_idx = lr_df[lr_df['Learning Rate'] == best_lr].index[0]
                fig.add_annotation(
                    x=best_lr,
                    y=lr_df.iloc[best_idx]['Final Validation Loss'],
                    text="Best",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.learning_rate.get('best_lr') is not None:
                best_lr = st.session_state.learning_rate['best_lr']
                st.success(f"Best learning rate found: {best_lr}, with validation loss: {st.session_state.learning_rate['histories'][best_lr]['final_test_loss']:.4f}")
        else:
            st.info("Train models with different learning rates to see the results here!")

    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='subtitle'>Learning Rate Effects</div>", unsafe_allow_html=True)
        st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*XJXa-PEkH85rwcG2xFXI7g.png", caption="How learning rate affects convergence")
        
    with col2:
        st.markdown("<div class='subtitle'>Learning Rate Schedules</div>", unsafe_allow_html=True)
        st.markdown("""
        Instead of using a constant learning rate, you can use learning rate schedules:
        
        - **Time-based decay**: Reduces learning rate over time
        - **Step decay**: Reduces learning rate at specific intervals
        - **Exponential decay**: Reduces learning rate exponentially
        - **Cosine annealing**: Cycles learning rate with decreasing amplitude
        - **Adaptive methods**: Algorithms like Adam, RMSprop adjust learning rates automatically
        """)
        
        # Display a small graphic of various learning rate schedules
        lr_schedules = pd.DataFrame({
            'Epoch': list(range(100)),
            'Constant': [0.1] * 100,
            'Time-based': [0.1 / (1 + 0.01 * e) for e in range(100)],
            'Step': [0.1 * (0.5 ** (e // 25)) for e in range(100)],
            'Exponential': [0.1 * np.exp(-0.01 * e) for e in range(100)],
            'Cosine': [0.1 * (1 + np.cos(np.pi * e / 100)) / 2 for e in range(100)]
        })
        
        fig = px.line(
            lr_schedules, x='Epoch', y=['Constant', 'Time-based', 'Step', 'Exponential', 'Cosine'],
            title='Learning Rate Schedules',
            template='plotly_white',
            color_discrete_sequence=[
                AWS_COLORS['secondary'], AWS_COLORS['accent2'], 
                AWS_COLORS['primary'], 'green', 'purple'
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 When to use different learning rates?"):
        st.markdown("""
        ### Learning Rate Selection Guidelines
        
        | Scenario | Suggested Learning Rate | Reason |
        |----------|-------------------------|--------|
        | Initial exploration | Medium (0.01 - 0.1) | Provides a good balance to start with |
        | Fine-tuning | Small (0.0001 - 0.001) | Allows for precise refinement of weights |
        | Large dataset | Smaller rates | More stable updates due to many examples |
        | Small dataset | Larger rates | Helps escape local minima |
        | Deep networks | Smaller rates | Prevents exploding gradients |
        | Transfer learning | Very small rates | Preserves pre-trained knowledge |
        
        **Learning Rate Finder**: Modern practice often involves using a learning rate finder that tests multiple learning rates in a single run to identify the optimal rate.
        """)

# Tab 3: Early Stopping
with tabs[2]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='subtitle'>What is Early Stopping?</div>", unsafe_allow_html=True)
        st.markdown("""
        Early stopping is a regularization technique that stops training when the model's performance on a validation dataset stops improving, preventing overfitting.
        
        **Key benefits:**
        - Prevents overfitting
        - Saves computational resources
        - Helps find optimal training duration
        - Works well with other regularization techniques
        """)
    
    with col2:
        # Display a relevant animation
        lottie_es = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_ot5guwfo.json")
        if lottie_es:
            st_lottie(lottie_es, height=200, key="es_animation")
        else:
            st.image("https://machinelearningmastery.com/wp-content/uploads/2019/09/Example-of-a-Learning-Curve-Showing-an-Overfit-Model.png", caption="Early stopping concept")

    st.markdown("---")
    st.markdown("<div class='subtitle'>Interactive Demo: Early Stopping in Action</div>", unsafe_allow_html=True)
    
    # Early stopping visualization demo
    if st.session_state.early_stopping['data'] is None:
        # Generate more complex regression data
        np.random.seed(42)
        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        y = 0.5 * X.ravel() ** 3 - X.ravel() ** 2 + 2 * X.ravel() + 2 + np.random.normal(0, 3, size=X.shape[0])
        
        # Add polynomial features to make it more complex
        X_poly = np.column_stack([X, X**2, X**3, X**4, X**5])
        
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
        
        st.session_state.early_stopping['data'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_orig': X  # original feature for plotting
        }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='highlight'>See how early stopping prevents overfitting by monitoring validation performance.</div>", unsafe_allow_html=True)
        
        # Parameters for training
        max_epochs = st.slider("Maximum number of epochs", 50, 500, 200, step=10, key="es_max_epochs")
        patience = st.slider("Patience (epochs before early stopping)", 5, 50, 20, step=5, key="es_patience")
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            value=0.01,
            format_func=lambda x: f"{x:.4f}",
            key="es_lr"
        )
        
        if st.button("Train with Early Stopping", key="train_es_model"):
            with st.spinner("Training model with early stopping..."):
                # Get data from session state
                data = st.session_state.early_stopping['data']
                X_train, y_train = data['X_train'], data['y_train']
                X_test, y_test = data['X_test'], data['y_test']
                
                # Initialize model
                model = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=1, random_state=42)
                
                # Training with early stopping
                train_losses = []
                val_losses = []
                
                best_val_loss = float('inf')
                best_epoch = 0
                best_model_coef = None
                best_model_intercept = None
                
                for epoch in range(max_epochs):
                    # Train for one epoch
                    model.partial_fit(X_train, y_train)
                    
                    # Calculate losses
                    train_pred = model.predict(X_train)
                    train_loss = mean_squared_error(y_train, train_pred)
                    
                    val_pred = model.predict(X_test)
                    val_loss = mean_squared_error(y_test, val_pred)
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    
                    # Check if this is the best model so far
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_model_coef = model.coef_.copy()
                        best_model_intercept = model.intercept_
                    
                    # Early stopping check
                    if epoch - best_epoch >= patience:
                        break
                
                # Store results in session state
                st.session_state.early_stopping['history'] = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_epoch': best_epoch,
                    'final_epoch': epoch,
                    'best_val_loss': best_val_loss
                }
                
                # Store best model
                st.session_state.early_stopping['best_model'] = {
                    'coef': best_model_coef,
                    'intercept': best_model_intercept
                }
    
    with col2:
        # Display results if model has been trained
        if 'history' in st.session_state.early_stopping and isinstance(st.session_state.early_stopping['history'], dict):
            st.markdown("<div class='subtitle'>Training Results with Early Stopping</div>", unsafe_allow_html=True)
            
            history = st.session_state.early_stopping['history']
            
            # Create training curve plot
            epochs = list(range(1, len(history.get('train_losses', [])) + 1))
            df = pd.DataFrame({
                'Epoch': epochs,
                'Training Loss': history['train_losses'],
                'Validation Loss': history['val_losses']
            })
            
            fig = px.line(
                df, x='Epoch', y=['Training Loss', 'Validation Loss'],
                title=f'Loss Curves with Early Stopping',
                template='plotly_white',
                color_discrete_sequence=[AWS_COLORS['secondary'], AWS_COLORS['accent2']]
            )
            
            # Add vertical line for best epoch
            best_epoch = history['best_epoch']
            fig.add_vline(x=best_epoch + 1, line_dash="dash", line_color="green", 
                         annotation_text=f"Best Epoch: {best_epoch + 1}", 
                         annotation_position="top right")
            
            # Add vertical line for final epoch if stopped early
            final_epoch = history['final_epoch']
            if final_epoch < max_epochs - 1:
                fig.add_vline(x=final_epoch + 1, line_dash="dot", line_color="red",
                             annotation_text="Stopped Training", 
                             annotation_position="bottom right")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Epoch", f"{history['best_epoch'] + 1}")
            col2.metric("Training Epochs", f"{history['final_epoch'] + 1}")
            col3.metric("Best Validation Loss", f"{history['best_val_loss']:.4f}")
            
            if history['final_epoch'] < max_epochs - 1:
                st.success(f"Early stopping activated! Saved {max_epochs - history['final_epoch'] - 1} epochs of unnecessary training.")
            else:
                st.info("Training completed without early stopping. Try increasing max epochs or decreasing patience.")
        else:
            st.info("Train a model to see early stopping in action!")

    st.markdown("---")
    
    # Show model fit visualization
    if 'best_model' in st.session_state.early_stopping:
        st.markdown("<div class='subtitle'>Model Fit Visualization</div>", unsafe_allow_html=True)
        
        data = st.session_state.early_stopping['data']
        X_orig = data['X_orig']
        
        # Create polynomial features for prediction
        X_poly_viz = np.column_stack([X_orig, X_orig**2, X_orig**3, X_orig**4, X_orig**5])
        
        # Get best model from session state
        best_model = st.session_state.early_stopping['best_model']
        
        # Function to make predictions with a model
        def predict_with_model(X, coef, intercept):
            return X.dot(coef) + intercept
        
        # Create predictions with best model
        best_preds = predict_with_model(X_poly_viz, best_model['coef'], best_model['intercept'])
        
        # Create a plot showing the data and model fit
        fig = go.Figure()
        
        # Add the original data points
        fig.add_trace(go.Scatter(
            x=X_orig.flatten(),
            y=data['y_train'],
            mode='markers',
            name='Training Data',
            marker=dict(color=AWS_COLORS['primary'])
        ))
        
        # Add the best model prediction
        fig.add_trace(go.Scatter(
            x=X_orig.flatten(),
            y=best_preds,
            mode='lines',
            name='Best Model (Early Stopping)',
            line=dict(color=AWS_COLORS['secondary'], width=3)
        ))
        
        fig.update_layout(
            title='Model Fit with Early Stopping',
            xaxis_title='Feature Value',
            yaxis_title='Target',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("🎯 How to implement Early Stopping"):
            st.markdown("""
            ### Early Stopping Implementation Steps
            
            1. **Split your data** into training, validation, and test sets
            2. **Define a patience parameter** - how many epochs to wait for improvement
            3. **Track validation loss** during training
            4. **Save model** whenever validation performance improves
            5. **Stop training** when no improvement for 'patience' epochs
            6. **Return best model** from saved checkpoint
            
            ```python
            # Pseudocode for early stopping
            best_val_loss = float('inf')
            patience = 20
            wait = 0
            best_model = None
            
            for epoch in range(max_epochs):
                # Train for one epoch
                train_model(model, train_data)
                
                # Evaluate on validation set
                val_loss = evaluate(model, val_data)
                
                # Check if improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    best_model = copy_model(model)
                else:
                    wait += 1
                
                # Check early stopping condition
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Return best model
            return best_model
            ```
            """)
    
    with col2:
        with st.expander("📊 Early Stopping vs Other Regularization"):
            st.markdown("""
            ### Early Stopping vs Other Regularization Techniques
            
            | Technique | How it Works | Pros | Cons |
            |-----------|-------------|------|------|
            | **Early Stopping** | Stops training when validation loss stops improving | Easy to implement, Computationally efficient | Requires validation set, May be sensitive to random variations |
            | **L1/L2 Regularization** | Adds penalty term to loss based on weights | Can induce sparsity (L1), Smoother solutions | Requires tuning regularization strength |
            | **Dropout** | Randomly deactivates neurons during training | Simple, works well for deep networks | May require longer training time |
            | **Batch Normalization** | Normalizes layer inputs | Speeds up training, Reduces sensitivity to initialization | Adds complexity, Can interact with other regularization techniques |
            
            Early stopping is often used together with other regularization techniques for better results!
            """)
            
            # Show a comparison image
            st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*0KMm0nouBZJZwvUVi2T2uw.png", caption="Early stopping can prevent overfitting")

# Add a footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
ML Concepts Explorer © 2025 | Built with Streamlit | Images and content for educational purposes only
</div>
""", unsafe_allow_html=True)
