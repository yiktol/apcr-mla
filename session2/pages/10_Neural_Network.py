import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import base64
from PIL import Image
import io
import seaborn as sns
from streamlit_lottie import st_lottie
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Neural Network Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- AWS Color Scheme ----
AWS_COLORS = {
    'orange': '#FF9900',
    'dark_blue': '#232F3E',
    'light_blue': '#1E88E5',
    'light_grey': '#F2F2F2',
    'dark_grey': '#545B64',
    'white': '#FFFFFF',
    'teal': '#007E99'
}

# ---- Custom CSS for AWS style ----
st.markdown("""
<style>
    /* AWS color scheme */
    .stApp {
        background-color: #FFFFFF;
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
    h1, h2, h3, h4 {
        color: #232F3E;
    }
    .highlight {
        background-color: #FF9900;
        border-radius: 4px;
        padding: 0.25em 0.5em;
    }
    .stButton>button {
        background-color: #FF9900;
        color: #232F3E;
        font-weight: bold;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #E88E00;
    }
    # .stSidebar {
    #     background-color: #232F3E;
    #     color: white;
    # }
    footer {
        font-size: 0.8rem;
        padding: 10px;
        text-align: center;
        border-top: 1px solid #ddd;
    }
    /* Responsive layout */
    @media (max-width: 768px) {
        .responsive-container {
            flex-direction: column !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---- Session Management ----
def initialize_session_state():
    if "neural_network" not in st.session_state:
        st.session_state.neural_network = None
    if "train_losses" not in st.session_state:
        st.session_state.train_losses = []
    if "input_features" not in st.session_state:
        st.session_state.input_features = 2
    if "hidden_neurons" not in st.session_state:
        st.session_state.hidden_neurons = 5
    if "dataset_type" not in st.session_state:
        st.session_state.dataset_type = "moons"
    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False
    if "trained_epochs" not in st.session_state:
        st.session_state.trained_epochs = 0
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None

# Initialize session state
initialize_session_state()


# Session management in sidebar
st.sidebar.header("Session Management")
if st.sidebar.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()
    st.sidebar.success("Session has been reset!")

# ---- Helper Functions ----
def load_lottie_url(url: str):
    """Load a Lottie animation file from URL"""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot the decision boundary for a neural network model"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).detach().numpy()
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu_r)
    ax.set_xlabel('Feature 1', fontsize=14)
    ax.set_ylabel('Feature 2', fontsize=14)
    ax.set_title(title, fontsize=16)
    return fig

def create_dataset(dataset_type, samples=1000):
    """Create a dataset for classification"""
    if dataset_type == "moons":
        X, y = make_moons(n_samples=samples, noise=0.2, random_state=42)
    elif dataset_type == "circles":
        X, y = make_circles(n_samples=samples, noise=0.2, factor=0.5, random_state=42)
    else:  # blobs
        X, y = make_classification(n_samples=samples, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=42, 
                                   n_clusters_per_class=1)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

# ---- Neural Network Model ----
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        output = self.sigmoid(self.fc2(hidden))
        return output
    
    def get_weights(self):
        """Return the weights of the network for visualization"""
        weights = {
            'w1': self.fc1.weight.detach().numpy(),
            'b1': self.fc1.bias.detach().numpy(),
            'w2': self.fc2.weight.detach().numpy(),
            'b2': self.fc2.bias.detach().numpy()
        }
        return weights

# ---- Main Application ----
# Title and description
st.title("🧠 Neural Network")
st.markdown("""
This interactive application helps you understand how neural networks work through visualizations and hands-on examples.
Explore each tab to learn different aspects of neural networks!
""")

# Tab-based navigation with emojis
tabs = st.tabs([
    "📚 Introduction", 
    "🧩 Building Blocks", 
    "🔄 Training Process", 
    "🛠️ Interactive Playground",
    "🔍 Advanced Concepts"
])

# ---- Tab 1: Introduction ----
with tabs[0]:
    st.header("📚 Introduction to Neural Networks")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### What are Neural Networks?
        
        Neural networks are computational systems inspired by the structure and function of the human brain. 
        They consist of interconnected nodes (or "neurons") that work together to process information and learn patterns.
        
        ### Key Features of Neural Networks:
        
        - **Learning from data**: They adjust themselves based on examples
        - **Pattern recognition**: They can identify complex patterns in data
        - **Non-linear relationships**: They can model complex non-linear relationships
        - **Generalization**: They can apply learning to new, unseen data
        
        ### Applications of Neural Networks:
        
        - 🖼️ Image recognition and computer vision
        - 🗣️ Natural language processing
        - 🎮 Gaming and reinforcement learning
        - 📈 Financial prediction and risk assessment
        - 🏥 Medical diagnostics
        """)
    
    with col2:
        # Load and display a Lottie animation
        brain_animation = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_UJNc2t.json")
        if brain_animation:
            st_lottie(brain_animation, speed=1, height=400, key="brain_animation")
        else:
            st.image("https://cdn.pixabay.com/photo/2017/09/05/11/37/ai-2717282_1280.jpg", 
                     caption="Neural Network Visualization")
    
    st.markdown("---")
    
    st.subheader("From Biological to Artificial Neurons")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Biological Neurons
        
        In the human brain:
        - Dendrites receive signals from other neurons
        - The cell body processes these signals
        - If the combined signal is strong enough, the neuron fires
        - The axon transmits the signal to other neurons
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Neuron.svg/1200px-Neuron.svg.png", 
                 caption="Biological Neuron", width=400)
    
    with col2:
        st.markdown("""
        ### Artificial Neurons
        
        In neural networks:
        - **Inputs** (x₁, x₂, ...) represent features or signals
        - **Weights** (w₁, w₂, ...) determine the importance of each input
        - **Bias** adds a constant term for flexibility
        - **Activation function** determines the output based on the weighted sum
        """)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw neuron body
        circle = plt.Circle((0.5, 0.5), 0.2, fill=True, color=AWS_COLORS['light_blue'], alpha=0.7)
        ax.add_patch(circle)
        
        # Draw inputs
        ax.arrow(0.1, 0.7, 0.2, -0.1, head_width=0.03, head_length=0.03, fc=AWS_COLORS['orange'], ec=AWS_COLORS['orange'], lw=2)
        ax.arrow(0.1, 0.5, 0.2, 0, head_width=0.03, head_length=0.03, fc=AWS_COLORS['orange'], ec=AWS_COLORS['orange'], lw=2)
        ax.arrow(0.1, 0.3, 0.2, 0.1, head_width=0.03, head_length=0.03, fc=AWS_COLORS['orange'], ec=AWS_COLORS['orange'], lw=2)
        
        # Draw output
        ax.arrow(0.7, 0.5, 0.2, 0, head_width=0.03, head_length=0.03, fc=AWS_COLORS['dark_blue'], ec=AWS_COLORS['dark_blue'], lw=2)
        
        # Labels
        ax.text(0.05, 0.7, 'x₁', fontsize=14)
        ax.text(0.05, 0.5, 'x₂', fontsize=14)
        ax.text(0.05, 0.3, 'x₃', fontsize=14)
        ax.text(0.5, 0.5, 'Σ', fontsize=18, ha='center', va='center', color='white')
        ax.text(0.95, 0.5, 'Output', fontsize=14, ha='right')
        ax.text(0.3, 0.65, 'w₁', fontsize=10, color=AWS_COLORS['dark_blue'])
        ax.text(0.3, 0.52, 'w₂', fontsize=10, color=AWS_COLORS['dark_blue'])
        ax.text(0.3, 0.38, 'w₃', fontsize=10, color=AWS_COLORS['dark_blue'])
        
        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        st.pyplot(fig)
        
        st.markdown("""
        The artificial neuron calculates: `output = activation_function(w₁x₁ + w₂x₂ + ... + bias)`
        """)
    
    st.markdown("---")
    
    st.subheader("Common Activation Functions")
    
    col1, col2, col3 = st.columns(3)
    
    x = np.linspace(-5, 5, 100)
    
    with col1:
        st.markdown("### Sigmoid")
        st.markdown("Maps input to a value between 0 and 1")
        st.markdown("Formula: `σ(x) = 1 / (1 + e^(-x))`")
        
        fig, ax = plt.subplots()
        ax.plot(x, 1 / (1 + np.exp(-x)), color=AWS_COLORS['orange'], lw=3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_title('Sigmoid Function')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### ReLU")
        st.markdown("Rectified Linear Unit - Simple and efficient")
        st.markdown("Formula: `ReLU(x) = max(0, x)`")
        
        fig, ax = plt.subplots()
        ax.plot(x, np.maximum(0, x), color=AWS_COLORS['light_blue'], lw=3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_title('ReLU Function')
        st.pyplot(fig)
        
    with col3:
        st.markdown("### Tanh")
        st.markdown("Maps input to a value between -1 and 1")
        st.markdown("Formula: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`")
        
        fig, ax = plt.subplots()
        ax.plot(x, np.tanh(x), color=AWS_COLORS['teal'], lw=3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_title('Tanh Function')
        st.pyplot(fig)

# ---- Tab 2: Building Blocks ----
with tabs[1]:
    st.header("🧩 Building Blocks of Neural Networks")
    
    st.subheader("Neural Network Architecture")
    
    # Create a basic neural network diagram
    st.markdown("""
    Neural networks consist of layers of interconnected neurons. 
    The three main types of layers are:
    
    1. **Input Layer**: Receives the initial data
    2. **Hidden Layer(s)**: Process the information
    3. **Output Layer**: Produces the final result
    """)
    
    # Neural Network Architecture Visualization
    def plot_nn_architecture(input_size=3, hidden_sizes=[4], output_size=2):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine number of layers
        n_layers = len(hidden_sizes) + 2  # input + hidden + output
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Colors
        input_color = AWS_COLORS['light_blue']
        hidden_color = AWS_COLORS['orange']
        output_color = AWS_COLORS['teal']
        
        # Max layer height
        max_neurons = max(layer_sizes)
        vertical_distance = 0.8 / max_neurons
        
        # Position layers horizontally
        horizontal_distance = 1.0 / (n_layers - 1)
        
        # Draw neurons and connections
        layer_positions = []
        
        for l, layer_size in enumerate(layer_sizes):
            x = l * horizontal_distance
            
            # Save neuron positions for this layer
            neurons = []
            
            # Place neurons in this layer
            for n in range(layer_size):
                if layer_size == 1:
                    y = 0.5
                else:
                    y = 0.1 + (n * vertical_distance * max_neurons / layer_size)
                
                # Choose color based on layer
                if l == 0:
                    color = input_color
                elif l == len(layer_sizes) - 1:
                    color = output_color
                else:
                    color = hidden_color
                
                # Draw neuron
                circle = plt.Circle((x, y), 0.02, fill=True, color=color)
                ax.add_patch(circle)
                
                # Label
                if l == 0:
                    ax.text(x - 0.05, y, f"x{n+1}", ha='right', va='center')
                elif l == len(layer_sizes) - 1:
                    ax.text(x + 0.05, y, f"y{n+1}", ha='left', va='center')
                
                neurons.append((x, y))
            
            layer_positions.append(neurons)
        
        # Draw connections between layers
        for l in range(len(layer_sizes) - 1):
            for i, (x1, y1) in enumerate(layer_positions[l]):
                for j, (x2, y2) in enumerate(layer_positions[l + 1]):
                    ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5)
        
        # Add layer labels
        for l in range(n_layers):
            x = l * horizontal_distance
            if l == 0:
                ax.text(x, 0.02, "Input\nLayer", ha='center', va='center')
            elif l == n_layers - 1:
                ax.text(x, 0.02, "Output\nLayer", ha='center', va='center')
            else:
                ax.text(x, 0.02, f"Hidden\nLayer {l}", ha='center', va='center')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_nn_architecture(input_size=4, hidden_sizes=[5, 3], output_size=2)
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        ### Architecture Components:
        
        - **Neurons**: Individual processing units
        - **Connections**: Pathways between neurons (with weights)
        - **Layers**: Groups of neurons that process information together
        - **Weights**: Parameters that determine the strength of connections
        - **Biases**: Additional parameters that shift the activation function
        
        Most neural networks follow a **feedforward** architecture, where information flows from input to output without cycles.
        """)
    
    st.markdown("---")
    
    st.subheader("Mathematical Foundation")
    
    st.markdown("""
    ### Forward Propagation
    
    This is how a neural network processes input data to make predictions:
    
    1. At each layer, compute the weighted sum of inputs
    2. Apply an activation function
    3. Pass the result to the next layer
    
    For a single neuron:
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.latex(r'''
        z = \sum_{i=1}^{n} w_i x_i + b
        ''')
        st.markdown("Where:")
        st.markdown("- $z$ is the weighted sum")
        st.markdown("- $w_i$ are the weights")
        st.markdown("- $x_i$ are the inputs")
        st.markdown("- $b$ is the bias term")
    
    with col2:
        st.latex(r'''
        a = \sigma(z)
        ''')
        st.markdown("Where:")
        st.markdown("- $a$ is the activation (output)")
        st.markdown("- $\sigma$ is the activation function")
        st.markdown("- $z$ is the weighted sum from step 1")
    
    # Interactive forward propagation example
    st.markdown("---")
    
    st.subheader("Interactive Forward Propagation Example")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Adjust Input Values")
        x1 = st.slider("Input x₁", -10.0, 10.0, 1.0, 0.1)
        x2 = st.slider("Input x₂", -10.0, 10.0, 2.0, 0.1)
        
        st.markdown("### Adjust Weights and Bias")
        w1 = st.slider("Weight w₁", -2.0, 2.0, 0.5, 0.1)
        w2 = st.slider("Weight w₂", -2.0, 2.0, -0.5, 0.1)
        bias = st.slider("Bias", -5.0, 5.0, 0.0, 0.1)
        
        activation_fn = st.selectbox(
            "Activation Function",
            ["Sigmoid", "ReLU", "Tanh"]
        )
    
    with col2:
        # Calculate neuron's output
        z = w1 * x1 + w2 * x2 + bias
        
        if activation_fn == "Sigmoid":
            a = 1 / (1 + np.exp(-z))
            fn_name = "sigmoid"
        elif activation_fn == "ReLU":
            a = max(0, z)
            fn_name = "ReLU"
        else:  # Tanh
            a = np.tanh(z)
            fn_name = "tanh"
        
        st.markdown("### Neuron Computation")
        st.markdown(f"""
        **Step 1**: Calculate weighted sum (z)  
        $z = w_1 x_1 + w_2 x_2 + bias$  
        $z = ({w1:.2f} \\times {x1:.2f}) + ({w2:.2f} \\times {x2:.2f}) + {bias:.2f}$  
        $z = {z:.4f}$
        
        **Step 2**: Apply activation function  
        $a = {fn_name}(z)$  
        $a = {fn_name}({z:.4f})$  
        $a = {a:.4f}$
        """)
        
        # Visualize the neuron
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw neuron body
        circle = plt.Circle((0.5, 0.5), 0.15, fill=True, color=AWS_COLORS['light_blue'], alpha=0.7)
        ax.add_patch(circle)
        
        # Draw inputs
        ax.arrow(0.1, 0.7, 0.25, -0.1, head_width=0.03, head_length=0.03, fc=AWS_COLORS['orange'], ec=AWS_COLORS['orange'], lw=2)
        ax.arrow(0.1, 0.3, 0.25, 0.1, head_width=0.03, head_length=0.03, fc=AWS_COLORS['orange'], ec=AWS_COLORS['orange'], lw=2)
        
        # Draw output
        ax.arrow(0.65, 0.5, 0.2, 0, head_width=0.03, head_length=0.03, fc=AWS_COLORS['dark_blue'], ec=AWS_COLORS['dark_blue'], lw=2)
        
        # Labels
        ax.text(0.05, 0.7, f'x₁ = {x1:.2f}', fontsize=12)
        ax.text(0.05, 0.3, f'x₂ = {x2:.2f}', fontsize=12)
        ax.text(0.5, 0.5, f'Σ', fontsize=14, ha='center', va='center', color='white')
        ax.text(0.9, 0.5, f'a = {a:.4f}', fontsize=12, ha='center')
        
        ax.text(0.3, 0.65, f'w₁ = {w1:.2f}', fontsize=10, color=AWS_COLORS['dark_blue'])
        ax.text(0.3, 0.35, f'w₂ = {w2:.2f}', fontsize=10, color=AWS_COLORS['dark_blue'])
        ax.text(0.5, 0.33, f'bias = {bias:.2f}', fontsize=10, color=AWS_COLORS['dark_grey'])
        
        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        st.pyplot(fig)
    
    # Provide information about layers
    st.markdown("---")
    st.subheader("Types of Neural Network Layers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Dense (Fully Connected)")
        st.markdown("""
        - Each neuron connected to all neurons in previous layer
        - Versatile but can have many parameters
        - Used in most basic neural networks
        """)
        st.latex(r"y = \sigma(Wx + b)")
    
    with col2:
        st.markdown("### Convolutional")
        st.markdown("""
        - Specialized for grid-like data (images)
        - Uses filters to detect patterns
        - Greatly reduces parameters
        - Core of computer vision networks
        """)
        st.image("https://miro.medium.com/max/1400/1*ciDgQEjViWLnCbmX-EeSrA.gif", 
                 caption="Convolution Operation", width=250)
    
    with col3:
        st.markdown("### Recurrent")
        st.markdown("""
        - Has connections that form cycles
        - Can process sequential data
        - Maintains internal memory
        - Used for text, time series, speech
        """)
        st.image("https://miro.medium.com/max/1000/1*sX6T0Y4_95Hm2Q3ICsCYig.gif", 
                 caption="Recurrent Network", width=250)

# ---- Tab 3: Training Process ----
with tabs[2]:
    st.header("🔄 Training Neural Networks")
    
    st.markdown("""
    Training a neural network involves adjusting its weights and biases to minimize prediction errors. 
    This process consists of several key components:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loss Functions")
        st.markdown("""
        Loss functions measure how far the network's predictions are from the actual values. Common loss functions include:
        
        - **Mean Squared Error (MSE)**: For regression problems
        - **Binary Cross-Entropy**: For binary classification
        - **Categorical Cross-Entropy**: For multi-class classification
        """)
        
        # Visualize MSE
        x = np.linspace(-3, 3, 100)
        y_mse = x**2
        
        fig, ax = plt.subplots()
        ax.plot(x, y_mse, color=AWS_COLORS['light_blue'], lw=3)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Loss Value')
        ax.set_title('Mean Squared Error Loss')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Optimization Algorithms")
        st.markdown("""
        Optimizers determine how to update the weights based on the calculated gradients:
        
        - **Gradient Descent**: Basic algorithm that follows the negative gradient
        - **Stochastic Gradient Descent (SGD)**: Uses random subsets of data
        - **Adam**: Adaptive algorithm with momentum and bias correction
        - **RMSprop**: Uses a moving average of squared gradients
        """)
        
        # Simple animation of gradient descent
        def plot_gradient_descent():
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create a simple function (bowl shaped)
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            Z = X**2 + Y**2
            
            # Plot contour
            contour = ax.contour(X, Y, Z, levels=20, cmap='Blues', alpha=0.8)
            ax.clabel(contour, inline=True, fontsize=8)
            
            # Plot gradient descent path
            path_x = [-4]
            path_y = [4]
            learning_rate = 0.3
            
            for _ in range(10):
                # Compute gradient
                grad_x = 2 * path_x[-1]
                grad_y = 2 * path_y[-1]
                
                # Update position
                new_x = path_x[-1] - learning_rate * grad_x
                new_y = path_y[-1] - learning_rate * grad_y
                
                path_x.append(new_x)
                path_y.append(new_y)
            
            ax.plot(path_x, path_y, 'ro-', markersize=8, linewidth=2)
            ax.set_xlabel('Weight 1')
            ax.set_ylabel('Weight 2')
            ax.set_title('Gradient Descent Optimization')
            return fig
        
        st.pyplot(plot_gradient_descent())
    
    st.markdown("---")
    
    st.subheader("Backpropagation: How Neural Networks Learn")
    
    st.markdown("""
    Backpropagation is the key algorithm for training neural networks. It consists of two main phases:
    
    1. **Forward Pass**: Compute predictions and loss
    2. **Backward Pass**: Compute gradients and update weights
    """)
    
    # Backpropagation animation
    backprop_lottie = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_0fVTtS.json")
    if backprop_lottie:
        st_lottie(backprop_lottie, speed=1, height=400, key="backprop_animation")
    
    st.markdown("""
    ### The Backpropagation Algorithm:
    
    1. **Forward propagation** to compute the output and error
    2. **Compute output layer gradients**
    3. **Propagate gradients backward** through the network
    4. **Update all weights and biases** using the computed gradients
    """)
    
    st.markdown("---")
    
    st.subheader("Interactive Learning Process Visualization")
    
    # Animated training visualization
    st.markdown("This animation shows how a neural network learns to classify data points over time:")
    
    # Create dataset
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.5).astype(float)
    
    # Split data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Create animation frames
    epochs = [0, 5, 20, 100]
    
    def create_model_at_epoch(epoch):
        # Create and train model
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
        if epoch > 0:
            # Training logic
            criterion = nn.BCELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.1)
            
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            
            for e in range(epoch):
                # Forward pass
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return model
    
    # Plot for each epoch
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, epoch in enumerate(epochs):
        ax = axes[i]
        model = create_model_at_epoch(epoch)
        
        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Predict on mesh points
        with torch.no_grad():
            inputs = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
            Z = model(inputs).numpy().reshape(xx.shape)
        
        # Plot decision boundary and data points
        ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.6)
        ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='red', label='Class 0')
        ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='blue', label='Class 1')
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f'After {epoch} epochs')
        if i == 0:
            ax.legend()
    
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Observations:
    
    - At 0 epochs (untrained), the decision boundary is simply a straight line
    - After 5 epochs, the network begins to curve the decision boundary
    - After 20 epochs, the circle shape becomes more defined
    - After 100 epochs, the network has learned the circular decision boundary well
    
    This demonstrates how neural networks progressively learn complex patterns from data through many iterations of training.
    """)
    
    st.markdown("---")
    
    st.subheader("Challenges in Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overfitting")
        st.markdown("""
        **Overfitting** occurs when a model performs well on training data but poorly on new data.
        
        **Solutions:**
        - Get more training data
        - Use regularization (L1, L2)
        - Implement dropout
        - Early stopping
        - Data augmentation
        """)
        
        # Visualize overfitting
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 50)
        y_true = 0.5 * x + np.sin(x) + np.random.normal(0, 0.5, size=len(x))
        
        # Simple model (underfit)
        p_underfit = np.polyfit(x, y_true, 1)
        y_underfit = np.polyval(p_underfit, x)
        
        # Complex model (overfit)
        p_overfit = np.polyfit(x, y_true, 15)
        y_overfit = np.polyval(p_overfit, x)
        
        # Good model
        p_good = np.polyfit(x, y_true, 3)
        y_good = np.polyval(p_good, x)
        
        ax.scatter(x, y_true, alpha=0.7, label='Data')
        ax.plot(x, y_underfit, 'r-', label='Underfit', linewidth=2)
        ax.plot(x, y_good, 'g-', label='Good fit', linewidth=2)
        ax.plot(x, y_overfit, 'b-', label='Overfit', linewidth=2)
        
        ax.legend()
        ax.set_title('Underfitting vs Overfitting')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Vanishing/Exploding Gradients")
        st.markdown("""
        **Vanishing gradients**: Gradients become very small, making learning slow or impossible.
        
        **Exploding gradients**: Gradients become extremely large, causing unstable updates.
        
        **Solutions:**
        - Use ReLU activation instead of sigmoid/tanh
        - Implement gradient clipping
        - Use batch normalization
        - Apply proper weight initialization
        """)
        
        # Visualize vanishing gradient
        fig, ax = plt.subplots()
        x = np.linspace(-6, 6, 100)
        sigmoid = 1 / (1 + np.exp(-x))
        sigmoid_derivative = sigmoid * (1 - sigmoid)
        
        ax.plot(x, sigmoid, 'b-', label='Sigmoid', linewidth=2)
        ax.plot(x, sigmoid_derivative, 'r-', label='Derivative', linewidth=2)
        ax.fill_between(x, sigmoid_derivative, alpha=0.2, color='r')
        
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title('Vanishing Gradient Problem')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        Notice how the derivative of the sigmoid function (red) approaches zero when input values are very large or very small,
        causing gradients to vanish during backpropagation.
        """)

# ---- Tab 4: Interactive Playground ----
with tabs[3]:
    st.header("🛠️ Neural Network Interactive Playground")
    
    st.markdown("""
    Build and train your own neural network to see how it learns to classify different datasets.
    Experiment with network architecture and training parameters!
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        
        # Dataset selection
        dataset_type = st.selectbox(
            "Select Dataset",
            ["moons", "circles", "blobs"],
            index=["moons", "circles", "blobs"].index(st.session_state.dataset_type)
        )
        
        if dataset_type != st.session_state.dataset_type:
            st.session_state.dataset_type = dataset_type
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = create_dataset(dataset_type)
            st.session_state.neural_network = None
            st.session_state.trained_epochs = 0
            st.session_state.train_losses = []
        
        # If dataset not created yet, create it
        if st.session_state.X_train is None:
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = create_dataset(dataset_type)
        
        # Network architecture
        st.markdown("### Network Architecture")
        hidden_neurons = st.slider("Number of Hidden Neurons", 1, 20, st.session_state.hidden_neurons)
        if hidden_neurons != st.session_state.hidden_neurons:
            st.session_state.hidden_neurons = hidden_neurons
            st.session_state.neural_network = None
            st.session_state.trained_epochs = 0
            st.session_state.train_losses = []
        
        # Training parameters
        st.markdown("### Training Parameters")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.01, 0.05, 0.1, 0.5],
            value=0.1
        )
        
        epochs = st.slider("Number of Epochs", 1, 100, 10)
        
        # Initialize model if needed
        if st.session_state.neural_network is None:
            st.session_state.neural_network = SimpleNN(
                input_size=2,
                hidden_size=hidden_neurons,
                output_size=1
            )
        
        # Training button
        start_training = st.button("Train Network")
    
    with col2:
        st.subheader("Dataset Visualization")
        
        # Visualize the dataset
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            st.session_state.X_train[:, 0],
            st.session_state.X_train[:, 1],
            c=st.session_state.y_train,
            cmap=plt.cm.RdYlBu_r,
            edgecolor='k',
            s=60,
            alpha=0.7
        )
        ax.set_title(f"Dataset: {dataset_type.capitalize()}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Train the model if requested
    if start_training:
        st.session_state.training_in_progress = True
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(st.session_state.X_train)
        y_tensor = torch.FloatTensor(st.session_state.y_train).unsqueeze(1)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(st.session_state.neural_network.parameters(), lr=learning_rate)
        
        # Training progress container
        progress_container = st.empty()
        loss_container = st.empty()
        
        losses = []
        
        # Training loop
        for i in range(epochs):
            # Forward pass
            outputs = st.session_state.neural_network(X_tensor)
            loss = criterion(outputs, y_tensor)
            losses.append(loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            progress_container.progress((i + 1) / epochs)
            loss_container.markdown(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}")
            
            # Sleep a bit to show progress
            time.sleep(0.05)
        
        st.session_state.train_losses.extend(losses)
        st.session_state.trained_epochs += epochs
        st.session_state.training_in_progress = False
        
        # Display completion message
        st.success(f"Training completed! The network has been trained for {st.session_state.trained_epochs} epochs in total.")
    
    # Display results
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Decision Boundary")
        
        if st.session_state.neural_network is not None and st.session_state.trained_epochs > 0:
            # Plot decision boundary
            fig = plot_decision_boundary(
                st.session_state.neural_network,
                np.vstack((st.session_state.X_train, st.session_state.X_test)),
                np.hstack((st.session_state.y_train, st.session_state.y_test)),
                f"Decision Boundary after {st.session_state.trained_epochs} epochs"
            )
            st.pyplot(fig)
        else:
            st.info("Train the model to see the decision boundary.")
    
    with col2:
        st.subheader("Training Loss")
        
        if len(st.session_state.train_losses) > 0:
            # Plot loss curve
            fig, ax = plt.subplots()
            ax.plot(range(1, len(st.session_state.train_losses) + 1), st.session_state.train_losses, marker='o', linestyle='-', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss over Epochs')
            ax.grid(True, alpha=0.3)
            
            # If many epochs, use log scale for y-axis
            if max(st.session_state.train_losses) / min(st.session_state.train_losses) > 100:
                ax.set_yscale('log')
            
            st.pyplot(fig)
        else:
            st.info("Train the model to see the loss curve.")
        
        # Display model evaluation
        if st.session_state.neural_network is not None and st.session_state.trained_epochs > 0:
            # Convert test data to PyTorch tensors
            X_test_tensor = torch.FloatTensor(st.session_state.X_test)
            
            # Make predictions
            with torch.no_grad():
                y_pred = st.session_state.neural_network(X_test_tensor).numpy().flatten()
            
            # Convert predictions to binary
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred_binary == st.session_state.y_test)
            
            st.markdown("### Model Evaluation")
            st.markdown(f"**Test Accuracy:** {accuracy:.2%}")
            
            # Create a confusion matrix
            conf_matrix = pd.crosstab(
                pd.Series(st.session_state.y_test, name='Actual'),
                pd.Series(y_pred_binary, name='Predicted')
            )
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    
    st.markdown("---")
    
    # Network visualization
    st.subheader("Neural Network Visualization")
    
    if st.session_state.neural_network is not None:
        # Get the weights
        weights = st.session_state.neural_network.get_weights()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Visualize network architecture
            st.markdown("### Network Architecture")
            
            fig = plot_nn_architecture(
                input_size=2, 
                hidden_sizes=[st.session_state.hidden_neurons], 
                output_size=1
            )
            st.pyplot(fig)
        
        with col2:
            # Visualize weights
            st.markdown("### Weight Visualization")
            st.markdown("The size of each connection represents the absolute weight value. Blue indicates positive weights, red indicates negative weights.")
            
            # Create weight visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Number of nodes in each layer
            n_input = 2
            n_hidden = st.session_state.hidden_neurons
            n_output = 1
            
            # Position nodes
            input_pos = [(0.1, 0.2 + 0.6 * i / (n_input-1)) for i in range(n_input)]
            hidden_pos = [(0.5, 0.1 + 0.8 * i / (n_hidden-1)) for i in range(n_hidden)]
            output_pos = [(0.9, 0.5)]
            
            # Draw nodes
            for i, (x, y) in enumerate(input_pos):
                circle = plt.Circle((x, y), 0.03, fill=True, color=AWS_COLORS['light_blue'])
                ax.add_patch(circle)
                ax.text(x - 0.05, y, f"x{i+1}", ha='right', va='center')
            
            for i, (x, y) in enumerate(hidden_pos):
                circle = plt.Circle((x, y), 0.03, fill=True, color=AWS_COLORS['orange'])
                ax.add_patch(circle)
            
            for i, (x, y) in enumerate(output_pos):
                circle = plt.Circle((x, y), 0.03, fill=True, color=AWS_COLORS['teal'])
                ax.add_patch(circle)
                ax.text(x + 0.05, y, "output", ha='left', va='center')
            
            # Draw connections from input to hidden with weights
            max_weight = max(np.max(np.abs(weights['w1'])), np.max(np.abs(weights['w2'])))
            
            # Input to hidden
            for i in range(n_input):
                for j in range(n_hidden):
                    weight = weights['w1'][j, i]
                    width = 3 * np.abs(weight) / max_weight
                    color = 'red' if weight < 0 else 'blue'
                    ax.plot([input_pos[i][0], hidden_pos[j][0]], 
                            [input_pos[i][1], hidden_pos[j][1]], 
                            color=color, linewidth=width, alpha=0.7)
            
            # Hidden to output
            for i in range(n_hidden):
                weight = weights['w2'][0, i]
                width = 3 * np.abs(weight) / max_weight
                color = 'red' if weight < 0 else 'blue'
                ax.plot([hidden_pos[i][0], output_pos[0][0]], 
                        [hidden_pos[i][1], output_pos[0][1]], 
                        color=color, linewidth=width, alpha=0.7)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            st.pyplot(fig)
            
            # Display weight values numerically
            if st.checkbox("Show weight values"):
                st.markdown("### Input to Hidden Layer Weights")
                st.dataframe(pd.DataFrame(weights['w1'], 
                                         columns=[f"Input {i+1}" for i in range(n_input)],
                                         index=[f"Hidden {i+1}" for i in range(n_hidden)]))
                
                st.markdown("### Hidden to Output Layer Weights")
                st.dataframe(pd.DataFrame(weights['w2'], 
                                         columns=[f"Hidden {i+1}" for i in range(n_hidden)],
                                         index=["Output"]))

# ---- Tab 5: Advanced Concepts ----
with tabs[4]:
    st.header("🔍 Advanced Neural Network Concepts")
    
    st.markdown("""
    Beyond the basics, there are many advanced techniques and architectures 
    that make neural networks more powerful and versatile.
    """)
    
    advanced_topic = st.selectbox(
        "Choose an advanced topic to explore:",
        ["Regularization Techniques", 
         "Advanced Architectures", 
         "Transfer Learning", 
         "Generative Models"]
    )
    
    if advanced_topic == "Regularization Techniques":
        st.subheader("Regularization Techniques")
        
        st.markdown("""
        Regularization helps prevent overfitting by constraining the model's complexity.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### L1 and L2 Regularization")
            st.markdown("""
            **L1 Regularization** (Lasso):
            - Adds the absolute value of weights to the loss function
            - Tends to produce sparse models with many zero weights
            - Formula: λ * Σ|w|
            
            **L2 Regularization** (Ridge):
            - Adds the squared value of weights to the loss function
            - Prevents any weight from becoming too large
            - Formula: λ * Σw²
            """)
            
            # Visualize L1 vs L2
            fig, ax = plt.subplots()
            x = np.linspace(-2, 2, 100)
            y = np.linspace(-2, 2, 100)
            X, Y = np.meshgrid(x, y)
            
            # L1 norm contours
            l1_z = abs(X) + abs(Y)
            l1_contour = ax.contour(X, Y, l1_z, levels=[1], colors='blue', linestyles='solid', linewidths=2)
            
            # L2 norm contours
            l2_z = X**2 + Y**2
            l2_contour = ax.contour(X, Y, l2_z, levels=[1], colors='red', linestyles='solid', linewidths=2)
            
            ax.legend([l1_contour.collections[0], l2_contour.collections[0]], ['L1 Norm', 'L2 Norm'])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title('L1 vs L2 Regularization')
            ax.set_xlabel('Weight 1')
            ax.set_ylabel('Weight 2')
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Dropout")
            st.markdown("""
            **Dropout** is a technique where randomly selected neurons are ignored during training.
            
            **How it works:**
            - During each training iteration, neurons are temporarily "dropped" with probability p
            - Forces the network to learn redundant representations
            - Acts as an ensemble of multiple networks
            
            **Benefits:**
            - Reduces overfitting dramatically
            - Improves generalization
            - Computationally inexpensive
            """)
            
            # Dropout animation
            st.image("https://miro.medium.com/max/1400/1*iWQzxhVlvadk6VAJjsgXgg.png",
                    caption="Dropout Visualization", width=400)
    
    elif advanced_topic == "Advanced Architectures":
        st.subheader("Advanced Neural Network Architectures")
        
        architecture_type = st.radio(
            "Select an architecture type:",
            ["Convolutional Neural Networks (CNNs)", 
             "Recurrent Neural Networks (RNNs)",
             "Transformers"]
        )
        
        if architecture_type == "Convolutional Neural Networks (CNNs)":
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                ### Convolutional Neural Networks (CNNs)
                
                CNNs are specialized for processing grid-like data such as images. They use:
                
                - **Convolutional layers**: Apply filters to detect features
                - **Pooling layers**: Downsample to reduce dimensions
                - **Fully connected layers**: Final classification
                
                **Key advantages:**
                - Parameter sharing reduces model size
                - Spatial hierarchies of features
                - Translation invariance
                
                CNNs have revolutionized computer vision tasks like image classification, object detection, and segmentation.
                """)
            
            with col2:
                # Show CNN architecture
                st.image("https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg",
                        caption="CNN Architecture")
        
        elif architecture_type == "Recurrent Neural Networks (RNNs)":
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                ### Recurrent Neural Networks (RNNs)
                
                RNNs are designed for sequential data by maintaining a hidden state that captures information from previous steps.
                
                **Variations:**
                - **LSTM** (Long Short-Term Memory): Solves vanishing gradient problem with gates
                - **GRU** (Gated Recurrent Unit): Simpler version of LSTM
                
                **Applications:**
                - Natural language processing
                - Time series prediction
                - Speech recognition
                - Music generation
                """)
            
            with col2:
                # Show RNN unfolding
                st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png",
                        caption="RNN Unfolded Over Time")
        
        else:  # Transformers
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                ### Transformer Architecture
                
                Transformers have revolutionized NLP and are now used in vision as well. Key components:
                
                - **Self-attention mechanism**: Weighs importance of different parts of input
                - **Multi-head attention**: Multiple parallel attention mechanisms
                - **Positional encoding**: Adds position information
                
                **Famous models:**
                - BERT, GPT, T5 (for NLP)
                - Vision Transformer (ViT)
                
                Transformers have achieved breakthrough results in machine translation, text generation, and other language tasks.
                """)
            
            with col2:
                # Show transformer architecture
                st.image("https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png",
                        caption="Transformer Architecture")
    
    elif advanced_topic == "Transfer Learning":
        st.subheader("Transfer Learning")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### What is Transfer Learning?
            
            Transfer learning involves taking a model trained on one task and reusing it as the starting point for a model on a second task.
            
            **Benefits:**
            - Requires less training data for the new task
            - Faster training time
            - Often results in better performance
            
            **Common approaches:**
            1. **Feature extraction**: Use pre-trained model as fixed feature extractor
            2. **Fine-tuning**: Further train some or all layers on new data
            
            **Applications:**
            - Using ImageNet pre-trained models for specific image tasks
            - Using pre-trained word embeddings or language models for NLP tasks
            """)
            
            # Create a flowchart for transfer learning
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Draw boxes
            pretrained = plt.Rectangle((0.1, 0.5), 0.3, 0.3, fill=True, color=AWS_COLORS['light_blue'], alpha=0.7)
            new_model = plt.Rectangle((0.6, 0.5), 0.3, 0.3, fill=True, color=AWS_COLORS['orange'], alpha=0.7)
            
            ax.add_patch(pretrained)
            ax.add_patch(new_model)
            
            # Draw arrow
            ax.arrow(0.4, 0.65, 0.19, 0, head_width=0.03, head_length=0.03, fc=AWS_COLORS['teal'], ec=AWS_COLORS['teal'], lw=2)
            
            # Labels
            ax.text(0.25, 0.65, "Pre-trained Model", ha='center', va='center')
            ax.text(0.25, 0.58, "(Source Task)", ha='center', va='center', fontsize=10)
            ax.text(0.75, 0.65, "New Model", ha='center', va='center')
            ax.text(0.75, 0.58, "(Target Task)", ha='center', va='center', fontsize=10)
            ax.text(0.5, 0.7, "Transfer Knowledge", ha='center', va='center', fontsize=10)
            
            # Settings
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### Transfer Learning Process
            
            1. Select a pre-trained model (e.g., ResNet, BERT)
            2. Decide whether to use as feature extractor or fine-tune
            3. Replace and retrain the output layer for your specific task
            4. Optionally fine-tune deeper layers
            
            **When to use Transfer Learning:**
            - When you have limited data
            - When the source and target tasks are related
            - To save training time and computational resources
            """)
            
            # Show example of transfer learning in action
            st.image("https://miro.medium.com/max/1400/1*9GTEzcO8KxxrfutmtsPs3Q.png",
                    caption="Transfer Learning for Image Classification")
    
    else:  # Generative Models
        st.subheader("Generative Neural Networks")
        
        generative_model = st.radio(
            "Choose a generative model type:",
            ["Generative Adversarial Networks (GANs)", 
             "Variational Autoencoders (VAEs)",
             "Diffusion Models"]
        )
        
        if generative_model == "Generative Adversarial Networks (GANs)":
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                ### Generative Adversarial Networks (GANs)
                
                GANs consist of two neural networks that compete against each other:
                
                - **Generator**: Creates synthetic samples
                - **Discriminator**: Distinguishes real from generated samples
                
                **Training process:**
                1. Generator creates fake samples
                2. Discriminator tries to identify real vs. fake
                3. Generator improves based on feedback
                4. Process repeats until generator creates convincing fakes
                
                **Applications:**
                - Image synthesis and editing
                - Style transfer
                - Data augmentation
                - Creating realistic simulations
                """)
            
            with col2:
                # Show GAN architecture
                st.image("https://developers.google.com/static/machine-learning/gan/images/gan_diagram.svg",
                        caption="GAN Architecture")
                
                # Show GAN progression
                st.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2017/06/11000153/g10.png",
                        caption="GAN Training Progression")
        
        elif generative_model == "Variational Autoencoders (VAEs)":
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                ### Variational Autoencoders (VAEs)
                
                VAEs are generative models that learn to encode data into a latent space and then decode it back.
                
                **Architecture:**
                - **Encoder**: Compresses input into a latent representation
                - **Latent Space**: Probabilistic distribution of features
                - **Decoder**: Reconstructs the input from latent representation
                
                **Key features:**
                - Regularized latent space for smooth interpolation
                - Probabilistic encoder outputs mean and variance
                - Can generate new samples by sampling from latent space
                
                **Applications:**
                - Image generation
                - Anomaly detection
                - Drug discovery
                - Data compression
                """)
            
            with col2:
                # Show VAE architecture
                st.image("https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png",
                        caption="VAE Architecture")
        
        else:  # Diffusion Models
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                ### Diffusion Models
                
                Diffusion models work by gradually adding noise to data and then learning to reverse this process.
                
                **Process:**
                1. **Forward diffusion**: Gradually add noise to data
                2. **Reverse diffusion**: Learn to denoise step by step
                
                **Key advantages:**
                - High quality outputs
                - More stable training than GANs
                - Good mode coverage (diversity)
                
                **Applications:**
                - Image generation (DALL-E, Imagen, Stable Diffusion)
                - Audio synthesis
                - Video generation
                - 3D structure generation
                """)
            
            with col2:
                # Show diffusion process
                st.image("https://miro.medium.com/max/1400/1*50kQyKrKxxAhPFIwJzIKAQ.png",
                        caption="Diffusion Process")

# ---- Footer ----
st.markdown("""
<footer>
© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</footer>
""", unsafe_allow_html=True)