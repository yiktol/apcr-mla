import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import make_classification, make_regression, load_iris
from PIL import Image, ImageOps, ImageEnhance
import altair as alt
from io import BytesIO
import base64
import time
import string
import nltk
from nltk.corpus import wordnet
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import utils.common as common
import utils.authenticate as authenticate


def initialize_session_state():
    """Initialize the session state variables if they don't exist."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.sample_data = None
        st.session_state.train_data = None
        st.session_state.valid_data = None
        st.session_state.test_data = None
        st.session_state.kfold_data = None
        st.session_state.shuffled_data = None
        st.session_state.augmented_images = None
        st.session_state.augmented_text = None
        st.session_state.augmented_timeseries = None
        # Initialize quiz state
        st.session_state.quiz_score = 0
        st.session_state.quiz_submitted = False
        st.session_state.quiz_answers = {}


def setup_page_config():
    """Configure the page settings."""
    st.set_page_config(
        page_title="ML Data Preparation",
        page_icon="📊",
        layout="wide"
    )
    
setup_page_config()

def apply_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #4361EE;
            text-align: left;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #3A0CA3;
            margin-bottom: 1rem;
        }
        .section {
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #F0F8FF;
            padding: 1rem;
            border-left: 5px solid #4361EE;
            margin-bottom: 1rem;
            border-radius: 5px;
        }
        .example-box {
            background-color: #F0FFF0;
            padding: 1rem;
            border-left: 5px solid #4CC9F0;
            margin-bottom: 1rem;
            border-radius: 5px;
        }
        .warning-box {
            background-color: #FFF0F0;
            padding: 1rem;
            border-left: 5px solid #F72585;
            margin-bottom: 1rem;
            border-radius: 5px;
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
        .stButton>button {
            background-color: #FF9900;
            color: white;
        }
        .stButton>button:hover {
            background-color: #FFAC31;
        }
    </style>
    """, unsafe_allow_html=True)


def setup_sidebar():
    """Set up the sidebar with session management."""
    with st.sidebar:
        common.render_sidebar()

    with st.sidebar.expander("About This App", expanded=False):
        st.info("""
        This application helps you understand:
        - Data Splitting techniques
        - Data Shuffling methods
        - Data Augmentation strategies

        Perfect for data scientists preparing data for machine learning models.
        """)


def render_main_header():
    """Render the main page header."""
    st.markdown('<h1 class="main-header">Data Preparation for Machine Learning</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Preparing your data correctly is one of the most crucial steps in building effective machine learning models. 
    This interactive guide will help you understand key techniques for data splitting, shuffling, and augmentation.
    </div>
    """, unsafe_allow_html=True)


def render_data_splitting_tab():
    """Render the content of the data splitting tab."""
    st.markdown('<h2 class="sub-header">📊 Data Splitting</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Why Split Your Data?</h3>
    <p>When building machine learning models, we split our data to:</p>
    <ul>
        <li><strong>Train the model</strong> with a substantial portion of the data</li>
        <li><strong>Validate model performance</strong> during development</li>
        <li><strong>Test the final model</strong> on unseen data</li>
    </ul>
    <p>This approach helps ensure your model generalizes well to new, unseen data in production.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for the train/validation/test explanation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color:#E9ECEF; padding:10px; border-radius:5px; height:200px;">
        <h3 style="color:#4361EE; text-align:center;">Training Data</h3>
        <ul>
            <li>Typically 60-80% of data</li>
            <li>Used to train the model</li>
            <li>Model learns patterns from this data</li>
            <li>Performance on this data can be optimistic</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color:#E9ECEF; padding:10px; border-radius:5px; height:200px;">
        <h3 style="color:#4CC9F0; text-align:center;">Validation Data</h3>
        <ul>
            <li>Typically 10-20% of data</li>
            <li>Used during model development</li>
            <li>For hyperparameter tuning</li>
            <li>Prevents overfitting to training data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color:#E9ECEF; padding:10px; border-radius:5px; height:200px;">
        <h3 style="color:#F72585; text-align:center;">Test Data</h3>
        <ul>
            <li>Typically 10-20% of data</li>
            <li>Used only once at the end</li>
            <li>Final evaluation of model</li>
            <li>Simulates real-world performance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data splitting techniques subtabs
    split_tabs = st.tabs(["🔪 Simple Hold-out", "🔄 Cross-Validation", "🧪 Interactive Example"])
    
    with split_tabs[0]:
        render_holdout_method()
    
    with split_tabs[1]:
        render_cross_validation()
    
    with split_tabs[2]:
        render_interactive_split_example()


def render_holdout_method():
    """Render the simple hold-out method section."""
    st.markdown("""
    <h3>Simple Hold-out Method</h3>
    <div class="info-box">
    The hold-out method is the simplest approach to splitting data. It involves randomly dividing the dataset into:
    <ul>
        <li><strong>Training set</strong> - Used to train the model (e.g., 60-80%)</li>
        <li><strong>Validation set</strong> - Used to tune hyperparameters (e.g., 10-20%)</li>
        <li><strong>Test set</strong> - Used for final evaluation (e.g., 10-20%)</li>
    </ul>
    </div>
    
    <div class="example-box">
    <h4>Advantages</h4>
    <ul>
        <li>Simple and quick to implement</li>
        <li>Less computationally expensive</li>
        <li>Works well with large datasets</li>
    </ul>
    </div>
    
    <div class="warning-box">
    <h4>Limitations</h4>
    <ul>
        <li>Performance can vary depending on which samples end up in each split</li>
        <li>Inefficient use of data, especially with small datasets</li>
        <li>May lead to higher variance in model performance estimates</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual representation of simple hold-out
    st.markdown("<h4>Visual Representation</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh("Dataset", 1, color="lightgray", height=0.5)
    ax.barh("Dataset", 0.7, color="#4361EE", height=0.5, label="Training (70%)")
    ax.barh("Dataset", 0.15, left=0.7, color="#4CC9F0", height=0.5, label="Validation (15%)")
    ax.barh("Dataset", 0.15, left=0.85, color="#F72585", height=0.5, label="Test (15%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel("Data Distribution")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    st.pyplot(fig)


def render_cross_validation():
    """Render the cross-validation section."""
    st.markdown("""
    <h3>Cross-Validation</h3>
    <div class="info-box">
    Cross-validation is a more robust technique that makes better use of your data. The most common form is k-fold cross-validation:
    <ol>
        <li>Divide the dataset into k equal parts (folds)</li>
        <li>Train the model k times, each time using a different fold as the validation set</li>
        <li>Average the performance across all k iterations</li>
    </ol>
    This ensures that every data point is used for both training and validation.
    </div>
    
    <div class="example-box">
    <h4>Advantages</h4>
    <ul>
        <li>Makes better use of limited data</li>
        <li>Provides more reliable performance estimates</li>
        <li>Reduces variance in performance evaluation</li>
        <li>Every observation is used for both training and validation</li>
    </ul>
    </div>
    
    <div class="warning-box">
    <h4>Limitations</h4>
    <ul>
        <li>More computationally expensive (trains k models)</li>
        <li>Takes longer to execute</li>
        <li>More complex to implement</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual representation of K-fold cross-validation
    st.markdown("<h4>K-Fold Cross-Validation (k=5)</h4>", unsafe_allow_html=True)
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        # Draw the full dataset bar
        ax.barh(f"Fold {i+1}", 1, color="lightgray", height=0.5)
        
        # Highlight the validation fold
        for j in range(5):
            if j == i:
                # Validation fold
                ax.barh(f"Fold {i+1}", 0.2, left=j*0.2, color="#F72585", height=0.5)
            else:
                # Training folds
                ax.barh(f"Fold {i+1}", 0.2, left=j*0.2, color="#4361EE", height=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add "Training" and "Validation" text for the first fold
        if i == 0:
            ax.text(0.4, 0.8, "Training", color="white", fontweight="bold", ha="center")
            ax.text(0.1, 0.8, "Validation", color="white", fontweight="bold", ha="center")
    
    axes[-1].set_xlabel("Data Distribution")
    fig.tight_layout()
    
    # Add a legend
    blue_patch = plt.Rectangle((0, 0), 1, 1, color="#4361EE")
    pink_patch = plt.Rectangle((0, 0), 1, 1, color="#F72585")
    fig.legend([blue_patch, pink_patch], ["Training", "Validation"], 
              loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2)
    
    st.pyplot(fig)
    
    st.markdown("""
    <div class="info-box">
    <h4>Common Variations</h4>
    <ul>
        <li><strong>Stratified K-Fold</strong>: Preserves the class distribution in each fold (important for imbalanced datasets)</li>
        <li><strong>Leave-One-Out</strong>: Extreme case where k equals the number of samples (used for very small datasets)</li>
        <li><strong>Nested Cross-Validation</strong>: Used when tuning hyperparameters, with an inner and outer cross-validation loop</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def render_interactive_split_example():
    """Render the interactive data splitting example."""
    st.markdown("<h3>Interactive Data Splitting Example</h3>", unsafe_allow_html=True)
    
    # Generate sample data if not already generated
    if st.session_state.sample_data is None:
        # Generate a synthetic dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        
        feature_names = ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"]
        df = pd.DataFrame(X, columns=feature_names)
        df["Target"] = y
        st.session_state.sample_data = df
    
    # Display a sample of the dataset
    st.markdown("<h4>Sample Dataset</h4>", unsafe_allow_html=True)
    st.dataframe(st.session_state.sample_data.head(5))
    st.markdown(f"Dataset Shape: {st.session_state.sample_data.shape[0]} rows, {st.session_state.sample_data.shape[1]} columns")
    
    # Add controls for splitting parameters
    st.markdown("<h4>Configure Your Data Split</h4>", unsafe_allow_html=True)
    
    split_method = st.radio("Select Split Method", ["Hold-out Method", "K-Fold Cross-Validation"])
    
    if split_method == "Hold-out Method":
        handle_holdout_split()
    else:  # K-Fold Cross-Validation
        handle_kfold_split()


def handle_holdout_split():
    """Handle the hold-out splitting method in the interactive example."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_size = st.slider("Training Set %", 50, 90, 70, 5)
    
    with col2:
        valid_size_max = 100 - train_size
        valid_pct = st.slider("Validation Set %", 0, valid_size_max, min(20, valid_size_max), 5)
    
    with col3:
        test_size = 100 - train_size - valid_pct
        st.metric("Test Set %", test_size)
    
    shuffle_data = st.checkbox("Shuffle data before splitting", value=True)
    random_seed = st.number_input("Random seed (for reproducibility)", value=42, min_value=0, max_value=100)
    
    if st.button("Split Data", key="split_holdout"):
        # Calculate actual proportions
        test_prop = test_size / 100
        valid_prop = valid_pct / 100
        train_prop = train_size / 100
        
        # First split into train and temp (valid+test)
        temp_size = valid_prop + test_prop
        train_idx, temp_idx = train_test_split(
            range(len(st.session_state.sample_data)),
            test_size=temp_size,
            random_state=random_seed,
            shuffle=shuffle_data
        )
        
        # Then split temp into valid and test
        if temp_size > 0:
            ratio_valid_to_test = valid_prop / temp_size if temp_size > 0 else 0
            valid_idx, test_idx = train_test_split(
                temp_idx,
                test_size=1-ratio_valid_to_test,
                random_state=random_seed,
                shuffle=shuffle_data
            )
        else:
            valid_idx = []
            test_idx = []
        
        # Create the subsets
        train_data = st.session_state.sample_data.iloc[train_idx].copy()
        valid_data = st.session_state.sample_data.iloc[valid_idx].copy() if len(valid_idx) > 0 else pd.DataFrame()
        test_data = st.session_state.sample_data.iloc[test_idx].copy() if len(test_idx) > 0 else pd.DataFrame()
        
        # Store in session state
        st.session_state.train_data = train_data
        st.session_state.valid_data = valid_data
        st.session_state.test_data = test_data
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"<h5>Training Set: {len(train_data)} samples</h5>", unsafe_allow_html=True)
            st.dataframe(train_data.head(3))
        
        with col2:
            st.markdown(f"<h5>Validation Set: {len(valid_data)} samples</h5>", unsafe_allow_html=True)
            if not valid_data.empty:
                st.dataframe(valid_data.head(3))
            else:
                st.info("No validation set (0%)")
        
        with col3:
            st.markdown(f"<h5>Test Set: {len(test_data)} samples</h5>", unsafe_allow_html=True)
            if not test_data.empty:
                st.dataframe(test_data.head(3))
            else:
                st.info("No test set (0%)")
        
        # Visualize the distribution of target variable in each split
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        if not train_data.empty:
            train_data["Target"].value_counts(normalize=True).plot(kind="bar", ax=ax[0])
            ax[0].set_title("Training Set Target Distribution")
            ax[0].set_ylabel("Proportion")
        
        if not valid_data.empty:
            valid_data["Target"].value_counts(normalize=True).plot(kind="bar", ax=ax[1])
            ax[1].set_title("Validation Set Target Distribution")
            ax[1].set_ylabel("")
        else:
            ax[1].set_title("No Validation Set")
            
        if not test_data.empty:
            test_data["Target"].value_counts(normalize=True).plot(kind="bar", ax=ax[2])
            ax[2].set_title("Test Set Target Distribution")
            ax[2].set_ylabel("")
        else:
            ax[2].set_title("No Test Set")
        
        fig.tight_layout()
        st.pyplot(fig)


def handle_kfold_split():
    """Handle the K-fold cross-validation splitting method in the interactive example."""
    n_folds = st.slider("Number of folds (k)", 2, 10, 5)
    shuffle_data = st.checkbox("Shuffle data before splitting", value=True)
    random_seed = st.number_input("Random seed (for reproducibility)", value=42, min_value=0, max_value=100)
    
    if st.button("Perform K-Fold Split", key="split_kfold"):
        # Initialize K-Fold
        kf = KFold(n_splits=n_folds, shuffle=shuffle_data, random_state=random_seed)
        
        # Store the fold assignments
        fold_assignments = np.zeros(len(st.session_state.sample_data), dtype=int)
        for i, (_, val_idx) in enumerate(kf.split(st.session_state.sample_data)):
            fold_assignments[val_idx] = i
        
        # Add fold column to the data
        kfold_data = st.session_state.sample_data.copy()
        kfold_data["Fold"] = fold_assignments
        
        # Store in session state
        st.session_state.kfold_data = kfold_data
        
        # Display results
        st.markdown(f"<h5>K-Fold Cross-Validation Results ({n_folds} folds)</h5>", unsafe_allow_html=True)
        st.dataframe(kfold_data.head())
        
        # Count samples per fold
        fold_counts = kfold_data["Fold"].value_counts().sort_index()
        
        # Plot the fold distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        fold_counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Number of Samples in Each Fold (Total: {len(kfold_data)})")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Number of Samples")
        st.pyplot(fig)
        
        # Show target distribution by fold
        st.markdown("<h5>Target Distribution by Fold</h5>", unsafe_allow_html=True)
        
        # Calculate target distribution for each fold
        target_dist = []
        for fold in range(n_folds):
            fold_data = kfold_data[kfold_data["Fold"] == fold]
            target_counts = fold_data["Target"].value_counts(normalize=True)
            target_dist.append(target_counts)
        
        target_dist_df = pd.DataFrame(target_dist).fillna(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        target_dist_df.plot(kind="bar", ax=ax)
        ax.set_title("Target Class Distribution Across Folds")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Proportion")
        ax.legend(title="Target Class")
        st.pyplot(fig)
        
        # Simulate cross-validation
        st.markdown("<h5>Simulated Cross-Validation Iterations</h5>", unsafe_allow_html=True)
        
        for fold in range(n_folds):
            train_mask = kfold_data["Fold"] != fold
            val_mask = kfold_data["Fold"] == fold
            
            train_data = kfold_data[train_mask]
            val_data = kfold_data[val_mask]
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Visualization for this fold
                fig, ax = plt.subplots(figsize=(8, 1))
                ax.barh("Dataset", 1, color="lightgray", height=0.5)
                
                # Plot each fold segment
                for i in range(n_folds):
                    if i == fold:
                        # Validation fold
                        ax.barh("Dataset", 1/n_folds, left=i/n_folds, color="#F72585", height=0.5)
                    else:
                        # Training folds
                        ax.barh("Dataset", 1/n_folds, left=i/n_folds, color="#4361EE", height=0.5)
                
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                
                fig.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown(f"""
                <div style="border:1px solid #ddd; padding:8px; border-radius:5px;">
                <strong>Fold {fold+1}</strong><br>
                Training: {len(train_data)} samples<br>
                Validation: {len(val_data)} samples
                </div>
                """, unsafe_allow_html=True)


def render_data_shuffling_tab():
    """Render the content of the data shuffling tab."""
    st.markdown('<h2 class="sub-header">🔀 Data Shuffling</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Why Shuffle Your Data?</h3>
    <p>Data shuffling is the process of randomizing the order of samples in your dataset. It is a crucial step in machine learning for several reasons:</p>
    <ul>
        <li><strong>Prevents order bias</strong> - Ensures the model doesn't learn patterns that depend on the order of the data</li>
        <li><strong>Improves convergence</strong> - Many optimization algorithms converge faster with shuffled data</li>
        <li><strong>Reduces variance</strong> - Different shuffles can produce different models; evaluating across shuffles provides more robust results</li>
        <li><strong>Better generalization</strong> - Models trained on shuffled data tend to generalize better to unseen data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Shuffling techniques subtabs
    shuffle_tabs = st.tabs(["🎲 Random Permutation", "🔄 Epoch-based Shuffling", "📦 Mini-batch Shuffling", "🧪 Interactive Example"])
    
    with shuffle_tabs[0]:
        render_random_permutation()
    
    with shuffle_tabs[1]:
        render_epoch_shuffling()
    
    with shuffle_tabs[2]:
        render_minibatch_shuffling()
    
    with shuffle_tabs[3]:
        render_interactive_shuffle_example()


def render_random_permutation():
    """Render the random permutation section."""
    st.markdown("""
    <h3>Random Permutation</h3>
    <div class="info-box">
    <p>Random permutation involves randomly reordering <strong>all data points</strong> in your dataset. This is the most common and straightforward shuffling technique.</p>
    
    <h4>How It Works:</h4>
    <ol>
        <li>Generate a random permutation of indices from 0 to n-1 (where n is the dataset size)</li>
        <li>Reorder the entire dataset according to these indices</li>
    </ol>
    
    <h4>Implementation in Python:</h4>
    <pre>
    import numpy as np
    
    # Generate random permutation
    indices = np.random.permutation(len(dataset))
    
    # Shuffle dataset
    shuffled_dataset = dataset[indices]
    </pre>
    </div>
    
    <div class="example-box">
    <h4>Advantages</h4>
    <ul>
        <li>Simple to implement</li>
        <li>Completely randomizes the dataset</li>
        <li>Effectively removes any ordering biases</li>
    </ul>
    </div>
    
    <div class="warning-box">
    <h4>When to Use</h4>
    <p>Random permutation is ideal:</p>
    <ul>
        <li>Before splitting data into train/validation/test sets</li>
        <li>For batch gradient descent (when using the entire dataset for each update)</li>
        <li>When the original order of data may contain patterns you don't want the model to learn</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual representation of random permutation
    st.markdown("<h4>Visual Representation</h4>", unsafe_allow_html=True)
    
    # Create a simple visualization of random permutation
    n_samples = 20
    original_order = np.arange(n_samples)
    np.random.seed(42)
    shuffled_order = np.random.permutation(original_order)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
    
    # Original data visualization with gradient color
    cmap = plt.cm.viridis
    colors = [cmap(i/n_samples) for i in range(n_samples)]
    
    for i, color in zip(original_order, colors):
        ax1.barh(0, 1, left=i, height=0.5, color=color)
    ax1.set_title("Original Order")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(-0.5, n_samples-0.5)
    
    # Shuffled data visualization with same colors but shuffled
    for i, (orig_i, color) in enumerate(zip(shuffled_order, colors)):
        ax2.barh(0, 1, left=i, height=0.5, color=color)
    ax2.set_title("After Random Permutation")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(-0.5, n_samples-0.5)
    
    plt.tight_layout()
    st.pyplot(fig)


def render_epoch_shuffling():
    """Render the epoch-based shuffling section."""
    st.markdown("""
    <h3>Epoch-based Shuffling</h3>
    <div class="info-box">
    <p>In epoch-based shuffling, the dataset is reshuffled at the beginning of each training epoch, but the order remains fixed during the epoch.</p>
    
    <h4>How It Works:</h4>
    <ol>
        <li>At the start of each epoch, generate a new random permutation of the dataset</li>
        <li>Use this order throughout the epoch for all batches</li>
        <li>Repeat for each new epoch</li>
    </ol>
    
    <h4>Implementation in Python:</h4>
    <pre>
    import numpy as np
    
    for epoch in range(num_epochs):
        # Generate new shuffling for this epoch
        indices = np.random.permutation(len(dataset))
        shuffled_dataset = dataset[indices]
        
        # Train on shuffled dataset for this epoch
        train_on_dataset(shuffled_dataset)
    </pre>
    </div>
    
    <div class="example-box">
    <h4>Advantages</h4>
    <ul>
        <li>Balances randomness with computational efficiency</li>
        <li>Ensures the model sees data in a different order in each epoch</li>
        <li>Good for mini-batch gradient descent</li>
        <li>More memory-efficient than re-shuffling for every batch</li>
    </ul>
    </div>
    
    <div class="warning-box">
    <h4>When to Use</h4>
    <p>Epoch-based shuffling is ideal for:</p>
    <ul>
        <li>Training deep learning models with multiple epochs</li>
        <li>When using mini-batch optimization</li>
        <li>When memory or computational constraints make per-batch shuffling impractical</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual representation of epoch-based shuffling
    st.markdown("<h4>Visual Representation</h4>", unsafe_allow_html=True)
    
    n_samples = 20
    n_epochs = 3
    
    fig, axes = plt.subplots(n_epochs + 1, 1, figsize=(10, 6))
    
    # Original data
    cmap = plt.cm.viridis
    colors = [cmap(i/n_samples) for i in range(n_samples)]
    
    axes[0].barh(np.zeros(n_samples), np.ones(n_samples), left=np.arange(n_samples), height=0.5, color=colors)
    axes[0].set_title("Original Data")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Different shuffling for each epoch
    np.random.seed(42)
    for i in range(n_epochs):
        shuffled_order = np.random.permutation(n_samples)
        shuffled_colors = [colors[j] for j in shuffled_order]
        
        axes[i+1].barh(np.zeros(n_samples), np.ones(n_samples), left=np.arange(n_samples), height=0.5, color=shuffled_colors)
        axes[i+1].set_title(f"Epoch {i+1} Shuffling")
        axes[i+1].set_xticks([])
        axes[i+1].set_yticks([])
    
    plt.tight_layout()
    st.pyplot(fig)


def render_minibatch_shuffling():
    """Render the mini-batch shuffling section."""
    st.markdown("""
    <h3>Mini-batch Shuffling</h3>
    <div class="info-box">
    <p>In mini-batch shuffling, the dataset is re-shuffled before creating each mini-batch, providing maximum randomness during training.</p>
    
    <h4>How It Works:</h4>
    <ol>
        <li>Shuffle the entire dataset</li>
        <li>Divide it into mini-batches</li>
        <li>Process each mini-batch</li>
        <li>Reshuffle and create new mini-batches for the next iteration</li>
    </ol>
    
    <h4>Implementation in Python:</h4>
    <pre>
    import numpy as np
    
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            # Generate new shuffling for each batch
            indices = np.random.permutation(len(dataset))
            batch_indices = indices[:batch_size]  # Take first batch_size elements
            mini_batch = dataset[batch_indices]
            
            # Train on mini-batch
            train_on_mini_batch(mini_batch)
    </pre>
    </div>
    
    <div class="example-box">
    <h4>Advantages</h4>
    <ul>
        <li>Maximum randomness in training</li>
        <li>Can help escape local minima</li>
        <li>Potentially better model generalization</li>
        <li>Each batch is truly independent</li>
    </ul>
    </div>
    
    <div class="warning-box">
    <h4>Limitations</h4>
    <ul>
        <li>Computationally expensive</li>
        <li>May not cover the entire dataset evenly</li>
        <li>Risk of some samples being seen multiple times while others are not seen at all</li>
        <li>Generally not recommended for most applications</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual representation of mini-batch shuffling
    st.markdown("<h4>Visual Representation</h4>", unsafe_allow_html=True)
    
    n_samples = 20
    batch_size = 4
    n_batches = 3
    
    fig, axes = plt.subplots(n_batches + 1, 1, figsize=(10, 6))
    
    # Original data
    cmap = plt.cm.viridis
    colors = [cmap(i/n_samples) for i in range(n_samples)]
    
    axes[0].barh(np.zeros(n_samples), np.ones(n_samples), left=np.arange(n_samples), height=0.5, color=colors)
    axes[0].set_title("Full Dataset")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Different mini-batches
    np.random.seed(42)
    for i in range(n_batches):
        shuffled_order = np.random.permutation(n_samples)[:batch_size]
        shuffled_colors = [colors[j] for j in shuffled_order]
        
        # Create a bar for each element in the mini-batch
        axes[i+1].barh(np.zeros(batch_size), np.ones(batch_size), left=np.arange(batch_size), height=0.5, color=shuffled_colors)
        axes[i+1].set_title(f"Mini-batch {i+1}")
        axes[i+1].set_xticks([])
        axes[i+1].set_yticks([])
        axes[i+1].set_xlim(0, batch_size)
    
    plt.tight_layout()
    st.pyplot(fig)


def render_interactive_shuffle_example():
    """Render the interactive data shuffling example."""
    st.markdown("<h3>Interactive Data Shuffling Example</h3>", unsafe_allow_html=True)
    
    # Generate sample time series data if not already generated
    if st.session_state.shuffled_data is None:
        # Create a simple time series dataset with a trend
        np.random.seed(42)
        x = np.arange(100)
        y = 0.05 * x + np.sin(x/5) + np.random.normal(0, 0.2, size=100)
        
        shuffled_data = pd.DataFrame({
            'x': x,
            'y': y,
            'group': np.repeat(['A', 'B', 'C', 'D', 'E'], 20)  # Creating groups for visualization
        })
        
        st.session_state.shuffled_data = shuffled_data
    
    # Display the original data
    st.markdown("<h4>Original Data (with inherent pattern)</h4>", unsafe_allow_html=True)
    
    fig = px.scatter(st.session_state.shuffled_data, x='x', y='y', color='group',
                    title="Original Data with Time Trend")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display a sample of the original data
    with st.expander("View data sample"):
        st.dataframe(st.session_state.shuffled_data.head(10))
    
    # Add controls for shuffling
    st.markdown("<h4>Choose Shuffling Method</h4>", unsafe_allow_html=True)
    
    shuffle_method = st.radio(
        "Select Shuffling Method",
        ["Random Permutation", "Batch-based Shuffling"],
        key="shuffle_method"
    )
    
    if shuffle_method == "Random Permutation":
        handle_random_permutation_shuffling()
    else:  # Batch-based Shuffling
        handle_batch_shuffling()


def handle_random_permutation_shuffling():
    """Handle the random permutation shuffling method in the interactive example."""
    seed = st.number_input("Random Seed", min_value=0, max_value=100, value=42)
    
    if st.button("Shuffle Data", key="shuffle_random"):
        # Shuffle the data
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(st.session_state.shuffled_data))
        shuffled_df = st.session_state.shuffled_data.iloc[shuffled_indices].reset_index(drop=True)
        
        # Plot the shuffled data
        fig = px.scatter(shuffled_df, x=shuffled_df.index, y='y', color='group',
                        title="Data after Random Permutation")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the impact on trend
        original_corr = st.session_state.shuffled_data['x'].corr(st.session_state.shuffled_data['y'])
        shuffled_corr = shuffled_df.index.to_series().corr(shuffled_df['y'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original X-Y Correlation", f"{original_corr:.3f}")
        with col2:
            st.metric("Shuffled Index-Y Correlation", f"{shuffled_corr:.3f}", 
                      delta=f"{shuffled_corr-original_corr:.3f}")
        
        if abs(shuffled_corr) < abs(original_corr)/2:
            st.success("✅ Shuffling successfully reduced the temporal pattern correlation!")
        else:
            st.warning("⚠️ The correlation is still significant. Try a different seed or method.")
        
        # Show shuffled data
        with st.expander("View shuffled data"):
            st.dataframe(shuffled_df.head(10))


def handle_batch_shuffling():
    """Handle the batch-based shuffling method in the interactive example."""
    batch_size = st.slider("Batch Size", 5, 50, 20)
    n_batches = st.slider("Number of Batches to Display", 1, 5, 3)
    
    if st.button("Create Shuffled Batches", key="shuffle_batch"):
        # Create different shuffled batches
        fig, axs = plt.subplots(n_batches, 1, figsize=(10, 3*n_batches))
        
        if n_batches == 1:
            axs = [axs]
        
        for i in range(n_batches):
            # Shuffle indices
            np.random.seed(i)  # Different seed for each batch
            shuffled_indices = np.random.permutation(len(st.session_state.shuffled_data))[:batch_size]
            batch_df = st.session_state.shuffled_data.iloc[shuffled_indices]
            
            # Plot
            scatter = axs[i].scatter(range(batch_size), batch_df['y'], 
                                 c=[{'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}[g] for g in batch_df['group']], 
                                 cmap='viridis')
            axs[i].set_title(f"Mini-batch {i+1} (size={batch_size})")
            axs[i].set_xlabel("Batch Index")
            axs[i].set_ylabel("y Value")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display impact on learning
        st.markdown("<h4>Impact on Model Learning</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <p>When using mini-batch training with shuffled data:</p>
        <ul>
            <li>The model sees different patterns in each batch</li>
            <li>Temporal or sequential patterns are disrupted</li>
            <li>Gradient updates become more stochastic, potentially helping escape local minima</li>
            <li>Learning is usually more stable and generalizes better</li>
        </ul>
        </div>
        
        <div class="warning-box">
        <p><strong>Note:</strong> For data where sequence matters (like time series), you typically would NOT want to shuffle. Instead, you'd use techniques like sliding windows or recurrent neural networks that can capture temporal dependencies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show batch statistics
        batch_stats = []
        for i in range(n_batches):
            np.random.seed(i)
            shuffled_indices = np.random.permutation(len(st.session_state.shuffled_data))[:batch_size]
            batch_df = st.session_state.shuffled_data.iloc[shuffled_indices]
            
            batch_stats.append({
                "Batch": i+1,
                "Mean y": batch_df['y'].mean(),
                "StdDev y": batch_df['y'].std(),
                "Group A%": (batch_df['group'] == 'A').mean() * 100,
                "Group B%": (batch_df['group'] == 'B').mean() * 100,
                "Group C%": (batch_df['group'] == 'C').mean() * 100,
                "Group D%": (batch_df['group'] == 'D').mean() * 100,
                "Group E%": (batch_df['group'] == 'E').mean() * 100,
            })
        
        st.dataframe(pd.DataFrame(batch_stats).set_index("Batch").round(2))


def render_data_augmentation_tab():
    """Render the content of the data augmentation tab."""
    st.markdown('<h2 class="sub-header">🔄 Data Augmentation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>What is Data Augmentation?</h3>
    <p>Data augmentation is a technique used to artificially increase the size and diversity of your training dataset by creating modified versions of existing samples. It helps models:</p>
    <ul>
        <li><strong>Generalize better</strong> - Learn more robust features that are invariant to certain transformations</li>
        <li><strong>Reduce overfitting</strong> - Especially when training with limited data</li>
        <li><strong>Handle class imbalance</strong> - By generating more samples for underrepresented classes</li>
        <li><strong>Improve model robustness</strong> - To variations in real-world data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Augmentation techniques subtabs
    aug_tabs = st.tabs(["🖼️ Image Augmentation", "📝 Text Augmentation", "📈 Time Series Augmentation"])
    
    with aug_tabs[0]:
        render_image_augmentation()
    
    with aug_tabs[1]:
        render_text_augmentation()
    
    with aug_tabs[2]:
        render_timeseries_augmentation()


def render_image_augmentation():
    """Render the image augmentation section."""
    st.markdown("""
    <h3>Image Augmentation</h3>
    <div class="info-box">
    <p>Image augmentation creates new training examples by applying various transformations to existing images. These transformations preserve the semantic content while altering the visual appearance.</p>
    
    <h4>Common Techniques:</h4>
    <ul>
        <li><strong>Geometric Transformations</strong>: Flipping, rotation, scaling, cropping, translation</li>
        <li><strong>Color Space Transformations</strong>: Brightness, contrast, saturation, hue adjustments</li>
        <li><strong>Noise Injection</strong>: Adding Gaussian noise, salt-and-pepper noise</li>
        <li><strong>Advanced Methods</strong>: CutMix, MixUp, Random erasing, style transfer</li>
    </ul>
    </div>
    
    <div class="example-box">
    <h4>Benefits in Computer Vision</h4>
    <ul>
        <li><strong>Viewpoint Invariance</strong>: Helps models recognize objects from different angles</li>
        <li><strong>Lighting Robustness</strong>: Improves performance under various lighting conditions</li>
        <li><strong>Scale Invariance</strong>: Enables detection of objects at different sizes</li>
        <li><strong>Occlusion Handling</strong>: Teaches models to recognize partially hidden objects</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive image augmentation demo
    st.markdown("<h4>Interactive Image Augmentation Demo</h4>", unsafe_allow_html=True)
    demo_image = Image.open("images/animals.png")
    
    # Allow user to upload an image
    uploaded_image = st.file_uploader("Upload your own image for augmentation", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Uploaded Image", use_container_width=True)
    else:
        image = demo_image
        st.image(image, caption="Demo Image", use_container_width=True)
        
    # Create augmentation options
    st.markdown("<h5>Choose Augmentation Techniques</h5>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        flip_h = st.checkbox("Horizontal Flip", value=False)
        flip_v = st.checkbox("Vertical Flip", value=False)
        rotate = st.checkbox("Rotate", value=False)
        if rotate:
            rotate_angle = st.slider("Rotation Angle", -180, 180, 0)
    
    with col2:
        brightness = st.checkbox("Adjust Brightness", value=False)
        if brightness:
            brightness_factor = st.slider("Brightness Factor", 0.5, 1.5, 1.0, 0.1)
        
        contrast = st.checkbox("Adjust Contrast", value=False)
        if contrast:
            contrast_factor = st.slider("Contrast Factor", 0.5, 1.5, 1.0, 0.1)
    
    with col3:
        add_noise = st.checkbox("Add Noise", value=False)
        if add_noise:
            noise_intensity = st.slider("Noise Intensity", 0.0, 0.5, 0.1, 0.05)
        
        crop = st.checkbox("Random Crop", value=False)
        
    if st.button("Generate Augmented Images"):
        handle_image_augmentation(image, flip_h, flip_v, rotate, rotate_angle if rotate else 0, 
                                  brightness, brightness_factor if brightness else 1.0, 
                                  contrast, contrast_factor if contrast else 1.0, 
                                  add_noise, noise_intensity if add_noise else 0.1, 
                                  crop)


def handle_image_augmentation(image, flip_h, flip_v, rotate, rotate_angle, 
                              brightness, brightness_factor, contrast, contrast_factor, 
                              add_noise, noise_intensity, crop):
    """Handle the image augmentation operations and display results."""
    # Store the original image
    images = [image]
    captions = ["Original"]
    
    # Apply selected augmentations
    if flip_h:
        img_h_flipped = ImageOps.mirror(image)
        images.append(img_h_flipped)
        captions.append("Horizontal Flip")
    
    if flip_v:
        img_v_flipped = ImageOps.flip(image)
        images.append(img_v_flipped)
        captions.append("Vertical Flip")
    
    if rotate:
        img_rotated = image.rotate(rotate_angle, expand=True)
        images.append(img_rotated)
        captions.append(f"Rotated ({rotate_angle}°)")
    
    if brightness:
        enhancer = ImageEnhance.Brightness(image)
        img_bright = enhancer.enhance(brightness_factor)
        images.append(img_bright)
        captions.append(f"Brightness ({brightness_factor})")
    
    if contrast:
        enhancer = ImageEnhance.Contrast(image)
        img_contrast = enhancer.enhance(contrast_factor)
        images.append(img_contrast)
        captions.append(f"Contrast ({contrast_factor})")
    
    if add_noise:
        # Convert to numpy array to add noise
        img_array = np.array(image).astype('float32')
        
        # Add Gaussian noise
        height, width, _ = img_array.shape
        noise = np.random.normal(0, 255 * noise_intensity, (height, width, 3))
        img_array = img_array + noise
        
        # Clip values to valid range
        img_array = np.clip(img_array, 0, 255).astype('uint8')
        
        # Convert back to PIL Image
        img_noisy = Image.fromarray(img_array)
        
        images.append(img_noisy)
        captions.append(f"With Noise ({noise_intensity})")
    
    if crop:
        # Get dimensions
        width, height = image.size
        
        # Define crop parameters (crop 80% of original size)
        crop_size = (int(width * 0.8), int(height * 0.8))
        
        # Calculate random position for crop
        left = np.random.randint(0, width - crop_size[0])
        top = np.random.randint(0, height - crop_size[1])
        right = left + crop_size[0]
        bottom = top + crop_size[1]
        
        # Crop image
        img_cropped = image.crop((left, top, right, bottom))
        
        # Resize back to original dimensions for better comparison
        img_cropped = img_cropped.resize((width, height))
        
        images.append(img_cropped)
        captions.append("Random Crop")
    
    # Save augmented images to session state
    st.session_state.augmented_images = (images, captions)
    
    # Display the results in a grid
    num_images = len(images)
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    for i in range(rows):
        row_cols = st.columns(cols)
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                row_cols[j].image(images[idx], caption=captions[idx], use_container_width=True)


def render_text_augmentation():
    """Render the text augmentation section."""
    st.markdown("""
    <h3>Text Augmentation</h3>
    <div class="info-box">
    <p>Text augmentation creates variations of text data while preserving its meaning or intent. This helps NLP models generalize better by exposing them to diverse linguistic expressions.</p>
    
    <h4>Common Techniques:</h4>
    <ul>
        <li><strong>Synonym Replacement</strong>: Replace words with their synonyms</li>
        <li><strong>Random Insertion/Deletion</strong>: Add or remove words randomly</li>
        <li><strong>Word Swapping</strong>: Change the order of words</li>
        <li><strong>Back-Translation</strong>: Translate text to another language and back</li>
        <li><strong>Text Generation</strong>: Use language models to create variations</li>
    </ul>
    </div>
    
    <div class="example-box">
    <h4>Benefits in NLP</h4>
    <ul>
        <li><strong>Improved Robustness</strong>: Models become less sensitive to specific phrasings</li>
        <li><strong>Better Generalization</strong>: Performance improves on unseen text variations</li>
        <li><strong>Handling Data Scarcity</strong>: Particularly useful for low-resource languages or domains</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive text augmentation demo
    st.markdown("<h4>Interactive Text Augmentation Demo</h4>", unsafe_allow_html=True)
    
    # Check if NLTK wordnet is downloaded
    try:
        wordnet.synsets('test')
    except LookupError:
        with st.spinner("Downloading required NLTK data (first run only)..."):
            nltk.download('wordnet')
    
    # Input text
    default_text = "Machine learning is an exciting field that involves developing algorithms that can learn from data."
    input_text = st.text_area("Enter text to augment", value=default_text, height=100)
    
    # Augmentation options
    st.markdown("<h5>Choose Augmentation Techniques</h5>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_synonym = st.checkbox("Synonym Replacement", value=True)
        if use_synonym:
            synonym_percent = st.slider("Percentage of words to replace", 10, 50, 30)
        
        use_deletion = st.checkbox("Random Word Deletion", value=False)
        if use_deletion:
            deletion_percent = st.slider("Percentage of words to delete", 5, 25, 10)
    
    with col2:
        use_swap = st.checkbox("Random Word Swap", value=False)
        if use_swap:
            swap_count = st.slider("Number of swaps", 1, 5, 2)
        
        use_capitalization = st.checkbox("Random Capitalization", value=False)
        
    if st.button("Generate Augmented Text"):
        handle_text_augmentation(input_text, use_synonym, synonym_percent, use_deletion, deletion_percent, 
                                 use_swap, swap_count, use_capitalization)


def handle_text_augmentation(input_text, use_synonym, synonym_percent, use_deletion, deletion_percent, 
                             use_swap, swap_count, use_capitalization):
    """Handle the text augmentation operations and display results."""
    # Process the text
    words = input_text.split()
    num_words = len(words)
    augmented_texts = [input_text]
    
    # Synonym replacement
    if use_synonym and num_words > 0:
        new_words = words.copy()
        num_to_replace = max(1, int(num_words * synonym_percent / 100))
        replace_indices = random.sample(range(num_words), min(num_to_replace, num_words))
        
        for idx in replace_indices:
            word = words[idx]
            # Skip very short words, punctuation, and special characters
            if len(word) <= 2 or not word.isalpha():
                continue
            
            # Find synonyms
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            
            # Replace with a synonym if available
            if synonyms:
                new_words[idx] = random.choice(synonyms).replace('_', ' ')
        
        augmented_texts.append(' '.join(new_words))
    
    # Random deletion
    if use_deletion and num_words > 3:  # Ensure we don't delete too much from short texts
        new_words = words.copy()
        num_to_delete = max(1, int(num_words * deletion_percent / 100))
        delete_indices = sorted(random.sample(range(num_words), min(num_to_delete, num_words - 2)), reverse=True)
        
        for idx in delete_indices:
            del new_words[idx]
        
        augmented_texts.append(' '.join(new_words))
    
    # Random swap
    if use_swap and num_words > 3:
        new_words = words.copy()
        for _ in range(min(swap_count, num_words-1)):
            idx1, idx2 = random.sample(range(num_words), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        augmented_texts.append(' '.join(new_words))
    
    # Random capitalization
    if use_capitalization:
        new_words = words.copy()
        for i in range(num_words):
            if random.random() < 0.15 and new_words[i].isalpha():  # 15% chance to capitalize
                new_words[i] = new_words[i].upper()
        
        augmented_texts.append(' '.join(new_words))
    
    # Save augmented texts to session state
    st.session_state.augmented_text = augmented_texts
    
    # Display results
    st.markdown("<h5>Augmented Text Variations:</h5>", unsafe_allow_html=True)
    
    for i, text in enumerate(augmented_texts):
        with st.expander(f"Version {i+1}{' (Original)' if i==0 else ''}"):
            st.markdown(f"<div style='background-color:#f8f9fa; padding:10px; border-radius:5px;'>{text}</div>", unsafe_allow_html=True)
    
    # Highlight the differences (for the first augmentation)
    if len(augmented_texts) > 1:
        st.markdown("<h5>Differences Highlighted:</h5>", unsafe_allow_html=True)
        
        orig_words = augmented_texts[0].split()
        aug_words = augmented_texts[1].split()
        
        # Find differences and highlight them
        highlighted_text = ""
        for i in range(min(len(orig_words), len(aug_words))):
            if orig_words[i] != aug_words[i]:
                highlighted_text += f"<mark style='background-color:#FFEB3B'>{aug_words[i]}</mark> "
            else:
                highlighted_text += aug_words[i] + " "
        
        # Add any remaining words
        if len(aug_words) > len(orig_words):
            for i in range(len(orig_words), len(aug_words)):
                highlighted_text += f"<mark style='background-color:#FFEB3B'>{aug_words[i]}</mark> "
        
        st.markdown(f"<div style='background-color:#f8f9fa; padding:10px; border-radius:5px;'>{highlighted_text}</div>", unsafe_allow_html=True)


def render_timeseries_augmentation():
    """Render the time series augmentation section."""
    st.markdown("""
    <h3>Time Series Augmentation</h3>
    <div class="info-box">
    <p>Time series augmentation generates synthetic time series data that preserves the essential temporal patterns while introducing variations. This helps time series models generalize better.</p>
    
    <h4>Common Techniques:</h4>
    <ul>
        <li><strong>Time Warping</strong>: Stretching or compressing segments of the time series</li>
        <li><strong>Magnitude Warping</strong>: Scaling the amplitude of segments</li>
        <li><strong>Jittering</strong>: Adding random noise to the values</li>
        <li><strong>Slicing</strong>: Taking random windows or segments of the time series</li>
        <li><strong>Permutation</strong>: Rearranging blocks or segments while preserving local temporal structure</li>
    </ul>
    </div>
    
    <div class="example-box">
    <h4>Benefits in Time Series Analysis</h4>
    <ul>
        <li><strong>Improved Forecasting</strong>: Models become more robust to variations in future data</li>
        <li><strong>Better Anomaly Detection</strong>: More effective identification of unusual patterns</li>
        <li><strong>Handling Seasonality</strong>: Better adaptation to seasonal or cyclic variations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive time series augmentation demo
    st.markdown("<h4>Interactive Time Series Augmentation Demo</h4>", unsafe_allow_html=True)
    
    # Generate synthetic time series if not already done
    if not hasattr(st.session_state, 'time_series_data'):
        # Create a synthetic time series with trend, seasonality, and noise
        t = np.linspace(0, 8*np.pi, 400)
        trend = 0.1 * t
        seasonality = 5 * np.sin(t) + 2 * np.sin(3*t)
        noise = np.random.normal(0, 0.5, len(t))
        
        time_series = trend + seasonality + noise
        
        # Create a DataFrame
        dates = pd.date_range(start='2023-01-01', periods=len(time_series), freq='D')
        ts_df = pd.DataFrame({'date': dates, 'value': time_series})
        
        st.session_state.time_series_data = ts_df
    
    # Plot the original time series
    fig = px.line(st.session_state.time_series_data, x='date', y='value', title='Original Time Series')
    st.plotly_chart(fig, use_container_width=True)
    
    # Augmentation options
    st.markdown("<h5>Choose Augmentation Techniques</h5>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_jitter = st.checkbox("Add Jitter (Noise)", value=True)
        if use_jitter:
            jitter_intensity = st.slider("Jitter Intensity", 0.1, 2.0, 0.5, 0.1)
        
        use_scaling = st.checkbox("Magnitude Scaling", value=False)
        if use_scaling:
            scale_factor = st.slider("Scaling Factor", 0.5, 2.0, 1.0, 0.1)
    
    with col2:
        use_time_warp = st.checkbox("Time Warping", value=False)
        if use_time_warp:
            warp_factor = st.slider("Warping Factor", 0.7, 1.3, 1.0, 0.05)
        
        use_window = st.checkbox("Random Window", value=False)
        if use_window:
            window_size = st.slider("Window Size (days)", 30, 300, 100)
    
    if st.button("Generate Augmented Time Series"):
        handle_timeseries_augmentation(use_jitter, jitter_intensity if use_jitter else 0.5,
                                      use_scaling, scale_factor if use_scaling else 1.0,
                                      use_time_warp, warp_factor if use_time_warp else 1.0,
                                      use_window, window_size if use_window else 100)


def handle_timeseries_augmentation(use_jitter, jitter_intensity, use_scaling, scale_factor,
                                  use_time_warp, warp_factor, use_window, window_size):
    """Handle the time series augmentation operations and display results."""
    original_ts = st.session_state.time_series_data.copy()
    augmented_series = [original_ts]
    titles = ["Original"]
    
    # Add jitter
    if use_jitter:
        jittered = original_ts.copy()
        noise = np.random.normal(0, jitter_intensity, len(jittered))
        jittered['value'] = jittered['value'] + noise
        augmented_series.append(jittered)
        titles.append(f"With Jitter (σ={jitter_intensity})")
    
    # Apply magnitude scaling
    if use_scaling:
        scaled = original_ts.copy()
        scaled['value'] = scaled['value'] * scale_factor
        augmented_series.append(scaled)
        titles.append(f"Magnitude Scaled (×{scale_factor})")
    
    # Apply time warping
    if use_time_warp:
        warped = original_ts.copy()
        
        # Simple time warping by resampling
        x = np.linspace(0, 1, len(warped))
        warped_x = x**warp_factor  # Non-linear transformation
        
        # Interpolate values at new warped positions
        original_values = warped['value'].values
        interp_func = lambda x_new: np.interp(x_new, x, original_values)
        warped_values = interp_func(warped_x)
        
        warped['value'] = warped_values
        augmented_series.append(warped)
        titles.append(f"Time Warped (factor={warp_factor})")
    
    # Extract random window
    if use_window:
        if len(original_ts) > window_size:
            start_idx = np.random.randint(0, len(original_ts) - window_size)
            windowed = original_ts.iloc[start_idx:start_idx + window_size].copy()
            windowed = windowed.reset_index(drop=True)
            augmented_series.append(windowed)
            titles.append(f"Random Window ({window_size} days)")
    
    # Plot the results
    fig = go.Figure()
    
    # Add each series
    for ts, title in zip(augmented_series, titles):
        fig.add_trace(go.Scatter(x=ts['date'], y=ts['value'], name=title))
    
    fig.update_layout(
        title="Original vs. Augmented Time Series",
        xaxis_title="Date",
        yaxis_title="Value",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistical comparison
    st.markdown("<h5>Statistical Comparison:</h5>", unsafe_allow_html=True)
    
    stats_data = []
    for i, (ts, title) in enumerate(zip(augmented_series, titles)):
        stats = {
            "Series": title,
            "Mean": ts['value'].mean(),
            "Std Dev": ts['value'].std(),
            "Min": ts['value'].min(),
            "Max": ts['value'].max()
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df.set_index("Series").round(2))
    
    # Save to session state
    st.session_state.augmented_timeseries = (augmented_series, titles)


def render_knowledge_check_tab():
    """Render the knowledge check quiz tab."""
    st.header("Test Your Knowledge")
    st.markdown("Let's see how well you understand data preparation techniques for machine learning!")
    
    # Quiz questions based on the content in the other tabs
    questions = [
        {
            "question": "What is the main purpose of splitting data into training, validation, and test sets?",
            "options": [
                "To make the model train faster", 
                "To ensure the model generalizes well to unseen data", 
                "To reduce memory requirements",
                "To make the code more organized"
            ],
            "correct": "To ensure the model generalizes well to unseen data",
            "explanation": "Data splitting helps ensure your model can perform well on new, unseen data by evaluating it on separate validation and test sets that weren't used during training."
        },
        {
            "question": "In a k-fold cross-validation with k=5, what percentage of the data is used for training in each fold?",
            "options": ["20%", "50%", "80%", "95%"],
            "correct": "80%",
            "explanation": "In k-fold cross-validation with k=5, the data is divided into 5 equal parts. In each iteration, 1 part (20%) is used for validation and the remaining 4 parts (80%) are used for training."
        },
        {
            "question": "Why is data shuffling important before splitting into train/test sets?",
            "options": [
                "It makes the model converge faster", 
                "It prevents the model from learning the order of samples", 
                "It ensures the train and test sets have similar distributions",
                "All of the above"
            ],
            "correct": "It ensures the train and test sets have similar distributions",
            "explanation": "Shuffling data before splitting helps ensure that both train and test sets have similar statistical distributions, avoiding bias that might exist in the original order of the data."
        },
        {
            "question": "Which of the following is NOT a common image augmentation technique?",
            "options": ["Horizontal flipping", "Brightness adjustment", "Dropout regularization", "Random cropping"],
            "correct": "Dropout regularization",
            "explanation": "Dropout is a regularization technique applied during model training, not a data augmentation method. The other options (flipping, brightness adjustment, and cropping) are common image augmentation techniques."
        },
        {
            "question": "For time series data with important temporal patterns, which approach is typically most appropriate?",
            "options": [
                "Random shuffling of all data points", 
                "Using sliding window approaches without shuffling", 
                "Mini-batch shuffling",
                "Epoch-based shuffling"
            ],
            "correct": "Using sliding window approaches without shuffling",
            "explanation": "Time series data often contains important sequential patterns. Using sliding window approaches preserves these temporal dependencies, while shuffling would disrupt the time-based patterns that the model needs to learn."
        },
    ]
    
    # Display questions
    for q_idx, question in enumerate(questions):
        st.subheader(f"Question {q_idx+1}")
        st.markdown(f"**{question['question']}**")
        
        # If quiz is not submitted, show radio buttons
        if not st.session_state.quiz_submitted:
            st.session_state.quiz_answers[f"q{q_idx}"] = st.radio(
                f"Select your answer for question {q_idx+1}:",
                question["options"],
                index=None,
                key=f"radio_{q_idx}"
            )
        # If quiz is submitted, show results
        else:
            user_answer = st.session_state.quiz_answers.get(f"q{q_idx}")
            if user_answer == question["correct"]:
                st.success(f"✅ Your answer: {user_answer}")
                st.info(f"Explanation: {question['explanation']}")
            else:
                st.error(f"❌ Your answer: {user_answer}")
                st.info(f"Correct answer: {question['correct']}")
                st.info(f"Explanation: {question['explanation']}")
        
        st.markdown("---")
    
    # Submit or reset buttons
    if not st.session_state.quiz_submitted:
        if st.button("Submit Answers"):
            submit_quiz(questions)
    else:
        st.header(f"Your Score: {st.session_state.quiz_score}/{len(questions)}")
        
        # Score interpretation
        if st.session_state.quiz_score == len(questions):
            st.balloons()
            st.success("🏆 Perfect score! You're a data preparation expert!")
        elif st.session_state.quiz_score >= len(questions) * 0.8:
            st.success("🎓 Great job! You have a strong understanding of data preparation techniques.")
        elif st.session_state.quiz_score >= len(questions) * 0.6:
            st.warning("📚 Good effort! Review the explanations to strengthen your knowledge.")
        else:
            st.error("🔄 You might want to revisit the earlier sections to reinforce your understanding.")
        
        if st.button("Take Quiz Again"):
            reset_quiz()


def submit_quiz(questions):
    """Handle the quiz submission."""
    score = 0
    for q_idx, question in enumerate(questions):
        if st.session_state.quiz_answers.get(f"q{q_idx}") == question["correct"]:
            score += 1
    st.session_state.quiz_score = score
    st.session_state.quiz_submitted = True


def reset_quiz():
    """Reset the quiz state."""
    st.session_state.quiz_score = 0
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers = {}


def render_resources_tab():
    """Render the resources tab."""
    st.markdown('<h2 class="sub-header">📚 Resources</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Further Learning Resources</h3>
    <p>To deepen your knowledge about data preparation for machine learning, explore these resources:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h4>Data Splitting & Cross-Validation</h4>
        <ul>
            <li><a href="https://scikit-learn.org/stable/modules/cross_validation.html" target="_blank">Scikit-learn: Cross-validation</a></li>
            <li><a href="https://www.tensorflow.org/tutorials/keras/overfit_and_underfit" target="_blank">TensorFlow: Overfitting and Underfitting</a></li>
            <li><a href="https://machinelearningmastery.com/k-fold-cross-validation/" target="_blank">Machine Learning Mastery: k-Fold Cross-Validation</a></li>
        </ul>
        
        <h4>Data Shuffling</h4>
        <ul>
            <li><a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader" target="_blank">PyTorch DataLoader Shuffling</a></li>
            <li><a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle" target="_blank">TensorFlow Dataset Shuffling</a></li>
            <li><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html" target="_blank">Pandas DataFrame Sampling and Shuffling</a></li>
        </ul>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <h4>Data Augmentation</h4>
        <ul>
            <li><a href="https://www.tensorflow.org/tutorials/images/data_augmentation" target="_blank">TensorFlow: Image Data Augmentation</a></li>
            <li><a href="https://pytorch.org/vision/stable/transforms.html" target="_blank">PyTorch: Torchvision Transforms</a></li>
            <li><a href="https://nlpaug.readthedocs.io/en/latest/" target="_blank">nlpaug: Text Augmentation Library</a></li>
            <li><a href="https://github.com/uchidalab/time_series_augmentation" target="_blank">Time Series Augmentation Techniques</a></li>
        </ul>
        
        <h4>AWS Resources</h4>
        <ul>
            <li><a href="https://aws.amazon.com/sagemaker/data-wrangler/" target="_blank">Amazon SageMaker Data Wrangler</a></li>
            <li><a href="https://aws.amazon.com/sagemaker/feature-store/" target="_blank">Amazon SageMaker Feature Store</a></li>
            <li><a href="https://aws.amazon.com/sagemaker/clarify/" target="_blank">Amazon SageMaker Clarify</a></li>
        </ul>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="example-box">
    <h4>Best Practices Summary</h4>
    <ol>
        <li><strong>Data Splitting</strong>: Always split your data before any model training to avoid data leakage</li>
        <li><strong>Cross-Validation</strong>: Use k-fold cross-validation for smaller datasets to maximize training data usage</li>
        <li><strong>Stratification</strong>: For classification tasks, use stratified splits to maintain class distribution</li>
        <li><strong>Shuffling</strong>: Always shuffle your data before splitting unless order is important (e.g., time series)</li>
        <li><strong>Augmentation</strong>: Apply domain-appropriate augmentation techniques to improve model robustness</li>
        <li><strong>Consistency</strong>: Use the same random seed across experiments for reproducibility</li>
        <li><strong>Evaluation</strong>: Always evaluate on a separate test set that was not used during model development</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>Common Pitfalls to Avoid</h4>
    <ul>
        <li><strong>Data Leakage</strong>: Performing data transformations before splitting can leak information</li>
        <li><strong>Peeking at Test Data</strong>: Never use test data for model selection or hyperparameter tuning</li>
        <li><strong>Inappropriate Shuffling</strong>: Don't shuffle time series or sequential data that has temporal dependencies</li>
        <li><strong>Overfitting to Validation</strong>: Too many iterations of model tuning can lead to overfitting to validation data</li>
        <li><strong>Unrealistic Augmentation</strong>: Ensure augmented data remains realistic for your problem domain</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Quick reference cheatsheet
    with st.expander("📋 Quick Reference Cheatsheet"):
        st.markdown("""
        ```python
        # Data Splitting - Simple Hold-out
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
        )
        
        # Further split training data to create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        
        # K-Fold Cross-Validation
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # Train and evaluate model
        
        # Stratified K-Fold for classification
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # Train and evaluate model
        ```
        """)


def render_footer():
    """Render the footer of the app."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app."""
    # Initialize session state variables
    initialize_session_state()
    
    
    # Apply custom CSS
    apply_custom_css()
    
    # Setup sidebar
    setup_sidebar()
    
    # Render main header
    render_main_header()
    
    # Tab navigation
    tabs = st.tabs([
        "📊 Data Splitting",
        "🔀 Data Shuffling",
        "🔄 Data Augmentation",
        "📋 Knowledge Check",
        "📚 Resources"
    ])
    
    # Render each tab's content
    with tabs[0]:
        render_data_splitting_tab()
    
    with tabs[1]:
        render_data_shuffling_tab()
    
    with tabs[2]:
        render_data_augmentation_tab()
    
    with tabs[3]:
        render_knowledge_check_tab()
    
    with tabs[4]:
        render_resources_tab()
    
    # Render footer
    render_footer()


# Main execution flow
if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()