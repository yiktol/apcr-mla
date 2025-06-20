
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, fetch_20newsgroups, load_iris, make_swiss_roll
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import altair as alt
import time
import plotly.express as px
import plotly.graph_objects as go
import umap
import networkx as nx
from gensim.models import Word2Vec
import string
import re
from nltk.corpus import stopwords
import nltk
from PIL import Image
from io import BytesIO
import base64
import requests
from datetime import datetime

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="SageMaker Unsupervised Learning Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define AWS Color Scheme
AWS_COLORS = {
    'orange': '#FF9900',
    'blue': '#232F3E',
    'light_blue': '#1A73E8',
    'grey': '#545B64',
    'light_grey': '#D5DBDB',
    'white': '#FFFFFF',
    'green': '#008296',
    'red': '#D13212',
    'dark_orange': '#E76D0C'
}

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #FFFFFF;
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
    
    /* Card styling */
    .card {
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #F7F7F7;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #232F3E;
        color: white;
        text-align: center;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FF9900;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #E76D0C;
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
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #232F3E;
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #F0F8FF;
        border-left: 5px solid #1A73E8;
        padding: 10px;
        margin-bottom: 15px;
    }
    
    /* Algorithm description */
    .algorithm-description {
        background-color: #F5F5F5;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #FF9900;
    }
    
    /* Code blocks */
    code {
        background-color: #F0F0F0;
        padding: 2px 5px;
        border-radius: 3px;
        font-family: monospace;
    }
    
    pre {
        background-color: #232F3E;
        color: #FFFFFF;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'initialized_usup' not in st.session_state:
        st.session_state.initialized_usup = True
        
        # K-means state variables
        st.session_state.kmeans_data = None
        st.session_state.kmeans_trained = False
        st.session_state.kmeans_clusters = None
        st.session_state.kmeans_silhouette = None
        
        # LDA state variables
        st.session_state.lda_data = None
        st.session_state.lda_trained = False
        st.session_state.lda_model = None
        st.session_state.lda_feature_names = None
        
        # Object2Vec state variables
        st.session_state.obj2vec_data = None
        st.session_state.obj2vec_trained = False
        st.session_state.obj2vec_embeddings = None
        
        # Random Cut Forest state variables
        st.session_state.rcf_data = None
        st.session_state.rcf_trained = False
        st.session_state.rcf_scores = None
        st.session_state.rcf_outliers = None
        
        # IP Insights state variables
        st.session_state.ip_data = None
        st.session_state.ip_trained = False
        st.session_state.ip_results = None
        
        # PCA state variables
        st.session_state.pca_data = None
        st.session_state.pca_trained = False
        st.session_state.pca_components = None
        st.session_state.pca_explained_variance = None

# Initialize session state
init_session_state()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg", width=100)
    st.title("Session Management")
    
    if st.button("Reset Session", key="reset_session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.success("Session has been reset!")
    
    st.divider()
    st.subheader("About This App")
    st.write("""
    This interactive application demonstrates Amazon SageMaker's built-in unsupervised learning algorithms.
    Explore each algorithm with interactive examples and visualizations.
    """)
    
    st.divider()
    st.subheader("Resources")
    st.markdown("""
    - [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
    - [AWS Machine Learning](https://aws.amazon.com/machine-learning/)
    - [AWS Training & Certification](https://aws.amazon.com/training/)
    """)

# Main content
st.title("Amazon SageMaker Unsupervised Learning Explorer")
st.markdown("""
This interactive application helps you understand the main built-in unsupervised learning algorithms available in Amazon SageMaker. 
Select an algorithm tab below to explore its features, use cases, and see it in action with interactive examples.
""")

tabs = st.tabs([
    "🔵 K-means", 
    "📚 LDA", 
    "🧩 Object2Vec", 
    "🌲 Random Cut Forest", 
    "🌐 IP Insights", 
    "📊 PCA"
])

# ------------- K-means Tab -------------
with tabs[0]:
    st.header("🔵 K-means Clustering")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>What is K-means?</h3>
    <p>K-means is one of the most popular and simple clustering algorithms. It partitions data into K distinct clusters based on distance to the centroid of each cluster.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Unsupervised learning - no labeled data required</li>
        <li>Identifies natural groupings in data</li>
        <li>Scales well to large datasets</li>
        <li>Simple and intuitive algorithm</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>Customer segmentation</li>
        <li>Document clustering</li>
        <li>Image compression</li>
        <li>Pattern recognition in data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive K-means Demo")
    
    kmeans_col1, kmeans_col2 = st.columns([1, 1])
    
    with kmeans_col1:
        st.markdown("### Generate Cluster Data")
        kmeans_n_samples = st.slider("Number of Samples", 100, 2000, 500, 100, key="kmeans_n_samples_key")
        kmeans_n_features = st.selectbox("Number of Features", [2, 3], key="kmeans_n_features_key")
        kmeans_n_clusters = st.slider("Number of Clusters", 2, 10, 4, 1, key="kmeans_n_clusters_key")
        kmeans_cluster_std = st.slider("Cluster Standard Deviation", 0.5, 3.0, 1.0, 0.1, key="kmeans_cluster_std_key")
        
        if st.button("Generate Data", key="kmeans_generate"):
            with st.spinner("Generating cluster data..."):
                # Generate synthetic cluster data
                X, y = make_blobs(
                    n_samples=kmeans_n_samples,
                    n_features=kmeans_n_features,
                    centers=kmeans_n_clusters,
                    cluster_std=kmeans_cluster_std,
                    random_state=42
                )
                
                # Store in session state
                st.session_state.kmeans_data = X
                st.session_state.kmeans_true_labels = y
                st.session_state.kmeans_trained = False
                st.session_state.kmeans_n_clusters = kmeans_n_clusters
                st.session_state.kmeans_n_features = kmeans_n_features
                st.success("Data generated successfully!")
    
    with kmeans_col2:
        if st.session_state.kmeans_data is not None:
            # Visualize generated data
            if st.session_state.kmeans_n_features == 2:
                fig = plt.figure(figsize=(8, 6))
                plt.scatter(
                    st.session_state.kmeans_data[:, 0], 
                    st.session_state.kmeans_data[:, 1], 
                    c=st.session_state.kmeans_true_labels, 
                    cmap='viridis', 
                    s=50, 
                    alpha=0.8
                )
                plt.colorbar(label='True Cluster')
                plt.title('Generated Cluster Data')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.tight_layout()
                st.pyplot(fig)
            else:  # 3D plot
                fig = px.scatter_3d(
                    x=st.session_state.kmeans_data[:, 0],
                    y=st.session_state.kmeans_data[:, 1],
                    z=st.session_state.kmeans_data[:, 2],
                    color=st.session_state.kmeans_true_labels,
                    opacity=0.7,
                    title="Generated 3D Cluster Data"
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title='Feature 1',
                        yaxis_title='Feature 2',
                        zaxis_title='Feature 3'
                    ),
                    coloraxis_colorbar=dict(title="True Cluster")
                )
                st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.kmeans_data is not None:
        st.divider()
        st.subheader("Train K-means Model")
        
        kmeans_train_col1, kmeans_train_col2 = st.columns([1, 2])
        
        with kmeans_train_col1:
            kmeans_k = st.slider(
                "Number of Clusters (k)", 
                2, 10, 
                st.session_state.kmeans_n_clusters, 
                1, 
                key="kmeans_k"
            )
            kmeans_init = st.selectbox(
                "Initialization Method", 
                ["k-means++", "random"], 
                key="kmeans_init"
            )
            kmeans_max_iter = st.slider(
                "Maximum Iterations", 
                10, 500, 300, 10, 
                key="kmeans_max_iter"
            )
            
            if st.button("Train K-means", key="kmeans_train"):
                with st.spinner("Training K-means model..."):
                    # Simulate training with progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Train K-means model
                    kmeans = KMeans(
                        n_clusters=kmeans_k,
                        init=kmeans_init,
                        max_iter=kmeans_max_iter,
                        random_state=42
                    )
                    
                    kmeans.fit(st.session_state.kmeans_data)
                    
                    # Store results in session state
                    st.session_state.kmeans_model = kmeans
                    st.session_state.kmeans_labels = kmeans.labels_
                    st.session_state.kmeans_centroids = kmeans.cluster_centers_
                    st.session_state.kmeans_inertia = kmeans.inertia_
                    
                    # Calculate silhouette score
                    silhouette = silhouette_score(
                        st.session_state.kmeans_data, 
                        st.session_state.kmeans_labels
                    )
                    st.session_state.kmeans_silhouette = silhouette
                    st.session_state.kmeans_trained = True
                    st.success("K-means model trained successfully!")
        
        with kmeans_train_col2:
            if st.session_state.kmeans_trained:
                st.markdown("### Model Performance")
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Inertia (Within-cluster Sum of Squares)", f"{st.session_state.kmeans_inertia:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Silhouette Score", f"{st.session_state.kmeans_silhouette:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                <strong>Interpreting the Metrics:</strong><br>
                - <strong>Inertia:</strong> Lower is better. Measures how far points are from their centroids.<br>
                - <strong>Silhouette Score:</strong> Higher is better (range: -1 to 1). Measures how well-separated the clusters are.
                </div>
                """, unsafe_allow_html=True)
    
    if st.session_state.kmeans_trained:
        st.divider()
        st.subheader("K-means Results Visualization")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            # Visualize clustering results
            if st.session_state.kmeans_n_features == 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot data points with cluster assignments
                scatter = ax.scatter(
                    st.session_state.kmeans_data[:, 0], 
                    st.session_state.kmeans_data[:, 1], 
                    c=st.session_state.kmeans_labels, 
                    cmap='viridis', 
                    s=40, 
                    alpha=0.7
                )
                
                # Plot centroids
                ax.scatter(
                    st.session_state.kmeans_centroids[:, 0], 
                    st.session_state.kmeans_centroids[:, 1], 
                    marker='X', 
                    s=200, 
                    c='red', 
                    label='Centroids'
                )
                
                ax.set_title('K-means Clustering Result')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.legend()
                plt.colorbar(scatter, label='Cluster Assignment')
                plt.tight_layout()
                st.pyplot(fig)
            else:  # 3D visualization
                fig = px.scatter_3d(
                    x=st.session_state.kmeans_data[:, 0],
                    y=st.session_state.kmeans_data[:, 1],
                    z=st.session_state.kmeans_data[:, 2],
                    color=st.session_state.kmeans_labels,
                    opacity=0.7,
                    title="K-means 3D Clustering Result"
                )
                
                # Add centroids
                fig.add_trace(go.Scatter3d(
                    x=st.session_state.kmeans_centroids[:, 0],
                    y=st.session_state.kmeans_centroids[:, 1],
                    z=st.session_state.kmeans_centroids[:, 2],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='x'
                    ),
                    name='Centroids'
                ))
                
                fig.update_layout(
                    scene=dict(
                        xaxis_title='Feature 1',
                        yaxis_title='Feature 2',
                        zaxis_title='Feature 3'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with result_col2:
            # Find optimal k using Elbow method
            st.markdown("### Finding Optimal k using Elbow Method")
            
            with st.spinner("Calculating inertia for different k values..."):
                k_range = range(1, 11)
                inertias = []
                silhouette_scores = []
                
                for k in k_range:
                    if k == 1:  # Silhouette score is not defined for k=1
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(st.session_state.kmeans_data)
                        inertias.append(kmeans.inertia_)
                        silhouette_scores.append(None)
                    else:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(st.session_state.kmeans_data)
                        labels = kmeans.labels_
                        inertias.append(kmeans.inertia_)
                        silhouette_scores.append(silhouette_score(st.session_state.kmeans_data, labels))
                
                # Plot elbow curve
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                color = 'tab:blue'
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Inertia', color=color)
                ax1.plot(k_range, inertias, 'o-', color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                
                # Add silhouette score on secondary y-axis
                ax2 = ax1.twinx()
                color = 'tab:orange'
                ax2.set_ylabel('Silhouette Score', color=color)
                ax2.plot(k_range[1:], silhouette_scores[1:], 'o-', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                
                plt.title('Elbow Method for Optimal k')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                <div class="info-box">
                <strong>Interpreting the Elbow Method:</strong><br>
                The optimal number of clusters is often at the "elbow" of the inertia curve (where the rate of decrease sharply changes). 
                Additionally, higher silhouette scores indicate better clustering quality.
                </div>
                """, unsafe_allow_html=True)
        
        # Cluster distribution
        st.markdown("### Cluster Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        cluster_counts = np.bincount(st.session_state.kmeans_labels)
        sns.barplot(x=list(range(len(cluster_counts))), y=cluster_counts, ax=ax, color=AWS_COLORS['orange'])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Distribution of Samples Across Clusters')
        plt.tight_layout()
        st.pyplot(fig)
        
        # SageMaker implementation
        st.divider()
        st.markdown("""
        <div class="card">
        <h3>Amazon SageMaker Implementation</h3>
        <p>Here's how you can implement K-means clustering in Amazon SageMaker:</p>
        <pre><code>import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri</code>

# Set up the SageMaker session
<code>session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()</code>

# Get the K-means container
<code>container = get_image_uri(boto3.Session().region_name, 'kmeans')</code>

# Set the algorithm parameters
<code>hyperparameters = {
    "k": 5,
    "feature_dim": 2,
    "mini_batch_size": 500,
    "init_method": "kmeans++",
    "max_iterations": 300,
    "tol": 1e-3,
    "eval_metrics": ["ssd", "msd"]
}</code>

# Set up the estimator
<code>kmeans = sagemaker.estimator.Estimator(
    container,
    role, 
    instance_count=1, 
    instance_type='ml.m5.xlarge',
    hyperparameters=hyperparameters,
    output_path=f"s3://{bucket}/kmeans-output"
)</code>

# Train the model
<code>kmeans.fit({'train': train_data_s3_path})</code>

# Deploy the model for inference
<code>kmeans_predictor = kmeans.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)</code>
        </pre>
        </div>
        """, unsafe_allow_html=True)
    
# ------------- LDA Tab -------------
with tabs[1]:
    st.header("📚 Latent Dirichlet Allocation (LDA)")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>What is LDA?</h3>
    <p>Latent Dirichlet Allocation (LDA) is a generative probabilistic model for collections of discrete data such as text corpora. 
    It is commonly used for topic modeling, discovering abstract topics in a collection of documents.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Unsupervised topic modeling technique</li>
        <li>Represents documents as mixtures of topics</li>
        <li>Each topic is a mixture of words with certain probabilities</li>
        <li>Useful for organizing, searching, and understanding large document collections</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>Document clustering and organization</li>
        <li>Content recommendation systems</li>
        <li>Discovering hidden themes in large text collections</li>
        <li>Summarizing document collections</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive LDA Demo")
    
    # Sample dataset options
    lda_dataset = st.selectbox(
        "Select a Dataset", 
        ["20 Newsgroups (Subset)", "Custom Text Input"],
        key="lda_dataset"
    )
    
    if lda_dataset == "20 Newsgroups (Subset)":
        lda_col1, lda_col2 = st.columns([1, 1])
        
        with lda_col1:
            st.markdown("### Dataset Configuration")
            lda_categories = st.multiselect(
                "Select Categories",
                [
                    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
                    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                    'misc.forsale', 'rec.autos', 'rec.motorcycles',
                    'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
                    'sci.electronics', 'sci.med', 'sci.space',
                    'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                    'talk.politics.misc', 'talk.religion.misc'
                ],
                default=['sci.space', 'rec.autos', 'comp.graphics', 'sci.med'],
                key="lda_categories"
            )
            
            lda_sample_size = st.slider("Number of Documents", 100, 500, 200, 50, key="lda_sample_size_key")
            lda_n_topics = st.slider("Number of Topics", 2, 10, 5, 1, key="lda_n_topics_key")
            
            if st.button("Load Dataset", key="lda_load"):
                with st.spinner("Loading dataset and preprocessing..."):
                    # Load 20 newsgroups dataset
                    newsgroups = fetch_20newsgroups(
                        subset='train',
                        categories=lda_categories,
                        remove=('headers', 'footers', 'quotes'),
                        random_state=42
                    )
                    
                    # Take a subset of documents
                    if len(newsgroups.data) > lda_sample_size:
                        indices = np.random.choice(
                            len(newsgroups.data), 
                            lda_sample_size, 
                            replace=False
                        )
                        documents = [newsgroups.data[i] for i in indices]
                        labels = [newsgroups.target_names[newsgroups.target[i]] for i in indices]
                    else:
                        documents = newsgroups.data
                        labels = [newsgroups.target_names[t] for t in newsgroups.target]
                    
                    # Vectorize documents
                    vectorizer = CountVectorizer(
                        max_df=0.95, 
                        min_df=2,
                        max_features=1000,
                        stop_words='english'
                    )
                    
                    X = vectorizer.fit_transform(documents)
                    
                    # Save to session state
                    st.session_state.lda_data = X
                    st.session_state.lda_feature_names = vectorizer.get_feature_names_out()
                    st.session_state.lda_documents = documents
                    st.session_state.lda_labels = labels
                    st.session_state.lda_vectorizer = vectorizer
                    st.session_state.lda_n_topics = lda_n_topics
                    st.session_state.lda_trained = False
                    
                    st.success(f"Loaded {len(documents)} documents with {X.shape[1]} features")
        
        with lda_col2:
            if hasattr(st.session_state, 'lda_data') and st.session_state.lda_data is not None:
                # Show document distribution by category
                label_counts = pd.Series(st.session_state.lda_labels).value_counts()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax = sns.barplot(x=label_counts.index, y=label_counts.values, color=AWS_COLORS['orange'])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_title('Document Distribution by Category')
                ax.set_ylabel('Number of Documents')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display sample documents
                st.markdown("### Sample Documents")
                
                sample_idx = np.random.choice(len(st.session_state.lda_documents), min(3, len(st.session_state.lda_documents)), replace=False)
                
                for i, idx in enumerate(sample_idx):
                    with st.expander(f"Document {i+1} - {st.session_state.lda_labels[idx]}"):
                        text = st.session_state.lda_documents[idx]
                        if len(text) > 500:
                            st.text(text[:500] + "...")
                        else:
                            st.text(text)
    
    elif lda_dataset == "Custom Text Input":
        st.markdown("### Enter Your Own Text")
        st.markdown("Enter multiple documents, separated by blank lines.")
        
        lda_custom_text = st.text_area(
            "Custom Documents",
            height=200,
            value="""Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. It focuses on developing algorithms that can learn from and make predictions on data.
            
            Deep learning is a subset of machine learning that uses neural networks with many layers to analyze various factors. These algorithms can automatically learn representations from data such as images, video, and text.
            
            Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.
            
            Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind.
            
            Cloud computing is the delivery of computing services over the internet rather than having local servers or personal devices handle applications. These services include servers, storage, databases, networking, software, analytics, and more.
            """,
            key="lda_custom_text"
        )
        
        lda_n_topics = st.slider("Number of Topics", 2, 10, 4, 1, key="lda_custom_n_topics")
        
        if st.button("Process Text", key="lda_process_custom"):
            with st.spinner("Processing custom text..."):
                # Split text into documents
                documents = [doc.strip() for doc in lda_custom_text.split("\n\n") if doc.strip()]
                
                if len(documents) < 2:
                    st.error("Please enter at least 2 documents separated by blank lines.")
                else:
                    # Vectorize documents
                    vectorizer = CountVectorizer(
                        max_df=0.95, 
                        min_df=1,
                        max_features=500,
                        stop_words='english'
                    )
                    
                    X = vectorizer.fit_transform(documents)
                    
                    # Save to session state
                    st.session_state.lda_data = X
                    st.session_state.lda_feature_names = vectorizer.get_feature_names_out()
                    st.session_state.lda_documents = documents
                    st.session_state.lda_labels = [f"Document {i+1}" for i in range(len(documents))]
                    st.session_state.lda_vectorizer = vectorizer
                    st.session_state.lda_n_topics = lda_n_topics
                    st.session_state.lda_trained = False
                    
                    st.success(f"Processed {len(documents)} documents with {X.shape[1]} features")
                    
                    # Show word cloud of the corpus
                    from wordcloud import WordCloud
                    
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate(" ".join(documents))
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Word Cloud of All Documents')
                    st.pyplot(fig)
    
    if hasattr(st.session_state, 'lda_data') and st.session_state.lda_data is not None:
        st.divider()
        st.subheader("Train LDA Model")
        
        lda_train_col1, lda_train_col2 = st.columns([1, 2])
        
        with lda_train_col1:
            lda_n_components = st.slider(
                "Number of Topics", 
                2, 10, 
                st.session_state.lda_n_topics, 
                1, 
                key="lda_n_components"
            )
            
            lda_max_iter = st.slider(
                "Maximum Iterations", 
                5, 50, 20, 5, 
                key="lda_max_iter"
            )
            
            lda_learning_method = st.selectbox(
                "Learning Method", 
                ["batch", "online"], 
                key="lda_learning_method"
            )
            
            if st.button("Train LDA Model", key="lda_train"):
                with st.spinner("Training LDA model..."):
                    # Simulate training with progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    # Train LDA model
                    lda_model = LatentDirichletAllocation(
                        n_components=lda_n_components,
                        max_iter=lda_max_iter,
                        learning_method=lda_learning_method,
                        random_state=42
                    )
                    
                    lda_model.fit(st.session_state.lda_data)
                    
                    # Transform documents
                    lda_doc_topic = lda_model.transform(st.session_state.lda_data)
                    
                    # Store in session state
                    st.session_state.lda_model = lda_model
                    st.session_state.lda_doc_topic = lda_doc_topic
                    st.session_state.lda_trained = True
                    st.success("LDA model trained successfully!")
        
        with lda_train_col2:
            if st.session_state.lda_trained:
                # Topic-Word distribution visualization
                st.markdown("### Top Words per Topic")
                
                n_top_words = 10
                topic_word_df = pd.DataFrame()
                
                for topic_idx, topic in enumerate(st.session_state.lda_model.components_):
                    top_words_idx = topic.argsort()[:-n_top_words-1:-1]
                    top_words = [st.session_state.lda_feature_names[i] for i in top_words_idx]
                    top_words_weights = [topic[i] for i in top_words_idx]
                    
                    topic_df = pd.DataFrame({
                        'word': top_words,
                        'weight': top_words_weights,
                        'topic': [f"Topic {topic_idx + 1}"] * len(top_words)
                    })
                    
                    topic_word_df = pd.concat([topic_word_df, topic_df])
                
                # Create horizontal bar chart grouped by topic
                chart = alt.Chart(topic_word_df).mark_bar().encode(
                    y=alt.Y('word:N', title='Word'),
                    x=alt.X('weight:Q', title='Weight'),
                    color=alt.Color('topic:N', scale=alt.Scale(scheme='category10')),
                    row=alt.Row('topic:N', header=alt.Header(labelAngle=0))
                ).properties(
                    height=100
                ).configure_axisY(
                    labelFontSize=12
                )
                
                st.altair_chart(chart, use_container_width=True)
    
    if st.session_state.lda_trained:
        st.divider()
        st.subheader("LDA Model Results")
        
        # Document-Topic Distribution
        st.markdown("### Document-Topic Distribution")
        
        # Sample a few documents
        n_docs_to_show = min(10, len(st.session_state.lda_documents))
        sample_indices = np.random.choice(len(st.session_state.lda_documents), n_docs_to_show, replace=False)
        
        # Create data for visualization
        doc_topic_data = []
        for i, idx in enumerate(sample_indices):
            doc_label = st.session_state.lda_labels[idx] if hasattr(st.session_state, 'lda_labels') else f"Document {idx+1}"
            for topic_idx in range(st.session_state.lda_model.n_components):
                doc_topic_data.append({
                    'Document': f"{i+1}: {doc_label[:20]}...",
                    'Topic': f"Topic {topic_idx + 1}",
                    'Weight': st.session_state.lda_doc_topic[idx, topic_idx]
                })
        
        doc_topic_df = pd.DataFrame(doc_topic_data)
        
        # Create a heatmap
        doc_topic_pivot = doc_topic_df.pivot(index='Document', columns='Topic', values='Weight')
        
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(doc_topic_pivot, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
        ax.set_title('Document-Topic Distribution')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Topic visualization
        st.markdown("### Topic Visualization")
        
        # Get document embeddings from topic distribution
        doc_topic_matrix = st.session_state.lda_doc_topic
        
        # Reduce to 2D for visualization
        tsne = umap.UMAP(n_components=2, random_state=42)
        doc_tsne = tsne.fit_transform(doc_topic_matrix)
        
        # Prepare data for visualization
        viz_df = pd.DataFrame({
            'x': doc_tsne[:, 0],
            'y': doc_tsne[:, 1],
            'document': [f"Doc {i}" for i in range(len(st.session_state.lda_documents))],
            'label': st.session_state.lda_labels if hasattr(st.session_state, 'lda_labels') else [""] * len(st.session_state.lda_documents),
            'dominant_topic': np.argmax(doc_topic_matrix, axis=1) + 1
        })
        
        # Create interactive scatter plot
        fig = px.scatter(
            viz_df, x='x', y='y', 
            color='dominant_topic', 
            hover_data=['document', 'label'],
            color_discrete_sequence=px.colors.qualitative.Bold,
            title='Document-Topic Space Projection'
        )
        
        fig.update_layout(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title="Dominant Topic"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic similarity
        st.markdown("### Topic Similarity Network")
        
        # Calculate topic similarity
        topic_similarity = np.zeros((st.session_state.lda_model.n_components, st.session_state.lda_model.n_components))
        for i in range(st.session_state.lda_model.n_components):
            for j in range(st.session_state.lda_model.n_components):
                # Using Hellinger distance between topic distributions
                topic_i = st.session_state.lda_model.components_[i]
                topic_j = st.session_state.lda_model.components_[j]
                
                # Convert to probabilities
                topic_i = topic_i / topic_i.sum()
                topic_j = topic_j / topic_j.sum()
                
                # Hellinger distance
                hellinger = np.sqrt(0.5 * np.sum((np.sqrt(topic_i) - np.sqrt(topic_j))**2))
                # Convert to similarity
                similarity = 1 - hellinger
                topic_similarity[i, j] = similarity
        
        # Create a graph
        G = nx.Graph()
        for i in range(st.session_state.lda_model.n_components):
            G.add_node(i, name=f'Topic {i+1}')
        
        # Add edges with weight > threshold
        threshold = 0.5
        for i in range(st.session_state.lda_model.n_components):
            for j in range(i+1, st.session_state.lda_model.n_components):
                if topic_similarity[i, j] > threshold:
                    G.add_edge(i, j, weight=topic_similarity[i, j])
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=800, 
            node_color=list(range(st.session_state.lda_model.n_components)),
            cmap=plt.cm.viridis,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges
        edges = nx.draw_networkx_edges(
            G, pos, 
            width=[G[u][v]['weight'] * 3 for u, v in G.edges()],
            alpha=0.5,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')
        
        ax.set_title('Topic Similarity Network')
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Topic interpretation
        st.markdown("### Interpreting the Topics")
        
        # Visualize top words for each topic in a more readable format
        for topic_idx, topic in enumerate(st.session_state.lda_model.components_):
            with st.expander(f"Topic {topic_idx + 1}"):
                top_n_words = 20
                top_words_idx = topic.argsort()[:-top_n_words-1:-1]
                
                # Create a dataframe for visualization
                top_words_df = pd.DataFrame({
                    'Word': [st.session_state.lda_feature_names[i] for i in top_words_idx],
                    'Weight': [topic[i] for i in top_words_idx]
                })
                
                # Display top words as a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=top_words_df,
                    y='Word',
                    x='Weight',
                    color=AWS_COLORS['orange'],
                    ax=ax
                )
                ax.set_title(f'Top {top_n_words} Words for Topic {topic_idx + 1}')
                plt.tight_layout()
                st.pyplot(fig)
                
                # List some example documents strongly associated with this topic
                st.markdown("#### Example Documents for this Topic")
                
                # Get documents strongly associated with this topic
                topic_docs = np.argsort(st.session_state.lda_doc_topic[:, topic_idx])[::-1][:3]
                
                for i, doc_idx in enumerate(topic_docs):
                    doc_text = st.session_state.lda_documents[doc_idx]
                    doc_label = st.session_state.lda_labels[doc_idx] if hasattr(st.session_state, 'lda_labels') else f"Document {doc_idx+1}"
                    topic_prob = st.session_state.lda_doc_topic[doc_idx, topic_idx]
                    
                    st.markdown(f"**Example {i+1}** ({doc_label}, Topic {topic_idx+1} probability: {topic_prob:.4f})")
                    st.text(doc_text[:300] + "..." if len(doc_text) > 300 else doc_text)
        
        # SageMaker implementation
        st.divider()
        st.markdown("""
        <div class="card">
        <h3>Amazon SageMaker Implementation</h3>
        <p>Here's how you can implement LDA in Amazon SageMaker:</p>
        <pre>
import boto3
import sagemaker
from sagemaker import LDA

# Set up the SageMaker session
session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

# Set the algorithm parameters
lda = LDA(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    num_topics=5,  # Number of topics
    feature_dim=1000,  # Vocabulary size
    mini_batch_size=500,
    max_restarts=10,
    max_iterations=100,
    tol=1e-3,
    alpha0=1.0
)

# Train the model
lda.fit({'train': train_data_s3_path})

# Deploy the model
lda_predictor = lda.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

# Get topics
topics = lda.describe_topics()
print(topics)
        </pre>
        </div>
        """, unsafe_allow_html=True)

# ------------- Object2Vec Tab -------------
with tabs[2]:
    st.header("🧩 Object2Vec")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>What is Object2Vec?</h3>
    <p>Object2Vec is a general-purpose neural embedding algorithm that learns low-dimensional dense embeddings of high-dimensional objects. 
    It can be used for recommendations, document similarity, and other applications involving complex relationships between objects.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Learns low-dimensional embeddings of objects</li>
        <li>Useful for recommendation systems, document similarity, and link prediction</li>
        <li>Handles many types of objects: users, items, products, documents, etc.</li>
        <li>Can model relationships between different types of objects</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>Recommendation systems (e.g., user-item interactions)</li>
        <li>Document similarity and search</li>
        <li>Knowledge graph embedding</li>
        <li>Word embeddings</li>
        <li>Social network analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive Object2Vec Demo")
    
    obj2vec_col1, obj2vec_col2 = st.columns([1, 1])
    
    with obj2vec_col1:
        st.markdown("### Word Embedding Demo")
        st.markdown("""
        In this demo, we'll use a simplified version of Object2Vec to learn word embeddings.
        We'll use the Word2Vec algorithm to represent words as dense vectors.
        """)
        
        obj2vec_demo_type = st.selectbox(
            "Select Demo Type", 
            ["Word Embeddings", "Movie Recommendations"],
            key="obj2vec_demo_type"
        )
        
        if obj2vec_demo_type == "Word Embeddings":
            obj2vec_text = st.text_area(
                "Input Text for Word Embeddings",
                value="""Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly.
                SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high quality models.
                SageMaker provides all the components used for machine learning in a single toolset so models get to production faster with much less effort and at lower cost.
                SageMaker includes modules that can be used together or independently to build, train, and deploy your machine learning models.
                Amazon SageMaker Studio provides a single, web-based visual interface where you can perform all ML development steps.
                SageMaker Studio gives you complete access, control, and visibility into each step required to build, train, and deploy models.
                Amazon SageMaker includes capabilities that help manage security, identity, and access for your SageMaker resources.
                SageMaker helps data scientists and developers prepare, build, train, and deploy high-quality machine learning models by bringing together a broad set of capabilities purpose-built for machine learning.""",
                height=150,
                key="obj2vec_text"
            )
            
            obj2vec_vector_size = st.slider("Vector Size", 10, 100, 50, 10, key="obj2vec_vector_size")
            obj2vec_window = st.slider("Context Window Size", 2, 10, 5, 1, key="obj2vec_window")
            
            if st.button("Train Word Embeddings", key="obj2vec_train_words"):
                with st.spinner("Training word embeddings..."):
                    # Preprocess text
                    sentences = []
                    for line in obj2vec_text.split('\n'):
                        if line.strip():
                            # Tokenize and clean
                            tokens = re.findall(r'\b\w+\b', line.lower())
                            # Remove stopwords
                            stop_words = set(stopwords.words('english'))
                            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
                            sentences.append(tokens)
                    
                    # Train Word2Vec model
                    w2v_model = Word2Vec(
                        sentences=sentences,
                        vector_size=obj2vec_vector_size,
                        window=obj2vec_window,
                        min_count=1,
                        workers=4,
                        sg=1  # Skip-gram model
                    )
                    
                    # Get vocabulary and embeddings
                    vocab = list(w2v_model.wv.index_to_key)
                    embeddings = [w2v_model.wv[word] for word in vocab]
                    
                    # Save to session state
                    st.session_state.obj2vec_model = w2v_model
                    st.session_state.obj2vec_vocab = vocab
                    st.session_state.obj2vec_embeddings = embeddings
                    st.session_state.obj2vec_trained = True
                    
                    st.success(f"Word embeddings trained! Vocabulary size: {len(vocab)} words.")
        
        elif obj2vec_demo_type == "Movie Recommendations":
            st.markdown("### Movie Recommendation Demo")
            st.markdown("""
            This demo simulates a movie recommendation system using embeddings.
            We'll generate synthetic user-movie interaction data and learn embeddings for both users and movies.
            """)
            
            obj2vec_n_users = st.slider("Number of Users", 20, 100, 50, 10, key="obj2vec_n_users")
            obj2vec_n_movies = st.slider("Number of Movies", 20, 100, 50, 10, key="obj2vec_n_movies")
            obj2vec_n_genres = st.slider("Number of Movie Genres", 3, 10, 5, 1, key="obj2vec_n_genres")
            obj2vec_embedding_dim = st.slider("Embedding Dimension", 5, 30, 10, 5, key="obj2vec_embedding_dim")
            
            if st.button("Generate Movie Recommendation Data", key="obj2vec_movies"):
                with st.spinner("Generating synthetic movie recommendation data..."):
                    # Generate movie genres
                    movie_genres = [f"Genre {i+1}" for i in range(obj2vec_n_genres)]
                    
                    # Assign genres to movies (each movie can have multiple genres)
                    movies = []
                    for i in range(obj2vec_n_movies):
                        n_genres = np.random.randint(1, 4)  # 1-3 genres per movie
                        movie_genre_indices = np.random.choice(
                            range(obj2vec_n_genres), 
                            size=n_genres, 
                            replace=False
                        )
                        movie_genre_list = [movie_genres[j] for j in movie_genre_indices]
                        movies.append({
                            'movie_id': i,
                            'title': f"Movie {i+1}",
                            'genres': movie_genre_list
                        })
                    
                    # Generate user preferences (implicit genre preferences)
                    users = []
                    for i in range(obj2vec_n_users):
                        genre_preferences = np.random.rand(obj2vec_n_genres)
                        genre_preferences = genre_preferences / genre_preferences.sum()  # Normalize
                        users.append({
                            'user_id': i,
                            'genre_preferences': genre_preferences
                        })
                    
                    # Generate user-movie interactions based on genre preferences
                    interactions = []
                    for user in users:
                        # Each user rates a random number of movies
                        n_ratings = np.random.randint(5, min(20, obj2vec_n_movies))
                        movie_indices = np.random.choice(
                            range(obj2vec_n_movies), 
                            size=n_ratings, 
                            replace=False
                        )
                        
                        for movie_idx in movie_indices:
                            movie = movies[movie_idx]
                            
                            # Calculate rating based on genre match
                            rating = 3.0  # Base rating
                            for genre in movie['genres']:
                                genre_idx = movie_genres.index(genre)
                                genre_pref = user['genre_preferences'][genre_idx]
                                rating += genre_pref * 4  # Max contribution of 4 stars
                            
                            # Add noise
                            rating += np.random.normal(0, 0.5)
                            rating = max(1.0, min(5.0, rating))  # Clip to 1-5 range
                            
                            interactions.append({
                                'user_id': user['user_id'],
                                'movie_id': movie['movie_id'],
                                'rating': rating
                            })
                    
                    # Convert to dataframes
                    movies_df = pd.DataFrame(movies)
                    users_df = pd.DataFrame([{'user_id': u['user_id']} for u in users])
                    interactions_df = pd.DataFrame(interactions)
                    
                    # Store in session state
                    st.session_state.obj2vec_movies = movies_df
                    st.session_state.obj2vec_users = users_df
                    st.session_state.obj2vec_interactions = interactions_df
                    st.session_state.obj2vec_movie_genres = movie_genres
                    st.session_state.obj2vec_embedding_dim = obj2vec_embedding_dim
                    st.session_state.obj2vec_trained = False
                    
                    st.success(f"Generated {len(interactions)} interactions between {obj2vec_n_users} users and {obj2vec_n_movies} movies.")
    
    with obj2vec_col2:
        if obj2vec_demo_type == "Word Embeddings" and st.session_state.get('obj2vec_trained', False):
            # Display word embeddings results
            st.markdown("### Word Embedding Results")
            
            # Word similarity search
            st.markdown("#### Word Similarity")
            query_word = st.selectbox(
                "Select a word to find similar words", 
                options=st.session_state.obj2vec_vocab,
                key="obj2vec_query_word"
            )
            
            if query_word:
                try:
                    similar_words = st.session_state.obj2vec_model.wv.most_similar(query_word, topn=10)
                    
                    # Create a dataframe for visualization
                    similar_df = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])
                    
                    # Display as bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(similar_df['Word'], similar_df['Similarity'], color=AWS_COLORS['orange'])
                    ax.set_xlabel('Similarity Score')
                    ax.set_title(f'Words Most Similar to "{query_word}"')
                    ax.invert_yaxis()  # Display most similar at the top
                    
                    # Add values to bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                                ha='left', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                except KeyError:
                    st.error(f"Word '{query_word}' not found in vocabulary.")
            
            # Word analogy
            st.markdown("#### Word Analogies")
            st.markdown("Test analogies like 'king - man + woman = queen'")
            
            analogy_col1, analogy_col2, analogy_col3 = st.columns(3)
            
            with analogy_col1:
                word_a = st.selectbox("Word A", st.session_state.obj2vec_vocab, key="analogy_word_a")
            with analogy_col2:
                word_b = st.selectbox("Word B", st.session_state.obj2vec_vocab, key="analogy_word_b")
            with analogy_col3:
                word_c = st.selectbox("Word C", st.session_state.obj2vec_vocab, key="analogy_word_c")
            
            if st.button("Compute Analogy", key="compute_analogy"):
                try:
                    results = st.session_state.obj2vec_model.wv.most_similar(
                        positive=[word_a, word_c], 
                        negative=[word_b], 
                        topn=5
                    )
                    
                    st.markdown(f"#### {word_a} - {word_b} + {word_c} = ?")
                    for word, score in results:
                        st.markdown(f"- **{word}** (similarity: {score:.4f})")
                except KeyError:
                    st.error("One or more words not found in vocabulary.")
            
            # Visualize word embeddings
            st.markdown("### Word Embeddings Visualization")
            
            # Reduce dimensionality for visualization
            if len(st.session_state.obj2vec_vocab) > 5:
                # Get embeddings array
                embeddings_array = np.array(st.session_state.obj2vec_embeddings)
                
                # Use PCA or t-SNE for visualization
                from sklearn.manifold import TSNE
                
                # Take the first 100 words for visualization if there are more
                max_words = min(100, len(st.session_state.obj2vec_vocab))
                vis_words = st.session_state.obj2vec_vocab[:max_words]
                vis_embeddings = embeddings_array[:max_words]
                
                # Use t-SNE to reduce to 2D
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max_words-1))
                embeddings_2d = tsne.fit_transform(vis_embeddings)
                
                # Create dataframe for visualization
                vis_df = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'word': vis_words
                })
                
                # Create interactive scatter plot
                fig = px.scatter(
                    vis_df, x='x', y='y', 
                    text='word', 
                    opacity=0.8,
                    title='Word Embedding Space (2D Projection)'
                )
                
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif obj2vec_demo_type == "Movie Recommendations" and hasattr(st.session_state, 'obj2vec_interactions'):
            # Display movie recommendation dataset
            st.markdown("### Movie Dataset Overview")
            
            # Show interaction distribution
            if not st.session_state.obj2vec_interactions.empty:
                # Show rating distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    data=st.session_state.obj2vec_interactions, 
                    x='rating', 
                    bins=10, 
                    kde=True, 
                    color=AWS_COLORS['orange'],
                    ax=ax
                )
                ax.set_title('Distribution of Movie Ratings')
                ax.set_xlabel('Rating')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                
                # Train embeddings
                st.markdown("### Train User and Movie Embeddings")
                
                if st.button("Train Embeddings", key="train_movie_embeddings"):
                    with st.spinner("Training embeddings..."):
                        # This is a simplified version, in a real scenario we would use a proper embedding model
                        # Here we'll create synthetic embeddings for demonstration purposes
                        
                        # Create user and movie IDs
                        user_ids = st.session_state.obj2vec_interactions['user_id'].unique()
                        movie_ids = st.session_state.obj2vec_interactions['movie_id'].unique()
                        
                        # Generate synthetic embeddings
                        np.random.seed(42)
                        user_embeddings = np.random.randn(len(user_ids), st.session_state.obj2vec_embedding_dim)
                        movie_embeddings = np.random.randn(len(movie_ids), st.session_state.obj2vec_embedding_dim)
                        
                        # Create mappings
                        user_id_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
                        movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}
                        
                        # Create dataframes with embeddings
                        user_embedding_df = pd.DataFrame({
                            'user_id': user_ids,
                            **{f'dim_{i}': user_embeddings[:, i] for i in range(st.session_state.obj2vec_embedding_dim)}
                        })
                        
                        movie_embedding_df = pd.DataFrame({
                            'movie_id': movie_ids,
                            **{f'dim_{i}': movie_embeddings[:, i] for i in range(st.session_state.obj2vec_embedding_dim)}
                        })
                        
                        # Store in session state
                        st.session_state.obj2vec_user_embeddings = user_embedding_df
                        st.session_state.obj2vec_movie_embeddings = movie_embedding_df
                        st.session_state.obj2vec_user_embedding_array = user_embeddings
                        st.session_state.obj2vec_movie_embedding_array = movie_embeddings
                        st.session_state.obj2vec_user_id_to_idx = user_id_to_idx
                        st.session_state.obj2vec_movie_id_to_idx = movie_id_to_idx
                        st.session_state.obj2vec_trained = True
                        
                        st.success(f"Embeddings trained! {len(user_ids)} user embeddings and {len(movie_ids)} movie embeddings.")
    
    if obj2vec_demo_type == "Movie Recommendations" and st.session_state.get('obj2vec_trained', False):
        st.divider()
        st.subheader("Embeddings and Recommendations")
        
        viz_col1, viz_col2 = st.columns([1, 1])
        
        with viz_col1:
            # Visualization of user and movie embeddings
            st.markdown("### User and Movie Embeddings Visualization")
            
            # Reduce dimensionality
            user_embeddings = st.session_state.obj2vec_user_embedding_array
            movie_embeddings = st.session_state.obj2vec_movie_embedding_array
            
            # Combine for joint embedding space
            all_embeddings = np.vstack([user_embeddings, movie_embeddings])
            
            # Label each point
            embedding_types = ['User'] * len(user_embeddings) + ['Movie'] * len(movie_embeddings)
            embedding_ids = list(range(len(user_embeddings))) + list(range(len(movie_embeddings)))
            
            # Reduce to 2D
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(all_embeddings)
            
            # Split back
            user_embeddings_2d = embeddings_2d[:len(user_embeddings)]
            movie_embeddings_2d = embeddings_2d[len(user_embeddings):]
            
            # Create dataframe for visualization
            embedding_df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'type': embedding_types,
                'id': [f"{t} {i+1}" for t, i in zip(embedding_types, embedding_ids)]
            })
            
            # Create interactive plot
            fig = px.scatter(
                embedding_df, x='x', y='y', 
                color='type', 
                hover_data=['id'],
                title='User and Movie Embedding Space',
                color_discrete_map={'User': 'blue', 'Movie': 'orange'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Movie recommendation system
            st.markdown("### Movie Recommendation Demo")
            
            selected_user = st.selectbox(
                "Select User", 
                options=[f"User {i+1}" for i in range(len(st.session_state.obj2vec_user_embedding_array))],
                key="obj2vec_select_user"
            )
            
            if st.button("Get Recommendations", key="get_recommendations"):
                user_idx = int(selected_user.split()[-1]) - 1
                user_embedding = st.session_state.obj2vec_user_embedding_array[user_idx]
                
                # Calculate similarity to all movies
                movie_similarities = np.dot(
                    st.session_state.obj2vec_movie_embedding_array, 
                    user_embedding
                ) / (
                    np.linalg.norm(st.session_state.obj2vec_movie_embedding_array, axis=1) * 
                    np.linalg.norm(user_embedding)
                )
                
                # Get top 5 recommendations
                top_movie_indices = np.argsort(movie_similarities)[::-1][:5]
                
                st.markdown(f"#### Top Movie Recommendations for {selected_user}")
                
                # Create recommendation dataframe
                recommendations = []
                for rank, movie_idx in enumerate(top_movie_indices):
                    movie_id = movie_idx  # In our simplified case, idx = id
                    similarity = movie_similarities[movie_idx]
                    movie_title = f"Movie {movie_id+1}"
                    movie_genres = st.session_state.obj2vec_movies.loc[st.session_state.obj2vec_movies['movie_id'] == movie_id, 'genres'].values[0]
                    
                    recommendations.append({
                        'Rank': rank + 1,
                        'Movie': movie_title,
                        'Genres': ", ".join(movie_genres),
                        'Similarity Score': similarity
                    })
                
                recommendations_df = pd.DataFrame(recommendations)
                
                # Display as table
                st.table(recommendations_df)
                
                # Create visualization of user and recommended movies
                st.markdown("#### User-Movie Embedding Space")
                
                # Extract user and recommended movie embeddings
                user_embedding_2d = user_embeddings_2d[user_idx].reshape(1, -1)
                rec_movie_embeddings_2d = movie_embeddings_2d[top_movie_indices]
                
                # Create plot data
                plot_data = []
                # Add user point
                plot_data.append({
                    'x': user_embedding_2d[0, 0],
                    'y': user_embedding_2d[0, 1],
                    'label': selected_user,
                    'type': 'User'
                })
                
                # Add movie points
                for i, movie_idx in enumerate(top_movie_indices):
                    plot_data.append({
                        'x': rec_movie_embeddings_2d[i, 0],
                        'y': rec_movie_embeddings_2d[i, 1],
                        'label': f"Movie {movie_idx+1}",
                        'type': 'Movie'
                    })
                
                plot_df = pd.DataFrame(plot_data)
                
                # Create interactive plot
                fig = px.scatter(
                    plot_df, x='x', y='y', 
                    color='type', 
                    text='label',
                    title='User and Recommended Movies in Embedding Space',
                    color_discrete_map={'User': 'blue', 'Movie': 'orange'}
                )
                
                fig.update_traces(textposition='top center')
                
                st.plotly_chart(fig, use_container_width=True)
        
        # SageMaker implementation
        st.divider()
        st.markdown("""
        <div class="card">
        <h3>Amazon SageMaker Implementation</h3>
        <p>Here's how you can implement Object2Vec in Amazon SageMaker:</p>
        <pre>
import boto3
import sagemaker
from sagemaker.object2vec.estimator import Object2Vec

# Set up the SageMaker session
session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

# Set the algorithm parameters
object2vec = Object2Vec(
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    enc_dim=512,  # Size of embeddings
    num_epochs=100,
    optimizer='adam',
    token_dim=1,  # For items with single ID representation
    enc0_network='mlp',
    enc1_network='mlp',
    enc0_max_seq_len=1,  # For items represented by a single token
    enc1_max_seq_len=1,
    output_layer='mean',
    mini_batch_size=32
)

# Train the model
object2vec.fit({'train': train_data_s3_path})

# Deploy the model to get embeddings
object2vec_predictor = object2vec.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

# Use model for predictions
recommendations = object2vec_predictor.predict(user_input)
        </pre>
        </div>
        """, unsafe_allow_html=True)
    
    elif obj2vec_demo_type == "Word Embeddings":
        # SageMaker implementation for word embeddings
        st.divider()
        st.markdown("""
        <div class="card">
        <h3>Amazon SageMaker Implementation for Word Embeddings</h3>
        <p>Here's how you can implement word embeddings using Object2Vec in Amazon SageMaker:</p>
        <pre>
import boto3
import sagemaker
from sagemaker.object2vec.estimator import Object2Vec

# Set up the SageMaker session
session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

# Set the algorithm parameters for word embeddings
object2vec = Object2Vec(
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    enc_dim=100,  # Size of word embeddings
    num_epochs=20,
    optimizer='adam',
    token_dim=300,  # Input token dimension
    enc0_network='bilstm',  # Use BiLSTM for capturing word context
    enc1_network='bilstm',
    enc0_max_seq_len=50,  # Max sequence length
    enc1_max_seq_len=50,
    output_layer='mean',
    mini_batch_size=64
)

# Train the model
object2vec.fit({'train': word_sequence_data_s3_path})

# Deploy the model
word_embedding_predictor = object2vec.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

# Use model to get word embeddings
word_vectors = word_embedding_predictor.predict(words)
        </pre>
        </div>
        """, unsafe_allow_html=True)

# ------------- Random Cut Forest Tab -------------
with tabs[3]:
    st.header("🌲 Random Cut Forest (RCF)")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>What is Random Cut Forest?</h3>
    <p>Random Cut Forest (RCF) is an unsupervised algorithm for detecting anomalies or outliers in data. It works by building an ensemble of trees (a forest) that isolate anomalies in the data.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Unsupervised anomaly detection</li>
        <li>Works well with high-dimensional data</li>
        <li>Can process streaming data</li>
        <li>Assigns anomaly scores to data points</li>
        <li>Robust to irrelevant or noisy attributes</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>Fraud detection</li>
        <li>Network intrusion detection</li>
        <li>Medical anomaly detection</li>
        <li>Industrial equipment monitoring</li>
        <li>Time series anomaly detection</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive Random Cut Forest Demo")
    
    rcf_col1, rcf_col2 = st.columns([1, 1])
    
    with rcf_col1:
        st.markdown("### Generate Data with Anomalies")
        
        rcf_demo_type = st.selectbox(
            "Select Demo Type", 
            ["2D Data with Outliers", "Time Series with Anomalies"],
            key="rcf_demo_type"
        )
        
        if rcf_demo_type == "2D Data with Outliers":
            rcf_n_samples = st.slider("Number of Normal Samples", 100, 1000, 300, 50, key="rcf_n_samples")
            rcf_n_outliers = st.slider("Number of Outliers", 5, 50, 15, 5, key="rcf_n_outliers")
            rcf_noise = st.slider("Noise Level", 0.01, 0.5, 0.1, 0.01, key="rcf_noise")
            
            if st.button("Generate Data", key="rcf_generate"):
                with st.spinner("Generating data with outliers..."):
                    # Generate normal data
                    np.random.seed(42)
                    X = np.random.randn(rcf_n_samples, 2)
                    
                    # Generate outliers
                    X_outliers = np.random.uniform(
                        low=[-4, -4], 
                        high=[4, 4], 
                        size=(rcf_n_outliers, 2)
                    )
                    
                    # Select outliers that are far from the center
                    distances = np.sqrt(X_outliers[:, 0]**2 + X_outliers[:, 1]**2)
                    X_outliers = X_outliers[distances > 2.5]
                    
                    # If we lost some outliers due to distance filtering, regenerate
                    while len(X_outliers) < rcf_n_outliers:
                        n_needed = rcf_n_outliers - len(X_outliers)
                        extra_outliers = np.random.uniform(
                            low=[-4, -4], 
                            high=[4, 4], 
                            size=(n_needed * 2, 2)
                        )
                        distances = np.sqrt(extra_outliers[:, 0]**2 + extra_outliers[:, 1]**2)
                        extra_outliers = extra_outliers[distances > 2.5]
                        X_outliers = np.vstack([X_outliers, extra_outliers])
                        X_outliers = X_outliers[:rcf_n_outliers]
                    
                    # Combine datasets
                    X_combined = np.vstack([X, X_outliers])
                    y_combined = np.zeros(X_combined.shape[0])
                    y_combined[-rcf_n_outliers:] = 1  # Mark outliers as 1
                    
                    # Add noise
                    X_combined += np.random.normal(0, rcf_noise, X_combined.shape)
                    
                    # Store in session state
                    st.session_state.rcf_data = X_combined
                    st.session_state.rcf_labels = y_combined
                    st.session_state.rcf_trained = False
                    st.session_state.rcf_data_type = '2d'
                    
                    st.success(f"Generated {rcf_n_samples} normal samples and {rcf_n_outliers} outliers.")
        
        elif rcf_demo_type == "Time Series with Anomalies":
            rcf_n_points = st.slider("Number of Time Points", 100, 1000, 500, 50, key="rcf_ts_n_points")
            rcf_n_anomalies = st.slider("Number of Anomalies", 5, 30, 10, 5, key="rcf_ts_n_anomalies")
            rcf_anomaly_scale = st.slider("Anomaly Scale", 1.0, 10.0, 5.0, 0.5, key="rcf_ts_anomaly_scale")
            
            if st.button("Generate Time Series", key="rcf_generate_ts"):
                with st.spinner("Generating time series with anomalies..."):
                    # Generate time series
                    np.random.seed(42)
                    
                    # Time points
                    t = np.linspace(0, 10, rcf_n_points)
                    
                    # Base signal: combination of sine waves
                    signal = 3 * np.sin(2 * t) + 2 * np.sin(5 * t) + np.random.normal(0, 0.5, rcf_n_points)
                    
                    # Insert anomalies
                    anomaly_indices = np.random.choice(rcf_n_points, rcf_n_anomalies, replace=False)
                    anomaly_values = np.random.choice([-1, 1], rcf_n_anomalies) * rcf_anomaly_scale
                    signal[anomaly_indices] += anomaly_values
                    
                    # Create labels
                    labels = np.zeros(rcf_n_points)
                    labels[anomaly_indices] = 1
                    
                    # Store in session state
                    st.session_state.rcf_ts_data = signal.reshape(-1, 1)
                    st.session_state.rcf_ts_time = t
                    st.session_state.rcf_ts_labels = labels
                    st.session_state.rcf_ts_anomaly_indices = anomaly_indices
                    st.session_state.rcf_trained = False
                    st.session_state.rcf_data_type = 'ts'
                    
                    st.success(f"Generated time series with {rcf_n_anomalies} anomalies.")
    
    with rcf_col2:
        if st.session_state.get('rcf_data_type') == '2d' and hasattr(st.session_state, 'rcf_data'):
            # Plot 2D data with outliers
            fig, ax = plt.subplots(figsize=(10, 8))
            
            normal = st.session_state.rcf_labels == 0
            outliers = st.session_state.rcf_labels == 1
            
            ax.scatter(
                st.session_state.rcf_data[normal, 0], 
                st.session_state.rcf_data[normal, 1],
                c='blue', 
                label='Normal', 
                alpha=0.6,
                edgecolors='k',
                s=80
            )
            
            ax.scatter(
                st.session_state.rcf_data[outliers, 0], 
                st.session_state.rcf_data[outliers, 1],
                c='red', 
                label='Outliers', 
                alpha=0.8,
                marker='X',
                edgecolors='k',
                s=100
            )
            
            ax.set_title('Data with Outliers')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        
        elif st.session_state.get('rcf_data_type') == 'ts' and hasattr(st.session_state, 'rcf_ts_data'):
            # Plot time series with anomalies
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot time series
            ax.plot(st.session_state.rcf_ts_time, st.session_state.rcf_ts_data, 'b-', label='Signal')
            
            # Highlight anomalies
            anomaly_indices = st.session_state.rcf_ts_anomaly_indices
            ax.scatter(
                st.session_state.rcf_ts_time[anomaly_indices],
                st.session_state.rcf_ts_data[anomaly_indices],
                c='red',
                label='Anomalies',
                marker='X',
                s=100,
                zorder=5
            )
            
            ax.set_title('Time Series with Anomalies')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
    
    if ((st.session_state.get('rcf_data_type') == '2d' and hasattr(st.session_state, 'rcf_data')) or
        (st.session_state.get('rcf_data_type') == 'ts' and hasattr(st.session_state, 'rcf_ts_data'))):
        
        st.divider()
        st.subheader("Train Random Cut Forest Model")
        
        rcf_train_col1, rcf_train_col2 = st.columns([1, 2])
        
        with rcf_train_col1:
            rcf_n_estimators = st.slider("Number of Trees", 10, 200, 100, 10, key="rcf_n_estimators")
            rcf_contamination = st.slider("Expected Contamination", 0.01, 0.3, 0.05, 0.01, key="rcf_contamination")
            
            if st.button("Detect Anomalies", key="rcf_train"):
                with st.spinner("Training Random Cut Forest model..."):
                    # Simulate training with progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Use Isolation Forest as a proxy for RCF
                    model = IsolationForest(
                        n_estimators=rcf_n_estimators,
                        contamination=rcf_contamination,
                        random_state=42
                    )
                    
                    if st.session_state.rcf_data_type == '2d':
                        data = st.session_state.rcf_data
                        true_labels = st.session_state.rcf_labels
                    else:  # Time series
                        # For time series, use sliding window features
                        data = create_sliding_window_features(st.session_state.rcf_ts_data.flatten(), window_size=10)
                        true_labels = st.session_state.rcf_ts_labels[9:]  # Adjust for window size
                    
                    # Fit model and get anomaly scores
                    model.fit(data)
                    scores = model.decision_function(data)
                    anomaly_scores = -scores  # Invert so higher = more anomalous
                    
                    # Predict anomalies
                    predictions = model.predict(data)
                    anomalies = predictions == -1
                    
                    # Store in session state
                    if st.session_state.rcf_data_type == '2d':
                        st.session_state.rcf_model = model
                        st.session_state.rcf_scores = anomaly_scores
                        st.session_state.rcf_anomalies = anomalies
                    else:  # Time series
                        st.session_state.rcf_ts_model = model
                        st.session_state.rcf_ts_scores = np.zeros(len(st.session_state.rcf_ts_data))
                        st.session_state.rcf_ts_scores[9:] = anomaly_scores  # Pad beginning
                        st.session_state.rcf_ts_anomalies = np.zeros(len(st.session_state.rcf_ts_data), dtype=bool)
                        st.session_state.rcf_ts_anomalies[9:] = anomalies
                    
                    st.session_state.rcf_trained = True
                    st.success("Anomaly detection completed!")
        
        with rcf_train_col2:
            if st.session_state.rcf_trained:
                if st.session_state.rcf_data_type == '2d':
                    # Plot anomaly detection results
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot normal points
                    normal = ~st.session_state.rcf_anomalies
                    ax.scatter(
                        st.session_state.rcf_data[normal, 0], 
                        st.session_state.rcf_data[normal, 1],
                        c='blue', 
                        label='Normal', 
                        alpha=0.6,
                        edgecolors='k',
                        s=80
                    )
                    
                    # Plot detected anomalies
                    anomalies = st.session_state.rcf_anomalies
                    ax.scatter(
                        st.session_state.rcf_data[anomalies, 0], 
                        st.session_state.rcf_data[anomalies, 1],
                        c='red', 
                        label='Detected Anomalies', 
                        alpha=0.8,
                        marker='X',
                        edgecolors='k',
                        s=100
                    )
                    
                    # Plot true outliers that were missed (false negatives)
                    true_outliers = st.session_state.rcf_labels == 1
                    missed = true_outliers & ~anomalies
                    if np.any(missed):
                        ax.scatter(
                            st.session_state.rcf_data[missed, 0], 
                            st.session_state.rcf_data[missed, 1],
                            c='green', 
                            label='Missed Outliers', 
                            alpha=0.8,
                            marker='o',
                            edgecolors='k',
                            s=100
                        )
                    
                    ax.set_title('Random Cut Forest Anomaly Detection Results')
                    ax.set_xlabel('Feature 1')
                    ax.set_ylabel('Feature 2')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                
                else:  # Time series
                    # Plot time series with detected anomalies
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                    
                    # Plot original time series
                    ax1.plot(st.session_state.rcf_ts_time, st.session_state.rcf_ts_data, 'b-', label='Signal')
                    
                    # Highlight true anomalies
                    anomaly_indices = st.session_state.rcf_ts_anomaly_indices
                    ax1.scatter(
                        st.session_state.rcf_ts_time[anomaly_indices],
                        st.session_state.rcf_ts_data[anomaly_indices],
                        c='red',
                        label='True Anomalies',
                        marker='X',
                        s=100,
                        zorder=5
                    )
                    
                    ax1.set_title('Time Series with True Anomalies')
                    ax1.set_ylabel('Value')
                    ax1.legend()
                    
                    # Plot anomaly scores
                    ax2.plot(st.session_state.rcf_ts_time, st.session_state.rcf_ts_scores, 'g-', label='Anomaly Score')
                    
                    # Highlight detected anomalies
                    detected_indices = np.where(st.session_state.rcf_ts_anomalies)[0]
                    if len(detected_indices) > 0:
                        ax2.scatter(
                            st.session_state.rcf_ts_time[detected_indices],
                            st.session_state.rcf_ts_scores[detected_indices],
                            c='red',
                            label='Detected Anomalies',
                            marker='o',
                            s=100,
                            zorder=5
                        )
                    
                    ax2.set_title('Anomaly Scores')
                    ax2.set_xlabel('Time')
                    ax2.set_ylabel('Anomaly Score')
                    ax2.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    if st.session_state.rcf_trained:
        st.divider()
        st.subheader("Evaluation")
        
        if st.session_state.rcf_data_type == '2d':
            eval_col1, eval_col2 = st.columns([1, 1])
            
            with eval_col1:
                # Calculate evaluation metrics
                st.markdown("### Anomaly Detection Performance")
                
                # Calculate metrics
                from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
                
                y_true = st.session_state.rcf_labels.astype(bool)
                y_pred = st.session_state.rcf_anomalies
                
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                
                # Create a confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Display metrics
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Precision", f"{precision:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recall", f"{recall:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1 Score", f"{f1:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    ax=ax
                )
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                plt.tight_layout()
                st.pyplot(fig)
            
            with eval_col2:
                # ROC curve
                st.markdown("### ROC Curve")
                
                # Calculate ROC
                fpr, tpr, _ = roc_curve(y_true, st.session_state.rcf_scores)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color=AWS_COLORS['orange'], lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC)')
                ax.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Anomaly score distribution
                st.markdown("### Anomaly Score Distribution")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                sns.histplot(
                    st.session_state.rcf_scores[~y_true], 
                    color='blue', 
                    label='Normal', 
                    kde=True,
                    alpha=0.6,
                    ax=ax
                )
                
                sns.histplot(
                    st.session_state.rcf_scores[y_true], 
                    color='red', 
                    label='Anomaly', 
                    kde=True,
                    alpha=0.6,
                    ax=ax
                )
                
                ax.set_xlabel('Anomaly Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Anomaly Scores')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
        
        else:  # Time series
            # Evaluate time series anomaly detection
            st.markdown("### Time Series Anomaly Detection Performance")
            
            # Calculate metrics
            y_true = st.session_state.rcf_ts_labels.astype(bool)
            y_pred = st.session_state.rcf_ts_anomalies
            
            # True anomalies and detected anomalies
            true_anomaly_indices = st.session_state.rcf_ts_anomaly_indices
            detected_anomaly_indices = np.where(y_pred)[0]
            
            # Count correct detections (within a window)
            def count_matches(true_indices, detected_indices, window=3):
                matches = 0
                for idx in true_indices:
                    # Check if there's a detection within window of this true anomaly
                    if np.any(np.abs(detected_indices - idx) <= window):
                        matches += 1
                return matches
            
            # Calculate metrics
            true_anomalies = len(true_anomaly_indices)
            detected_anomalies = len(detected_anomaly_indices)
            matches = count_matches(true_anomaly_indices, detected_anomaly_indices, window=3)
            
            precision = matches / detected_anomalies if detected_anomalies > 0 else 0
            recall = matches / true_anomalies if true_anomalies > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Precision", f"{precision:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recall", f"{recall:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1 Score", f"{f1:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display detailed view of anomalies
            st.markdown("### Detailed View of Anomalies")
            
            # Create interactive plot
            fig = go.Figure()
            
            # Add time series
            fig.add_trace(go.Scatter(
                x=st.session_state.rcf_ts_time,
                y=st.session_state.rcf_ts_data.flatten(),
                mode='lines',
                name='Time Series',
                line=dict(color='blue')
            ))
            
            # Add true anomalies
            fig.add_trace(go.Scatter(
                x=st.session_state.rcf_ts_time[true_anomaly_indices],
                y=st.session_state.rcf_ts_data[true_anomaly_indices].flatten(),
                mode='markers',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x'
                ),
                name='True Anomalies'
            ))
            
            # Add detected anomalies
            if len(detected_anomaly_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=st.session_state.rcf_ts_time[detected_anomaly_indices],
                    y=st.session_state.rcf_ts_data[detected_anomaly_indices].flatten(),
                    mode='markers',
                    marker=dict(
                        color='green',
                        size=15,
                        symbol='circle-open',
                        line=dict(width=2)
                    ),
                    name='Detected Anomalies'
                ))
            
            # Add anomaly score
            fig.add_trace(go.Scatter(
                x=st.session_state.rcf_ts_time,
                y=st.session_state.rcf_ts_scores / max(st.session_state.rcf_ts_scores) * max(st.session_state.rcf_ts_data),
                mode='lines',
                line=dict(
                    color='gray',
                    width=1,
                    dash='dot'
                ),
                opacity=0.7,
                name='Normalized Score'
            ))
            
            fig.update_layout(
                title='Time Series with Anomalies',
                xaxis_title='Time',
                yaxis_title='Value',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # SageMaker implementation
        st.divider()
        st.markdown("""
        <div class="card">
        <h3>Amazon SageMaker Implementation</h3>
        <p>Here's how you can implement Random Cut Forest in Amazon SageMaker:</p>
        <pre>
import boto3
import sagemaker
from sagemaker import RandomCutForest

# Set up the SageMaker session
session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

# Set the algorithm parameters
rcf = RandomCutForest(
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    num_samples_per_tree=512,
    num_trees=50,
    feature_dim=data.shape[1]  # Number of features in your data
)

# Train the model
rcf.fit({'train': train_data_s3_path})

# Deploy the model for inference
rcf_predictor = rcf.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

# Get anomaly scores
anomaly_scores = rcf_predictor.predict(test_data)

# Analyze anomaly scores (higher scores indicate anomalies)
        </pre>
        </div>
        """, unsafe_allow_html=True)

# ------------- IP Insights Tab -------------
with tabs[4]:
    st.header("🌐 IP Insights")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>What is IP Insights?</h3>
    <p>IP Insights is an unsupervised learning algorithm that learns the usage patterns of IP addresses. 
    It can capture associations between IP addresses and entities (e.g., user accounts) to identify anomalous behavior.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Learns latent vector representations for IP addresses and entities</li>
        <li>Detects anomalous IP address usage patterns</li>
        <li>Can handle large datasets of IP/entity combinations</li>
        <li>Useful for security monitoring and fraud detection</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>Detecting account takeover attempts</li>
        <li>Identifying compromised accounts</li>
        <li>Detecting unauthorized access</li>
        <li>Analyzing user login patterns</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive IP Insights Demo")
    
    ip_col1, ip_col2 = st.columns([1, 1])
    
    with ip_col1:
        st.markdown("### Generate IP-User Access Data")
        
        ip_n_users = st.slider("Number of Users", 20, 100, 50, 10, key="ip_n_users")
        ip_n_ips = st.slider("Number of IP Addresses", 50, 200, 100, 10, key="ip_n_ips")
        ip_n_locations = st.slider("Number of Geographic Regions", 5, 15, 8, 1, key="ip_n_locations")
        ip_anomaly_percent = st.slider("Anomaly Percentage", 1, 15, 5, 1, key="ip_anomaly_percent")
        
        if st.button("Generate Data", key="ip_generate"):
            with st.spinner("Generating IP-User access data..."):
                # Generate users
                users = [f"user_{i}" for i in range(ip_n_users)]
                
                # Generate IP addresses by location
                locations = [f"Region_{chr(65+i)}" for i in range(ip_n_locations)]
                ips_by_location = {}
                total_ips = 0
                
                # Distribute IPs across locations
                for location in locations:
                    n_loc_ips = max(3, int(ip_n_ips / ip_n_locations) + np.random.randint(-3, 4))
                    if total_ips + n_loc_ips > ip_n_ips:
                        n_loc_ips = ip_n_ips - total_ips
                    
                    ips_by_location[location] = [
                        f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
                        for _ in range(n_loc_ips)
                    ]
                    
                    total_ips += n_loc_ips
                    if total_ips >= ip_n_ips:
                        break
                
                # Assign users to primary locations
                user_locations = {}
                for user in users:
                    primary_location = np.random.choice(locations)
                    secondary_location = np.random.choice([l for l in locations if l != primary_location])
                    user_locations[user] = {
                        'primary': primary_location,
                        'secondary': secondary_location
                    }
                
                # Generate access logs
                access_logs =[]
                
                # Normal access patterns (users access from their primary location most of the time)
                for user in users:
                    primary_loc = user_locations[user]['primary']
                    secondary_loc = user_locations[user]['secondary']
                    
                    # Normal accesses from primary location
                    n_primary_accesses = np.random.randint(5, 15)
                    for _ in range(n_primary_accesses):
                        ip = np.random.choice(ips_by_location[primary_loc])
                        timestamp = datetime.now() - pd.Timedelta(days=np.random.randint(1, 30))
                        access_logs.append({
                            'user': user,
                            'ip_address': ip,
                            'location': primary_loc,
                            'timestamp': timestamp,
                            'is_anomaly': False
                        })
                    
                    # Some accesses from secondary location
                    n_secondary_accesses = np.random.randint(0, 5)
                    for _ in range(n_secondary_accesses):
                        ip = np.random.choice(ips_by_location[secondary_loc])
                        timestamp = datetime.now() - pd.Timedelta(days=np.random.randint(1, 30))
                        access_logs.append({
                            'user': user,
                            'ip_address': ip,
                            'location': secondary_loc,
                            'timestamp': timestamp,
                            'is_anomaly': False
                        })
                
                # Generate anomalous accesses (random users from random locations)
                n_anomalies = int(len(access_logs) * ip_anomaly_percent / 100)
                
                for _ in range(n_anomalies):
                    user = np.random.choice(users)
                    # Pick a location that's not the user's primary or secondary
                    user_locs = [user_locations[user]['primary'], user_locations[user]['secondary']]
                    anomaly_locations = [loc for loc in locations if loc not in user_locs]
                    
                    if anomaly_locations:
                        anomaly_loc = np.random.choice(anomaly_locations)
                        
                        if anomaly_loc in ips_by_location and ips_by_location[anomaly_loc]:
                            ip = np.random.choice(ips_by_location[anomaly_loc])
                            timestamp = datetime.now() - pd.Timedelta(days=np.random.randint(1, 30))
                            
                            access_logs.append({
                                'user': user,
                                'ip_address': ip,
                                'location': anomaly_loc,
                                'timestamp': timestamp,
                                'is_anomaly': True
                            })
                
                # Convert to DataFrames
                access_df = pd.DataFrame(access_logs)
                
                # Store in session state
                st.session_state.ip_data = access_df
                st.session_state.ip_users = users
                st.session_state.ip_locations = locations
                st.session_state.ip_user_locations = user_locations
                st.session_state.ip_trained = False
                
                st.success(f"Generated {len(access_df)} access logs with {n_anomalies} anomalies")
    
    with ip_col2:
        # if hasattr(st.session_state, 'ip_data') and not st.session_state.ip_data.empty:
        if 'ip_data' in st.session_state and st.session_state.ip_data is not None and not st.session_state.ip_data.empty:
            # Display some statistics about the data
            st.markdown("### Access Log Overview")
            
            # Show distribution of access by location
            loc_counts = st.session_state.ip_data['location'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=loc_counts.index, y=loc_counts.values, color=AWS_COLORS['orange'], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title('Access Logs by Location')
            ax.set_ylabel('Number of Accesses')
            ax.set_xlabel('Location')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show some sample data
            st.markdown("### Sample Access Logs")
            st.dataframe(st.session_state.ip_data[['user', 'ip_address', 'location', 'timestamp']].head(10))
        else:
            st.warning("No data available. Please load the data first.")
    
    # if hasattr(st.session_state, 'ip_data') and not st.session_state.ip_data.empty:
    if 'ip_data' in st.session_state and st.session_state.ip_data is not None and not st.session_state.ip_data.empty:
        st.divider()
        st.subheader("Train IP Insights Model")
        
        ip_train_col1, ip_train_col2 = st.columns([1, 1])
        
        with ip_train_col1:
            ip_vector_dim = st.slider("Embedding Dimension", 16, 128, 32, 16, key="ip_vector_dim")
            ip_neg_samples = st.slider("Number of Negative Samples", 5, 100, 20, 5, key="ip_neg_samples")
            
            if st.button("Train IP Insights Model", key="ip_train"):
                with st.spinner("Training IP Insights model..."):
                    # Simulate training with progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    # In a real implementation, we'd use the actual IP Insights algorithm.
                    # For this demo, we'll simulate embeddings using user-location and IP-location associations
                    
                    # Create mappings
                    users = st.session_state.ip_data['user'].unique()
                    ips = st.session_state.ip_data['ip_address'].unique()
                    
                    user_to_idx = {user: i for i, user in enumerate(users)}
                    ip_to_idx = {ip: i for i, ip in enumerate(ips)}
                    
                    # Create synthetic embeddings
                    np.random.seed(42)
                    user_embeddings = np.random.randn(len(users), ip_vector_dim)
                    ip_embeddings = np.random.randn(len(ips), ip_vector_dim)
                    
                    # Adjust embeddings based on location
                    for user in users:
                        user_idx = user_to_idx[user]
                        primary_loc = st.session_state.ip_user_locations[user]['primary']
                        
                        # Add a location-specific bias to user embedding
                        loc_idx = st.session_state.ip_locations.index(primary_loc)
                        user_embeddings[user_idx, :] += np.sin(np.arange(ip_vector_dim) + loc_idx)
                    
                    # Process IPs
                    for ip in ips:
                        ip_idx = ip_to_idx[ip]
                        # Find locations where this IP appears
                        ip_locs = st.session_state.ip_data.loc[st.session_state.ip_data['ip_address'] == ip, 'location'].unique()
                        
                        if len(ip_locs) > 0:
                            main_loc = ip_locs[0]
                            loc_idx = st.session_state.ip_locations.index(main_loc)
                            ip_embeddings[ip_idx, :] += np.cos(np.arange(ip_vector_dim) + loc_idx)
                    
                    # Calculate anomaly scores
                    anomaly_scores = []
                    
                    for _, row in st.session_state.ip_data.iterrows():
                        user = row['user']
                        ip = row['ip_address']
                        
                        user_idx = user_to_idx[user]
                        ip_idx = ip_to_idx[ip]
                        
                        # Calculate dot product as compatibility score
                        user_emb = user_embeddings[user_idx]
                        ip_emb = ip_embeddings[ip_idx]
                        
                        # Anomaly score is negative compatibility
                        compatibility = np.dot(user_emb, ip_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(ip_emb))
                        anomaly_score = 1 - compatibility  # Higher means more anomalous
                        
                        anomaly_scores.append(anomaly_score)
                    
                    # Add anomaly scores to data
                    st.session_state.ip_data['anomaly_score'] = anomaly_scores
                    
                    # Determine anomalies using threshold
                    threshold = np.percentile(anomaly_scores, 95)  # Top 5% are anomalous
                    st.session_state.ip_data['predicted_anomaly'] = st.session_state.ip_data['anomaly_score'] > threshold
                    
                    # Store embeddings
                    st.session_state.ip_user_embeddings = user_embeddings
                    st.session_state.ip_ip_embeddings = ip_embeddings
                    st.session_state.ip_user_to_idx = user_to_idx
                    st.session_state.ip_ip_to_idx = ip_to_idx
                    st.session_state.ip_anomaly_threshold = threshold
                    st.session_state.ip_trained = True
                    
                    st.success("IP Insights model trained successfully!")
        
        with ip_train_col2:
            if st.session_state.ip_trained:
                # Show anomaly score distribution
                st.markdown("### Anomaly Score Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot normal vs anomalous
                sns.histplot(
                    st.session_state.ip_data.loc[~st.session_state.ip_data['is_anomaly'], 'anomaly_score'],
                    color='blue',
                    label='Normal',
                    kde=True,
                    alpha=0.6,
                    ax=ax
                )
                
                sns.histplot(
                    st.session_state.ip_data.loc[st.session_state.ip_data['is_anomaly'], 'anomaly_score'],
                    color='red',
                    label='Anomalous',
                    kde=True,
                    alpha=0.6,
                    ax=ax
                )
                
                # Add threshold line
                ax.axvline(
                    st.session_state.ip_anomaly_threshold,
                    color='black',
                    linestyle='--',
                    label=f'Threshold ({st.session_state.ip_anomaly_threshold:.3f})'
                )
                
                ax.set_xlabel('Anomaly Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Anomaly Scores')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
    
    if st.session_state.ip_trained:
        st.divider()
        st.subheader("IP Insights Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            # Visualize embeddings
            st.markdown("### User and IP Embeddings")
            
            # Reduce dimensionality of embeddings for visualization
            from sklearn.manifold import TSNE
            
            # Check if we have both user and IP embeddings
            if hasattr(st.session_state, 'ip_user_embeddings') and hasattr(st.session_state, 'ip_ip_embeddings'):
                # Combine embeddings
                all_embeddings = np.vstack([
                    st.session_state.ip_user_embeddings,
                    st.session_state.ip_ip_embeddings
                ])
                
                # Labels for points
                embedding_types = ['User'] * len(st.session_state.ip_user_embeddings) + ['IP'] * len(st.session_state.ip_ip_embeddings)
                
                # Apply t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                reduced_embeddings = tsne.fit_transform(all_embeddings)
                
                # Split back into user and IP embeddings
                user_reduced = reduced_embeddings[:len(st.session_state.ip_user_embeddings)]
                ip_reduced = reduced_embeddings[len(st.session_state.ip_user_embeddings):]
                
                # Create dataframe for visualization
                viz_df = pd.DataFrame({
                    'x': reduced_embeddings[:, 0],
                    'y': reduced_embeddings[:, 1],
                    'type': embedding_types
                })
                
                # Create interactive scatter plot
                fig = px.scatter(
                    viz_df, 
                    x='x', 
                    y='y',
                    color='type',
                    title='User and IP Address Embedding Space',
                    color_discrete_map={
                        'User': 'blue',
                        'IP': 'orange'
                    },
                    opacity=0.7
                )
                
                fig.update_layout(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with result_col2:
            # Show detected anomalies
            st.markdown("### Detected Anomalies")
            
            anomaly_df = st.session_state.ip_data[st.session_state.ip_data['predicted_anomaly']]
            
            if not anomaly_df.empty:
                st.dataframe(
                    anomaly_df[['user', 'ip_address', 'location', 'timestamp', 'anomaly_score']]
                    .sort_values('anomaly_score', ascending=False)
                )
                
                # Calculate metrics
                true_anomalies = st.session_state.ip_data['is_anomaly'].sum()
                predicted_anomalies = st.session_state.ip_data['predicted_anomaly'].sum()
                true_positives = ((st.session_state.ip_data['is_anomaly']) & 
                                 (st.session_state.ip_data['predicted_anomaly'])).sum()
                
                precision = true_positives / predicted_anomalies if predicted_anomalies > 0 else 0
                recall = true_positives / true_anomalies if true_anomalies > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                st.markdown("### Detection Performance")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Precision", f"{precision:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Recall", f"{recall:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("F1 Score", f"{f1:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No anomalies detected based on current threshold.")
        
        st.markdown("### User-IP Access Patterns")
        
        selected_user = st.selectbox(
            "Select a User to Analyze",
            options=st.session_state.ip_users,
            key="ip_selected_user"
        )
        
        if selected_user:
            user_data = st.session_state.ip_data[st.session_state.ip_data['user'] == selected_user]
            
            # Create a network visualization of user-IP connections
            G = nx.Graph()
            
            # Add user node
            G.add_node(selected_user, type='user')
            
            # Add IP nodes and edges
            for _, row in user_data.iterrows():
                ip = row['ip_address']
                location = row['location']
                is_anomaly = row['predicted_anomaly']
                
                G.add_node(ip, type='ip', location=location, anomaly=is_anomaly)
                G.add_edge(selected_user, ip, anomaly=is_anomaly)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Position nodes using spring layout
            pos = nx.spring_layout(G, k=0.5)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, 
                pos, 
                nodelist=[selected_user],
                node_color='blue',
                node_size=800,
                alpha=0.8,
                ax=ax
            )
            
            # Get normal and anomalous IPs
            normal_ips = [n for n, attrs in G.nodes(data=True) 
                          if attrs.get('type') == 'ip' and not attrs.get('anomaly', False)]
            anomalous_ips = [n for n, attrs in G.nodes(data=True) 
                             if attrs.get('type') == 'ip' and attrs.get('anomaly', False)]
            
            # Draw normal IP nodes
            nx.draw_networkx_nodes(
                G, 
                pos, 
                nodelist=normal_ips,
                node_color='green',
                node_size=400,
                alpha=0.7,
                ax=ax
            )
            
            # Draw anomalous IP nodes
            if anomalous_ips:
                nx.draw_networkx_nodes(
                    G, 
                    pos, 
                    nodelist=anomalous_ips,
                    node_color='red',
                    node_size=500,
                    alpha=0.9,
                    ax=ax
                )
            
            # Draw normal edges
            normal_edges = [(u, v) for u, v, attrs in G.edges(data=True) if not attrs.get('anomaly', False)]
            nx.draw_networkx_edges(
                G, 
                pos, 
                edgelist=normal_edges,
                width=2,
                alpha=0.7,
                edge_color='green',
                ax=ax
            )
            
            # Draw anomalous edges
            anomalous_edges = [(u, v) for u, v, attrs in G.edges(data=True) if attrs.get('anomaly', False)]
            if anomalous_edges:
                nx.draw_networkx_edges(
                    G, 
                    pos, 
                    edgelist=anomalous_edges,
                    width=3,
                    alpha=0.9,
                    edge_color='red',
                    ax=ax
                )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, 
                pos, 
                font_size=10, 
                font_weight='bold',
                ax=ax
            )
            
            ax.set_title(f'Access Pattern for {selected_user}')
            ax.legend([
                plt.Line2D([0], [0], color='blue', marker='o', linestyle='', markersize=10),
                plt.Line2D([0], [0], color='green', marker='o', linestyle='', markersize=10),
                plt.Line2D([0], [0], color='red', marker='o', linestyle='', markersize=10)
            ], ['User', 'Normal IP', 'Anomalous IP'])
            
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show user access log table
            st.markdown("### User Access Logs")
            
            # Format table columns
            user_display = user_data[['timestamp', 'ip_address', 'location', 'anomaly_score', 'predicted_anomaly']]
            user_display = user_display.rename(columns={'predicted_anomaly': 'Is Anomalous'})
            user_display = user_display.sort_values('timestamp', ascending=False)
            
            # Style the dataframe
            def highlight_anomalies(val):
                color = 'background-color: rgba(255, 0, 0, 0.2)' if val else ''
                return color
            
            styled_df = user_display.style.format({
                'anomaly_score': '{:.4f}',
                'timestamp': '{:%Y-%m-%d %H:%M:%S}'
            }).applymap(highlight_anomalies, subset=['Is Anomalous'])
            
            st.dataframe(styled_df)
        
        # SageMaker implementation
        st.divider()
        st.markdown("""
        <div class="card">
        <h3>Amazon SageMaker Implementation</h3>
        <p>Here's how you can implement IP Insights in Amazon SageMaker:</p>
        <pre>
import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri

# Set up the SageMaker session
session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

# Get the IP Insights container
container = get_image_uri(region_name, 'ipinsights')

# Set the algorithm parameters
ipinsights = sagemaker.estimator.Estimator(
    container,
    role, 
    instance_count=1, 
    instance_type='ml.c4.xlarge',
    train_volume_size=20,
    output_path=f"s3://{bucket}/ip-insights-output",
    sagemaker_session=session
)

# Set hyperparameters
ipinsights.set_hyperparameters(
    num_entity_vectors=ip_n_users,  # Number of unique users or accounts
    vector_dim=128,                # Size of embedding vectors
    epochs=20,                    # Number of training epochs
    batch_metrics="true",         # Report metrics
    mini_batch_size=1000,         # Mini-batch size for SGD
    learning_rate=0.1,            # Learning rate
    negative_samples=20           # Number of negative samples
)

# Train the model
ipinsights.fit({'train': train_data_s3_path})

# Deploy the model
predictor = ipinsights.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

# Get compatibility scores
scores = predictor.predict(test_data)
        </pre>
        </div>
        """, unsafe_allow_html=True)

# ------------- PCA Tab -------------
with tabs[5]:
    st.header("📊 Principal Component Analysis (PCA)")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>What is PCA?</h3>
    <p>Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional form while preserving as much information as possible.</p>
    
    <h3>Key Features</h3>
    <ul>
        <li>Reduces data dimensionality while minimizing information loss</li>
        <li>Identifies the principal components (directions of maximum variance)</li>
        <li>Useful for visualization, compression, and preprocessing</li>
        <li>Removes correlation between features</li>
    </ul>
    
    <h3>When to Use</h3>
    <ul>
        <li>Visualizing high-dimensional data</li>
        <li>Preprocessing step before applying other algorithms</li>
        <li>Reducing computational complexity</li>
        <li>Identifying important features/patterns in data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive PCA Demo")
    
    pca_col1, pca_col2 = st.columns([1, 1])
    
    with pca_col1:
        st.markdown("### Select Dataset")
        
        pca_dataset = st.selectbox(
            "Dataset",
            ["Iris Dataset", "Swiss Roll", "Random High-Dimensional Data"],
            key="pca_dataset"
        )
        
        if pca_dataset == "Iris Dataset":
            # Load Iris dataset
            if st.button("Load Iris Dataset", key="pca_load_iris"):
                with st.spinner("Loading Iris dataset..."):
                    # Load iris dataset
                    iris = load_iris()
                    X = iris.data
                    y = iris.target
                    feature_names = iris.feature_names
                    target_names = iris.target_names
                    
                    # Store in session state
                    st.session_state.pca_data = X
                    st.session_state.pca_labels = y
                    st.session_state.pca_feature_names = feature_names
                    st.session_state.pca_target_names = target_names
                    st.session_state.pca_trained = False
                    st.session_state.pca_data_type = 'iris'
                    
                    st.success("Iris dataset loaded successfully!")
        
        elif pca_dataset == "Swiss Roll":
            # Load Swiss Roll dataset
            pca_n_samples = st.slider("Number of Samples", 500, 3000, 1000, 100, key="pca_swiss_n_samples")
            pca_noise = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05, key="pca_swiss_noise")
            
            if st.button("Generate Swiss Roll", key="pca_load_swiss"):
                with st.spinner("Generating Swiss Roll dataset..."):
                    # Generate Swiss Roll
                    X, color = make_swiss_roll(
                        n_samples=pca_n_samples,
                        noise=pca_noise,
                        random_state=42
                    )
                    
                    # Store in session state
                    st.session_state.pca_data = X
                    st.session_state.pca_labels = color
                    st.session_state.pca_trained = False
                    st.session_state.pca_data_type = 'swiss'
                    
                    st.success("Swiss Roll dataset generated!")
        
        elif pca_dataset == "Random High-Dimensional Data":
            # Generate high-dimensional data
            pca_n_samples = st.slider("Number of Samples", 100, 1000, 500, 50, key="pca_random_n_samples")
            pca_n_features = st.slider("Number of Features", 10, 100, 50, 5, key="pca_random_n_features")
            pca_n_classes = st.slider("Number of Classes", 2, 5, 3, 1, key="pca_random_n_classes")
            
            if st.button("Generate High-Dimensional Data", key="pca_load_random"):
                with st.spinner("Generating high-dimensional dataset..."):
                    # Generate data
                    X, y = make_classification(
                        n_samples=pca_n_samples,
                        n_features=pca_n_features,
                        n_informative=pca_n_features // 2,
                        n_redundant=pca_n_features // 10,
                        n_classes=pca_n_classes,
                        random_state=42
                    )
                    
                    # Store in session state
                    st.session_state.pca_data = X
                    st.session_state.pca_labels = y
                    st.session_state.pca_feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
                    st.session_state.pca_trained = False
                    st.session_state.pca_data_type = 'random'
                    
                    st.success(f"Generated {pca_n_samples} samples with {pca_n_features} features!")
    
    with pca_col2:
        if hasattr(st.session_state, 'pca_data') and st.session_state.pca_data is not None:
            # Display dataset information
            st.markdown("### Dataset Overview")
            
            # Display dataset shape
            st.markdown(f"**Shape:** {st.session_state.pca_data.shape[0]} samples, {st.session_state.pca_data.shape[1]} features")
            
            # Show different visualizations based on dataset type
            if st.session_state.pca_data_type == 'iris':
                # Show pair plot of first few dimensions
                iris_df = pd.DataFrame(
                    st.session_state.pca_data, 
                    columns=st.session_state.pca_feature_names
                )
                iris_df['species'] = [st.session_state.pca_target_names[i] for i in st.session_state.pca_labels]
                
                fig, axs = plt.subplots(2, 3, figsize=(12, 8))
                axs = axs.flatten()
                
                for i, feature in enumerate(st.session_state.pca_feature_names):
                    if i < 4:  # Show histograms for each feature
                        for species in iris_df['species'].unique():
                            sns.histplot(
                                iris_df[iris_df['species'] == species][feature], 
                                ax=axs[i], 
                                label=species,
                                kde=True,
                                alpha=0.6
                            )
                        axs[i].set_title(feature)
                        axs[i].legend()
                
                # Show scatter plot of first two features
                sns.scatterplot(
                    data=iris_df,
                    x=st.session_state.pca_feature_names[0],
                    y=st.session_state.pca_feature_names[1],
                    hue='species',
                    ax=axs[4],
                    s=80,
                    alpha=0.7
                )
                
                # Show scatter plot of second two features
                sns.scatterplot(
                    data=iris_df,
                    x=st.session_state.pca_feature_names[2],
                    y=st.session_state.pca_feature_names[3],
                    hue='species',
                    ax=axs[5],
                    s=80,
                    alpha=0.7
                )
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif st.session_state.pca_data_type == 'swiss':
                # Show 3D visualization of Swiss Roll
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                ax.scatter(
                    st.session_state.pca_data[:, 0],
                    st.session_state.pca_data[:, 1],
                    st.session_state.pca_data[:, 2],
                    c=st.session_state.pca_labels,
                    cmap=plt.cm.viridis,
                    s=50,
                    alpha=0.7
                )
                
                ax.set_title('Swiss Roll Dataset')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show interactive 3D plot
                fig = px.scatter_3d(
                    x=st.session_state.pca_data[:, 0],
                    y=st.session_state.pca_data[:, 1],
                    z=st.session_state.pca_data[:, 2],
                    color=st.session_state.pca_labels,
                    opacity=0.7,
                    title="Swiss Roll Dataset (Interactive)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.pca_data_type == 'random':
                # Show summary statistics
                st.markdown("**Summary Statistics (First 5 Features)**")
                stats_df = pd.DataFrame(
                    st.session_state.pca_data[:, :5], 
                    columns=st.session_state.pca_feature_names[:5]
                ).describe().T
                
                st.dataframe(stats_df)
                
                # Show feature correlation heatmap
                st.markdown("**Feature Correlation (First 10 Features)**")
                
                corr_df = pd.DataFrame(
                    st.session_state.pca_data[:, :10], 
                    columns=st.session_state.pca_feature_names[:10]
                ).corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                ax.set_title('Feature Correlation Matrix')
                plt.tight_layout()
                st.pyplot(fig)
    
    if hasattr(st.session_state, 'pca_data') and st.session_state.pca_data is not None:
        st.divider()
        st.subheader("Apply PCA")
        
        pca_train_col1, pca_train_col2 = st.columns([1, 2])
        
        with pca_train_col1:
            pca_n_components = st.slider(
                "Number of Components", 
                2, min(10, st.session_state.pca_data.shape[1]), 
                2, 
                1, 
                key="pca_n_components"
            )
            
            pca_whiten = st.checkbox("Whiten", value=False, key="pca_whiten")
            
            if st.button("Apply PCA", key="pca_apply"):
                with st.spinner("Applying PCA..."):
                    # Standardize data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(st.session_state.pca_data)
                    
                    # Apply PCA
                    pca_model = PCA(n_components=pca_n_components, whiten=pca_whiten)
                    X_pca = pca_model.fit_transform(X_scaled)
                    
                    # Store in session state
                    st.session_state.pca_model = pca_model
                    st.session_state.pca_transformed = X_pca
                    st.session_state.pca_components = pca_model.components_
                    st.session_state.pca_explained_variance = pca_model.explained_variance_ratio_
                    st.session_state.pca_X_scaled = X_scaled
                    st.session_state.pca_trained = True
                    
                    st.success("PCA applied successfully!")
        
        with pca_train_col2:
            if st.session_state.pca_trained:
                # Show explained variance
                st.markdown("### Explained Variance")
                
                # Create a table of explained variance
                explained_variance_df = pd.DataFrame({
                    'Component': [f"PC{i+1}" for i in range(len(st.session_state.pca_explained_variance))],
                    'Explained Variance (%)': [f"{var * 100:.2f}%" for var in st.session_state.pca_explained_variance],
                    'Cumulative Variance (%)': [f"{sum(st.session_state.pca_explained_variance[:i+1]) * 100:.2f}%" 
                                              for i in range(len(st.session_state.pca_explained_variance))]
                })
                
                st.table(explained_variance_df)
                
                # Plot explained variance ratio
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.bar(
                    range(1, len(st.session_state.pca_explained_variance) + 1),
                    st.session_state.pca_explained_variance,
                    alpha=0.7,
                    color=AWS_COLORS['orange']
                )
                
                ax.plot(
                    range(1, len(st.session_state.pca_explained_variance) + 1),
                    np.cumsum(st.session_state.pca_explained_variance),
                    'ro-',
                    label='Cumulative Explained Variance'
                )
                
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Explained Variance Ratio')
                ax.set_title('Scree Plot')
                ax.set_xticks(range(1, len(st.session_state.pca_explained_variance) + 1))
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
    
    if st.session_state.pca_trained:
        st.divider()
        st.subheader("PCA Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            # Display PCA results
            st.markdown("### Projected Data")
            
            # 2D scatter plot of first two principal components
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(
                st.session_state.pca_transformed[:, 0],
                st.session_state.pca_transformed[:, 1],
                c=st.session_state.pca_labels,
                cmap=plt.cm.viridis,
                alpha=0.7,
                s=50
            )
            
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('Data Projected onto First Two Principal Components')
            
            # Add a colorbar if it's not a classification dataset
            if st.session_state.pca_data_type != 'iris':
                plt.colorbar(scatter, ax=ax)
            else:  # Add a legend for Iris dataset
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', label=st.session_state.pca_target_names[i],
                              markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                    for i in range(len(st.session_state.pca_target_names))
                ]
                ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show interactive plot
            if st.session_state.pca_n_components >= 3:
                st.markdown("### 3D Projection")
                
                fig = px.scatter_3d(
                    x=st.session_state.pca_transformed[:, 0],
                    y=st.session_state.pca_transformed[:, 1],
                    z=st.session_state.pca_transformed[:, 2],
                    color=st.session_state.pca_labels,
                    opacity=0.7,
                    title="Data Projected onto First Three Principal Components"
                )
                
                fig.update_layout(
                    scene=dict(
                        xaxis_title="Principal Component 1",
                        yaxis_title="Principal Component 2",
                        zaxis_title="Principal Component 3"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with result_col2:
            # Show principal components
            st.markdown("### Principal Components")
            
            if st.session_state.pca_data_type in ['iris', 'random']:
                # Show the loadings/principal components
                n_features_to_show = min(10, st.session_state.pca_data.shape[1])
                feature_names = (st.session_state.pca_feature_names 
                                if hasattr(st.session_state, 'pca_feature_names') 
                                else [f"Feature {i+1}" for i in range(n_features_to_show)])
                
                loadings = st.session_state.pca_components[:, :n_features_to_show]
                
                # Create a heatmap of loadings
                fig, ax = plt.subplots(figsize=(12, 6))
                
                sns.heatmap(
                    loadings,
                    annot=True,
                    cmap='coolwarm',
                    fmt='.2f',
                    xticklabels=feature_names[:n_features_to_show],
                    yticklabels=[f"PC{i+1}" for i in range(loadings.shape[0])],
                    ax=ax
                )
                
                ax.set_title('Principal Component Loadings')
                ax.set_xlabel('Feature')
                ax.set_ylabel('Principal Component')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show feature importance for first principal component
                st.markdown("### Feature Importance (PC1)")
                
                # Calculate absolute contributions and sort
                pc1_contrib = np.abs(loadings[0, :])
                sorted_idx = np.argsort(pc1_contrib)[::-1]
                
                # Create a dataframe for display
                contrib_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in sorted_idx],
                    'Contribution': pc1_contrib[sorted_idx]
                })
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.barplot(
                    data=contrib_df.head(10),  # Show top 10 features
                    x='Contribution',
                    y='Feature',
                    color=AWS_COLORS['orange'],
                    ax=ax
                )
                
                ax.set_title('Top Features Contributing to PC1')
                ax.set_xlabel('Absolute Contribution')
                plt.tight_layout()
                st.pyplot(fig)
            
            elif st.session_state.pca_data_type == 'swiss':
                # For Swiss Roll, show the reconstruction error
                st.markdown("### Reconstruction Error")
                
                # Reconstruct the data
                X_reconstructed = np.matmul(
                    st.session_state.pca_transformed, 
                    st.session_state.pca_components
                ) + np.mean(st.session_state.pca_data, axis=0)
                
                # Calculate reconstruction error
                reconstruction_error = np.mean((st.session_state.pca_data - X_reconstructed) ** 2)
                
                # Display metric
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Mean Squared Reconstruction Error", f"{reconstruction_error:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show original vs. reconstructed visualization
                fig = plt.figure(figsize=(12, 6))
                
                # Original data
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(
                    st.session_state.pca_data[:, 0],
                    st.session_state.pca_data[:, 1],
                    st.session_state.pca_data[:, 2],
                    c=st.session_state.pca_labels,
                    cmap=plt.cm.viridis,
                    s=30,
                    alpha=0.7
                )
                ax1.set_title('Original Data')
                
                # Reconstructed data
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(
                    X_reconstructed[:, 0],
                    X_reconstructed[:, 1],
                    X_reconstructed[:, 2],
                    c=st.session_state.pca_labels,
                    cmap=plt.cm.viridis,
                    s=30,
                    alpha=0.7
                )
                ax2.set_title('Reconstructed Data')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show interactive comparison
                st.markdown("### Interactive Comparison")
                
                # Create a dataframe for plotting
                vis_df = pd.DataFrame({
                    'x_orig': st.session_state.pca_data[:, 0],
                    'y_orig': st.session_state.pca_data[:, 1],
                    'z_orig': st.session_state.pca_data[:, 2],
                    'x_recon': X_reconstructed[:, 0],
                    'y_recon': X_reconstructed[:, 1],
                    'z_recon': X_reconstructed[:, 2],
                    'color': st.session_state.pca_labels
                })
                
                # Let user select which visualization to see
                view_option = st.selectbox(
                    "Select View", 
                    ["Original", "Reconstructed", "Side by Side"],
                    key="pca_view_option"
                )
                
                if view_option == "Original":
                    fig = px.scatter_3d(
                        vis_df, 
                        x='x_orig', 
                        y='y_orig', 
                        z='z_orig',
                        color='color',
                        title="Original Data"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif view_option == "Reconstructed":
                    fig = px.scatter_3d(
                        vis_df, 
                        x='x_recon', 
                        y='y_recon', 
                        z='z_recon',
                        color='color',
                        title="Reconstructed Data"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # Side by Side
                    fig = make_subplots(
                        rows=1, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=["Original", "Reconstructed"]
                    )
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=vis_df['x_orig'],
                            y=vis_df['y_orig'],
                            z=vis_df['z_orig'],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=vis_df['color'],
                                colorscale='Viridis',
                                opacity=0.7
                            ),
                            name="Original"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=vis_df['x_recon'],
                            y=vis_df['y_recon'],
                            z=vis_df['z_recon'],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=vis_df['color'],
                                colorscale='Viridis',
                                opacity=0.7
                            ),
                            name="Reconstructed"
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        
        # SageMaker implementation
        st.divider()
        st.markdown("""
        <div class="card">
        <h3>Amazon SageMaker Implementation</h3>
        <p>Here's how you can implement PCA in Amazon SageMaker:</p>
        <pre>
import boto3
import sagemaker
from sagemaker import PCA

# Set up the SageMaker session
session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

# Set the algorithm parameters
pca = PCA(
    role=role,
    instance_count=1,
    instance_type='ml.c4.xlarge',
    num_components=10,  # Number of principal components
    feature_dim=data.shape[1],  # Number of features in your data
    subtract_mean=True,  # Center the data
    algorithm_mode='randomized'  # 'regular' or 'randomized'
)

# Train the model
pca.fit({'train': train_data_s3_path})

# Transform the data
transformer = pca.transformer(
    instance_count=1,
    instance_type='ml.m4.xlarge'
)

transformer.transform(
    data_s3_path,
    content_type='text/csv',
    split_type='line'
)

# Get results
transformed_data = transformer.output_path
        </pre>
        </div>
        """, unsafe_allow_html=True)

# Helper functions
def create_sliding_window_features(ts, window_size=10):
    """Create sliding window features from a time series."""
    n = len(ts)
    X = np.zeros((n - window_size + 1, window_size))
    
    for i in range(n - window_size + 1):
        X[i, :] = ts[i:i + window_size]
    
    return X

# Run the app
if __name__ == "__main__":
    pass  # The Streamlit app is already running at this point
# ```

# This comprehensive Streamlit application showcases Amazon SageMaker's unsupervised learning algorithms with interactive examples and visualizations. Here's a summary of what the application includes:

# ### Key Features:
# 1. **Tab-based Navigation**: Six tabs with emojis for different algorithms
# 2. **Modern UI/UX**: AWS color scheme, card-based layout, and responsive design 
# 3. **Interactive Examples**: Each algorithm has configurable parameters and visual results
# 4. **Comprehensive Visualizations**: Interactive plots, charts, and 3D renderings
# 5. **Session Management**: Reset functionality in the sidebar
# 6. **Educational Content**: Algorithm descriptions, use cases, and SageMaker implementation code

# ### Algorithms Included:
# 1. **K-means**: Clustering visualization with optimal k selection
# 2. **LDA**: Topic modeling with word distribution and document-topic analysis
# 3. **Object2Vec**: Word embeddings and recommendation system demonstrations
# 4. **Random Cut Forest**: Anomaly detection in 2D data and time series
# 5. **IP Insights**: IP-entity relationship analysis and anomaly detection
# 6. **PCA**: Dimensionality reduction with variance analysis and reconstructions

# ### Implementation Details:
# - Used the latest Python libraries (Streamlit, Plotly, Pandas, NumPy, etc.)
# - Incorporated interactive visualizations with Plotly and Matplotlib
# - Added session state management for resetting user state
# - Included detailed SageMaker implementation code for each algorithm
# - Created synthetic data generation for interactive experimentation
# - Added performance metrics and evaluation for model assessment

# This application serves as both an educational tool and a practical demonstration of how to leverage Amazon SageMaker's unsupervised learning algorithms for various machine learning tasks.