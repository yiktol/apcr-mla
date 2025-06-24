
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import uuid
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
import random
import os
from datetime import datetime
import altair as alt
import utils.common as common
import utils.authenticate as authenticate
# Configure page settings
st.set_page_config(
    page_title="The Curse of Dimensionality",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Session management functions
def initialize_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
        
    if 'dimension_slider' not in st.session_state:
        st.session_state.dimension_slider = 2
        
    if 'knn_k_value' not in st.session_state:
        st.session_state.knn_k_value = 5
        
    if 'distance_calculation_done' not in st.session_state:
        st.session_state.distance_calculation_done = False
        
    if 'distance_data' not in st.session_state:
        st.session_state.distance_data = {}
        
    if 'pca_dataset' not in st.session_state:
        st.session_state.pca_dataset = None
        
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = 42
        
    if 'show_tooltips' not in st.session_state:
        st.session_state.show_tooltips = True


# Helper function to display tooltip
def tooltip(text, tooltip_text):
    if st.session_state.show_tooltips:
        return f"{text} ℹ️"
    else:
        return text

# Visualization functions
def plot_curse_overview():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Parameters
    dimensions = np.arange(1, 101)
    sample_sizes = [100, 1000, 10000, 100000]
    coverage_percentages = {}
    
    for sample in sample_sizes:
        coverage = sample / (2**dimensions)
        coverage_percentages[sample] = np.minimum(coverage * 100, 100)  # Cap at 100%
    
    # Create plot
    for sample, coverage in coverage_percentages.items():
        ax.plot(dimensions, coverage, label=f'Sample size: {sample}', linewidth=2.5)
    
    ax.set_title('How Sample Coverage Decreases with More Dimensions', fontsize=16)
    ax.set_xlabel('Number of Dimensions (Features)', fontsize=14)
    ax.set_ylabel('How Much of the Space We Can Cover (%)', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    return fig

def generate_distance_concentration_data(n_dimensions, n_points=1000):
    if n_dimensions in st.session_state.distance_data:
        return st.session_state.distance_data[n_dimensions]
    
    # Generate random data points
    np.random.seed(42)
    data = np.random.random((n_points, n_dimensions))
    
    # Calculate pairwise distances
    distances = pdist(data)
    
    # Store in session state
    st.session_state.distance_data[n_dimensions] = {
        'distances': distances,
        'mean': np.mean(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances)
    }
    
    return st.session_state.distance_data[n_dimensions]

def plot_distance_distribution(n_dimensions):
    distance_data = generate_distance_concentration_data(n_dimensions)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram of distances
    sns.histplot(distance_data['distances'], kde=True, ax=ax)
    
    # Add mean and std lines
    mean = distance_data['mean']
    std = distance_data['std']
    ax.axvline(mean, color='r', linestyle='--', label=f'Average distance: {mean:.3f}')
    ax.axvline(mean - std, color='g', linestyle=':', label=f'Range of typical distances: {mean:.3f} ± {std:.3f}')
    ax.axvline(mean + std, color='g', linestyle=':')
    
    ax.set_title(f'How Far Apart Points Are in {n_dimensions}-Dimensional Space', fontsize=16)
    ax.set_xlabel('Distance Between Points', fontsize=14)
    ax.set_ylabel('How Many Point Pairs at This Distance', fontsize=14)
    ax.legend()
    
    return fig

def plot_distance_stats():
    dimensions = list(range(2, 101, 5))
    means = []
    stds = []
    min_distances = []
    max_distances = []
    
    # Calculate stats for different dimensions
    for dim in dimensions:
        stats = generate_distance_concentration_data(dim)
        means.append(stats['mean'])
        stds.append(stats['std'])
        min_distances.append(stats['min'])
        max_distances.append(stats['max'])
    
    # Calculate ratio of std/mean
    relative_stds = [std/mean for std, mean in zip(stds, means)]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean and standard deviation
    ax1.plot(dimensions, means, 'b-', label='Average distance')
    ax1.fill_between(dimensions, 
                    [m - s for m, s in zip(means, stds)], 
                    [m + s for m, s in zip(means, stds)], 
                    alpha=0.2, color='b',
                    label='Range of typical distances')
    ax1.set_title('Average Distance Increases with Dimensions', fontsize=16)
    ax1.set_xlabel('Number of Dimensions', fontsize=14)
    ax1.set_ylabel('Average Distance Between Points', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Relative standard deviation (std/mean) - shows distance concentration
    ax2.plot(dimensions, relative_stds, 'r-')
    ax2.set_title('Points Become Similarly Distant in Higher Dimensions', fontsize=16)
    ax2.set_xlabel('Number of Dimensions', fontsize=14)
    ax2.set_ylabel('How Much Distances Vary (Lower = More Similar)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_knn_accuracy_plot(max_dim=50):
    accuracies = []
    dimensions = list(range(2, max_dim + 1, 2))
    
    # Generate classification data for different dimensions
    for dim in dimensions:
        # Generate dataset with increasing dimensions
        X, y = make_classification(n_samples=1000, n_features=dim, 
                                  n_informative=min(dim, 10),  # Keep informative features limited
                                  n_redundant=0, 
                                  random_state=st.session_state.random_seed)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                           random_state=st.session_state.random_seed)
        
        # Train KNN classifier
        k = st.session_state.knn_k_value
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Evaluate accuracy
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dimensions, accuracies, 'o-', linewidth=2)
    ax.set_title(f'KNN Accuracy Decreases with More Dimensions (k={k})', fontsize=16)
    ax.set_xlabel('Number of Dimensions (Features)', fontsize=14)
    ax.set_ylabel('Accuracy (How Often Predictions Are Correct)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    return fig

def generate_pca_demonstration():
    if st.session_state.pca_dataset is None:
        # Generate high-dimensional dataset
        n_samples = 1000
        n_dimensions = 50
        n_informative = 5  # Only a few dimensions contain useful information
        
        X, y = make_classification(n_samples=n_samples, n_features=n_dimensions,
                                 n_informative=n_informative, n_redundant=n_dimensions-n_informative-10,
                                 n_classes=3, random_state=st.session_state.random_seed)
        
        st.session_state.pca_dataset = (X, y)
    else:
        X, y = st.session_state.pca_dataset
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Number of components for different thresholds
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
    components_needed = [np.where(cumulative_variance >= t)[0][0] + 1 for t in thresholds]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Explained variance by component
    ax1.bar(range(1, 21), explained_variance[:20], alpha=0.7)
    ax1.set_title('How Much Information Each Component Contains', fontsize=16)
    ax1.set_xlabel('Principal Component', fontsize=14)
    ax1.set_ylabel('Information Captured (%)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative explained variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', linewidth=2)
    ax2.set_title('Total Information Captured vs. Number of Components', fontsize=16)
    ax2.set_xlabel('Number of Components', fontsize=14)
    ax2.set_ylabel('Total Information Captured (%)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal lines for thresholds
    for t in thresholds:
        ax2.axhline(y=t, linestyle='--', alpha=0.5, color='red')
        ax2.text(45, t+0.01, f'{int(t*100)}%', color='red')
    
    plt.tight_layout()
    
    # Create scatter plot of first two PCA components
    fig2 = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y.astype(str),
                    title="Data Compressed to 2 Dimensions with PCA",
                    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
                    color_discrete_sequence=px.colors.qualitative.Bold)
    
    return fig, fig2, components_needed, thresholds

def generate_volume_plot():
    # Create data for volume comparisons
    radii = np.linspace(0.1, 1.0, 10)
    dimensions = np.arange(1, 11)
    
    volumes = np.zeros((len(radii), len(dimensions)))
    
    # Calculate volumes for unit balls with different radii and dimensions
    for i, r in enumerate(radii):
        for j, d in enumerate(dimensions):
            # Volume formula for d-dimensional ball: V = (π^(d/2) / Γ(d/2 + 1)) * r^d
            # We'll use a simplified formula proportional to r^d
            volumes[i, j] = r**d
    
    # Normalize volumes for each dimension
    for j in range(len(dimensions)):
        max_vol = volumes[-1, j]  # Volume at max radius
        volumes[:, j] = volumes[:, j] / max_vol
    
    # Create interactive 3D plot
    fig = go.Figure(data=[go.Surface(z=volumes, x=dimensions, y=radii)])
    
    fig.update_layout(
        title='How Volume Shifts Toward the Edges in Higher Dimensions',
        scene=dict(
            xaxis_title='Number of Dimensions',
            yaxis_title='Distance from Center (0 to 1)',
            zaxis_title='Relative Volume',
            xaxis=dict(nticks=10),
            yaxis=dict(nticks=10),
            zaxis=dict(nticks=10),
        ),
        width=800,
        height=600,
    )
    
    return fig

def create_simple_dimension_viz(dim):
    """Creates a simple visualization for 1D, 2D and 3D spaces"""
    if dim == 1:
        # Create 1D visualization - points on a line
        fig, ax = plt.subplots(figsize=(8, 2))
        points = np.linspace(0, 1, 10)
        ax.scatter(points, np.zeros_like(points), s=80, color='blue')
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(-0.1, 1.1)
        ax.set_title("1D Space: A Line")
        ax.set_xlabel("Position")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        return fig
    elif dim == 2:
        # Create 2D visualization - points on a plane
        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.random.random(100)
        y = np.random.random(100)
        ax.scatter(x, y, s=50, color='blue', alpha=0.7)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("2D Space: A Plane")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, alpha=0.3)
        return fig
    elif dim == 3:
        # Create 3D visualization - points in a cube
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        x = np.random.random(200)
        y = np.random.random(200)
        z = np.random.random(200)
        ax.scatter(x, y, z, s=50, color='blue', alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title("3D Space: A Cube")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        return fig
    else:
        return None

def main():
    # Initialize session state
    initialize_session()
    
    # Apply AWS styling
    common.apply_styles()
    
    # Sidebar content
    with st.sidebar:
        common.render_sidebar()
        
        # About section (collapsible)
        with st.expander("About this App", expanded=False):
            st.write("""
            This interactive app explains the "Curse of Dimensionality" - 
            an important concept in machine learning that shows why having lots of features
            can actually make learning harder.
            
            We use simple visualizations to make this complex topic easy to understand,
            even if you're new to machine learning!
            """)
        
    
    # Main content
    st.title("The Curse of Dimensionality in Machine Learning")
    st.markdown("""
    ### What you'll learn
    How having too many features (dimensions) can make machine learning harder, and what to do about it.
    
    This app is designed to be beginner-friendly. We'll explain everything step by step!
    """)
    
    # Create tabs for different sections
    tabs = st.tabs([
        "Introduction 📚",
        "Space & Sparsity 🌌",
        "Distance Effects 📏",
        "Impact on Algorithms 🤖",
        "Dimensionality Reduction ✂️",
        "Practical Solutions 🛠️"
    ])
    
    # Tab 1 - Introduction
    with tabs[0]:
        st.header("Understanding the Curse of Dimensionality")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### What is the Curse of Dimensionality?
            
            The "curse of dimensionality" refers to problems that happen when working with data that has many features (dimensions).
            
            **In simple terms:**
            > The more features you add to your data, the harder it becomes to find patterns, because the data gets spread out too thinly.
            
            **Think of it like this:**
            - Finding a person in a line (1D) is easy
            - Finding a person in a field (2D) is harder
            - Finding a person in a building (3D) is even harder
            - Finding a person in a 100-dimensional space? Nearly impossible!
            """)
            
            st.markdown("""
            ### Why should you care about this?
            
            Understanding the curse of dimensionality helps you:
            
            1. Know when you have **too many features** for your dataset
            2. Understand why some algorithms **struggle with high-dimensional data**
            3. Learn techniques to **reduce dimensions** effectively
            4. Build **better machine learning models** with the right number of features
            """)
            
            with st.expander("Glossary of Key Terms"):
                st.markdown("""
                - **Dimension**: A feature or attribute in your data (like height, weight, age)
                - **Sparsity**: When data points are spread far apart in your feature space
                - **Distance concentration**: When all points seem similarly distant in high dimensions
                - **Dimensionality reduction**: Techniques to decrease the number of features while preserving important information
                - **Feature selection**: Choosing only the most important features for your model
                """)
        
        with col2:
            st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*jDFQHxpKmNUGR6J7FngAmg.png", 
                    caption="High dimensions are different from what we can visualize")
            
            # Simple dimension visualization
            dim_choice = st.radio("Choose a dimension to visualize:", [1, 2, 3], horizontal=True)
            simple_viz = create_simple_dimension_viz(dim_choice)
            if simple_viz:
                st.pyplot(simple_viz)
                if dim_choice == 1:
                    st.caption("In 1D, we can easily find and organize points on a line")
                elif dim_choice == 2:
                    st.caption("In 2D, we can still visualize the entire space")
                else:
                    st.caption("In 3D, visualization gets harder, but we can still manage")
            
            st.info("Humans can visualize 1D, 2D, and 3D spaces, but machine learning often deals with hundreds or thousands of dimensions!")
            
            # Interactive element - simple visualization
            overview_fig = plot_curse_overview()
            st.pyplot(overview_fig)
            st.caption("This graph shows how quickly coverage drops as dimensions increase. With 10 dimensions, even 100,000 samples covers almost nothing!")
    
    # Tab 2 - Volume & Sparsity
    with tabs[1]:
        st.header("Volume Growth and Data Sparsity")
        
        st.markdown("""
        ### What Happens in Higher Dimensions?
        
        As we add more dimensions (features) to our data:
        1. The space gets **exponentially larger**
        2. Our data points become more **spread out**
        3. This makes it **harder to find patterns**
        
        Think of it like this: Finding patterns in high-dimensional data is like trying to find a few specific 
        coins scattered across an entire country, rather than just in your living room.
        """)
        
        # Simple illustration first
        st.subheader("Simple Illustration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            Imagine you want to cover:
            - A 1-meter line (1D) - you need 10 points spaced 10cm apart
            - A 1×1 meter square (2D) - you need 100 points in a 10×10 grid
            - A 1×1×1 meter cube (3D) - you need 1,000 points in a 10×10×10 grid
            
            Each time you add a dimension, you need **10 times more points** to maintain the same coverage!
            """)
        
        with col2:
            st.image("https://miro.medium.com/v2/resize:fit:640/format:webp/0*6wUMEAT8So1GKhVp", 
                    caption="As dimensions increase, you need exponentially more data points")
        
        # Interactive volume visualization with simpler explanation
        st.subheader("The Volume Problem in Higher Dimensions")
        st.markdown("""
        The interactive visualization below shows an important effect in higher dimensions:
        - In low dimensions, volume is mostly concentrated toward the center of a sphere
        - In higher dimensions, most of the volume shifts toward the outer edges
        - This means data points tend to be found near the boundaries, not the center
        
        This makes finding "nearby" points much harder!
        """)
        
        # Interactive volume visualization
        volume_fig = generate_volume_plot()
        st.plotly_chart(volume_fig)
        
        st.markdown("""
        **What this means in simple terms:** 
        
        In high dimensions, most points end up being far away from each other, near the edges of the space.
        This makes it hard for algorithms to determine which points are truly similar.
        """)
        
        # Sparsity example with simplified explanation
        st.subheader("The Data Coverage Problem")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### Why We Need More Data As Dimensions Increase
            
            Let's see how many samples we need to cover just 10% of our data space:
            
            * 1D space: 10 samples (like 10 points on a line)
            * 2D space: 100 samples (like a 10×10 grid)
            * 3D space: 1,000 samples (like a 10×10×10 cube)
            * 10D space: 10 billion samples!
            
            This is why machine learning with many features often requires huge amounts of data.
            """)
            
        with col2:
            # Create an interactive widget to show sampling in different dimensions
            dim_slider = st.slider("Number of dimensions:", 1, 10, 2)
            samples_needed = 10**dim_slider
            
            st.metric("Samples needed for 10% coverage:", f"{samples_needed:,}")
            
            # Visual representation of sample growth
            if dim_slider <= 4:
                grid_size = 10
                # Show grid visualization for dimensions 1-4
                if dim_slider == 1:
                    data = np.random.choice([0, 1], size=grid_size, p=[0.9, 0.1])
                    fig, ax = plt.subplots(figsize=(10, 1))
                    ax.scatter(range(grid_size), np.zeros(grid_size), c=['red' if x else 'blue' for x in data], s=100)
                    ax.set_ylim(-1, 1)
                    ax.set_title("1D: 10 samples to cover 10% of a line")
                    ax.axis('off')
                    st.pyplot(fig)
                elif dim_slider == 2:
                    data = np.random.choice([0, 1], size=(grid_size, grid_size), p=[0.99, 0.01])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(data, cmap='Blues')
                    ax.set_title("2D: 100 samples to cover 10% of a square")
                    ax.axis('off')
                    st.pyplot(fig)
                elif dim_slider == 3:
                    st.markdown("3D: Need 1,000 samples to cover 10% of a cube")
                    st.image("https://miro.medium.com/v2/resize:fit:1400/1*XEu9cHGq-dOOjQJfKTZ8jA.png", 
                            width=300, caption="3D sparse sampling illustration")
                else:
                    st.markdown("4D: Need 10,000 samples for 10% coverage")
            else:
                st.error(f"With {dim_slider} dimensions, we need {samples_needed:,} samples - that's too many to visualize!")
        
        # Real-world implications
        st.subheader("What This Means For Machine Learning")
        
        st.markdown("""
        ### Real-World Implications
        
        1. **Insufficient Data**: Most real-world datasets don't have enough samples for high-dimensional spaces
        
        2. **Overfitting Risk**: Models can find "patterns" in sparse data that don't actually exist
        
        3. **Need for Dimensionality Reduction**: We often need to reduce the number of features to make learning feasible
        
        4. **Feature Selection Importance**: Choosing only the most relevant features becomes critical
        
        > **In simple terms:** When you have too many features and not enough data, your machine learning model might "memorize" your training data instead of learning actual patterns.
        """)

    # Tab 3 - Distance Effects
    with tabs[2]:
        st.header("Distance Effects in High Dimensions")
        
        st.markdown("""
        ### The Strange Behavior of Distances
        
        In our everyday 3D world, we have good intuition about distances:
        - Some points are close to us
        - Some points are far away
        - Most points are somewhere in between
        
        But in high-dimensional spaces, something strange happens: **almost all points become similarly distant from each other**.
        """)
        
        # Real-world analogy first
        st.subheader("Real-World Analogy")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            Imagine you're comparing people based on different characteristics:
            
            **With 3 features:**
            - Height, weight, and age
            - Some people will be very similar to you
            - Others will be very different
            - Most will be somewhat different
            
            **With 100 features:**
            - Height, weight, age, hair length, foot size, arm length, income, words per minute typing...
            - Almost everyone will seem about equally different from you
            - It becomes hard to find your "nearest neighbors"
            
            This is called **distance concentration**, and it's a major problem for machine learning!
            """)
        
        with col2:
            st.image("https://i.sstatic.net/f1WOm.png", 
                    caption="In high dimensions, most points end up at similar distances from each other")
        
        # Interactive distance distribution exploration
        st.subheader("Explore How Distances Behave")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dimension = st.slider("Select number of dimensions:", 2, 100, 10, key="distance_dim_slider")
            
            st.markdown("""
            As dimensions increase:
            
            * The average distance between points increases
            * All distances become more similar to each other
            * Finding truly "close" points becomes nearly impossible
            
            This is why many algorithms that rely on finding similar data points struggle in high dimensions.
            """)
            
            if st.button("Calculate Distance Statistics", key="calc_dist_btn"):
                st.session_state.distance_calculation_done = True
                with st.spinner("Calculating distances between points..."):
                    # Pre-generate data for all dimensions up to the selected one
                    for d in range(2, dimension+1):
                        generate_distance_concentration_data(d)
                
                st.success("Distance statistics calculated!")
        
        with col2:
            if st.session_state.distance_calculation_done:
                dist_fig = plot_distance_distribution(dimension)
                st.pyplot(dist_fig)
                
                stats_fig = plot_distance_stats()
                st.pyplot(stats_fig)
                st.caption("Left: Average distance grows with dimensions. Right: Distances become more similar (the concentration effect)")
            else:
                st.info("Click 'Calculate Distance Statistics' to see how distances behave in higher dimensions")
        
        # Explain the implications with everyday examples
        st.subheader("Why This Matters")
        
        st.markdown("""
        ### Real-World Examples of Distance Concentration Problems
        
        **1. Recommendation Systems**
        - With many features, all products appear similarly different from what you've purchased
        - Makes it harder to find truly "similar" items to recommend
        
        **2. Image Recognition**
        - Images converted to pixels have thousands of dimensions
        - Without dimension reduction, similar images can appear distant
        
        **3. Medical Diagnosis**
        - Patient data might include hundreds of measurements
        - Finding similar cases becomes difficult with too many features
        
        The key insight is that **more data (features) can actually give you less information** if you don't handle dimensions properly!
        """)
    
    # Tab 4 - Impact on Algorithms
    with tabs[3]:
        st.header("How Algorithms Are Affected")
        
        st.markdown("""
        ### Machine Learning in High Dimensions
        
        Different algorithms handle high-dimensional data in different ways. Some are severely affected, while others are more resistant.
        
        Let's explore how common algorithms behave when dimensions increase:
        """)
        
        # Algorithm effects expander sections - with beginner-friendly explanations first
        with st.expander("K-Nearest Neighbors (KNN)", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                ### How KNN Works
                
                **K-Nearest Neighbors (KNN)** makes predictions by:
                1. Finding the K training examples closest to a new data point
                2. Taking a "vote" among these neighbors
                3. Assigning the majority class as the prediction
                
                **Example:** If you want to predict if someone will like a movie, KNN finds K people with similar tastes and checks what movies they liked.
                
                ### Why KNN Struggles in High Dimensions
                
                * When all points are similarly distant (distance concentration), finding "true neighbors" is like finding a needle in a haystack
                * Classification boundaries become less reliable
                * Accuracy often plummets
                """)
                
                # Interactive controls
                st.subheader("Try It Yourself")
                k_value = st.slider("Number of neighbors (k):", 1, 20, 5, key="knn_k_slider")
                st.session_state.knn_k_value = k_value
                
                st.markdown("""
                **What is k?** 
                The number of nearby points KNN uses to make a decision. Higher values make the model smoother but might miss local patterns.
                """)
                
                random_seed = st.slider("Random seed (changes the dataset):", 0, 100, 42)
                if random_seed != st.session_state.random_seed:
                    st.session_state.random_seed = random_seed
                
                if st.button("Run KNN Experiment"):
                    with st.spinner("Testing KNN accuracy across different dimensions..."):
                        knn_fig = generate_knn_accuracy_plot()
                    st.success("Experiment completed!")
            
            with col2:
                try:
                    knn_fig = generate_knn_accuracy_plot()
                    st.pyplot(knn_fig)
                    st.caption("KNN accuracy typically decreases as dimensions increase (with the same amount of data)")
                    
                    # Visualization of why KNN fails
                    st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*VMf6J5M50X0h4p4n76YDRg.png",
                            caption="In high dimensions, the boundary between classes becomes more complex")
                except:
                    st.info("Click 'Run KNN Experiment' to see how KNN accuracy changes with dimensions")
        
        with st.expander("Other Popular Algorithms"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Decision Trees & Random Forests
                
                **How they work:**
                Decision trees split data based on feature values to create decision rules.
                
                **In high dimensions:**
                * ✅ Less sensitive to distance problems
                * ✅ Can ignore irrelevant features
                * ❌ May overfit with too many features
                * ❌ Struggle to capture complex interactions
                
                **Example:** Random Forests combine many trees and use feature subsets, making them more resistant to the curse of dimensionality.
                """)
                
                st.markdown("""
                ### Neural Networks
                
                **How they work:**
                Neural networks learn complex patterns through layers of connected neurons.
                
                **In high dimensions:**
                * ✅ Can learn non-linear patterns
                * ✅ Feature learning happens automatically
                * ❌ Need much more data as dimensions increase
                * ❌ Prone to overfitting without regularization
                
                **Key insight:** Neural networks can work well in high dimensions, but require appropriate regularization and sufficient data.
                """)
            
            with col2:
                st.markdown("""
                ### Support Vector Machines (SVM)
                
                **How they work:**
                SVMs find a boundary (hyperplane) that maximally separates classes.
                
                **In high dimensions:**
                * ✅ Kernel trick helps with non-linear boundaries
                * ✅ Regularization is built-in (margin maximization)
                * ❌ Performance can still degrade in very high dimensions
                * ❌ Selection of appropriate kernel becomes crucial
                
                **Example:** Text classification uses high-dimensional word spaces where SVMs often perform well.
                """)
                
                st.markdown("""
                ### Clustering Algorithms
                
                **How they work:**
                Clustering groups similar data points together without labels.
                
                **In high dimensions:**
                * ❌ K-means struggles with Euclidean distance issues
                * ❌ Density-based methods have trouble defining "density"
                * ❌ Clusters may be meaningless in high dimensions
                * ✅ Specialized high-dimensional clustering algorithms exist
                
                **Tip:** Always combine clustering with dimensionality reduction for high-dimensional data.
                """)
        
        # Visual comparison
        st.subheader("Algorithm Performance in High Dimensions")
        
        # Simple visual representation of algorithm sensitivity
        sensitivity_data = {
            'Algorithm': ['KNN', 'Linear Regression', 'Decision Trees', 'Random Forest', 'SVM', 'Neural Networks'],
            'Sensitivity': [0.9, 0.6, 0.5, 0.4, 0.5, 0.6]
        }
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(sensitivity_data['Algorithm'], sensitivity_data['Sensitivity'], color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Sensitivity to High Dimensions')
        ax.set_title('How Much Each Algorithm Is Affected by High Dimensions')
        
        # Add annotations
        for i, v in enumerate(sensitivity_data['Sensitivity']):
            if v >= 0.7:
                label = "Highly sensitive"
            elif v >= 0.5:
                label = "Moderately sensitive"
            else:
                label = "Less sensitive"
            ax.text(v + 0.01, i, label, va='center')
        
        st.pyplot(fig)
        st.caption("Algorithms that rely heavily on distances (like KNN) are most affected by high dimensions")
        
        # Computational complexity explanation
        st.subheader("Performance and Memory Costs")
        
        st.markdown("""
        ### As Dimensions Increase, So Do Computational Costs
        
        Even for algorithms that can handle high dimensions mathematically, there are practical limitations:
        
        - **Training time** increases with dimensions
        - **Memory requirements** grow substantially
        - **Prediction speed** gets slower
        
        This can make some models impractical for real-time applications with high-dimensional data.
        """)
        
        # Simple complexity table
        complexity_data = {
            'Algorithm': ['KNN', 'K-means', 'Decision Trees', 'Linear Regression', 'Neural Network', 'SVM'],
            'How Training Time Scales': ['Fast for training, slow for prediction', 'Moderate, increases with dimensions', 'Moderate, scales with data size', 'Fast, even with many dimensions', 'Slow, especially with many dimensions', 'Slow with large datasets'],
            'Best For': ['Small to medium datasets with relevant features', 'Clear cluster structures with few dimensions', 'Mixed feature types, moderate dimensions', 'Linear relationships, even in high dimensions', 'Complex patterns with sufficient data', 'Complex boundaries with medium-sized datasets']
        }
        
        complexity_df = pd.DataFrame(complexity_data)
        st.table(complexity_df)
    
    # Tab 5 - Dimensionality Reduction
    with tabs[4]:
        st.header("Dimensionality Reduction: The Solution")
        
        st.markdown("""
        ### Solving the Curse with Fewer Dimensions
        
        Dimensionality reduction is like compressing a large image file - you keep the important information while reducing the size.
        
        **The basic idea:**
        > Transform your data from many dimensions to fewer dimensions, while preserving as much important information as possible.
        """)
        
        # Simple analogy
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### Real-World Analogy
            
            Imagine you're describing people. You could use hundreds of measurements:
            
            - Height, weight, age
            - Circumference of head, neck, arms, legs
            - Length of fingers, toes, nose
            - Hundreds more measurements...
            
            Or you could use just a few key features that capture most of the variation:
            
            - Height (tall vs. short)
            - Build (thin vs. stocky)
            - Age (young vs. old)
            
            Dimensionality reduction works the same way - finding the few dimensions that matter most!
            """)
        
        with col2:
            st.image("https://media.geeksforgeeks.org/wp-content/uploads/20250526125548648108/What-is-Dimensionality-Reduction-.webp", 
                    caption="Dimensionality reduction simplifies data while keeping important patterns")
        
        # PCA Explanation
        st.subheader("How PCA Works - A Simple Explanation")
        
        st.markdown("""
        **Principal Component Analysis (PCA)** is the most popular dimensionality reduction technique. Here's how it works:
        
        1. **Find directions of maximum variance**: PCA looks for directions where your data varies the most
        2. **Rank these directions**: The direction with most variation becomes the first principal component, then second, etc.
        3. **Keep only the important ones**: We can keep just the top components that contain most of the information
        
        Think of it like taking a 3D object and looking at its shadow from the angle that shows the most detail.
        """)
        
        # PCA Interactive Demo
        st.subheader("PCA in Action")
        
        pca_fig, pca_scatter, components_needed, thresholds = generate_pca_demonstration()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.pyplot(pca_fig)
            st.caption("Left: How much information each component contains. Right: Total information captured as we add components.")
        
        with col2:
            st.plotly_chart(pca_scatter, use_container_width=True)
            st.caption("Our 50-dimensional data compressed to just 2 dimensions while still showing clear patterns")
            
            # Display components needed in a more beginner-friendly way
            st.markdown("### The Data Compression Power of PCA")
            
            original_dims = 50
            st.markdown(f"Our original data had **{original_dims} dimensions** (features).")
            st.markdown("With PCA, we can reduce it to:")
            
            for t, c in zip(thresholds, components_needed):
                percent = int(t*100)
                compression = round((original_dims - c) / original_dims * 100)
                st.markdown(f"* **{c} dimensions** to keep {percent}% of the information ({compression}% smaller!)")
            
            st.success(f"We reduced from {original_dims} to just {components_needed[2]} dimensions while preserving 90% of the information!")
        
        # Other dimensionality reduction techniques - beginner-friendly
        st.subheader("Other Ways to Reduce Dimensions")
        
        technique_col1, technique_col2 = st.columns(2)
        
        with technique_col1:
            st.markdown("""
            ### Linear Techniques
            
            **Principal Component Analysis (PCA)**
            * **How it works**: Finds directions of maximum variance
            * **When to use it**: Good first approach for most datasets
            * **Limitation**: Only finds linear relationships
            
            **Linear Discriminant Analysis (LDA)**
            * **How it works**: Finds dimensions that best separate classes
            * **When to use it**: When you have labeled data (classification)
            * **Limitation**: Only works for classification problems
            """)
            
            st.image("https://i.sstatic.net/7fW4l.png",
                    caption="PCA finds directions of maximum variance")
        
        with technique_col2:
            st.markdown("""
            ### Non-linear Techniques
            
            **t-SNE**
            * **How it works**: Preserves local neighborhood structure
            * **When to use it**: For visualization of complex data
            * **Limitation**: Primarily for visualization, not general reduction
            
            **UMAP**
            * **How it works**: Similar to t-SNE but preserves more global structure
            * **When to use it**: Fast visualization of high-dimensional data
            * **Limitation**: Results can vary based on parameters
            
            **Autoencoders**
            * **How it works**: Neural networks that compress then reconstruct data
            * **When to use it**: Complex data with non-linear patterns
            * **Limitation**: Requires more data and tuning
            """)
            
            st.image("https://www.researchgate.net/profile/Christopher-Watkins-4/publication/51235435/figure/fig4/AS:214030204575752@1428040185624/Data-set-2-t-SNE-mappings-and-nearest-neighbour-plots-provide-a-means-to-evaluate-and.png",
                    caption="t-SNE can reveal clusters in complex data")
        
        # Beginner-friendly comparison
        st.markdown("### Which Method Should You Use?")
        
        technique_data = {
            'Method': ['PCA', 'LDA', 't-SNE', 'UMAP', 'Autoencoders'],
            'Best Used For': ['General purpose, first approach', 'Classification problems', 'Visualizing complex data', 'Better t-SNE alternative', 'Very complex patterns'],
            'Complexity': ['Easy', 'Easy', 'Medium', 'Medium', 'Hard'],
            'Speed': ['Fast', 'Fast', 'Slow', 'Medium', 'Slow']
        }
        
        technique_df = pd.DataFrame(technique_data)
        st.table(technique_df)
        st.caption("Start with simpler methods like PCA before trying more complex approaches")
    
    # Tab 6 - Practical Solutions
    with tabs[5]:
        st.header("Practical Solutions and Best Practices")
        
        st.markdown("""
        ### How to Handle High-Dimensional Data
        
        Now that you understand the curse of dimensionality, let's look at practical strategies to address it in your machine learning projects.
        """)
        
        # Different levels of solutions
        with st.expander("Beginner-Friendly Approaches", expanded=True):
            st.markdown("""
            ### Start with These Simple Techniques
            
            1. **Remove obviously irrelevant features**
               * Features with many missing values
               * Features that don't vary much
               * Features with obvious redundancy
            
            2. **Try PCA as your first dimensionality reduction method**
               * Easy to implement in most libraries
               * Works well for many datasets
               * Provides clear measure of information retained
            
            3. **Use built-in feature selection in models**
               * Random Forests have feature importance
               * Regularized models (Lasso) can eliminate features
               * Gradient boosting models rank feature importance
            
            4. **Visualize your data after reduction**
               * Check if classes/clusters are separable
               * Look for outliers and patterns
               * Test if the reduced data makes intuitive sense
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*SWtyRgP8s_1j72rJ4nVAzQ.png", 
                        caption="Feature importance from tree-based models")
            
            with col2:
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*8A2oRCM1sJ2Dx7P_-vLgHw.png", 
                        caption="PCA for visualization")
            
            with col3:
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*9HRMQ8hTXHBEhQRU8LKP6Q.png", 
                        caption="Feature correlation matrix")
        
        # More advanced approaches
        with st.expander("Intermediate & Advanced Approaches"):
            st.markdown("""
            ### As You Get More Comfortable
            
            **Feature Engineering Techniques:**
            * Create meaningful combinations of features
            * Transform features to better represent relationships
            * Apply domain knowledge to create better features
            
            **More Advanced Dimensionality Reduction:**
            * Try non-linear methods like t-SNE or UMAP
            * Explore autoencoders for very complex data
            * Consider supervised dimensionality reduction
            
            **Specialized Algorithms:**
            * Use algorithms designed for high dimensions
            * Consider ensemble methods with different feature subsets
            * Implement regularization techniques
            
            **Validation Strategies:**
            * Use cross-validation to prevent overfitting
            * Test different dimensionality reduction parameters
            * Validate your reduced model against the full model
            """)
        
        # Decision guide
        st.subheader("Which Approach Should You Use?")
        
        decision_flow = """
        digraph {
            node [shape=box, style="rounded,filled", fillcolor=lightblue, fontsize=12];
            start [label="High-Dimensional Data", fillcolor=lightgreen];
            q1 [label="How many features\nrelative to samples?", shape=diamond, fillcolor=lightyellow];
            q2 [label="Do you understand\nthe features well?", shape=diamond, fillcolor=lightyellow];
            q3 [label="Is visualization\nimportant?", shape=diamond, fillcolor=lightyellow];
            q4 [label="What's your goal?", shape=diamond, fillcolor=lightyellow];
            
            a1 [label="Use feature selection\n(Random Forest importance\nor Lasso)"];
            a2 [label="Use domain knowledge to\nselect/engineer features"];
            a3 [label="Try t-SNE or UMAP\nfor visualization"];
            a4 [label="Use PCA for general\ndimensionality reduction"];
            a5 [label="Try supervised dimension\nreduction like LDA"];
            
            start -> q1;
            q1 -> a1 [label="Many more\nfeatures\nthan samples"];
            q1 -> q2 [label="Similar\nnumbers"];
            q2 -> a2 [label="Yes"];
            q2 -> q3 [label="No"];
            q3 -> a3 [label="Yes"];
            q3 -> q4 [label="No"];
            q4 -> a4 [label="Preprocessing"];
            q4 -> a5 [label="Classification"];
        }
        """
        
        st.graphviz_chart(decision_flow)
        st.caption("Simple decision guide for handling high-dimensional data")
        
        # Case study - simplified and more relatable
        st.subheader("Real-World Example: Medical Diagnosis")
        
        st.markdown("""
        ### Challenge:
        
        A hospital wants to predict patient outcomes based on lab results, which include 500 different measurements.
        
        ### Problem:
        * Only 200 patient records available
        * Many lab tests are related or redundant
        * Models were overfitting and performing poorly on new patients
        
        ### Solution:
        
        1. **Medical experts selected 50 most relevant tests** based on clinical knowledge
        
        2. **Used PCA to further reduce to 15 dimensions** that captured 90% of the variation
        
        3. **Applied Random Forest model** with cross-validation on the reduced data
        
        ### Results:
        
        * Prediction accuracy improved from 65% to 82%
        * Model could explain which factors were most important
        * Reduced lab tests needed for new patients
        
        > **Key Lesson:** Sometimes less data (fewer dimensions) gives better results!
        """)
        
        # Practical checklist
        st.subheader("Your High-Dimensional Data Checklist")
        
        st.markdown("""
        Use this checklist for your high-dimensional data projects:
        """)
        
        checks = [
            "Explore your data and understand feature distributions",
            "Check for highly correlated features",
            "Remove or combine redundant features", 
            "Apply feature selection to keep only important features",
            "Try dimensionality reduction methods like PCA",
            "Start with algorithms that handle high dimensions better",
            "Use cross-validation to prevent overfitting",
            "Apply appropriate regularization techniques",
            "Visualize reduced data to check for patterns",
            "Document which features are most important"
        ]
        
        for i, check in enumerate(checks):
            st.checkbox(check, key=f"checklist_{i}")
        
        # Final summary
        st.subheader("Key Takeaways")
        
        st.markdown("""
        ### Remember These Core Principles:
        
        1. **More features isn't always better** - they can make learning harder
        
        2. **Distance-based methods struggle in high dimensions** - be careful with KNN, K-means, etc.
        
        3. **Always consider dimensionality reduction** as a preprocessing step
        
        4. **Feature selection is critical** - quality beats quantity
        
        5. **Visualization helps understanding** - reduce to 2D or 3D to see patterns
        """)
        
        st.success("Congratulations! You now understand the curse of dimensionality and how to address it in your machine learning projects.")

    # Footer with copyright
    st.markdown("---")
    st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")

# Main execution flow
if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()
