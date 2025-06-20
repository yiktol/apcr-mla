import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import io
import utils.common as common
import utils.authenticate as authenticate


def set_page_config():
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="Data Cleaning for ML",
        page_icon="🧹",
        layout="wide"
    )
set_page_config()

def set_custom_css():
    """Set custom CSS for the app."""
    # AWS color scheme
    aws_colors = {
        'orange': '#FF9900',
        'dark_blue': '#232F3E',
        'light_blue': '#1A73E8',
        'teal': '#007DBC',
        'light_gray': '#F2F3F3',
        'dark_gray': '#545B64'
    }
    
    st.markdown("""
        <style>
        .main {
            background-color: #F2F3F3;
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
        div[data-testid="stSidebarNav"] li div a {
            margin-left: 1rem;
            padding: 0rem;
            width: 300px;
            border-radius: 0.5rem;
        }
        div[data-testid="stSidebarNav"] li div::focus-visible {
            background-color: rgba(151, 166, 195, 0.15);
        }
        div[data-baseweb="card"] {
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #e6e6e6;
        }
        .css-1y4p8pa {
            max-width: 100%;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    return aws_colors


def generate_sample_data(n_samples=100, seed=42):
    """Generate sample data for the app."""
    np.random.seed(seed)
    
    # Generate a DataFrame with some patterns and issues
    df = pd.DataFrame({
        'age': np.random.randint(18, 90, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'years_experience': np.random.randint(0, 40, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], n_samples, 
                                    p=[0.3, 0.3, 0.2, 0.1, 0.1]),
        'job_category': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Education', 'Other'], n_samples),
        'satisfaction': np.random.choice([1, 2, 3, 4, 5, None], n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1]),
    })
    
    # Add some duplicates
    duplicate_indices = np.random.choice(range(n_samples), size=int(n_samples*0.1), replace=False)
    for idx in duplicate_indices:
        duplicate_idx = np.random.randint(0, n_samples)
        df.iloc[idx] = df.iloc[duplicate_idx]
    
    # Add some correlations
    df['bonus'] = df['income'] * np.random.uniform(0.05, 0.2, n_samples) + np.random.normal(0, 2000, n_samples)
    
    # Add some outliers
    outlier_indices = np.random.choice(range(n_samples), size=5, replace=False)
    df.loc[outlier_indices, 'income'] = np.random.normal(200000, 50000, len(outlier_indices))
    
    # Add some more missing values
    for col in ['age', 'income', 'years_experience']:
        missing_indices = np.random.choice(range(n_samples), size=int(n_samples*0.05), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # Add a target column for classification
    X, y = make_classification(n_samples=n_samples, n_features=1, n_informative=1, n_redundant=0,
                             n_classes=2, n_clusters_per_class=1, weights=[0.8, 0.2], random_state=seed)
    df['target'] = y
    
    return df


def reset_data():
    """Reset dataset to original state."""
    if 'df' in st.session_state:
        del st.session_state.df
    if 'df_original' in st.session_state:
        del st.session_state.df_original


def initialize_quiz_state():
    """Initialize quiz state variables."""
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}


def initialize_data():
    """Initialize sample data in session state."""
    if 'df' not in st.session_state:
        st.session_state.df = generate_sample_data()
        st.session_state.df_original = st.session_state.df.copy()


def render_sidebar():
    """Render the sidebar elements."""
    common.render_sidebar()
    
    # Reset data button
    if st.sidebar.button("↺ Reset to Original Data", key="reset_2", use_container_width=True):
        reset_data()
        st.rerun()
        
    with st.sidebar.expander("About This App", expanded=False):
        st.info("This application demonstrates various data cleaning techniques for machine learning.")


def render_intro_tab(aws_colors):
    """Render the Introduction tab content."""
    st.header("Introduction to Data Cleaning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Why Data Cleaning Matters
        
        Data cleaning is a critical step in the machine learning pipeline. Raw data often contains issues that can negatively impact model performance:
        
        - **Missing values** can lead to biased models
        - **Duplicates** can give undue weight to certain examples
        - **Outliers** can skew distributions and impact model training
        - **Inconsistent formats** can prevent the model from recognizing patterns
        - **Imbalanced data** can lead to biased predictions
        
        This application demonstrates various techniques to address these issues.
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2940&auto=format&fit=crop", 
                 caption="Data cleaning is an essential part of ML", use_container_width=True)
    
    st.subheader("Sample Dataset")
    st.dataframe(st.session_state.df)
    
    # Basic statistics about the dataset
    st.subheader("Data Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Shape:", st.session_state.df.shape)
        st.write("Missing values:", st.session_state.df.isna().sum().sum())
        st.write("Duplicated rows:", st.session_state.df.duplicated().sum())
    
    with col2:
        # Create pie chart for target distribution
        fig = px.pie(
            values=st.session_state.df['target'].value_counts().values,
            names=st.session_state.df['target'].value_counts().index,
            title='Target Distribution',
            color_discrete_sequence=[aws_colors['teal'], aws_colors['orange']]
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)
    
    # Reset data button
    if st.button("↺ Reset to Original Data"):
        reset_data()
        st.rerun()


def render_missing_values_tab():
    """Render the Missing Values tab content."""
    st.header("Handling Missing Values")
    
    st.markdown("""
    ### Missing Value Strategies
    
    Missing values can significantly affect your model's performance. Here are some common techniques for handling them:
    
    1. **Drop rows** with missing values (risky if you have limited data)
    2. **Impute values** using mean, median, mode, or more complex strategies
    3. **Use algorithms** that handle missing values natively
    
    The best approach depends on why the data is missing and how much data you have.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Visualize missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(st.session_state.df.isnull(), cmap='viridis', yticklabels=False, cbar=False, ax=ax)
        plt.title('Missing Value Heatmap')
        st.pyplot(fig)
        
    with col2:
        # Missing value statistics
        missing_stats = pd.DataFrame({
            'Missing Values': st.session_state.df.isnull().sum(),
            'Percentage': st.session_state.df.isnull().sum() / len(st.session_state.df) * 100
        })
        fig = px.bar(
            missing_stats, 
            y=missing_stats.index, 
            x='Percentage',
            orientation='h',
            title='Percentage of Missing Values by Column',
            color_discrete_sequence=['#007DBC']  # teal color
        )
        fig.update_layout(yaxis_title="", xaxis_title="Percent Missing")
        st.plotly_chart(fig)
    
    # Options for handling missing values
    st.subheader("Apply Missing Value Handling")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        missing_strategy = st.radio(
            "Choose a strategy for handling missing values:",
            ["Drop Rows", "Mean Imputation", "Median Imputation", "Mode Imputation"]
        )
        
        columns_to_handle = st.multiselect(
            "Select columns to handle missing values:",
            st.session_state.df.columns.tolist(),
            default=[col for col in st.session_state.df.columns if st.session_state.df[col].isnull().any()]
        )
    
    with col2:
        if st.button("Apply Missing Value Strategy"):
            if missing_strategy == "Drop Rows":
                st.session_state.df = st.session_state.df.dropna(subset=columns_to_handle)
                st.success(f"Dropped rows with missing values in selected columns. {len(st.session_state.df_original) - len(st.session_state.df)} rows removed.")
            else:
                for col in columns_to_handle:
                    if col in st.session_state.df.columns and st.session_state.df[col].isnull().any():
                        if st.session_state.df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(st.session_state.df[col]):
                            # For categorical/text columns, use mode imputation regardless of selection
                            mode_val = st.session_state.df[col].mode()[0]
                            st.session_state.df[col].fillna(mode_val, inplace=True)
                            st.info(f"Column '{col}' is categorical - using mode imputation with value: {mode_val}")
                        else:
                            # For numeric columns, use the selected strategy
                            if missing_strategy == "Mean Imputation":
                                mean_val = st.session_state.df[col].mean()
                                st.session_state.df[col].fillna(mean_val, inplace=True)
                                st.info(f"Imputed missing values in '{col}' with mean: {mean_val:.2f}")
                            elif missing_strategy == "Median Imputation":
                                median_val = st.session_state.df[col].median()
                                st.session_state.df[col].fillna(median_val, inplace=True)
                                st.info(f"Imputed missing values in '{col}' with median: {median_val:.2f}")
                            elif missing_strategy == "Mode Imputation":
                                mode_val = st.session_state.df[col].mode()[0]
                                st.session_state.df[col].fillna(mode_val, inplace=True)
                                st.info(f"Imputed missing values in '{col}' with mode: {mode_val}")
    
    # Display the resulting dataframe
    st.subheader("Resulting Dataset")
    st.dataframe(st.session_state.df)
    
    # Show before and after statistics
    st.subheader("Before vs After")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Before:")
        st.write(f"Total rows: {len(st.session_state.df_original)}")
        st.write(f"Missing values: {st.session_state.df_original.isna().sum().sum()}")
    
    with col2:
        st.write("After:")
        st.write(f"Total rows: {len(st.session_state.df)}")
        st.write(f"Missing values: {st.session_state.df.isna().sum().sum()}")


def render_duplicates_tab():
    """Render the Duplicates tab content."""
    st.header("Handling Duplicates")
    
    st.markdown("""
    ### Why Handle Duplicates?
    
    Duplicate records in your dataset can:
    
    - Give unnecessary weight to certain examples during model training
    - Artificially inflate validation metrics
    - Leak information between training and testing sets
    
    It's important to identify and address duplicates before model training.
    """)
    
    # Identify duplicates
    duplicated = st.session_state.df.duplicated(keep=False)
    num_duplicates = duplicated.sum()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Total Duplicate Rows", num_duplicates)
        
        if num_duplicates > 0:
            st.subheader("Sample of Duplicated Records")
            st.dataframe(st.session_state.df[duplicated].head())
    
    with col2:
        # Visualization of duplicates
        labels = ['Unique', 'Duplicate']
        values = [len(st.session_state.df) - num_duplicates, num_duplicates]
        
        fig = px.pie(
            values=values,
            names=labels,
            title='Duplicate vs Unique Records',
            color_discrete_sequence=['#007DBC', '#FF9900']  # teal and orange
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)
    
    # Options for handling duplicates
    st.subheader("Apply Duplicate Handling")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        duplicate_strategy = st.radio(
            "Choose a strategy for handling duplicates:",
            ["Keep First Occurrence", "Keep Last Occurrence", "Drop All Duplicates"]
        )
        
        columns_for_dupes = st.multiselect(
            "Consider these columns when identifying duplicates (leave empty for all):",
            st.session_state.df.columns.tolist(),
            default=[]
        )
    
    with col2:
        if st.button("Remove Duplicates"):
            before_count = len(st.session_state.df)
            
            if duplicate_strategy == "Keep First Occurrence":
                st.session_state.df = st.session_state.df.drop_duplicates(
                    subset=columns_for_dupes if columns_for_dupes else None, 
                    keep='first'
                )
            elif duplicate_strategy == "Keep Last Occurrence":
                st.session_state.df = st.session_state.df.drop_duplicates(
                    subset=columns_for_dupes if columns_for_dupes else None, 
                    keep='last'
                )
            else:  # "Drop All Duplicates"
                st.session_state.df = st.session_state.df.drop_duplicates(
                    subset=columns_for_dupes if columns_for_dupes else None, 
                    keep=False
                )
                
            after_count = len(st.session_state.df)
            st.success(f"Removed {before_count - after_count} duplicate records.")
    
    # Display the resulting dataframe
    st.subheader("Resulting Dataset")
    st.dataframe(st.session_state.df)
    
    # Show the dataframe shape before and after
    st.subheader("Before vs After")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Before:")
        st.write(f"Total rows: {len(st.session_state.df_original)}")
        original_dups = st.session_state.df_original.duplicated().sum()
        st.write(f"Duplicate rows: {original_dups}")
    
    with col2:
        st.write("After:")
        st.write(f"Total rows: {len(st.session_state.df)}")
        current_dups = st.session_state.df.duplicated().sum()
        st.write(f"Duplicate rows: {current_dups}")


def render_feature_scaling_tab():
    """Render the Feature Scaling tab content."""
    st.header("Feature Scaling")
    
    st.markdown("""
    ### Why Scale Features?
    
    Feature scaling is crucial for many machine learning algorithms, especially:
    
    - Algorithms that use distance calculations (K-Nearest Neighbors, K-Means)
    - Algorithms with gradient descent optimization (Linear Regression, Neural Networks)
    - Algorithms that use regularization (Ridge, Lasso)
    
    Scaling puts features on a similar scale so that no feature dominates due to its range.
    """)
    
    # Select only numeric columns for scaling
    numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    # Visualize original distributions
    st.subheader("Original Numeric Feature Distributions")
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 3 * n_rows))
    for i, col in enumerate(numeric_cols):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.histplot(st.session_state.df[col], kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Scaling options
    st.subheader("Apply Feature Scaling")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        scaling_method = st.radio(
            "Choose a scaling method:",
            ["MinMax Scaling", "Standard Scaling", "Robust Scaling"]
        )
        
        columns_to_scale = st.multiselect(
            "Select numeric columns to scale:",
            numeric_cols,
            default=numeric_cols
        )
    
    with col2:
        st.markdown("""
        ### Scaling Methods Explained
        
        - **MinMax Scaling**: Transforms features to a range between 0 and 1
        - **Standard Scaling**: Transforms features to have mean=0 and standard deviation=1
        - **Robust Scaling**: Scales based on median and quantiles (less sensitive to outliers)
        """)
        
        if st.button("Apply Scaling"):
            if scaling_method == "MinMax Scaling":
                scaler = MinMaxScaler()
            elif scaling_method == "Standard Scaling":
                scaler = StandardScaler()
            else:  # "Robust Scaling"
                scaler = RobustScaler()
                
            scaled_data = scaler.fit_transform(st.session_state.df[columns_to_scale])
            
            # Replace the original columns with scaled values
            for i, col in enumerate(columns_to_scale):
                st.session_state.df[col] = scaled_data[:, i]
                
            st.success(f"Applied {scaling_method} to {len(columns_to_scale)} columns.")
    
    # Display the resulting dataframe
    st.subheader("Resulting Dataset")
    st.dataframe(st.session_state.df)
    
    # Visualize scaled distributions
    st.subheader("Scaled Numeric Feature Distributions")
    
    # Create new figure for scaled distributions
    fig = plt.figure(figsize=(15, 3 * n_rows))
    for i, col in enumerate(columns_to_scale):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.histplot(st.session_state.df[col], kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)


def render_feature_selection_tab():
    """Render the Feature Selection tab content."""
    st.header("Feature Selection")
    
    st.markdown("""
    ### Why Select Features?
    
    Feature selection helps to:
    
    - Reduce dimensionality and model complexity
    - Improve model performance and interpretability
    - Reduce training time and computational requirements
    - Avoid the "curse of dimensionality"
    
    This demo shows how to drop redundant features or those with high correlation.
    """)
    
    # Show correlation matrix
    numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    corr_matrix = st.session_state.df[numeric_cols].corr()
    
    st.subheader("Feature Correlation Matrix")
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=px.colors.sequential.Blues
    )
    st.plotly_chart(fig)
    
    # Options for feature selection
    st.subheader("Apply Feature Selection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selection_method = st.radio(
            "Choose a feature selection method:",
            ["Drop Selected Features", "Drop Highly Correlated Features"]
        )
        
        if selection_method == "Drop Selected Features":
            features_to_drop = st.multiselect(
                "Select features to drop:",
                st.session_state.df.columns.tolist()
            )
        else:  # "Drop Highly Correlated Features"
            correlation_threshold = st.slider(
                "Correlation threshold (drop one feature from pairs exceeding this):",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05
            )
    
    with col2:
        if selection_method == "Drop Selected Features":
            if st.button("Drop Selected Features"):
                if features_to_drop:
                    st.session_state.df = st.session_state.df.drop(columns=features_to_drop)
                    st.success(f"Dropped {len(features_to_drop)} selected features.")
                else:
                    st.warning("No features selected to drop.")
        else:  # "Drop Highly Correlated Features"
            if st.button("Drop Highly Correlated Features"):
                # Find pairs of highly correlated features
                features_to_drop = set()
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                            # Drop the feature with higher mean correlation with other features
                            corr_i = corr_matrix[numeric_cols[i]].abs().mean()
                            corr_j = corr_matrix[numeric_cols[j]].abs().mean()
                            
                            if corr_i > corr_j:
                                features_to_drop.add(numeric_cols[i])
                            else:
                                features_to_drop.add(numeric_cols[j])
                
                if features_to_drop:
                    st.session_state.df = st.session_state.df.drop(columns=list(features_to_drop))
                    st.success(f"Dropped {len(features_to_drop)} highly correlated features: {', '.join(features_to_drop)}")
                else:
                    st.info("No features found with correlation exceeding the threshold.")
    
    # Display the resulting dataframe
    st.subheader("Resulting Dataset")
    st.dataframe(st.session_state.df)
    
    # Show the dataframe shape before and after
    st.subheader("Before vs After")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Before:")
        st.write(f"Total features: {st.session_state.df_original.shape[1]}")
        st.write(f"Feature list: {', '.join(st.session_state.df_original.columns.tolist())}")
    
    with col2:
        st.write("After:")
        st.write(f"Total features: {st.session_state.df.shape[1]}")
        st.write(f"Feature list: {', '.join(st.session_state.df.columns.tolist())}")
        st.write(f"Features removed: {set(st.session_state.df_original.columns) - set(st.session_state.df.columns)}")


def render_balance_dataset_tab():
    """Render the Balance Dataset tab content."""
    st.header("Balance Dataset")
    
    st.markdown("""
    ### Why Balance Your Dataset?
    
    Class imbalance is a common problem in machine learning where one class has significantly more examples than other classes.
    
    This can lead to:
    - Models that are biased toward the majority class
    - Poor performance on minority classes
    - Misleading evaluation metrics
    
    Balancing techniques help ensure the model learns equally from all classes.
    """)
    
    # Show class distribution
    if 'target' not in st.session_state.df.columns:
        st.warning("Target column not found in the dataset. Skipping balance demonstration.")
    else:
        # Calculate class distribution
        class_counts = st.session_state.df['target'].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Current Class Distribution")
            
            fig = px.bar(
                x=class_counts.index.astype(str),
                y=class_counts.values,
                labels={'x': 'Target Class', 'y': 'Count'},
                title='Class Distribution',
                color=class_counts.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig)
            
        with col2:
            # Calculate imbalance metrics
            total_samples = len(st.session_state.df)
            majority_class = class_counts.idxmax()
            majority_count = class_counts.max()
            minority_class = class_counts.idxmin()
            minority_count = class_counts.min()
            imbalance_ratio = majority_count / minority_count
            
            st.subheader("Imbalance Statistics")
            st.write(f"Total samples: {total_samples}")
            st.write(f"Majority class ({majority_class}): {majority_count} samples ({majority_count/total_samples:.1%})")
            st.write(f"Minority class ({minority_class}): {minority_count} samples ({minority_count/total_samples:.1%})")
            st.write(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Options for balancing
        st.subheader("Apply Balancing Technique")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            balancing_method = st.radio(
                "Choose a balancing method:",
                ["Random Oversampling", "Random Undersampling", "SMOTE (Synthetic Minority Over-sampling)"]
            )
            
        with col2:
            st.markdown("""
            ### Balancing Methods Explained
            
            - **Random Oversampling**: Randomly duplicate examples from minority classes
            - **Random Undersampling**: Randomly remove examples from majority classes
            - **SMOTE**: Generate synthetic examples for minority classes
            """)
            
            if st.button("Apply Balancing"):
                apply_balancing(balancing_method)
        
        # Display the resulting dataframe
        st.subheader("Resulting Dataset")
        st.dataframe(st.session_state.df)
        
        # Show before and after class distribution
        st.subheader("Before vs After Class Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            original_class_counts = st.session_state.df_original['target'].value_counts()
            fig = px.pie(
                values=original_class_counts.values,
                names=original_class_counts.index.astype(str),
                title='Before Balancing',
                color_discrete_sequence=['#007DBC', '#FF9900']  # teal and orange
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig)
        
        with col2:
            current_class_counts = st.session_state.df['target'].value_counts()
            fig = px.pie(
                values=current_class_counts.values,
                names=current_class_counts.index.astype(str),
                title='After Balancing',
                color_discrete_sequence=['#007DBC', '#FF9900']  # teal and orange
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig)


def apply_balancing(balancing_method):
    """Apply a balancing technique to the dataset."""
    # First, ensure we handle missing values before balancing
    has_missing = st.session_state.df.isnull().any().any()
    
    if has_missing:
        st.warning("Dataset contains missing values. These will be imputed before balancing.")
        # Simple imputation for all columns
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
        categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns
        
        # Create a copy to avoid warnings
        temp_df = st.session_state.df.copy()
        
        # Impute numeric columns with mean
        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            temp_df[numeric_cols] = num_imputer.fit_transform(temp_df[numeric_cols])
        
        # Impute categorical columns with most frequent value
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if temp_df[col].isnull().any():
                    mode_val = temp_df[col].mode()[0]
                    temp_df[col].fillna(mode_val, inplace=True)
        
        # Use the imputed dataframe for balancing
        X = temp_df.drop(columns=['target'])
        y = temp_df['target']
    else:
        X = st.session_state.df.drop(columns=['target'])
        y = st.session_state.df['target']
    
    # Handle non-numeric features for all methods
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Create dummy variables for categorical columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        st.info(f"Created dummy variables for categorical columns: {', '.join(categorical_cols)}")
    
    # Apply balancing technique
    try:
        if balancing_method == "Random Oversampling":
            sampler = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        elif balancing_method == "Random Undersampling":
            sampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        else:  # "SMOTE"
            sampler = SMOTE(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Recreate the dataframe - we only keep the numeric columns and target
        # since categorical columns were converted to dummies
        resampled_df = pd.concat([X_resampled, pd.Series(y_resampled, name='target')], axis=1)
        
        # Update the session state
        # Keep only numeric columns and target from original df to avoid issues
        if has_missing or len(categorical_cols) > 0:
            st.info("Note: Categorical columns have been converted to dummy variables in the balanced dataset.")
        
        st.session_state.df = resampled_df
        
        # Show success message
        new_class_counts = st.session_state.df['target'].value_counts()
        st.success(f"Applied {balancing_method}. New class distribution: {dict(new_class_counts)}")
    
    except Exception as e:
        st.error(f"Error applying balancing technique: {str(e)}")
        st.info("Make sure there are no missing values and all features are properly encoded.")


def render_feature_engineering_tab():
    """Render the Feature Engineering tab content."""
    st.header("Feature Engineering")
    
    st.markdown("""
    ### Why Engineer Features?
    
    Feature engineering is the process of creating new features from existing ones to:
    
    - Extract more signal from your data
    - Create features that better represent the underlying patterns
    - Improve model performance by providing more relevant information
    - Transform features into a format that better suits the modeling algorithm
    
    Good feature engineering requires domain knowledge and creativity.
    """)
    
    # Show original features
    st.subheader("Original Features")
    st.dataframe(st.session_state.df.head())
    
    # Feature engineering options
    st.subheader("Apply Feature Engineering")
    
    feature_engineering_options = [
        "Create Polynomial Features",
        "Create Interaction Features",
        "Create Ratio Features",
        "Bin Numeric Features",
        "Custom Feature (Mathematical Expression)"
    ]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        engineering_method = st.selectbox(
            "Choose a feature engineering method:",
            feature_engineering_options
        )
        
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if engineering_method == "Create Polynomial Features":
            degree = st.slider("Polynomial degree:", min_value=2, max_value=3, value=2)
            features_for_poly = st.multiselect(
                "Select numeric features for polynomial expansion:",
                numeric_cols,
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
            )
            
        elif engineering_method == "Create Interaction Features":
            features_for_interaction = st.multiselect(
                "Select features for interaction terms:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
        elif engineering_method == "Create Ratio Features":
            numerator = st.selectbox("Select numerator feature:", numeric_cols)
            denominator = st.selectbox(
                "Select denominator feature:", 
                [col for col in numeric_cols if col != numerator],
                index=0 if len(numeric_cols) > 1 else None
            )
            
        elif engineering_method == "Bin Numeric Features":
            feature_to_bin = st.selectbox("Select feature to bin:", numeric_cols)
            num_bins = st.slider("Number of bins:", min_value=2, max_value=10, value=5)
            
        elif engineering_method == "Custom Feature (Mathematical Expression)":
            available_features = ", ".join(numeric_cols)
            st.info(f"Available features: {available_features}")
            custom_expression = st.text_input(
                "Enter a mathematical expression using available features:",
                value="age * income / 1000" if "age" in numeric_cols and "income" in numeric_cols else ""
            )
            new_feature_name = st.text_input("Name for the new feature:", "custom_feature")
    
    with col2:
        if st.button("Create New Features"):
            apply_feature_engineering(engineering_method, locals())
    
    # Display the resulting dataframe
    st.subheader("Resulting Dataset with Engineered Features")
    st.dataframe(st.session_state.df)
    
    # Show feature statistics before and after
    st.subheader("Before vs After")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Before:")
        st.write(f"Number of features: {st.session_state.df_original.shape[1]}")
    
    with col2:
        st.write("After:")
        st.write(f"Number of features: {st.session_state.df.shape[1]}")
        new_features = set(st.session_state.df.columns) - set(st.session_state.df_original.columns)
        st.write(f"New features created: {', '.join(new_features)}")


def apply_feature_engineering(engineering_method, variables):
    """Apply feature engineering methods to the dataset."""
    try:
        if engineering_method == "Create Polynomial Features":
            features_for_poly = variables.get('features_for_poly', [])
            degree = variables.get('degree', 2)
            
            if features_for_poly:
                # Create polynomial features
                for feature in features_for_poly:
                    for d in range(2, degree + 1):
                        new_feature_name = f"{feature}_power_{d}"
                        st.session_state.df[new_feature_name] = st.session_state.df[feature] ** d
                
                st.success(f"Created polynomial features of degree {degree} for {len(features_for_poly)} features.")
            else:
                st.warning("No features selected for polynomial expansion.")
                
        elif engineering_method == "Create Interaction Features":
            features_for_interaction = variables.get('features_for_interaction', [])
            
            if len(features_for_interaction) >= 2:
                # Create interaction terms
                created = 0
                for i in range(len(features_for_interaction)):
                    for j in range(i+1, len(features_for_interaction)):
                        feat1 = features_for_interaction[i]
                        feat2 = features_for_interaction[j]
                        new_feature_name = f"{feat1}_x_{feat2}"
                        st.session_state.df[new_feature_name] = st.session_state.df[feat1] * st.session_state.df[feat2]
                        created += 1
                        
                st.success(f"Created {created} interaction features.")
            else:
                st.warning("Need at least 2 features to create interaction terms.")
                
        elif engineering_method == "Create Ratio Features":
            numerator = variables.get('numerator')
            denominator = variables.get('denominator')
            
            if numerator and denominator:
                # Create ratio feature
                new_feature_name = f"{numerator}_to_{denominator}"
                # Handle division by zero
                st.session_state.df[new_feature_name] = st.session_state.df[numerator] / (st.session_state.df[denominator] + 1e-8)
                st.success(f"Created ratio feature: {new_feature_name}")
            else:
                st.warning("Both numerator and denominator must be selected.")
                
        elif engineering_method == "Bin Numeric Features":
            feature_to_bin = variables.get('feature_to_bin')
            num_bins = variables.get('num_bins', 5)
            
            if feature_to_bin:
                # Create binned feature
                new_feature_name = f"{feature_to_bin}_binned"
                st.session_state.df[new_feature_name] = pd.qcut(
                    st.session_state.df[feature_to_bin], 
                    q=num_bins, 
                    labels=[f"Bin_{i+1}" for i in range(num_bins)],
                    duplicates='drop'
                )
                st.success(f"Created binned feature: {new_feature_name} with {num_bins} bins.")
            else:
                st.warning("No feature selected for binning.")
                
        elif engineering_method == "Custom Feature (Mathematical Expression)":
            custom_expression = variables.get('custom_expression')
            new_feature_name = variables.get('new_feature_name')
            numeric_cols = variables.get('numeric_cols', [])
            
            if custom_expression and new_feature_name:
                # Replace feature names with dataframe references
                for feature in numeric_cols:
                    custom_expression = custom_expression.replace(
                        feature, 
                        f"st.session_state.df['{feature}']"
                    )
                    
                # Evaluate the expression
                st.session_state.df[new_feature_name] = eval(custom_expression)
                st.success(f"Created custom feature: {new_feature_name}")
            else:
                st.warning("Both expression and feature name are required.")
                
    except Exception as e:
        st.error(f"Error creating features: {str(e)}")


def render_data_conversion_tab():
    """Render the Data Conversion tab content."""
    st.header("Data Conversion")
    
    st.markdown("""
    ### Why Convert Data Types?
    
    Converting data types is important for:
    
    - Ensuring compatibility with ML algorithms (many require numeric inputs)
    - Reducing memory usage and improving performance
    - Correctly representing the meaning of the data (e.g., categorical vs. continuous)
    - Enabling certain operations or transformations
    
    Common conversions include encoding categorical variables and normalizing text.
    """)
    
    # Show current data types
    st.subheader("Current Data Types")
    
    dtypes_df = pd.DataFrame({
        'Column': st.session_state.df.dtypes.index,
        'Data Type': st.session_state.df.dtypes.values.astype(str)
    })
    st.table(dtypes_df)
    
    # Data conversion options
    st.subheader("Apply Data Conversion")
    
    conversion_options = [
        "Encode Categorical Variables",
        "Convert Numeric Types",
        "Convert to Datetime",
        "Convert to String",
        "Extract from Datetime"
    ]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        conversion_method = st.selectbox(
            "Choose a conversion method:",
            conversion_options
        )
        
        # Set up UI based on selected conversion method
        conversion_params = setup_conversion_ui(conversion_method)
    
    with col2:
        if st.button("Apply Conversion"):
            apply_data_conversion(conversion_method, conversion_params)
    
    # Display the resulting dataframe
    st.subheader("Resulting Dataset with Converted Data")
    st.dataframe(st.session_state.df)
    
    # Show data types after conversion
    st.subheader("Updated Data Types")
    
    new_dtypes_df = pd.DataFrame({
        'Column': st.session_state.df.dtypes.index,
        'Data Type': st.session_state.df.dtypes.values.astype(str)
    })
    st.table(new_dtypes_df)
    
    # Show changes
    st.subheader("Changes in Dataset Structure")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Before:")
        st.write(f"Shape: {st.session_state.df_original.shape}")
        st.write(f"Columns: {st.session_state.df_original.shape[1]}")
        st.write(f"Data types: {st.session_state.df_original.dtypes.value_counts().to_dict()}")
    
    with col2:
        st.write("After:")
        st.write(f"Shape: {st.session_state.df.shape}")
        st.write(f"Columns: {st.session_state.df.shape[1]}")
        st.write(f"Data types: {st.session_state.df.dtypes.value_counts().to_dict()}")


def setup_conversion_ui(conversion_method):
    """Set up the UI for data conversion based on the selected method."""
    params = {}
    
    if conversion_method == "Encode Categorical Variables":
        categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not categorical_cols:
            st.warning("No categorical columns found in the dataset.")
        else:
            params['columns_to_encode'] = st.multiselect(
                "Select categorical columns to encode:",
                categorical_cols,
                default=categorical_cols
            )
            
            params['encoding_type'] = st.radio(
                "Select encoding type:",
                ["One-Hot Encoding", "Label Encoding"]
            )
            
    elif conversion_method == "Convert Numeric Types":
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        params['columns_to_convert'] = st.multiselect(
            "Select columns to convert:",
            st.session_state.df.columns.tolist(),
            default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        )
        
        params['target_dtype'] = st.selectbox(
            "Target data type:",
            ["int", "float", "int32", "float32", "int64", "float64"]
        )
        
    elif conversion_method == "Convert to Datetime":
        params['columns_for_datetime'] = st.multiselect(
            "Select columns to convert to datetime:",
            st.session_state.df.columns.tolist()
        )
        
        params['date_format'] = st.text_input(
            "Date format (optional, leave blank for auto-detection):",
            value=""
        )
        
    elif conversion_method == "Convert to String":
        params['columns_to_string'] = st.multiselect(
            "Select columns to convert to string:",
            st.session_state.df.columns.tolist()
        )
        
    elif conversion_method == "Extract from Datetime":
        datetime_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        
        if not datetime_cols:
            st.warning("No datetime columns found in the dataset.")
        else:
            params['datetime_col'] = st.selectbox(
                "Select datetime column:",
                datetime_cols
            )
            
            params['extraction_type'] = st.multiselect(
                "Select components to extract:",
                ["Year", "Month", "Day", "Hour", "Minute", "Second", "Day of Week", "Quarter"]
            )
    
    return params


def apply_data_conversion(conversion_method, params):
    """Apply data conversion based on the selected method and parameters."""
    try:
        if conversion_method == "Encode Categorical Variables":
            columns_to_encode = params.get('columns_to_encode', [])
            encoding_type = params.get('encoding_type')
            
            if columns_to_encode:
                if encoding_type == "One-Hot Encoding":
                    # Apply one-hot encoding
                    encoded_df = pd.get_dummies(
                        st.session_state.df, 
                        columns=columns_to_encode,
                        drop_first=True
                    )
                    st.session_state.df = encoded_df
                    st.success(f"Applied one-hot encoding to {len(columns_to_encode)} columns.")
                    
                else:  # Label Encoding
                    # Apply label encoding
                    for col in columns_to_encode:
                        st.session_state.df[col] = pd.factorize(st.session_state.df[col])[0]
                    st.success(f"Applied label encoding to {len(columns_to_encode)} columns.")
            else:
                st.warning("No columns selected for encoding.")
                
        elif conversion_method == "Convert Numeric Types":
            columns_to_convert = params.get('columns_to_convert', [])
            target_dtype = params.get('target_dtype')
            
            if columns_to_convert:
                for col in columns_to_convert:
                    try:
                        st.session_state.df[col] = st.session_state.df[col].astype(target_dtype)
                    except:
                        st.warning(f"Could not convert column '{col}' to {target_dtype}.")
                st.success(f"Converted columns to {target_dtype}.")
            else:
                st.warning("No columns selected for conversion.")
                
        elif conversion_method == "Convert to Datetime":
            columns_for_datetime = params.get('columns_for_datetime', [])
            date_format = params.get('date_format', '')
            
            if columns_for_datetime:
                for col in columns_for_datetime:
                    try:
                        if date_format:
                            st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], format=date_format)
                        else:
                            st.session_state.df[col] = pd.to_datetime(st.session_state.df[col])
                    except:
                        st.warning(f"Could not convert column '{col}' to datetime.")
                st.success(f"Converted {len(columns_for_datetime)} columns to datetime.")
            else:
                st.warning("No columns selected for datetime conversion.")
                
        elif conversion_method == "Convert to String":
            columns_to_string = params.get('columns_to_string', [])
            
            if columns_to_string:
                for col in columns_to_string:
                    st.session_state.df[col] = st.session_state.df[col].astype(str)
                st.success(f"Converted {len(columns_to_string)} columns to string.")
            else:
                st.warning("No columns selected for string conversion.")
                
        elif conversion_method == "Extract from Datetime":
            datetime_col = params.get('datetime_col')
            extraction_type = params.get('extraction_type', [])
            
            if datetime_col and extraction_type:
                for component in extraction_type:
                    if component == "Year":
                        st.session_state.df[f"{datetime_col}_year"] = st.session_state.df[datetime_col].dt.year
                    elif component == "Month":
                        st.session_state.df[f"{datetime_col}_month"] = st.session_state.df[datetime_col].dt.month
                    elif component == "Day":
                        st.session_state.df[f"{datetime_col}_day"] = st.session_state.df[datetime_col].dt.day
                    elif component == "Hour":
                        st.session_state.df[f"{datetime_col}_hour"] = st.session_state.df[datetime_col].dt.hour
                    elif component == "Minute":
                        st.session_state.df[f"{datetime_col}_minute"] = st.session_state.df[datetime_col].dt.minute
                    elif component == "Second":
                        st.session_state.df[f"{datetime_col}_second"] = st.session_state.df[datetime_col].dt.second
                    elif component == "Day of Week":
                        st.session_state.df[f"{datetime_col}_dayofweek"] = st.session_state.df[datetime_col].dt.dayofweek
                    elif component == "Quarter":
                        st.session_state.df[f"{datetime_col}_quarter"] = st.session_state.df[datetime_col].dt.quarter
                st.success(f"Extracted {len(extraction_type)} components from {datetime_col}.")
            else:
                st.warning("Both datetime column and components to extract must be selected.")
                
    except Exception as e:
        st.error(f"Error during conversion: {str(e)}")


def render_knowledge_check_tab():
    """Render the Knowledge Check tab content."""
    st.header("Test Your Knowledge")
    st.markdown("Let's see how well you understand data cleaning techniques for machine learning!")
    
    # Quiz questions
    questions = [
        {
            "question": "Which technique is most appropriate for handling categorical variables with a meaningful order?",
            "options": ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding", "Binary Encoding"],
            "correct": "Ordinal Encoding",
            "explanation": "Ordinal encoding is specifically designed for categorical variables that have a natural order or hierarchy, such as education levels (High School < Bachelor < Master < PhD)."
        },
        {
            "question": "When dealing with outliers in your dataset, which feature scaling method is least sensitive to them?",
            "options": ["MinMax Scaling", "Standard Scaling", "Robust Scaling", "No scaling"],
            "correct": "Robust Scaling",
            "explanation": "Robust Scaling uses the median and quantiles instead of mean and standard deviation, making it less sensitive to outliers in the data."
        },
        {
            "question": "What problem might arise if you apply SMOTE to balance your dataset without first handling missing values?",
            "options": [
                "The minority class will still be underrepresented", 
                "SMOTE will fail because it doesn't accept missing values encoded as NaN",
                "The majority class will be oversampled instead", 
                "SMOTE will automatically fill in the missing values"
            ],
            "correct": "SMOTE will fail because it doesn't accept missing values encoded as NaN",
            "explanation": "SMOTE requires complete data with no missing values. If your dataset contains NaN values, you need to impute them before applying SMOTE."
        },
        {
            "question": "Which feature selection approach is most appropriate when you have several highly correlated features?",
            "options": [
                "Keep all features to maximize information", 
                "Drop all correlated features", 
                "Drop one feature from each pair of highly correlated features", 
                "Create interaction features between correlated features"
            ],
            "correct": "Drop one feature from each pair of highly correlated features",
            "explanation": "When features are highly correlated, they provide redundant information. Keeping one feature from each correlated pair reduces dimensionality while retaining the important information."
        },
        {
            "question": "What is the main reason for applying feature scaling in machine learning pipelines?",
            "options": [
                "To convert categorical features to numerical ones", 
                "To ensure all features contribute equally to the model", 
                "To remove outliers from the dataset", 
                "To create polynomial features"
            ],
            "correct": "To ensure all features contribute equally to the model",
            "explanation": "Feature scaling puts all features on a similar scale so that no feature dominates the model due to its range. This is particularly important for algorithms that use distance calculations or gradient descent."
        }
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
            st.success("🏆 Perfect score! You're a data cleaning expert!")
        elif st.session_state.quiz_score >= len(questions) * 0.8:
            st.success("🎓 Great job! You have a strong understanding of data cleaning techniques.")
        elif st.session_state.quiz_score >= len(questions) * 0.6:
            st.warning("📚 Good effort! Review the explanations to strengthen your knowledge.")
        else:
            st.error("🔄 You might want to revisit the earlier sections to reinforce your understanding.")
        
        if st.button("Take Quiz Again"):
            reset_quiz()


def submit_quiz(questions):
    """Handle quiz submission and calculate score."""
    score = 0
    for q_idx, question in enumerate(questions):
        if st.session_state.quiz_answers.get(f"q{q_idx}") == question["correct"]:
            score += 1
    st.session_state.quiz_score = score
    st.session_state.quiz_submitted = True


def reset_quiz():
    """Reset quiz state."""
    st.session_state.quiz_score = 0
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers = {}


def render_footer():
    """Render the page footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main function to run the app."""
    aws_colors = set_custom_css()
    
    # Initialize quiz and data
    initialize_quiz_state()
    common.initialize_session_state()
    initialize_data()

    # Sidebar
    with st.sidebar:
        render_sidebar()

    # Header
    st.title("Data Cleaning for Machine Learning")
    st.markdown("Explore different data transformation techniques to prepare your data for ML models")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📚 Introduction", 
        "❓ Missing Values", 
        "🔄 Duplicates", 
        "📏 Feature Scaling", 
        "🎯 Feature Selection", 
        "⚖️ Balance Dataset", 
        "🛠️ Feature Engineering",
        "🔀 Data Conversion",
        "📋 Knowledge Check"
    ])

    with tab1:
        render_intro_tab(aws_colors)
        
    with tab2:
        render_missing_values_tab()
        
    with tab3:
        render_duplicates_tab()
        
    with tab4:
        render_feature_scaling_tab()
        
    with tab5:
        render_feature_selection_tab()
        
    with tab6:
        render_balance_dataset_tab()
        
    with tab7:
        render_feature_engineering_tab()
        
    with tab8:
        render_data_conversion_tab()
        
    with tab9:
        render_knowledge_check_tab()

    # Footer
    render_footer()


if __name__ == "__main__":
    # Check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()