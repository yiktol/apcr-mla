
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="ML Model Evaluation | AWS Learning",
    page_icon="📊",
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'dataset' not in st.session_state:
        st.session_state.dataset = 'breast_cancer'
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
    if 'y_prob' not in st.session_state:
        st.session_state.y_prob = None

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
    options=["Breast Cancer", "Wine Classification", "Synthetic Data"],
    key="dataset_selection"
)


st.sidebar.divider()

with st.sidebar.expander(label='About this application' ,expanded=False):
    st.markdown("""
This application focuses on model evaluation techniques for classification problems in machine learning, covering five essential areas:

- **Confusion Matrix**: Visualize and understand true/false positives and negatives with adjustable thresholds
- **Accuracy & Precision**: Learn when these metrics are appropriate and their limitations
- **Recall & F1 Score**: Explore the balance between different types of errors with an interactive calculator
- **AUC-ROC Curves**: Visualize model performance across all possible thresholds
- **Heat Maps**: Master different visualization techniques for interpreting model results
    """)



if dataset_option == "Breast Cancer":
    st.session_state.dataset = 'breast_cancer'
elif dataset_option == "Wine Classification":
    st.session_state.dataset = 'wine'
else:
    st.session_state.dataset = 'synthetic'

# Function to prepare dataset
@st.cache_data
def prepare_dataset(dataset_name):
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        class_names = data.target_names
    elif dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        class_names = data.target_names
    else:  # synthetic
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10, n_classes=2, 
            random_state=42, n_clusters_per_class=2
        )
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
        class_names = ["Class 0", "Class 1"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    return X_train, X_test, y_train, y_test, y_pred, y_prob, model, feature_names, class_names

# Load dataset based on selection
X_train, X_test, y_train, y_test, y_pred, y_prob, model, feature_names, class_names = prepare_dataset(st.session_state.dataset)

# Store in session state
st.session_state.X_train = X_train
st.session_state.X_test = X_test
st.session_state.y_train = y_train
st.session_state.y_test = y_test
st.session_state.y_pred = y_pred
st.session_state.y_prob = y_prob
st.session_state.model = model

# Helper functions for visualizations
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a Plotly heatmap
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Label", y="True Label"),
        x=class_names,
        y=class_names,
        color_continuous_scale='YlOrRd'
    )
    
    fig.update_layout(
        title="Confusion Matrix",
        title_x=0.5,
        width=600,
        height=500,
        coloraxis_showscale=False
    )
    
    return fig

def plot_metrics_comparison(y_true, y_pred):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }
    
    fig = px.bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        color=list(metrics.values()),
        color_continuous_scale='YlOrRd',
        labels={'x': 'Metric', 'y': 'Score'},
        text_auto='.3f'
    )
    
    fig.update_layout(
        title="Model Performance Metrics",
        title_x=0.5,
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        coloraxis_showscale=False
    )
    
    return fig

def plot_roc_curve(y_true, y_prob, class_names):
    n_classes = len(class_names)
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            line=dict(color='darkorange', width=2),
            name=f'ROC curve (AUC = {roc_auc:.3f})'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(color='navy', width=2, dash='dash'),
            name='Random Guess'
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            title_x=0.5,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.7, y=0.05),
            width=700,
            height=500
        )
    else:
        # Multi-class classification
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        fig = go.Figure()
        
        colors = px.colors.qualitative.Plotly[:n_classes]
        
        for i, color, class_name in zip(range(n_classes), colors, class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                line=dict(color=color, width=2),
                name=f'{class_name} (AUC = {roc_auc:.3f})'
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(color='navy', width=2, dash='dash'),
            name='Random Guess'
        ))
        
        fig.update_layout(
            title='Multi-class ROC Curve',
            title_x=0.5,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.7, y=0.05),
            width=700,
            height=500
        )
    
    return fig

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
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
        labels={'x': 'Importance', 'y': 'Feature'},
        title="Feature Importance"
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        title_x=0.5,
        coloraxis_showscale=False
    )
    
    return fig

def plot_precision_recall_tradeoff(y_test, y_prob):
    thresholds = np.arange(0, 1, 0.05)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        # For binary classification
        if y_prob.shape[1] == 2:
            y_pred = (y_prob[:, 1] >= threshold).astype(int)
            precisions.append(precision_score(y_test, y_pred, average='binary'))
            recalls.append(recall_score(y_test, y_pred, average='binary'))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds, y=precisions,
        mode='lines+markers',
        name='Precision',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds, y=recalls,
        mode='lines+markers',
        name='Recall',
        line=dict(color='red', width=2)
    ))
    
    # Add optimal threshold marker
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    fig.add_trace(go.Scatter(
        x=[optimal_threshold], y=[precisions[optimal_idx]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='star'),
        name=f'Optimal Threshold: {optimal_threshold:.2f}'
    ))
    
    fig.update_layout(
        title='Precision-Recall Tradeoff',
        title_x=0.5,
        xaxis_title='Threshold',
        yaxis_title='Score',
        legend=dict(x=0.01, y=0.01),
        width=700,
        height=500
    )
    
    return fig

# Main content with tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Introduction", 
    "🔄 Confusion Matrix", 
    "🎯 Accuracy & Precision", 
    "📊 Recall & F1 Score", 
    "📈 AUC-ROC", 
    "🔥 Heat Maps", 
    "❓ Knowledge Check"
])

# Introduction Tab
with tab1:
    st.title("Model Evaluation for Classification Problems")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Welcome to the Interactive Machine Learning Model Evaluation Course!</h3>
        <p>In this interactive e-learning module, you'll learn about essential metrics and techniques 
        for evaluating classification models in machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3,2])
    
    with col1:
        st.markdown("""
        ### What You'll Learn
        
        In this module, you'll explore:
        
        1. **Confusion Matrix**: Understanding true positives, false positives, true negatives, and false negatives
        2. **Accuracy**: The proportion of correct predictions among the total predictions
        3. **Precision & Recall**: Balancing specificity and sensitivity in your model
        4. **F1 Score**: The harmonic mean of precision and recall
        5. **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
        6. **Heat Maps**: Visualization techniques for model evaluation
        
        Each section includes interactive examples that allow you to explore these concepts using real datasets.
        
        ### How to Use This Module
        
        1. Navigate through the tabs to explore different evaluation metrics
        2. Use the sidebar to select different datasets
        3. Interact with the visualizations to understand the concepts better
        4. Test your knowledge with the quiz in the last section
        """)
        
        st.info("👈 Select a different dataset from the sidebar to see how evaluation metrics change across different problems!")
        
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/0*8yher9Uy3W5gcWX_", caption="Model Evaluation Flow")
        
        st.markdown("""
        ### Dataset Info
        
        Currently Selected: **{}**
        
        - Training samples: {}
        - Testing samples: {}
        - Features: {}
        - Classes: {}
        """.format(
            dataset_option,
            len(st.session_state.X_train),
            len(st.session_state.X_test),
            len(feature_names),
            len(class_names)
        ))

# Confusion Matrix Tab
with tab2:
    st.title("Confusion Matrix")
    
    st.markdown("""
    <div class="concept-box">
        <h3>What is a Confusion Matrix?</h3>
        <p>A confusion matrix is a table that visualizes the performance of a classification algorithm. 
        Each row represents the actual class, and each column represents the predicted class.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        ### Components of a Confusion Matrix
        
        - **True Positive (TP)**: Correctly predicted positive class
        - **True Negative (TN)**: Correctly predicted negative class
        - **False Positive (FP)**: Incorrectly predicted positive class (Type I error)
        - **False Negative (FN)**: Incorrectly predicted negative class (Type II error)
        
        ### For Binary Classification
        
        |                | Predicted Positive | Predicted Negative |
        |----------------|--------------------|--------------------|
        | Actual Positive| True Positive (TP) | False Negative (FN)|
        | Actual Negative| False Positive (FP)| True Negative (TN) |
        
        ### Why It's Important
        
        The confusion matrix helps us understand:
        - Where the model is making mistakes
        - What types of errors are occurring
        - The balance between different types of errors
        """)
        
        st.markdown('<p class="highlight">A good model should maximize TPs and TNs while minimizing FPs and FNs.</p>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Interactive Confusion Matrix")
        
        # Let user adjust prediction threshold for binary classification
        if len(class_names) == 2 and st.session_state.y_prob is not None:
            threshold = st.slider(
                "Prediction Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="cm_threshold"
            )
            
            # Apply threshold to get predictions
            custom_preds = (st.session_state.y_prob[:, 1] >= threshold).astype(int)
            
            # Plot confusion matrix with custom threshold
            cm_fig = plot_confusion_matrix(st.session_state.y_test, custom_preds, class_names)
            st.plotly_chart(cm_fig, key="cm_plot")
            
            # Show metrics based on this threshold
            tn, fp, fn, tp = confusion_matrix(st.session_state.y_test, custom_preds).ravel()
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("True Positives (TP)", tp)
                st.metric("False Positives (FP)", fp)
            
            with metrics_col2:
                st.metric("True Negatives (TN)", tn)
                st.metric("False Negatives (FN)", fn)
                
        else:
            # For multiclass, just show the standard confusion matrix
            cm_fig = plot_confusion_matrix(st.session_state.y_test, st.session_state.y_pred, class_names)
            st.plotly_chart(cm_fig, key="cm_plot_2")
    
    st.markdown("""
    ### Interpreting the Confusion Matrix
    
    Try adjusting the threshold slider above to see how it affects the confusion matrix. Notice how:
    
    - Increasing the threshold typically reduces false positives but increases false negatives
    - Decreasing the threshold typically increases false positives but reduces false negatives
    
    This illustrates the fundamental trade-off in classification problems!
    """)

# Accuracy & Precision Tab
with tab3:
    st.title("Accuracy & Precision")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Understanding Accuracy and Precision</h3>
        <p>These are fundamental metrics for evaluating classification models, each providing different insights into model performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Accuracy
        
        **Definition:** The proportion of correct predictions (both true positives and true negatives) among the total number of predictions.
        
        **Formula:** 
        $$Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}$$
        
        **When to use:**
        - When classes are balanced
        - When all types of errors are equally important
        
        **Limitations:**
        - Can be misleading with imbalanced classes
        - Doesn't tell you what types of errors are occurring
        """)
        
        acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        st.metric("Current Model Accuracy", f"{acc:.4f}", f"{(acc - 0.5) * 100:.1f}% better than random")
    
    with col2:
        st.markdown("""
        ### Precision
        
        **Definition:** The proportion of correct positive predictions among all positive predictions.
        
        **Formula:** 
        $$Precision = \\frac{TP}{TP + FP}$$
        
        **When to use:**
        - When false positives are costly
        - When you need to be confident in positive predictions
        - Example: Spam detection (don't want to mark legitimate emails as spam)
        
        **Limitations:**
        - Doesn't consider false negatives
        - Can be manipulated by making very few positive predictions
        """)
        
        prec = precision_score(st.session_state.y_test, st.session_state.y_pred, average='weighted')
        st.metric("Current Model Precision", f"{prec:.4f}")
    
    st.markdown("### Interactive Visualization")
    
    # Precision-Recall tradeoff visualization (for binary classification)
    if len(class_names) == 2:
        st.subheader("Precision-Recall Tradeoff")
        pr_fig = plot_precision_recall_tradeoff(st.session_state.y_test, st.session_state.y_prob)
        st.plotly_chart(pr_fig, key="pr_plot")
        
        st.markdown("""
        #### Understanding the Tradeoff
        
        As you can see in the graph above, precision and recall have an inverse relationship when you adjust the classification threshold:
        
        - **Higher threshold:** Increases precision but decreases recall
        - **Lower threshold:** Increases recall but decreases precision
        
        This is why finding the right balance is crucial for your specific application.
        """)
    
    # Compare metrics across different metrics
    st.subheader("Model Performance Metrics")
    metrics_fig = plot_metrics_comparison(st.session_state.y_test, st.session_state.y_pred)
    st.plotly_chart(metrics_fig, key="metrics_plot")
    
    # Class-specific precision
    if len(class_names) > 2:
        st.subheader("Class-specific Precision")
        class_precision = precision_score(st.session_state.y_test, st.session_state.y_pred, average=None)
        
        precision_df = pd.DataFrame({
            'Class': class_names,
            'Precision': class_precision
        })
        
        prec_bar = px.bar(
            precision_df, 
            x='Class', 
            y='Precision',
            color='Precision',
            color_continuous_scale='YlOrRd',
            text_auto='.3f'
        )
        
        prec_bar.update_layout(
            title="Precision by Class",
            title_x=0.5,
            coloraxis_showscale=False,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(prec_bar, key="prec_bar_plot")

# Recall & F1 Score Tab
with tab4:
    st.title("Recall & F1 Score")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Balancing False Negatives and Overall Performance</h3>
        <p>Recall and F1 Score help us understand different aspects of model performance, particularly when dealing with imbalanced datasets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Recall (Sensitivity)
        
        **Definition:** The proportion of actual positives that were correctly identified.
        
        **Formula:** 
        $$Recall = \\frac{TP}{TP + FN}$$
        
        **Also known as:**
        - Sensitivity
        - True Positive Rate
        
        **When to use:**
        - When false negatives are costly
        - When missing positive cases is a serious issue
        - Example: Disease detection (don't want to miss cases)
        
        **Limitations:**
        - Doesn't consider false positives
        - Can achieve perfect recall by classifying everything as positive
        """)
        
        rec = recall_score(st.session_state.y_test, st.session_state.y_pred, average='weighted')
        st.metric("Current Model Recall", f"{rec:.4f}")
    
    with col2:
        st.markdown("""
        ### F1 Score
        
        **Definition:** The harmonic mean of precision and recall.
        
        **Formula:** 
        $$F1 = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$$
        
        **When to use:**
        - When you need a balance between precision and recall
        - When the dataset is imbalanced
        - When both false positives and false negatives are important
        
        **Limitations:**
        - Doesn't consider true negatives
        - Weights precision and recall equally (may not be ideal for all cases)
        
        **Why harmonic mean?** It penalizes extreme values more than arithmetic mean.
        """)
        
        f1 = f1_score(st.session_state.y_test, st.session_state.y_pred, average='weighted')
        st.metric("Current Model F1 Score", f"{f1:.4f}")
    
    # Class-specific recall
    st.subheader("Class-specific Recall")
    class_recall = recall_score(st.session_state.y_test, st.session_state.y_pred, average=None)
    
    recall_df = pd.DataFrame({
        'Class': class_names,
        'Recall': class_recall
    })
    
    recall_bar = px.bar(
        recall_df, 
        x='Class', 
        y='Recall',
        color='Recall',
        color_continuous_scale='YlOrRd',
        text_auto='.3f'
    )
    
    recall_bar.update_layout(
        title="Recall by Class",
        title_x=0.5,
        coloraxis_showscale=False,
        yaxis_range=[0, 1]
    )
    
    st.plotly_chart(recall_bar,key="recall_bar")
    
    # Interactive visualization
    st.subheader("Interactive F1 Score Calculator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        demo_precision = st.slider("Precision:", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        demo_recall = st.slider("Recall:", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    
    with col2:
        demo_f1 = 2 * (demo_precision * demo_recall) / (demo_precision + demo_recall) if (demo_precision + demo_recall) > 0 else 0
        
        st.markdown(f"""
        ### Calculated F1 Score
        
        $$F1 = 2 \\times \\frac{{{demo_precision:.2f} \\times {demo_recall:.2f}}}{{{demo_precision:.2f} + {demo_recall:.2f}}} = {demo_f1:.4f}$$
        """)
        
        # Create gauge chart for F1 score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = demo_f1,
            title = {'text': "F1 Score"},
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
        st.plotly_chart(fig, key="f1_gauge")
    
    st.markdown("""
    ### Understanding the Relationship
    
    F1 Score balances precision and recall. Notice in the calculator above:
    
    - When either precision or recall is low, F1 is closer to the lower value
    - Only when both metrics are high does the F1 score become high
    - F1 score is particularly harsh when there's a large gap between precision and recall
    
    This property makes F1 score especially useful for imbalanced datasets, where accuracy might be misleading.
    """)
    
    # Classification report
    st.subheader("Full Classification Report")
    
    report = classification_report(st.session_state.y_test, st.session_state.y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Style the dataframe
    st.dataframe(report_df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}',
        'f1-score': '{:.2f}',
        'support': '{:.0f}'
    }).background_gradient(cmap='YlOrRd', subset=['precision', 'recall', 'f1-score']))

# AUC-ROC Tab
with tab5:
    st.title("AUC-ROC Curve")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Evaluating Model Performance Across Thresholds</h3>
        <p>The Area Under the Receiver Operating Characteristic curve is a powerful tool for evaluating classification models regardless of the threshold chosen.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        ### What is ROC?
        
        The **Receiver Operating Characteristic (ROC)** curve plots:
        
        - **X-axis**: False Positive Rate (FPR) = FP / (FP + TN)
        - **Y-axis**: True Positive Rate (TPR) = TP / (TP + FN)
        
        It shows the tradeoff between sensitivity (recall) and specificity at various threshold settings.
        
        ### What is AUC?
        
        **AUC (Area Under the Curve)** measures the entire two-dimensional area underneath the ROC curve.
        
        - AUC = 1.0: Perfect classifier
        - AUC = 0.5: No better than random guessing (diagonal line)
        - AUC < 0.5: Worse than random guessing
        
        ### Why is AUC-ROC Useful?
        
        - Scale-invariant: Measures how well predictions are ranked
        - Classification-threshold-invariant: Measures performance across all thresholds
        - Especially useful when dealing with imbalanced datasets
        """)
        
        st.markdown('<p class="highlight">AUC-ROC is particularly valuable when you need to compare different models without setting a specific classification threshold.</p>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Interactive ROC Curve")
        
        roc_fig = plot_roc_curve(st.session_state.y_test, st.session_state.y_prob, class_names)
        st.plotly_chart(roc_fig, key='roc_fig')
        
        st.markdown("""
        #### Understanding the ROC Curve
        
        - **Perfect classifier**: Curve passes through the top left corner (0,1)
        - **Random classifier**: Diagonal line from (0,0) to (1,1)
        - **Better models**: Curves closer to the top-left corner
        """)
    
    st.subheader("Interpreting ROC Curves")
    
    st.markdown("""
    ### What the Curve Tells Us
    
    Each point on the ROC curve represents a different threshold for classifying positive vs. negative:
    
    - Moving up and right (lower threshold): More positives predicted (both true and false)
    - Moving down and left (higher threshold): Fewer positives predicted
    
    ### Practical Applications
    
    The optimal threshold depends on your specific problem:
    
    - **Medical testing for deadly disease**: You might prioritize high recall (TPR) and accept more false positives
    - **Fraud detection on financial transactions**: You might want to balance precision and recall depending on the cost of false positives vs. false negatives
    
    ### Limitations of AUC-ROC
    
    - Doesn't work well with highly imbalanced datasets (consider Precision-Recall AUC instead)
    - Assumes equal importance of false positives and false negatives
    - Doesn't directly show the actual classification performance at any specific threshold
    """)
    
    # Compare models (simulated)
    st.subheader("Comparing Different Models")
    
    # Create simulated AUC values for different models
    models = ['Random Forest', 'Logistic Regression', 'SVM', 'Neural Network', 'Decision Tree']
    aucs = [0.92, 0.88, 0.90, 0.93, 0.82]  # Simulated AUC values
    
    # Create a bar chart
    model_fig = px.bar(
        x=models,
        y=aucs,
        color=aucs,
        color_continuous_scale='YlOrRd',
        labels={'x': 'Model', 'y': 'AUC-ROC'},
        text=[f"{auc:.3f}" for auc in aucs]
    )
    
    model_fig.update_layout(
        title="Comparing Models by AUC-ROC",
        title_x=0.5,
        xaxis_title="Model",
        yaxis_title="AUC-ROC Score",
        coloraxis_showscale=False,
        yaxis_range=[0.5, 1]
    )
    
    st.plotly_chart(model_fig, key='model_fig')

# Heat Maps Tab
with tab6:
    st.title("Heat Maps & Visualization Techniques")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Visualizing Model Performance with Heat Maps</h3>
        <p>Heat maps provide an intuitive way to visualize complex relationships and patterns in your model's performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Types of Visualization Techniques
    
    Heat maps are versatile tools for visualizing various aspects of model evaluation:
    
    1. **Confusion Matrix Heat Maps**: Visualize classification results
    2. **Feature Correlation Heat Maps**: Identify relationships between features
    3. **Feature Importance Heat Maps**: Show which features contribute most to predictions
    4. **Prediction Probability Heat Maps**: Visualize confidence across classes
    """)
    
    viz_option = st.radio(
        "Select a visualization technique to explore:",
        ["Confusion Matrix Heat Map", "Feature Correlation Heat Map", "Feature Importance", "Prediction Probability Heat Map"],
        horizontal=True
    )
    
    if viz_option == "Confusion Matrix Heat Map":
        st.markdown("""
        ### Confusion Matrix Heat Map
        
        Heat maps make confusion matrices more intuitive:
        - **Darker colors**: Higher values
        - **Diagonal elements**: Correct predictions (should be darker)
        - **Off-diagonal elements**: Errors (should be lighter)
        """)
        
        # Create a normalized confusion matrix
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred, normalize='true')
        
        # Create heatmap with annotations
        fig = px.imshow(
            cm,
            text_auto='.2f',
            labels=dict(x="Predicted Label", y="True Label", color="Proportion"),
            x=class_names,
            y=class_names,
            color_continuous_scale='YlOrRd'
        )
        
        fig.update_layout(
            title="Normalized Confusion Matrix Heat Map",
            title_x=0.5,
            width=700,
            height=600
        )
        
        st.plotly_chart(fig, key='confusion_heatmap')
        
        st.markdown("""
        **Interpretation Tips:**
        - The diagonal elements represent the proportion of correctly classified instances for each class
        - Off-diagonal elements show misclassifications and patterns of confusion between classes
        - Perfect classification would show values of 1.0 along the diagonal and 0.0 elsewhere
        """)
    
    elif viz_option == "Feature Correlation Heat Map":
        st.markdown("""
        ### Feature Correlation Heat Map
        
        Correlation heat maps help identify:
        - **Positively correlated features**: Move together (1.0 = perfect positive correlation)
        - **Negatively correlated features**: Move in opposite directions (-1.0 = perfect negative correlation)
        - **Uncorrelated features**: No relationship (0.0 = no correlation)
        
        This helps with feature selection and understanding data relationships.
        """)
        
        # Select a subset of features for better visualization
        max_features = min(10, len(feature_names))
        feature_indices = np.random.choice(len(feature_names), max_features, replace=False) if len(feature_names) > max_features else range(len(feature_names))
        
        # Create correlation matrix
        X_subset = st.session_state.X_train[:, feature_indices]
        corr_matrix = np.corrcoef(X_subset, rowvar=False)
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            labels=dict(x="Feature", y="Feature", color="Correlation"),
            x=[feature_names[i] for i in feature_indices],
            y=[feature_names[i] for i in feature_indices],
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            title="Feature Correlation Heat Map",
            title_x=0.5,
            width=800,
            height=700
        )
        
        st.plotly_chart(fig, key='correlation_heatmap')
        
        st.markdown("""
        **Why Feature Correlation Matters:**
        
        - **Highly correlated features** can cause multicollinearity in some models
        - Can help identify **redundant features** for dimensionality reduction
        - Reveals **relationships in your data** that might not be obvious
        """)
    
    elif viz_option == "Feature Importance":
        st.markdown("""
        ### Feature Importance Visualization
        
        Feature importance helps us understand:
        - Which features have the most predictive power
        - Where to focus feature engineering efforts
        - What the model considers most relevant for predictions
        """)
        
        # Plot feature importance
        fi_fig = plot_feature_importance(st.session_state.model, feature_names)
        st.plotly_chart(fi_fig, key='feature_importance')
        
        st.markdown("""
        **Why Feature Importance Matters:**
        
        - Helps with **feature selection** for model simplification
        - Provides **insights into the problem domain**
        - Can guide **feature engineering** efforts
        - Improves **model interpretability**
        """)
    
    else:  # Prediction Probability Heat Map
        st.markdown("""
        ### Prediction Probability Heat Map
        
        This visualization shows:
        - How confident the model is in its predictions
        - Distribution of prediction probabilities across classes
        - Areas where the model is uncertain
        """)
        
        # Create a sample of test instances for visualization
        sample_size = min(30, len(st.session_state.y_test))
        sample_indices = np.random.choice(len(st.session_state.y_test), sample_size, replace=False)
        
        # Get probabilities for the sample
        sample_probs = st.session_state.y_prob[sample_indices]
        
        # Create heatmap
        fig = px.imshow(
            sample_probs,
            labels=dict(x="Class", y="Sample", color="Probability"),
            x=class_names,
            y=[f"Sample {i+1}" for i in range(sample_size)],
            color_continuous_scale='YlOrRd',
            zmin=0,
            zmax=1
        )
        
        fig.update_layout(
            title="Prediction Probability Heat Map",
            title_x=0.5,
            width=700,
            height=600
        )
        
        st.plotly_chart(fig, key='gi8')
        
        st.markdown("""
        **Interpretation:**
        
        - **Darker cells** indicate higher prediction probability for that class
        - For each sample (row), probabilities across all classes sum to 1.0
        - **Confident predictions** show one very dark cell and others light
        - **Uncertain predictions** show more evenly distributed probabilities
        
        This visualization helps identify where your model is struggling to make clear predictions.
        """)
    
    st.markdown("""
    ### Best Practices for Heat Map Visualizations
    
    1. **Choose appropriate color scales**:
       - Sequential scales (e.g., YlOrRd) for values ranging from low to high
       - Diverging scales (e.g., RdBu) for values centered around a meaningful midpoint
    
    2. **Include annotations** when there aren't too many cells
    
    3. **Normalize data** when appropriate to make comparisons fair
    
    4. **Order rows and columns** to reveal patterns (e.g., cluster similar features)
    
    5. **Provide clear context and interpretation** to help viewers understand what they're seeing
    """)

# Knowledge Check Tab
with tab7:
    st.title("Knowledge Check")
    
    st.markdown("""
    <div class="concept-box">
        <h3>Test Your Understanding</h3>
        <p>Answer these five questions to check your understanding of model evaluation metrics and techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quiz questions
    quiz = [
        {
            "question": "Which metric is most sensitive to imbalanced datasets?",
            "options": ["F1 Score", "Accuracy", "ROC-AUC", "Precision"],
            "answer": "Accuracy"
        },
        {
            "question": "In a confusion matrix, what does the term 'False Positive' refer to?",
            "options": [
                "Predicted negative but actually positive",
                "Predicted positive but actually negative",
                "Correctly predicted positive",
                "Correctly predicted negative"
            ],
            "answer": "Predicted positive but actually negative"
        },
        {
            "question": "Which metric would be most important for a cancer detection system where missing a positive case is very serious?",
            "options": ["Precision", "Recall", "Accuracy", "Specificity"],
            "answer": "Recall"
        },
        {
            "question": "What does an AUC-ROC value of 0.5 indicate?",
            "options": [
                "Perfect classification",
                "Classification no better than random guessing",
                "Classification worse than random guessing",
                "Imbalanced dataset"
            ],
            "answer": "Classification no better than random guessing"
        },
        {
            "question": "What is the F1 Score?",
            "options": [
                "The arithmetic mean of precision and recall",
                "The harmonic mean of precision and recall",
                "The geometric mean of precision and recall",
                "The weighted average of precision and recall"
            ],
            "answer": "The harmonic mean of precision and recall"
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
        st.plotly_chart(fig, key='fig6')
        
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
            st.success("🎉 Perfect score! You've mastered model evaluation concepts!")
        elif st.session_state.quiz_score >= len(quiz) * 0.8:
            st.success("🌟 Great job! You have a strong understanding of model evaluation!")
        elif st.session_state.quiz_score >= len(quiz) * 0.6:
            st.info("👍 Good effort! Review the concepts you missed to strengthen your understanding.")
        else:
            st.warning("📚 You may need more practice. Try reviewing the material again.")
        
        if st.button("Retake Quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score = 0
            st.rerun()

# Footer
st.markdown("""
<footer>
© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</footer>
""", unsafe_allow_html=True)
# ```

# ## Explanation of the Application

# This Streamlit application serves as an interactive e-learning tool for model evaluation in machine learning, specifically focusing on classification problems. Here's how it's structured:

# 1. **Setup and Configuration**
#    - Uses modern Python libraries like Streamlit, scikit-learn, Matplotlib, Seaborn, and Plotly
#    - Sets up AWS-themed styling with custom CSS
#    - Configures session state management

# 2. **Navigation**
#    - Tab-based navigation with emojis for easy access to different concepts
#    - Sidebar for session management and dataset selection

# 3. **Content Sections**
#    - **Introduction**: Overview of model evaluation concepts
#    - **Confusion Matrix**: Interactive visualization with adjustable thresholds
#    - **Accuracy & Precision**: Definitions, formulas, and visualizations
#    - **Recall & F1 Score**: Interactive calculator and metrics by class
#    - **AUC-ROC**: Interactive ROC curves and model comparison
#    - **Heat Maps**: Various visualization techniques with sample data
#    - **Knowledge Check**: Five-question quiz with feedback

# 4. **Interactive Elements**
#    - Dataset selection (Breast Cancer, Wine, Synthetic)
#    - Adjustable thresholds to see impacts on metrics
#    - Interactive visualizations that respond to user inputs
#    - Quiz with instant feedback

# 5. **Visualizations**
#    - Confusion matrices as heat maps
#    - ROC curves
#    - Bar charts for metrics comparison
#    - Feature correlation heat maps
#    - Prediction probability visualizations

# 6. **User Experience**
#    - Clean, modern AWS-themed styling
#    - Responsive layout
#    - Clear explanations with highlighted important concepts
#    - Session reset functionality

# The application is designed to be educational, engaging, and interactive, allowing users to explore model evaluation concepts through hands-on examples and visualizations.