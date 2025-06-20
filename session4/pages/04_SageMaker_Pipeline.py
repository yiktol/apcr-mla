
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import json
import time
import uuid
import random
from datetime import datetime, timedelta
from PIL import Image
import io
import base64
from streamlit_lottie import st_lottie
import requests
from typing import Dict, List, Any, Optional, Tuple, Union


# Custom utility functions
def load_lottieurl(url: str) -> Dict:
    """
    Load a Lottie animation from a URL
    
    Args:
        url: URL of the Lottie animation JSON
    
    Returns:
        Lottie animation as a dictionary, or None if loading fails
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def create_dataset_sample() -> pd.DataFrame:
    """
    Create a sample dataset for demonstration
    
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    n = 1000
    
    # Create synthetic features
    data = {
        'age': np.random.randint(18, 95, n),
        'income': np.random.normal(65000, 25000, n),
        'education_years': np.random.randint(8, 24, n),
        'employment_years': np.random.randint(0, 40, n),
        'debt_to_income': np.random.uniform(0, 1.2, n),
        'credit_score': np.random.randint(300, 850, n),
        'has_mortgage': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'gender': np.random.choice(['Male', 'Female'], n, p=[0.51, 0.49])
    }
    
    # Create target variable (loan approval) with some bias
    loan_probability = (
        0.5 +
        0.1 * (data['credit_score'] > 700) +
        0.15 * (data['income'] > 80000) +
        0.05 * (data['education_years'] > 16) +
        0.05 * (data['debt_to_income'] < 0.4) +
        0.05 * (np.array([g == 'Male' for g in data['gender']]))  # Add slight gender bias
    )
    
    # Ensure probabilities are valid
    loan_probability = np.clip(loan_probability, 0.001, 0.999)
    
    # Generate loan approval
    data['loan_approved'] = np.random.binomial(1, loan_probability)
    
    return pd.DataFrame(data)


def generate_data_report(df: pd.DataFrame) -> Dict:
    """
    Generate a data report for a DataFrame
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary containing data report metrics
    """
    report = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_types": {},
        "missing_values": {},
        "mean_values": {},
        "median_values": {},
        "min_values": {},
        "max_values": {},
        "unique_values_count": {},
        "top_correlations": []
    }
    
    # Generate column types and statistics
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            report["column_types"][col] = "numeric"
            report["mean_values"][col] = round(float(df[col].mean()), 2)
            report["median_values"][col] = round(float(df[col].median()), 2)
            report["min_values"][col] = round(float(df[col].min()), 2)
            report["max_values"][col] = round(float(df[col].max()), 2)
        else:
            report["column_types"][col] = "categorical"
            
        report["missing_values"][col] = int(df[col].isna().sum())
        report["unique_values_count"][col] = int(df[col].nunique())
    
    # Get top correlations for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.loc[col1, col2]
                corr_pairs.append((col1, col2, corr))
        
        # Sort by absolute correlation
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Take top 5
        for col1, col2, corr in corr_pairs[:5]:
            report["top_correlations"].append({
                "feature1": col1,
                "feature2": col2,
                "correlation": round(float(corr), 3)
            })
    
    return report


def create_transformation_examples(df: pd.DataFrame) -> Dict:
    """
    Create examples of data transformations for demonstration
    
    Args:
        df: DataFrame to transform
    
    Returns:
        Dictionary containing transformation examples
    """
    # Take a sample for demonstration
    sample = df.sample(5).copy()
    
    # Numeric transformations
    numeric_cols = ['age', 'income', 'credit_score', 'debt_to_income']
    sample_numeric = sample[numeric_cols].copy()
    
    # Scaled values
    scaled_sample = {}
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        scaled_sample[col] = [(x - min_val) / (max_val - min_val) for x in sample[col].values]
    
    # Categorical transformations
    gender_encoding = sample['gender'].map({'Male': 0, 'Female': 1})
    
    # One-hot encoding example
    one_hot_gender = pd.get_dummies(sample['gender'], prefix='gender')
    
    return {
        "original": sample.to_dict('records'),
        "scaled": scaled_sample,
        "encoded_gender": gender_encoding.tolist(),
        "one_hot_gender": one_hot_gender.to_dict('records')
    }


def create_pipeline_graph() -> nx.DiGraph:
    """
    Create a directed graph for the SageMaker Pipeline visualization
    
    Returns:
        NetworkX DiGraph object
    """
    G = nx.DiGraph()
    
    # Add pipeline nodes
    G.add_node("data_input", label="Data Input", node_type="data", description="S3 data source")
    G.add_node("processing", label="Data Processing", node_type="processing", 
              description="Feature engineering, preprocessing, splitting")
    G.add_node("training", label="Model Training", node_type="training",
              description="XGBoost model training")
    G.add_node("evaluation", label="Model Evaluation", node_type="evaluation",
              description="Model metric calculation")
    G.add_node("condition", label="Accuracy Check", node_type="condition",
              description="Min. accuracy threshold check")
    G.add_node("register_true", label="Register Model", node_type="register",
              description="Register model in Model Registry")
    G.add_node("register_false", label="Skip Registration", node_type="skip",
              description="Model doesn't meet quality bar")
    G.add_node("clarify", label="Bias & Explainability", node_type="clarify",
              description="Bias detection and feature importance")
    
    # Add edges to show flow
    G.add_edge("data_input", "processing")
    G.add_edge("processing", "training")
    G.add_edge("training", "evaluation")
    G.add_edge("evaluation", "condition")
    G.add_edge("condition", "register_true", condition="accuracy >= 0.85")
    G.add_edge("condition", "register_false", condition="accuracy < 0.85")
    G.add_edge("register_true", "clarify")
    
    return G


def draw_pipeline_graph(
    G: nx.DiGraph, 
    highlight_path: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 7)
) -> plt.Figure:
    """
    Draw the pipeline graph using NetworkX and Matplotlib
    
    Args:
        G: NetworkX DiGraph to draw
        highlight_path: Optional list of node names to highlight
        figsize: Figure size tuple (width, height)
    
    Returns:
        Matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    
    # Define node positions for a clean pipeline view
    pos = {
        "data_input": (1, 2),
        "processing": (3, 2),
        "training": (5, 2),
        "evaluation": (7, 2),
        "condition": (9, 2),
        "register_true": (11, 3),
        "register_false": (11, 1),
        "clarify": (13, 3)
    }
    
    # Define colors based on node type
    node_colors = {
        "data": "#3F88C5",      # blue
        "processing": "#FFA500", # orange
        "training": "#00A1C9",   # teal
        "evaluation": "#59BA47", # green
        "condition": "#FFDB58",  # yellow
        "register": "#59BA47",   # green
        "skip": "#D13212",       # red
        "clarify": "#9D1F63"     # purple
    }
    
    # Define node shapes
    node_shapes = {
        "data": "s",       # square
        "processing": "o",  # circle  
        "training": "o",    # circle
        "evaluation": "o",  # circle
        "condition": "d",   # diamond
        "register": "o",    # circle
        "skip": "o",        # circle
        "clarify": "o"      # circle
    }
    
    # Prepare node lists by type for separate drawing
    nodes_by_type = {}
    for node_type in set(nx.get_node_attributes(G, 'node_type').values()):
        nodes_by_type[node_type] = [node for node in G.nodes() 
                                   if G.nodes[node].get('node_type') == node_type]
    
    # Draw nodes by type
    for node_type, nodes in nodes_by_type.items():
        # Skip empty lists
        if not nodes:
            continue
        
        # Get color for this type
        color = node_colors.get(node_type, "#CCCCCC")
        
        # Filter nodes based on highlight path
        if highlight_path:
            highlighted_nodes = [n for n in nodes if n in highlight_path]
            regular_nodes = [n for n in nodes if n not in highlight_path]
            
            # Draw highlighted nodes with special formatting
            if highlighted_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=highlighted_nodes,
                    node_shape=node_shapes.get(node_type, 'o'),
                    node_color=color,
                    node_size=1800,
                    edgecolors='black',
                    linewidths=3
                )
            
            # Draw regular nodes with standard formatting
            if regular_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=regular_nodes,
                    node_shape=node_shapes.get(node_type, 'o'),
                    node_color=color,
                    node_size=1500,
                    alpha=0.5,
                    edgecolors='black',
                    linewidths=1
                )
        else:
            # Draw all nodes normally
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_shape=node_shapes.get(node_type, 'o'),
                node_color=color,
                node_size=1500,
                edgecolors='black',
                linewidths=1
            )
    
    # Draw edges with different formatting based on highlight path
    for edge in G.edges():
        start_node, end_node = edge
        
        # Check if edge is part of the highlight path
        is_highlighted = (highlight_path and 
                          start_node in highlight_path and 
                          end_node in highlight_path)
        
        # Get condition if it exists
        condition = G.edges[edge].get('condition', '')
        
        # Format based on highlighting
        if is_highlighted:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[edge],
                width=3.0,
                arrowsize=20,
                arrowstyle='-|>',
                edge_color='#FF9900'  # AWS orange for highlighted edges
            )
        else:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[edge],
                width=1.5,
                arrowsize=15,
                arrowstyle='-|>',
                edge_color='gray',
                alpha=0.7
            )
        
        # Add edge labels for conditions
        if condition:
            edge_center = ((pos[start_node][0] + pos[end_node][0]) / 2,
                          (pos[start_node][1] + pos[end_node][1]) / 2)
            
            # Adjust y position for conditional edges
            if start_node == "condition":
                if end_node == "register_true":
                    edge_center = (edge_center[0], edge_center[1] + 0.25)
                else:  # register_false
                    edge_center = (edge_center[0], edge_center[1] - 0.25)
            
            # Draw the condition text
            plt.text(
                edge_center[0], edge_center[1],
                condition,
                fontsize=9,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'),
                horizontalalignment='center',
                verticalalignment='center'
            )
    
    # Add labels to nodes
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_weight='bold',
        verticalalignment='center'
    )
    
    # Add descriptions under nodes
    descriptions = nx.get_node_attributes(G, 'description')
    for node, description in descriptions.items():
        plt.text(
            pos[node][0], pos[node][1] - 0.3,
            description,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=8,
            color='#444444',
            wrap=True
        )
    
    plt.title("Amazon SageMaker Pipeline - Model Build Workflow", fontsize=16, pad=20)
    plt.axis('off')  # Turn off axis
    
    return plt.gcf()


def create_feature_importance_chart(n_features: int = 8) -> alt.Chart:
    """
    Create a feature importance chart
    
    Args:
        n_features: Number of features to display
    
    Returns:
        Altair Chart object
    """
    # Create sample feature importance data
    np.random.seed(42)
    
    feature_names = ['credit_score', 'income', 'debt_to_income', 'age', 
                     'employment_years', 'education_years', 'has_mortgage', 'gender']
    
    # Create importance scores with some randomness but sensible values
    importance_base = np.array([0.32, 0.27, 0.18, 0.09, 0.06, 0.04, 0.03, 0.01])
    importance = importance_base + np.random.normal(0, 0.02, len(feature_names))
    importance = np.maximum(importance, 0)  # No negative values
    importance = importance / importance.sum()  # Re-normalize
    
    # Create DataFrame
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names[:n_features],
        'Importance': importance[:n_features]
    })
    
    # Sort by importance
    feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)
    
    # Create the chart
    chart = alt.Chart(feature_imp_df).mark_bar().encode(
        y=alt.Y('Feature:N', sort='-x', title=None),
        x=alt.X('Importance:Q', title='Feature Importance'),
        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='oranges'))
    ).properties(
        title='Feature Importance (SHAP Values)',
        width=500,
        height=300
    )
    
    return chart


def create_bias_metrics() -> pd.DataFrame:
    """
    Create sample bias metrics for demonstration
    
    Returns:
        DataFrame with bias metrics
    """
    metrics = {
        'Metric': [
            'Disparate Impact (DI)',
            'Class Imbalance (CI)',
            'Difference in Positive Proportions (DPP)',
            'Accuracy Difference',
            'Recall Difference',
            'Precision Difference'
        ],
        'Value': [
            0.83,  # DI < 0.8 indicates potential bias
            0.18,  # CI > 0.15 indicates class imbalance 
            0.09,  # DPP > 0.05 indicates potential bias
            0.05,  # Accuracy difference > 0.05 can indicate bias
            0.11,  # Recall difference > 0.1 can indicate bias
            0.07   # Precision difference > 0.05 can indicate bias
        ],
        'Threshold': [
            0.8,  # DI threshold
            0.15,  # CI threshold
            0.05,  # DPP threshold
            0.05,  # Accuracy threshold
            0.1,  # Recall threshold
            0.05   # Precision threshold
        ],
        'Status': [
            'Warning',  # DI < 0.8, warning
            'Warning',  # CI > 0.15, warning
            'Warning',  # DPP > 0.05, warning
            'Warning',  # Accuracy diff > 0.05, warning
            'Warning',  # Recall diff > 0.1, warning
            'Warning'   # Precision diff > 0.05, warning
        ]
    }
    
    return pd.DataFrame(metrics)


def create_bias_visualization() -> alt.Chart:
    """
    Create a visualization for bias analysis
    
    Returns:
        Altair Chart object
    """
    # Create data for approval rates by gender
    gender_data = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Approval_Rate': [0.72, 0.60]
    })
    
    # Create a bar chart
    chart = alt.Chart(gender_data).mark_bar().encode(
        x=alt.X('Gender:N', title=None),
        y=alt.Y('Approval_Rate:Q', title='Approval Rate', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Gender:N', scale=alt.Scale(domain=['Male', 'Female'], 
                                                   range=['#00A1C9', '#FF9900']))
    ).properties(
        title='Loan Approval Rate by Gender',
        width=400,
        height=300
    )
    
    return chart


def generate_sample_code(step: str) -> str:
    """
    Generate sample code for different steps in the SageMaker Pipeline
    
    Args:
        step: Which pipeline step to show code for
    
    Returns:
        String containing sample Python code
    """
    if step == "pipeline_definition":
        return '''
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ClarifyCheckStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.clarify.processor import SageMakerClarifyProcessor
from sagemaker.xgboost import XGBoost

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sagemaker_session.boto_region_name

# Define parameters
input_data = "s3://sagemaker-{region}/loan-data/input/loan_data.csv"
output_data_prefix = "s3://sagemaker-{region}/loan-data/output"
model_package_group_name = "LoanPredictionModels"

# Define preprocessing step
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1
)

preprocessing_step = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation")
    ],
    code="preprocessing.py"
)

# Define training step
xgb_estimator = XGBoost(
    entry_point="train.py",
    framework_version="1.3-1",
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100,
        "max_depth": 5,
        "eta": 0.2,
        "subsample": 0.8
    },
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"{output_data_prefix}/model"
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Define evaluation step for model quality
evaluation_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    code="evaluate.py"
)

# Define model evaluation property file
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

# Define condition step for model registration
condition_step = ConditionStep(
    name="CheckModelAccuracy",
    conditions=[ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value"
        ),
        right=0.85  # Accept models with 85% accuracy or higher
    )],
    if_steps=[register_model_step],
    else_steps=[]
)

# Define the bias analysis check step
clarify_processor = SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sagemaker_session
)

bias_check_config = {
    "dataset_type": "text/csv",
    "headers": ["age", "income", "education_years", "employment_years", "debt_to_income", 
                "credit_score", "has_mortgage", "gender", "loan_approved"],
    "label": "loan_approved",
    "sensitive_attributes": ["gender"],
    "facet": [{"name_or_index": "gender", "value_or_threshold": "Female"}],
    "metrics": ["CI", "DPL", "DI", "DCA"]
}

bias_check_step = ClarifyCheckStep(
    name="CheckBias",
    clarify_check_config=bias_check_config,
    model_name=training_step.properties.TrainingJobName,
    skip_check=False,
    register_new_baseline=True
)

# Define the explainability check step
explainability_check_config = {
    "dataset_type": "text/csv",
    "headers": ["age", "income", "education_years", "employment_years", "debt_to_income", 
                "credit_score", "has_mortgage", "gender", "loan_approved"],
    "label": "loan_approved",
    "methods": "shap",
    "num_samples": 100
}

explainability_check_step = ClarifyCheckStep(
    name="CheckExplainability",
    clarify_check_config=explainability_check_config,
    model_name=training_step.properties.TrainingJobName,
    skip_check=False,
    register_new_baseline=True
)

# Create the pipeline
pipeline = Pipeline(
    name="LoanApprovalPipeline",
    parameters=[
        processing_instance_type,
        processing_instance_count,
        training_instance_type,
        model_approval_status
    ],
    steps=[
        preprocessing_step,
        training_step,
        eval_step,
        condition_step,
        bias_check_step,
        explainability_check_step
    ]
)

# Submit the pipeline definition
pipeline.upsert(role_arn=role)
execution = pipeline.start()
'''.strip()

    elif step == "data_processing":
        return '''
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Define preprocessing functions
def load_and_clean_data(input_file):
    """Load and clean the dataset"""
    df = pd.read_csv(input_file)
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = ['age', 'income', 'education_years', 'employment_years', 
                  'debt_to_income', 'credit_score']
    for col in numeric_cols:
        if col in df and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Drop rows with remaining missing values
    df = df.dropna()
    
    return df

def preprocess_features(df):
    """Apply feature engineering and preprocessing"""
    # Feature engineering
    df['income_to_debt_ratio'] = df['income'] / (df['debt_to_income'] * df['income'] + 1)
    df['credit_income_ratio'] = df['credit_score'] / (df['income'] / 10000)
    
    # Convert categorical variables
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    return df

# Main processing function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    
    # Define paths
    input_path = args.input_data
    train_path = "/opt/ml/processing/train"
    validation_path = "/opt/ml/processing/validation"
    test_path = "/opt/ml/processing/test"
    
    # Create output directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_clean_data(input_path)
    df = preprocess_features(df)
    
    # Split data into features and target
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    
    # Create train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Define preprocessing for numeric columns
    numeric_features = ['age', 'income', 'education_years', 'employment_years', 
                       'debt_to_income', 'credit_score', 'income_to_debt_ratio', 
                       'credit_income_ratio']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Apply preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor for inference
    joblib.dump(preprocessor, os.path.join("/opt/ml/processing", "preprocessor.joblib"))
    
    # Convert to DataFrame and save processed datasets
    train_df = pd.DataFrame(X_train_processed)
    train_df['target'] = y_train.values
    
    val_df = pd.DataFrame(X_val_processed)
    val_df['target'] = y_val.values
    
    test_df = pd.DataFrame(X_test_processed)
    test_df['target'] = y_test.values
    
    # Write to CSV files
    train_df.to_csv(os.path.join(train_path, "train.csv"), header=False, index=False)
    val_df.to_csv(os.path.join(validation_path, "validation.csv"), header=False, index=False)
    test_df.to_csv(os.path.join(test_path, "test.csv"), header=False, index=False)
    
    print("Preprocessing completed!")
'''.strip()

    elif step == "model_training":
        return '''
import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--min_child_weight", type=float, default=1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int, default=100)
    
    # SageMaker parameters
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    
    args = parser.parse_args()
    
    # Load training and validation data
    train_df = pd.read_csv(f"{args.train}/train.csv", header=None)
    val_df = pd.read_csv(f"{args.validation}/validation.csv", header=None)
    
    # Last column is the target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    
    X_val = val_df.iloc[:, :-1]
    y_val = val_df.iloc[:, -1]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Training parameters
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "objective": args.objective,
        "eval_metric": "logloss"
    }
    
    # Train the model with early stopping
    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=10,
        evals_result=evals_result,
        verbose_eval=10
    )
    
    # Save the model
    model_file = os.path.join(args.model_dir, "xgboost-model")
    model.save_model(model_file)
    
    print(f"Model saved at {model_file}")
    print(f"Final training logloss: {evals_result['train']['logloss'][-1]}")
    print(f"Final validation logloss: {evals_result['validation']['logloss'][-1]}")
'''.strip()

    elif step == "model_evaluation":
        return '''
import argparse
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-path", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()
    
    # Load the model
    model_path = os.path.join(args.model_path, "model.tar.gz")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load test data
    test_df = pd.read_csv(os.path.join(args.test_path, "test.csv"), header=None)
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create JSON output with metrics
    metrics = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy},
            "precision": {"value": precision},
            "recall": {"value": recall},
            "f1": {"value": f1},
            "auc": {"value": auc}
        }
    }
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save metrics to output path
    with open(os.path.join(args.output_path, "evaluation.json"), "w") as f:
        f.write(json.dumps(metrics))
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 8))
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(args.output_path, "confusion_matrix.png"))
    
    print(f"Evaluation metrics:\n{json.dumps(metrics, indent=2)}")
'''.strip()

    elif step == "conditional_registration":
        return '''
# This would be part of the pipeline definition
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet

# Define the evaluation report property file
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

# Create a RegisterModel step
register_model_step = RegisterModel(
    name="RegisterModel",
    estimator=xgb_estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large", "ml.m5.xlarge"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status
)

# Define condition step for model registration
condition_step = ConditionStep(
    name="CheckModelAccuracy",
    conditions=[ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value"
        ),
        right=0.85  # Accept models with 85% accuracy or higher
    )],
    if_steps=[register_model_step],
    else_steps=[]  # Can add notification or other steps for rejected models
)
'''.strip()

    else:  # bias and explainability
        return '''
# This would be part of the pipeline definition
from sagemaker.clarify.processor import SageMakerClarifyProcessor
from sagemaker.workflow.clarify_check_step import ClarifyCheckStep
from sagemaker.workflow.quality_check_step import QualityCheckStep

# Define the clarify processor
clarify_processor = SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sagemaker_session
)

# Define bias configuration
bias_config = {
    "dataset_type": "text/csv",
    "dataset_uri": preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
    "headers": ["age", "income", "education_years", "employment_years", "debt_to_income", 
                "credit_score", "has_mortgage", "gender", "loan_approved"],
    "label": "loan_approved",
    "sensitive_attributes": ["gender"],
    "facet": [{"name_or_index": "gender", "value_or_threshold": "Female"}],
    "methods": {
        "pre_training_bias": {"methods": ["CI", "DPL", "DI", "DCA"]}
    }
}

# Define bias check step
bias_check_step = ClarifyCheckStep(
    name="CheckBias",
    clarify_check_config=bias_config,
    model_name=training_step.properties.TrainingJobName,
    skip_check=False,
    register_new_baseline=True
)

# Define explainability configuration
explainability_config = {
    "dataset_type": "text/csv",
    "dataset_uri": preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
    "headers": ["age", "income", "education_years", "employment_years", "debt_to_income", 
                "credit_score", "has_mortgage", "gender", "loan_approved"],
    "label": "loan_approved",
    "methods": {
        "shap": {
            "baseline": [0.5] * 8,
            "num_samples": 100,
            "agg_method": "mean_abs"
        }
    }
}

# Define explainability check step
explainability_check_step = ClarifyCheckStep(
    name="CheckExplainability",
    clarify_check_config=explainability_config,
    model_name=training_step.properties.TrainingJobName,
    skip_check=False,
    register_new_baseline=True
)
'''.strip()


def create_model_metrics() -> pd.DataFrame:
    """
    Create sample model evaluation metrics
    
    Returns:
        DataFrame with model metrics
    """
    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [0.87, 0.83, 0.91, 0.87, 0.92],
        'Threshold': [0.85, 0.80, 0.80, 0.80, 0.85],
        'Status': ['Passed', 'Passed', 'Passed', 'Passed', 'Passed']
    }
    
    return pd.DataFrame(metrics)


def create_confusion_matrix() -> plt.Figure:
    """
    Create a sample confusion matrix visualization
    
    Returns:
        Matplotlib Figure object
    """
    # Define confusion matrix values
    cm = np.array([[680, 120], [80, 720]])
    
    # Create figure with labels
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Predicted Negative', 'Predicted Positive'],
               yticklabels=['Actual Negative', 'Actual Positive'])
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    return fig


def create_model_comparison() -> pd.DataFrame:
    """
    Create a sample model comparison DataFrame
    
    Returns:
        DataFrame with model comparison
    """
    comparison = {
        'Model Version': ['v1', 'v2 (Current)', 'v3 (Candidate)'],
        'Accuracy': [0.82, 0.85, 0.87],
        'Precision': [0.79, 0.81, 0.83],
        'Recall': [0.85, 0.88, 0.91],
        'F1 Score': [0.82, 0.84, 0.87],
        'ROC AUC': [0.88, 0.90, 0.92],
        'Status': ['Rejected', 'In Production', 'Candidate']
    }
    
    return pd.DataFrame(comparison)


def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'dataset' not in st.session_state:
        st.session_state.dataset = create_dataset_sample()
        
    if 'data_report' not in st.session_state:
        st.session_state.data_report = generate_data_report(st.session_state.dataset)
        
    if 'transformation_examples' not in st.session_state:
        st.session_state.transformation_examples = create_transformation_examples(st.session_state.dataset)
        
    if 'pipeline_graph' not in st.session_state:
        st.session_state.pipeline_graph = create_pipeline_graph()
        
    if 'highlight_path' not in st.session_state:
        st.session_state.highlight_path = None
        
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = create_model_metrics()
        
    if 'model_comparison' not in st.session_state:
        st.session_state.model_comparison = create_model_comparison()
        
    if 'bias_metrics' not in st.session_state:
        st.session_state.bias_metrics = create_bias_metrics()


def reset_session():
    """
    Reset the session state
    """
    # Keep only user_id and reset all other state
    user_id = st.session_state.user_id
    st.session_state.clear()
    st.session_state.user_id = user_id


# Main application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="SageMaker Pipelines Learning",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # AWS Color Scheme
    AWS_COLORS = {
        "orange": "#FF9900",
        "teal": "#00A1C9", 
        "blue": "#232F3E",
        "gray": "#E9ECEF",
        "light_gray": "#F8F9FA",
        "white": "#FFFFFF",
        "dark_gray": "#545B64",
        "green": "#59BA47",
        "red": "#D13212",
        "purple": "#9D1F63"
    }
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {
            background-color: #F8F9FA;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #F8F9FA;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            white-space: pre-wrap;
            border-radius: 6px;
            font-weight: 600;
            background-color: #FFFFFF;
            color: #232F3E;
            border: 1px solid #E9ECEF;
            padding: 5px 15px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF9900 !important;
            color: #FFFFFF !important;
            border: 1px solid #FF9900 !important;
        }
        .stButton button {
            background-color: #FF9900;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 8px 16px;
        }
        .stButton button:hover {
            background-color: #EC7211;
        }
        .info-box {
            background-color: #E6F2FF;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #00A1C9;
        }
        .code-box {
            background-color: #232F3E;
            color: #FFFFFF;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            margin: 15px 0;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 0;
            right: 0;
            left: 0;
            background-color: #232F3E;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 12px;
        }
        h1, h2, h3 {
            color: #232F3E;
        }
        .comparison-table th {
            background-color: #E9ECEF;
        }
        .comparison-table td, .comparison-table th {
            padding: 8px;
            border: 1px solid #ddd;
        }
        .comparison-table tr:hover {
            background-color: #f5f5f5;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            grid-gap: 10px;
        }
        .metric-item {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }
        .step-card {
            border-left: 5px solid #FF9900;
            background-color: white;
            padding: 15px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .highlight {
            background-color: #FFEFDB;
            padding: 2px 5px;
            border-radius: 3px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for session management
    with st.sidebar:
        st.markdown("### Session Management")
        st.info(f"User ID: {st.session_state.user_id}")
        
        if st.button("🔄 Reset Session"):
            reset_session()
            st.rerun()
        
        st.divider()
        
        # Information about the application
        with st.expander("📚 About This App", expanded=False):
            st.markdown("""
                This interactive learning application demonstrates Amazon SageMaker
                Pipelines for building and managing ML workflows. Explore each step of the 
                model build process from data processing to deployment.
            """)
            
            # AWS learning resources
            st.subheader("Additional Resources")
            st.markdown("""
                - [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
                - [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
                - [AWS Machine Learning University](https://aws.amazon.com/machine-learning/mlu/)
            """)
        
    # Main app header
    st.title("Amazon SageMaker Pipelines: Model Build Workflow")
    st.markdown("""
    Learn about SageMaker Pipelines - a tool for building, automating, managing, and 
    scaling ML workflows on Amazon SageMaker.
    """)
    
    # Animation for the header
    lottie_url = "https://assets4.lottiefiles.com/packages/lf20_wzrpopkv.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=200, key="header_animation")
    
    # Tab-based navigation with emoji
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Overview", 
        "🧹 Data Processing", 
        "🤖 Model Training",
        "📊 Model Evaluation",
        "✅ Conditional Registration",
        "⚖️ Bias & Explainability"
    ])
    
    # OVERVIEW TAB
    with tab1:
        st.header("SageMaker Pipelines Overview")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Amazon SageMaker Pipelines is a purpose-built, easy-to-use continuous integration and 
            continuous delivery (CI/CD) service for machine learning (ML). With SageMaker Pipelines, 
            you can create, automate, and manage end-to-end ML workflows at scale.
            
            **Key benefits:**
            
            - **Orchestrate ML workflows** with an easy-to-use Python SDK
            - **Automate** different steps of the ML workflow
            - **Reuse workflow components** across projects
            - **Track lineage** of models from data preparation to deployment
            - **Version and audit** all aspects of the workflow
            """)
            
        with col2:
            st.image("https://d1.awsstatic.com/re19/Diagram_Amazon-SageMaker-Pipelines.42b2928acaa3ef5f8e39141ea06189873654a4d0.png",
                    caption="SageMaker Pipelines Architecture", use_container_width=True)
        
        # The complete pipeline visualization
        st.subheader("The Complete ML Workflow")
        
        # Get graph from session state
        pipeline_graph = st.session_state.pipeline_graph
        
        # Highlight path selector
        highlight_options = [
            "None",
            "Complete Workflow",
            "Data Processing Path",
            "Model Training & Evaluation",
            "Successful Model Path",
            "Rejected Model Path"
        ]
        
        selected_path = st.selectbox(
            "Highlight workflow path:", 
            highlight_options,
            index=0
        )
        
        # Set the highlight path based on selection
        if selected_path == "None":
            st.session_state.highlight_path = None
        elif selected_path == "Complete Workflow":
            st.session_state.highlight_path = ["data_input", "processing", "training", "evaluation", 
                                              "condition", "register_true", "clarify"]
        elif selected_path == "Data Processing Path":
            st.session_state.highlight_path = ["data_input", "processing"]
        elif selected_path == "Model Training & Evaluation":
            st.session_state.highlight_path = ["processing", "training", "evaluation"]
        elif selected_path == "Successful Model Path":
            st.session_state.highlight_path = ["evaluation", "condition", "register_true", "clarify"]
        elif selected_path == "Rejected Model Path":
            st.session_state.highlight_path = ["evaluation", "condition", "register_false"]
        
        # Draw the pipeline graph with the selected highlight path
        pipeline_fig = draw_pipeline_graph(pipeline_graph, st.session_state.highlight_path)
        st.pyplot(pipeline_fig, use_container_width=True)
        
        # Pipeline components explanation
        st.subheader("Pipeline Components")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Pipeline Steps
            
            - **Processing Step**: Data preprocessing and feature engineering
            - **Training Step**: Model training with hyperparameter configurations
            - **Evaluation Step**: Calculate model metrics on test data
            - **Condition Step**: Make decisions based on model metrics
            - **Register Model Step**: Add approved models to the model registry
            - **Clarify Step**: Bias detection and model explainability
            """)
        
        with col2:
            st.markdown("""
            ### Pipeline Properties
            
            - **Lineage Tracking**: Automatic model and data provenance
            - **Model Registry Integration**: Version and catalog models
            - **Conditional Execution**: Logic-based path selection
            - **Parameter Passing**: Share information between steps
            - **Error Handling**: Automatic retry and failure management
            - **Parallel Execution**: Run steps concurrently when possible
            """)
        
        # SageMaker Pipelines SDK
        st.subheader("SageMaker Pipelines SDK")
        
        st.markdown("""
        SageMaker Pipelines is built around a Python SDK that allows you to define pipelines
        as a series of interconnected steps. Here's a high-level example:
        """)
        
        st.code('''
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ConditionStep

# Define steps
preprocessing_step = ProcessingStep(...)
training_step = TrainingStep(...)
eval_step = ProcessingStep(...)
condition_step = ConditionStep(...)

# Create pipeline
pipeline = Pipeline(
    name="LoanModelPipeline",
    steps=[preprocessing_step, training_step, eval_step, condition_step]
)

# Submit pipeline definition
pipeline.upsert(role_arn=role)
execution = pipeline.start()
        ''', language='python')
        
        # Show complete pipeline code
        with st.expander("See the complete pipeline definition"):
            st.code(generate_sample_code("pipeline_definition"), language="python")
        
        # Key advantages section
        st.subheader("Key Advantages of SageMaker Pipelines")
        
        advantages = {
            "Advantage": [
                "Automation & Consistency", 
                "Reusability & Modularity", 
                "Visibility & Governance",
                "Scalability",
                "Conditional Processing",
                "Integration with SageMaker Features"
            ],
            "Description": [
                "Automatically execute all steps in your ML workflow in a consistent, repeatable manner.",
                "Build reusable components that can be shared across teams and projects.",
                "Track lineage of all artifacts and ensure governance with full auditability.",
                "Scale processing across multiple instances without managing infrastructure.",
                "Add conditional logic to create dynamic workflows based on results.",
                "Seamlessly integrate with other SageMaker features like Experiments, Model Registry, and Clarify."
            ]
        }
        
        advantages_df = pd.DataFrame(advantages)
        st.table(advantages_df)
    
    # DATA PROCESSING TAB
    with tab2:
        st.header("🧹 Data Processing")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Data processing is a critical first step in an ML pipeline. In this phase, you prepare raw data 
            for model training by cleaning, transforming, and splitting it into training, validation, and test sets.
            
            **SageMaker Processing provides:**
            
            - Managed data processing infrastructure
            - Custom code execution for preprocessing
            - Distributed processing across multiple instances
            - Integration with pipeline parameter passing
            """)
        
        with col2:
            st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/Processing-1.png",
                    caption="SageMaker Processing", use_container_width=True)
        
        # Data processing flow
        st.subheader("Data Processing Flow")
        
        # Create steps visualization 
        processing_steps = [
            {"number": "1", "name": "Data Ingestion", "description": "Load raw data from S3"},
            {"number": "2", "name": "Data Cleaning", "description": "Handle missing values and outliers"},
            {"number": "3", "name": "Feature Engineering", "description": "Transform features and create new ones"},
            {"number": "4", "name": "Data Splitting", "description": "Create train/validation/test splits"},
            {"number": "5", "name": "Data Export", "description": "Save processed datasets back to S3"}
        ]
        
        for step in processing_steps:
            st.markdown(f"""
            <div class="step-card">
                <h3>{step['number']}. {step['name']}</h3>
                <p>{step['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset exploration
        st.subheader("Dataset Exploration")
        
        dataset = st.session_state.dataset
        data_report = st.session_state.data_report
        
        # Show dataset sample
        with st.expander("View Dataset Sample", expanded=True):
            st.dataframe(dataset.head(5))
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", data_report["num_rows"])
        
        with col2:
            st.metric("Columns", data_report["num_columns"])
        
        with col3:
            st.metric("Target: Approval Rate", f"{dataset['loan_approved'].mean():.1%}")
        
        with col4:
            st.metric("Missing Values", sum(data_report["missing_values"].values()))
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Credit score distribution colored by outcome
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Split by outcome
            approved = dataset[dataset['loan_approved'] == 1]['credit_score']
            rejected = dataset[dataset['loan_approved'] == 0]['credit_score']
            
            # Plot histograms
            ax.hist(approved, bins=20, alpha=0.6, label='Approved', color='#59BA47')
            ax.hist(rejected, bins=20, alpha=0.6, label='Rejected', color='#D13212')
            
            ax.set_xlabel('Credit Score')
            ax.set_ylabel('Count')
            ax.set_title('Credit Score Distribution by Loan Approval')
            ax.legend()
            
            st.pyplot(fig)
        
        with col2:
            # Income distribution colored by outcome
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Split by outcome
            approved = dataset[dataset['loan_approved'] == 1]['income']
            rejected = dataset[dataset['loan_approved'] == 0]['income']
            
            # Plot histograms
            ax.hist(approved, bins=20, alpha=0.6, label='Approved', color='#59BA47')
            ax.hist(rejected, bins=20, alpha=0.6, label='Rejected', color='#D13212')
            
            ax.set_xlabel('Income')
            ax.set_ylabel('Count')
            ax.set_title('Income Distribution by Loan Approval')
            ax.legend()
            
            st.pyplot(fig)
        
        # Feature correlations
        st.subheader("Feature Correlations")
        
        # Create correlation matrix
        numeric_cols = ['age', 'income', 'education_years', 'employment_years', 
                      'debt_to_income', 'credit_score', 'loan_approved']
        corr_matrix = dataset[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        plt.title('Feature Correlation Matrix')
        st.pyplot(fig)
        
        # Feature transformations
        st.subheader("Feature Transformations")
        
        # Get transformation examples
        transformations = st.session_state.transformation_examples
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Data")
            st.dataframe(pd.DataFrame(transformations['original']).iloc[:, :5])
        
        with col2:
            st.markdown("#### Scaled Features")
            st.dataframe(pd.DataFrame(transformations['scaled']))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Label Encoded Gender")
            st.dataframe(pd.DataFrame({
                'Original': [d['gender'] for d in transformations['original']],
                'Encoded': transformations['encoded_gender']
            }))
        
        with col2:
            st.markdown("#### One-Hot Encoded Gender")
            st.dataframe(pd.DataFrame(transformations['one_hot_gender']))
        
        # Processing step code
        st.subheader("Processing Step in Pipeline")
        
        st.markdown("""
        In SageMaker Pipelines, you define a processing step using the `ProcessingStep` class. 
        This step can run custom processing code with frameworks like scikit-learn.
        """)
        
        st.code('''
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

# Define the processor
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1
)

# Define the processing step
preprocessing_step = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation")
    ],
    code="preprocessing.py"
)
        ''', language='python')
        
        # Show processing script
        with st.expander("View processing script (preprocessing.py)"):
            st.code(generate_sample_code("data_processing"), language='python')
        
        # Best practices
        st.subheader("Best Practices for Data Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Preprocessing Strategies
            
            - **Save preprocessors** for inference-time consistency
            - **Handle missing values** consistently in train and inference
            - **Normalize numerical features** to improve model convergence
            - **Create reusable transformers** for feature engineering
            - **Use feature stores** for sharing features across projects
            """)
        
        with col2:
            st.markdown("""
            ### Pipeline Integration Tips
            
            - **Output clear artifacts** with consistent naming
            - **Include data validation** to catch data drift
            - **Add data quality metrics** to track dataset changes
            - **Consider distributed processing** for large datasets
            - **Document transformations** for model explainability
            """)
    
    # MODEL TRAINING TAB
    with tab3:
        st.header("🤖 Model Training")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Model training is where your machine learning algorithm learns patterns from the
            processed training data. SageMaker Pipelines integrates seamlessly with SageMaker's
            training infrastructure for managed, distributed training.
            
            **Key benefits of SageMaker training:**
            
            - Managed infrastructure for model training
            - Support for all major ML frameworks
            - Distributed training across multiple instances
            - Integration with SageMaker Experiments
            - Automatic model artifacts storage
            """)
        
        with col2:
            st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/your-algorithms/sagemaker-algo.png",
                    caption="SageMaker Training", use_container_width=True)
        
        # Algorithm selection
        st.subheader("Algorithm Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Built-in Algorithms
            
            - **XGBoost**: Gradient boosting for tabular data
            - **Linear Learner**: Linear/logistic regression
            - **Random Cut Forest**: Anomaly detection
            - **Neural Topic Model**: Topic modeling
            - **Image Classification**: Computer vision
            - **Many others** for specific use cases
            """)
        
        with col2:
            st.markdown("""
            ### Framework Support
            
            - **TensorFlow**: Deep learning
            - **PyTorch**: Deep learning
            - **MXNet**: Deep learning
            - **scikit-learn**: Traditional ML
            - **HuggingFace**: NLP models
            - **R**: Statistical modeling
            """)
            
        with col3:
            st.markdown("""
            ### Custom Containers
            
            - **Extend existing frameworks** with custom logic
            - **Bring your own container** for complete control
            - **Use pre-built containers** from ECR
            - **Add custom dependencies** to standard frameworks
            - **Implement proprietary algorithms** or techniques
            """)
        
        # Training configuration visualization
        st.subheader("Training Configuration")
        
        # Show hyperparameter tuning options
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple hyperparameter UI
            st.markdown("#### XGBoost Hyperparameters")
            
            max_depth = st.slider("max_depth", 3, 10, 5)
            eta = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
            num_round = st.slider("num_round", 50, 1000, 100, 50)
            subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.1)
            
            # Show updated hyperparameter dict
            hyperparams = {
                "max_depth": max_depth,
                "eta": eta,
                "num_round": num_round,
                "subsample": subsample,
                "objective": "binary:logistic",
                "eval_metric": "logloss"
            }
            
            st.json(hyperparams)
        
        with col2:
            # Instance configuration
            st.markdown("#### Training Infrastructure")
            
            instance_type = st.selectbox(
                "Training Instance Type",
                ["ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.c5.xlarge", "ml.g4dn.xlarge"]
            )
            
            instance_count = st.slider("Instance Count", 1, 5, 1)
            
            # Additional options
            use_spot = st.checkbox("Use Spot Instances", True)
            max_wait_time = st.slider("Max Wait Time (seconds)", 3600, 14400, 7200, 1800)
            
            # Calculate cost estimate
            hourly_rates = {
                "ml.m5.large": 0.115,
                "ml.m5.xlarge": 0.23,
                "ml.m5.2xlarge": 0.46,
                "ml.c5.xlarge": 0.19,
                "ml.g4dn.xlarge": 0.736
            }
            
            # Apply spot discount if selected
            rate = hourly_rates[instance_type]
            if use_spot:
                rate *= 0.3  # 70% discount for spot
            
            # Estimate time to train in hours (simplified)
            est_train_time = 0.5 * (num_round / 100) * (1 / instance_count)
            
            # Calculate cost
            est_cost = rate * instance_count * est_train_time
            
            # Display cost estimate
            st.metric("Estimated Training Cost", f"${est_cost:.2f}")
        
        # Learning curves visualization
        st.subheader("Training Progress Monitoring")
        
        # Generate simulated training curves
        epochs = 100
        x = np.arange(1, epochs + 1)
        
        # Training loss curve
        train_loss = 0.7 * np.exp(-0.02 * x) + 0.1 + np.random.normal(0, 0.02, epochs)
        
        # Validation loss curve (higher than training, showing some overfitting)
        val_loss = 0.7 * np.exp(-0.015 * x) + 0.2 + np.random.normal(0, 0.03, epochs)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training and validation loss
        ax.plot(x, train_loss, label='Training Loss', color='#00A1C9')
        ax.plot(x, val_loss, label='Validation Loss', color='#FF9900')
        
        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss Curves')
        
        # Add a legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Show the plot
        st.pyplot(fig)
        
        # Training step in pipeline
        st.subheader("Training Step in Pipeline")
        
        st.markdown("""
        In SageMaker Pipelines, the `TrainingStep` class is used to define and configure the
        model training process. This step manages the training job execution and tracks metrics.
        """)
        
        st.code('''
from sagemaker.xgboost import XGBoost
from sagemaker.workflow.steps import TrainingStep

# Define the XGBoost estimator
xgb_estimator = XGBoost(
    entry_point="train.py",
    framework_version="1.3-1",
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100,
        "max_depth": 5,
        "eta": 0.2,
        "subsample": 0.8
    },
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"{output_data_prefix}/model"
)

# Define the training step
training_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)
        ''', language='python')
        
        # Show training script
        with st.expander("View training script (train.py)"):
            st.code(generate_sample_code("model_training"), language='python')
        
        # Integration with Experiments
        st.subheader("Integration with SageMaker Experiments")
        
        st.markdown("""
        SageMaker Pipelines integrates with SageMaker Experiments to automatically
        track training runs, parameters, and metrics. This provides:
        
        - Automatic experiment tracking for each pipeline execution
        - Hyperparameter and metric logging
        - Run comparison across pipeline iterations
        - Lineage tracking from data to model artifacts
        
        You can view and analyze experiments in SageMaker Studio:
        """)
        
        st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/studio/studio-experiments.png",
                caption="SageMaker Experiments in Studio", use_container_width=True)
        
        # Best practices
        st.subheader("Best Practices for Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Training Optimization
            
            - **Use appropriate instance types** for your algorithm
            - **Enable early stopping** to prevent overfitting
            - **Implement spot training** to reduce costs
            - **Optimize algorithm parameters** for your dataset
            - **Scale distributed training** for large datasets
            """)
        
        with col2:
            st.markdown("""
            ### Pipeline Integration Tips
            
            - **Track hyperparameters** explicitly as pipeline parameters
            - **Add validation datasets** to assess model during training
            - **Store evaluation metrics** for condition steps
            - **Use experiment tracking** to compare runs
            - **Implement checkpointing** for spot instances
            """)
    
    # MODEL EVALUATION TAB
    with tab4:
        st.header("📊 Model Evaluation")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Model evaluation is a critical step that assesses the performance of the trained model
            on unseen data. In SageMaker Pipelines, this is typically implemented as a processing
            step that runs after training to calculate metrics.
            
            **Key aspects of model evaluation:**
            
            - Performance metrics calculation
            - Confusion matrix and error analysis
            - Metric storage for conditional steps
            - Comparison with baseline models
            - Visualization of results
            """)
        
        with col2:
            st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/model-monitor/model-monitor-model-quality-how-it-works.png",
                    caption="Model Evaluation Process", use_container_width=True)
        
        # Model metrics
        st.subheader("Performance Metrics")
        
        # Display evaluation metrics
        metrics = st.session_state.model_metrics
        
        # Create a metrics dashboard
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # metric_val = float(metrics[metrics['Metric'] == 'Accuracy']['Value'])
            # threshold = float(metrics[metrics['Metric'] == 'Accuracy']['Threshold'])
            accuracy_metrics = metrics[metrics['Metric'] == 'Accuracy']
            if not accuracy_metrics.empty:
                metric_val = float(accuracy_metrics['Value'].iloc[0])
                threshold = float(accuracy_metrics['Threshold'].iloc[0])
            else:
                raise ValueError("No accuracy metrics found in the data")
            
            st.metric("Accuracy", f"{metric_val:.2f}", f"{metric_val-threshold:+.2f}")
        
        with col2:
            metric_val = float(metrics[metrics['Metric'] == 'Precision']['Value'].iloc[0])
            threshold = float(metrics[metrics['Metric'] == 'Precision']['Threshold'].iloc[0])
            st.metric("Precision", f"{metric_val:.2f}", f"{metric_val-threshold:+.2f}")
            
        with col3:
            metric_val = float(metrics[metrics['Metric'] == 'Recall']['Value'].iloc[0])
            threshold = float(metrics[metrics['Metric'] == 'Recall']['Threshold'].iloc[0])
            st.metric("Recall", f"{metric_val:.2f}", f"{metric_val-threshold:+.2f}")
            
        with col4:
            metric_val = float(metrics[metrics['Metric'] == 'F1 Score']['Value'].iloc[0])
            threshold = float(metrics[metrics['Metric'] == 'F1 Score']['Threshold'].iloc[0])
            st.metric("F1 Score", f"{metric_val:.2f}", f"{metric_val-threshold:+.2f}")
            
        with col5:
            metric_val = float(metrics[metrics['Metric'] == 'ROC AUC']['Value'].iloc[0])
            threshold = float(metrics[metrics['Metric'] == 'ROC AUC']['Threshold'].iloc[0])
            st.metric("ROC AUC", f"{metric_val:.2f}", f"{metric_val-threshold:+.2f}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        conf_matrix_fig = create_confusion_matrix()
        st.pyplot(conf_matrix_fig)
        
        # ROC curve
        st.subheader("ROC Curve")
        
        # Create simulated ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-3 * fpr)  # Simulated ROC curve shape
        
        # Add some noise to make it look more realistic
        tpr = np.minimum(tpr + np.random.normal(0, 0.03, len(tpr)), 1)
        
        # Calculate AUC
        auc = metrics[metrics['Metric'] == 'ROC AUC']['Value'].values[0]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color='#FF9900', linewidth=2)
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        # Add labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')
        
        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Show the plot
        st.pyplot(fig)
        
        # Model comparison
        st.subheader("Model Version Comparison")
        
        # Display model comparison table
        model_comp = st.session_state.model_comparison
        
        # Apply formatting and colors to table
        st.dataframe(model_comp, use_container_width=True)
        
        # Evaluation step in pipeline
        st.subheader("Evaluation Step in Pipeline")
        
        st.markdown("""
        In SageMaker Pipelines, model evaluation is typically implemented as a processing step
        that runs after training to calculate performance metrics.
        """)
        
        st.code('''
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile

# Define evaluation processor
evaluation_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1
)

# Define evaluation step
eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    code="evaluate.py"
)

# Define model evaluation property file for accessing metrics
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)
        ''', language='python')
        
        # Show evaluation script
        with st.expander("View evaluation script (evaluate.py)"):
            st.code(generate_sample_code("model_evaluation"), language='python')
        
        # Best practices
        st.subheader("Best Practices for Model Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Metric Selection
            
            - **Choose metrics appropriate to your problem** (classification vs regression)
            - **Consider business impact** of false positives vs false negatives
            - **Set thresholds based on requirements**, not arbitrary values
            - **Use multiple metrics** for comprehensive evaluation
            - **Evaluate on representative test data**
            """)
        
        with col2:
            st.markdown("""
            ### Pipeline Integration Tips
            
            - **Output metrics in consistent JSON format** for condition steps
            - **Save visualizations** for easier analysis
            - **Track metrics across versions** for comparison
            - **Consider specialized evaluation** for your domain
            - **Add failure handling** for robustness
            """)
    
    # CONDITIONAL MODEL REGISTRATION TAB
    with tab5:
        st.header("✅ Conditional Model Registration")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Conditional registration is a powerful feature of SageMaker Pipelines that lets you
            automatically decide whether a model should be registered in the SageMaker Model Registry
            based on its evaluation metrics. This ensures only models that meet quality thresholds
            enter your deployment process.
            
            **Key benefits:**
            
            - Automated quality gating for models
            - Prevention of low-quality model deployment
            - Integration with model registry versioning
            - Control of model approval workflow
            - Seamless CI/CD integration
            """)
        
        with col2:
            st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/model_registry_block_diagram.png",
                    caption="SageMaker Model Registry", use_container_width=True)
        
        # Condition step explanation
        st.subheader("How Conditional Steps Work")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### Condition Components
            
            1. **Condition expression**: A logical comparison like "accuracy > 0.85"
            
            2. **Property access**: Reference to metrics from previous steps
            
            3. **If steps**: Actions to take when condition is true
            
            4. **Else steps**: Optional actions when condition is false
            
            5. **PropertyFile**: Helper to access values from evaluation outputs
            """)
        
        with col2:
            # Create visualization of condition flow
            cond_fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Draw condition node
            condition = plt.Rectangle((4, 5), 2, 2, fc='#FFDB58', ec='#000000')
            ax.add_patch(condition)
            ax.text(5, 6, "Condition\naccuracy >= 0.85", ha='center', va='center')
            
            # Draw true/false paths
            ax.arrow(6, 6, 2, 1, head_width=0.3, head_length=0.3, fc='#59BA47', ec='#59BA47')
            ax.arrow(4, 5, -2, -1, head_width=0.3, head_length=0.3, fc='#D13212', ec='#D13212')
            
            # Draw result nodes
            register = plt.Rectangle((8, 6.5), 2, 1, fc='#59BA47', ec='#000000')
            ax.add_patch(register)
            ax.text(9, 7, "Register Model", ha='center', va='center')
            
            skip = plt.Rectangle((1, 3.5), 2, 1, fc='#D13212', ec='#000000')
            ax.add_patch(skip)
            ax.text(2, 4, "Skip Registration", ha='center', va='center')
            
            # Add true/false labels
            ax.text(7, 7, "True", ha='center', va='center')
            ax.text(3, 4, "False", ha='center', va='center')
            
            st.pyplot(cond_fig)
        
        # Condition configuration
        st.subheader("Condition Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Metric Threshold Settings")
            
            acc_threshold = st.slider("Accuracy Threshold", 0.75, 0.95, 0.85, 0.01)
            
            # Create a gauge chart showing current model vs threshold
            current_accuracy = 0.87  # From model metrics
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_accuracy,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Model Accuracy"},
                gauge = {
                    'axis': {'range': [0.75, 0.95]},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': acc_threshold
                    },
                    'steps': [
                        {'range': [0.75, acc_threshold], 'color': "#FFEFDB"},
                        {'range': [acc_threshold, 0.95], 'color': "#E6F2FF"}
                    ],
                    'bar': {'color': '#FF9900'}
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Registration decision
            if current_accuracy >= acc_threshold:
                st.success(f"✅ Model PASSES threshold ({acc_threshold:.2f}) and will be registered")
            else:
                st.error(f"❌ Model FAILS threshold ({acc_threshold:.2f}) and will NOT be registered")
        
        with col2:
            st.markdown("#### Approval Status Options")
            
            approval_status = st.radio(
                "Model Approval Status",
                ["PendingManualApproval", "Approved", "Rejected"]
            )
            
            st.markdown("""
            **Approval Workflow:**
            
            1. **PendingManualApproval**: Requires manual review before deployment
            2. **Approved**: Ready for automated deployment
            3. **Rejected**: Should not be deployed
            
            You can use custom logic to determine the initial approval status, such as:
            - Auto-approve models above a very high threshold
            - Always require review for critical applications
            - Reject models with poor performance on specific metrics
            """)
        
        # Conditional step in pipeline
        st.subheader("Conditional Step in Pipeline")
        
        st.markdown("""
        In SageMaker Pipelines, conditional registration is implemented using `ConditionStep` that
        evaluates a condition and executes the appropriate next steps.
        """)
        
        st.code(generate_sample_code("conditional_registration"), language='python')
        
        # Model registry integration
        st.subheader("Model Registry Integration")
        
        st.markdown("""
        The SageMaker Model Registry provides a centralized repository for versioning and tracking models:
        
        - **Model Package Groups**: Collections of related model versions
        - **Model Packages**: Individual model versions with metadata
        - **Approval Status**: Track which models are approved for deployment
        - **Deployment History**: Record of where models have been deployed
        - **Lifecycle Automation**: Connect model registry events to AWS Lambda
        
        Pipeline integration with Model Registry enables a complete CI/CD workflow from training to deployment.
        """)
        
        # Show model registry browser UI
        st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/model_registry/model_package_group_detail_page.PNG",
                caption="SageMaker Model Registry Browser", use_container_width=True)
        
        # Best practices
        st.subheader("Best Practices for Conditional Registration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Condition Configuration
            
            - **Set clear quality thresholds** based on business requirements
            - **Use multiple conditions** for comprehensive quality gates
            - **Consider confidence intervals**, not just point estimates
            - **Implement baseline comparisons** to ensure models improve
            - **Add notification mechanisms** for rejected models
            """)
        
        with col2:
            st.markdown("""
            ### Model Registry Management
            
            - **Establish a naming convention** for model package groups
            - **Include model metadata** for easier searching and filtering
            - **Implement multi-stage approval** for critical models
            - **Set up automatic deployment** for approved models
            - **Define model lifecycle policies** for archiving old versions
            """)
    
    # BIAS AND EXPLAINABILITY TAB
    with tab6:
        st.header("⚖️ Bias & Explainability")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Bias detection and model explainability are essential components of responsible AI.
            SageMaker Clarify integrates with Pipelines to automatically analyze models for potential
            biases and provide explanations for their predictions.
            
            **Key capabilities:**
            
            - **Bias detection**: Pre-training and post-training bias metrics
            - **Feature importance**: SHAP value calculation for model explanations
            - **Visual reports**: Interactive visualizations of bias and explanations
            - **Pipeline integration**: Automated checks as part of model building
            - **Continuous monitoring**: Ongoing bias and drift detection
            """)
        
        with col2:
            st.image("https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/clarify/clarify-pipeline.png",
                    caption="SageMaker Clarify in Pipelines", use_container_width=True)
        
        # Bias detection
        st.subheader("Bias Detection")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            SageMaker Clarify can detect both pre-training and post-training bias in your data and models.
            It provides several metrics for measuring different aspects of bias:
            
            - **Class Imbalance (CI)**: Measures imbalance in outcome distribution
            - **Difference in Positive Proportions (DPP)**: Measures statistical parity
            - **Disparate Impact (DI)**: Measures ratio of positive outcomes
            - **Accuracy Difference**: Compares model accuracy across groups
            - **Recall Difference**: Compares model recall across groups
            - **And many more specialized metrics**
            """)
            
            # Show bias metrics table
            bias_metrics = st.session_state.bias_metrics
            
            # Format the table - add red or green styling based on threshold
            def style_status(val):
                if val == 'Warning':
                    return 'color: red; font-weight: bold'
                else:
                    return 'color: green; font-weight: bold'
                    
            st.dataframe(bias_metrics.style.map(style_status, subset=['Status']))
        
        with col2:
            # Show bias visualization
            st.markdown("#### Approval Rate by Gender")
            
            bias_chart = create_bias_visualization()
            st.altair_chart(bias_chart, use_container_width=True)
            
            st.markdown("""
            This chart shows a potential bias in loan approvals, with women receiving 
            approvals at a lower rate than men (60% vs. 72%). Further investigation 
            would be needed to determine if this disparity is due to legitimate factors
            or represents unfair bias.
            """)
        
        # Model explainability
        st.subheader("Model Explainability")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            SageMaker Clarify uses SHAP (SHapley Additive exPlanations) to explain model predictions.
            SHAP values represent each feature's contribution to a specific prediction, helping you
            understand how models make decisions.
            
            **Benefits of model explainability:**
            
            - **Transparency**: Understand why models make certain predictions
            - **Fairness verification**: Ensure models don't rely on sensitive attributes
            - **Regulatory compliance**: Support documentation for model governance
            - **Model improvement**: Identify and address problematic patterns
            - **Trust building**: Provide stakeholders with understandable explanations
            """)
            
            # Show global feature importance chart
            st.markdown("#### Global Feature Importance")
            
            feature_imp_chart = create_feature_importance_chart()
            st.altair_chart(feature_imp_chart, use_container_width=True)
        
        with col2:
            st.markdown("#### Individual Prediction Explanation")
            
            # Sample customer for explanation
            customer = {
                "age": 42,
                "income": 85000,
                "education_years": 16,
                "employment_years": 12,
                "debt_to_income": 0.38,
                "credit_score": 710,
                "has_mortgage": 1,
                "gender": "Female"
            }
            
            # Display customer details
            st.json(customer)
            
            # Create SHAP waterfall chart for the customer
            fig = go.Figure(go.Waterfall(
                name="Prediction explanation", 
                orientation="h",
                measure=["relative", "relative", "relative", "relative", 
                        "relative", "relative", "relative", "relative", "total"],
                y=["credit_score", "income", "debt_to_income", "employment_years", 
                  "education_years", "age", "has_mortgage", "gender", "prediction"],
                x=[0.15, 0.12, -0.08, 0.05, 0.03, 0.01, 0.02, -0.01, 0.79],
                text=["0.15", "0.12", "-0.08", "0.05", "0.03", "0.01", "0.02", "-0.01", "0.79"],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#D13212"}},
                increasing={"marker": {"color": "#59BA47"}},
                totals={"marker": {"color": "#FF9900"}}
            ))
            
            fig.update_layout(
                title="SHAP Values for Individual Prediction",
                waterfallgap=0.2,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Clarify integration in pipeline
        st.subheader("Clarify Steps in Pipeline")
        
        st.markdown("""
        SageMaker Pipelines integrates with SageMaker Clarify through dedicated `ClarifyCheckStep` 
        components for bias detection and explainability analysis.
        """)
        
        st.code(generate_sample_code("bias_and_explainability"), language='python')
        
        # Best practices
        st.subheader("Best Practices for Bias & Explainability")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Bias Mitigation
            
            - **Identify sensitive attributes** early in the process
            - **Measure bias with multiple metrics** for comprehensive view
            - **Compare pre- and post-training bias** to understand model impact
            - **Establish bias thresholds** for acceptable model behavior
            - **Consider bias-aware algorithms** or preprocessing techniques
            - **Implement ongoing bias monitoring** in production
            """)
        
        with col2:
            st.markdown("""
            ### Explainability Enhancement
            
            - **Choose inherently interpretable models** when possible
            - **Generate global and local explanations** for complete understanding
            - **Validate explanations** against domain knowledge
            - **Use explanations for feature engineering** guidance
            - **Document explanations** for model governance
            - **Present explanations in business terms** to stakeholders
            """)
        
        # Integration with model monitoring
        st.subheader("Ongoing Monitoring for Bias & Drift")
        
        st.markdown("""
        After deployment, SageMaker Model Monitor can continuously check for:
        
        - **Data quality drift**: Changes in feature distributions
        - **Model quality drift**: Degradation in model performance
        - **Bias drift**: Changes in bias metrics over time
        - **Feature attribution drift**: Changes in feature importance
        
        This creates a complete cycle of responsible AI - from development to deployment to monitoring.
        """)
        
        st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/05/20/ML-4709-image001-new.jpg",
                caption="Continuous Monitoring for ML Models", use_container_width=True)

    # Add footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    main()
