
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import json
import time
import uuid
from datetime import datetime, timedelta
import random
import networkx as nx
from PIL import Image
import io
import base64
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import plotly.express as px


def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
        
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "cv_model"
        
    if 'selected_instance_family' not in st.session_state:
        st.session_state.selected_instance_family = "ml.g4dn"
        
    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = generate_benchmark_results()
        
    if 'instance_details' not in st.session_state:
        st.session_state.instance_details = generate_instance_details()
        
    if 'model_details' not in st.session_state:
        st.session_state.model_details = generate_model_details()
        
    if 'recommendation_results' not in st.session_state:
        st.session_state.recommendation_results = None


def reset_session():
    """
    Reset the session state
    """
    # Keep only user_id and reset all other state
    user_id = st.session_state.user_id
    st.session_state.clear()
    st.session_state.user_id = user_id


# Data generation functions
def generate_instance_details():
    """
    Generate details about SageMaker instance types
    """
    instances = {
        "ml.t2": {
            "name": "T2 - Low-cost CPU",
            "description": "Burstable performance instances for development and low-intensity workloads",
            "use_cases": ["Development", "Testing", "Small batch transform jobs"],
            "types": {
                "ml.t2.medium": {"vCPU": 2, "Memory": "4 GiB", "network": "Low to Moderate"},
                "ml.t2.large": {"vCPU": 2, "Memory": "8 GiB", "network": "Low to Moderate"},
                "ml.t2.xlarge": {"vCPU": 4, "Memory": "16 GiB", "network": "Moderate"},
                "ml.t2.2xlarge": {"vCPU": 8, "Memory": "32 GiB", "network": "Moderate"}
            },
            "good_for": ["Low-traffic endpoints", "Development environments", "Cost optimization"],
            "limitations": ["Limited burst performance", "No GPU acceleration", "Not for high-throughput workloads"],
            "cost_range": "$0.05 - $0.40 per hour",
            "icon": "cpu"
        },
        "ml.c5": {
            "name": "C5 - Compute optimized",
            "description": "Compute-optimized instances for compute-intensive workloads",
            "use_cases": ["High-performance inference", "Batch processing", "CPU-optimized models"],
            "types": {
                "ml.c5.large": {"vCPU": 2, "Memory": "4 GiB", "network": "Up to 10 Gbps"},
                "ml.c5.xlarge": {"vCPU": 4, "Memory": "8 GiB", "network": "Up to 10 Gbps"},
                "ml.c5.2xlarge": {"vCPU": 8, "Memory": "16 GiB", "network": "Up to 10 Gbps"},
                "ml.c5.4xlarge": {"vCPU": 16, "Memory": "32 GiB", "network": "Up to 10 Gbps"},
                "ml.c5.9xlarge": {"vCPU": 36, "Memory": "72 GiB", "network": "10 Gbps"},
                "ml.c5.18xlarge": {"vCPU": 72, "Memory": "144 GiB", "network": "25 Gbps"}
            },
            "good_for": ["CPU-intensive operations", "High throughput requirements", "Cost-effective compute"],
            "limitations": ["No GPU acceleration", "Limited for memory-intensive workloads"],
            "cost_range": "$0.10 - $3.40 per hour",
            "icon": "cpu"
        },
        "ml.g4dn": {
            "name": "G4DN - GPU accelerated compute",
            "description": "GPU-based instances for graphics-intensive applications and machine learning inference",
            "use_cases": ["Deep learning inference", "Computer vision", "NLP models"],
            "types": {
                "ml.g4dn.xlarge": {"vCPU": 4, "Memory": "16 GiB", "GPU": "1 NVIDIA T4", "GPU Memory": "16 GiB"},
                "ml.g4dn.2xlarge": {"vCPU": 8, "Memory": "32 GiB", "GPU": "1 NVIDIA T4", "GPU Memory": "16 GiB"},
                "ml.g4dn.4xlarge": {"vCPU": 16, "Memory": "64 GiB", "GPU": "1 NVIDIA T4", "GPU Memory": "16 GiB"},
                "ml.g4dn.8xlarge": {"vCPU": 32, "Memory": "128 GiB", "GPU": "1 NVIDIA T4", "GPU Memory": "16 GiB"},
                "ml.g4dn.12xlarge": {"vCPU": 48, "Memory": "192 GiB", "GPU": "4 NVIDIA T4", "GPU Memory": "64 GiB"},
                "ml.g4dn.16xlarge": {"vCPU": 64, "Memory": "256 GiB", "GPU": "1 NVIDIA T4", "GPU Memory": "16 GiB"}
            },
            "good_for": ["GPU-accelerated inference", "Real-time computer vision", "Deep learning models"],
            "limitations": ["Higher cost than CPU instances", "May be over-provisioned for simpler models"],
            "cost_range": "$0.75 - $6.00 per hour",
            "icon": "gpu"
        },
        "ml.inf1": {
            "name": "Inf1 - AWS Inferentia",
            "description": "Instances for machine learning inference featuring AWS Inferentia chips",
            "use_cases": ["High-performance deep learning inference", "Cost-optimized ML inference", "Production deployments"],
            "types": {
                "ml.inf1.xlarge": {"vCPU": 4, "Memory": "8 GiB", "Inferentia": "1 chip", "network": "Up to 25 Gbps"},
                "ml.inf1.2xlarge": {"vCPU": 8, "Memory": "16 GiB", "Inferentia": "1 chip", "network": "Up to 25 Gbps"},
                "ml.inf1.6xlarge": {"vCPU": 24, "Memory": "48 GiB", "Inferentia": "4 chips", "network": "25 Gbps"},
                "ml.inf1.24xlarge": {"vCPU": 96, "Memory": "192 GiB", "Inferentia": "16 chips", "network": "100 Gbps"}
            },
            "good_for": ["High throughput inference", "Cost-optimized TensorFlow/PyTorch", "Optimizing inference costs"],
            "limitations": ["Requires model compilation with AWS Neuron", "Limited flexibility compared to GPUs"],
            "cost_range": "$0.40 - $5.00 per hour",
            "icon": "inferentia"
        },
        "ml.r5": {
            "name": "R5 - Memory optimized",
            "description": "Memory-optimized instances for memory-intensive applications",
            "use_cases": ["Large model hosting", "Memory-intensive preprocessing", "In-memory analytics"],
            "types": {
                "ml.r5.large": {"vCPU": 2, "Memory": "16 GiB", "network": "Up to 10 Gbps"},
                "ml.r5.xlarge": {"vCPU": 4, "Memory": "32 GiB", "network": "Up to 10 Gbps"},
                "ml.r5.2xlarge": {"vCPU": 8, "Memory": "64 GiB", "network": "Up to 10 Gbps"},
                "ml.r5.4xlarge": {"vCPU": 16, "Memory": "128 GiB", "network": "Up to 10 Gbps"},
                "ml.r5.12xlarge": {"vCPU": 48, "Memory": "384 GiB", "network": "12 Gbps"},
                "ml.r5.24xlarge": {"vCPU": 96, "Memory": "768 GiB", "network": "25 Gbps"}
            },
            "good_for": ["Memory-intensive models", "Large embeddings", "NLP models with large vocabularies"],
            "limitations": ["No GPU acceleration", "Higher cost for compute-bound workloads"],
            "cost_range": "$0.15 - $7.00 per hour",
            "icon": "cpu"
        },
        "ml.m5": {
            "name": "M5 - General purpose",
            "description": "General purpose instances providing a balance of compute, memory, and network resources",
            "use_cases": ["General inference workloads", "Diverse model types", "Balanced resource needs"],
            "types": {
                "ml.m5.large": {"vCPU": 2, "Memory": "8 GiB", "network": "Up to 10 Gbps"},
                "ml.m5.xlarge": {"vCPU": 4, "Memory": "16 GiB", "network": "Up to 10 Gbps"},
                "ml.m5.2xlarge": {"vCPU": 8, "Memory": "32 GiB", "network": "Up to 10 Gbps"},
                "ml.m5.4xlarge": {"vCPU": 16, "Memory": "64 GiB", "network": "Up to 10 Gbps"},
                "ml.m5.12xlarge": {"vCPU": 48, "Memory": "192 GiB", "network": "12 Gbps"},
                "ml.m5.24xlarge": {"vCPU": 96, "Memory": "384 GiB", "network": "25 Gbps"}
            },
            "good_for": ["Versatile deployment", "Balanced workloads", "Cost-effective general purpose"],
            "limitations": ["Not optimized for specific workload types", "No specialized hardware"],
            "cost_range": "$0.10 - $5.00 per hour",
            "icon": "cpu"
        },
        "ml.p4d": {
            "name": "P4d - Accelerated GPU compute",
            "description": "High-performance computing instances with NVIDIA A100 GPUs for machine learning",
            "use_cases": ["High-performance inference", "Complex deep learning models", "Multi-model serving"],
            "types": {
                "ml.p4d.24xlarge": {
                    "vCPU": 96, 
                    "Memory": "1152 GiB", 
                    "GPU": "8 NVIDIA A100", 
                    "GPU Memory": "320 GiB (40 GiB each)",
                    "network": "400 Gbps"
                }
            },
            "good_for": ["Highest performance needs", "Large model inference", "Multi-model inference"],
            "limitations": ["Highest cost tier", "May be significantly over-provisioned for simple models"],
            "cost_range": "$30.00 - $35.00 per hour",
            "icon": "gpu"
        },
        "ml.trn1": {
            "name": "Trn1 - AWS Trainium",
            "description": "Instances powered by AWS Trainium chips for high-performance training and inference",
            "use_cases": ["Deep learning training", "Cost-optimized deep learning", "Large language models"],
            "types": {
                "ml.trn1.2xlarge": {"vCPU": 8, "Memory": "32 GiB", "Trainium": "1 chip", "network": "Up to 12.5 Gbps"},
                "ml.trn1.32xlarge": {"vCPU": 128, "Memory": "512 GiB", "Trainium": "16 chips", "network": "800 Gbps"}
            },
            "good_for": ["Cost-effective deep learning", "NLP training and inference", "Transfer learning"],
            "limitations": ["Requires model compilation with AWS Neuron SDK", "Currently supports certain frameworks"],
            "cost_range": "$1.50 - $25.00 per hour",
            "icon": "trainium"
        },
        "ml.inf2": {
            "name": "Inf2 - AWS Inferentia2",
            "description": "Second-generation machine learning inference instances with AWS Inferentia2 chips",
            "use_cases": ["Advanced deep learning inference", "Large language models", "Transformer-based architectures"],
            "types": {
                "ml.inf2.xlarge": {"vCPU": 4, "Memory": "16 GiB", "Inferentia": "1 chip", "network": "Up to 25 Gbps"},
                "ml.inf2.8xlarge": {"vCPU": 32, "Memory": "128 GiB", "Inferentia": "1 chip", "network": "50 Gbps"},
                "ml.inf2.24xlarge": {"vCPU": 96, "Memory": "384 GiB", "Inferentia": "6 chips", "network": "100 Gbps"},
                "ml.inf2.48xlarge": {"vCPU": 192, "Memory": "768 GiB", "Inferentia": "12 chips", "network": "200 Gbps"}
            },
            "good_for": ["High-performance requirements", "Cost-efficient LLM inference", "Advanced ML applications"],
            "limitations": ["Requires model compilation with AWS Neuron", "Higher cost than first-generation"],
            "cost_range": "$1.00 - $18.00 per hour",
            "icon": "inferentia"
        }
    }
    
    return instances


def generate_model_details():
    """
    Generate example model details for demonstration
    """
    models = {
        "cv_model": {
            "name": "Computer Vision Model (ResNet-50)",
            "description": "Image classification model based on ResNet-50 architecture",
            "framework": "PyTorch",
            "type": "Computer Vision",
            "size": "98 MB",
            "latency_sensitivity": "Medium",
            "io_bound": False,
            "compute_bound": True,
            "memory_bound": False,
            "input_tensor_shape": "(1, 3, 224, 224)",
            "output_shape": "(1, 1000)",
            "recommended_instances": ["ml.g4dn.xlarge", "ml.inf1.xlarge", "ml.c5.2xlarge"],
            "traffic_pattern": "Spiky, with peak periods",
            "endpoint_type_recommendations": ["Real-time endpoint", "Serverless Inference"],
            "performance_metrics": {
                "latency_importance": 80,
                "throughput_importance": 60,
                "cost_importance": 50
            }
        },
        "nlp_model": {
            "name": "Text Classification Model (BERT-base)",
            "description": "BERT-based text classification model fine-tuned for sentiment analysis",
            "framework": "TensorFlow",
            "type": "Natural Language Processing",
            "size": "438 MB",
            "latency_sensitivity": "Medium-High",
            "io_bound": False,
            "compute_bound": True,
            "memory_bound": True,
            "input_tensor_shape": "Variable sequence length",
            "output_shape": "(1, 2)",
            "recommended_instances": ["ml.g4dn.2xlarge", "ml.inf1.2xlarge", "ml.c5.4xlarge"],
            "traffic_pattern": "Consistent throughout day",
            "endpoint_type_recommendations": ["Real-time endpoint", "Serverless Inference"],
            "performance_metrics": {
                "latency_importance": 70,
                "throughput_importance": 50,
                "cost_importance": 60
            }
        },
        "tabular_model": {
            "name": "Gradient Boosting Model (XGBoost)",
            "description": "XGBoost model for tabular data prediction",
            "framework": "XGBoost",
            "type": "Tabular",
            "size": "12 MB",
            "latency_sensitivity": "High",
            "io_bound": True,
            "compute_bound": False,
            "memory_bound": False,
            "input_tensor_shape": "Variable features",
            "output_shape": "(1,)",
            "recommended_instances": ["ml.c5.xlarge", "ml.m5.xlarge", "ml.t2.xlarge"],
            "traffic_pattern": "High during business hours",
            "endpoint_type_recommendations": ["Real-time endpoint", "Serverless Inference"],
            "performance_metrics": {
                "latency_importance": 90,
                "throughput_importance": 40,
                "cost_importance": 70
            }
        },
        "large_nlp_model": {
            "name": "Large Language Model (FLAN-T5 XL)",
            "description": "Large language model for text generation and comprehension",
            "framework": "PyTorch",
            "type": "Natural Language Processing",
            "size": "9.8 GB",
            "latency_sensitivity": "Low",
            "io_bound": False,
            "compute_bound": True,
            "memory_bound": True,
            "input_tensor_shape": "Variable sequence length",
            "output_shape": "Variable sequence length",
            "recommended_instances": ["ml.g5.12xlarge", "ml.inf2.8xlarge", "ml.p4d.24xlarge"],
            "traffic_pattern": "Moderate, consistent",
            "endpoint_type_recommendations": ["Real-time endpoint", "Asynchronous Inference"],
            "performance_metrics": {
                "latency_importance": 40,
                "throughput_importance": 70,
                "cost_importance": 60
            }
        },
        "recommender_model": {
            "name": "Recommendation Engine (DeepFM)",
            "description": "Deep factorization machine for personalized recommendations",
            "framework": "TensorFlow",
            "type": "Recommendation",
            "size": "256 MB",
            "latency_sensitivity": "High",
            "io_bound": True,
            "compute_bound": True,
            "memory_bound": True,
            "input_tensor_shape": "Variable features",
            "output_shape": "Variable (ranked list)",
            "recommended_instances": ["ml.g4dn.4xlarge", "ml.r5.2xlarge", "ml.c5.4xlarge"],
            "traffic_pattern": "Spiky, high during peak shopping hours",
            "endpoint_type_recommendations": ["Real-time endpoint"],
            "performance_metrics": {
                "latency_importance": 85,
                "throughput_importance": 75,
                "cost_importance": 50
            }
        }
    }
    
    return models


def generate_benchmark_results():
    """
    Generate simulated benchmark results for instances across model types
    """
    # Create instance types for benchmarking
    instance_types = [
        "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge",
        "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
        "ml.r5.xlarge", "ml.r5.2xlarge",
        "ml.g4dn.xlarge", "ml.g4dn.2xlarge", "ml.g4dn.4xlarge",
        "ml.inf1.xlarge", "ml.inf1.2xlarge",
        "ml.g5.xlarge", "ml.g5.2xlarge",
        "ml.inf2.xlarge"
    ]
    
    # Create benchmark results for CV model
    cv_model_results = {
        instance_type: {
            "latency_ms": base_latency * latency_factor,
            "throughput": base_throughput * throughput_factor,
            "cost_per_1M": base_cost * cost_factor,
            "cpu_utilization": random.uniform(30, 95)
        }
        for instance_type, (base_latency, base_throughput, base_cost, latency_factor, throughput_factor, cost_factor) in {
            "ml.c5.xlarge": (35, 400, 4.5, 1.0, 1.0, 1.0),
            "ml.c5.2xlarge": (25, 700, 7.0, 1.0, 1.0, 1.0),
            "ml.c5.4xlarge": (18, 1200, 13.0, 1.0, 1.0, 1.0),
            "ml.m5.xlarge": (38, 350, 4.2, 1.0, 1.0, 1.0),
            "ml.m5.2xlarge": (28, 600, 7.5, 1.0, 1.0, 1.0),
            "ml.m5.4xlarge": (22, 1000, 14.0, 1.0, 1.0, 1.0),
            "ml.r5.xlarge": (36, 370, 5.1, 1.0, 1.0, 1.0),
            "ml.r5.2xlarge": (26, 650, 9.5, 1.0, 1.0, 1.0),
            "ml.g4dn.xlarge": (12, 1500, 8.2, 1.0, 1.0, 1.0),
            "ml.g4dn.2xlarge": (10, 2200, 14.5, 1.0, 1.0, 1.0),
            "ml.g4dn.4xlarge": (8, 3500, 28.0, 1.0, 1.0, 1.0),
            "ml.inf1.xlarge": (9, 2800, 4.8, 1.0, 1.0, 1.0),
            "ml.inf1.2xlarge": (7, 4200, 9.2, 1.0, 1.0, 1.0),
            "ml.g5.xlarge": (9, 2200, 10.1, 1.0, 1.0, 1.0),
            "ml.g5.2xlarge": (7, 3500, 19.0, 1.0, 1.0, 1.0),
            "ml.inf2.xlarge": (6, 4000, 6.5, 1.0, 1.0, 1.0)
        }.items()
    }
    
    # Create benchmark results for NLP model
    nlp_model_results = {
        instance_type: {
            "latency_ms": base_latency * random.uniform(0.9, 1.1),
            "throughput": base_throughput * random.uniform(0.9, 1.1),
            "cost_per_1M": base_cost * random.uniform(0.9, 1.1),
            "cpu_utilization": random.uniform(30, 95)
        }
        for instance_type, (base_latency, base_throughput, base_cost) in {
            "ml.c5.xlarge": (70, 180, 9.0),
            "ml.c5.2xlarge": (55, 280, 14.0),
            "ml.c5.4xlarge": (40, 480, 25.0),
            "ml.m5.xlarge": (75, 160, 8.5),
            "ml.m5.2xlarge": (60, 240, 13.0),
            "ml.m5.4xlarge": (45, 420, 23.0),
            "ml.r5.xlarge": (72, 170, 9.5),
            "ml.r5.2xlarge": (58, 260, 18.0),
            "ml.g4dn.xlarge": (22, 600, 16.0),
            "ml.g4dn.2xlarge": (18, 900, 29.0),
            "ml.g4dn.4xlarge": (15, 1400, 56.0),
            "ml.inf1.xlarge": (19, 900, 9.0),
            "ml.inf1.2xlarge": (16, 1300, 17.0),
            "ml.g5.xlarge": (18, 800, 19.0),
            "ml.g5.2xlarge": (14, 1200, 35.0),
            "ml.inf2.xlarge": (12, 1500, 12.0)
        }.items()
    }
    
    # Create benchmark results for Tabular model
    tabular_model_results = {
        instance_type: {
            "latency_ms": base_latency * random.uniform(0.9, 1.1),
            "throughput": base_throughput * random.uniform(0.9, 1.1),
            "cost_per_1M": base_cost * random.uniform(0.9, 1.1),
            "cpu_utilization": random.uniform(30, 95)
        }
        for instance_type, (base_latency, base_throughput, base_cost) in {
            "ml.c5.xlarge": (12, 5000, 1.6),
            "ml.c5.2xlarge": (9, 8000, 2.4),
            "ml.c5.4xlarge": (7, 12000, 4.5),
            "ml.m5.xlarge": (14, 4500, 1.5),
            "ml.m5.2xlarge": (10, 7000, 2.6),
            "ml.m5.4xlarge": (8, 10000, 4.8),
            "ml.r5.xlarge": (15, 4200, 2.0),
            "ml.r5.2xlarge": (11, 6500, 3.8),
            "ml.g4dn.xlarge": (8, 7000, 4.0),
            "ml.g4dn.2xlarge": (7, 9000, 7.5),
            "ml.g4dn.4xlarge": (6, 12000, 14.0),
            "ml.inf1.xlarge": (9, 6500, 2.8),
            "ml.inf1.2xlarge": (8, 8500, 5.0),
            "ml.g5.xlarge": (7, 8000, 5.0),
            "ml.g5.2xlarge": (6, 10000, 9.0),
            "ml.inf2.xlarge": (7, 7500, 3.5)
        }.items()
    }
    
    # Create benchmark results for large NLP model
    large_nlp_model_results = {
        instance_type: {
            "latency_ms": base_latency * random.uniform(0.9, 1.1),
            "throughput": base_throughput * random.uniform(0.9, 1.1),
            "cost_per_1M": base_cost * random.uniform(0.9, 1.1),
            "cpu_utilization": random.uniform(40, 98)
        }
        for instance_type, (base_latency, base_throughput, base_cost) in {
            "ml.c5.xlarge": (550, 8, 130.0),
            "ml.c5.2xlarge": (320, 14, 92.0),
            "ml.c5.4xlarge": (160, 28, 82.0),
            "ml.m5.xlarge": (580, 7, 145.0),
            "ml.m5.2xlarge": (350, 12, 108.0),
            "ml.m5.4xlarge": (175, 25, 95.0),
            "ml.r5.xlarge": (520, 8, 135.0),
            "ml.r5.2xlarge": (300, 15, 102.0),
            "ml.g4dn.xlarge": (120, 42, 52.0),
            "ml.g4dn.2xlarge": (85, 62, 70.0),
            "ml.g4dn.4xlarge": (48, 110, 125.0),
            "ml.inf1.xlarge": (85, 60, 28.0),
            "ml.inf1.2xlarge": (55, 95, 46.0),
            "ml.g5.xlarge": (90, 55, 60.0),
            "ml.g5.2xlarge": (65, 75, 95.0),
            "ml.inf2.xlarge": (45, 110, 38.0)
        }.items()
    }
    
    # Create benchmark results for recommender model
    recommender_model_results = {
        instance_type: {
            "latency_ms": base_latency * random.uniform(0.9, 1.1),
            "throughput": base_throughput * random.uniform(0.9, 1.1),
            "cost_per_1M": base_cost * random.uniform(0.9, 1.1),
            "cpu_utilization": random.uniform(40, 98)
        }
        for instance_type, (base_latency, base_throughput, base_cost) in {
            "ml.c5.xlarge": (95, 120, 18.0),
            "ml.c5.2xlarge": (65, 220, 12.0),
            "ml.c5.4xlarge": (42, 380, 19.0),
            "ml.m5.xlarge": (105, 110, 16.0),
            "ml.m5.2xlarge": (72, 180, 15.0),
            "ml.m5.4xlarge": (48, 340, 25.0),
            "ml.r5.xlarge": (92, 115, 19.0),
            "ml.r5.2xlarge": (60, 190, 19.0),
            "ml.g4dn.xlarge": (35, 400, 21.0),
            "ml.g4dn.2xlarge": (25, 520, 38.0),
            "ml.g4dn.4xlarge": (18, 780, 72.0),
            "ml.inf1.xlarge": (30, 450, 15.0),
            "ml.inf1.2xlarge": (22, 650, 26.0),
            "ml.g5.xlarge": (28, 440, 26.0),
            "ml.g5.2xlarge": (20, 600, 48.0),
            "ml.inf2.xlarge": (25, 520, 19.0)
        }.items()
    }
    
    return {
        "cv_model": cv_model_results,
        "nlp_model": nlp_model_results,
        "tabular_model": tabular_model_results,
        "large_nlp_model": large_nlp_model_results,
        "recommender_model": recommender_model_results
    }


def generate_recommendation(model_id, load_profile="moderate", cost_sensitivity="medium"):
    """
    Generate a recommendation based on model type and constraints
    """
    benchmark_results = st.session_state.benchmark_results[model_id]
    model_details = st.session_state.model_details[model_id]
    
    # Define weights based on cost sensitivity
    if cost_sensitivity == "high":
        cost_weight = 0.6
        latency_weight = 0.2
        throughput_weight = 0.2
    elif cost_sensitivity == "medium":
        cost_weight = 0.33
        latency_weight = 0.34
        throughput_weight = 0.33
    else:  # low
        cost_weight = 0.2
        latency_weight = 0.5
        throughput_weight = 0.3
    
    # Adjust weights based on model's performance metrics importance
    metrics = model_details["performance_metrics"]
    latency_weight *= metrics["latency_importance"] / 100
    throughput_weight *= metrics["throughput_importance"] / 100
    cost_weight *= metrics["cost_importance"] / 100
    
    # Normalize weights to sum to 1
    total_weight = latency_weight + throughput_weight + cost_weight
    latency_weight /= total_weight
    throughput_weight /= total_weight
    cost_weight /= total_weight
    
    # Define target load based on load profile
    if load_profile == "high":
        target_load = 10000  # requests per minute
    elif load_profile == "moderate":
        target_load = 1000  # requests per minute
    else:  # low
        target_load = 100  # requests per minute
    
    # Calculate scores for each instance
    instance_scores = {}
    for instance, metrics in benchmark_results.items():
        # Normalize metrics (lower is better for latency and cost, higher is better for throughput)
        max_latency = max(item["latency_ms"] for item in benchmark_results.values())
        max_throughput = max(item["throughput"] for item in benchmark_results.values())
        max_cost = max(item["cost_per_1M"] for item in benchmark_results.values())
        
        latency_score = 1 - (metrics["latency_ms"] / max_latency)
        throughput_score = metrics["throughput"] / max_throughput
        cost_score = 1 - (metrics["cost_per_1M"] / max_cost)
        
        # Calculate weighted score
        score = (latency_weight * latency_score + 
                 throughput_weight * throughput_score + 
                 cost_weight * cost_score)
        
        # Filter out instances that can't handle the load
        requests_per_second = target_load / 60
        if metrics["throughput"] < requests_per_second:
            continue
        
        instance_scores[instance] = {
            "score": score,
            "latency_ms": metrics["latency_ms"],
            "throughput": metrics["throughput"],
            "cost_per_1M": metrics["cost_per_1M"],
            "cpu_utilization": metrics["cpu_utilization"]
        }
    
    # Sort instances by score
    sorted_instances = sorted(instance_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    # Select top 3 instances if available
    top_instances = sorted_instances[:3] if len(sorted_instances) >= 3 else sorted_instances
    
    # Calculate estimated monthly cost
    monthly_requests = target_load * 60 * 24 * 30  # requests per month
    for instance, metrics in instance_scores.items():
        metrics["monthly_cost"] = (monthly_requests / 1000000) * metrics["cost_per_1M"]
    
    # Format results
    recommendations = [{
        "instance_type": instance,
        "score": round(metrics["score"] * 100),  # Convert to percentage
        "latency_ms": round(metrics["latency_ms"], 1),
        "throughput_per_second": round(metrics["throughput"]),
        "cost_per_1M": round(metrics["cost_per_1M"], 2),
        "monthly_cost": round(metrics["monthly_cost"], 2),
        "cpu_utilization": round(metrics["cpu_utilization"]),
        "rank": i + 1
    } for i, (instance, metrics) in enumerate(top_instances)]
    
    # Add explanation of recommendation
    result = {
        "recommendations": recommendations,
        "weights": {
            "latency": round(latency_weight * 100),
            "throughput": round(throughput_weight * 100),
            "cost": round(cost_weight * 100)
        },
        "load_profile": {
            "name": load_profile,
            "requests_per_minute": target_load,
            "monthly_requests": monthly_requests
        },
        "model_details": model_details,
        "timestamps": {
            "start_time": "2023-12-01T10:00:00Z",
            "end_time": "2023-12-01T11:30:00Z",
            "duration_minutes": 90
        }
    }
    
    return result


def load_lottie_url(url: str):
    """
    Load lottie animation from URL
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


# Visualization functions
def create_instance_family_comparison_chart(instance_family_id):
    """
    Create a visualization comparing different sizes within an instance family
    """
    instance_details = st.session_state.instance_details
    
    # Check if instance family exists
    if instance_family_id not in instance_details:
        return None
    
    family = instance_details[instance_family_id]
    
    # Extract instance types and their vCPU and Memory
    instance_types = []
    vcpus = []
    memory = []
    
    for instance_type, specs in family["types"].items():
        instance_types.append(instance_type)
        vcpus.append(specs["vCPU"])
        
        # Extract numeric memory value (remove "GiB")
        mem_str = specs["Memory"].replace(" GiB", "")
        memory.append(float(mem_str))
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add bars for vCPU
    fig.add_trace(go.Bar(
        x=instance_types,
        y=vcpus,
        name='vCPUs',
        marker_color='#00A1C9',
        text=vcpus,
        textposition='auto'
    ))
    
    # Add line for Memory
    fig.add_trace(go.Scatter(
        x=instance_types,
        y=memory,
        name='Memory (GiB)',
        marker_color='#FF9900',
        mode='lines+markers',
        line=dict(width=4),
        marker=dict(size=10),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{family['name']} Instance Comparison",
        xaxis_title="Instance Type",
        yaxis=dict(
            title="vCPUs",
            # titlefont=dict(color="#00A1C9"),
            tickfont=dict(color="#00A1C9")
        ),
        yaxis2=dict(
            title="Memory (GiB)",
            # titlefont=dict(color="#FF9900"),
            tickfont=dict(color="#FF9900"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_instance_comparison_chart(model_id, highlight_instances=None):
    """
    Create a visualization comparing different instances for a specific model type
    """
    benchmark_results = st.session_state.benchmark_results[model_id]
    
    # Prepare data for visualization
    instance_types = list(benchmark_results.keys())
    latency = [benchmark_results[i]["latency_ms"] for i in instance_types]
    throughput = [benchmark_results[i]["throughput"] for i in instance_types]
    cost = [benchmark_results[i]["cost_per_1M"] for i in instance_types]
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Create color list with highlights
    if highlight_instances:
        colors = ['#FF9900' if instance in highlight_instances else '#00A1C9' for instance in instance_types]
        marker_size = [10 if instance in highlight_instances else 6 for instance in instance_types]
    else:
        colors = ['#00A1C9'] * len(instance_types)
        marker_size = [6] * len(instance_types)
    
    # Add bars for latency
    fig.add_trace(go.Bar(
        x=instance_types,
        y=latency,
        name='Latency (ms)',
        marker_color=colors,
        opacity=0.7,
        text=[f"{l:.1f} ms" for l in latency],
        textposition='auto'
    ))
    
    # Add line for throughput
    fig.add_trace(go.Scatter(
        x=instance_types,
        y=throughput,
        name='Throughput (req/sec)',
        marker_color='#59BA47',
        mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=marker_size),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Instance Performance Comparison for {st.session_state.model_details[model_id]['name']}",
        xaxis=dict(
            title="Instance Type",
            tickangle=45
        ),
        yaxis=dict(
            title="Latency (ms)",
            # titlefont=dict(color="#00A1C9"),
            tickfont=dict(color="#00A1C9")
        ),
        yaxis2=dict(
            title="Throughput (req/sec)",
            # titlefont=dict(color="#59BA47"),
            tickfont=dict(color="#59BA47"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_cost_performance_chart(model_id):
    """
    Create a visualization showing cost vs performance for different instances
    """
    benchmark_results = st.session_state.benchmark_results[model_id]
    
    # Prepare data for visualization
    instance_types = list(benchmark_results.keys())
    latency = [benchmark_results[i]["latency_ms"] for i in instance_types]
    throughput = [benchmark_results[i]["throughput"] for i in instance_types]
    cost = [benchmark_results[i]["cost_per_1M"] for i in instance_types]
    
    # Create categories for different instance families
    families = []
    colors = []
    family_color_map = {
        "ml.c5": "#00A1C9",  # teal
        "ml.m5": "#FF9900",  # orange
        "ml.r5": "#59BA47",  # green
        "ml.g4dn": "#232F3E",  # blue
        "ml.inf1": "#D13212",  # red
        "ml.g5": "#8C51A5",  # purple
        "ml.inf2": "#FF5A5A",  # coral
        "ml.p4d": "#545B64"  # gray
    }
    
    for instance in instance_types:
        family = ".".join(instance.split(".")[:2])
        families.append(family)
        colors.append(family_color_map.get(family, "#545B64"))
    
    # Create scatter plot of cost vs latency, with size representing throughput
    fig = px.scatter(
        x=cost,
        y=latency,
        size=throughput,
        color=families,
        color_discrete_map=family_color_map,
        hover_name=instance_types,
        labels={
            "x": "Cost per 1M requests ($)",
            "y": "Latency (ms)",
            "size": "Throughput (req/sec)",
            "color": "Instance Family"
        },
        title=f"Cost vs. Performance for {st.session_state.model_details[model_id]['name']}"
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        xaxis=dict(title="Cost per 1M requests ($)"),
        yaxis=dict(title="Latency (ms)"),
        legend=dict(
            title="Instance Family",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add annotations
    fig.add_annotation(
        x=0.05,
        y=0.05,
        xref="paper",
        yref="paper",
        text="Lower left is better (lower cost, lower latency)",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#545B64",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # Invert y-axis so lower latency is higher on the chart
    fig.update_yaxes(autorange="reversed")
    
    return fig


def create_recommendation_visualization(results):
    """
    Create visualizations for the recommendation results
    """
    if not results:
        return None
    
    recommendations = results["recommendations"]
    weights = results["weights"]
    
    # Create performance comparison chart
    instance_types = [r["instance_type"] for r in recommendations]
    scores = [r["score"] for r in recommendations]
    latency = [r["latency_ms"] for r in recommendations]
    throughput = [r["throughput_per_second"] for r in recommendations]
    cost = [r["monthly_cost"] for r in recommendations]
    
    # Create bar chart of scores
    score_fig = go.Figure()
    
    # Add bars for scores
    score_fig.add_trace(go.Bar(
        x=instance_types,
        y=scores,
        name='Overall Score',
        marker_color=['#00A1C9', '#59BA47', '#FF9900'][:len(instance_types)],
        text=[f"{s}%" for s in scores],
        textposition='auto'
    ))
    
    # Update layout
    score_fig.update_layout(
        title="Recommended Instance Types (Overall Score)",
        xaxis_title="Instance Type",
        yaxis=dict(
            title="Score (%)",
            range=[0, 100]
        ),
        height=400
    )
    
    # Create detailed comparison chart
    detail_fig = go.Figure()
    
    # Add bars for latency
    detail_fig.add_trace(go.Bar(
        x=instance_types,
        y=latency,
        name='Latency (ms)',
        marker_color='#00A1C9',
        text=[f"{l:.1f} ms" for l in latency],
        textposition='auto'
    ))
    
    # Add bars for throughput
    detail_fig.add_trace(go.Bar(
        x=instance_types,
        y=throughput,
        name='Throughput (req/sec)',
        marker_color='#59BA47',
        text=[f"{t:.0f}" for t in throughput],
        textposition='auto'
    ))
    
    # Add bars for monthly cost
    detail_fig.add_trace(go.Bar(
        x=instance_types,
        y=cost,
        name='Monthly Cost ($)',
        marker_color='#FF9900',
        text=[f"${c:.2f}" for c in cost],
        textposition='auto'
    ))
    
    # Update layout
    detail_fig.update_layout(
        title="Detailed Comparison of Recommended Instances",
        xaxis_title="Instance Type",
        yaxis_title="Value (normalized)",
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create weights pie chart
    weights_fig = go.Figure()
    
    # Add pie chart for weights
    weights_fig.add_trace(go.Pie(
        labels=['Latency', 'Throughput', 'Cost'],
        values=[weights['latency'], weights['throughput'], weights['cost']],
        marker_colors=['#00A1C9', '#59BA47', '#FF9900']
    ))
    
    # Update layout
    weights_fig.update_layout(
        title="Weight Factors in Recommendation",
        height=400
    )
    
    return {
        "score_chart": score_fig,
        "detail_chart": detail_fig,
        "weights_chart": weights_fig
    }


def create_inference_recommender_architecture():
    """
    Create a visualization of SageMaker Inference Recommender architecture
    """
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("Model", type="model", color="#00A1C9")
    G.add_node("Inference Recommender", type="service", color="#FF9900")
    G.add_node("Instance Recommendations", type="recommendations", color="#59BA47")
    G.add_node("Default Job", type="job", color="#232F3E")
    G.add_node("Advanced Job", type="job", color="#232F3E")
    G.add_node("Model Registry", type="registry", color="#D13212")
    G.add_node("CPU Instances", type="instances", color="#545B64")
    G.add_node("GPU Instances", type="instances", color="#545B64")
    G.add_node("Inferentia", type="instances", color="#545B64")
    G.add_node("Load Tests", type="test", color="#8C51A5")
    G.add_node("Traffic Patterns", type="patterns", color="#8C51A5")
    
    # Add edges
    G.add_edge("Model", "Inference Recommender")
    G.add_edge("Model Registry", "Inference Recommender")
    G.add_edge("Inference Recommender", "Default Job")
    G.add_edge("Inference Recommender", "Advanced Job")
    G.add_edge("Default Job", "Instance Recommendations")
    G.add_edge("Advanced Job", "Instance Recommendations")
    G.add_edge("CPU Instances", "Default Job")
    G.add_edge("GPU Instances", "Default Job")
    G.add_edge("Inferentia", "Default Job")
    G.add_edge("CPU Instances", "Advanced Job")
    G.add_edge("GPU Instances", "Advanced Job")
    G.add_edge("Inferentia", "Advanced Job")
    G.add_edge("Load Tests", "Advanced Job")
    G.add_edge("Traffic Patterns", "Advanced Job")
    
    # Define position layout
    pos = {
        "Model": (0, 6),
        "Model Registry": (0, 4),
        "Inference Recommender": (3, 5),
        "Default Job": (6, 6),
        "Advanced Job": (6, 4),
        "Instance Recommendations": (9, 5),
        "CPU Instances": (3, 8),
        "GPU Instances": (5, 8),
        "Inferentia": (7, 8),
        "Load Tests": (3, 2),
        "Traffic Patterns": (7, 2)
    }
    
    return G, pos


def draw_architecture_diagram(G, pos):
    """
    Draw the architecture diagram
    """
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, width=2.0, alpha=0.7, arrowstyle='->', edge_color='#aaaaaa')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='white')
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()


# Main application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Amazon SageMaker Inference Recommender",
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
        "red": "#D13212"
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
        .warning-box {
            background-color: #FFF8E6;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #FF9900;
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
        .status-card {
            border: 1px solid #E9ECEF;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            background-color: #FFFFFF;
            transition: transform 0.2s;
        }
        .status-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .status-1 {
            border-left: 5px solid #59BA47;
        }
        .status-2 {
            border-left: 5px solid #D13212;
        }
        .status-3 {
            border-left: 5px solid #FF9900;
        }
        .metrics-table th {
            font-weight: normal;
            color: #545B64;
        }
        .metrics-table td {
            font-weight: bold;
        }
        .instance-family-card {
            border: 1px solid #E9ECEF;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #FFFFFF;
            transition: transform 0.2s;
            cursor: pointer;
        }
        .instance-family-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #00A1C9;
        }
        .instance-family-selected {
            border-color: #FF9900;
            border-width: 2px;
        }
        .model-card {
            border: 1px solid #E9ECEF;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #FFFFFF;
            transition: transform 0.2s;
            cursor: pointer;
        }
        .model-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #00A1C9;
        }
        .model-card-selected {
            border-color: #FF9900;
            border-width: 2px;
        }
        .instance-tag {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .instance-tag-cpu {
            background-color: #E9ECEF;
            color: #232F3E;
        }
        .instance-tag-gpu {
            background-color: #00A1C9;
            color: white;
        }
        .instance-tag-inferentia {
            background-color: #FF9900;
            color: white;
        }
        .result-card {
            border: 1px solid #E9ECEF;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #FFFFFF;
        }
        .result-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid #E9ECEF;
            padding-bottom: 10px;
        }
        .result-rank {
            background-color: #232F3E;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        st.title("Session Management")
        st.info(f"User ID: {st.session_state.user_id}")
        
        if st.button("Reset Session"):
            reset_session()
            st.rerun()
        
        st.divider()
        # Information about the application
        with st.expander("About this application", expanded=False):
            st.markdown("""
                This e-learning application demonstrates Amazon SageMaker Inference Recommender, a capability 
                that helps you choose the best instance type for your ML model deployment.
                
                Navigate through the tabs to learn about different ML instance options and how 
                Inference Recommender works to optimize your model deployment.
            """)
            
            # Load lottie animation
            lottie_url = "https://assets1.lottiefiles.com/packages/lf20_AbQGpP.json"
            lottie_json = load_lottie_url(lottie_url)
            if lottie_json:
                st_lottie(lottie_json, height=200, key="sidebar_animation")
            
            # Additional resources section
            st.subheader("Additional Resources")
            st.markdown("""
                - [SageMaker Inference Recommender Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-recommender.html)
                - [SageMaker Instance Types](https://docs.aws.amazon.com/sagemaker/latest/dg/instance-types-az.html)
                - [Inference Recommender Best Practices](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-sagemaker-inference-recommender/)
            """)
    
    # Main app header
    st.title("Amazon SageMaker Inference Recommender")
    st.markdown("Learn how to optimize your ML model deployments by finding the best instance type for your specific model and workload.")
    
    # Tab-based navigation with emoji
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview", 
        "🖥️ ML Instance Types",
        "🚀 Inference Recommender",
        "🧪 Interactive Demo"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("Introduction to SageMaker Inference Recommender")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            **Amazon SageMaker Inference Recommender** helps you select the best ML instance type for your model 
            and workload requirements. It runs a series of load tests to benchmark your model's performance on 
            different instance types, then recommends the most suitable option based on performance, cost, or a 
            balance of both.
            
            **Key benefits:**
            - **Optimize costs**: Find the most cost-effective instance that meets your performance needs
            - **Improve performance**: Identify instances that minimize latency or maximize throughput
            - **Simplify deployment decisions**: Remove guesswork from instance selection
            - **Test at scale**: Evaluate model performance under various traffic patterns
            - **Automate benchmarking**: Avoid manual testing across multiple instance types
            """)
            
            st.markdown("""
            <div class="info-box">
            <h3>Why Instance Selection Matters</h3>
            <p>Choosing the right instance type for your ML model deployment has significant impacts:</p>
            <ul>
                <li><strong>Cost optimization</strong>: Rightsizing can reduce inference costs by up to 50%</li>
                <li><strong>Performance</strong>: The right instance can reduce latency by 3-10x</li>
                <li><strong>Reliability</strong>: Properly provisioned endpoints handle traffic spikes better</li>
                <li><strong>User experience</strong>: Faster inference means better application responsiveness</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Load graphics illustrating the concept
            st.image("https://d1.awsstatic.com/re19/Sagemaker/SageMaker-Inference-Recommender.ef2fbd5b3a1ff4672318b215fc501d0a962741f9.png", 
                     use_container_width=True)
            st.caption("Amazon SageMaker Inference Recommender workflow")
        
        st.subheader("How It Works")
        
        # Architecture diagram
        G, pos = create_inference_recommender_architecture()
        fig = draw_architecture_diagram(G, pos)
        st.pyplot(fig)
        
        # Three columns for three steps
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>1️⃣ Default Recommendation</h3>
                <p>Quick benchmarking to identify suitable instance types:</p>
                <ul>
                    <li>Register your model in the SageMaker Model Registry</li>
                    <li>Create a recommendation job with basic parameters</li>
                    <li>Test your model on a subset of instance types</li>
                    <li>Results in minutes with initial recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
                <h3>2️⃣ Advanced Benchmarking</h3>
                <p>In-depth testing with custom workload patterns:</p>
                <ul>
                    <li>Define custom traffic patterns (e.g., spikes, sustained load)</li>
                    <li>Specify your performance and cost requirements</li>
                    <li>Run extensive load tests across multiple instance types</li>
                    <li>Detailed analysis of model behavior under load</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="card">
                <h3>3️⃣ Inference Optimization</h3>
                <p>Deploy based on optimized recommendations:</p>
                <ul>
                    <li>Select from recommended instance types based on priorities</li>
                    <li>Deploy to SageMaker endpoints with confidence</li>
                    <li>Monitor actual performance against predictions</li>
                    <li>Re-evaluate as model or traffic patterns evolve</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Instance Selection Challenges")
        
        st.markdown("""
        <div class="warning-box">
        <h3>Common Challenges in ML Inference Deployment</h3>
        <p>Without proper guidance, organizations often face these instance selection issues:</p>
        <ul>
            <li><strong>Overprovisioning</strong>: Selecting instances that are far more powerful (and expensive) than needed</li>
            <li><strong>Underprovisioning</strong>: Choosing instances that can't handle traffic spikes, leading to high latency or throttling</li>
            <li><strong>Framework mismatch</strong>: Using instances that don't optimize for specific ML frameworks</li>
            <li><strong>Inefficient hardware selection</strong>: Using GPU instances for CPU-bound models or vice versa</li>
            <li><strong>Ignoring specialized hardware</strong>: Missing opportunities to use purpose-built inference accelerators like AWS Inferentia</li>
        </ul>
        <p>SageMaker Inference Recommender helps solve these challenges through data-driven recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Case study
        st.subheader("Real-World Impact")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Case Study: E-commerce Recommendation System</h3>
                <p>A large online retailer needed to optimize their product recommendation model deployment:</p>
                <ul>
                    <li><strong>Challenge</strong>: High inference costs with inconsistent latency</li>
                    <li><strong>Solution</strong>: Used Inference Recommender to test 15 instance types</li>
                    <li><strong>Result</strong>: Switched from ml.c5.4xlarge to ml.inf1.xlarge instances</li>
                    <li><strong>Impact</strong>: 62% cost reduction, 40% lower latency, better customer experience</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Create a simple ROI chart
            months = list(range(1, 13))
            old_cost = [10000] * 12  # $10K per month with old instance
            new_cost = [3800] * 12  # $3.8K per month with new instance
            cumulative_savings = [old_cost[0] - new_cost[0]]
            for i in range(1, 12):
                cumulative_savings.append(cumulative_savings[i-1] + old_cost[i] - new_cost[i])
            
            fig = go.Figure()
            
            # Add monthly costs
            fig.add_trace(go.Bar(
                x=months,
                y=old_cost,
                name='Before Optimization',
                marker_color='#D13212'  # red
            ))
            
            fig.add_trace(go.Bar(
                x=months,
                y=new_cost,
                name='After Optimization',
                marker_color='#59BA47'  # green
            ))
            
            # Add cumulative savings line
            fig.add_trace(go.Scatter(
                x=months,
                y=cumulative_savings,
                name='Cumulative Savings',
                mode='lines+markers',
                marker_color='#FF9900',  # orange
                line=dict(width=3),
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title="Cost Savings from Inference Optimization",
                xaxis_title="Month",
                yaxis_title="Monthly Cost ($)",
                yaxis2=dict(
                    title="Cumulative Savings ($)",
                    # titlefont=dict(color='#FF9900'),
                    tickfont=dict(color='#FF9900'),
                    overlaying="y",
                    side="right"
                ),
                barmode='group',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: ML INSTANCE TYPES
    with tab2:
        st.header("Amazon SageMaker ML Instance Types")
        
        st.markdown("""
        SageMaker provides a wide range of instance types optimized for different ML workloads. 
        Understanding the characteristics and best use cases for each instance family helps you 
        make better deployment decisions.
        """)
        
        # Instance type families
        st.subheader("Instance Families")
        
        # Create a grid of instance family cards
        instance_details = st.session_state.instance_details
        
        # Function to create instance family cards
        def create_instance_family_card(family_id, details):
            # Determine icon based on family type
            icon = "⚡" if details["icon"] == "gpu" else "🧠" if details["icon"] == "inferentia" or details["icon"] == "trainium" else "🖥️"
            
            # Determine if this is the selected family
            is_selected = st.session_state.selected_instance_family == family_id
            selected_class = "instance-family-selected" if is_selected else ""
            
            # Create HTML for the card
            html = f"""
            <div class="instance-family-card {selected_class}" data-family-id="{family_id}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 20px; font-weight: bold;">{icon} {details['name']}</div>
                </div>
                <p style="margin-bottom: 10px; font-size: 14px; color: #545B64;">{details['description']}</p>
                <div style="display: flex; flex-wrap: wrap; margin-bottom: 10px;">
            """
            
            # Add instance type tags
            for instance_type in list(details['types'].keys())[:3]:
                html += f'<span class="instance-tag instance-tag-{"gpu" if "gpu" in details["icon"] else "inferentia" if "inferentia" in details["icon"] else "cpu"}">{instance_type}</span>'
            
            if len(details['types']) > 3:
                html += f'<span class="instance-tag instance-tag-cpu">+{len(details["types"]) - 3} more</span>'
            
            # Close tags and add cost range
            html += f"""
                </div>
                <div style="font-size: 13px;"><strong>Cost range:</strong> {details['cost_range']}</div>
            </div>
            """
            
            return html
        
        # Create a grid of cards
        col1, col2 = st.columns(2)
        
        # Split instance families between columns
        families = list(instance_details.keys())
        col1_families = families[:len(families)//2 + len(families)%2]
        col2_families = families[len(families)//2 + len(families)%2:]
        
        # Render column 1 cards
        with col1:
            for family_id in col1_families:
                st.markdown(create_instance_family_card(family_id, instance_details[family_id]), unsafe_allow_html=True)
                
                # Add selection button
                if st.button(f"Select {family_id}", key=f"select_{family_id}"):
                    st.session_state.selected_instance_family = family_id
                    st.rerun()
        
        # Render column 2 cards
        with col2:
            for family_id in col2_families:
                st.markdown(create_instance_family_card(family_id, instance_details[family_id]), unsafe_allow_html=True)
                
                # Add selection button
                if st.button(f"Select {family_id}", key=f"select_{family_id}"):
                    st.session_state.selected_instance_family = family_id
                    st.rerun()
        
        st.divider()
        
        # Display selected family details
        selected_family = st.session_state.selected_instance_family
        if selected_family in instance_details:
            family = instance_details[selected_family]
            
            st.subheader(f"{family['name']} Details")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {family['description']}")
                
                st.markdown("**Best use cases:**")
                for use_case in family['use_cases']:
                    st.markdown(f"- {use_case}")
                
                st.markdown("**Advantages:**")
                for advantage in family['good_for']:
                    st.markdown(f"- {advantage}")
                
                st.markdown("**Limitations:**")
                for limitation in family['limitations']:
                    st.markdown(f"- {limitation}")
            
            with col2:
                st.markdown("**Cost Range:**")
                st.info(family['cost_range'])
                
                # Create tag cloud for instance types
                st.markdown("**Available sizes:**")
                
                instance_types_html = "<div style='display: flex; flex-wrap: wrap;'>"
                for instance_type in family['types'].keys():
                    tag_class = "gpu" if "gpu" in family["icon"] else "inferentia" if "inferentia" in family["icon"] else "cpu"
                    instance_types_html += f'<span class="instance-tag instance-tag-{tag_class}">{instance_type}</span>'
                instance_types_html += "</div>"
                
                st.markdown(instance_types_html, unsafe_allow_html=True)
            
            # Instance comparison chart
            st.subheader(f"{family['name']} Size Comparison")
            
            # Create comparison chart
            fig = create_instance_family_comparison_chart(selected_family)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Instance specifications table
            st.subheader("Instance Specifications")
            
            # Convert the instance types dictionary to a DataFrame for display
            instance_specs = []
            for instance_type, specs in family['types'].items():
                instance_specs.append({
                    "Instance Type": instance_type,
                    **specs
                })
            
            specs_df = pd.DataFrame(instance_specs)
            st.dataframe(specs_df, use_container_width=True)
        
        # Instance selection guidance
        st.subheader("Instance Selection Guidance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="status-card status-1">
                <h4>CPU-Based Instances (C5, M5)</h4>
                <p><strong>Best for:</strong></p>
                <ul>
                    <li>Traditional ML algorithms (XGBoost, LightGBM)</li>
                    <li>Small-to-medium deep learning models</li>
                    <li>Text processing with simple NLP models</li>
                    <li>Low latency requirements with simple models</li>
                    <li>Cost-sensitive deployments</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="status-card status-3">
                <h4>GPU-Based Instances (G4, G5)</h4>
                <p><strong>Best for:</strong></p>
                <ul>
                    <li>Complex deep learning models</li>
                    <li>Computer vision applications</li>
                    <li>Transformer-based NLP models</li>
                    <li>High throughput requirements</li>
                    <li>Batch inference with large models</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="status-card status-2">
                <h4>Inferentia Instances (Inf1, Inf2)</h4>
                <p><strong>Best for:</strong></p>
                <ul>
                    <li>Production deep learning deployments</li>
                    <li>Cost-optimized deep learning inference</li>
                    <li>Models compiled with AWS Neuron SDK</li>
                    <li>High-throughput inference workloads</li>
                    <li>Balance of performance and cost</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-world considerations
        st.subheader("Real-World Considerations")
        
        st.markdown("""
        <div class="warning-box">
        <h3>Beyond Instance Specifications</h3>
        <p>When selecting an instance type, consider these additional factors:</p>
        <ul>
            <li><strong>Traffic patterns:</strong> Does your model serve steady traffic or experience spikes?</li>
            <li><strong>Batch size:</strong> Can you batch requests to improve throughput?</li>
            <li><strong>Auto-scaling:</strong> Will you use auto-scaling to handle variable load?</li>
            <li><strong>Framework optimization:</strong> Is your model optimized for the hardware you're using?</li>
            <li><strong>Cost structure:</strong> Do you prioritize lower per-request cost or lower overall instance cost?</li>
            <li><strong>Memory requirements:</strong> Does your model have large artifacts that need substantial memory?</li>
        </ul>
        <p>SageMaker Inference Recommender helps address these considerations through its testing methodology.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 3: INFERENCE RECOMMENDER
    with tab3:
        st.header("How SageMaker Inference Recommender Works")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Amazon SageMaker Inference Recommender helps you find the optimal instance type for your ML model 
            by running benchmark tests and load tests on a variety of SageMaker instance types.
            
            **Two types of recommendation jobs:**
            
            1. **Default Jobs**: Quick benchmarking across a smaller set of instance types to get initial recommendations within minutes.
            
            2. **Advanced Jobs**: In-depth load testing with custom traffic patterns that simulate your expected workloads, providing detailed performance metrics.
            """)
            
            st.markdown("""
            <div class="info-box">
            <h3>Key Features</h3>
            <ul>
                <li><strong>Automated benchmarking</strong> across multiple instance types</li>
                <li><strong>Custom traffic patterns</strong> to simulate real-world scenarios</li>
                <li><strong>Performance metrics</strong> including latency, throughput, and cost</li>
                <li><strong>Endpoint invocation</strong> to verify model functionality</li>
                <li><strong>Custom payloads</strong> to test with your specific input data</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Illustration of recommendation workflow
            st.image("https://d1.awsstatic.com/products/sagemaker/inference-recommender-architecture.5225972a3c9ca5a2a58bd2229f145f0b0d96c2e6.png", 
                     caption="SageMaker Inference Recommender Workflow", use_container_width=True)
        
        # Job types comparison
        st.subheader("Default vs. Advanced Jobs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Default Jobs</h3>
                <ul>
                    <li><strong>Duration:</strong> Minutes</li>
                    <li><strong>Instances tested:</strong> Subset based on model profile</li>
                    <li><strong>Traffic pattern:</strong> Simple benchmark testing</li>
                    <li><strong>Best for:</strong> Initial exploration and quick guidance</li>
                    <li><strong>Setup complexity:</strong> Minimal</li>
                </ul>
                <h4>Sample Code:</h4>
                <div style="background-color: #232F3E; padding: 15px; border-radius: 5px; font-family: monospace; color: white; font-size: 12px; overflow-x: auto;">
                import boto3<br>
                <br>
                sm_client = boto3.client('sagemaker')<br>
                <br>
                response = sm_client.create_inference_recommendations_job(<br>
                JobName='sample-default-job',<br>
                JobType='Default',<br>
                RoleArn='arn:aws:iam::123456789012:role/service-role/SageMaker-Execution',<br>
                InputConfig={<br>
                'ModelPackageVersionArn': 'arn:aws:sagemaker:us-west-2:123456789012:model-package/mymodel/1'<br>
                }<br>
                )
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Advanced Jobs</h3>
                <ul>
                    <li><strong>Duration:</strong> Hours</li>
                    <li><strong>Instances tested:</strong> Wider range or custom selection</li>
                    <li><strong>Traffic pattern:</strong> Custom patterns like spikes or sustained load</li>
                    <li><strong>Best for:</strong> Production workloads with specific requirements</li>
                    <li><strong>Setup complexity:</strong> More detailed configuration</li>
                </ul>
                <h4>Sample Code:</h4>
                <div style="background-color: #232F3E; padding: 15px; border-radius: 5px; font-family: monospace; color: white; font-size: 12px; overflow-x: auto;">
                response = sm_client.create_inference_recommendations_job(<br>
                    JobName='sample-advanced-job',<br>
                    JobType='Advanced',<br>
                    RoleArn='arn:aws:iam::123456789012:role/service-role/SageMaker-Execution',<br>
                    InputConfig={<br>
                    'ModelPackageVersionArn': 'arn:aws:sagemaker:us-west-2:123456789012:model-package/mymodel/1',<br>
                    'JobDurationInSeconds': 7200,<br>
                    'TrafficPattern': {<br>
                    'TrafficType': 'PHASES',<br>
                    'Phases': [<br>
                        {<br>
                        'InitialNumberOfUsers': 1,<br>
                        'SpawnRate': 1,<br>
                        'DurationInSeconds': 120<br>
                        },<br>
                {<br>
                    'InitialNumberOfUsers': 5,<br>
                    'SpawnRate': 1,<br>
                    'DurationInSeconds': 600<br>
                }<br>
                ]<br>
                },<br>
                    'ResourceLimit': {<br>
                'MaxNumberOfTests': 10<br>
                    },<br>
                    'EndpointConfigurations': [<br>
                    {<br>
                    'InstanceType': 'ml.c5.xlarge'<br>
                    },<br>
                    {<br>
                    'InstanceType': 'ml.g4dn.xlarge'<br>
                    }<br>
                ]<br>
                }<br>
                )
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Traffic pattern visualization
        st.subheader("Traffic Patterns")
        
        st.markdown("""
        Advanced jobs allow you to simulate different traffic patterns to test how your model performs 
        under various conditions. These patterns help you understand how your model will behave in 
        real-world scenarios with varying loads.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a traffic pattern visualization
            fig = go.Figure()
            
            # Add line for traffic pattern 1: Phases
            x1 = list(range(0, 3600, 60))  # 1 hour in 1-minute intervals
            y1 = []
            
            # Initial traffic
            for _ in range(10):
                y1.append(50)
            
            # Ramp up
            for i in range(10):
                y1.append(50 + i * 15)
            
            # Sustained load
            for _ in range(20):
                y1.append(200)
            
            # Spike
            for i in range(10):
                spike_value = 200 + i * 40 if i < 5 else 400 - (i - 5) * 40
                y1.append(spike_value)
            
            # Recovery
            for i in range(10):
                y1.append(200 - i * 15)
            
            fig.add_trace(go.Scatter(
                x=x1,
                y=y1,
                mode='lines',
                name='Phase-based Pattern',
                line=dict(color='#00A1C9', width=3)
            ))
            
            # Update layout
            fig.update_layout(
                title="Phase-Based Traffic Pattern",
                xaxis_title="Time (seconds)",
                yaxis_title="Number of Users",
                height=400,
                margin=dict(l=20, r=20, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Phase-Based Pattern**
            
            Define specific phases with different user loads:
            
            ```python
            'TrafficPattern': {
                'TrafficType': 'PHASES',
                'Phases': [
                    {
                        'InitialNumberOfUsers': 50,
                        'SpawnRate': 0,
                        'DurationInSeconds': 600
                    },
                    {
                        'InitialNumberOfUsers': 50,
                        'SpawnRate': 1,
                        'DurationInSeconds': 600
                    },
                    {
                        'InitialNumberOfUsers': 200,
                        'SpawnRate': 0,
                        'DurationInSeconds': 1200
                    }
                    # More phases...
                ]
            }
            ```
            """)
        
        with col2:
            # Create a traffic pattern visualization
            fig = go.Figure()
            
            # Add line for traffic pattern 2: Steps
            x2 = list(range(0, 3600, 60))  # 1 hour in 1-minute intervals
            y2 = []
            
            for i in range(60):
                step = i // 10
                y2.append(50 * (step + 1))
            
            fig.add_trace(go.Scatter(
                x=x2,
                y=y2,
                mode='lines+markers',
                name='Step-based Pattern',
                line=dict(color='#FF9900', width=3)
            ))
            
            # Update layout
            fig.update_layout(
                title="Step-Based Traffic Pattern",
                xaxis_title="Time (seconds)",
                yaxis_title="Number of Users",
                height=400,
                margin=dict(l=20, r=20, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Step-Based Pattern**
            
            Gradually increase load in steps:
            
            ```python
            'TrafficPattern': {
                'TrafficType': 'STEPS',
                'Steps': {
                    'IncrementSize': 50,
                    'DurationInSeconds': 600,
                    'StartingNumberOfUsers': 50,
                    'MaximumNumberOfUsers': 300
                }
            }
            ```
            """)
        
        # Recommendation metrics
        st.subheader("Recommendation Metrics")
        
        st.markdown("""
        Inference Recommender provides several key metrics to help you evaluate instance performance:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 36px; color: #00A1C9;">⏱️</div>
                <h3>Latency</h3>
                <p>The time it takes for your model to respond to requests</p>
                <ul style="text-align: left;">
                    <li>Average latency</li>
                    <li>P50 latency (median)</li>
                    <li>P95 latency (95th percentile)</li>
                    <li>P99 latency (99th percentile)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 36px; color: #FF9900;">⚡</div>
                <h3>Throughput</h3>
                <p>The number of requests your model can handle per second</p>
                <ul style="text-align: left;">
                    <li>Requests per second</li>
                    <li>Maximum throughput</li>
                    <li>Sustained throughput</li>
                    <li>Throughput stability</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 36px; color: #59BA47;">💰</div>
                <h3>Cost</h3>
                <p>Cost metrics to help with budget optimization</p>
                <ul style="text-align: left;">
                    <li>Instance cost per hour</li>
                    <li>Cost per 1M requests</li>
                    <li>Cost per 1K inference hours</li>
                    <li>Estimated monthly cost</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Resource utilization
        st.subheader("Resource Utilization Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Collected Metrics</h3>
                <p>Inference Recommender also collects these resource utilization metrics:</p>
                <ul>
                    <li><strong>CPU Utilization:</strong> Percentage of CPU usage</li>
                    <li><strong>Memory Utilization:</strong> Percentage of memory usage</li>
                    <li><strong>GPU Utilization:</strong> Percentage of GPU usage (for GPU instances)</li>
                    <li><strong>GPU Memory Utilization:</strong> Percentage of GPU memory usage</li>
                    <li><strong>Disk I/O:</strong> Disk read/write operations</li>
                    <li><strong>Network I/O:</strong> Network in/out bytes</li>
                </ul>
                <p>These metrics help identify potential bottlenecks in your model's performance.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a resource utilization chart
            fig = go.Figure()
            
            # Time series for a hypothetical load test
            time_points = list(range(0, 60))
            
            # Create utilization curves
            cpu_util = [30 + i * 0.8 if i < 30 else 54 - (i - 30) * 0.3 for i in time_points]
            memory_util = [45 + i * 0.4 if i < 30 else 57 - (i - 30) * 0.15 for i in time_points]
            gpu_util = [60 + i * 0.5 if i < 30 else 75 - (i - 30) * 0.2 for i in time_points]
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=time_points,
                y=cpu_util,
                mode='lines',
                name='CPU Utilization',
                line=dict(color='#00A1C9', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=memory_util,
                mode='lines',
                name='Memory Utilization',
                line=dict(color='#FF9900', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=gpu_util,
                mode='lines',
                name='GPU Utilization',
                line=dict(color='#59BA47', width=2)
            ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=0,
                y0=80,
                x1=60,
                y1=80,
                line=dict(color="#D13212", width=2, dash="dash")
            )
            
            # Add annotation
            fig.add_annotation(
                x=55,
                y=80,
                text="Recommended max utilization",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=20
            )
            
            # Update layout
            fig.update_layout(
                title="Resource Utilization During Load Test",
                xaxis_title="Time (minutes)",
                yaxis_title="Utilization (%)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=40),
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Best practices
        st.subheader("Best Practices")
        
        st.markdown("""
        <div class="warning-box">
        <h3>Making the Most of Inference Recommender</h3>
        <ul>
            <li><strong>Register your model</strong> in the SageMaker Model Registry for easier management</li>
            <li><strong>Start with default jobs</strong> to get quick initial recommendations</li>
            <li><strong>Define realistic traffic patterns</strong> that match your expected workloads</li>
            <li><strong>Test with real payloads</strong> to get accurate performance measurements</li>
            <li><strong>Consider all metrics</strong> (not just latency or cost) when making decisions</li>
            <li><strong>Re-run recommendations</strong> when your model or workload patterns change</li>
            <li><strong>Test serverless inference</strong> as an option for variable workloads</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 4: INTERACTIVE DEMO
    with tab4:
        st.header("Interactive Inference Recommender Demo")
        
        st.markdown("""
        This interactive demo allows you to explore how Inference Recommender works with different 
        model types and deployment requirements. Choose a model, specify your workload parameters, 
        and see recommendations for optimal instance types.
        """)
        
        # Step 1: Select a model
        st.subheader("Step 1: Select a Model")
        
        model_details = st.session_state.model_details
        
        # Create model cards
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        model_ids = list(model_details.keys())
        for i, model_id in enumerate(model_ids):
            model = model_details[model_id]
            with cols[i % 3]:
                # Determine if this is the selected model
                is_selected = st.session_state.selected_model == model_id
                selected_class = "model-card-selected" if is_selected else ""
                
                st.markdown(f"""
                <div class="model-card {selected_class}">
                    <div style="font-weight: bold; font-size: 16px;">{model["name"]}</div>
                    <div style="font-size: 12px; color: #545B64; margin-bottom: 10px;">{model["framework"]} | {model["type"]}</div>
                    <div style="font-size: 14px; margin-bottom: 10px;">{model["description"]}</div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px;">
                        <div><strong>Size:</strong> {model["size"]}</div>
                        <div><strong>Latency Sensitivity:</strong> {model["latency_sensitivity"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Select {model['name'].split('(')[0].strip()}", key=f"select_model_{model_id}"):
                    st.session_state.selected_model = model_id
                    st.rerun()
        
        st.divider()
        
        # Step 2: Configure Recommendation Job
        st.subheader("Step 2: Configure Recommendation Job")
        
        # Get selected model
        selected_model = st.session_state.selected_model
        model = model_details[selected_model]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>Model Details: {model["name"]}</h3>
                <div style="display: flex; gap: 20px; margin-bottom: 15px;">
                    <div style="flex: 1;">
                        <p><strong>Framework:</strong> {model["framework"]}</p>
                        <p><strong>Type:</strong> {model["type"]}</p>
                        <p><strong>Size:</strong> {model["size"]}</p>
                        <p><strong>Latency Sensitivity:</strong> {model["latency_sensitivity"]}</p>
                    </div>
                    <div style="flex: 1;">
                        <p><strong>Input Shape:</strong> {model["input_tensor_shape"]}</p>
                        <p><strong>Output Shape:</strong> {model["output_shape"]}</p>
                        <p><strong>Traffic Pattern:</strong> {model["traffic_pattern"]}</p>
                    </div>
                </div>
                <div>
                    <h4>Resource Bottlenecks:</h4>
                    <div style="display: flex; gap: 10px; margin-top: 10px;">
                        <div style="background-color: {"#59BA47" if model["compute_bound"] else "#E9ECEF"}; padding: 5px 15px; border-radius: 15px; color: {"white" if model["compute_bound"] else "#545B64"};">
                            Compute-bound
                        </div>
                        <div style="background-color: {"#59BA47" if model["memory_bound"] else "#E9ECEF"}; padding: 5px 15px; border-radius: 15px; color: {"white" if model["memory_bound"] else "#545B64"};">
                            Memory-bound
                        </div>
                        <div style="background-color: {"#59BA47" if model["io_bound"] else "#E9ECEF"}; padding: 5px 15px; border-radius: 15px; color: {"white" if model["io_bound"] else "#545B64"};">
                            I/O-bound
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a radar chart of performance metrics importance
            metrics = model["performance_metrics"]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=[metrics["latency_importance"], metrics["throughput_importance"], metrics["cost_importance"]],
                theta=["Latency", "Throughput", "Cost"],
                fill='toself',
                name='Performance Priorities',
                line_color='#FF9900'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Performance Priorities",
                height=300,
                margin=dict(l=30, r=30, t=40, b=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Job configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Job Configuration")
            
            job_type = st.radio(
                "Job Type",
                ["Default", "Advanced"],
                index=0
            )
            
            if job_type == "Default":
                st.info("Default jobs provide quick recommendations within minutes, testing on a pre-selected set of instance types.")
            else:
                st.info("Advanced jobs run detailed load tests to simulate your expected traffic patterns, providing more accurate recommendations.")
        
        with col2:
            st.markdown("#### Workload Requirements")
            
            load_profile = st.radio(
                "Traffic Load Profile",
                ["Low", "Moderate", "High"],
                index=1
            )
            
            cost_sensitivity = st.radio(
                "Cost Sensitivity",
                ["Low", "Medium", "High"],
                index=1
            )
        
        # Run the recommendation job
        if st.button("Run Recommendation Job", type="primary"):
            with st.spinner("Running inference recommendation job..."):
                # Simulate job running time
                progress_bar = st.progress(0)
                
                for i in range(100):
                    # Update progress
                    progress_bar.progress(i + 1)
                    time.sleep(0.05)
                
                # Generate recommendation
                results = generate_recommendation(selected_model, load_profile.lower(), cost_sensitivity.lower())
                st.session_state.recommendation_results = results
        
        # Display recommendation results
        if st.session_state.recommendation_results:
            st.divider()
            st.subheader("Recommendation Results")
            
            results = st.session_state.recommendation_results
            
            # Display job summary
            st.markdown(f"""
            <div class="card">
                <h3>Job Summary</h3>
                <div style="display: flex; gap: 20px;">
                    <div style="flex: 1;">
                        <p><strong>Model:</strong> {results["model_details"]["name"]}</p>
                        <p><strong>Job Type:</strong> {job_type}</p>
                        <p><strong>Job Duration:</strong> {results["timestamps"]["duration_minutes"]} minutes</p>
                    </div>
                    <div style="flex: 1;">
                        <p><strong>Traffic Profile:</strong> {load_profile} ({results["load_profile"]["requests_per_minute"]} requests/minute)</p>
                        <p><strong>Cost Sensitivity:</strong> {cost_sensitivity}</p>
                        <p><strong>Monthly Requests:</strong> {results["load_profile"]["monthly_requests"]:,}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create visualizations
            charts = create_recommendation_visualization(results)
            
            if charts:
                # Display overall score chart
                st.plotly_chart(charts["score_chart"], use_container_width=True)
                
                # Display weights chart and detailed comparison side by side
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.plotly_chart(charts["weights_chart"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(charts["detail_chart"], use_container_width=True)
            
            # Display recommendation cards
            st.subheader("Recommended Instance Types")
            
            for rec in results["recommendations"]:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-card-header">
                        <div style="display: flex; align-items: center;">
                            <div class="result-rank">{rec["rank"]}</div>
                            <div style="margin-left: 15px; font-size: 18px; font-weight: bold;">{rec["instance_type"]}</div>
                        </div>
                        <div>
                            <span style="font-size: 20px; font-weight: bold; color: #00A1C9;">{rec["score"]}%</span>
                            <span style="color: #545B64; margin-left: 5px;">Score</span>
                        </div>
                    </div>
                    <div style="display: flex; margin-bottom: 20px;">
                        <div style="flex: 1; text-align: center; padding: 10px; border-right: 1px solid #E9ECEF;">
                            <div style="font-size: 22px; font-weight: bold; color: #00A1C9;">{rec["latency_ms"]}</div>
                            <div style="color: #545B64;">Latency (ms)</div>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 10px; border-right: 1px solid #E9ECEF;">
                            <div style="font-size: 22px; font-weight: bold; color: #59BA47;">{rec["throughput_per_second"]}</div>
                            <div style="color: #545B64;">Requests/Second</div>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 10px;">
                            <div style="font-size: 22px; font-weight: bold; color: #FF9900;">${rec["monthly_cost"]}</div>
                            <div style="color: #545B64;">Monthly Cost</div>
                        </div>
                    </div>
                    <div style="display: flex;">
                        <div style="flex: 1;">
                            <div style="margin-bottom: 5px; color: #545B64;">CPU Utilization</div>
                            <div style="height: 8px; background-color: #E9ECEF; border-radius: 4px;">
                                <div style="height: 8px; width: {rec["cpu_utilization"]}%; background-color: {"#59BA47" if rec["cpu_utilization"] < 80 else "#FF9900" if rec["cpu_utilization"] < 90 else "#D13212"}; border-radius: 4px;"></div>
                            </div>
                            <div style="text-align: right; font-size: 12px;">{rec["cpu_utilization"]}%</div>
                        </div>
                        <div style="flex: 1; margin-left: 20px;">
                            <div style="margin-bottom: 5px; color: #545B64;">Cost per 1M Requests</div>
                            <div style="font-size: 18px; font-weight: bold;">${rec["cost_per_1M"]}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display instance comparison chart
            st.subheader("Performance Comparison")
            
            # Get recommended instance types
            recommended_instances = [rec["instance_type"] for rec in results["recommendations"]]
            
            # Create instance comparison chart
            comparison_chart = create_instance_comparison_chart(selected_model, recommended_instances)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Display cost vs performance chart
            st.subheader("Cost vs. Performance Analysis")
            
            # Create cost vs performance chart
            cost_perf_chart = create_cost_performance_chart(selected_model)
            st.plotly_chart(cost_perf_chart, use_container_width=True)
            
            # Deployment recommendations
            st.subheader("Deployment Recommendations")
            
            st.markdown("""
            <div class="warning-box">
            <h3>Next Steps</h3>
            <p>Based on the recommendation analysis, consider these deployment options:</p>
            <ol>
                <li><strong>Deploy to your recommended instance type</strong> using SageMaker endpoints</li>
                <li><strong>Configure auto-scaling</strong> to handle traffic variations efficiently</li>
                <li><strong>Set up model monitoring</strong> to track performance and detect drift</li>
                <li><strong>Implement multi-model endpoints</strong> if serving multiple models with similar requirements</li>
                <li><strong>Consider serverless inference</strong> for unpredictable or sporadic workloads</li>
            </ol>
            <p>Periodically re-evaluate your instance choices as your models and traffic patterns evolve.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample deployment code
            st.subheader("Sample Deployment Code")
            
            # Get top recommendation
            top_instance = results["recommendations"][0]["instance_type"]
            
            st.code(f"""
# Deploy model to recommended instance type
import boto3
import sagemaker
from sagemaker.model import Model

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create model
model = Model(
    image_uri='<ecr-image-uri>',
    model_data='s3://my-bucket/model-artifacts/model.tar.gz',
    role=role
)

# Deploy model to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='{top_instance}',
    endpoint_name='my-optimized-endpoint'
)

# Configure autoscaling (optional)
client = boto3.client('application-autoscaling')

client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/my-optimized-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=5
)

client.put_scaling_policy(
    PolicyName='MyScalingPolicy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/my-optimized-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={{
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {{
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }},
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }}
)

print(f"Model deployed to {top_instance} with autoscaling configured")
            """)
    
    # Add footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
