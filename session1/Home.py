import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io
import utils.common as common
import utils.authenticate as authenticate

# Set page config
st.set_page_config(
    page_title="ML Engineer - Associate Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def custom_header(text, level="main"):
    """Display a custom header with appropriate styling."""
    if level == "main":
        st.markdown(f'<div class="main-header">{text}</div>', unsafe_allow_html=True)
    elif level == "sub":
        st.markdown(f'<div class="sub-header">{text}</div>', unsafe_allow_html=True)
    elif level == "section":
        st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def info_box(text, box_type="info"):
    """Create a custom information box with styling."""
    box_class = f"{box_type}-box"
    st.markdown(f"""
        <div class="{box_class}">
            <div markdown="1">
                {text}
        """, unsafe_allow_html=True)


def definition_box(term, definition):
    """Display a definition box for key terminology."""
    st.markdown(f"""
    <div class="definition">
        <strong>{term}:</strong> {definition}
    </div>
    """, unsafe_allow_html=True)


def reset_session():
    """Reset all session state variables."""
    st.session_state['quiz_score'] = 0
    st.session_state['quiz_attempted'] = False
    st.session_state['name'] = ""
    st.session_state['visited_ML_Lifecycle'] = False
    st.session_state['visited_Data_Collection'] = False
    st.session_state['visited_Data_Transformation'] = False
    st.session_state['visited_Feature_Engineering'] = False
    st.session_state['visited_Data_Integrity'] = False
    st.rerun()


def setup_sidebar():
    """Setup and render the sidebar elements."""
    with st.sidebar:
        st.image("images/mla_badge.png", width=150)
        st.markdown("### ML Engineer - Associate")
        st.markdown("#### Domain 1: Data Preparation")
        
        common.render_sidebar()


def render_home_tab():
    """Render the Home tab content."""
    custom_header("AWS Partner Certification Readiness")
    st.markdown("## Machine Learning Engineer - Associate")
    
    st.markdown("### Domain 1: Data Preparation for Machine Learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        info_box("""
        This interactive e-learning application covers the main topics of Domain 1 from the AWS Machine Learning Engineer - Associate certification.
        
        Domain 1 focuses on **Data Preparation for Machine Learning**, which accounts for a significant portion of the certification exam.
        
        Navigate through the content using the tabs above to learn about:
        - Machine Learning Lifecycle
        - Data Collection
        - Data Transformation
        - Feature Engineering
        - Data Integrity
        
        Test your knowledge with the quiz when you're ready!
        """, "info")
        
        st.markdown("### Learning Outcomes")
        st.markdown("""
        By the end of this module, you will be able to:
        - Explain the ML lifecycle and its key phases
        - Understand data collection and storage options on AWS
        - Perform data transformation and preprocessing
        - Apply feature engineering techniques
        - Ensure data integrity and prepare data for modeling
        - Identify appropriate AWS services for each phase of data preparation
        """)
    
    with col2:
        st.image("images/mla_badge_big.png", width=250)
        
        if st.session_state['quiz_attempted']:
            st.success(f"Current Quiz Score: {st.session_state['quiz_score']}/5")
        
        st.info("Use the tabs above to navigate through different sections!")
        
    st.markdown("---")
    
    st.markdown("### Task Statements in Domain 1")
    
    task_col1, task_col2, task_col3 = st.columns(3)
    
    with task_col1:
        st.markdown("#### Task 1.1: Ingest and store data")
        st.markdown("""
        - Identify data sources
        - Select appropriate data formats
        - Choose storage solutions
        - Implement data ingestion pipelines
        """)
    
    with task_col2:
        st.markdown("#### Task 1.2: Transform data and perform feature engineering")
        st.markdown("""
        - Clean and preprocess data
        - Handle missing values and outliers
        - Perform feature extraction
        - Create informative features
        """)
    
    with task_col3:
        st.markdown("#### Task 1.3: Ensure data integrity and prepare for modeling")
        st.markdown("""
        - Handle class imbalance
        - Split datasets appropriately
        - Shuffle and augment data
        - Prepare data for model training
        """)


def render_ml_lifecycle_tab():
    """Render the ML Lifecycle tab content."""
    # Mark as visited
    st.session_state['visited_ML_Lifecycle'] = True
    
    custom_header("Machine Learning Lifecycle")
    
    st.markdown("""
    The machine learning lifecycle encompasses all the stages involved in developing, deploying, and maintaining a machine learning model.
    Understanding this lifecycle is crucial for successful implementation of ML projects.
    """)
    
    # ML Lifecycle diagram
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image("images/ml_lifecycle.png", caption="Machine Learning Lifecycle")
    
    with col2:
        info_box("""
        Key phases in the ML Lifecycle:
        1. Frame the ML problem
        2. Prepare data
        3. Model development
        4. Deploy
        5. Monitor
        
        At Amazon, we use a process called "working backwards" which focuses on outcomes and questions that clarify these outcomes.
        """, "tip")
    
    custom_header("ML Lifecycle Detailed", "sub")
    
    st.markdown("""
    The machine learning process always starts with a business problem and data related to that problem. It's important to consider whether 
    the business problem can truly be framed as a machine learning problem before proceeding with the ML lifecycle.
    """)
    
    with st.expander("Business Problem Framing Questions"):
        st.markdown("""
        Before starting any ML project, ask these key questions:
        
        - What business problem are you experiencing?
        - Who is experiencing the problem?
        - When and where is the problem experienced?
        - How is the problem identified or manifested?
        - Why must the problem be resolved?
        - What happens if the problem isn't solved?
        - What are the goals for resolving the problem?
        
        Then assess if ML is appropriate:
        
        - Is your problem domain well known enough for business rules instead of ML?
        - How many unknown variables do you have? (If countable on two hands, it may not be an ML problem)
        - Can the business tolerate errors in prediction?
        - How often does the context for the decision change?
        - How stable are the parameters?
        
        Remember: Most business problems may not require ML and can be resolved with traditional programming approaches.
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. Frame the ML Problem
        - Identify business problem
        - Define ML objectives
        - Determine evaluation metrics
        - Establish success criteria
        - Assess whether ML is the right approach
        
        ### 2. Prepare Data
        - Collect data from various sources
        - Pre-process data (cleaning, handling missing values)
        - Analysis and visualization
        - Feature Engineering
        - Split Dataset (training, validation, testing)
        """)
    
    with col2:
        st.markdown("""
        ### 3. Model Development
        - Select algorithm based on problem type
        - Train model using training dataset
        - Evaluate performance with validation set
        - Tune hyperparameters iteratively
        - Test final model on unseen test data
        
        ### 4. Deploy
        - Implement model in production environment
        - Integrate with applications and systems
        - Set up inference endpoints
        - Ensure scalability and performance
        
        ### 5. Monitor
        - Track model performance metrics
        - Detect concept drift and data drift
        - Debug errors and issues
        - Retrain as needed with fresh data
        - Maintain model relevance over time
        """)
    
    custom_header("The Iterative Nature of ML", "section")
    
    st.markdown("""
    Machine learning is highly iterative. You may need to revisit earlier phases as you discover new insights:
    
    - If business goals aren't met, you might need to collect more data (data augmentation)
    - Feature augmentation might be needed to improve model performance
    - Model drift over time requires retraining with new data
    - Monitoring feedback from production helps refine the model
    """)
    
    st.image("https://miro.medium.com/max/1400/1*Nv2NNALuokZEcV6hYEHdGA.png", caption="The Iterative ML Process")
    
    custom_header("AWS AI/ML Stack", "section")
    
    st.image("images/ml_stack.png", caption="AWS AI/ML Stack")
    
    info_box("""
    **The AWS Machine Learning Stack consists of 3 layers:**
    
    1. **AI Services Layer**: Requires no ML expertise. Pretrained and auto-trained models for specific ML problems like vision, speech, text, and more. Developers point their data to REST APIs to receive specific outputs.
    
    2. **ML Services Layer**: Amazon SageMaker enables labeling, building, training, and deploying machine learning models. SageMaker removes the heavy lifting from each step of the ML process.
    
    3. **ML Frameworks and Infrastructure Layer**: Addresses highly complex ML problems with flexibility but requires in-depth ML expertise. Includes support for popular frameworks like PyTorch, TensorFlow, and MXNet.
    """, "info")
    
    custom_header("Amazon SageMaker AI", "section")
    
    st.image("images/sagemaker_workflow.png", caption="Amazon SageMaker Workflow", width=800)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### SageMaker AI Components")
        st.markdown("""
        **PREPARE:**
        - Data Wrangler: Aggregate and prepare data for ML
        - Ground Truth: Create high-quality datasets
        - Feature Store: Store, catalog, search features
        - Processing: Built-in Python, BYO R/Spark
        - Clarify: Detect bias and understand predictions
        
        **BUILD:**
        - Studio Notebooks & Instances: Fully managed Jupyter Notebooks
        - Studio Lab: Free ML development environment
        - Built-in Algorithms: Tabular, NLP, and vision algorithms
        - JumpStart: UI-based discovery, training, deployment
        - Autopilot: Automatically create ML models
        """)
    
    with col2:
        st.markdown("""
        **TRAIN & TUNE:**
        - Fully Managed Training: Broad hardware options
        - Distributed Training Libraries: High-performance training
        - Training Compiler: Faster deep learning model training
        - Automatic Model Tuning: Hyperparameter optimization
        - Managed Spot Training: Reduce cost up to 90%
        - Debugger and Profiler: Debug and profile training
        
        **DEPLOY & MANAGE:**
        - Fully Managed Deployment: Low latency, high throughput
        - Options: Real-Time, Serverless, Asynchronous, Batch
        - Multi-Model & Container Endpoints: Reduce costs
        - Shadow Testing: Validate model performance in production
        - Inference Recommender: Automatically select compute
        - Model Monitor: Maintain accuracy of deployed models
        """)
    
    info_box("""
    **Canvas, MLOps, and Governance:**
    
    - **Amazon SageMaker Canvas**: Generate ML predictions without code
    - **MLOps**: Pipelines, Projects, and Model Registry for workflow automation and CI/CD
    - **Governance**: Model cards, Dashboard, Permissions for oversight
    """, "success")


def render_data_collection_tab():
    """Render the Data Collection tab content."""
    # Mark as visited
    st.session_state['visited_Data_Collection'] = True
    
    custom_header("Data Collection")
    
    st.markdown("""
    Data collection is the foundation of any machine learning project. This phase involves gathering, storing,
    and organizing the data that will be used to train and test machine learning models.
    """)
    
    custom_header("Fundamentals of Data Collection", "sub")
    
    st.markdown("""
    Effective data collection requires understanding:
    
    1. **Data Sources**: Where your data comes from (databases, APIs, logs, sensors, etc.)
    2. **Data Volume**: How much data is needed for effective model training
    3. **Data Variety**: Different types and formats of data required
    4. **Data Velocity**: How quickly data is generated and needs processing
    5. **Data Quality**: Ensuring collected data is accurate and relevant
    """)
    
    custom_header("Data Structure and Types", "sub")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Structured Data")
        st.image("images/structuredata.png", width=200)
        st.markdown("""
        - Organized in tabular format
        - Examples: SQL databases, spreadsheets
        - Well-defined schema
        - Easily queryable
        - Fixed fields and record lengths
        """)
    
    with col2:
        st.markdown("### Semi-structured Data")
        st.image("images/semistructuredata.png", width=200)
        st.markdown("""
        - Has organizational properties but not in relational databases
        - Examples: XML, JSON, NoSQL databases
        - Flexible schema
        - Tags and attributes with hierarchy
        - Self-describing structure
        """)
    
    with col3:
        st.markdown("### Unstructured Data")
        st.image("images/unstructuredata.png", width=200)
        st.markdown("""
        - No pre-defined model or organization
        - Examples: Text files, social media posts, images, videos
        - Requires advanced processing
        - Often requires feature extraction techniques
        - Makes up 80-90% of all data
        """)
    
    custom_header("Storage Formats", "section")
    
    st.markdown("""
    Choosing the right storage format is crucial for efficiency in data processing and analysis.
    Different data formats impact the efficiency of storage and speed of analysis, and should match the data structure and operations needed.
    """)
    
    data = {
        'Features': ['Data storage', 'Write performance', 'Read performance', 'Block compression', 'Schema evolution', 'Use cases'],
        'CSV': ['Row', 'Fast', 'Slow', '', '', 'Simple data interchange, logs'],
        'Avro RecordIO': ['Row', 'Medium', 'Medium', 'X', 'X', 'Machine learning data storage'],
        'ORC': ['Column', 'Slow', 'Fast', 'X', 'X', 'Big data processing, Hive/SQL optimizations'],
        'Parquet': ['Column', 'Slow', 'Fast', 'X', 'X', 'Big data processing, analytical queries'],
        'JSON': ['object-notation', 'Medium', 'Medium', '', '', 'Flexible data exchange, Web applications'],
        'JSONL': ['object-notation', 'Fast', 'Medium', '', '', 'Logs, stream processing']
    }
    
    df = pd.DataFrame(data)
    st.table(df)
    
    with st.expander("Storage Formats Explained"):
        st.markdown("""
        ### Row-based Formats
        
        **CSV (Comma-Separated Values)**
        - Lightweight, space-efficient text files for tabular data
        - Each line is a row with columns separated by commas
        - Simple format for storing different data types
        - Less efficient for analytics compared to columnar formats
        
        **Avro RecordIO**
        - Row-based storage that stores records sequentially
        - Benefits ML workloads that need to iterate over datasets multiple times
        - Defines a schema that structures the data
        - Improves data processing speeds compared to schema-less formats
        
        ### Column-based Formats
        
        **Parquet**
        - Columnar storage typically used in analytics and data warehousing
        - Beneficial for ML workloads due to compression capabilities
        - Improves both storage space and performance
        - Ideal for analytical queries on specific columns
        
        **ORC (Optimized Row Columnar)**
        - Similar to Parquet, used in big data workloads
        - Used with Apache Hive and Spark
        - Efficient compression and improved performance
        - Widely chosen for ML workloads due to performance benefits
        
        ### Object-notation Formats
        
        **JSON (JavaScript Object Notation)**
        - Document-based format that's human and machine readable
        - Flexible data structure makes it suitable for ML
        - Compact, hierarchical, and easy to parse
        - Represented in objects and arrays with key-value pairs
        
        **JSONL (JSON Lines)**
        - JSON objects separated by new lines instead of being nested
        - Improves efficiency for processing individual objects
        - Better handling of large datasets for ML workloads
        - Can map to columnar formats like Parquet for additional benefits
        """)
    
    custom_header("Data Ingestion", "section")
    
    st.markdown("""
    Data ingestion is the process of importing data from various sources into storage systems for immediate use or for later analysis. 
    AWS offers several services for real-time and batch data ingestion.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Real-time Data Streaming")
        st.image("images/realtime-data-processing.png", width=600)
        st.markdown("""
        **Amazon Kinesis**
        - Kinesis Data Streams: Collect and process large streams of data records in real time
        - Kinesis Data Firehose: Easily deliver streaming data to Amazon S3, Redshift, and more
        - Amazon Managed Service for Apache Flink: Process and analyze data using SQL
        
        **Amazon MSK (Managed Streaming for Apache Kafka)**
        - Fully managed, highly available Apache Kafka service
        - MSK Connect: Run fully managed Kafka Connect workloads
        """)
    
    with col2:
        st.markdown("### Batch Processing and Data Transfer")
        st.image("images/batch_data_processing.png", width=600)
        st.markdown("""
        **AWS CLI & SDKs**
        - Command line interface and software development kits for interacting with AWS services
        
        **AWS Glue**
        - Fully managed extract, transform, and load (ETL) service
        - Connects to various data sources and prepares data for analysis
        
        **AWS Database Migration Service (DMS)**
        - Migrates databases to AWS with minimal downtime
        - Extracts data in various formats (SQL, JSON, CSV, XML)
        
        **AWS DataSync**
        - Efficiently transfer data between on-premises systems and AWS services
        
        **AWS Snowball**
        - Physical device for transferring large amounts of data into and out of AWS
        - Used when network transfers are infeasible
        """)
    
    with st.expander("More About Amazon Kinesis"):
        st.markdown("""
        **Amazon Kinesis** offers a suite of services for real-time data streaming and processing:
        
        1. **Amazon Kinesis Data Streams**
           - Collects and processes large streams of data records in real-time
           - Scales to handle terabytes of data per hour
           - Stores data for up to 365 days
           - Applications can access data with millisecond latency
        
        2. **Amazon Kinesis Data Firehose**
           - Easiest way to load streaming data into data stores
           - Automatically scales to match throughput
           - Can transform data before delivery
           - No need to write applications or manage resources
        
        3. **Amazon Managed Service for Apache Flink**
           - Process and analyze streaming data using SQL, Python, or Java
           - Formerly called "Amazon Kinesis Data Analytics"
           - Run standard Apache Flink applications
           - Automatically provisions resources and scales infrastructure
        """)
    
    with st.expander("More About Amazon MSK"):
        st.markdown("""
        **Amazon Managed Streaming for Apache Kafka (MSK)** provides a fully managed Apache Kafka service:
        
        1. **Core MSK Service**
           - Fully managed Apache Kafka clusters
           - Automatic patching and maintenance
           - High availability with multi-AZ deployment
           - Secure by default with encryption and authentication
        
        2. **Amazon MSK Connect**
           - Runs fully managed Kafka Connect workloads
           - Makes it easy to stream data to and from Kafka
           - Supports common connectors for databases, file systems, and search indexes
           - Automatically scales connector workers
        
        Apache Kafka is a distributed data store optimized for ingesting and processing streaming data in real-time, allowing systems to publish, subscribe to, and process streams of records.
        """)
    
    custom_header("AWS Storage Services", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/storage_services.png", caption="AWS Storage Services",width=700)
    
    with col2:
        st.markdown("""
        **AWS provides various storage services for ML:**
        
        **Block Storage:**
        - Amazon EBS (Elastic Block Store)
          - High-performance, low-latency block storage
          - Directly attached to Amazon EC2 instances
          - Ideal for hosting pre-trained models
        
        **File Storage:**
        - Amazon EFS (Elastic File System)
        - Amazon FSx (for various file systems)
          - Windows File Server
          - Lustre (high-performance computing)
          - NetApp ONTAP
          - OpenZFS
        
        **Object Storage:**
        - Amazon S3 (Simple Storage Service)
          - Highly scalable, durable object storage
          - Central data lake for ML datasets
          - Versioning for model management
          - Integration with other AWS services
        """)
        
        info_box("Amazon S3 is commonly used for ML datasets due to its scalability, durability, and integration with other AWS services like SageMaker", "tip")
    
    with st.expander("Amazon S3 for Machine Learning"):
        st.markdown("""
        **Amazon S3 (Simple Storage Service)** is a highly scalable, available, and redundant object-storage service accessed through an API. It's particularly well-suited for ML workloads:
        
        **Key Benefits for ML:**
        
        1. **Data Ingestion and Storage**
           - Store large datasets required for ML training
           - Data can be ingested through streaming or batch processing
           - Scalable and durable for large volumes of data
        
        2. **Model Training and Evaluation**
           - Store training and validation datasets
           - Maintain versioning of different model iterations
           - Compare model performance across versions
        
        3. **Integration with AWS ML Services**
           - SageMaker can directly access data in S3 for training
           - Amazon Kinesis can stream data into S3 buckets
           - AWS Glue can process data stored in S3
        
        **Storage Classes:**
        - Standard: Frequent access, lowest latency
        - Intelligent-Tiering: Automatic cost optimization
        - Infrequent Access: Lower cost for less frequently accessed data
        - Glacier: Archival storage for rarely accessed data
        
        **Considerations:**
        - Higher latency compared to local storage
        - Network-based data access
        - May require caching strategies for latency-sensitive workloads
        """)
    
    custom_header("Data Lakes and Warehouses", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Lake")
        st.image("images/datalake.png", width=400)
        st.markdown("""
        **A data lake is a centralized repository that helps you store structured, semi-structured, and unstructured data at any scale.**
        
        **Key Characteristics:**
        - Store data at any scale without transformation
        - No need to define schema upfront (schema-on-read)
        - Support for all data types and formats
        - Commonly built on Amazon S3
        - Ideal for machine learning workloads
        - Enables analytics on raw data
        
        **Benefits for ML:**
        - Preserves all raw data that might be useful for future ML projects
        - Allows experimentation with different data transformations
        - Supports both batch and real-time analytics
        - Cost-effective storage for large datasets
        """)
    
    with col2:
        st.markdown("### Data Warehouse")
        st.image("images/datawarehouse.png", width=400)
        st.markdown("""
        **A data warehouse is a central repository of information coming from one or more data sources, optimized for analytics.**
        
        **Key Characteristics:**
        - Structured data with predefined schema
        - Schema-on-write approach
        - Optimized for analytical queries
        - Examples: Amazon Redshift
        - Used for business intelligence and reporting
        - Often fed by ETL processes
        
        **Benefits for ML:**
        - Clean, transformed data ready for analysis
        - Fast query performance for feature exploration
        - Integration with BI tools for data visualization
        - Ability to join data from multiple sources
        """)
    
    with st.expander("Data Lake vs. Data Warehouse for ML"):
        st.markdown("""
        **When to use a Data Lake for ML:**
        - When you need to store raw, unprocessed data
        - When you're not sure which data will be valuable
        - When you need flexibility in data processing approaches
        - For exploratory data analysis and experimentation
        - When dealing with unstructured data like images or text
        
        **When to use a Data Warehouse for ML:**
        - When working with structured, well-understood data
        - When you need fast query performance for feature engineering
        - For well-defined, repeatable ML pipelines
        - When integrating with business reporting systems
        - For regular retraining of models with consistent data structures
        
        **Hybrid Approach:**
        Many organizations use both:
        1. Store raw data in a data lake
        2. Process and transform selected data for a data warehouse
        3. Use both for different stages of the ML lifecycle
        """)
    
    custom_header("AWS Glue", "section")
    
    st.image("images/glue.png", caption="AWS Glue Workflow",width=800)
    
    st.markdown("""
    **AWS Glue** is a fully managed extract, transform, and load (ETL) service that makes it convenient to prepare and load data for analytics.
    
    **Key Components:**
    
    1. **AWS Glue Data Catalog**
       - Catalog structured and semi-structured data
       - Central metadata repository
    
    2. **AWS Glue Connectors**
       - Connect to various data sources
       - Discover schema with AWS Glue crawlers
    
    3. **AWS Glue DataBrew**
       - Visual data preparation tool
       - No-code data transformation
    
    4. **AWS Glue Studio**
       - Visual ETL development
       - Interactive data transformation
    
    **Data Sources:**
    - Amazon RDS
    - Other databases
    - On-premises data
    - Streaming data
    
    **Destinations:**
    - Data warehouses
    - Operational data stores
    - Data lakes
    - Data mesh
    """)
    
    custom_header("Data Labeling Services", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Amazon Mechanical Turk")
        st.markdown("""
        **Amazon Mechanical Turk** provides access to an on-demand global workforce to quickly complete data transformation microtasks at low cost.
        
        **Key Features:**
        - Access to a diverse, on-demand workforce
        - Human intelligence for tasks difficult for computers
        - Available through UI or API
        - Scalable human judgment without full-time hires
        
        **Use Cases:**
        - Image annotation
        - Text annotation
        - Data collection
        - Data cleanup
        - Transcription
        """)
    
    with col2:
        st.markdown("### Amazon SageMaker Ground Truth")
        st.markdown("""
        **SageMaker Ground Truth** streamlines data labeling by using human workers and machine learning.
        
        **Key Features:**
        - Built-in workflows for data labeling
        - Uses human workers from Mechanical Turk
        - Active learning to reduce manual labeling
        - High-quality training data creation
        
        **Ground Truth Plus:**
        - Fully managed data labeling service
        - Expert labelers
        - Additional data types and formats
        - Faster turnaround times
        - Expert review of labeled datasets
        
        **Use Cases:**
        - Image classification
        - Object detection
        - Text classification
        - Named entity recognition
        """)


def render_data_transformation_tab():
    """Render the Data Transformation tab content."""
    # Mark as visited
    st.session_state['visited_Data_Transformation'] = True
    
    custom_header("Data Transformation")
    
    st.markdown("""
    Data transformation is the process of converting data from one format or structure to another.
    It is a crucial step in the data preparation phase of the machine learning lifecycle, often taking up to 80% of a data scientist's time.
    """)
    
    custom_header("Data Cleaning", "sub")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **Data cleaning** is the process of identifying and correcting (or removing) errors, inconsistencies, 
        and inaccuracies in datasets to improve data quality.
        
        Common data quality issues include:
        - **Missing values**: Empty fields or placeholders like NULL, NaN, or blank
        - **Duplicate records**: Identical or near-identical rows
        - **Outliers**: Values that deviate significantly from the rest of the data
        - **Inconsistent formatting**: Different date formats, casing, or naming conventions
        - **Data entry errors**: Typos or incorrect values
        - **Irrelevant data**: Features that don't contribute to the model
        """)
        
        # Example of data with issues
        data = {
            'Age': [39, 25, 50, 38, 49, 52, 131, 54, 38, 40],
            'Workclass': ['State-gov', 'Private', 'Self-emp-not-inc', 'Private', 'Private', 'Self-emp-not-inc', 'Private', '', 'Private', 'Private'],
            'Education': ['Bachelors', 'Masters', 'Bach', 'HS-grad', '9th', 'HS-grad', 'masters', 'Some-college', 'HS-grad', 'Assoc-voc'],
            'Occupation': ['Adm-clerical', 'Farming-fishing', 'Exec-managerial', 'Handlers-cleaners', 'blank', 'Exec-managerial', 'Prof-specialty', '?', 'Handlers-cleaners', 'Craft-repair'],
            'Hours_per_week': [40, 99, 13, 40, 16, 45, 50, 60, 40, 40],
            'Income': ['<=50K', '>50K', '<=50K', '<=50K', '<=50K', '>50K', '>50K', '>50K', '<=50K', '>50K']
        }
        df = pd.DataFrame(data)
        
        st.markdown("### Example dataset with quality issues:")
        st.dataframe(df, use_container_width=True)
        
        st.markdown("""
        **Issues in this dataset:**
        - Missing values (blank, ?)
        - Inconsistent formatting (Bach vs. Bachelors, masters vs. Masters)
        - Outliers (Age 131, 99 hours per week)
        - Duplicate rows (rows 4 and 9 are similar)
        """)
    
    with col2:
        info_box("""
        **Data Cleaning Techniques:**
        
        1. **Handle Missing Values**
           - Remove rows/columns with missing data
           - Impute with statistical measures (mean, median, mode)
           - Use advanced imputation methods (KNN, regression)
           - Use placeholder values for categorical data
        
        2. **Remove Duplicates**
           - Identify exact or nearly identical records
           - Remove or merge duplicates
           - Check for partial duplications
        
        3. **Handle Outliers**
           - Detect using statistical methods (z-score, IQR)
           - Remove extreme values
           - Cap or transform outliers
           - Analyze outliers for potential insights
        
        4. **Standardize Formats**
           - Consistent date formats (YYYY-MM-DD)
           - Consistent text case (uppercase/lowercase)
           - Standardize categorical values
           - Normalize units of measurement
        
        5. **Correct Errors**
           - Fix typos and inconsistencies
           - Validate values against constraints
           - Use business rules to identify errors
           - Cross-reference with reliable sources
        """, "info")
        
        st.image("images/datacleaning.png", caption="Data Cleaning Process")
    
    custom_header("Identifying Outliers", "section")
    
    st.markdown("""
    **Outliers** are data points that differ significantly from the majority of the data. They can be caused by 
    natural variations or errors in data collection.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Types of Outliers")
        st.markdown("""
        **Natural Outliers**
        - Accurate representations of extreme but valid data points
        - Example: An extremely tall individual in a height dataset
        - Should be kept but may need special handling
        
        **Artificial Outliers**
        - Anomalous data points due to error or improper collection
        - Example: A body temperature that is unrealistically high
        - Often need to be corrected or removed
        """)
        
        st.markdown("### Detecting Outliers")
        st.markdown("""
        **Statistical Methods:**
        - Z-score (standard deviations from mean)
        - IQR (Interquartile Range) method
        - DBSCAN clustering
        - Isolation Forest
        
        **Visual Methods:**
        - Box plots
        - Scatter plots
        - Histograms
        """)
    
    with col2:
        # Create sample data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 100)
        data_with_outliers = np.append(normal_data, [0, 5, 95, 100])
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data_with_outliers, ax=ax)
        ax.set_title('Box Plot Showing Outliers')
        ax.set_xlabel('Values')
        st.pyplot(fig)
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data_with_outliers, kde=True, ax=ax)
        ax.set_title('Histogram Showing Distribution with Outliers')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    st.markdown("""
    **Methods for Handling Outliers:**
    
    1. **Keep them**: If they are valid data points that represent natural variation
    2. **Remove them**: If they are errors or if their presence significantly impacts analysis
    3. **Transform them**: Apply logarithmic or other transformations to compress the range
    4. **Cap them**: Set maximum and minimum thresholds (Winsorizing)
    5. **Separate analysis**: Analyze outliers separately to understand their causes
    """)
    
    custom_header("Categorical Encoding", "section")
    
    st.markdown("""
    Categorical encoding is the process of converting categorical variables into a format that machine learning algorithms can work with.
    Most ML algorithms require numerical input, so categorical data needs to be converted to numbers.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Types of Categorical Data")
        st.markdown("""
        - **Binary**: Two possible values (Yes/No, True/False)
        - **Nominal**: Categories with no order (e.g., colors, cities)
        - **Ordinal**: Categories with natural order (e.g., small, medium, large)
        """)
        
        # Visual representation of types
        categories = ['Binary', 'Nominal', 'Ordinal']
        examples = ['Yes/No', 'Red/Blue/Green', 'Small/Medium/Large']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(categories, [1, 2, 3], color=['#FF9900', '#232F3E', '#1E88E5'])
        
        for i, (cat, ex) in enumerate(zip(categories, examples)):
            ax.text(i, 0.5, ex, ha='center', color='white', fontweight='bold')
        
        ax.set_ylim(0, 3.5)
        ax.set_title('Types of Categorical Variables')
        ax.set_yticks([])
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Label Encoding")
        st.image("images/label_encoding.png", caption="Label Encoding",width=300)
        st.markdown("""
        - Maps each category to a number
        - Good for ordinal data where order matters
        - Example: Small → 1, Medium → 2, Large → 3
        - Warning: Creates implied ordering for nominal data
        - Simple implementation with scikit-learn's LabelEncoder
        """)
    
    with col3:
        st.markdown("### One-Hot Encoding")
        st.image("images/one-hot-encoding.png", caption="One-Hot Encoding")
        st.markdown("""
        - Creates binary columns for each category
        - Good for nominal data with no inherent order
        - Avoids false relationships between categories
        - Example: Red → [1,0,0], Green → [0,1,0], Blue → [0,0,1]
        - Can lead to high dimensionality with many categories
        - Implemented in pandas with get_dummies()
        """)
    
    info_box("""
    **When to use which encoding method:**
    
    - **Label Encoding**: Best for ordinal data where order matters (education levels, ratings)
    
    - **One-Hot Encoding**: Best for nominal data with no inherent order (colors, product categories)
    
    - **Binary Encoding**: Useful when there are many categories; converts to binary representation
    
    - **Frequency or Target Encoding**: Replaces categories with statistical measures related to the target
    
    - **Embedding**: For high-cardinality categorical features in deep learning models
    """, "tip")
    
    with st.expander("Code Example: Categorical Encoding"):
        st.code("""
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        
        # Sample data
        data = {
            'Color': ['Red', 'Blue', 'Green', 'Red', 'Green'],
            'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small']
        }
        df = pd.DataFrame(data)
        
        # Label Encoding (for ordinal data - Size)
        le = LabelEncoder()
        df['Size_Encoded'] = le.fit_transform(df['Size'])
        
        # One-Hot Encoding (for nominal data - Color)
        color_dummies = pd.get_dummies(df['Color'], prefix='Color')
        df = pd.concat([df, color_dummies], axis=1)
        
        print(df)
        """, language="python")
    
    custom_header("AWS Services for Data Transformation", "section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Amazon SageMaker Data Wrangler")
        st.image("images/data-wrangler.png", width=600)
        st.markdown("""
        **Key Features:**
        - Visual, no-code interface for data preparation
        - 300+ built-in transformations
        - Connect to multiple data sources
        - Data quality and visualization capabilities
        - Integration with SageMaker ecosystem
        - Reusable data flow templates
        
        **Benefits:**
        - Reduces time spent on data preparation
        - No coding required for common transformations
        - Streamlines ML workflow
        - Ensures data quality before model training
        """)
    
    with col2:
        st.markdown("### AWS Glue")
        st.image("images/glue.png", width=600)
        st.markdown("""
        **Key Features:**
        - Fully managed ETL service
        - Connect to various data sources
        - Transform data with Apache Spark
        - Catalog data with AWS Glue Data Catalog
        - Create ETL jobs with minimal coding
        
        **AWS Glue Components:**
        - **Glue Data Catalog**: Central metadata repository
        - **Glue Crawlers**: Automatically discover schema
        - **Glue ETL**: Transform data using Spark or Python
        - **Glue DataBrew**: Visual data preparation
        - **Glue Studio**: Visual ETL development
        """)
    
    st.markdown("### Other AWS Services for Data Transformation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### AWS Glue DataBrew")
        st.markdown("""
        - Visual data preparation tool
        - 250+ pre-built transformations
        - No coding required
        - Visualize data quality
        - Profile datasets automatically
        """)
    
    with col2:
        st.markdown("#### AWS Lambda")
        st.markdown("""
        - Serverless compute service
        - Run lightweight data processing
        - Real-time data transformation
        - Event-driven architecture
        - Pay only for compute time used
        """)
    
    with col3:
        st.markdown("#### Amazon EMR")
        st.markdown("""
        - Managed Hadoop framework
        - Run Spark, Hive, Presto
        - Process large-scale data
        - Advanced transformations
        - Cost-effective big data processing
        """)
    
    with col4:
        st.markdown("#### AWS Batch")
        st.markdown("""
        - Run batch computing workloads
        - Schedule data processing jobs
        - Dynamic resource provisioning
        - Priority-based job queues
        - Integration with AWS Step Functions
        """)
    
    with st.expander("AWS Glue DataBrew In-Depth"):
        st.markdown("""
        **AWS Glue DataBrew** is a visual data preparation tool that makes it easy for data analysts and data scientists to clean and normalize data for analytics and machine learning.
        
        **Key Features:**
        
        1. **Visual Data Preparation**
           - No-code UI for data transformation
           - 250+ pre-built transformations
           - Live previews of transformations
           - Recipe-based approach for reusable steps
        
        2. **Data Quality & Profiling**
           - Automatic data profiling
           - Data quality validation
           - Anomaly detection
           - Statistical summaries
        
        3. **Data Transformation Capabilities**
           - Handle missing values
           - Normalize and standardize data
           - Filter and sort records
           - Join and pivot datasets
           - Aggregate and group data
        
        4. **Integration**
           - Works with data in Amazon S3, data lakes, and databases
           - Export transformations to AWS Glue ETL jobs
           - Integration with SageMaker for ML workflows
           - Publish cleaned data to various destinations
        
        5. **Security & Governance**
           - IAM role-based access control
           - Data encryption in transit and at rest
           - Audit logs for tracking changes
           - Versioning for recipes
        
        **Use Cases:**
        - Preparing data for machine learning models
        - Cleaning data for analytics and dashboards
        - Standardizing data from multiple sources
        - Automating regular data preparation tasks
        """)


def render_feature_engineering_tab():
    """Render the Feature Engineering tab content."""
    # Mark as visited
    st.session_state['visited_Feature_Engineering'] = True
    
    custom_header("Feature Engineering")
    
    st.markdown("""
    Feature engineering is the process of transforming raw data into features that better represent the underlying problem to predictive models,
    resulting in improved model accuracy on unseen data.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Feature engineering is often considered both an art and a science:
        
        - It requires domain knowledge and creativity
        - It involves transforming raw data into a format that ML algorithms can work with
        - Well-engineered features can significantly improve model performance
        - Feature engineering is often the most time-consuming part of ML projects
        - Different algorithms benefit from different types of feature engineering
        """)
    
    with col2:
        info_box("""
        **Key Terminology:**
        
        - **Feature**: An attribute or independent variable used in a predictive model (also called predictor)
        
        - **Target/Label**: What you're trying to predict (dependent variable or response)
        
        - **Feature Engineering**: Process of reshaping data to get more value out of it
        
        - **Feature Selection**: Process of selecting specific features that have the most valuable data
        """, "info")
    
    definition_box("Feature Engineering", "The process of using domain knowledge and data transformations to create new features that make machine learning algorithms work better.")
    
    custom_header("The Curse of Dimensionality", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        The **curse of dimensionality** refers to the problems that arise when analyzing data in high-dimensional spaces.
        
        **Key Challenges:**
        
        - The amount of data needed to generalize accurately grows exponentially with dimensions
        - Models become more complex and prone to overfitting
        - Distance metrics become less meaningful in high dimensions
        - Computational resources required increase significantly
        - Classification performance typically declines with too many dimensions
        
        **This is why feature selection and dimensionality reduction are important parts of feature engineering**
        """)
        
        st.image("images/curse_dimensionality.png", caption="Curse of Dimensionality",width=600)
    
    with col2:
        st.markdown("### Impact on Model Performance")
        
        # Create sample data to show impact
        dims = np.arange(1, 101, 10)
        accuracy = 90 - 20 * np.log10(dims/10)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(dims, accuracy)
        ax.set_xlabel('Number of Features (Dimensions)')
        ax.set_ylabel('Model Performance')
        ax.set_title('Impact of Dimensionality on Performance')
        ax.grid(True)
        
        st.pyplot(fig)
        
        info_box("""
        **Best Practice:**
        
        Ideally, you should try to include as many meaningful features as possible before performance declines.
        
        Feature engineering and selection help balance this tradeoff by:
        
        - Creating more informative features
        - Removing redundant or irrelevant features
        - Reducing noise in the data
        """, "tip")
    
    custom_header("Feature Engineering Techniques", "sub")
    
    st.markdown("""
    Different types of feature engineering techniques can be applied depending on the data type and the specific problem.
    The goal is to create features that capture the important patterns in the data relevant to the prediction task.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Feature Creation")
        st.markdown("""
        Creating new features based on:
        - Domain knowledge
        - Data patterns
        - Mathematical combinations
        - Interaction terms
        - Rolling statistics (for time series)
        
        **Examples:**
        - Ratio of height to weight (BMI)
        - Days since last purchase
        - Distance between geographic points
        - Total household income from multiple sources
        - Age derived from birth date
        """)
        
        st.markdown("### Feature Transformation")
        st.markdown("""
        Changing the scale or distribution of features:
        - Log transformation for skewed data
        - Polynomial features for non-linear relationships
        - Box-Cox transformation for normalization
        - Binning continuous variables
        
        **Examples:**
        - Log of income (reduces skew)
        - Square of age (captures non-linear effects)
        - Sine/cosine transformations of cyclical features
        - Discretizing continuous features into bins
        """)
    
    with col2:
        st.markdown("### Feature Selection")
        st.markdown("""
        Selecting the most relevant features:
        - **Filter methods**: Statistical measures (correlation, chi-square)
        - **Wrapper methods**: Search algorithms (recursive feature elimination)
        - **Embedded methods**: Built into algorithms (LASSO, decision trees)
        
        **Examples:**
        - Selecting top-k features by correlation with target
        - Using feature importance from tree-based models
        - Applying L1 regularization for feature selection
        - Eliminating features with low variance
        """)
        
        st.markdown("### Feature Scaling")
        st.markdown("""
        Normalizing feature ranges:
        - Min-Max Scaling: Scales to [0,1] range
        - Standard Scaling: Z-score normalization (mean=0, std=1)
        - Robust Scaling: Uses median and quartiles (robust to outliers)
        
        **Examples:**
        - Scaling age to [0,1] range using Min-Max scaling
        - Standardizing income to mean=0, std=1
        - Using robust scaling when outliers are present
        """)
    
    custom_header("Feature Scaling Methods Comparison", "section")
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(50, 10, 100)
    data = np.append(data, [5, 95])  # Add outliers
    
    # Create different scalers
    original = data
    min_max = (data - data.min()) / (data.max() - data.min())
    standard = (data - data.mean()) / data.std()
    robust = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
    
    # Plot the distributions
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    
    axs[0].hist(original, bins=20, alpha=0.7)
    axs[0].set_title('Original Data')
    
    axs[1].hist(min_max, bins=20, alpha=0.7)
    axs[1].set_title('Min-Max Scaling [0,1]')
    
    axs[2].hist(standard, bins=20, alpha=0.7)
    axs[2].set_title('Standard Scaling (Z-score)')
    
    axs[3].hist(robust, bins=20, alpha=0.7)
    axs[3].set_title('Robust Scaling (using quartiles)')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Characteristics of each scaling method:**
    
    1. **Min-Max Scaling**
       - Transforms features to a range between 0 and 1
       - Formula: X_scaled = (X - X_min) / (X_max - X_min)
       - Sensitive to outliers
       - Preserves the shape of the original distribution
    
    2. **Standard Scaling (Z-score)**
       - Transforms features to have mean=0 and standard deviation=1
       - Formula: X_scaled = (X - μ) / σ
       - Affected by outliers but less than Min-Max
       - Useful for algorithms that assume features are normally distributed
    
    3. **Robust Scaling**
       - Uses median and interquartile range instead of mean and standard deviation
       - Formula: X_scaled = (X - median(X)) / (Q3 - Q1)
       - Resistant to outliers
       - Better for data with outliers or skewed distributions
    """)
    
    info_box("""
    **Scaling Method Selection:**
    
    - **Min-Max Scaling**: Best when you need values in a specific range [0,1] and data has no significant outliers
    
    - **Standard Scaling**: Best for data with approximately normal distribution and algorithms that assume normality
    
    - **Robust Scaling**: Best when outliers are present or when dealing with skewed distributions
    
    Many algorithms like Support Vector Machines, K-Nearest Neighbors, and Neural Networks benefit significantly from properly scaled features.
    """, "tip")
    
    with st.expander("Code Example: Feature Scaling"):
        st.code("""
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
        # Sample data with outliers
        data = np.random.normal(50, 10, 100)
        data = np.append(data, [5, 95])  # Add outliers
        df = pd.DataFrame({'feature': data})
        
        # Apply different scalers
        min_max_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()
        robust_scaler = RobustScaler()
        
        df['min_max_scaled'] = min_max_scaler.fit_transform(df[['feature']])
        df['standard_scaled'] = standard_scaler.fit_transform(df[['feature']])
        df['robust_scaled'] = robust_scaler.fit_transform(df[['feature']])
        
        print(df.describe())
        """, language="python")
    
    custom_header("Feature Engineering for Different Data Types", "section")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Numeric Features")
        st.markdown("""
        **Techniques:**
        - Mathematical transformations (log, square root)
        - Binning/discretization
        - Polynomial features
        - Interaction terms
        - Rolling statistics (for time series)
        
        **Example:**
        Converting raw age to age groups or creating age^2 feature to capture non-linear relationships
        """)
    
    with col2:
        st.markdown("### Categorical Features")
        st.markdown("""
        **Techniques:**
        - Encoding (one-hot, label, target)
        - Feature hashing
        - Entity embeddings
        - Count/frequency encoding
        - Grouping rare categories
        
        **Example:**
        Converting product categories to embeddings that capture semantic relationships between categories
        """)
    
    with col3:
        st.markdown("### Text Features")
        st.markdown("""
        **Techniques:**
        - Bag of words
        - TF-IDF
        - Word embeddings (Word2Vec, GloVe)
        - N-grams
        - Topic modeling
        - Text statistics (length, counts)
        
        **Example:**
        Converting product descriptions to TF-IDF vectors that capture important terms
        """)
    
    custom_header("Amazon SageMaker Feature Store", "section")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/feature-store-intro-diagram.png", width=700)
        
        st.markdown("""
        **Amazon SageMaker Feature Store** is a purpose-built repository for machine learning features.
        
        **Key capabilities:**
        - Store, discover, and share features securely
        - Online and offline feature storage modes
        - Millisecond latency for online inference
        - Consistent feature values across training and inference
        - Visual search for feature discovery
        - Sharing and collaboration across teams
        - Time travel capabilities to access historical values
        """)
    
    with col2:
        st.markdown("### Benefits of Feature Store")
        st.markdown("""
        - **Reuse features** across multiple models
        - **Reduce duplication** of feature engineering work
        - **Ensure consistency** between training and inference
        - **Track lineage** of features for governance
        - **Reduce latency** for real-time predictions
        - **Centralize feature management** for teams
        - **Version features** over time
        - **Improve governance** with metadata tracking
        """)
        
        info_box("""
        **Dual Storage Pattern:**
        
        SageMaker Feature Store supports both:
        
        - **Online store**: Low-latency, high-availability storage for real-time inference
        - **Offline store**: High-throughput storage for model training and batch inference
        
        This ensures feature consistency across training and inference, eliminating training-serving skew.
        """, "success")
    
    with st.expander("How SageMaker Feature Store Works"):
        st.markdown("""
        **How SageMaker Feature Store Works:**
        
        1. **Create Feature Groups**
           - Define schema with feature definitions
           - Configure online and offline storage options
           - Set up record identifiers and time feature names
        
        2. **Ingest Features**
           - Batch ingestion using DataFrame APIs
           - Streaming ingestion for real-time updates
           - Automatic synchronization between online and offline stores
        
        3. **Retrieve Features**
           - Get online features for real-time inference
           - Query offline features for model training
           - Point-in-time joins for historical feature values
        
        4. **Search and Share**
           - Discover features using the Feature Store Catalog
           - Share features across teams and projects
           - Track feature metadata and lineage
        
        **Example Use Case:**
        
        A recommendation system team creates customer features (demographics, past purchases, browsing history)
        and stores them in Feature Store. Multiple teams can now use these same features for different models
        (product recommendations, churn prediction, customer segmentation), ensuring consistency and reducing
        duplicate work.
        """)


def render_data_integrity_tab():
    """Render the Data Integrity tab content."""
    # Mark as visited
    st.session_state['visited_Data_Integrity'] = True
    
    custom_header("Data Integrity and Preparation for Modeling")
    
    st.markdown("""
    Ensuring data integrity and properly preparing data for modeling are crucial steps before training a model.
    These processes help create high-quality datasets that lead to more accurate and reliable models.
    
    In this section, we'll explore how to handle class imbalance, split datasets appropriately, and use techniques like data shuffling and augmentation to prepare data for modeling.
    """)
    
    custom_header("Class Imbalance", "sub")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **Class imbalance** occurs when the classes in your target variable are not represented equally in your dataset.
        
        For example, in a fraud detection dataset, fraudulent transactions might represent only 0.1% of all transactions.
        
        **Challenges with imbalanced data:**
        - Models tend to favor the majority class
        - Poor performance on minority classes
        - Misleading evaluation metrics
        - Risk of bias against minority groups
        - Critical events often belong to minority classes
        
        **Why it matters:**
        - Models optimize for overall accuracy, which can lead to ignoring minority classes
        - In many business cases (fraud, disease diagnosis), the minority class is the more important one
        - Standard evaluation metrics like accuracy become misleading
        """)
        
        # Create a sample imbalanced dataset visualization
        labels = ['Not Fraud (99.9%)', 'Fraud (0.1%)']
        sizes = [999, 1]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#5DADE2', '#EC7063'])
        ax.axis('equal')
        plt.title('Example: Imbalanced Fraud Detection Dataset')
        
        st.pyplot(fig)
    
    with col2:
        info_box("""
        **Detecting Class Imbalance:**
        
        1. Calculate the ratio of classes
           - Count observations in each class
           - Compute percentage distribution
        
        2. Visualize class distribution
           - Bar charts or pie charts
           - Count plots
        
        3. Check performance metrics by class
           - Confusion matrix
           - Class-specific metrics
        
        **Metrics affected by imbalance:**
        - Accuracy can be misleading (99.9% accuracy by always predicting "not fraud")
        
        **Better metrics for imbalanced data:**
        - Precision: True positives / (True positives + False positives)
        - Recall: True positives / (True positives + False negatives)
        - F1-score: Harmonic mean of precision and recall
        - Area Under ROC Curve (AUC)
        - Precision-Recall Curve
        """, "warning")
        
        st.markdown("### Amazon SageMaker Clarify")
        st.image("https://d1.awsstatic.com/SageMaker-Clarify-How-it-works.3e1739db812b5d9ad35ec0f2164769565df7c749.png", width=400)
        
        st.markdown("""
        **SageMaker Clarify** helps detect bias in your data and models:
        
        - Identify imbalances during data preparation
        - Evaluate bias in trained models
        - Generate automated bias reports
        - Track bias drift over time
        - Support explainability in predictions
        """)
    
    custom_header("Techniques to Handle Class Imbalance", "section")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Random Oversampling")
        st.image("https://miro.medium.com/max/640/1*YPA2BrCmGpBMWy-M27yKKA.png", width=200)
        st.markdown("""
        - Duplicates examples from the minority class
        - Increases minority samples until balanced
        - Easy to implement
        - Risk of overfitting to minority samples
        - No new information added
        
        **In AWS:**
        Available in SageMaker Data Wrangler's balance data transform
        """)
    
    with col2:
        st.markdown("### Random Undersampling")
        st.image("https://miro.medium.com/max/640/1*6Vq_7NT7_lXbmBIirJ_htg.png", width=200)
        st.markdown("""
        - Removes examples from the majority class
        - Reduces majority samples until balanced
        - May lose valuable information
        - Works well with abundant data
        - Less computationally expensive
        
        **In AWS:**
        Available in SageMaker Data Wrangler's balance data transform
        """)
    
    with col3:
        st.markdown("### SMOTE")
        st.image("https://miro.medium.com/max/640/1*c87MvS3jzBY6Ja5xwBZbgA.png", width=200)
        st.markdown("""
        - Synthetic Minority Oversampling Technique
        - Creates synthetic examples from minority class
        - Uses interpolation between existing samples
        - Reduces overfitting compared to random oversampling
        - Adds diversity to minority class
        
        **In AWS:**
        Available in SageMaker Data Wrangler's balance data transform
        """)
    
    st.markdown("""
    **Additional Techniques:**
    
    1. **Algorithm-level Approaches**
       - Cost-sensitive learning: Assign higher cost to minority class errors
       - Class weights: Weight samples inversely proportional to class frequencies
       - Ensemble methods: Techniques like RUSBoost or EasyEnsemble
    
    2. **Advanced Sampling**
       - ADASYN: Adaptive Synthetic Sampling
       - Borderline-SMOTE: Focuses on samples near the decision boundary
       - Cluster-based oversampling: Creates synthetic examples within clusters
    
    3. **Anomaly Detection**
       - For extreme imbalance, treat as anomaly detection
       - One-class classification or novelty detection
    """)
    
    with st.expander("Using SageMaker Data Wrangler for Class Imbalance"):
        st.markdown("""
        **Using Amazon SageMaker Data Wrangler to Balance Data:**
        
        SageMaker Data Wrangler offers built-in transforms for handling class imbalance:
        
        1. **Random Oversampling in Data Wrangler**
           - Automatically oversamples minority class by duplicating samples
           - Simply select the transform and specify target column and desired ratio
        
        2. **Random Undersampling in Data Wrangler**
           - Automatically removes samples from majority class
           - Specify target column and desired balance ratio
        
        3. **SMOTE in Data Wrangler**
           - Generates synthetic minority samples through interpolation
           - Supports both numeric and non-numeric features
           - Non-numeric features are handled by copying from original samples
           - Configure parameters like k-neighbors and sampling ratio
        
        **Implementation Steps:**
        1. Open your data flow in SageMaker Data Wrangler
        2. Add a new transform step
        3. Choose "Balance data" from the transform menu
        4. Select the balancing method (Random oversampling, Random undersampling, or SMOTE)
        5. Configure parameters like target column and sampling strategy
        6. Preview and apply the transformation
        """)
    
    custom_header("Dataset Splitting", "sub")
    
    st.markdown("""
    Dataset splitting is the process of dividing your dataset into subsets for training, validation, and testing.
    Proper splitting ensures your model generalizes well to new, unseen data.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("https://miro.medium.com/max/1400/1*Nv2NNALuokZEcV6hYEHdGA.png", caption="Training, Validation, and Test Splits")
        
        st.markdown("""
        **Common Splitting Ratios:**
        - 70% Training, 15% Validation, 15% Test
        - 80% Training, 10% Validation, 10% Test
        
        **Purpose of each split:**
        
        - **Training data**: Used to train the model parameters
          - Largest portion of the dataset
          - Model learns patterns and relationships from this data
        
        - **Validation data**: Used for tuning hyperparameters and preventing overfitting
          - Not used in training the model parameters
          - Helps select the best model configuration
          - Used for early stopping to prevent overfitting
        
        - **Test data**: Used for final evaluation of model performance
          - Completely held-out data not seen during development
          - Provides unbiased estimate of model performance
          - Simulates real-world performance on unseen data
        """)
    
    with col2:
        st.markdown("### Splitting Techniques")
        
        st.markdown("#### Simple Hold-out")
        st.markdown("""
        - Random split into training, validation, and test sets
        - Simple and commonly used
        - Can be problematic with small datasets
        - May not preserve data distribution
        
        ```python
        from sklearn.model_selection import train_test_split
        
        # First split: training + validation vs test
        train_val, test = train_test_split(
            data, test_size=0.2, random_state=42
        )
        
        # Second split: training vs validation
        train, val = train_test_split(
            train_val, test_size=0.125, random_state=42
        )
        ```
        """)
        
        st.markdown("#### Cross-validation")
        st.markdown("""
        - Data is divided into k folds
        - Model trained k times, each time using k-1 folds for training and 1 fold for validation
        - Results are averaged across all runs
        - More robust estimate of performance
        - Computationally more expensive
        
        ```python
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(data):
            train_data, val_data = data[train_index], data[val_index]
            # Train and evaluate model
        ```
        """)
        
        info_box("""
        **Best Practices for Splitting:**
        
        - Maintain same distribution across splits (stratified sampling)
        - Stratify splits for class balance
        - Consider time-based splits for time series
        - Use same random seed for reproducibility
        - Ensure no data leakage between splits
        """, "tip")
    
    custom_header("Data Shuffling", "section")
    
    st.markdown("""
    **Data shuffling** randomizes the order of examples in your dataset before they are used for training a machine learning model.
    Shuffling ensures that the training examples are presented to the model in a random order rather than in a fixed sequence.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Benefits of Data Shuffling")
        st.markdown("""
        - **Prevents learning sequence-dependent patterns**
          - Model learns actual relationships instead of order
          - Reduces bias from sequential ordering
        
        - **Improves convergence in optimization**
          - Helps stochastic gradient descent converge faster
          - Creates more diverse mini-batches
        
        - **Reduces bias in training**
          - Ensures even exposure to all types of examples
          - Prevents overfitting to specific data sequences
        
        - **Enhances model generalization**
          - Model becomes more robust to different data presentations
          - Less sensitive to ordering effects
        """)
    
    with col2:
        st.markdown("### Data Shuffling Techniques")
        
        st.markdown("""
        **Random Permutation**
        - Randomly swaps positions of all data points
        - Complete reshuffling of the dataset
        - Like thoroughly shuffling a deck of cards
        
        **Epoch-based Shuffling**
        - Reshuffles data at the beginning of each training epoch
        - Provides different data ordering in each pass
        - Standard in most ML frameworks
        
        **Mini-batch Shuffling**
        - Shuffles data before creating mini-batches
        - Ensures diverse examples in each batch
        - Improves training stability
        """)
    
    st.markdown("""
    **Implementation Example:**
    
    ```python
    # Random permutation
    import numpy as np
    from sklearn.utils import shuffle
    
    # For NumPy arrays
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
    
    # For pandas DataFrames
    df_shuffled = df.sample(frac=1, random_state=42)
    
    # In PyTorch DataLoader (epoch-based shuffling)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    ```
    """)
    
    custom_header("Data Augmentation", "section")
    
    st.markdown("""
    **Data augmentation** creates new training examples by applying transformations to existing data. This technique increases
    the diversity and size of the training dataset without actually collecting new data.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Image-based Augmentation")
        st.image("https://miro.medium.com/max/1400/1*mA1blos7f5LZbhWUHZI2dg.png", width=300)
        st.markdown("""
        **Techniques:**
        - Geometric transformations: rotation, flipping, scaling
        - Color space transformations: brightness, contrast
        - Random cropping and zooming
        - Noise injection
        - Image mixing (CutMix, MixUp)
        - Random erasing/cutout
        
        **AWS Services:**
        - SageMaker built-in image classification algorithms
        - SageMaker image preprocessing libraries
        """)
    
    with col2:
        st.markdown("#### Text-based Augmentation")
        st.image("https://miro.medium.com/max/1400/1*zBXBfFMAH4gv-JBzhEDDkA.png", width=300)
        st.markdown("""
        **Techniques:**
        - Synonym replacement
        - Random insertion/deletion/swapping
        - Back-translation
        - Text generation using language models
        - Entity replacement
        - Contextual augmentation
        
        **AWS Services:**
        - Amazon Translate for back-translation
        - Amazon Comprehend for entity recognition
        - SageMaker JumpStart for language models
        """)
    
    with col3:
        st.markdown("#### Time Series Augmentation")
        st.image("https://miro.medium.com/max/1400/1*QpGO_DpGGgcj9c4UoFEQjA.png", width=300)
        st.markdown("""
        **Techniques:**
        - Time warping
        - Magnitude warping
        - Window slicing
        - Jittering (adding noise)
        - Synthetic data generation
        - Trend and seasonality adjustments
        
        **AWS Services:**
        - Amazon Forecast
        - SageMaker time series algorithms
        - SageMaker Processing for custom transformations
        """)
    
    st.markdown("""
    **Benefits of Data Augmentation:**
    
    1. **Increases dataset size**
       - More examples for model training
       - Helps when data collection is expensive or limited
    
    2. **Improves model generalization**
       - Reduces overfitting
       - Makes model robust to variations
    
    3. **Addresses class imbalance**
       - Creates additional examples for minority classes
       - Improves detection of under-represented categories
    
    4. **Enhances model robustness**
       - Makes model invariant to certain transformations
       - Improves performance in real-world conditions
    """)
    
    with st.expander("Advanced Data Augmentation Techniques"):
        st.markdown("""
        **Advanced Data Augmentation Approaches:**
        
        1. **Generative Adversarial Networks (GANs)**
           - Use deep learning to generate entirely new, synthetic examples
           - Particularly effective for images and complex data types
           - Can create highly realistic data
        
        2. **Neural Style Transfer**
           - Applies style of one image to content of another
           - Creates diverse visual variations
           - Useful for artistic and creative applications
        
        3. **Feature Space Augmentation**
           - Performs augmentation in feature space rather than input space
           - Particularly useful for tabular data
           - Can generate more realistic synthetic examples
        
        4. **Automated Augmentation**
           - AutoAugment and RandAugment
           - Uses search algorithms to find optimal augmentation strategies
           - Adapts augmentation policy to specific datasets
        
        5. **Consistency Training**
           - Uses augmentations in semi-supervised learning
           - Enforces consistent predictions across augmented examples
           - Leverages unlabeled data effectively
        
        **Implementation in AWS:**
        
        - Use SageMaker Studio notebooks to implement custom augmentation pipelines
        - Leverage SageMaker Processing jobs for data preprocessing at scale
        - Integrate with SageMaker Training jobs to apply augmentation during model training
        - Use SageMaker Pipelines to orchestrate end-to-end workflows including augmentation steps
        """)
    
    custom_header("Configuration for Model Training", "section")
    
    st.markdown("""
    After handling class imbalance, splitting your dataset, and applying techniques like data shuffling and augmentation,
    it's time to prepare your final configuration for model training.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Preparation Checklist")
        st.markdown("""
        **Before training your model, ensure you have:**
        
        1. **Cleaned your data**
           - Handled missing values
           - Corrected or removed erroneous values
           - Addressed outliers appropriately
        
        2. **Transformed your features**
           - Applied necessary encoding for categorical variables
           - Scaled numerical features appropriately
           - Created informative features through feature engineering
        
        3. **Balanced your dataset**
           - Applied techniques to handle class imbalance
           - Ensured proper representation of all classes
        
        4. **Split your data properly**
           - Created training, validation, and test sets
           - Maintained class distribution across splits
           - Avoided data leakage between splits
        
        5. **Implemented data augmentation**
           - Applied appropriate augmentation for your data type
           - Created diverse training examples
        """)
    
    with col2:
        st.markdown("### Setting Up Training in SageMaker")
        st.markdown("""
        **Amazon SageMaker provides several ways to configure your training jobs:**
        
        1. **Configure Data Sources**
           - Specify S3 locations for training and validation datasets
           - Set up data channels for different data splits
           - Configure input data format (CSV, Parquet, etc.)
        
        2. **Resource Configuration**
           - Select instance type for training
           - Set instance count for distributed training
           - Configure compute resources based on dataset size
        
        3. **Hyperparameter Configuration**
           - Set algorithm-specific hyperparameters
           - Configure learning rate, batch size, etc.
           - Set up hyperparameter tuning jobs if needed
        
        4. **Output Configuration**
           - Specify S3 location for model artifacts
           - Configure metrics to capture during training
           - Set up debugging and profiling options
        """)
    
    st.markdown("""
    **Example SageMaker Training Configuration:**
    
    ```python
    import sagemaker
    from sagemaker.estimator import Estimator
    
    # Initialize SageMaker session
    session = sagemaker.Session()
    
    # Configure training job
    estimator = Estimator(
        image_uri="<algorithm-image-uri>",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size=30,
        max_run=3600,
        input_mode="File",
        output_path=f"s3://{bucket}/output",
        hyperparameters={
            "max_depth": 5,
            "eta": 0.2,
            "gamma": 4,
            "min_child_weight": 6,
            "subsample": 0.8,
            "num_round": 100,
        }
    )
    
    # Configure data inputs
    estimator.fit(
        {
            "train": f"s3://{bucket}/train",
            "validation": f"s3://{bucket}/validation"
        },
        wait=True
    )
    ```
    """)


def render_quiz_tab():
    """Render the Knowledge Check quiz tab content."""
    custom_header("Test Your Knowledge")
    
    st.markdown("""
    This quiz will test your understanding of the key concepts covered in Domain 1: Data Preparation for Machine Learning.
    Answer the following questions to evaluate your knowledge of data preparation for ML.
    """)
    
    # Define quiz questions
    questions = [
        {
            "question": "Which AWS storage service is commonly used for storing machine learning datasets due to its scalability, durability, and integration capabilities?",
            "options": ["Amazon S3", "Amazon EBS", "Amazon RDS", "AWS Fargate"],
            "correct": "Amazon S3",
            "explanation": "Amazon S3 is commonly used for ML datasets due to its scalability, durability, and integration with other AWS services. It provides a central data lake for ingesting, extracting, and transforming data."
        },
        {
            "question": "What technique would you use to convert categorical data like ['Red', 'Blue', 'Green'] into numerical values for ML models when there's no inherent order to the categories?",
            "options": ["Feature scaling", "Label encoding", "One-hot encoding", "Dimensionality reduction"],
            "correct": "One-hot encoding",
            "explanation": "One-hot encoding creates binary columns for each category value, which is appropriate for nominal categorical data with no inherent order, like colors. This avoids implying a numerical relationship between categories."
        },
        {
            "question": "Which technique is most suitable for handling class imbalance by creating synthetic examples of the minority class?",
            "options": ["Random undersampling", "SMOTE", "Feature scaling", "Cross-validation"],
            "correct": "SMOTE",
            "explanation": "SMOTE (Synthetic Minority Oversampling Technique) creates synthetic examples for the minority class by interpolating between existing minority samples. This helps address class imbalance without simply duplicating existing examples."
        },
        {
            "question": "What is the primary purpose of splitting a dataset into training, validation, and test sets?",
            "options": [
                "To ensure the model works with different types of data", 
                "To evaluate model performance on unseen data and prevent overfitting", 
                "To speed up the training process", 
                "To reduce the amount of data needed for training"
            ],
            "correct": "To evaluate model performance on unseen data and prevent overfitting",
            "explanation": "Splitting the dataset helps ensure the model generalizes well to new data. The training set is used to train the model, the validation set for hyperparameter tuning and avoiding overfitting, and the test set provides an unbiased evaluation of the final model's performance."
        },
        {
            "question": "Which AWS service helps detect bias in ML models, provides explanations for model predictions, and monitors bias drift over time?",
            "options": ["Amazon SageMaker Data Wrangler", "Amazon SageMaker Feature Store", "Amazon SageMaker Clarify", "Amazon Comprehend"],
            "correct": "Amazon SageMaker Clarify",
            "explanation": "Amazon SageMaker Clarify helps identify biases in data and models, explains model predictions (both overall behavior and individual predictions), and detects drift in bias and model behavior over time. It also generates automated reports on bias and explanations."
        },
        {
            "question": "Which feature scaling method is most appropriate when your dataset contains significant outliers?",
            "options": ["Min-Max Scaling", "Standard Scaling", "Robust Scaling", "Unit Vector Scaling"],
            "correct": "Robust Scaling",
            "explanation": "Robust Scaling uses the median and interquartile range instead of mean and standard deviation, making it resistant to outliers. It's the most appropriate scaling method when your dataset contains significant outliers."
        },
        {
            "question": "What is the 'curse of dimensionality' in machine learning?",
            "options": [
                "The challenge of visualizing more than 3 dimensions", 
                "The exponential increase in complexity and data needed as dimensions increase", 
                "The computational cost of training deep neural networks", 
                "The difficulty in selecting appropriate hyperparameters"
            ],
            "correct": "The exponential increase in complexity and data needed as dimensions increase",
            "explanation": "The curse of dimensionality refers to problems that arise when analyzing data in high-dimensional spaces. As dimensions increase, the volume of the space increases exponentially, requiring exponentially more data points to maintain the same data density, which affects model performance."
        },
        {
            "question": "Which AWS service provides a visual, no-code interface with 300+ built-in transformations for data preparation?",
            "options": ["AWS Glue", "Amazon SageMaker Data Wrangler", "Amazon EMR", "AWS Lambda"],
            "correct": "Amazon SageMaker Data Wrangler",
            "explanation": "Amazon SageMaker Data Wrangler provides a visual, no-code interface with 300+ built-in transformations for data preparation, making it easy to clean and transform data without writing code."
        },
        {
            "question": "What is the primary benefit of data shuffling in machine learning?",
            "options": [
                "It reduces training time", 
                "It ensures the model doesn't learn order-dependent patterns", 
                "It eliminates the need for data scaling", 
                "It automatically removes duplicates"
            ],
            "correct": "It ensures the model doesn't learn order-dependent patterns",
            "explanation": "Data shuffling randomizes the order of examples in your dataset before training. This ensures the model doesn't learn patterns based on the order of data presentation, which helps with stochastic gradient descent optimization and reduces bias from sequential patterns."
        },
        {
            "question": "What problem does Amazon SageMaker Feature Store solve?",
            "options": [
                "Automated model selection", 
                "Reusing features across models and ensuring consistency between training and inference", 
                "Hyperparameter optimization", 
                "Distributed model training"
            ],
            "correct": "Reusing features across models and ensuring consistency between training and inference",
            "explanation": "Amazon SageMaker Feature Store allows teams to reuse features across multiple models, reducing duplicate feature engineering work. It also ensures consistency between training and inference by providing the same feature values to both processes, eliminating training-serving skew."
        }
    ]
    
    # Check if the quiz has been attempted
    if not st.session_state['quiz_attempted']:
        # Create a form for the quiz
        with st.form("quiz_form"):
            st.markdown("### Answer the following questions:")
            
            # Track user answers
            user_answers = []
            
            # Display 5 random questions
            np.random.seed(42)  # For reproducibility
            selected_questions = np.random.choice(questions, size=5, replace=False)
            
            # Display each question
            for i, q in enumerate(selected_questions):
                st.markdown(f"**Question {i+1}:** {q['question']}")
                answer = st.radio(f"Select your answer for question {i+1}:", q['options'], index=None,key=f"q{i}")
                user_answers.append((answer, q['correct'], q['explanation']))
            
            # Submit button
            submitted = st.form_submit_button("Submit Quiz")
            
            if submitted:
                # Calculate score
                score = sum([1 for ua, corr, _ in user_answers if ua == corr])
                st.session_state['quiz_score'] = score
                st.session_state['quiz_attempted'] = True
                st.session_state['quiz_answers'] = user_answers
                st.rerun()
    else:
        # Display results
        score = st.session_state['quiz_score']
        user_answers = st.session_state.get('quiz_answers', [])
        
        st.markdown(f"### Your Score: {score}/5")
        
        if score == 5:
            st.success("🎉 Perfect score! You've mastered the concepts of Data Preparation for ML!")
        elif score >= 3:
            st.success("👍 Good job! You have a solid understanding of the concepts.")
        else:
            st.warning("📚 You might want to review the content again to strengthen your understanding.")
        
        # Show correct answers
        st.markdown("### Review Questions and Answers:")
        
        for i, (user_answer, correct_answer, explanation) in enumerate(user_answers):
            st.markdown(f"**Question {i+1}**")
            st.markdown(f"**Your answer:** {user_answer}")
            
            if user_answer == correct_answer:
                st.markdown(f"**✅ Correct!**")
            else:
                st.markdown(f"**❌ Incorrect. The correct answer is:** {correct_answer}")
            
            st.markdown(f"**Explanation:** {explanation}")
            
            if i < len(user_answers) - 1:
                st.markdown("---")
        
        # Option to retake the quiz
        if st.button("Retake Quiz"):
            st.session_state['quiz_attempted'] = False
            st.rerun()


def render_resources_tab():
    """Render the Resources tab content."""
    custom_header("Additional Resources")
    
    st.markdown("""
    Explore these resources to deepen your understanding of Data Preparation for Machine Learning.
    These materials provide additional context and practical guidance for implementing the concepts covered in this module.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AWS Documentation")
        st.markdown("""
        - [Feature Engineering in Machine Learning](https://aws.amazon.com/what-is/feature-engineering/)
        - [Amazon SageMaker Data Wrangler](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-transform.html)
        - [Amazon SageMaker Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)
        - [Amazon SageMaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-detect-data-bias.html)
        - [AWS Glue](https://docs.aws.amazon.com/glue/latest/dg/how-it-works.html)
        - [Amazon Redshift ML](https://docs.aws.amazon.com/redshift/latest/dg/machine_learning.html)
        - [AWS Lake Formation](https://docs.aws.amazon.com/lake-formation/latest/dg/what-is-lake-formation.html)
        - [Model Access Training Data Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data-best-practices.html)
        """)
        
        st.markdown("### AWS Blog Posts")
        st.markdown("""
        - [Balance your data for machine learning with Amazon SageMaker Data Wrangler](https://aws.amazon.com/blogs/machine-learning/balance-your-data-for-machine-learning-with-amazon-sagemaker-data-wrangler/)
        - [Amazon Kinesis Data Firehose Zero Buffering](https://aws.amazon.com/about-aws/whats-new/2023/12/amazon-kinesis-data-firehose-zero-buffering/)
        - [Building ML Models with Amazon SageMaker Feature Store](https://aws.amazon.com/blogs/machine-learning/building-ml-models-with-amazon-sagemaker-feature-store/)
        - [Detecting and Handling Data Drift with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/detecting-and-handling-data-drift-with-amazon-sagemaker/)
        - [Categorical Data Encoding Techniques for ML](https://aws.amazon.com/blogs/machine-learning/categorical-data-encoding-techniques-for-machine-learning-with-amazon-sagemaker/)
        """)
    
    with col2:
        st.markdown("### Training Courses")
        st.markdown("""
        - [AWS Technical Essentials](https://aws.amazon.com/training/learn-about/technical-essentials/)
        - [Getting Started with AWS Storage](https://aws.amazon.com/training/learn-about/storage/)
        - [AWS Cloud Quest: Machine Learning](https://aws.amazon.com/training/learn-about/cloud-quest/)
        - [AWS Machine Learning Specialty Certification](https://aws.amazon.com/certification/certified-machine-learning-specialty/)
        - [Data Science on AWS](https://www.coursera.org/learn/data-science-on-aws)
        - [Practical Data Science with Amazon SageMaker](https://www.coursera.org/specializations/practical-data-science)
        """)
        
        st.markdown("### Tools and Services")
        st.markdown("""
        - [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
        - [AWS Glue](https://aws.amazon.com/glue/)
        - [Amazon S3](https://aws.amazon.com/s3/)
        - [Amazon Kinesis](https://aws.amazon.com/kinesis/)
        - [Amazon EMR](https://aws.amazon.com/emr/)
        - [AWS Lambda](https://aws.amazon.com/lambda/)
        - [Amazon Mechanical Turk](https://www.mturk.com/)
        - [Amazon DataBrew](https://aws.amazon.com/glue/features/databrew/)
        """)
    
    custom_header("Deep Dive: Key Concepts", "sub")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Data Collection")
        st.markdown("""
        **Key Resources:**
        
        - [Building Data Lakes on AWS](https://aws.amazon.com/big-data/datalakes-and-analytics/what-is-a-data-lake/)
        - [Real-time Data Processing with Amazon Kinesis](https://docs.aws.amazon.com/streams/latest/dev/introduction.html)
        - [Amazon S3 Storage Classes](https://aws.amazon.com/s3/storage-classes/)
        - [Data Lake vs. Data Warehouse Architecture](https://aws.amazon.com/blogs/big-data/build-a-lake-house-architecture-on-aws/)
        - [Guide to Data Formats for ML](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html)
        """)
    
    with col2:
        st.markdown("#### Data Transformation")
        st.markdown("""
        **Key Resources:**
        
        - [Data Cleaning Best Practices](https://aws.amazon.com/blogs/machine-learning/build-a-strong-foundation-for-ml-with-automated-data-preparation-using-amazon-sagemaker-data-wrangler/)
        - [Handling Missing Values in AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-transform.html#data-wrangler-transform-handle-missing)
        - [Categorical Encoding Techniques](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-transform.html#data-wrangler-transform-cat-encode)
        - [AWS Glue for ETL Jobs](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming.html)
        - [Automating Data Preparation with DataBrew](https://docs.aws.amazon.com/databrew/latest/dg/what-is.html)
        """)
    
    with col3:
        st.markdown("#### Feature Engineering")
        st.markdown("""
        **Key Resources:**
        
        - [Feature Selection Methods](https://aws.amazon.com/blogs/machine-learning/feature-selection-for-machine-learning/)
        - [Dimensionality Reduction Techniques](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
        - [Scaling and Normalization in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-transform.html#data-wrangler-transform-featurize)
        - [Feature Store Architecture](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-quotas.html)
        - [Managing ML Features with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-getting-started.html)
        """)
    
    custom_header("Additional ML Resources", "sub")
    
    st.markdown("""
    ### Foundational Machine Learning Resources
    
    - [Machine Learning on AWS](https://aws.amazon.com/machine-learning/)
    - [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
    - [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)
    - [Amazon SageMaker Examples GitHub Repository](https://github.com/aws/amazon-sagemaker-examples)
    - [AWS Machine Learning Foundations Course](https://www.udacity.com/course/aws-machine-learning-foundations--ud090)
    
    ### Data Preparation Specific Resources
    
    - [AWS Data Wrangler GitHub](https://github.com/awslabs/aws-data-wrangler)
    - [Data Visualization on AWS](https://aws.amazon.com/data-visualization/)
    - [AWS Data Exchange](https://aws.amazon.com/data-exchange/)
    - [Data Preprocessing for Machine Learning with AWS](https://www.youtube.com/watch?v=P1SYJ7Iy2F8)
    
    ### ML Certification Preparation
    
    - [AWS Machine Learning Engineer Path](https://aws.amazon.com/training/learn-about/machine-learning/)
    - [Sample Exam Questions](https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Sample-Questions.pdf)
    - [Exam Readiness: AWS Certified Machine Learning - Specialty](https://www.aws.training/Details/eLearning?id=42183)
    """)


def render_footer():
    """Render the page footer."""
    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("images/aws_logo.png", width=70)
    with col2:
        st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")


def initialize_state():
    """Initialize session state variables."""

    common.initialize_session_state()

    # Initialize session state variables
    if 'quiz_score' not in st.session_state:
        st.session_state['quiz_score'] = 0
    if 'quiz_attempted' not in st.session_state:
        st.session_state['quiz_attempted'] = False
    if 'name' not in st.session_state:
        st.session_state['name'] = ""
    if 'visited_ML_Lifecycle' not in st.session_state:
        st.session_state['visited_ML_Lifecycle'] = False
    if 'visited_Data_Collection' not in st.session_state:
        st.session_state['visited_Data_Collection'] = False
    if 'visited_Data_Transformation' not in st.session_state:
        st.session_state['visited_Data_Transformation'] = False
    if 'visited_Feature_Engineering' not in st.session_state:
        st.session_state['visited_Feature_Engineering'] = False
    if 'visited_Data_Integrity' not in st.session_state:
        st.session_state['visited_Data_Integrity'] = False

    # Apply custom CSS
    common.apply_styles()


def main():
    """Main function to run the Streamlit application."""
    initialize_state()
    
    # Setup sidebar
    setup_sidebar()
    
    # Create tabs
    tabs = st.tabs([
        "🏠 Home", 
        "🔄 ML Lifecycle", 
        "📊 Data Collection", 
        "⚙️ Data Transformation", 
        "🧩 Feature Engineering", 
        "🛡️ Data Integrity", 
        "❓ Knowledge Check", 
        "📚 Resources"
    ])
    
    # Render content for each tab
    with tabs[0]:
        render_home_tab()
    
    with tabs[1]:
        render_ml_lifecycle_tab()
    
    with tabs[2]:
        render_data_collection_tab()
    
    with tabs[3]:
        render_data_transformation_tab()
    
    with tabs[4]:
        render_feature_engineering_tab()
    
    with tabs[5]:
        render_data_integrity_tab()
    
    with tabs[6]:
        render_quiz_tab()
    
    with tabs[7]:
        render_resources_tab()
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()