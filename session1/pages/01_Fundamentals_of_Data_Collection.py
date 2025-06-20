
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import utils.common as common
import utils.authenticate as authenticate


# Set page configuration
st.set_page_config(
    page_title="AWS Data Structures & Storage", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS Color Scheme
AWS_COLORS = {
    "orange": "#FF9900",
    "light_orange": "#FFAC31",
    "dark_blue": "#232F3E",
    "light_blue": "#1A73E8",
    "teal": "#00A1C9",
    "red": "#D13212",
    "green": "#7AA116",
    "purple": "#8C4FFF"
}

def apply_custom_styles():
    """Apply custom CSS styling for AWS look and feel"""
    st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #232F3E;
        color: white;
    }
    h1, h2, h3 {
        color: #232F3E;
    }
    .stButton>button {
        background-color: #FF9900;
        color: white;
    }
    .stButton>button:hover {
        background-color: #FFAC31;
    }
    .highlight {
        background-color: #FFAC31;
        padding: 10px;
        border-radius: 5px;
    }
    .card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
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
    </style>
    """, unsafe_allow_html=True)

def initialize_quiz_state():
    """Initialize the quiz state variables if not already set"""
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}

def draw_box(ax, x, y, width, height, color, label, fontsize=10):
    """Helper function to draw a box in matplotlib diagrams"""
    rect = plt.Rectangle((x, y), width, height, facecolor=color, alpha=0.6, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize)

def draw_arrow(ax, start_x, start_y, end_x, end_y):
    """Helper function to draw an arrow in matplotlib diagrams"""
    ax.arrow(start_x, start_y, end_x-start_x, end_y-start_y, 
            head_width=0.02, head_length=0.02, fc=AWS_COLORS['dark_blue'], 
            ec=AWS_COLORS['dark_blue'])

def show_data_structures_tab():
    """Content for the Data Structures tab"""
    st.header("Data Structures and Types")
    st.markdown("""
    Data can be categorized into three main types based on its structure. Understanding these types is crucial for 
    selecting the appropriate storage and processing methods in AWS.
    """)
    
    # Create three columns for the data types
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="card">
        <h3 style="color:{AWS_COLORS['orange']}">Structured Data</h3>
        <p><strong>Format:</strong> Tabular</p>
        <p><strong>Examples:</strong> SQL databases, spreadsheets</p>
        <p><strong>Features:</strong> Organized schema, easily searchable</p>
        <p><strong>AWS Services:</strong> Amazon RDS, Amazon Redshift</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example of structured data
        st.subheader("Example")
        df_structured = pd.DataFrame({
            'customer_id': [101, 102, 103, 104],
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [34, 45, 29, 52],
            'purchase_amount': [125.50, 89.99, 210.75, 55.25]
        })
        st.dataframe(df_structured)
    
    with col2:
        st.markdown(f"""
        <div class="card">
        <h3 style="color:{AWS_COLORS['teal']}">Semi-Structured Data</h3>
        <p><strong>Format:</strong> Elements within files</p>
        <p><strong>Examples:</strong> XML, JSON, NoSQL databases</p>
        <p><strong>Features:</strong> Some organization without rigid schema</p>
        <p><strong>AWS Services:</strong> DynamoDB, DocumentDB</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example of semi-structured data
        st.subheader("Example")
        semi_structured = {
            "customer": {
                "id": 101,
                "name": "Alice",
                "contacts": {
                    "email": "alice@example.com",
                    "phone": "555-123-4567"
                },
                "purchases": [
                    {"item": "Laptop", "price": 899.99},
                    {"item": "Mouse", "price": 25.50}
                ]
            }
        }
        st.json(semi_structured)
    
    with col3:
        st.markdown(f"""
        <div class="card">
        <h3 style="color:{AWS_COLORS['purple']}">Unstructured Data</h3>
        <p><strong>Format:</strong> Files without predefined model</p>
        <p><strong>Examples:</strong> Text files, images, videos</p>
        <p><strong>Features:</strong> No pre-defined organization</p>
        <p><strong>AWS Services:</strong> Amazon S3, Amazon OpenSearch</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example of unstructured data
        st.subheader("Example")
        st.markdown("""
        ```
        From: alice@example.com
        To: support@company.com
        Subject: Product feedback
        
        I recently purchased your product and wanted to share my thoughts.
        Overall, I'm quite satisfied with the functionality, but I noticed
        that the battery life is shorter than advertised. Could you provide
        some tips to optimize battery usage?
        
        Best regards,
        Alice
        ```
        """)
    
    # Data processing visualization
    st.header("Data Processing Flow")
    st.markdown("How different data types flow through AWS services:")
    
    # Create a flow chart image using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f9f9f9')
    
    # Create nodes
    nodes = {
        'structured': (0.2, 0.7),
        'semi': (0.2, 0.5),
        'unstructured': (0.2, 0.3),
        'rds': (0.5, 0.7),
        'dynamo': (0.5, 0.5),
        's3': (0.5, 0.3),
        'redshift': (0.8, 0.6),
        'ml': (0.8, 0.4)
    }
    
    # Draw nodes
    ax.plot(nodes['structured'][0], nodes['structured'][1], 'o', ms=15, color=AWS_COLORS['orange'])
    ax.plot(nodes['semi'][0], nodes['semi'][1], 'o', ms=15, color=AWS_COLORS['teal'])
    ax.plot(nodes['unstructured'][0], nodes['unstructured'][1], 'o', ms=15, color=AWS_COLORS['purple'])
    
    ax.plot(nodes['rds'][0], nodes['rds'][1], 'o', ms=15, color=AWS_COLORS['light_blue'])
    ax.plot(nodes['dynamo'][0], nodes['dynamo'][1], 'o', ms=15, color=AWS_COLORS['light_blue'])
    ax.plot(nodes['s3'][0], nodes['s3'][1], 'o', ms=15, color=AWS_COLORS['light_blue'])
    
    ax.plot(nodes['redshift'][0], nodes['redshift'][1], 'o', ms=15, color=AWS_COLORS['green'])
    ax.plot(nodes['ml'][0], nodes['ml'][1], 'o', ms=15, color=AWS_COLORS['green'])
    
    # Draw arrows
    draw_arrow(ax, nodes['structured'][0], nodes['structured'][1], 
             nodes['rds'][0]-0.02, nodes['rds'][1])
    
    draw_arrow(ax, nodes['semi'][0], nodes['semi'][1], 
             nodes['dynamo'][0]-0.02, nodes['dynamo'][1])
    
    draw_arrow(ax, nodes['unstructured'][0], nodes['unstructured'][1], 
             nodes['s3'][0]-0.02, nodes['s3'][1])
    
    draw_arrow(ax, nodes['rds'][0], nodes['rds'][1], 
             nodes['redshift'][0]-0.02, nodes['redshift'][1])
    
    draw_arrow(ax, nodes['dynamo'][0], nodes['dynamo'][1], 
             nodes['ml'][0]-0.02, nodes['ml'][1])
    
    draw_arrow(ax, nodes['s3'][0], nodes['s3'][1], 
             nodes['ml'][0]-0.02, nodes['ml'][1])
    
    # Add labels
    ax.text(nodes['structured'][0], nodes['structured'][1]+0.03, 'Structured Data', 
            ha='center', va='center', fontsize=10)
    ax.text(nodes['semi'][0], nodes['semi'][1]+0.03, 'Semi-structured Data', 
            ha='center', va='center', fontsize=10)
    ax.text(nodes['unstructured'][0], nodes['unstructured'][1]+0.03, 'Unstructured Data', 
            ha='center', va='center', fontsize=10)
    
    ax.text(nodes['rds'][0], nodes['rds'][1]+0.03, 'Amazon RDS', 
            ha='center', va='center', fontsize=10)
    ax.text(nodes['dynamo'][0], nodes['dynamo'][1]+0.03, 'Amazon DynamoDB', 
            ha='center', va='center', fontsize=10)
    ax.text(nodes['s3'][0], nodes['s3'][1]+0.03, 'Amazon S3', 
            ha='center', va='center', fontsize=10)
    
    ax.text(nodes['redshift'][0], nodes['redshift'][1]+0.03, 'Amazon Redshift', 
            ha='center', va='center', fontsize=10)
    ax.text(nodes['ml'][0], nodes['ml'][1]+0.03, 'ML Processing', 
            ha='center', va='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    st.pyplot(fig)

def show_storage_formats_tab():
    """Content for the Storage Formats tab"""
    st.header("Storage Formats for Machine Learning")
    st.markdown("""
    Choosing the right storage format for your data is essential for efficient processing and analysis in machine learning workflows.
    Each format has advantages and disadvantages depending on your use case.
    """)
    
    # Create comparison table for storage formats
    st.subheader("Comparison of Common Storage Formats")
    
    storage_data = {
        'Feature': ['Data Storage', 'Write Performance', 'Read Performance', 
                    'Compression', 'Schema Evolution', 'Best For'],
        'CSV': ['Row', 'Fast', 'Slow', 'No', 'No', 'Simple data interchange, logs'],
        'Parquet': ['Column', 'Slow', 'Fast', 'Yes', 'Yes', 'Big data processing, analytical queries'],
        'Avro': ['Row', 'Medium', 'Medium', 'No', 'Yes', 'Schema evolution'],
        'ORC': ['Column', 'Slow', 'Fast', 'Yes', 'Yes', 'Hive/SQL optimizations'],
        'JSON': ['Object-notation', 'Medium', 'Medium', 'No', 'No', 'Flexible data exchange, Web applications'],
        'JSONL': ['Object-notation', 'Fast', 'Medium', 'No', 'No', 'Logs, stream processing']
    }
    
    df_storage = pd.DataFrame(storage_data)
    st.dataframe(df_storage, use_container_width=True)
    
    # Performance visualization for read/write
    st.subheader("Read vs. Write Performance")
    
    performance_data = {
        'Format': ['CSV', 'Parquet', 'Avro', 'ORC', 'JSON', 'JSONL'],
        'Read Speed': [30, 90, 60, 85, 55, 50],
        'Write Speed': [85, 40, 60, 45, 55, 80]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    # Convert to long format for Altair
    df_performance_long = pd.melt(
        df_performance, 
        id_vars=['Format'], 
        value_vars=['Read Speed', 'Write Speed'],
        var_name='Operation', 
        value_name='Performance Score'
    )
    
    chart = alt.Chart(df_performance_long).mark_bar().encode(
        x='Format',
        y='Performance Score',
        color=alt.Color('Operation', scale=alt.Scale(
            domain=['Read Speed', 'Write Speed'],
            range=[AWS_COLORS['orange'], AWS_COLORS['teal']]
        )),
        column='Operation'
    ).properties(
        width=300
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Storage format examples
    st.subheader("Format Examples")
    
    format_tabs = st.tabs(["CSV", "Parquet", "JSON", "JSONL"])
    
    with format_tabs[0]:
        st.markdown("#### CSV (Comma-Separated Values)")
        st.code('''
customerID,name,age,email,subscription_active
12345678,"Rosalez, Alejandro",32,alejandro_rosalez@example.com,false
87654321,"Candella, Pat",22,pat_candella@example.com,true
''')
        st.markdown("""
        **Advantages:**
        - Simple and human-readable
        - Widely supported by tools and applications
        - Efficient for writing data
        
        **Disadvantages:**
        - No compression
        - Slower reads for large datasets
        - No schema enforcement
        """)
    
    with format_tabs[1]:
        st.markdown("#### Parquet (Columnar Storage)")
        st.image("https://parquet.apache.org/images/parquet.png", width=200)
        st.markdown("""
        Parquet is a binary file format that stores data in a columnar fashion.
        
        **Advantages:**
        - Excellent compression
        - Fast read performance
        - Schema evolution support
        - Great for analytical queries
        
        **Disadvantages:**
        - Slower write performance
        - Not human-readable
        - More complex than CSV
        """)
    
    with format_tabs[2]:
        st.markdown("#### JSON (JavaScript Object Notation)")
        st.code('''
{
  "customers": [
    {
      "customerID": "12345678",
      "name": "Rosalez, Alejandro",
      "age": 32,
      "email": "alejandro_rosalez@example.com",
      "subscription": {
        "active": false,
        "last_support": "2022-01-12"
      }
    },
    {
      "customerID": "87654321",
      "name": "Candella, Pat",
      "age": 22,
      "email": "pat_candella@example.com",
      "subscription": {
        "active": true,
        "last_support": "2024-03-26"
      }
    }
  ]
}
''')
        st.markdown("""
        **Advantages:**
        - Human-readable
        - Flexible schema
        - Good for nested data
        
        **Disadvantages:**
        - Less efficient storage
        - Slower parsing for large datasets
        - No built-in compression
        """)
    
    with format_tabs[3]:
        st.markdown("#### JSONL (JSON Lines)")
        st.code('''
{"customerID": "12345678", "name": "Rosalez, Alejandro", "age": 32, "email": "alejandro_rosalez@example.com", "last_support": "2022-01-12", "subscription_active": false}
{"customerID": "87654321", "name": "Candella, Pat", "age": 22, "email": "pat_candella@example.com", "last_support": "2024-03-26", "subscription_active": true}
''')
        st.markdown("""
        **Advantages:**
        - Easy to process line by line
        - Good for streaming data
        - Each line is a complete JSON object
        
        **Disadvantages:**
        - Less efficient than columnar formats
        - No built-in compression
        - Not ideal for deeply nested data
        """)
    
    # AWS Service recommendation
    st.header("AWS Service Recommendations by Format")
    
    service_data = {
        'Format': ['CSV', 'Parquet', 'Avro', 'JSON/JSONL'],
        'Storage': ['S3, EFS', 'S3, EMR', 'S3, EMR', 'S3, DynamoDB'],
        'Processing': ['Glue, Lambda', 'Athena, EMR, Redshift Spectrum', 'EMR, Glue', 'Lambda, API Gateway, AppSync'],
        'ML Services': ['SageMaker', 'SageMaker, EMR', 'SageMaker, EMR', 'SageMaker']
    }
    
    df_services = pd.DataFrame(service_data)
    st.table(df_services)

def show_data_warehouses_vs_lakes_tab():
    """Content for the Data Warehouses vs Data Lakes tab"""
    st.header("Data Warehouses vs Data Lakes vs Databases")
    
    st.markdown("""
    Understanding the differences between data warehouses, data lakes, and traditional databases 
    is crucial for designing effective data architecture on AWS.
    """)
    
    # Visual comparison
    comparison_tabs = st.tabs(["Overview", "Data Lake", "Data Warehouse", "Database"])
    
    with comparison_tabs[0]:
        st.subheader("Comparison Overview")
        
        comparison_data = {
            'Feature': ['Data Structure', 'Data Processing', 'Schema', 'Users', 
                       'Use Cases', 'Cost', 'Storage', 'Primary AWS Service'],
            'Database': ['Structured', 'Real-time transactions', 'Schema-on-write', 'Application developers',
                        'Transactional applications, CRUD operations', 'Higher per TB', 'Limited by server',
                        'Amazon RDS, Aurora'],
            'Data Warehouse': ['Structured', 'Batch processing & analytics', 'Schema-on-write', 'Data analysts, business users',
                              'BI, reporting, structured analytics', 'Medium', 'Terabytes to petabytes',
                              'Amazon Redshift'],
            'Data Lake': ['Structured, Semi-structured, Unstructured', 'Batch & real-time', 'Schema-on-read',
                         'Data scientists, ML engineers', 'Advanced analytics, ML, data exploration', 'Low per TB', 
                         'Petabytes+', 'Amazon S3 + AWS Glue']
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Create a visual representation
        st.subheader("Visual Comparison")
        
        # Data flow chart using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f9f9f9')
        
        # Setup positions
        pos = {
            'sources': (0.1, 0.5),
            'database': (0.4, 0.8),
            'warehouse': (0.4, 0.5),
            'lake': (0.4, 0.2),
            'app': (0.7, 0.8),
            'bi': (0.7, 0.5),
            'ml': (0.7, 0.2)
        }
        
        # Draw nodes
        ax.plot(pos['sources'][0], pos['sources'][1], 'o', ms=15, color=AWS_COLORS['dark_blue'])
        
        ax.plot(pos['database'][0], pos['database'][1], 'o', ms=15, color=AWS_COLORS['orange'])
        ax.plot(pos['warehouse'][0], pos['warehouse'][1], 'o', ms=15, color=AWS_COLORS['teal'])
        ax.plot(pos['lake'][0], pos['lake'][1], 'o', ms=15, color=AWS_COLORS['purple'])
        
        ax.plot(pos['app'][0], pos['app'][1], 'o', ms=15, color=AWS_COLORS['green'])
        ax.plot(pos['bi'][0], pos['bi'][1], 'o', ms=15, color=AWS_COLORS['green'])
        ax.plot(pos['ml'][0], pos['ml'][1], 'o', ms=15, color=AWS_COLORS['green'])
        
        # Draw arrows
        draw_arrow(ax, pos['sources'][0], pos['sources'][1], 
                pos['database'][0]-0.02, pos['database'][1])
        
        draw_arrow(ax, pos['sources'][0], pos['sources'][1], 
                pos['warehouse'][0]-0.02, pos['warehouse'][1])
        
        draw_arrow(ax, pos['sources'][0], pos['sources'][1], 
                pos['lake'][0]-0.02, pos['lake'][1])
        
        draw_arrow(ax, pos['database'][0], pos['database'][1], 
                pos['app'][0]-0.02, pos['app'][1])
        
        draw_arrow(ax, pos['warehouse'][0], pos['warehouse'][1], 
                pos['bi'][0]-0.02, pos['bi'][1])
        
        draw_arrow(ax, pos['lake'][0], pos['lake'][1], 
                pos['ml'][0]-0.02, pos['ml'][1])
        
        # Labels
        ax.text(pos['sources'][0], pos['sources'][1]+0.05, 'Data Sources', 
                ha='center', va='center', fontsize=11)
        
        ax.text(pos['database'][0], pos['database'][1]+0.05, 'Database', 
                ha='center', va='center', fontsize=11)
        ax.text(pos['warehouse'][0], pos['warehouse'][1]+0.05, 'Data Warehouse', 
                ha='center', va='center', fontsize=11)
        ax.text(pos['lake'][0], pos['lake'][1]+0.05, 'Data Lake', 
                ha='center', va='center', fontsize=11)
        
        ax.text(pos['app'][0], pos['app'][1]+0.05, 'Applications', 
                ha='center', va='center', fontsize=11)
        ax.text(pos['bi'][0], pos['bi'][1]+0.05, 'BI & Reporting', 
                ha='center', va='center', fontsize=11)
        ax.text(pos['ml'][0], pos['ml'][1]+0.05, 'ML & Advanced Analytics', 
                ha='center', va='center', fontsize=11)
        
        ax.text(0.25, 0.85, 'Structured', ha='center', va='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        ax.text(0.25, 0.55, 'Structured', ha='center', va='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        ax.text(0.25, 0.25, 'All Data Types', ha='center', va='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        st.pyplot(fig)
    
    with comparison_tabs[1]:
        st.subheader("Data Lake")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['purple']}">What is a Data Lake?</h4>
            <p>A data lake is a centralized repository that allows you to store all your structured, semi-structured, 
            and unstructured data at any scale. You can store your data as-is, without having to first structure it, 
            and run different types of analytics.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['purple']}">Key Characteristics</h4>
            <ul>
            <li><strong>Schema on read:</strong> Data structure is defined when data is read</li>
            <li><strong>Stores all data types:</strong> Structured, semi-structured, and unstructured</li>
            <li><strong>Highly scalable:</strong> Can store petabytes of data</li>
            <li><strong>Low-cost storage:</strong> Uses cost-effective storage like Amazon S3</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['purple']}">AWS Implementation</h4>
            <p><strong>Core Services:</strong></p>
            <ul>
            <li>Amazon S3 (storage)</li>
            <li>AWS Glue (data catalog and ETL)</li>
            <li>Amazon Athena (query service)</li>
            <li>Amazon EMR (processing)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['purple']}">Ideal Use Cases</h4>
            <ul>
            <li>Machine learning training data</li>
            <li>Advanced analytics</li>
            <li>Data discovery and exploration</li>
            <li>When you need to store raw, unprocessed data</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Lake architecture diagram
        st.subheader("Data Lake Architecture on AWS")
        
        # Create a simple architecture diagram using matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#f9f9f9')
        
        # Hide axes
        ax.axis('off')
        
        # Draw boxes
        # Data sources
        draw_box(ax, 0.05, 0.7, 0.15, 0.2, AWS_COLORS['orange'], 'Structured\nData')
        draw_box(ax, 0.05, 0.4, 0.15, 0.2, AWS_COLORS['teal'], 'Semi-structured\nData')
        draw_box(ax, 0.05, 0.1, 0.15, 0.2, AWS_COLORS['purple'], 'Unstructured\nData')
        
        # Ingestion
        draw_box(ax, 0.3, 0.4, 0.15, 0.2, AWS_COLORS['light_blue'], 'Data\nIngestion\n(Kinesis, Transfer)')
        
        # Storage
        draw_box(ax, 0.55, 0.4, 0.15, 0.2, AWS_COLORS['dark_blue'], 'Amazon S3\nData Lake')
        
        # Processing
        draw_box(ax, 0.8, 0.7, 0.15, 0.2, AWS_COLORS['light_blue'], 'AWS Glue\n(Catalog & ETL)')
        draw_box(ax, 0.8, 0.4, 0.15, 0.2, AWS_COLORS['light_blue'], 'EMR\n(Processing)')
        draw_box(ax, 0.8, 0.1, 0.15, 0.2, AWS_COLORS['light_blue'], 'Athena\n(Query)')
        
        # Connect boxes with arrows
        # Data sources to ingestion
        draw_arrow(ax, 0.2, 0.8, 0.3, 0.5)
        draw_arrow(ax, 0.2, 0.5, 0.3, 0.5)
        draw_arrow(ax, 0.2, 0.2, 0.3, 0.5)
        
        # Ingestion to storage
        draw_arrow(ax, 0.45, 0.5, 0.55, 0.5)
        
        # Storage to processing
        draw_arrow(ax, 0.7, 0.5, 0.8, 0.8)
        draw_arrow(ax, 0.7, 0.5, 0.8, 0.5)
        draw_arrow(ax, 0.7, 0.5, 0.8, 0.2)
        
        st.pyplot(fig)
    
    with comparison_tabs[2]:
        st.subheader("Data Warehouse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['teal']}">What is a Data Warehouse?</h4>
            <p>A data warehouse is a system that aggregates structured data from different sources into a single, 
            central, consistent data store to support business intelligence activities, particularly analytics.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['teal']}">Key Characteristics</h4>
            <ul>
            <li><strong>Schema on write:</strong> Data is structured before loading</li>
            <li><strong>Optimized for analytics:</strong> High query performance</li>
            <li><strong>Structured data:</strong> Primarily deals with structured data</li>
            <li><strong>Historical data:</strong> Designed for historical analysis</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['teal']}">AWS Implementation</h4>
            <p><strong>Core Services:</strong></p>
            <ul>
            <li>Amazon Redshift (primary warehouse)</li>
            <li>AWS Glue (ETL processes)</li>
            <li>Amazon QuickSight (visualization)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['teal']}">Ideal Use Cases</h4>
            <ul>
            <li>Business intelligence reporting</li>
            <li>Historical data analysis</li>
            <li>Structured analytics</li>
            <li>When you need fast query performance</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Warehouse architecture diagram
        st.subheader("Data Warehouse Architecture on AWS")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#f9f9f9')
        
        # Hide axes
        ax.axis('off')
        
        # Draw boxes for data warehouse architecture
        # Data sources
        draw_box(ax, 0.05, 0.6, 0.15, 0.15, AWS_COLORS['orange'], 'ERP System')
        draw_box(ax, 0.05, 0.4, 0.15, 0.15, AWS_COLORS['orange'], 'CRM System')
        draw_box(ax, 0.05, 0.2, 0.15, 0.15, AWS_COLORS['orange'], 'Other Systems')
        
        # ETL
        draw_box(ax, 0.3, 0.4, 0.15, 0.2, AWS_COLORS['light_blue'], 'AWS Glue\n(ETL)')
        
        # Data Warehouse
        draw_box(ax, 0.55, 0.3, 0.15, 0.4, AWS_COLORS['teal'], 'Amazon\nRedshift\nData\nWarehouse')
        
        # BI and Reporting
        draw_box(ax, 0.8, 0.6, 0.15, 0.15, AWS_COLORS['green'], 'QuickSight\n(BI)')
        draw_box(ax, 0.8, 0.4, 0.15, 0.15, AWS_COLORS['green'], 'Business\nReporting')
        draw_box(ax, 0.8, 0.2, 0.15, 0.15, AWS_COLORS['green'], 'Executive\nDashboards')
        
        # Connect boxes with arrows
        # Sources to ETL
        draw_arrow(ax, 0.2, 0.675, 0.3, 0.5)
        draw_arrow(ax, 0.2, 0.475, 0.3, 0.5)
        draw_arrow(ax, 0.2, 0.275, 0.3, 0.5)
        
        # ETL to warehouse
        draw_arrow(ax, 0.45, 0.5, 0.55, 0.5)
        
        # Warehouse to BI tools
        draw_arrow(ax, 0.7, 0.6, 0.8, 0.675)
        draw_arrow(ax, 0.7, 0.5, 0.8, 0.475)
        draw_arrow(ax, 0.7, 0.4, 0.8, 0.275)
        
        st.pyplot(fig)
    
    with comparison_tabs[3]:
        st.subheader("Traditional Database")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['orange']}">What is a Database?</h4>
            <p>A database is an organized collection of structured data, typically stored and accessed electronically from a 
            computer system. Databases are designed to handle transactional workloads with CRUD operations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['orange']}">Key Characteristics</h4>
            <ul>
            <li><strong>ACID compliance:</strong> Ensures reliable transaction processing</li>
            <li><strong>Normalized structure:</strong> Optimized for updates and transactions</li>
            <li><strong>Real-time processing:</strong> Designed for immediate data access</li>
            <li><strong>Row-oriented:</strong> Optimized for row-level operations</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['orange']}">AWS Implementation</h4>
            <p><strong>Core Services:</strong></p>
            <ul>
            <li>Amazon RDS (relational databases)</li>
            <li>Amazon Aurora (MySQL and PostgreSQL compatible)</li>
            <li>Amazon DynamoDB (NoSQL)</li>
            <li>Amazon DocumentDB (document database)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
            <h4 style="color:{AWS_COLORS['orange']}">Ideal Use Cases</h4>
            <ul>
            <li>Transaction processing</li>
            <li>Application data storage</li>
            <li>User profile management</li>
            <li>When you need real-time data access</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Database architecture diagram
        st.subheader("Database Architecture on AWS")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#f9f9f9')
        
        # Hide axes
        ax.axis('off')
        
        # Draw boxes for database architecture
        # Applications
        draw_box(ax, 0.05, 0.6, 0.15, 0.15, AWS_COLORS['green'], 'Web App')
        draw_box(ax, 0.05, 0.4, 0.15, 0.15, AWS_COLORS['green'], 'Mobile App')
        draw_box(ax, 0.05, 0.2, 0.15, 0.15, AWS_COLORS['green'], 'Internal\nSystems')
        
        # Load Balancer
        draw_box(ax, 0.3, 0.4, 0.15, 0.2, AWS_COLORS['light_blue'], 'Application\nLoad Balancer')
        
        # Database instances
        draw_box(ax, 0.55, 0.5, 0.15, 0.2, AWS_COLORS['orange'], 'Primary\nDB Instance')
        draw_box(ax, 0.55, 0.2, 0.15, 0.2, AWS_COLORS['light_orange'], 'Replica\nDB Instance')
        
        # Services using DB
        draw_box(ax, 0.8, 0.6, 0.15, 0.15, AWS_COLORS['purple'], 'User\nAuthentication')
        draw_box(ax, 0.8, 0.4, 0.15, 0.15, AWS_COLORS['purple'], 'Order\nProcessing')
        draw_box(ax, 0.8, 0.2, 0.15, 0.15, AWS_COLORS['purple'], 'Inventory\nManagement')
        
        # Connect boxes with arrows
        # Apps to load balancer
        draw_arrow(ax, 0.2, 0.675, 0.3, 0.5)
        draw_arrow(ax, 0.2, 0.475, 0.3, 0.5)
        draw_arrow(ax, 0.2, 0.275, 0.3, 0.5)
        
        # Load balancer to primary DB
        draw_arrow(ax, 0.45, 0.5, 0.55, 0.6)
        
        # Primary to replica
        draw_arrow(ax, 0.625, 0.5, 0.625, 0.4)
        
        # Databases to services
        draw_arrow(ax, 0.7, 0.6, 0.8, 0.675)
        draw_arrow(ax, 0.7, 0.6, 0.8, 0.475)
        draw_arrow(ax, 0.7, 0.3, 0.8, 0.275)
        
        st.pyplot(fig)
        
    # AWS Services mapping
    st.header("AWS Services by Storage Type")
    
    service_col1, service_col2, service_col3 = st.columns(3)
    
    with service_col1:
        st.markdown(f"""
        <div class="card" style="border-left: 5px solid {AWS_COLORS['orange']};">
        <h4 style="color:{AWS_COLORS['orange']}">Database Services</h4>
        <ul>
        <li><strong>Amazon RDS:</strong> Managed relational databases</li>
        <li><strong>Amazon Aurora:</strong> High-performance MySQL/PostgreSQL</li>
        <li><strong>Amazon DynamoDB:</strong> NoSQL database service</li>
        <li><strong>Amazon DocumentDB:</strong> MongoDB-compatible document database</li>
        <li><strong>Amazon Neptune:</strong> Graph database service</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with service_col2:
        st.markdown(f"""
        <div class="card" style="border-left: 5px solid {AWS_COLORS['teal']};">
        <h4 style="color:{AWS_COLORS['teal']}">Data Warehouse Services</h4>
        <ul>
        <li><strong>Amazon Redshift:</strong> Primary data warehouse</li>
        <li><strong>Redshift Spectrum:</strong> Query data in S3</li>
        <li><strong>AWS Glue:</strong> ETL service</li>
        <li><strong>Amazon QuickSight:</strong> BI visualization</li>
        <li><strong>Amazon Data Pipeline:</strong> Workflow orchestration</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with service_col3:
        st.markdown(f"""
        <div class="card" style="border-left: 5px solid {AWS_COLORS['purple']};">
        <h4 style="color:{AWS_COLORS['purple']}">Data Lake Services</h4>
        <ul>
        <li><strong>Amazon S3:</strong> Object storage</li>
        <li><strong>AWS Lake Formation:</strong> Data lake builder</li>
        <li><strong>Amazon Athena:</strong> Interactive query service</li>
        <li><strong>Amazon EMR:</strong> Big data processing</li>
        <li><strong>Amazon Kinesis:</strong> Real-time data streaming</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_knowledge_check_tab():
    """Content for the Knowledge Check tab"""
    st.header("Test Your Knowledge")
    st.markdown("Let's see how well you understand AWS data storage concepts with this quick quiz!")
    
    # Initialize quiz state
    initialize_quiz_state()
    
    # Quiz questions
    questions = [
        {
            "question": "Which data structure is best for storing unpredictable data formats?",
            "options": ["Structured data in RDS", "Semi-structured data in DynamoDB", "Unstructured data in S3", "All data types in Redshift"],
            "correct": "Unstructured data in S3",
            "explanation": "Unstructured data, like files, videos, and images, is best stored in object storage like Amazon S3 which doesn't enforce a schema."
        },
        {
            "question": "Which storage format is best for analytical queries that need fast read access to specific columns?",
            "options": ["CSV", "JSON", "Parquet", "JSONL"],
            "correct": "Parquet",
            "explanation": "Parquet is a columnar format that allows fast access to specific columns, making it ideal for analytical queries."
        },
        {
            "question": "What is the main difference between a data warehouse and a data lake?",
            "options": [
                "Data warehouses are more expensive", 
                "Data lakes use schema-on-read, data warehouses use schema-on-write",
                "Data lakes can only store unstructured data", 
                "Data warehouses are cloud-based, data lakes are on-premises"
            ],
            "correct": "Data lakes use schema-on-read, data warehouses use schema-on-write",
            "explanation": "Data lakes store raw data and apply schema when reading (schema-on-read), while data warehouses structure data before loading (schema-on-write)."
        },
        {
            "question": "Which AWS service would you use to analyze data directly in S3 without loading it into a database?",
            "options": ["Amazon RDS", "Amazon Redshift", "Amazon Athena", "Amazon DynamoDB"],
            "correct": "Amazon Athena",
            "explanation": "Amazon Athena is a serverless query service that lets you analyze data directly in Amazon S3 without having to load it elsewhere."
        },
        {
            "question": "For real-time OLTP (Online Transaction Processing) applications, which storage solution is most appropriate?",
            "options": ["Data Lake (S3)", "Data Warehouse (Redshift)", "Database (RDS/Aurora)", "Archive Storage (Glacier)"],
            "correct": "Database (RDS/Aurora)",
            "explanation": "Databases like Amazon RDS and Aurora are optimized for real-time transaction processing with low latency read/write operations."
        }
    ]
    
    # Function to handle quiz submission
    def submit_quiz():
        score = 0
        for q_idx, question in enumerate(questions):
            if st.session_state.quiz_answers.get(f"q{q_idx}") == question["correct"]:
                score += 1
        st.session_state.quiz_score = score
        st.session_state.quiz_submitted = True
    
    # Function to reset quiz
    def reset_quiz():
        st.session_state.quiz_score = 0
        st.session_state.quiz_submitted = False
        st.session_state.quiz_answers = {}
    
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
            submit_quiz()
    else:
        st.header(f"Your Score: {st.session_state.quiz_score}/{len(questions)}")
        
        # Score interpretation
        if st.session_state.quiz_score == len(questions):
            st.balloons()
            st.success("🏆 Perfect score! You're an AWS data storage expert!")
        elif st.session_state.quiz_score >= len(questions) * 0.8:
            st.success("🎓 Great job! You have a strong understanding of AWS data storage concepts.")
        elif st.session_state.quiz_score >= len(questions) * 0.6:
            st.warning("📚 Good effort! Review the explanations to strengthen your knowledge.")
        else:
            st.error("🔄 You might want to revisit the earlier sections to reinforce your understanding.")
        
        if st.button("Take Quiz Again"):
            reset_quiz()

def show_footer():
    """Display the footer content"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit application"""
    
    # Apply custom CSS styling
    apply_custom_styles()
    
    # Initialize session state
    common.initialize_session_state()
    
    # Sidebar for session management
    with st.sidebar:
        common.render_sidebar()
        
        with st.expander("About This App", expanded=False):
            st.info("""
            This interactive tutorial helps you understand:
            - Data structure types
            - Storage formats
            - Differences between data warehouses, data lakes, and databases
            
            Designed for AWS Machine Learning Engineers
            """)
    
    # Main page content
    st.title("AWS Data Structures and Storage Tutorial")
    st.markdown("#### An interactive guide to understanding data storage in AWS")
    
    # Create tabs
    tabs = st.tabs([
        "📊 Data Structures", 
        "💾 Storage Formats", 
        "🏢 Data Warehouses vs Data Lakes", 
        "📋 Knowledge Check"
    ])
    
    # Tab 1: Data Structures
    with tabs[0]:
        show_data_structures_tab()
    
    # Tab 2: Storage Formats
    with tabs[1]:
        show_storage_formats_tab()
    
    # Tab 3: Data Warehouses vs Data Lakes
    with tabs[2]:
        show_data_warehouses_vs_lakes_tab()
    
    # Tab 4: Knowledge Check
    with tabs[3]:
        show_knowledge_check_tab()
    
    # Footer
    show_footer()

# Main execution flow
if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()
