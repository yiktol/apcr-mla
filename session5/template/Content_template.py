
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
    
    # Additional state initializations can be added here
    # Example:
    # if 'example_data' not in st.session_state:
    #     st.session_state.example_data = generate_example_data()


def reset_session():
    """
    Reset the session state
    """
    # Keep only user_id and reset all other state
    user_id = st.session_state.user_id
    st.session_state.clear()
    st.session_state.user_id = user_id


# Sample data generation functions
def generate_sample_data():
    """
    Generate sample data for demonstration purposes.
    Replace with actual data generation logic as needed.
    """
    # Sample code - replace with actual implementation
    return pd.DataFrame(np.random.randn(20, 3), columns=['A', 'B', 'C'])


def generate_sample_graph():
    """
    Generate a sample graph for visualization
    """
    G = nx.DiGraph()
    
    # Add nodes and edges here
    G.add_node("Node 1", type="type1", color="#3F88C5")
    G.add_node("Node 2", type="type2", color="#FF9900")
    
    G.add_edge("Node 1", "Node 2")
    
    return G


# Visualization functions
def create_sample_chart(df):
    """
    Create a sample chart using the provided DataFrame
    """
    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        legend_title="Legend"
    )
    return fig


def draw_network_graph(G, highlight_path=None):
    """
    Draw a network graph using NetworkX and Matplotlib
    
    Args:
        G (NetworkX.DiGraph): The graph to draw
        highlight_path (list): Optional list of node names to highlight
    """
    plt.figure(figsize=(10, 6))
    
    # Define node positions
    pos = nx.spring_layout(G)
    
    # Get node colors
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    # Draw edges
    edge_color = '#aaaaaa'
    highlight_color = '#FF9900'
    
    # Draw the graph components
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()


# Main application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="AWS Styled Template",
        page_icon="📊",
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
        st.subheader("About this application")
        st.markdown("""
            This is a template application styled with AWS design patterns.
            Customize this sidebar with relevant information about your application.
        """)
        
        # Additional resources section
        st.sidebar.subheader("Additional Resources")
        st.sidebar.markdown("""
            - [Resource Link 1](#)
            - [Resource Link 2](#)
            - [Resource Link 3](#)
        """)
    
    # Main app header
    st.title("AWS Styled Application Template")
    st.markdown("A responsive, modern template for Streamlit applications with AWS styling.")
    
   
    # Tab-based navigation with emoji
    tab1, tab2, tab3 = st.tabs([
        "📊 Tab One", 
        "🔍 Tab Two",
        "🛠️ Tab Three"
    ])
    
    # TAB 1: FIRST TAB
    with tab1:
        st.header("First Tab Content")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            This is the first tab of your application. Add your content here.
            
            **Key points:**
            - Point 1
            - Point 2
            - Point 3
            - Point 4
            """)
        
        with col2:
            # Placeholder for an image
            st.image("https://via.placeholder.com/600x400", caption="Placeholder Image", use_container_width=True)
        
        st.subheader("Interactive Component Example")
        
        # Sample plotly visualization
        sample_data = generate_sample_data()
        chart = create_sample_chart(sample_data)
        st.plotly_chart(chart, use_container_width=True)
        
        # Information box example
        st.markdown("""
        <div class="info-box">
        <h3>Information Box</h3>
        <p>This is an example of a styled information box that you can use to highlight important information.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Code example box
        st.subheader("Code Example")
        st.code('''
# This is a sample code snippet
def example_function():
    """
    This is an example function
    """
    return "Hello, World!"

# Call the function
result = example_function()
print(result)
        ''')
        
    # TAB 2: SECOND TAB
    with tab2:
        st.header("Second Tab Content")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            Content for the first column of the second tab.
            
            - Bullet point 1
            - Bullet point 2
            - Bullet point 3
            """)
            
            # Example metric cards
            st.subheader("Metrics Overview")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>89%</h3>
                    <p>Metric 1</p>
                </div>
                """, unsafe_allow_html=True)
                
            with metric_col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>76</h3>
                    <p>Metric 2</p>
                </div>
                """, unsafe_allow_html=True)
                
            with metric_col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>$1.2M</h3>
                    <p>Metric 3</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            Content for the second column of the second tab.
            
            1. Numbered item 1
            2. Numbered item 2
            3. Numbered item 3
            """)
            
            # Sample cards with different status indicators
            st.subheader("Status Cards")
            
            st.markdown("""
            <div class="status-card status-1">
                <h4>Status 1 Title</h4>
                <p>Status 1 Description</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="status-card status-2">
                <h4>Status 2 Title</h4>
                <p>Status 2 Description</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="status-card status-3">
                <h4>Status 3 Title</h4>
                <p>Status 3 Description</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Network graph visualization
        st.subheader("Graph Visualization Example")
        
        # Create sample graph
        sample_graph = generate_sample_graph()
        
        # Draw the graph
        graph_fig = draw_network_graph(sample_graph)
        st.pyplot(graph_fig)
        
        # Timeline visualization example
        st.subheader("Timeline Visualization")
        
        # Create timeline data
        timeline_data = [
            {"stage": "Stage 1", "description": "Description 1", "status": "status-1"},
            {"stage": "Stage 2", "description": "Description 2", "status": "status-2"},
            {"stage": "Stage 3", "description": "Description 3", "status": "status-3"},
            {"stage": "Stage 4", "description": "Description 4", "status": "status-1"}
        ]
        
        # Create timeline visualization with Plotly
        fig = go.Figure()
        
        # Add timeline nodes
        y_pos = 0
        status_colors = {
            "status-1": AWS_COLORS["green"],
            "status-2": AWS_COLORS["red"],
            "status-3": AWS_COLORS["orange"]
        }
        
        for i, stage in enumerate(timeline_data):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[y_pos],
                mode="markers+text",
                marker=dict(size=30, color=status_colors[stage["status"]]),
                text=[stage["stage"]],
                textposition="bottom center",
                hoverinfo="text",
                hovertext=[stage["description"]],
                name=stage["stage"]
            ))
        
        # Add connecting lines
        for i in range(len(timeline_data) - 1):
            fig.add_shape(
                type="line",
                x0=i,
                y0=y_pos,
                x1=i + 1,
                y1=y_pos,
                line=dict(color="gray", width=2, dash="dot")
            )
        
        # Configure layout
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add descriptions for timeline
        for stage in timeline_data:
            st.markdown(f"**{stage['stage']}:** {stage['description']}")
    
    # TAB 3: THIRD TAB
    with tab3:
        st.header("Third Tab Content")
        
        # Interactive form example
        st.subheader("Interactive Form Example")
        
        with st.form(key='example_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name")
                email = st.text_input("Email")
                
            with col2:
                option = st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
                value = st.slider("Select a value", 0, 100, 50)
                
            submit_button = st.form_submit_button(label='Submit')
            
            if submit_button:
                st.success("Form submitted!")
        
        # Expandable sections
        st.subheader("Expandable Sections")
        
        with st.expander("Section 1"):
            st.markdown("""
            This is an expandable section. You can put detailed content here.
            
            - Item 1
            - Item 2
            - Item 3
            """)
            
            st.code('''
            # Example code in expandable section
            def sample_function():
                return "Hello from Section 1"
            ''')
        
        with st.expander("Section 2"):
            st.markdown("""
            This is another expandable section.
            
            1. First item
            2. Second item
            3. Third item
            """)
            
            # Example of a simple chart in the expandable section
            chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['A', 'B', 'C'])
            st.line_chart(chart_data)
        
        with st.expander("Section 3"):
            st.markdown("""
            This is a third expandable section.
            
            * Point 1
            * Point 2
            * Point 3
            """)
            
            # Example of columns within expandable section
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Left column content**")
                st.image("https://via.placeholder.com/300x200", use_container_width=True)
                
            with col2:
                st.markdown("**Right column content**")
                st.code('''print("Hello from Section 3")''')
        
        # Final example - cards in a grid
        st.subheader("Card Grid Example")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Card Title 1</h4>
                <p>Card description goes here. This is a sample text to demonstrate the card layout.</p>
                <p><strong>Detail:</strong> Value</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
                <h4>Card Title 2</h4>
                <p>Card description goes here. This is a sample text to demonstrate the card layout.</p>
                <p><strong>Detail:</strong> Value</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="card">
                <h4>Card Title 3</h4>
                <p>Card description goes here. This is a sample text to demonstrate the card layout.</p>
                <p><strong>Detail:</strong> Value</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
