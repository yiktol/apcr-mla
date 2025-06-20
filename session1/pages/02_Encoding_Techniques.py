import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import base64
from io import BytesIO
import utils.common as common
import utils.authenticate as authenticate

# AWS Color Scheme
AWS_COLORS = {
    "orange": "#FF9900",
    "light_orange": "#FFAC31",
    "dark_blue": "#232F3E",
    "light_blue": "#1A73E8",
    "teal": "#00A1C9",
    "red": "#D13212",
    "green": "#7AA116",
    "purple": "#8C4FFF",
    "light_grey": "#F2F3F3"
}


def setup_page_config():
    """Configure the page settings"""
    st.set_page_config(
        page_title="ML Encoding Techniques",
        layout="wide",
        initial_sidebar_state="expanded"
    )

setup_page_config()

def apply_custom_styles():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #232F3E;
        color: white;
    }
    h1, h2 {
        color: #232F3E;
    }
    h3, h4, h5 {
        color: #FF9900;
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


def initialize_session_state():
    """Initialize all session state variables"""
    # Tab state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
        
    # Sample data
    if 'categorical_data' not in st.session_state:
        st.session_state.categorical_data = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'blue', 'red', 'green', 'red'],
            'size': ['small', 'medium', 'large', 'medium', 'small', 'large', 'large'],
            'rating': ['low', 'medium', 'high', 'low', 'medium', 'high', 'medium']
        })
        
    if 'binary_data' not in st.session_state:
        st.session_state.binary_data = pd.DataFrame({
            'has_feature': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
            'is_active': ['True', 'False', 'True', 'False', 'True', 'False', 'True']
        })
        
    if 'ordinal_data' not in st.session_state:
        st.session_state.ordinal_data = pd.DataFrame({
            'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'High School', 'Master'],
            'satisfaction': ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Medium', 'Low']
        })

    # Quiz state variables
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
        
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
        
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}


def render_sidebar():
    """Render the sidebar content"""
    with st.sidebar:
        common.render_sidebar()
        
        with st.expander("About This App", expanded=False):
            st.info("""
            This interactive tutorial demonstrates various encoding techniques 
            used in machine learning to convert categorical data into numerical format.
            
            Learn about:
            - Binary encoding
            - Nominal encoding
            - Ordinal encoding
            - One-hot encoding
            - Label encoding
            
            Try the interactive examples to see how each technique transforms data!
            """)


def render_header():
    """Render the page header and introduction"""
    st.title("Machine Learning Encoding Techniques")
    
    st.markdown("""
    <div class="card">
    <p>Machine learning algorithms require numerical data to work effectively. Encoding techniques transform 
    categorical (text-based) variables into numerical formats that ML models can process. Each technique 
    has specific use cases and impacts model performance differently.</p>
    </div>
    """, unsafe_allow_html=True)


def render_overview_tab():
    """Render the overview tab content"""
    st.header("Understanding Encoding Techniques")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Why Do We Need Encoding? 🤔
        
        Machine learning algorithms work with numbers, not text. When your data contains categorical features
        like colors, sizes, or any text-based attributes, you need to convert them to numbers before feeding
        them to a model.
        
        Different encoding techniques are appropriate for different types of categorical data:
        """)
        
        techniques = {
            "Binary Encoding": "For features with only two possible values (Yes/No, True/False)",
            "Label Encoding": "Assigns a unique integer to each category (0, 1, 2...)",
            "Ordinal Encoding": "For categories with a clear, meaningful order (Small, Medium, Large)",
            "One-Hot Encoding": "Creates binary columns for each category (best for nominal data)"
        }
        
        for technique, description in techniques.items():
            st.markdown(f"""
            <div style="margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: {AWS_COLORS['light_grey']}; border-left: 5px solid {AWS_COLORS['orange']}">
                <strong style="color: {AWS_COLORS['dark_blue']};">{technique}:</strong> {description}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Create a visual comparison of different encoding techniques
        fig, ax = plt.subplots(figsize=(8, 6))

        # Sample data for visualization
        categories = ['Red', 'Green', 'Blue']
        techniques = ['Original', 'Label', 'One-Hot', 'Ordinal']

        # Create the encoding comparison visual
        y_pos = np.arange(len(techniques))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(techniques)
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_title('Encoding Techniques Comparison')

        # Add text annotations for each cell
        for i, technique in enumerate(techniques):
            for j, category in enumerate(categories):
                if technique == 'Original':
                    text = category
                    color = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}[category]
                    ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')
                elif technique == 'Label':
                    values = [0, 1, 2]
                    ax.text(j, i, str(values[j]), ha='center', va='center')
                elif technique == 'One-Hot':
                    if category == 'Red':
                        text = "[1,0,0]"
                    elif category == 'Green':
                        text = "[0,1,0]" 
                    else:
                        text = "[0,0,1]"
                    ax.text(j, i, text, ha='center', va='center')
                elif technique == 'Ordinal':
                    values = [1, 2, 3]
                    ax.text(j, i, str(values[j]), ha='center', va='center')

        # Set grid
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # Hide spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)
        
        st.markdown("""
        <div style="text-align: center; font-style: italic; margin-top: 10px;">
            Visual representation of how different encoding techniques transform categorical data
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("When to Use Each Encoding Technique")
    
    # Create comparison table
    comparison_data = {
        'Technique': ['Binary Encoding', 'Label Encoding', 'Ordinal Encoding', 'One-Hot Encoding'],
        'Best For': ['Binary categories (Yes/No)', 'Multiple categories with no order', 'Ordered categories', 'Nominal categories (no inherent order)'],
        'Advantages': ['Simple, space-efficient', 'Simple implementation, space-efficient', 'Preserves ordering information', 'No ordinal relationship assumed'],
        'Disadvantages': ['Limited to two categories', 'Creates false ordinal relationships', 'Requires domain knowledge for ordering', 'Creates many new features'],
        'Example Use Case': ['Spam detection, sentiment (positive/negative)', 'Tree-based models, Target encoding', 'Education levels, satisfaction ratings', 'Colors, product categories, cities']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Style the dataframe
    def highlight_rows(row):
        styles = ['background-color: ' + AWS_COLORS['light_grey']] * len(row)
        return styles
    
    styled_df = df_comparison.style.apply(highlight_rows, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=300)
    
    # Visual decision flow
    st.subheader("Encoding Decision Flowchart")
    
    # Create a simple flowchart using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Hide axes
    ax.axis('off')
    
    # Define nodes and their positions
    nodes = {
        'start': (0.5, 0.9, "Categorical\nVariable"),
        'binary': (0.2, 0.7, "Binary?"),
        'ordered': (0.5, 0.5, "Has meaningful\norder?"),
        'high_card': (0.8, 0.3, "High\ncardinality?"),
        'bin_enc': (0.2, 0.3, "Binary\nEncoding"),
        'ord_enc': (0.5, 0.3, "Ordinal\nEncoding"),
        'onehot_enc': (0.65, 0.1, "One-Hot\nEncoding"),
        'label_enc': (0.95, 0.1, "Label\nEncoding")
    }
    
    # Define node colors
    node_colors = {
        'start': AWS_COLORS['dark_blue'],
        'binary': AWS_COLORS['light_blue'],
        'ordered': AWS_COLORS['light_blue'],
        'high_card': AWS_COLORS['light_blue'],
        'bin_enc': AWS_COLORS['orange'],
        'ord_enc': AWS_COLORS['orange'],
        'onehot_enc': AWS_COLORS['orange'],
        'label_enc': AWS_COLORS['orange']
    }
    
    # Draw nodes
    for node, (x, y, label) in nodes.items():
        circle = plt.Circle((x, y), 0.1, color=node_colors[node], alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if node_colors[node] != AWS_COLORS['orange'] else 'black')
    
    # Draw arrows
    def draw_arrow(start_node, end_node, label="", offset=(0, 0)):
        start_x, start_y, _ = nodes[start_node]
        end_x, end_y, _ = nodes[end_node]
        
        # Calculate angle for arrow placement
        angle = np.arctan2(end_y - start_y, end_x - start_x)
        start_x += 0.1 * np.cos(angle)
        start_y += 0.1 * np.sin(angle)
        end_x -= 0.1 * np.cos(angle)
        end_y -= 0.1 * np.sin(angle)
        
        ax.annotate("", 
                   xy=(end_x, end_y), 
                   xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle="->", color=AWS_COLORS['dark_blue'], lw=1.5))
        
        # Add label with offset
        if label:
            label_x = (start_x + end_x) / 2 + offset[0]
            label_y = (start_y + end_y) / 2 + offset[1]
            ax.text(label_x, label_y, label, ha='center', va='center', 
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Draw the connections
    draw_arrow('start', 'binary')
    draw_arrow('binary', 'bin_enc', 'Yes', (-0.05, 0))
    draw_arrow('binary', 'ordered', 'No', (0.05, 0))
    draw_arrow('ordered', 'ord_enc', 'Yes', (-0.05, 0))
    draw_arrow('ordered', 'high_card', 'No', (0.05, 0))
    draw_arrow('high_card', 'onehot_enc', 'No', (-0.05, -0.05))
    draw_arrow('high_card', 'label_enc', 'Yes', (0.05, -0.05))
    
    # Set limits to ensure all elements are visible
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    st.pyplot(fig)


def render_binary_encoding_tab():
    """Render the binary encoding tab content"""
    st.header("Binary Encoding")
    
    st.markdown("""
    <div class="card">
    <h3>What is Binary Encoding? 0️⃣1️⃣</h3>
    <p>Binary encoding is the simplest form of encoding, used when a categorical feature has only two possible values.
    Each value is mapped to either 0 or 1.</p>
    
    <h4>When to use Binary Encoding:</h4>
    <ul>
    <li>When your categorical variable has exactly two categories (Yes/No, True/False, Active/Inactive)</li>
    <li>For features representing presence or absence of a characteristic</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Data")
        st.dataframe(st.session_state.binary_data, use_container_width=True)
    
    with col2:
        st.subheader("After Binary Encoding")
        binary_encoded = st.session_state.binary_data.copy()
        
        # Perform binary encoding
        binary_encoded['has_feature'] = binary_encoded['has_feature'].map({'Yes': 1, 'No': 0})
        binary_encoded['is_active'] = binary_encoded['is_active'].map({'True': 1, 'False': 0})
        
        st.dataframe(binary_encoded, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("How Binary Encoding Works")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Binary encoding is straightforward:
        
        1. Identify a categorical feature with two possible values
        2. Decide which value corresponds to 1 and which to 0
        3. Replace the original values with their binary counterparts
        
        For example:
        - Yes → 1, No → 0
        - True → 1, False → 0
        - Success → 1, Failure → 0
        - Present → 1, Absent → 0
        
        ### Code Example
        ```python
        # Using mapping
        df['has_feature'] = df['has_feature'].map({'Yes': 1, 'No': 0})
        
        # Using scikit-learn's LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['has_feature'] = le.fit_transform(df['has_feature'])
        ```
        
        > Note: When using LabelEncoder for binary variables, make sure to check which value gets mapped to 1 and which to 0.
        """)
    
    with col2:
        # Create a visual representation of binary encoding
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create data for the visualization
        binary_values = ['Yes', 'No']
        encoded_values = [1, 0]
        
        # Create a simple mapping visualization
        y_pos = np.arange(len(binary_values))
        
        # Create bars for original values (transparent)
        ax.barh(y_pos, [0.5, 0.5], color='lightgray', alpha=0.3, height=0.4)
        
        # Create bars for encoded values
        bars = ax.barh(y_pos, encoded_values, color=AWS_COLORS['orange'], height=0.4)
        
        # Add text labels
        for i, v in enumerate(binary_values):
            ax.text(-0.1, i, v, ha='right', va='center', fontweight='bold', color=AWS_COLORS['dark_blue'])
            ax.text(encoded_values[i] + 0.05, i, str(encoded_values[i]), va='center', fontweight='bold')
        
        # Set the limits and remove spines
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, len(binary_values) - 0.5)
        ax.set_title('Binary Encoding Mapping', fontsize=14)
        ax.set_xlabel('Encoded Value')
        
        # Remove y-tick labels and set the ticks
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Show the plot
        st.pyplot(fig)
        
        # Add an example use case
        st.markdown("""
        <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; border-left: 5px solid #FF9900;">
        <strong>Example Use Case:</strong><br>
        Email spam classification where a message is either spam (1) or not spam (0).
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Try It Yourself")
    
    # Create a simple interactive binary encoding demo
    custom_col1, custom_col2 = st.columns(2)
    
    with custom_col1:
        category1 = st.text_input("Enter Category 1 (will be encoded as 1)", "Yes")
        category2 = st.text_input("Enter Category 2 (will be encoded as 0)", "No")
        
        # Create sample data
        sample_data = st.text_area("Enter sample data (one value per line)", "Yes\nNo\nYes\nYes\nNo").strip().split('\n')
    
    with custom_col2:
        if st.button("Encode Data", key="binary_encode"):
            # Create DataFrame
            df = pd.DataFrame({'Original': sample_data})
            
            # Encode
            df['Encoded'] = df['Original'].map({category1: 1, category2: 0})
            
            st.write("Encoded Data:")
            st.dataframe(df, use_container_width=True)
            
            # Create a simple bar chart
            value_counts = df['Original'].value_counts().reset_index()
            value_counts.columns = ['Category', 'Count']
            
            fig = px.bar(value_counts, x='Category', y='Count', 
                        color='Category',
                        color_discrete_map={category1: AWS_COLORS['orange'], category2: AWS_COLORS['teal']},
                        title="Distribution of Categories")
            
            st.plotly_chart(fig, use_container_width=True)


def render_label_encoding_tab():
    """Render the label encoding tab content"""
    st.header("Label Encoding")
    
    st.markdown("""
    <div class="card">
    <h3>What is Label Encoding? 🏷️</h3>
    <p>Label encoding transforms categorical variables into numerical labels by assigning a unique integer 
    to each category value. It's a simple way to convert text-based categories into numbers without creating 
    additional columns.</p>
    
    <h4>When to use Label Encoding:</h4>
    <ul>
    <li>When you have categorical features with multiple possible values</li>
    <li>When using tree-based models that can handle label-encoded data</li>
    <li>When you need to conserve memory (compared to one-hot encoding)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Data")
        sample_df = st.session_state.categorical_data[['color']].head()
        st.dataframe(sample_df, use_container_width=True)
    
    with col2:
        st.subheader("After Label Encoding")
        # Create a copy to avoid modifying the original
        encoded_df = sample_df.copy()
        
        # Apply label encoding
        label_encoder = LabelEncoder()
        encoded_df['color_encoded'] = label_encoder.fit_transform(encoded_df['color'])
        
        st.dataframe(encoded_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("How Label Encoding Works")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Label encoding works by replacing each unique category with a number:
        
        1. The encoder identifies all unique values in the categorical column
        2. Each unique value is assigned an integer (usually starting from 0)
        3. The original values are replaced with their corresponding integers
        
        For example, for colors:
        - blue → 0
        - green → 1
        - red → 2
        
        ### Code Example
        ```python
        # Using scikit-learn's LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        df['color_encoded'] = label_encoder.fit_transform(df['color'])
        
        # To get the original mapping
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(mapping)  # {'blue': 0, 'green': 1, 'red': 2}
        ```
        
        > ⚠️ **Warning**: Label encoding creates an arbitrary numerical order that doesn't necessarily reflect any meaningful relationship between categories. This can mislead models that assume numerical relationships.
        """)
    
    with col2:
        # Create a visual representation of label encoding
        # Get unique values from the color column and their encoded values
        unique_colors = sample_df['color'].unique()
        encoded_values = label_encoder.transform(unique_colors)
        
        # Create a mapping dataframe
        mapping_df = pd.DataFrame({'Category': unique_colors, 'Encoded': encoded_values})
        
        # Create a colorful horizontal bar chart
        fig = px.bar(mapping_df, x='Encoded', y='Category', orientation='h',
                    color='Category', 
                    color_discrete_map={'red': 'red', 'blue': 'blue', 'green': 'green'},
                    title='Label Encoding Mapping',
                    text='Encoded')
        
        # Update the layout
        fig.update_layout(
            xaxis_title='Encoded Value',
            yaxis_title='Original Category',
            showlegend=False
        )
        
        # Display bar chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add an example use case
        st.markdown("""
        <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; border-left: 5px solid #FF9900;">
        <strong>Example Use Case:</strong><br>
        Product category encoding for a recommendation system using a Random Forest model, where label encoding keeps the feature space manageable.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Advantages and Disadvantages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #E5F6E3; padding: 15px; border-radius: 5px;">
        <h4 style="color: #7AA116;">✅ Advantages</h4>
        <ul>
        <li>Memory efficient - creates only one additional column</li>
        <li>Simple to implement and understand</li>
        <li>Works well with tree-based models</li>
        <li>Preserves the original number of features</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #FEE7E4; padding: 15px; border-radius: 5px;">
        <h4 style="color: #D13212;">⚠️ Disadvantages</h4>
        <ul>
        <li>Creates a false ordinal relationship between categories</li>
        <li>Can mislead linear models by suggesting numerical ordering</li>
        <li>Not suitable for nominal categorical data</li>
        <li>May introduce bias in distance-based algorithms</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Try It Yourself")
    
    # Create a simple interactive label encoding demo
    st.markdown("Enter several category values (one per line) to see how label encoding works:")
    
    custom_categories = st.text_area("Categories", "Apple\nBanana\nOrange\nApple\nBanana\nGrape").strip().split('\n')
    
    if st.button("Encode Categories", key="label_encode_btn"):
        # Create DataFrame
        df = pd.DataFrame({'Original': custom_categories})
        
        # Apply label encoding
        le = LabelEncoder()
        df['Encoded'] = le.fit_transform(df['Original'])
        
        # Show mapping
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Encoded Data:")
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.write("Category Mapping:")
            mapping_df = pd.DataFrame(list(mapping.items()), columns=['Category', 'Encoded Value'])
            st.dataframe(mapping_df, use_container_width=True)
        
        # Create a bar chart showing frequency of each category
        fig = px.histogram(df, x='Original', color='Original',
                          title="Category Distribution",
                          labels={'Original': 'Category', 'count': 'Frequency'})
        
        st.plotly_chart(fig, use_container_width=True)


def render_ordinal_encoding_tab():
    """Render the ordinal encoding tab content"""
    st.header("Ordinal Encoding")
    
    st.markdown("""
    <div class="card">
    <h3>What is Ordinal Encoding? 🔢</h3>
    <p>Ordinal encoding transforms categorical variables into numerical values while preserving the natural 
    ordering between categories. This is ideal when your categories have a meaningful sequence or hierarchy.</p>
    
    <h4>When to use Ordinal Encoding:</h4>
    <ul>
    <li>When your categorical variable has a clear, meaningful order</li>
    <li>Examples: education levels (High School < Bachelor < Master < PhD)</li>
    <li>Examples: satisfaction ratings (Very Low < Low < Medium < High < Very High)</li>
    <li>When the ordinal relationship between categories is important for your model</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Data")
        st.dataframe(st.session_state.ordinal_data, use_container_width=True)
    
    with col2:
        st.subheader("After Ordinal Encoding")
        # Create a copy to avoid modifying the original
        encoded_df = st.session_state.ordinal_data.copy()
        
        # Define the ordinal categories and their order
        education_categories = ['High School', 'Bachelor', 'Master', 'PhD']
        satisfaction_categories = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        
        # Apply ordinal encoding
        education_mapping = {category: i for i, category in enumerate(education_categories)}
        satisfaction_mapping = {category: i for i, category in enumerate(satisfaction_categories)}
        
        encoded_df['education_encoded'] = encoded_df['education'].map(education_mapping)
        encoded_df['satisfaction_encoded'] = encoded_df['satisfaction'].map(satisfaction_mapping)
        
        st.dataframe(encoded_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("How Ordinal Encoding Works")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Ordinal encoding works by replacing categories with numbers that reflect their order or rank:
        
        1. Define the ordering or hierarchy of your categories
        2. Assign increasing integers based on that order
        3. Replace the original values with their corresponding integers
        
        For example, for education levels:
        - High School → 0
        - Bachelor → 1
        - Master → 2
        - PhD → 3
        
        ### Code Example
        ```python
        # Using a custom mapping
        education_categories = ['High School', 'Bachelor', 'Master', 'PhD']
        education_mapping = {category: i for i, category in enumerate(education_categories)}
        df['education_encoded'] = df['education'].map(education_mapping)
        
        # Using scikit-learn's OrdinalEncoder
        from sklearn.preprocessing import OrdinalEncoder
        
        encoder = OrdinalEncoder(categories=[education_categories])
        df[['education_encoded']] = encoder.fit_transform(df[['education']])
        ```
        
        > 💡 **Key Difference from Label Encoding**: With ordinal encoding, the order is defined manually based on domain knowledge. Label encoding assigns numbers arbitrarily, usually alphabetically.
        """)
    
    with col2:
        # Create a visual representation of ordinal encoding for education
        fig = go.Figure()
        
        for i, category in enumerate(education_categories):
            fig.add_trace(go.Bar(
                x=[i],
                y=[1],
                name=category,
                text=str(i),
                textposition='inside',
                marker_color=AWS_COLORS['teal'],
                width=0.6
            ))
        
        # Update the layout
        fig.update_layout(
            title='Ordinal Encoding for Education Levels',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(education_categories))),
                ticktext=education_categories,
                title='Education Level'
            ),
            yaxis=dict(
                visible=False
            ),
            showlegend=False,
            height=300
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Add an example use case
        st.markdown("""
        <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; border-left: 5px solid #FF9900;">
        <strong>Example Use Case:</strong><br>
        Predicting salary based on education level, where the hierarchy of education is meaningful for the model.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Ordinal vs. Label Encoding")
    
    st.markdown("""
    <table style="width:100%; border-collapse: collapse;">
    <tr style="background-color: #F2F3F3;">
        <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Feature</th>
        <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Label Encoding</th>
        <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Ordinal Encoding</th>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #ddd;">Order assignment</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Arbitrary (often alphabetical)</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Based on meaningful order</td>
    </tr>
    <tr style="background-color: #F2F3F3;">
        <td style="padding: 10px; border: 1px solid #ddd;">Domain knowledge</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Not required</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Required to establish order</td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #ddd;">Suitable for</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Nominal categories</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Ordinal categories</td>
    </tr>
    <tr style="background-color: #F2F3F3;">
        <td style="padding: 10px; border: 1px solid #ddd;">Implementation</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Automatic</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Requires manual category ordering</td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #ddd;">Model interpretation</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Numbers may mislead</td>
        <td style="padding: 10px; border: 1px solid #ddd;">Numbers represent true order</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Try It Yourself")
    
    # Create an interactive ordinal encoding demo
    st.markdown("Define your ordinal categories and their order:")
    
    # Let user input ordered categories
    ordered_categories = st.text_area("Enter categories in order (one per line, from lowest to highest)", 
                                    "Beginner\nIntermediate\nAdvanced\nExpert").strip().split('\n')
    
    # Let user input data to encode
    data_to_encode = st.text_area("Enter data to encode (one value per line)", 
                                "Intermediate\nBeginner\nExpert\nAdvanced\nIntermediate").strip().split('\n')
    
    if st.button("Apply Ordinal Encoding", key="ordinal_encode_btn"):
        # Create mapping
        ordinal_mapping = {category: i for i, category in enumerate(ordered_categories)}
        
        # Create DataFrame
        df = pd.DataFrame({'Original': data_to_encode})
        
        # Apply encoding
        df['Encoded'] = df['Original'].map(ordinal_mapping)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Encoded Data:")
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.write("Ordinal Mapping:")
            mapping_df = pd.DataFrame(list(ordinal_mapping.items()), columns=['Category', 'Encoded Value'])
            st.dataframe(mapping_df, use_container_width=True)
        
        # Create a visual representation of the ordinal mapping
        fig = px.bar(mapping_df, x='Category', y='Encoded Value', 
                    color='Encoded Value', 
                    color_continuous_scale=px.colors.sequential.Blues,
                    title="Ordinal Mapping Visualization")
        
        fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':ordered_categories})
        
        st.plotly_chart(fig, use_container_width=True)


def render_onehot_encoding_tab():
    """Render the one-hot encoding tab content"""
    st.header("One-Hot Encoding")
    
    st.markdown("""
    <div class="card">
    <h3>What is One-Hot Encoding? 🔥</h3>
    <p>One-hot encoding transforms categorical variables into a binary representation by creating a new column 
    for each unique category value. For each record, the column representing its category contains a 1 
    (or "hot"), while all other columns contain 0.</p>
    
    <h4>When to use One-Hot Encoding:</h4>
    <ul>
    <li>When your categorical variable has no intrinsic order (nominal data)</li>
    <li>When using models that assume relationships between numeric values</li>
    <li>When the number of unique categories is reasonably small</li>
    <li>For linear models, SVMs, neural networks, and other algorithms sensitive to numeric relationships</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Original Data")
        sample_df = st.session_state.categorical_data[['color']].head(3)
        st.dataframe(sample_df, use_container_width=True)
    
    with col2:
        st.subheader("After One-Hot Encoding")
        # Create a copy to avoid modifying the original
        onehot_encoded = pd.get_dummies(sample_df, columns=['color'], prefix=['color'])
        
        st.dataframe(onehot_encoded, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("How One-Hot Encoding Works")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        One-hot encoding works by following these steps:
        
        1. Identify all unique values in the categorical column
        2. Create a new binary column for each unique value
        3. For each record, set a value of 1 in the column that matches its category, and 0 in all other columns
        
        For example, for colors red, blue, and green:
        
        | Color | color_red | color_blue | color_green |
        |-------|-----------|------------|-------------|
        | red   | 1         | 0          | 0           |
        | blue  | 0         | 1          | 0           |
        | green | 0         | 0          | 1           |
        
        ### Code Example
        ```python
        # Using pandas get_dummies
        df_encoded = pd.get_dummies(df, columns=['color'], prefix=['color'])
        
        # Using scikit-learn's OneHotEncoder
        from sklearn.preprocessing import OneHotEncoder
        
        encoder = OneHotEncoder(sparse=False)
        encoded_array = encoder.fit_transform(df[['color']])
        
        # Convert the array back to a DataFrame
        encoded_df = pd.DataFrame(
            encoded_array, 
            columns=encoder.get_feature_names_out(['color'])
        )
        ```
        
        > 💡 **Note**: One-hot encoding increases the dimensionality of your dataset by creating N new columns for N unique categories (or N-1 if using "drop_first=True" to avoid multicollinearity).
        """)
    
    with col2:
        # Create a visual representation of one-hot encoding
        # Create sample data
        colors = ['red', 'blue', 'green']
        
        # Create a heatmap-like visualization
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        # Create a matrix for the heatmap
        data = np.eye(len(colors))
        
        # Plot the heatmap
        im = ax.imshow(data, cmap='YlOrRd')
        
        # Set the ticks
        ax.set_xticks(np.arange(len(colors)))
        ax.set_yticks(np.arange(len(colors)))
        
        # Label the axes
        ax.set_xticklabels([f"color_{c}" for c in colors])
        ax.set_yticklabels(colors)
        
        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(colors)):
            for j in range(len(colors)):
                text = ax.text(j, i, int(data[i, j]),
                              ha="center", va="center", color="black")
        
        # Set title
        ax.set_title("One-Hot Encoding Visualization")
        
        # Display the figure
        st.pyplot(fig)
        
        # Add an example use case
        st.markdown("""
        <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; border-left: 5px solid #FF9900;">
        <strong>Example Use Case:</strong><br>
        Encoding city names for a linear regression model predicting house prices, where each city should be treated independently without implying any ordering.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Advantages and Disadvantages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #E5F6E3; padding: 15px; border-radius: 5px;">
        <h4 style="color: #7AA116;">✅ Advantages</h4>
        <ul>
        <li>No implied ordinal relationship between categories</li>
        <li>Each category is treated independently</li>
        <li>Works well with most ML algorithms</li>
        <li>Improves model performance for linear models</li>
        <li>Allows models to learn separate weights for each category</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #FEE7E4; padding: 15px; border-radius: 5px;">
        <h4 style="color: #D13212;">⚠️ Disadvantages</h4>
        <ul>
        <li>Creates many new features (dimensionality explosion)</li>
        <li>May lead to sparse datasets with many zeros</li>
        <li>Memory intensive for high-cardinality features</li>
        <li>Can cause multicollinearity in linear models</li>
        <li>Computationally expensive for large datasets</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dummy Variable Trap explanation
    st.subheader("The Dummy Variable Trap")
    
    st.markdown("""
    <div class="card">
    <p>When using one-hot encoding with linear models, you might encounter the <strong>dummy variable trap</strong>. 
    This occurs because the sum of all one-hot encoded columns will always equal 1, creating perfect multicollinearity.</p>
    
    <p>To avoid this problem, you can drop one of the categorical columns:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Full one-hot encoding
pd.get_dummies(df, columns=['color'])
        """)
        
        full_onehot = pd.get_dummies(pd.DataFrame({'color': ['red', 'blue', 'green']}), columns=['color'])
        st.dataframe(full_onehot)
    
    with col2:
        st.code("""
# One-hot with first category dropped
pd.get_dummies(df, columns=['color'], drop_first=True)
        """)
        
        dropped_onehot = pd.get_dummies(pd.DataFrame({'color': ['red', 'blue', 'green']}), columns=['color'], drop_first=True)
        st.dataframe(dropped_onehot)
        
        st.markdown("""
        <div style="font-size: 0.9em; background-color: #E9F5FF; padding: 10px; border-radius: 5px;">
        When drop_first=True, the first category becomes the reference category. For example, if 'red' is dropped, 
        then 'color_blue=0' and 'color_green=0' would implicitly mean 'red'.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Try It Yourself")
    
    # Create an interactive one-hot encoding demo
    st.markdown("Enter several category values to see how one-hot encoding works:")
    
    # Let user input data
    user_categories = st.text_area("Enter data (one value per line)", "Apple\nBanana\nApple\nOrange\nBanana").strip().split('\n')
    drop_first = st.checkbox("Drop first category (avoid dummy variable trap)")
    
    if st.button("Apply One-Hot Encoding", key="onehot_encode_btn"):
        # Create DataFrame
        df = pd.DataFrame({'Category': user_categories})
        
        # Apply one-hot encoding
        encoded_df = pd.get_dummies(df, columns=['Category'], prefix=['cat'], drop_first=drop_first)
        
        # Combine with original for display
        result_df = pd.concat([df, encoded_df], axis=1)
        
        st.write("Data with One-Hot Encoding:")
        st.dataframe(result_df, use_container_width=True)
        
        # Create a heatmap visualization
        unique_categories = df['Category'].unique()
        
        # Create heatmap data
        heatmap_data = []
        column_names = encoded_df.columns.tolist()
        
        for cat in unique_categories:
            row_data = []
            for col in column_names:
                # Find a row with this category and get its value in this column
                matching_row = (df['Category'] == cat)
                if matching_row.any():
                    row_data.append(encoded_df.loc[matching_row, col].iloc[0])
                else:
                    row_data.append(0)
            heatmap_data.append(row_data)
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Encoded Columns", y="Original Categories", color="Value"),
            x=column_names,
            y=unique_categories,
            color_continuous_scale="Oranges",
            title="One-Hot Encoding Visualization"
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cardinality impact
        st.markdown(f"""
        <div style="background-color: #E9F5FF; padding: 10px; border-radius: 5px;">
        <strong>Impact on Dimensionality:</strong><br>
        Original feature count: 1<br>
        After one-hot encoding: {len(encoded_df.columns)} features
        </div>
        """, unsafe_allow_html=True)


def render_interactive_lab_tab():
    """Render the interactive lab tab content"""
    st.header("Interactive Encoding Lab")
    
    st.markdown("""
    <div class="card">
    <p>In this interactive lab, you can explore and compare different encoding techniques on the same dataset.
    Select a dataset, apply various encoding methods, and visualize the results to understand the differences.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset selection
    st.subheader("1. Select a Dataset")
    
    dataset_option = st.selectbox(
        "Choose a dataset",
        ["Sample Customer Data", "Product Categories", "Upload Your Own"],
        index=0
    )
    
    df = None
    if dataset_option == "Sample Customer Data":
        df = pd.DataFrame({
            'customer_id': range(1, 11),
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'age_group': ['18-25', '26-35', '36-45', '46-55', '55+', '18-25', '26-35', '36-45', '46-55', '55+'],
            'state': ['CA', 'NY', 'TX', 'FL', 'WA', 'CA', 'NY', 'TX', 'FL', 'WA'],
            'subscription': ['Basic', 'Premium', 'Premium', 'Basic', 'Basic', 'Premium', 'Basic', 'Premium', 'Basic', 'Premium'],
            'satisfaction': ['Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Medium']
        })
    elif dataset_option == "Product Categories":
        df = pd.DataFrame({
            'product_id': range(101, 111),
            'category': ['Electronics', 'Clothing', 'Home', 'Beauty', 'Food', 'Electronics', 'Clothing', 'Home', 'Beauty', 'Food'],
            'price_tier': ['Budget', 'Mid-range', 'Premium', 'Budget', 'Mid-range', 'Premium', 'Budget', 'Mid-range', 'Premium', 'Budget'],
            'in_stock': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'],
            'season': ['Winter', 'Spring', 'Summer', 'Fall', 'Winter', 'Spring', 'Summer', 'Fall', 'Winter', 'Spring']
        })
    else:  # Upload Your Own
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file to continue")
    
    # Display the original dataset
    if df is not None:
        st.write("Original Dataset:")
        st.dataframe(df, use_container_width=True)
        
        # Feature selection
        st.subheader("2. Select Features to Encode")
        
        # Identify potential categorical columns (exclude numeric columns)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            selected_features = st.multiselect(
                "Select categorical features to encode",
                options=categorical_cols,
                default=categorical_cols[:min(2, len(categorical_cols))]
            )
            
            if selected_features:
                # Encoding selection
                st.subheader("3. Apply Encoding Techniques")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    apply_label = st.checkbox("Label Encoding", value=True)
                
                with col2:
                    apply_onehot = st.checkbox("One-Hot Encoding", value=True)
                    drop_first_onehot = st.checkbox("Drop First Category", value=False)
                
                with col3:
                    apply_ordinal = st.checkbox("Ordinal Encoding", value=True)
                    
                    # Only show ordinal map creation if ordinal encoding is selected
                    if apply_ordinal and len(selected_features) > 0:
                        st.markdown("Define order for ordinal features:")
                        
                        # Create collapsible sections for each selected feature
                        ordinal_mappings = {}
                        for feature in selected_features:
                            with st.expander(f"Order for {feature}"):
                                unique_values = df[feature].unique().tolist()
                                ordered_values = st.multiselect(
                                    f"Arrange values from lowest to highest",
                                    options=unique_values,
                                    default=sorted(unique_values)
                                )
                                
                                if len(ordered_values) == len(unique_values):
                                    ordinal_mappings[feature] = {val: idx for idx, val in enumerate(ordered_values)}
                                else:
                                    st.warning(f"Please select all values for {feature}")
                
                # Process button
                if st.button("Process and Compare Encodings"):
                    # Initialize result dictionary
                    encoded_dfs = {"Original": df[selected_features].copy()}
                    
                    # Apply Label Encoding
                    if apply_label:
                        label_encoded = df[selected_features].copy()
                        for col in selected_features:
                            le = LabelEncoder()
                            label_encoded[col] = le.fit_transform(df[col])
                        encoded_dfs["Label Encoded"] = label_encoded
                    
                    # Apply One-Hot Encoding
                    if apply_onehot:
                        onehot_encoded = pd.get_dummies(
                            df[selected_features], 
                            columns=selected_features,
                            drop_first=drop_first_onehot
                        )
                        encoded_dfs["One-Hot Encoded"] = onehot_encoded
                    
                    # Apply Ordinal Encoding
                    if apply_ordinal and all(feature in ordinal_mappings for feature in selected_features):
                        ordinal_encoded = df[selected_features].copy()
                        for col in selected_features:
                            ordinal_encoded[col] = ordinal_encoded[col].map(ordinal_mappings[col])
                        encoded_dfs["Ordinal Encoded"] = ordinal_encoded
                    
                    # Display results
                    st.subheader("4. Encoding Results Comparison")
                    
                    # Create tabs for each encoding method
                    encoding_tabs = st.tabs(list(encoded_dfs.keys()))
                    
                    for i, (name, encoded_df) in enumerate(encoded_dfs.items()):
                        with encoding_tabs[i]:
                            st.dataframe(encoded_df, use_container_width=True)
                            
                            # Additional info for specific encoding types
                            if name == "Label Encoded":
                                st.info("Label encoding assigns a unique integer to each category value, often alphabetically.")
                            elif name == "One-Hot Encoded":
                                cols_before = len(selected_features)
                                cols_after = len(encoded_df.columns)
                                st.info(f"One-hot encoding expanded {cols_before} feature(s) into {cols_after} binary columns.")
                            elif name == "Ordinal Encoded":
                                st.info("Ordinal encoding preserves the order defined for each category.")
                    
                    # Show visualization for a single selected feature
                    if len(selected_features) > 0:
                        st.subheader("5. Visual Comparison")
                        
                        feature_to_visualize = st.selectbox(
                            "Select a feature to visualize encodings",
                            options=selected_features
                        )
                        
                        # Create visualization data
                        viz_data = pd.DataFrame({
                            'Category': df[feature_to_visualize].unique()
                        })
                        
                        # Add encoded values for each method
                        if apply_label:
                            le = LabelEncoder()
                            le.fit(df[feature_to_visualize])
                            viz_data['Label Encoded'] = le.transform(viz_data['Category'])
                        
                        if apply_ordinal and feature_to_visualize in ordinal_mappings:
                            viz_data['Ordinal Encoded'] = viz_data['Category'].map(ordinal_mappings[feature_to_visualize])
                        
                        # Calculate the frequency of each value
                        value_counts = df[feature_to_visualize].value_counts()
                        viz_data['Count'] = viz_data['Category'].map(lambda x: value_counts.get(x, 0))
                        
                        # Create side-by-side bar chart comparing encodings
                        fig = go.Figure()
                        
                        # Label encoding bars
                        if apply_label:
                            fig.add_trace(go.Bar(
                                x=viz_data['Category'],
                                y=viz_data['Label Encoded'],
                                name='Label Encoded',
                                marker_color=AWS_COLORS['teal'],
                                text=viz_data['Label Encoded'],
                                textposition='auto'
                            ))
                        
                        # Ordinal encoding bars
                        if apply_ordinal and feature_to_visualize in ordinal_mappings:
                            fig.add_trace(go.Bar(
                                x=viz_data['Category'],
                                y=viz_data['Ordinal Encoded'],
                                name='Ordinal Encoded',
                                marker_color=AWS_COLORS['orange'],
                                text=viz_data['Ordinal Encoded'],
                                textposition='auto'
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f'Encoding Comparison for {feature_to_visualize}',
                            xaxis_title='Category Value',
                            yaxis_title='Encoded Value',
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # If one-hot encoding is applied, show heatmap
                        if apply_onehot:
                            st.subheader(f"One-Hot Encoding for {feature_to_visualize}")
                            
                            # Get one-hot columns for the selected feature
                            onehot_columns = [col for col in encoded_dfs["One-Hot Encoded"].columns if col.startswith(feature_to_visualize)]
                            
                            if onehot_columns:
                                # Create a matrix for the heatmap
                                categories = df[feature_to_visualize].unique()
                                onehot_matrix = np.zeros((len(categories), len(onehot_columns)))
                                
                                # Fill the matrix
                                for i, category in enumerate(categories):
                                    # Get a sample row with this category
                                    sample_row = df[df[feature_to_visualize] == category].index[0]
                                    
                                    # Get the one-hot values for this sample
                                    for j, col in enumerate(onehot_columns):
                                        onehot_matrix[i, j] = encoded_dfs["One-Hot Encoded"].loc[sample_row, col]
                                
                                # Create heatmap
                                fig = px.imshow(
                                    onehot_matrix,
                                    labels=dict(x="One-Hot Columns", y="Original Categories", color="Value"),
                                    x=onehot_columns,
                                    y=categories,
                                    color_continuous_scale="Oranges",
                                    title=f"One-Hot Encoding for {feature_to_visualize}"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one feature to encode")
        else:
            st.error("No categorical features found in the dataset")
    
    # Key takeaways and summary
    st.markdown("---")
    st.subheader("Key Takeaways")
    
    st.markdown("""
    <div class="card">
    <h4>Choosing the Right Encoding Technique</h4>
    <ul>
    <li><strong>Binary Encoding:</strong> For yes/no or true/false features</li>
    <li><strong>Label Encoding:</strong> When the feature has no inherent order and you're using tree-based models</li>
    <li><strong>Ordinal Encoding:</strong> When the feature values have a meaningful order</li>
    <li><strong>One-Hot Encoding:</strong> When the feature has no inherent order and you're using linear models</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h4>Impact on Model Performance</h4>
    <p>The choice of encoding technique can significantly impact model performance:</p>
    <ul>
    <li>Using label encoding for nominal data can create false relationships in linear models</li>
    <li>One-hot encoding can lead to the "curse of dimensionality" if you have many categorical features with high cardinality</li>
    <li>Using the right encoding for your data type and model can improve accuracy and interpretability</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def submit_quiz():
    """Handle quiz submission logic"""
    questions = get_quiz_questions()
    score = 0
    for q_idx, question in enumerate(questions):
        if st.session_state.quiz_answers.get(f"q{q_idx}") == question["correct"]:
            score += 1
    st.session_state.quiz_score = score
    st.session_state.quiz_submitted = True


def reset_quiz():
    """Reset the quiz state"""
    st.session_state.quiz_score = 0
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers = {}


def get_quiz_questions():
    """Return the list of quiz questions"""
    return [
        {
            "question": "Which encoding technique is most appropriate for categorical variables with a clear inherent order?",
            "options": ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding", "Binary Encoding"],
            "correct": "Ordinal Encoding",
            "explanation": "Ordinal encoding is specifically designed for categorical variables that have a natural order or hierarchy, like education levels or satisfaction ratings."
        },
        {
            "question": "For nominal categorical data with high cardinality (many unique values), which encoding technique is typically recommended?",
            "options": ["One-Hot Encoding", "Label Encoding", "Binary Encoding", "Target Encoding"],
            "correct": "Label Encoding",
            "explanation": "Label encoding is often preferred for high cardinality features as one-hot encoding would create too many columns, potentially leading to the curse of dimensionality."
        },
        {
            "question": "What potential issue does label encoding introduce when used with linear models?",
            "options": [
                "It creates too many features", 
                "It implies an ordinal relationship that might not exist",
                "It's too memory-intensive", 
                "It removes important information from the original data"
            ],
            "correct": "It implies an ordinal relationship that might not exist",
            "explanation": "Label encoding assigns numeric values (like 0, 1, 2) to categories, which can mislead linear models into interpreting a relationship between categories that doesn't exist in reality."
        },
        {
            "question": "Which encoding method is most suitable for binary categorical features like Yes/No or True/False?",
            "options": ["One-Hot Encoding", "Binary Encoding", "Ordinal Encoding", "Count Encoding"],
            "correct": "Binary Encoding",
            "explanation": "Binary encoding is the simplest and most appropriate technique for variables with exactly two categories, mapping them to 0 and 1."
        },
        {
            "question": "What problem does dropping the first category (drop_first=True) solve in one-hot encoding?",
            "options": ["Reduces memory usage", "Improves model accuracy", "Avoids the dummy variable trap", "Makes the model train faster"],
            "correct": "Avoids the dummy variable trap",
            "explanation": "Dropping the first category helps avoid the dummy variable trap (multicollinearity) in linear models, as the full set of one-hot encoded columns would be perfectly correlated."
        }
    ]


def render_knowledge_check_tab():
    """Render the knowledge check quiz tab"""
    st.header("Test Your Knowledge")
    st.markdown("Let's see how well you understand encoding techniques for machine learning!")
    
    # Get quiz questions
    questions = get_quiz_questions()
    
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
            st.success("🏆 Perfect score! You're an encoding expert!")
        elif st.session_state.quiz_score >= len(questions) * 0.8:
            st.success("🎓 Great job! You have a strong understanding of encoding techniques.")
        elif st.session_state.quiz_score >= len(questions) * 0.6:
            st.warning("📚 Good effort! Review the explanations to strengthen your knowledge.")
        else:
            st.error("🔄 You might want to revisit the earlier sections to reinforce your understanding.")
        
        if st.button("Take Quiz Again"):
            reset_quiz()


def render_footer():
    """Render the page footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app"""
    # Setup page configuration and styling
    apply_custom_styles()
    
    # Initialize session state
    initialize_session_state()
    common.initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render page header
    render_header()
    
    # Create tabs
    tabs = st.tabs([
        "📊 Overview", 
        "0️⃣1️⃣ Binary Encoding", 
        "🏷️ Label Encoding", 
        "🔢 Ordinal Encoding", 
        "🔥 One-Hot Encoding", 
        "🧪 Interactive Lab",
        "📋 Knowledge Check"
    ])
    
    # Render content for each tab
    with tabs[0]:
        render_overview_tab()
        
    with tabs[1]:
        render_binary_encoding_tab()
        
    with tabs[2]:
        render_label_encoding_tab()
        
    with tabs[3]:
        render_ordinal_encoding_tab()
        
    with tabs[4]:
        render_onehot_encoding_tab()
        
    with tabs[5]:
        render_interactive_lab_tab()
        
    with tabs[6]:
        render_knowledge_check_tab()
    
    # Render footer
    render_footer()


# Main execution flow
if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()
