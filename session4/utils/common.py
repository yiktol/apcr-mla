import streamlit as st
import uuid
from datetime import datetime
import utils.authenticate as authenticate
import streamlit.components.v1 as components


def reset_session():
    """Reset the session state"""
    for key in st.session_state.keys():
        if key not in ["authenticated", "user_cognito_groups", "auth_code","user_info"]:
            del st.session_state[key]  
    
def render_sidebar():
    """Render the sidebar with session information and reset button"""
    st.markdown("#### 🔑 Session Info")
    if 'auth_code' not in st.session_state:
        st.caption(f"**Session ID:** {st.session_state.session_id[:8]}")
    else:
        st.caption(f"**Session ID:** {st.session_state['auth_code'][:8]}")

    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()
        st.success("Session has been reset successfully!")
        st.rerun()  # Force a rerun to refresh the page

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# def reset_session():
#     """Reset all session state variables."""
    
#     # Keep only the session ID but generate a new one
#     st.session_state.session_id = str(uuid.uuid4())[:8]
#     st.session_state.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
#     # Reset all game scores and submissions
#     for i in range(1, 9):  # 8 games
#         st.session_state[f"game{i}_score"] = 0
#         st.session_state[f"game{i}_submitted"] = [False] * 5

def show_tip(tip_text):
    """Display a formatted tip box with the provided text."""
    
    st.markdown(f"""
    <div class="tip-box">
        <strong>💡 Learning Tip:</strong> {tip_text}
    </div>
    """, unsafe_allow_html=True)

       
def apply_styles():
    """Apply custom styling to the Streamlit app."""
    
    # AWS color scheme
    aws_orange = "#FF9900"
    aws_dark = "#232F3E"
    aws_blue = "#1A73E8"
    aws_background = "#FFFFFF"
    
    # CSS for styling
    st.markdown(f"""
    <style>
        .main-header {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #FF9900;
            margin-bottom: 1rem;
        }}
        .sub-header {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #232F3E;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }}
        .section-header {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #232F3E;
            margin-top: 0.8rem;
            margin-bottom: 0.3rem;
        }}
        .info-box {{
            background-color: #F0F2F6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .success-box {{
            background-color: #D1FAE5;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .warning-box {{
            background-color: #FEF3C7;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .tip-box {{
            background-color: #E0F2FE;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #0EA5E9;
        }}
        .step-box {{
            background-color: #FFFFFF;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        .card {{
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: white;
            transition: transform 0.3s;
        }}
        .card:hover {{
            transform: translateY(-5px);
        }}
        .aws-orange {{
            color: #FF9900;
        }}
        .aws-blue {{
            color: #232F3E;
        }}
        hr {{
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        /* Make the tab content container take full height */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: #F8F9FA;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-left: 16px;
            padding-right: 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #FF9900 !important;
            color: white !important;
        }}
        .definition {{
            background-color: #EFF6FF;
            border-left: 5px solid #3B82F6;
            padding: 10px 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }}
        .code-box {{
            background-color: #F8F9FA;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            margin: 15px 0;
            border: 1px solid #E5E7EB;
        }}
        .stButton>button {{
            background-color: #FF9900;
            color: white;
        }}
        .stButton>button:hover {{
            background-color: #FFAC31;
        }}    
        .stApp {{
            color: {aws_dark};
            background-color: {aws_background};
            font-family: 'Amazon Ember', Arial, sans-serif;
        }}
        
        
        /* Success styling */
        .correct-answer {{
            background-color: #D4EDDA;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        
        /* Error styling */
        .incorrect-answer {{
            background-color: #F8D7DA;
            color: #721C24;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        
        /* Custom card styling */
        .game-card {{
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        /* Progress bar styling */
        .stProgress > div > div > div {{
            background-color: {aws_orange};
        }}
        
        /* Tip box styling */
        .tip-box {{
            background-color: #E7F3FE;
            border-left: 6px solid {aws_blue};
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        
        /* Make images responsive */
        img {{
            max-width: 100%;
            height: auto;
        }}
        
        /* Score display */
        .score-display {{
            font-size: 1.2rem;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin: 15px 0;
            background-color: #F1F8FF;
        }}
    </style>
    """, unsafe_allow_html=True)

def mermaid(code: str, height: int = 100) -> None:
    """Render Mermaid diagrams in Streamlit.
    
    Args:
        code: The Mermaid diagram code to render
        height: Height of the diagram container in pixels (default: 600)
    """
    components.html(
        f"""
        <div class="mermaid-container">
            <pre class="mermaid">{code.strip()}</pre>
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        </script>
        <style>
            .mermaid-container {{
                height: {height}px;
                # border: 1px solid #ccc;
                padding: 10px;
                overflow: auto;
            }}
        </style>
        """,
        height=height + 20,  # Account for container height + border and padding
    )