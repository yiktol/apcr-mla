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

def mermaid(code: str, height: int = None) -> None:
    """Render Mermaid diagrams in Streamlit with smart height adjustment and zoom controls.
    
    Args:
        code: The Mermaid diagram code to render
        height: Optional fixed height in pixels. If None, height is auto-calculated.
    """
    
    def estimate_diagram_height(mermaid_code: str) -> int:
        """Quick estimation for initial render."""
        lines = [line.strip() for line in mermaid_code.strip().split('\n') if line.strip()]
        
        # Count meaningful lines (exclude comments and styling)
        content_lines = [line for line in lines 
                        if not line.startswith('%%') 
                        and not line.startswith('classDef') 
                        and not line.startswith('class')]
        
        diagram_type = lines[0].lower() if lines else ""
        node_count = len([line for line in content_lines if '-->' in line or '{' in line])
        
        if 'flowchart' in diagram_type and ('td' in diagram_type or 'tb' in diagram_type):
            return max(300, min(800, 200 + node_count * 60))
        else:
            return max(250, min(600, 150 + node_count * 50))
    
    # Get initial height estimate
    if height is None:
        height = estimate_diagram_height(code)
    
    components.html(
        f"""
        <div class="mermaid-wrapper" id="mermaid-wrapper-{hash(code) % 10000}">
            <!-- Zoom Controls -->
            <div class="zoom-controls">
                <button id="zoom-in-{hash(code) % 10000}" class="zoom-btn" title="Zoom In">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <path d="m21 21-4.35-4.35"></path>
                        <line x1="8" y1="11" x2="14" y2="11"></line>
                        <line x1="11" y1="8" x2="11" y2="14"></line>
                    </svg>
                </button>
                <button id="zoom-out-{hash(code) % 10000}" class="zoom-btn" title="Zoom Out">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <path d="m21 21-4.35-4.35"></path>
                        <line x1="8" y1="11" x2="14" y2="11"></line>
                    </svg>
                </button>
                <button id="zoom-reset-{hash(code) % 10000}" class="zoom-btn" title="Reset Zoom">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 3l18 18"></path>
                        <path d="M9 9h6v6"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                </button>
                <span id="zoom-level-{hash(code) % 10000}" class="zoom-level">100%</span>
            </div>
            
            <!-- Mermaid Container -->
            <div class="mermaid-container" id="mermaid-container-{hash(code) % 10000}">
                <div class="mermaid-content" id="mermaid-content-{hash(code) % 10000}">
                    <pre class="mermaid">{code.strip()}</pre>
                </div>
            </div>
        </div>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
            
            mermaid.initialize({{ 
                startOnLoad: false, 
                theme: 'default',
                flowchart: {{ useMaxWidth: true }},
                themeVariables: {{ primaryColor: '#ff0000' }}
            }});
            
            const containerId = 'mermaid-container-{hash(code) % 10000}';
            const contentId = 'mermaid-content-{hash(code) % 10000}';
            const container = document.getElementById(containerId);
            const content = document.getElementById(contentId);
            const zoomInBtn = document.getElementById('zoom-in-{hash(code) % 10000}');
            const zoomOutBtn = document.getElementById('zoom-out-{hash(code) % 10000}');
            const zoomResetBtn = document.getElementById('zoom-reset-{hash(code) % 10000}');
            const zoomLevelSpan = document.getElementById('zoom-level-{hash(code) % 10000}');
            
            let currentZoom = 1.0;
            const minZoom = 1.0;
            const maxZoom = 5.0;
            const zoomStep = 0.2;
            
            function updateZoom(newZoom) {{
                currentZoom = Math.max(minZoom, Math.min(maxZoom, newZoom));
                content.style.transform = `scale(${{currentZoom}})`;
                zoomLevelSpan.textContent = Math.round(currentZoom * 100) + '%';
                
                // Update button states
                zoomInBtn.disabled = currentZoom >= maxZoom;
                zoomOutBtn.disabled = currentZoom <= minZoom;
            }}
            
            // Zoom event listeners
            zoomInBtn.addEventListener('click', () => updateZoom(currentZoom + zoomStep));
            zoomOutBtn.addEventListener('click', () => updateZoom(currentZoom - zoomStep));
            zoomResetBtn.addEventListener('click', () => updateZoom(1.0));
            
            // Mouse wheel zoom
            container.addEventListener('wheel', (e) => {{
                if (e.ctrlKey || e.metaKey) {{
                    e.preventDefault();
                    const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
                    updateZoom(currentZoom + delta);
                }}
            }});
            
            // Prevent default zoom behavior
            container.addEventListener('keydown', (e) => {{
                if ((e.ctrlKey || e.metaKey) && (e.key === '+' || e.key === '-' || e.key === '0')) {{
                    e.preventDefault();
                    if (e.key === '+') updateZoom(currentZoom + zoomStep);
                    else if (e.key === '-') updateZoom(currentZoom - zoomStep);
                    else if (e.key === '0') updateZoom(1.0);
                }}
            }});
            
            // Make container focusable for keyboard events
            container.setAttribute('tabindex', '0');
            
            // Render mermaid and adjust height
            mermaid.run().then(() => {{
                setTimeout(() => {{
                    const svg = container.querySelector('svg');
                    if (svg) {{
                        const rect = svg.getBoundingClientRect();
                        const newHeight = Math.max(rect.height + 60, {height});
                        container.style.minHeight = newHeight + 'px';
                        
                        // Initialize zoom level display
                        updateZoom(1.0);
                        
                        // Trigger a resize event for Streamlit
                        window.parent.postMessage({{
                            type: 'streamlit:componentReady',
                            height: newHeight + 60 // Extra space for controls
                        }}, '*');
                    }}
                }}, 500);
            }});
        </script>

        <style>
            .mermaid-wrapper {{
                position: relative;
                width: 100%;
                height: {height + 60}px;
                border: 1px solid #e1e5e9;
                border-radius: 6px;
                background: #fafafa;
            }}
            
            .zoom-controls {{
                position: absolute;
                top: 10px;
                right: 10px;
                display: flex;
                gap: 5px;
                align-items: center;
                background: rgba(255, 255, 255, 0.9);
                padding: 5px 8px;
                border-radius: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                z-index: 10;
                backdrop-filter: blur(4px);
            }}
            
            .zoom-btn {{
                background: #fff;
                border: 1px solid #ddd;
                border-radius: 4px;
                width: 28px;
                height: 28px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s ease;
                color: #555;
            }}
            
            .zoom-btn:hover:not(:disabled) {{
                background: #f0f0f0;
                border-color: #bbb;
                transform: translateY(-1px);
            }}
            
            .zoom-btn:active {{
                transform: translateY(0);
            }}
            
            .zoom-btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            
            .zoom-level {{
                font-size: 12px;
                color: #666;
                font-weight: 500;
                min-width: 35px;
                text-align: center;
                margin-left: 5px;
            }}
            
            .mermaid-container {{
                height: 100%;
                padding: 20px;
                overflow: auto;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                position: relative;
                cursor: grab;
            }}
            
            .mermaid-container:active {{
                cursor: grabbing;
            }}
            
            .mermaid-content {{
                transform-origin: top center;
                transition: transform 0.3s ease;
                min-width: 100%;
                display: flex;
                justify-content: center;
            }}
            
            /* Scrollbar styling */
            .mermaid-container::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            .mermaid-container::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 4px;
            }}
            
            .mermaid-container::-webkit-scrollbar-thumb {{
                background: #c1c1c1;
                border-radius: 4px;
            }}
            
            .mermaid-container::-webkit-scrollbar-thumb:hover {{
                background: #a8a8a8;
            }}
            
            /* Responsive design */
            @media (max-width: 640px) {{
                .zoom-controls {{
                    top: 5px;
                    right: 5px;
                    padding: 3px 6px;
                }}
                
                .zoom-btn {{
                    width: 24px;
                    height: 24px;
                }}
                
                .zoom-level {{
                    font-size: 11px;
                }}
            }}
        </style>
        """,
        height=height + 80,  # Extra space for zoom controls
    )