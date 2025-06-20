
import streamlit as st
import datetime

# Configure the page
st.set_page_config(
    page_title="Site Under Construction",
    page_icon="🚧",
    layout="wide",
)

# AWS Color Scheme
AWS_ORANGE = "#FF9900"
AWS_BLUE = "#232F3E"
AWS_LIGHT_BLUE = "#1A365D"
AWS_GRAY = "#D5DBDB"

# Custom CSS for AWS styling and responsive design
st.markdown("""
<style>
    /* AWS Color Theme and Modern Styling */
    .main {
        background-color: white;
        color: #232F3E;
    }
    .stApp {
        max-width: 100%;
    }
    .container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        text-align: center;
        min-height: 80vh;
    }
    h1 {
        color: #232F3E;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    .subtitle {
        color: #545B64;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .progress-container {
        width: 100%;
        max-width: 600px;
        margin: 2rem 0;
    }
    .progress-bar {
        height: 12px;
        background-color: #FF9900;
        width: 60%;
        border-radius: 6px;
    }
    .aws-footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 1rem;
        background-color: #232F3E;
        color: white;
        font-size: 0.9rem;
    }
    .icon {
        font-size: 5rem;
        margin-bottom: 2rem;
        color: #FF9900;
    }
    .card {
        background-color: #F8F8F8;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        max-width: 600px;
        width: 100%;
    }
    .button {
        background-color: #FF9900;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        margin-top: 1rem;
        transition: all 0.2s ease-in-out;
    }
    .button:hover {
        background-color: #EC7211;
        transform: translateY(-2px);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        h1 {
            font-size: 2.2rem;
        }
        .subtitle {
            font-size: 1.2rem;
        }
        .icon {
            font-size: 4rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Content container
# st.markdown('<div class="container">', unsafe_allow_html=True)

# Construction icon and title
st.markdown('<div class="icon">🚧</div>', unsafe_allow_html=True)
st.markdown('<h1>Website Under Construction</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">We\'re working hard to bring you an amazing experience.</div>', unsafe_allow_html=True)

# Progress bar
st.markdown("""
<div class="progress-container">
    <div class="progress-bar"></div>
</div>
""", unsafe_allow_html=True)

# Construction message card
st.markdown("""
<div class="card">
    <h3>What to expect?</h3>
    <p>We're building something special for you and appreciate your patience.</p>
</div>
""", unsafe_allow_html=True)

# Countdown timer
# launch_date = datetime.datetime(2025, 1, 1)
# remaining = launch_date - datetime.datetime.now()

# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Days", remaining.days)
# with col2:
#     st.metric("Hours", remaining.seconds // 3600)
# with col3:
#     st.metric("Minutes", (remaining.seconds // 60) % 60)

# Subscribe button (non-functional in this example)
# st.markdown("""
# <button class="button">Notify Me When Live</button>
# """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="aws-footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</div>
""", unsafe_allow_html=True)
