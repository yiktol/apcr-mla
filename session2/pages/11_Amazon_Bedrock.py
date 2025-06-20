
import streamlit as st
import logging
import sys
import boto3
from botocore.exceptions import ClientError
import os
from io import BytesIO
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Page configuration with improved styling
st.set_page_config(
    page_title="Amazon Bedrock",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for modern appearance
st.markdown("""
    <style>
    .stApp {
        # max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #2563EB;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .output-container {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .response-block {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
        margin-top: 1rem;
    }
    .token-metrics {
        display: flex;
        justify-content: space-between;
        background-color: #F0F4F8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
    .metric-item {
        text-align: center;
    }
    .metric-value {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #4B5563;
    }
    </style>
""", unsafe_allow_html=True)

# ------- API FUNCTIONS -------

def text_conversation(bedrock_client, model_id, system_prompts, messages, **params):
    """Sends messages to a model."""
    logger.info(f"Generating message with model {model_id}")
    
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=params,
            additionalModelRequestFields={}
        )
        
        # Log token usage
        token_usage = response['usage']
        logger.info(f"Input tokens: {token_usage['inputTokens']}")
        logger.info(f"Output tokens: {token_usage['outputTokens']}")
        logger.info(f"Total tokens: {token_usage['totalTokens']}")
        logger.info(f"Stop reason: {response['stopReason']}")
        
        return response
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None

def image_conversation(bedrock_client, model_id, input_text, image_bytes, image_format='jpeg'):
    """Sends a message with image to a model."""
    logger.info(f"Generating message with model {model_id}")
    
    # Message to send
    message = {
        "role": "user",
        "content": [
            {"text": input_text},
            {
                "image": {
                    "format": image_format,
                    "source": {"bytes": image_bytes}
                }
            }
        ]
    }
    
    messages = [message]
    
    # Send the message
    try:
        response = bedrock_client.converse(modelId=model_id, messages=messages)
        return response
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None

def stream_conversation(bedrock_client, model_id, messages, system_prompts, inference_config):
    """Sends messages to a model and streams the response."""
    logger.info(f"Streaming messages with model {model_id}")
    
    try:
        response = bedrock_client.converse_stream(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields={}
        )
        
        stream = response.get('stream')
        if stream:
            placeholder = st.empty()
            full_response = ''
            token_info = {'input': 0, 'output': 0, 'total': 0}
            latency_ms = 0
            
            for event in stream:
                if 'messageStart' in event:
                    role = event['messageStart']['role']
                    with placeholder.container():
                        st.markdown(f"**{role.title()}**")
                
                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']
                    part = chunk['delta']['text']
                    full_response += part
                    with placeholder.container():
                        st.markdown(f"**Response:**\n{full_response}")
                
                if 'messageStop' in event:
                    stop_reason = event['messageStop']['stopReason']
                    with placeholder.container():
                        st.markdown(f"**Response:**\n{full_response}")
                        st.caption(f"Stop reason: {stop_reason}")
                
                if 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        usage = metadata['usage']
                        token_info = {
                            'input': usage['inputTokens'],
                            'output': usage['outputTokens'],
                            'total': usage['totalTokens']
                        }
                    
                    if 'metrics' in event.get('metadata', {}):
                        latency_ms = metadata['metrics']['latencyMs']
            
            # Display token usage after streaming is complete
            st.markdown("### Response Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Tokens", token_info['input'])
            col2.metric("Output Tokens", token_info['output'])
            col3.metric("Total Tokens", token_info['total'])
            st.caption(f"Latency: {latency_ms}ms")
        
        return True
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return False

# ------- UI COMPONENTS -------

def parameter_sidebar():
    """Sidebar with model selection and parameter tuning."""
    with st.sidebar:
        st.markdown("<div class='sub-header'>Model Selection</div>", unsafe_allow_html=True)
        
        MODEL_CATEGORIES = {
            "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
            "Anthropic": ["anthropic.claude-v2:1", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"],
            "Cohere": ["cohere.command-text-v14:0", "cohere.command-r-plus-v1:0", "cohere.command-r-v1:0"],
            "Meta": ["meta.llama3-70b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0"],
            "Mistral": ["mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", 
                       "mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0"],
            "AI21": ["ai21.jamba-1-5-large-v1:0", "ai21.jamba-1-5-mini-v1:0"]
        }
        
        # Create selectbox for provider first
        provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()))
        
        # Then create selectbox for models from that provider
        model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider])
        
        st.markdown("<div class='sub-header'>Parameter Tuning</div>", unsafe_allow_html=True)
        
        with st.expander("Model Parameters", expanded=True):
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, 
                                help="Higher values make output more random, lower values more deterministic")
            
            top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1,
                             help="Controls diversity via nucleus sampling")
            
            max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=1024, step=50,
                                       help="Maximum number of tokens in the response")
            
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
        
        with st.expander("About", expanded=False):
            st.markdown("""
            This app demonstrates Amazon Bedrock's Converse API with different foundation models.
            You can interact with text-only conversations, image analysis, or streaming responses.
            
            For more information, visit the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/).
            """)
        
    return model_id, params

def text_interface(model_id, params):
    """Interface for text-only conversation."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        system_prompt = st.text_area(
            "System Prompt", 
        value="""You are a helpful AI assistant focused on providing accurate information.

IMPORTANT GUIDELINES FOR FACTUAL RESPONSES:
1. Prioritize accuracy over comprehensiveness. It's better to provide less information that is accurate than more information that might be incorrect.

2. Express appropriate uncertainty:
   - For well-established facts, speak confidently
   - For interpretations or less certain information, use qualifiers like "According to [source]," "Many experts believe," "It's generally understood that"
   - Avoid definitive statements on topics that have significant debate or uncertainty

3. When you don't know or are unsure:
   - Clearly state "I don't have specific information about that" or "I'm not certain about"
   - Do NOT fabricate information, sources, statistics, or quotes
   - Do NOT make up specific dates, numbers, or facts if you're unsure

4. Be transparent about your limitations:
   - Be clear about the boundaries of your knowledge
   - Acknowledge when a question is outside your training data or expertise
   - Consider recommending that the user verify important information from authoritative sources

5. If asked about events after your training data cutoff, clearly state that you don't have information beyond your training cutoff.

Remember: It is much better to acknowledge uncertainty than to provide potentially incorrect information.""",
        height=250,
        key="factuality_system_prompt"
    )
    with col2:
        st.caption("This defines the AI assistant's behavior")

    user_prompt = st.text_area(
        "Your Question", 
        value="What were the main outcomes of the 2023 UN Climate Change Conference?", 
        height=120,
        placeholder="Enter your question here..."
    )
    
    submit = st.button("Generate Response", type="primary", key="text_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not user_prompt.strip():
            st.warning("Please enter a question first.")
            return
            
        with st.status("Generating response...") as status:
            # Setup the system prompts and messages
            system_prompts = [{"text": system_prompt}]
            message = {
                "role": "user",
                "content": [{"text": user_prompt}]
            }
            messages = [message]
            
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Send request to the model
                response = text_conversation(bedrock_client, model_id, system_prompts, messages, **params)
                
                if response:
                    status.update(label="Response received!", state="complete")
                    
                    st.markdown("<div class='response-block'>", unsafe_allow_html=True)
                    
                    # Display the model's response
                    output_message = response['output']['message']
                    
                    st.markdown(f"**{output_message['role'].title()}**")
                    for content in output_message['content']:
                        st.markdown(content['text'])
                    
                    # Show token usage directly (no nested expander)
                    st.markdown("### Response Details")
                    token_usage = response['usage']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input Tokens", token_usage['inputTokens'])
                    col2.metric("Output Tokens", token_usage['outputTokens'])
                    col3.metric("Total Tokens", token_usage['totalTokens'])
                    st.caption(f"Stop reason: {response['stopReason']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")

def image_interface(model_id):
    """Interface for image analysis."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Image input options
    image_source = st.radio(
        "Select Image Source",
        options=["Upload Image", "Sample Image"],
        horizontal=True,
        key="image_source"
    )
    
    image_bytes = None
    image_format = None
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image_format = uploaded_file.name.split(".")[-1].lower()
            
            # Display the uploaded image
            image = Image.open(BytesIO(image_bytes))
            st.image(image, caption="Uploaded Image", use_container_width =True)
    else:
        # Sample image
        sample_image_path = "images/sg_skyline.jpeg"
        if os.path.exists(sample_image_path):
            with open(sample_image_path, "rb") as f:
                image_bytes = f.read()
            image_format = "jpeg"
            st.image(sample_image_path, caption="Sample Image", use_container_width =True)
        else:
            st.error("Sample image not found. Please use the upload option.")
    
    # Query input
    user_prompt = st.text_area(
        "Ask about the image", 
        value="What's in this image?", 
        height=80,
        placeholder="Enter your question about the image..."
    )
    
    submit = st.button("Analyze Image", type="primary", key="image_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not image_bytes:
            st.warning("Please provide an image first.")
            return
        
        if not user_prompt.strip():
            st.warning("Please enter a question about the image.")
            return
        
        with st.status("Analyzing image...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Process the image and get response
                response = image_conversation(bedrock_client, model_id, user_prompt, image_bytes, image_format)
                
                if response:
                    status.update(label="Analysis complete!", state="complete")
                    
                    st.markdown("<div class='response-block'>", unsafe_allow_html=True)
                    
                    # Display the model's response
                    output_message = response['output']['message']
                    
                    st.markdown(f"**{output_message['role'].title()}**")
                    for content in output_message['content']:
                        st.markdown(content['text'])
                    
                    # Show token usage directly (no nested expander)
                    st.markdown("### Response Details")
                    token_usage = response['usage']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input Tokens", token_usage['inputTokens'])
                    col2.metric("Output Tokens", token_usage['outputTokens'])
                    col3.metric("Total Tokens", token_usage['totalTokens'])
                    st.caption(f"Stop reason: {response['stopReason']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")

def streaming_interface(model_id, params):
    """Interface for streaming responses."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        system_prompt = st.text_area(
            "System Prompt", 
            value="You are an assistant that provides detailed and thoughtful responses.", 
            height=100,
            key="streaming_system"
        )
    with col2:
        st.caption("This defines the AI assistant's behavior")

    user_prompt = st.text_area(
        "Your Question", 
        value="Explain the concept of neural networks and how they work.", 
        height=120,
        placeholder="Enter your question here...",
        key="streaming_input"
    )
    
    submit = st.button("Generate Streaming Response", type="primary", key="streaming_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not user_prompt.strip():
            st.warning("Please enter a question first.")
            return
        
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        st.markdown("**Streaming Response:**")
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params)
            
            if success:
                st.success("Response streaming completed successfully.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------- MAIN APP -------

def main():
    # Header
    st.markdown("<h1 class='main-header'>Amazon Bedrock</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard demonstrates the Converse API capabilities in Amazon Bedrock.
    The API provides a unified interface to interact with various foundation models through text
    conversations, image analysis, and streaming responses.
    """)
    
    # Get model and parameters from sidebar
    model_id, params = parameter_sidebar()
    
    # Create tabs for different interaction modes
    tabs = st.tabs([
        "💬 Text Conversation", 
        "🖼️ Image Analysis", 
        "⚡ Streaming Response"
    ])
    
    # Populate each tab
    with tabs[0]:
        text_interface(model_id, params)
    
    with tabs[1]:
        image_interface(model_id)
    
    with tabs[2]:
        streaming_interface(model_id, params)

if __name__ == "__main__":
    main()
