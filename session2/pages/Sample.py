# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Streamlit application for generating messages with Anthropic Claude on Amazon Bedrock.
"""
import boto3
import json
import logging
import streamlit as st
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens):
    """
    Generate a message using Anthropic Claude on Amazon Bedrock.
    
    Args:
        bedrock_runtime: Bedrock runtime client
        model_id: ID of the model to use
        system_prompt: System prompt to use
        messages: List of messages in the conversation
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Response from the model
    """
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages
        }  
    )  

    try:
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())
        return response_body
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        st.error(f"A client error occurred: {message}")
        return None

# Set up the Streamlit page
st.set_page_config(page_title="Claude Message Generator", page_icon="🤖")
st.title("Anthropic Claude Message Generator")
st.write("Generate responses using Anthropic Claude on Amazon Bedrock")

# AWS region selector
region = st.sidebar.selectbox(
    "AWS Region",
    ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1"]
)

# Model selection
model_options = {
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 2.1": "anthropic.claude-v2:1",
    "Claude 4": "us.anthropic.claude-opus-4-20250514-v1:0"
}
selected_model_name = st.sidebar.selectbox("Select Claude Model", list(model_options.keys()))
model_id = model_options[selected_model_name]

# System prompt and max tokens
system_prompt = st.sidebar.text_area("System Prompt", value="Please respond only with emoji.", height=100)
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)

# Message input
user_message = st.text_area("Your message", value="Hello World", height=100)

# Prefill assistant response option
include_prefill = st.checkbox("Include prefilled assistant response")
prefill_content = st.text_input("Prefilled assistant content", value="<emoji>", disabled=not include_prefill)

# Generate button
if st.button("Generate Response"):
    try:
        with st.spinner("Generating response..."):
            # Create Bedrock runtime client
            bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=region)
            
            # Create messages
            messages = [{"role": "user", "content": user_message}]
            if include_prefill:
                messages.append({"role": "assistant", "content": prefill_content})
                
            # Generate message
            response = generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens)
            
            if response:
                # Display the response
                st.subheader("Response")
                st.write(response["content"][0]["text"])
                
                # Show raw JSON response
                with st.expander("View raw JSON response"):
                    st.json(response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add information about the application
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This application uses Amazon Bedrock to generate responses with Anthropic Claude models. "
    "You need proper AWS credentials configured to use this application."
)
st.sidebar.markdown("---")
st.sidebar.caption("© Amazon.com, Inc. or its affiliates")
