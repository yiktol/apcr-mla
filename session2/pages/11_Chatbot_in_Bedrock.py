import os
import streamlit as st
import boto3
import uuid
import tempfile
from typing import List, Dict
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain, RetrievalQA
from langchain_aws import ChatBedrockConverse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
import utils.common as common
import utils.authenticate as authenticate

# Configure the page
st.set_page_config(
    page_title="AI Chatbot with RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def model_selection_panel():
    """Model selection and parameters panel"""
    st.markdown("<h4>Model Selection</h4>", unsafe_allow_html=True)
    
    MODEL_CATEGORIES = {
        "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
        "Anthropic": ["anthropic.claude-v2:1", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"],
        "Cohere": ["cohere.command-r-v1:0","cohere.command-r-plus-v1:0"],
        "Meta": ["meta.llama3-8b-instruct-v1:0","meta.llama3-70b-instruct-v1:0"],
        "Mistral": ["mistral.mistral-small-2402-v1:0", "mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", 
                   "mistral.mistral-7b-instruct-v0:2"],
        "AI21":["ai21.jamba-1-5-mini-v1:0","ai21.jamba-1-5-large-v1:0"],
        "DeepSeek": ["us.deepseek.r1-v1:0"]
    }
    
    EMBEDDING_MODELS = [
        "amazon.titan-embed-text-v1", 
        "cohere.embed-english-v3", 
        "cohere.embed-multilingual-v3"
    ]

    # Create selectbox for provider first
    provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()), key="control_provider")
    
    # Then create selectbox for models from that provider
    model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider], key="control_model")
    
    # Embedding model selection for RAG
    # Use a different key for the widget - "control_embedding_model" instead of "embedding_model"
    embedding_model = st.selectbox("Select Embedding Model", options=EMBEDDING_MODELS, key="control_embedding_model")
    
    st.markdown("<h4>API Method</h4>", unsafe_allow_html=True)
    api_method = st.radio(
        "Select API Method", 
        ["Streaming", "Synchronous"], 
        index=0,
        help="Streaming provides real-time responses, while Synchronous waits for complete response",
        key="control_api_method"
    )
    st.markdown("<h4>Parameter Tuning</h4>", unsafe_allow_html=True)

    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.05,
        key="control_temperature",
        help="Higher values make output more random, lower values more deterministic"
    )
        
    top_p = st.slider(
        "Top P", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.9, 
        step=0.05,
        key="control_top_p",
        help="Controls diversity via nucleus sampling"
    )
       
    max_tokens = st.number_input(
        "Max Tokens", 
        min_value=50, 
        max_value=4096, 
        value=1024, 
        step=50,
        key="control_max_tokens",
        help="Maximum number of tokens in the response"
    )
                 
    # Toggle switch for enabling/disabling guardrails
    st.markdown("<h4>Bedrock Guardrails</h4>", unsafe_allow_html=True)
    enable_guardrails = st.toggle("Enable Guardrails", value=True, key="enable_guardrails")
    
    guardrail_id = ""
    guardrail_version = ""
    trace_enabled = False
    stream_mode = "sync"
    
    if enable_guardrails:
        guardrail_id = st.text_input("Guardrail ID", value="wibfn4fa6ifg", key="guardrail_id")
        guardrail_version = st.text_input("Guardrail Version", value="DRAFT", key="guardrail_version")
        
        trace_enabled = st.checkbox("Enable Trace", value=True, key="trace_enabled")
        stream_mode = st.radio(
            "Stream Processing Mode",
            ["sync", "async"],
            index=0,
            key="stream_mode"
        )
    
    guardrail_config = None
    if enable_guardrails:
        guardrail_config = {
            "guardrailIdentifier": guardrail_id,
            "guardrailVersion": guardrail_version,
            "trace": "enabled" if trace_enabled else "disabled",
            "streamProcessingMode": stream_mode
        }
        
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    
    return provider, model_id, embedding_model, params, api_method, guardrail_config

def init_styles():
    """Apply custom styling to the app."""
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user {
            background-color: #e3f2fd;
        }
        .chat-message.assistant {
            background-color: #f0f4c3;
        }
        .chat-header {
            position: sticky;
            background: linear-gradient(to right, #4776E6, #8E54E9);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
            text-align: center;
            top: 0;
        }
        .sidebar .sidebar-content {
            background-color: #f0f4f8;
        }
        /* Button styling */
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            font-weight: bold;
            background-color: #4CAF50;
            color: white;
        }
        /* Sidebar title styling */
        .sidebar-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 1.5rem;
            color: #333;
            text-align: center;
        }
        .memory-status, .rag-status {
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            margin: 10px 0;
        }
        .memory-enabled, .rag-enabled {
            background-color: #c8e6c9;
            color: #2e7d32;
        }
        .memory-disabled, .rag-disabled {
            background-color: #ffcdd2;
            color: #c62828;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            font-size: 0.8em;
            color: #666;
        }
        /* Info boxes */
        .info-box {
            background-color: #f0f7ff;
            border-left: 5px solid #0066cc;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        /* Controls container */
        .controls-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .document-box {
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .upload-section {
            background-color: #f1f8e9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables if they don't exist."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        
    if 'memory_enabled' not in st.session_state:
        st.session_state.memory_enabled = True
        
    if 'rag_enabled' not in st.session_state:
        st.session_state.rag_enabled = False

    if 'provider' not in st.session_state:
        st.session_state.provider = "Amazon"
    
    if 'model_id' not in st.session_state:
        st.session_state.model_id = "amazon.nova-micro-v1:0"
        
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "amazon.titan-embed-text-v1"
        
    if "guardrail_config" not in st.session_state:
        st.session_state.guardrail_config = None
    
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1024
        }
        
    if 'api_method' not in st.session_state:
        st.session_state.api_method = "Streaming"
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "text": "Hello! How can I assist you today?"}]
    
    if 'memory' not in st.session_state:
        st.session_state.memory = get_memory()
        
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
        
    if 'documents' not in st.session_state:
        st.session_state.documents = []
        
    if 'retrieval_qa' not in st.session_state:
        st.session_state.retrieval_qa = None

@st.cache_resource
def init_bedrock_client():
    """Initialize and cache the AWS Bedrock client."""
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
    )

def get_llm():
    """Get the language model with specified parameters."""
    model_kwargs = st.session_state.model_params
    
    llm = ChatBedrockConverse(
        client=init_bedrock_client(),
        model=st.session_state.model_id,
        temperature=model_kwargs["temperature"],
        top_p=model_kwargs["top_p"],
        max_tokens=model_kwargs["max_tokens"],
        guardrail_config=st.session_state.guardrail_config
    )
    
    return llm

def get_embeddings():
    """Get the embeddings model."""
    return BedrockEmbeddings(
        client=init_bedrock_client(),
        model_id=st.session_state.embedding_model
    )

def get_memory():
    """Create a conversation memory with the language model."""
    llm = get_llm()
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024)
    return memory

def process_document(file):
    """Process a document file and return a list of Document objects."""
    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Choose the appropriate loader based on file extension
    file_extension = os.path.splitext(file.name)[1].lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(tmp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(tmp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(tmp_file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(tmp_file_path)
        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(tmp_file_path)
        else:
            os.unlink(tmp_file_path)
            return None, f"Unsupported file type: {file_extension}"
        
        # Load documents
        documents = loader.load()
        os.unlink(tmp_file_path)
        return documents, None
    
    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return None, f"Error processing document: {str(e)}"

def initialize_vector_store(documents: List[Document]):
    """Initialize the vector store with documents."""
    if not documents:
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Create a vector store
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def get_retrieval_qa():
    """Create a retrieval QA chain."""
    if not st.session_state.vector_store:
        return None
    
    llm = get_llm()
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return retrieval_qa

def get_chat_response(input_text, use_memory=True, use_rag=False):
    """Generate a chat response using the LLM, optionally with RAG."""
    llm = get_llm()
    
    if use_rag and st.session_state.retrieval_qa:
        # Use RAG for answering
        try:
            response = st.session_state.retrieval_qa.invoke(input_text)
            chat_response = response["result"]
            
            # If memory is enabled, save the context
            if use_memory:
                st.session_state.memory.save_context({"input": input_text}, {"output": chat_response})
                
            return chat_response
        except Exception as e:
            return f"Error using RAG: {str(e)}"
    
    elif use_memory and st.session_state.memory_enabled:
        # Use conversation memory without RAG
        conversation = ConversationChain(
            llm=llm,
            memory=st.session_state.memory,
            verbose=False
        )
        chat_response = conversation.predict(input=input_text)
    else:
        # Use a simple prompt without memory or RAG
        chat_response = llm.invoke(f"User: {input_text}\nAI: ")
        if hasattr(chat_response, 'content'):
            chat_response = chat_response.content
    
    return chat_response

def reset_session():
    """Reset the chat session state."""
    st.session_state.memory = get_memory()
    st.session_state.chat_history = [{"role": "assistant", "text": "Session reset. How may I assist you?"}]
    st.session_state.memory_enabled = True
    st.session_state.session_id = str(uuid.uuid4())[:8]

def render_sidebar():
    """Render the sidebar content."""
    common.render_sidebar()

    clear_chat_btn = st.sidebar.button("🧹 Clear Chat History", key="clear_chat")
    
    if clear_chat_btn:
        st.session_state.chat_history = [{"role": "assistant", "text": "Chat history cleared. How can I help you?"}]
        st.rerun()
    st.markdown("---")
    
    with st.expander("About this Application", expanded=False):
        st.markdown("""
        **AI Chatbot with Amazon Bedrock and RAG**
        
        This application demonstrates multiple foundation models available through Amazon Bedrock integrated with LangChain.
        
        Features:
        - Multiple foundation models support
        - Conversation memory
        - Retrieval Augmented Generation (RAG)
        - Document upload and processing
        
        Built using:
        - Amazon Bedrock
        - LangChain
        - FAISS Vector Store
        - Streamlit
        """)
    
    st.markdown("<div class='footer'>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

def render_chat_messages():
    """Render the chat message history."""
    for message in st.session_state.chat_history:
        message_role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(message["role"], avatar="👤" if message_role == "user" else "🤖"):
            st.markdown(message["text"])

def render_chat_input():
    """Render the chat input area and process user messages."""
    with st._bottom:
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            input_text = st.chat_input("Type your message here...")

    if input_text:
        # Add user message to chat
        with st.chat_message("user", avatar="👤"):
            st.markdown(input_text)
        
        st.session_state.chat_history.append({"role": "user", "text": input_text})
        
        # Get and display AI response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            if st.session_state.api_method == "Streaming":
                # Streaming approach
                full_response = ""
                llm = get_llm()
                
                try:
                    # Determine whether to use RAG
                    use_rag = st.session_state.rag_enabled and st.session_state.vector_store is not None
                    
                    if use_rag:
                        # For RAG, we need to retrieve context first
                        retriever = st.session_state.vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 3}
                        )
                        relevant_docs = retriever.get_relevant_documents(input_text)
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        # Create prompt with RAG context
                        if st.session_state.memory_enabled:
                            # Get conversation history from memory
                            memory_variables = st.session_state.memory.load_memory_variables({})
                            memory_context = memory_variables.get("history", "")
                            
                            # Create prompt with memory context AND document context
                            if memory_context:
                                prompt = f"Previous conversation:\n{memory_context}\n\nHere is relevant information to answer the question:\n{context}\n\nUser: {input_text}\nAssistant: "
                            else:
                                prompt = f"Here is relevant information to answer the question:\n{context}\n\nUser: {input_text}\nAssistant: "
                        else:
                            prompt = f"Here is relevant information to answer the question:\n{context}\n\nUser: {input_text}\nAssistant: "
                    else:
                        # Handle non-RAG case
                        if st.session_state.memory_enabled:
                            # Get conversation history from memory
                            memory_variables = st.session_state.memory.load_memory_variables({})
                            memory_context = memory_variables.get("history", "")
                            
                            # Create prompt with memory context
                            if memory_context:
                                prompt = f"Previous conversation:\n{memory_context}\n\nUser: {input_text}\nAssistant: "
                            else:
                                prompt = f"User: {input_text}\nAssistant: "
                        else:
                            prompt = f"User: {input_text}\nAssistant: "
                    
                    # Make the streaming call
                    for chunk in llm.stream(prompt):
                        try:
                            # The chunk is an AIMessageChunk with a content attribute that is a list
                            if hasattr(chunk, 'content'):
                                content_list = chunk.content
                                # Process each item in the content list
                                if isinstance(content_list, list):
                                    for item in content_list:
                                        # Item might be a dict with 'text' key
                                        if isinstance(item, dict):
                                            # Check for 'text' key
                                            if 'text' in item:
                                                full_response += item['text']
                                            # Some items might have 'type' and 'text'
                                            elif 'type' in item and item['type'] == 'text' and 'text' in item:
                                                full_response += item['text']
                            
                        except Exception as e:
                            st.warning(f"Chunk processing issue: {str(e)}")
                            continue
                        
                        # Update the message placeholder with current accumulated response
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Final update without cursor
                    message_placeholder.markdown(full_response)
                    
                    # Update memory after streaming completes
                    if st.session_state.memory_enabled:
                        st.session_state.memory.save_context({"input": input_text}, {"output": full_response})
                
                except Exception as e:
                    st.error(f"Streaming error: {e}")
                    full_response = f"I encountered an error while processing your request. Please try again or switch to synchronous mode. Error: {str(e)}"
                    message_placeholder.markdown(full_response)
            else:
                # Synchronous approach
                with st.spinner("Thinking..."):
                    use_rag = st.session_state.rag_enabled and st.session_state.vector_store is not None
                    full_response = get_chat_response(input_text, 
                                                    use_memory=st.session_state.memory_enabled, 
                                                    use_rag=use_rag)
                    message_placeholder.markdown(full_response)
        
        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "text": full_response})

def render_chat_area():
    """Render the main chat area."""
    # Chat header
    memory_status = "with Memory" if st.session_state.memory_enabled else "without Memory"
    rag_status = "with RAG" if st.session_state.rag_enabled else "without RAG"
    model_info = st.session_state.model_id.split('.')[0].capitalize()
    model_name = st.session_state.model_id.split('.')[1].replace('-', ' ').title()
    if st.session_state.model_id.split('.')[0] == 'us':
        model_name += f" {st.session_state.model_id.split('.')[2].replace('-', ' ').title()}"
    
    st.markdown(f"<div class='chat-header'><h1>AI Assistant ({memory_status}, {rag_status})</h1><p>{model_info} {model_name}</p></div>", unsafe_allow_html=True)

    # Display chat messages
    render_chat_messages()
    
    # Chat input
    render_chat_input()

def render_rag_panel():
    """Render the RAG panel for document upload and settings."""
    st.markdown("#### RAG Settings")
    
    # RAG toggle
    rag_enabled = st.toggle("Enable RAG", value=st.session_state.rag_enabled, key="rag_toggle")
    st.session_state.rag_enabled = rag_enabled
    
    # Display RAG status
    if rag_enabled:
        st.markdown("<div class='rag-status rag-enabled'>RAG: ENABLED</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Bot will use your uploaded documents to provide more informed answers.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='rag-status rag-disabled'>RAG: DISABLED</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Bot will respond based on its general knowledge without using your documents.</div>", unsafe_allow_html=True)
    
    if rag_enabled:
        # Document upload section
        st.markdown("#### Document Upload")
        
        with st.container(border=True, key="upload_container"):
            uploaded_files = st.file_uploader(
                "Upload documents for RAG", 
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "csv", "md"],
                help="Supported file types: PDF, DOCX, TXT, CSV, MD"
            )
            
            # Process uploaded files
            if uploaded_files:
                process_btn = st.button("Process Documents", type="primary")
                
                if process_btn:
                    with st.spinner("Processing documents..."):
                        all_documents = []
                        
                        for file in uploaded_files:
                            docs, error = process_document(file)
                            if error:
                                st.error(f"Error processing {file.name}: {error}")
                            else:
                                all_documents.extend(docs)
                                st.success(f"Successfully processed {file.name}")
                        
                        if all_documents:
                            st.session_state.documents = all_documents
                            
                            # Create vector store
                            with st.spinner("Creating vector store..."):
                                vector_store = initialize_vector_store(all_documents)
                                if vector_store:
                                    st.session_state.vector_store = vector_store
                                    st.session_state.retrieval_qa = get_retrieval_qa()
                                    st.success(f"Vector store created with {len(all_documents)} documents")
                                    st.session_state.rag_enabled = True
                                else:
                                    st.error("Failed to create vector store")
        
        # Display currently loaded documents
        if st.session_state.documents:
            st.markdown("#### Loaded Documents")
            for i, doc in enumerate(st.session_state.documents):
                source = doc.metadata.get('source', 'Unknown source')
                if isinstance(source, str) and len(source) > 30:
                    # Get just the filename if it's a path
                    source = os.path.basename(source)
                
                st.markdown(f"""
                <div class="document-box">
                    <b>Document {i+1}:</b> {source}<br>
                    <small>Length: {len(doc.page_content)} chars</small>
                </div>
                """, unsafe_allow_html=True)
            
            clear_docs = st.button("🗑️ Clear All Documents")
            if clear_docs:
                st.session_state.documents = []
                st.session_state.vector_store = None
                st.session_state.retrieval_qa = None
                st.success("All documents cleared")
                st.rerun()

def render_control_panel():
    """Render the control panel for memory controls and model settings."""
    with st.container(border=True):
        # Memory Settings Section
        st.markdown("#### Memory Settings")
        
        # Memory toggle
        memory_enabled = st.toggle("Enable Conversation Memory", value=st.session_state.memory_enabled)
        st.session_state.memory_enabled = memory_enabled
        
        # Display memory status
        if memory_enabled:
            st.markdown("<div class='memory-status memory-enabled'>Memory: ENABLED</div>", unsafe_allow_html=True)
            st.markdown("<div class='info-box'>Bot will remember your conversation and maintain context.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='memory-status memory-disabled'>Memory: DISABLED</div>", unsafe_allow_html=True)
            st.markdown("<div class='info-box'>Bot will respond to each message independently without context.</div>", unsafe_allow_html=True)
        
        # RAG Panel
        render_rag_panel()
        
        # Model Selection and Parameters
        provider, model_id, embedding_model, params, api_method, guardrail_config = model_selection_panel()
        
        # Update session state with model selection
        if (model_id != st.session_state.model_id or 
            embedding_model != st.session_state.embedding_model or
            params != st.session_state.model_params or 
            api_method != st.session_state.api_method or 
            guardrail_config != st.session_state.guardrail_config):
            
            st.session_state.provider = provider
            st.session_state.model_id = model_id
            # Here we're updating the session state variable, not the widget
            st.session_state.embedding_model = embedding_model
            st.session_state.model_params = params
            st.session_state.api_method = api_method
            st.session_state.guardrail_config = guardrail_config
            st.session_state.memory = get_memory()  # Refresh memory with new model
            
            # If we change the embedding model, we need to recreate the vector store if documents exist
            if embedding_model != st.session_state.embedding_model and st.session_state.documents:
                with st.spinner("Recreating vector store with new embedding model..."):
                    vector_store = initialize_vector_store(st.session_state.documents)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.retrieval_qa = get_retrieval_qa()
                        st.success("Vector store updated with new embedding model")
            
            st.rerun()

def main():
    """Main application function."""   
    # Initialize app components
    init_styles()
    init_session_state()
    
    # Sidebar (left column)
    with st.sidebar:
        render_sidebar()
    
    # Main layout - 70/30 split
    chat_col, controls_col = st.columns([0.7, 0.3])
    
    # Chat column (70%)
    with chat_col:
        render_chat_area()
    
    # Controls column (30%)
    with controls_col:
        render_control_panel()

# Main execution flow
if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()
