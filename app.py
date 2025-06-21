import os
import tempfile
import streamlit as st
import qdrant_client
from llama_index.core.schema import Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.readers.file.pymu_pdf import PyMuPDFReader

# ================================
# INITIALIZATION FUNCTIONS
# ================================

@st.cache_resource
def init_qdrant():
    """Initialize Qdrant Cloud client"""
    try:
        # Get credentials from Streamlit secrets or environment
        qdrant_url = st.secrets.get("QDRANT_URL") or os.getenv("QDRANT_URL")
        qdrant_api_key = st.secrets.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            st.error("❌ Qdrant Cloud credentials not found!")
            st.info("💡 Please set QDRANT_URL and QDRANT_API_KEY in Streamlit secrets")
            st.stop()
        
        client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Test connection
        client.get_collections()
        return client
        
    except Exception as e:
        st.error(f"❌ Cannot connect to Qdrant Cloud: {str(e)}")
        st.info("💡 Please check your Qdrant Cloud credentials")
        st.stop()

@st.cache_resource
def init_embedding():
    """Initialize embedding model"""
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        trust_remote_code=True
    )

@st.cache_resource
def init_mistral():
    """Initialize Mistral AI model"""
    api_key = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        st.error("❌ MISTRAL_API_KEY not found!")
        st.info("💡 Please set your Mistral API key in Streamlit secrets")
        st.stop()
    
    return MistralAI(
        model="mistral-large-latest",
        temperature=0.7,
        max_tokens=512,
        api_key=api_key
    )

# ================================
# DATABASE SETUP FUNCTIONS
# ================================

def setup_database_from_pdf(pdf_path, client, collection_name):
    """
    Set up fraud scenarios database from PDF file using Qdrant Cloud
    
    Args:
        pdf_path (str): Path to the fraud scenarios PDF file
        client: Qdrant client instance
        collection_name (str): Name of the collection to create
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        if not os.path.exists(pdf_path):
            return False, f"PDF file not found at {pdf_path}"
        
        st.info("📄 Loading PDF document...")
        
        # Load PDF document
        loader = PyMuPDFReader()
        documents = loader.load(file_path=pdf_path)
        
        if not documents:
            return False, "No content found in PDF"
        
        st.success(f"✅ Successfully loaded {len(documents)} pages from PDF")
        
        # Initialize embedding model
        st.info("🔄 Loading embedding model...")
        embed_model = init_embedding()
        st.success("✅ Embedding model loaded successfully!")
        
        # Set up vector store
        st.info("🔄 Setting up vector store...")
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name=collection_name
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index and store documents
        st.info("🔄 Creating index and storing documents...")
        with st.spinner("Processing documents..."):
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=False  # Disable progress bar for cleaner UI
            )
        
        return True, f"Successfully indexed {len(documents)} document pages!"
        
    except Exception as e:
        return False, f"Error setting up database: {str(e)}"

def setup_database_from_upload(uploaded_file, client, collection_name):
    """Setup database from uploaded PDF file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Setup database using temporary file
        success, message = setup_database_from_pdf(tmp_path, client, collection_name)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return success, message
        
    except Exception as e:
        return False, f"Error processing uploaded PDF: {str(e)}"

@st.cache_resource
def check_and_setup_database(_client, collection_name, pdf_path=None):
    """
    Check if database exists, create if not
    
    Args:
        _client: Qdrant client (with underscore for caching)
        collection_name (str): Name of the collection
        pdf_path (str, optional): Path to PDF file for setup
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Check if collection exists
        collections = _client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            if pdf_path and os.path.exists(pdf_path):
                # Database doesn't exist, create it from PDF
                st.warning("⚠️ Database not found. Setting up for the first time...")
                success, message = setup_database_from_pdf(pdf_path, _client, collection_name)
                return success, message
            else:
                return False, "Database not found and no PDF provided for setup"
        else:
            return True, "Database ready"
        
    except Exception as e:
        return False, f"Error checking database: {str(e)}"

# ================================
# QUERY ENGINE FUNCTIONS
# ================================

@st.cache_resource
def init_query_engine(_client, collection_name):
    """Initialize the query engine for RAG"""
    try:
        # Set up embedding model and LLM
        embed_model = init_embedding()
        llm = init_mistral()
        
        # Configure global settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        # Set up vector store
        vector_store = QdrantVectorStore(
            client=_client, 
            collection_name=collection_name
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Create reranker
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
            top_n=3
        )
        
        # Define prompt template
        template = """Context information is below:
                    ---------------------
                    {context_str}
                    ---------------------
                    Based on the context above, analyze the query and provide the response in the following format:
                    
                    Scenario: [Describe the situation from matching context]
                    Remediation: [Provide specific prevention/remediation steps]
                    Points of contact: [List relevant contact information/helplines]
                    
                    If no relevant information is found in the context, respond with "No matching scenario found."
                    
                    Query: {query_str}
                    
                    Response:"""
        
        qa_prompt_tmpl = PromptTemplate(template)
        
        # Create and configure query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=10,
            node_postprocessors=[rerank]
        )
        
        # Update query engine with template
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
        
        return query_engine
        
    except Exception as e:
        st.error(f"Error initializing query engine: {str(e)}")
        raise e

# ================================
# MAIN STREAMLIT APP
# ================================

def main():
    st.title("🛡️ Fraud Prevention RAG Chatbot")
    st.caption("Powered by Mistral AI • Upload PDF or ask about fraud scenarios!")
    
    # Initialize Qdrant client
    client = init_qdrant()
    collection_name = "demo_29thJan"
    default_pdf_path = "fraud_scenarios.pdf"
    
    # ================================
    # SIDEBAR: Database Management
    # ================================
    
    with st.sidebar:
        st.header("📊 Database Management")
        
        # Check database status
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            st.success("✅ Database is ready!")
            
            # Option to refresh database
            if st.button("🔄 Refresh Database", help="Recreate database from PDF"):
                try:
                    # Delete existing collection
                    client.delete_collection(collection_name)
                    st.cache_resource.clear()  # Clear cache
                    st.success("Database cleared! Please refresh the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error refreshing database: {e}")
        else:
            st.warning("⚠️ Database not found!")
        
        st.divider()
        
        # PDF Upload Section
        st.subheader("📄 Setup Database")
        
        # Option 1: Upload PDF
        uploaded_file = st.file_uploader(
            "Upload Fraud Scenarios PDF", 
            type="pdf",
            help="Upload a PDF containing fraud scenarios to create/update the database"
        )
        
        if uploaded_file and st.button("🚀 Process Uploaded PDF"):
            with st.spinner("Processing PDF..."):
                success, message = setup_database_from_upload(uploaded_file, client, collection_name)
                if success:
                    st.success(message)
                    st.cache_resource.clear()  # Clear cache to reload
                    st.rerun()
                else:
                    st.error(message)
        
        # Option 2: Use default PDF
        if os.path.exists(default_pdf_path):
            st.info(f"📁 Default PDF found: `{default_pdf_path}`")
            if st.button("📄 Use Default PDF"):
                with st.spinner("Setting up database from default PDF..."):
                    success, message = setup_database_from_pdf(default_pdf_path, client, collection_name)
                    if success:
                        st.success(message)
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(message)
        else:
            st.info("💡 To use auto-setup, add your PDF to `data/fraud_scenarios.pdf`")
    
    # ================================
    # MAIN CHAT INTERFACE
    # ================================
    
    # Check if database is ready
    db_ready, db_message = check_and_setup_database(client, collection_name, default_pdf_path)
    
    if not db_ready:
        st.error(f"❌ Database not ready: {db_message}")
        st.info("👈 Please use the sidebar to upload a PDF or set up the database.")
        return
    
    # Initialize query engine
    try:
        query_engine = init_query_engine(client, collection_name)
        st.success("🤖 Chatbot is ready! Ask me about fraud scenarios.")
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💬 Ask me about fraud scenarios (e.g., 'I got a lottery email')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing fraud scenario..."):
                try:
                    response = query_engine.query(prompt)
                    st.markdown(str(response))
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ================================
# RUN THE APP
# ================================

if __name__ == "__main__":
    main()
