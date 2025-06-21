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
# PAGE CONFIG & STYLING
# ================================

st.set_page_config(
    page_title="🛡️ Fraud Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with animations
st.markdown("""
<style>
    /* Animated background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Animated header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: headerPulse 3s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes headerPulse {
        from { transform: scale(1); }
        to { transform: scale(1.02); }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    /* Chat container with glass effect */
    .chat-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Animated buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #764ba2 0%, #f093fb 50%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Floating animation for icons */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Pulse animation for status indicators */
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Progress bar animation */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        animation: progressGlow 2s ease-in-out infinite;
    }
    
    @keyframes progressGlow {
        0%, 100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
    }
    
    /* Chat message animations */
    .stChatMessage {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateX(-20px); 
        }
        to { 
            opacity: 1; 
            transform: translateX(0); 
        }
    }
    
    /* File upload area styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 2px dashed rgba(255,255,255,0.3);
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(255,255,255,0.6);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Loading spinner enhancement */
    .stSpinner {
        color: #667eea !important;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: rgba(35, 213, 171, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stError {
        background: rgba(231, 60, 126, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        animation: shake 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
</style>
""", unsafe_allow_html=True)

# ================================
# INITIALIZATION FUNCTIONS
# ================================

@st.cache_resource
def init_qdrant():
    """Initialize Qdrant Cloud client with animated loading"""
    try:
        qdrant_url = st.secrets.get("QDRANT_URL") or os.getenv("QDRANT_URL")
        qdrant_api_key = st.secrets.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            st.error("❌ Qdrant Cloud credentials not found!")
            st.info("💡 Please set QDRANT_URL and QDRANT_API_KEY in Streamlit secrets")
            st.stop()
        
        client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        client.get_collections()
        return client
        
    except Exception as e:
        st.error(f"❌ Cannot connect to Qdrant Cloud: {str(e)}")
        st.stop()

@st.cache_resource
def init_embedding():
    """Initialize embedding model"""
    return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

@st.cache_resource
def init_mistral():
    """Initialize Mistral AI model"""
    api_key = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("❌ MISTRAL_API_KEY not found!")
        st.stop()
    return MistralAI(model="mistral-large-latest", temperature=0.7, max_tokens=512, api_key=api_key)

# ================================
# DATABASE FUNCTIONS
# ================================

def setup_database_from_pdf(pdf_path, client, collection_name):
    """Set up database from PDF with progress tracking"""
    try:
        if not os.path.exists(pdf_path):
            return False, f"PDF file not found at {pdf_path}"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📄 Loading PDF document...")
        progress_bar.progress(20)
        
        loader = PyMuPDFReader()
        documents = loader.load(file_path=pdf_path)
        
        if not documents:
            return False, "No content found in PDF"
        
        status_text.text("🔄 Loading embedding model...")
        progress_bar.progress(40)
        embed_model = init_embedding()
        
        status_text.text("🔄 Setting up vector store...")
        progress_bar.progress(60)
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        status_text.text("🔄 Creating index and storing documents...")
        progress_bar.progress(80)
        
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model, show_progress=False
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Database setup complete!")
        
        return True, f"Successfully indexed {len(documents)} document pages!"
        
    except Exception as e:
        return False, f"Error setting up database: {str(e)}"

def setup_database_from_upload(uploaded_file, client, collection_name):
    """Setup database from uploaded PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        success, message = setup_database_from_pdf(tmp_path, client, collection_name)
        os.unlink(tmp_path)
        return success, message
        
    except Exception as e:
        return False, f"Error processing uploaded PDF: {str(e)}"

@st.cache_resource
def init_query_engine(_client, collection_name):
    """Initialize the query engine for RAG"""
    try:
        embed_model = init_embedding()
        llm = init_mistral()
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        vector_store = QdrantVectorStore(client=_client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
        
        rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
        
        template = """Context information is below:
                    ---------------------
                    {context_str}
                    ---------------------
                    Based on the context above, analyze the query and provide the response in the following format:
                    
                    🎯 **Scenario**: [Describe the situation from matching context]
                    🛡️ **Remediation**: [Provide specific prevention/remediation steps]
                    📞 **Points of Contact**: [List relevant contact information/helplines]
                    
                    If no relevant information is found in the context, respond with "❌ No matching scenario found."
                    
                    Query: {query_str}
                    
                    Response:"""
        
        qa_prompt_tmpl = PromptTemplate(template)
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=10, node_postprocessors=[rerank])
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        
        return query_engine
        
    except Exception as e:
        st.error(f"Error initializing query engine: {str(e)}")
        raise e

# ================================
# MAIN STREAMLIT APP
# ================================

def main():
    # Header with gradient background and animations
    st.markdown("""
    <div class="main-header">
        <h1 class="floating">🛡️ Fraud Guardian AI</h1>
        <p style="font-size: 1.2em; margin-top: 10px;">Your Intelligent Fraud Prevention Assistant</p>
        <p style="opacity: 0.9;">✨ Powered by Mistral AI • Upload PDF or ask about fraud scenarios! ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    client = init_qdrant()
    collection_name = "demo_29thJan"
    default_pdf_path = "fraud_scenarios.pdf"
    
    # ================================
    # SIDEBAR: Enhanced Database Management
    # ================================
    
    with st.sidebar:
        st.markdown("### 📊 Database Control Center")
        
        # Database Status with metrics
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            st.markdown('<div class="pulse">✅ Database Online</div>', unsafe_allow_html=True)
            try:
                collection_info = client.get_collection(collection_name)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Documents", collection_info.points_count)
                with col2:
                    st.metric("🔍 Vectors", collection_info.vectors_count or "N/A")
            except:
                pass
                
            # Database actions
            if st.button("🔄 Refresh Database", use_container_width=True):
                try:
                    client.delete_collection(collection_name)
                    st.cache_resource.clear()
                    st.success("🔄 Database refreshed! Please reload.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Database Offline")
            st.info("👇 Setup required below")
        
        st.markdown("---")
        
        # Enhanced PDF Upload Section
        st.markdown("### 📤 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose PDF file", 
            type="pdf",
            help="Upload fraud scenarios PDF to create/update database"
        )
        
        if uploaded_file:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB"
            }
            st.json(file_details)
            
            if st.button("🚀 Process Document", use_container_width=True):
                with st.spinner("Processing..."):
                    success, message = setup_database_from_upload(uploaded_file, client, collection_name)
                    if success:
                        st.success(message)
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(message)
        
        # Quick Setup Option
        if os.path.exists(default_pdf_path):
            st.markdown("### ⚡ Quick Setup")
            st.info(f"📁 Found: `{default_pdf_path}`")
            if st.button("⚡ Quick Setup", use_container_width=True):
                with st.spinner("Setting up..."):
                    success, message = setup_database_from_pdf(default_pdf_path, client, collection_name)
                    if success:
                        st.success(message)
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(message)
    
    # ================================
    # MAIN CHAT INTERFACE
    # ================================
    
    # Check database status
    if collection_name not in collection_names:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255,255,255,0.1); 
                    backdrop-filter: blur(10px); border-radius: 20px; margin: 2rem 0;">
            <div class="floating">
                <h3>🚀 Getting Started</h3>
                <p style="font-size: 1.1em; margin-top: 1rem;">
                    Upload a PDF document using the sidebar to begin your fraud prevention journey!
                </p>
                <div style="margin-top: 2rem;">
                    <span style="font-size: 3em; animation: pulse 2s infinite;">📄</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize query engine
    try:
        query_engine = init_query_engine(client, collection_name)
    except Exception as e:
        st.error(f"❌ Failed to initialize AI: {str(e)}")
        return
    
    # Chat interface with tabs
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📚 Quick Examples"])
    
    with tab1:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Welcome message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "👋 Hello! I'm your Fraud Guardian AI. Describe any suspicious situation and I'll help identify potential fraud and provide prevention guidance."
            })
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("💬 Describe your situation or ask about fraud prevention..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing scenario..."):
                    try:
                        response = query_engine.query(prompt)
                        st.markdown(str(response))
                        st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with tab2:
        st.markdown("### 🎯 Try These Examples")
        
        examples = [
            "🎰 I received an email saying I won a lottery I never entered",
            "📞 Someone called claiming to be from my bank asking for my PIN",
            "💳 I got a text about suspicious activity on my credit card",
            "🏠 Someone offered to buy my house for cash sight unseen",
            "💰 I was offered a work-from-home job that seems too good to be true"
        ]
        
        for i, example in enumerate(examples):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Initialize messages if not exists
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                # Add user message
                st.session_state.messages.append({"role": "user", "content": example})
                
                # Generate AI response immediately
                try:
                    with st.spinner("🔍 Analyzing scenario..."):
                        response = query_engine.query(example)
                        st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    st.success("✅ Response generated! Check the Chat tab.")
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error("Failed to generate response.")
                
                # Trigger rerun to show the conversation
                st.rerun()

if __name__ == "__main__":
    main()
