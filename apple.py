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

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .chat-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
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
    # Header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Fraud Guardian AI</h1>
        <p style="font-size: 1.2em; margin-top: 10px;">Your Intelligent Fraud Prevention Assistant</p>
        <p style="opacity: 0.9;">Powered by Mistral AI • Upload PDF or ask about fraud scenarios!</p>
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
            st.success("✅ Database Online")
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
        <div style="text-align: center; padding: 3rem;">
            <h3>🚀 Getting Started</h3>
            <p>Please upload a PDF document using the sidebar to begin!</p>
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
        
        for example in examples:
            if st.button(example, use_container_width=True):
                # Add to chat and switch to chat tab
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

if __name__ == "__main__":
    main()
