import os
import streamlit as st
import qdrant_client
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings

# Initialize Qdrant Cloud client
@st.cache_resource
def init_qdrant():
    try:
        # Get Qdrant Cloud credentials from environment variables
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            st.error("❌ Qdrant Cloud credentials not found!")
            st.info("💡 Please set QDRANT_URL and QDRANT_API_KEY environment variables")
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

# Initialize embedding model
@st.cache_resource
def init_embedding():
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        trust_remote_code=True
    )

# Initialize Mistral AI
@st.cache_resource
def init_mistral():
    return MistralAI(
        model="mistral-large-latest",
        temperature=0.7,
        max_tokens=512,
        api_key=os.getenv("MISTRAL_API_KEY")
    )

# Initialize query engine
@st.cache_resource
def init_query_engine(_client, collection_name):
    try:
        # Check if collection exists
        collections = _client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            st.error(f"❌ Collection '{collection_name}' not found!")
            st.info("💡 Please run `python setup_database.py` first to create the database.")
            st.stop()
        
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
        
        # Create and configure query engine with explicit LLM
        query_engine = index.as_query_engine(
            llm=llm,  # Explicitly pass the Mistral LLM
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

def main():
    st.title("🛡️ Fraud Prevention RAG Chatbot")
    st.caption("Powered by Mistral AI • Ask about fraud scenarios and get instant help!")
    
    # Check for Mistral API key in environment variables
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    
    if not mistral_api_key:
        st.error("❌ MISTRAL_API_KEY environment variable not found!")
        st.info("💡 Please set your Mistral API key as an environment variable.")
        st.code("export MISTRAL_API_KEY='your-api-key-here'")
        st.stop()
    
    # API key is available
    # API key is available
    try:
        # Initialize components
        client = init_qdrant()
        collection_name = "demo_29thJan"
        
        # Initialize query engine
        query_engine = init_query_engine(client, collection_name)
        
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
                    response = query_engine.query(prompt)
                    st.markdown(str(response))
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Make sure your Mistral API key is valid and you have sufficient credits.")

if __name__ == "__main__":
    main()
