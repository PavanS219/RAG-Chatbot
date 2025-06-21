import os
import qdrant_client
from llama_index.core.schema import Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file.pymu_pdf import PyMuPDFReader

def setup_fraud_database(pdf_path):
    """
    Set up fraud scenarios database from PDF file using Qdrant Cloud
    
    Args:
        pdf_path (str): Path to the fraud scenarios PDF file
    """
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        print("Please provide the correct path to your fraud scenarios PDF.")
        return
    
    print("Setting up fraud scenarios database from PDF...")
    
    # Get Qdrant Cloud credentials from environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        print("❌ Error: Qdrant Cloud credentials not found!")
        print("Please set the following environment variables:")
        print("- QDRANT_URL (your Qdrant Cloud cluster URL)")
        print("- QDRANT_API_KEY (your Qdrant Cloud API key)")
        return
    
    try:
        # Initialize Qdrant Cloud client
        print("Connecting to Qdrant Cloud...")
        client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Test connection
        client.get_collections()
        print("✅ Connected to Qdrant Cloud successfully!")
        
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant Cloud: {e}")
        print("Please check your QDRANT_URL and QDRANT_API_KEY")
        return
    
    try:
        # Initialize embedding model
        print("Loading embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            trust_remote_code=True
        )
        print("✅ Embedding model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return
    
    try:
        # Load PDF document
        print(f"Reading PDF from: {pdf_path}")
        
        # Using PyMuPDFReader (recommended for better text extraction)
        loader = PyMuPDFReader()
        documents = loader.load(file_path=pdf_path)
        
        print(f"✅ Successfully loaded {len(documents)} pages from PDF")
        
        # Optional: Print first few characters to verify content
        if documents:
            print(f"Sample content: {documents[0].text[:200]}...")
        
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install pymupdf")
        return
    
    try:
        # Set up vector store
        collection_name = "demo_29thJan"
        print(f"Setting up vector store with collection: {collection_name}")
        
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name=collection_name
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index and store documents
        print("Creating index and storing documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        print(f"✅ Successfully indexed {len(documents)} document pages!")
        print("🚀 Your data is now stored in Qdrant Cloud!")
        print("🚀 You can now run your Streamlit app!")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return

def main():
    print("🛡️ Fraud Scenarios Database Setup - Qdrant Cloud")
    print("=" * 50)
    
    # Check for environment variables first
    if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
        print("⚠️  Environment variables not set!")
        print("Please set:")
        print("export QDRANT_URL='your-qdrant-cloud-url'")
        print("export QDRANT_API_KEY='your-qdrant-api-key'")
        print()
    
    # Get PDF path from user
    pdf_path = "data/fraud_scenarios.pdf"
    
    
    
    # Setup database
    setup_fraud_database(pdf_path)

if __name__ == "__main__":
    main()
