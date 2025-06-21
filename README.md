# 🛡️ Fraud Prevention RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that helps users identify and prevent fraud scenarios. Built with **Streamlit**, **LlamaIndex**, **Qdrant Cloud**, and **Mistral AI**.

## 🌟 Features

- **🤖 AI-Powered Fraud Detection**: Leverages Mistral AI's large language model for intelligent responses
- **📚 Document-Based Knowledge**: Upload PDF documents containing fraud scenarios and prevention strategies
- **🔍 Advanced Search**: Uses vector embeddings and semantic search to find relevant information
- **⚡ Real-time Chat**: Interactive chat interface for immediate fraud prevention guidance
- **☁️ Cloud-Based**: Fully deployed on Streamlit Cloud with Qdrant Cloud vector database
- **🔄 Reranking**: Implements sentence transformer reranking for improved response accuracy
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Live Demo

Try the live application: [Your App URL](https://rag-chatbot-cpzgf39ylhhrflfymbecwq.streamlit.app/)

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM**: [Mistral AI](https://mistral.ai/) (mistral-large-latest)
- **Vector Database**: [Qdrant Cloud](https://qdrant.tech/)
- **RAG Framework**: [LlamaIndex](https://www.llamaindex.ai/)
- **Embeddings**: BAAI/bge-large-en-v1.5 (HuggingFace)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-2-v2
- **PDF Processing**: PyMuPDF
- **Deployment**: Streamlit Cloud

## 📋 Prerequisites

Before running this application, ensure you have:

- **Python 3.11** (Required for compatibility)
- **Qdrant Cloud Account** ([Sign up here](https://cloud.qdrant.io/))
- **Mistral AI API Key** ([Get it here](https://console.mistral.ai/))
- **Git** for version control

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n fraud-chatbot python=3.11
conda activate fraud-chatbot
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Mistral AI Configuration
MISTRAL_API_KEY=your-mistral-api-key
```

### 5. Prepare Your Data

Place your fraud scenarios PDF file in the root directory and name it `fraud_scenarios.pdf`, or use the upload feature in the app.

### 6. Run Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🌐 Deployment on Streamlit Cloud

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Connect your GitHub account
3. Select your repository
4. Set the main file path: `app.py`
5. Configure secrets (see below)

### 3. Configure Secrets

In your Streamlit Cloud app settings, add these secrets:

```toml
[secrets]
QDRANT_URL = "https://your-cluster-url.qdrant.io"
QDRANT_API_KEY = "your-qdrant-api-key"
MISTRAL_API_KEY = "your-mistral-api-key"
```

## 📁 Project Structure

```
rag-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version specification
├── .python-version       # Alternative Python version file
├── README.md             # Project documentation
├── .env                  # Environment variables (local only)
├── .gitignore           # Git ignore rules
├── fraud_scenarios.pdf   # Sample fraud scenarios document
└── assets/              # Images and other assets
    └── demo.png
```

## 🔑 Getting API Keys

### Qdrant Cloud Setup

1. Visit [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a free account
3. Create a new cluster
4. Copy the cluster URL and API key

### Mistral AI Setup

1. Visit [Mistral AI Console](https://console.mistral.ai/)
2. Create an account
3. Navigate to API Keys section
4. Generate a new API key

## 💡 Usage Guide

### Basic Usage

1. **Start the Application**: Access the web interface
2. **Upload PDF**: Use the sidebar to upload your fraud scenarios document
3. **Ask Questions**: Type questions like:
   - "I received a lottery winning email"
   - "Someone called asking for my bank details"
   - "I got a suspicious text about my credit card"

### Sample Questions

- "What should I do if I receive a phishing email?"
- "How to identify romance scams?"
- "Someone is asking for my SSN over the phone"
- "I think I'm being targeted by a tech support scam"

### Database Management

- **Upload New PDF**: Use the sidebar to replace the knowledge base
- **Refresh Database**: Clear and rebuild the vector database
- **Check Status**: Monitor database connectivity in the sidebar

## 🔧 Configuration

### Customizing the Model

Edit these parameters in `app.py`:

```python
# LLM Configuration
llm = MistralAI(
    model="mistral-large-latest",  # Change model
    temperature=0.7,               # Adjust creativity
    max_tokens=512,               # Response length
    api_key=api_key
)

# Embedding Configuration
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",  # Change embedding model
    trust_remote_code=True
)

# Search Configuration
query_engine = index.as_query_engine(
    similarity_top_k=10,  # Number of retrieved documents
    node_postprocessors=[rerank]
)
```

### Prompt Customization

Modify the prompt template in `app.py`:

```python
template = """Your custom prompt template here..."""
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
ModuleNotFoundError: No module named 'qdrant_client'
```
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

#### 2. API Connection Issues
```
Cannot connect to Qdrant Cloud
```
**Solution**: 
- Check your Qdrant URL and API key
- Verify network connectivity
- Ensure your Qdrant cluster is running

#### 3. Mistral AI Errors
```
Authentication failed
```
**Solution**:
- Verify your Mistral API key
- Check API quota and billing status
- Ensure the API key has proper permissions

#### 4. PDF Processing Issues
```
No content found in PDF
```
**Solution**:
- Ensure PDF is not password-protected
- Check if PDF contains readable text (not just images)
- Try a different PDF file

#### 5. Memory Issues
```
Out of memory error
```
**Solution**:
- Reduce `similarity_top_k` parameter
- Use smaller embedding models
- Process documents in smaller batches

### Performance Optimization

1. **Reduce Response Time**:
   - Lower `similarity_top_k` from 10 to 5
   - Use smaller embedding models
   - Implement caching for frequent queries

2. **Improve Accuracy**:
   - Increase `similarity_top_k` parameter
   - Use better embedding models
   - Fine-tune the prompt template

3. **Handle Large Documents**:
   - Split large PDFs into smaller sections
   - Implement document chunking strategies
   - Use batch processing for uploads

## 📊 Monitoring & Analytics

### Usage Tracking

The app automatically tracks:
- Number of queries processed
- Response times
- Error rates
- Database status

### Performance Metrics

Monitor these key metrics:
- **Response Time**: Target < 5 seconds
- **Accuracy**: Based on user feedback
- **Uptime**: Database and API availability

## 🔒 Security & Privacy

### Data Protection

- **No Data Persistence**: Chat history is not stored permanently
- **Secure APIs**: All API keys are encrypted in Streamlit secrets
- **HTTPS**: All communication is encrypted in transit

### Best Practices

1. **API Key Management**:
   - Never commit API keys to version control
   - Use environment variables or secrets management
   - Regularly rotate API keys

2. **Document Security**:
   - Only upload non-sensitive fraud prevention documents
   - Review document content before uploading
   - Remove personal information from documents

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/new-feature`
3. **Make Changes**: Follow the coding standards
4. **Add Tests**: Ensure your code is tested
5. **Submit Pull Request**: With detailed description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black app.py
flake8 app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">

**Built with ❤️ using Streamlit, LlamaIndex, and Mistral AI**

[⭐ Star this repo](https://github.com/your-username/rag-chatbot) | [🐛 Report Bug](https://github.com/your-username/rag-chatbot/issues) | [💡 Request Feature](https://github.com/your-username/rag-chatbot/issues)

</div>
