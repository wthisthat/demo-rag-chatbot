import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker

# load env var
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX")
KNOWLEDGE_BASE_DIR = "knowledge_documents/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "models/text-embedding-004"

def init_pinecone():
    """Initialize Pinecone based on API key & setup index if none exist"""
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    index_first_setup = False
    # create pinecone index if index doesn't exist
    if INDEX_NAME not in pinecone.list_indexes().names():
        index_first_setup = True
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=768, # gemini model (text-embedding-004) output dimension is 768
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1") 
        )

    return index_first_setup

def load_pdfs_from_directory(directory):
    """Load all PDF files from the given directory."""
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

def embed_documents_and_store(docs, index_name):
    """Embed documents and store them in Pinecone vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

def process_pdfs_to_vector_store():
    """Function to load pdfs, split pdf text into chunk, embed, and store embedding in a vector store."""
    pdf_files = load_pdfs_from_directory(KNOWLEDGE_BASE_DIR)
    for pdf in pdf_files:
        # load pdf and split into pages
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()

        # split pdf texts into smaller chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
        docs = text_splitter.split_documents(pages)

        # semantic chunk splitter (experiment, not in use)
        # text_splitter = SemanticChunker(GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL))
        
        embed_documents_and_store(docs, INDEX_NAME)