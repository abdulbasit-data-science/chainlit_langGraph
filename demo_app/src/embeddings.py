
from dotenv import load_dotenv
import os

from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
load_dotenv()
# Initialize embeddings
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
def get_embeddings():
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

# Pinecone API key

def setup_vector_store():
    # Initialize embeddings
    embeddings = get_embeddings()
   
    # pc = Pinecone(api_key=PINECONE_API_KEY)
    # index = pc.Index(PINECONE_INDEX_NAME)
    api_key=QDRANT_API_KEY
    # Create vector store
    return QdrantVectorStore.from_existing_collection(
    embedding= embeddings,
    api_key=api_key,
    collection_name="my_documents",
    url="https://875eff9a-49a2-4ddb-8f64-652448182974.us-east4-0.gcp.cloud.qdrant.io:6333",
)

# Create a singleton instance of the vector store
vector_store = setup_vector_store()