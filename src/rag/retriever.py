import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    """Retrieve relevant complaint chunks for RAG pipeline"""
    
    def __init__(self, 
                 vector_store_path: str = "../vector_store/prebuilt",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "complaint_chunks"):
        """
        Initialize RAG retriever
        
        Args:
            vector_store_path: Path to pre-built vector store
            embedding_model_name: Name of embedding model (must match vector store)
            collection_name: Name of collection in vector store
        """
        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        logger.info(f"Initializing RAG Retriever...")
        logger.info(f"  Vector store: {vector_store_path}")
        logger.info(f"  Embedding model: {embedding_model_name}")
        logger.info(f"  Collection: {collection_name}")
        
        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model()
        
        # Initialize vector store client
        self.vector_store, self.collection = self._initialize_vector_store()
        
        logger.info(f"✓ RAG Retriever initialized successfully")
        logger.info(f"  Collection size: {self.collection.count() if self.collection else 'Unknown'}")
    
    def _initialize_embedding_model(self) -> SentenceTransformer:
        """Initialize the embedding model"""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        try:
            model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"✓ Embedding model loaded, dimension: {model.get_sentence_embedding_dimension()}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_vector_store(self) -> tuple:
        """Initialize ChromaDB vector store"""
        logger.info(f"Loading vector store from {self.vector_store_path}")
        
        try:
            # Check if vector store exists
            if not os.path.exists(self.vector_store_path):
                logger.warning(f"Vector store not found at {self.vector_store_path}")
                logger.info("Looking for vector store in alternative locations...")
                
                # Try alternative locations
                possible_paths = [
                    "../data/raw/vector_store",  # Might be with data
                    "./vector_store/prebuilt",   # Relative path
                    "vector_store/prebuilt"      # Current directory
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.vector_store_path = path
                        logger.info(f"Found vector store at: {path}")
                        break
                else:
                    raise FileNotFoundError(f"Vector store not found. Checked: {possible_paths}")
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=self.vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get the collection
            try:
                collection = client.get_collection(self.collection_name)
                logger.info(f"✓ Loaded collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Collection {self.collection_name} not found: {e}")
                # List available collections
                collections = client.list_collections()
                logger.info(f"Available collections: {[c.name for c in collections]}")
                
                if collections:
                    # Use first available collection
                    collection = collections[0]
                    self.collection_name = collection.name
                    logger.info(f"Using collection: {self.collection_name}")
                else:
                    raise ValueError(f"No collections found in vector store")
            
            return client, collection
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Fallback: try to load from parquet file
            logger.info("Attempting to load embeddings from parquet file...")
            return self._fallback_initialization()
    
    def _fallback_initialization(self):
        """Fallback initialization using parquet file"""
        parquet_path = "../data/raw/complaint_embeddings.parquet"
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Neither vector store nor parquet file found at {parquet_path}")
        
        logger.info(f"Loading embeddings from parquet: {parquet_path}")
        
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            logger.info(f"✓ Loaded {len(df):,} embeddings from parquet")
            
            # Store embeddings in memory
            self.embeddings = np.stack(df['embedding'].values)
            self.metadata = df.drop(columns=['embedding']).to_dict('records')
            self.documents = df['text'].tolist()
            
            logger.info(f"  Embeddings shape: {self.embeddings.shape}")
            logger.info(f"  Metadata records: {len(self.metadata)}")
            
            # Create a simple in-memory vector store
            from sklearn.metrics.pairwise import cosine_similarity
            self.cosine_similarity = cosine_similarity
            
            return None, None  # Return None for client/collection
            
        except Exception as e:
            logger.error(f"Failed to load from parquet: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query using the embedding model"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding
    
    def retrieve(self, 
                 query: str, 
                 k: int = 5,
                 filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a query
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            filter_criteria: Optional filters (e.g., {"product_category": "Credit Cards"})
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")
        
        # Embed the query
        query_embedding = self.embed_query(query)
        
        # Perform search
        if self.collection:  # Using ChromaDB
            results = self._chromadb_search(query_embedding, k, filter_criteria)
        else:  # Using fallback in-memory store
            results = self._memory_search(query_embedding, k, filter_criteria)
        
        logger.info(f"Retrieved {len(results)} documents")
        
        # Log top result
        if results:
            top_result = results[0]
            logger.info(f"Top result score: {top_result['score']:.3f}")
            logger.info(f"Top result product: {top_result['metadata'].get('product_category', 'N/A')}")
            logger.info(f"Top result preview: {top_result['text'][:100]}...")
        
        return results
    
    def _chromadb_search(self, 
                        query_embedding: np.ndarray, 
                        k: int,
                        filter_criteria: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search using ChromaDB"""
        try:
            # Prepare where filter if provided
            where_filter = None
            if filter_criteria:
                where_filter = {"$and": []}
                for key, value in filter_criteria.items():
                    where_filter["$and"].append({key: value})
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'source': 'chromadb'
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            # Fall back to memory search
            logger.info("Falling back to memory search")
            return self._memory_search(query_embedding, k, filter_criteria)
    
    def _memory_search(self, 
                      query_embedding: np.ndarray, 
                      k: int,
                      filter_criteria: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search using in-memory embeddings (fallback)"""
        if not hasattr(self, 'embeddings'):
            raise ValueError("No embeddings loaded for memory search")
        
        # Calculate similarities
        similarities = self.cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Apply filters if provided
        indices = list(range(len(similarities)))
        
        if filter_criteria:
            filtered_indices = []
            for idx in indices:
                metadata = self.metadata[idx]
                match = True
                for key, value in filter_criteria.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(idx)
            indices = filtered_indices
        
        # Get top-k indices
        if not indices:
            logger.warning("No documents match filter criteria")
            return []
        
        # Get top-k from filtered indices
        top_k_indices = sorted(indices, key=lambda i: similarities[i], reverse=True)[:k]
        
        # Format results
        results = []
        for idx in top_k_indices:
            result = {
                'id': f"doc_{idx}",
                'text': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(similarities[idx]),
                'source': 'memory_fallback'
            }
            results.append(result)
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection"""
        info = {
            'collection_name': self.collection_name,
            'vector_store_path': self.vector_store_path,
            'embedding_model': self.embedding_model_name
        }
        
        if self.collection:
            info['size'] = self.collection.count()
            info['type'] = 'chromadb'
            
            # Sample metadata fields
            sample = self.collection.peek(limit=1)
            if sample['metadatas'] and sample['metadatas'][0]:
                info['metadata_fields'] = list(sample['metadatas'][0][0].keys())
        elif hasattr(self, 'embeddings'):
            info['size'] = len(self.embeddings)
            info['type'] = 'memory_fallback'
            if self.metadata and len(self.metadata) > 0:
                info['metadata_fields'] = list(self.metadata[0].keys())
        
        return info
    
    def test_retrieval(self, test_queries: Optional[List[str]] = None):
        """Test retrieval with sample queries"""
        if test_queries is None:
            test_queries = [
                "credit card billing problems",
                "issues with savings account",
                "loan application denied",
                "money transfer failed"
            ]
        
        print("\n" + "="*70)
        print("RETRIEVAL SYSTEM TEST")
        print("="*70)
        
        collection_info = self.get_collection_info()
        print(f"\n📊 Collection Info:")
        print(f"  Type: {collection_info.get('type', 'Unknown')}")
        print(f"  Size: {collection_info.get('size', 'Unknown'):,} documents")
        print(f"  Model: {collection_info.get('embedding_model', 'Unknown')}")
        
        print(f"\n🔍 Testing {len(test_queries)} queries:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            try:
                results = self.retrieve(query, k=2)
                
                if results:
                    for j, result in enumerate(results, 1):
                        print(f"   Result {j}:")
                        print(f"     Score: {result['score']:.3f}")
                        print(f"     Product: {result['metadata'].get('product_category', 'N/A')}")
                        print(f"     Issue: {result['metadata'].get('issue', 'N/A')}")
                        print(f"     Text: {result['text'][:80]}...")
                else:
                    print("   No results found")
                    
            except Exception as e:
                print(f"   Error: {e}")
        
        print("\n" + "="*70)
        print("RETRIEVAL TEST COMPLETE")
        print("="*70)

# Helper function for easy initialization
def create_retriever(vector_store_path: str = None,
                    embedding_model: str = None) -> RAGRetriever:
    """Create a retriever with sensible defaults"""
    
    # Use provided paths or defaults
    if vector_store_path is None:
        # Try to find vector store
        possible_paths = [
            "../vector_store/prebuilt",
            "./vector_store/prebuilt",
            "../data/raw/vector_store",
            "vector_store/prebuilt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                vector_store_path = path
                break
        else:
            vector_store_path = "../vector_store/prebuilt"
    
    if embedding_model is None:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    logger.info(f"Creating retriever with:")
    logger.info(f"  Vector store: {vector_store_path}")
    logger.info(f"  Embedding model: {embedding_model}")
    
    return RAGRetriever(
        vector_store_path=vector_store_path,
        embedding_model_name=embedding_model
    )

if __name__ == "__main__":
    print("Testing RAG Retriever...")
    
    # Create retriever
    retriever = create_retriever()
    
    # Test retrieval
    retriever.test_retrieval()
    
    # Show collection info
    info = retriever.get_collection_info()
    print(f"\n📋 Collection Information:")
    for key, value in info.items():
        if key != 'metadata_fields':
            print(f"  {key}: {value}")
    
    if 'metadata_fields' in info:
        print(f"\n📁 Metadata Fields:")
        for field in info['metadata_fields']:
            print(f"  • {field}")