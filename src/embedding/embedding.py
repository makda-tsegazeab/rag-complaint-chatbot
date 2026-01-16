from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import time
import json
import pandas as pd
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintEmbedder:
    """Generate embeddings for complaint chunks using sentence transformers"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        """
        Initialize embedder with specified model
        
        Args:
            model_name: Name of the sentence transformer model
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {device}")
        
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            load_time = time.time() - start_time
            
            # Get model info
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"✓ Model loaded in {load_time:.2f} seconds")
            logger.info(f"✓ Embedding dimension: {self.embedding_dimension}")
            logger.info(f"✓ Model max sequence length: {self.model.max_seq_length}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            
            # Try CPU as fallback
            if device == 'cuda':
                logger.info("Trying CPU as fallback...")
                try:
                    self.model = SentenceTransformer(model_name, device='cpu')
                    self.device = 'cpu'
                    self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                    logger.info(f"✓ Model loaded on CPU, dimension: {self.embedding_dimension}")
                except Exception as e2:
                    logger.error(f"Failed to load on CPU: {e2}")
                    raise
            else:
                raise
    
    def embed_text(self, text: str, show_progress: bool = False) -> np.ndarray:
        """Embed a single text string"""
        if not text or not text.strip():
            # Return zero vector for empty text
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.embedding_dimension)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                device=self.device
            )
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return np.zeros(self.embedding_dimension)
    
    def embed_texts(self, 
                   texts: List[str], 
                   batch_size: int = 32,
                   show_progress: bool = True) -> np.ndarray:
        """Embed multiple texts efficiently with batching"""
        if not texts:
            return np.array([])
        
        logger.info(f"Embedding {len(texts):,} texts with batch size {batch_size}")
        
        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and str(text).strip():
                valid_texts.append(str(text))
                valid_indices.append(i)
        
        if not valid_texts:
            logger.warning("No valid texts to embed")
            return np.zeros((len(texts), self.embedding_dimension))
        
        logger.info(f"  Valid texts: {len(valid_texts):,} of {len(texts):,}")
        
        start_time = time.time()
        
        try:
            # Embed in batches
            embeddings = self.model.encode(
                valid_texts, 
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                device=self.device
            )
            
            # Create full embeddings array with zeros for empty texts
            full_embeddings = np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)
            
            for idx, embedding in zip(valid_indices, embeddings):
                full_embeddings[idx] = embedding
            
            total_time = time.time() - start_time
            
            logger.info(f"✓ Embedding completed in {total_time:.2f} seconds "
                       f"({len(texts)/total_time:.1f} texts/second)")
            
            # Clear GPU memory if using CUDA
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            return full_embeddings
            
        except Exception as e:
            logger.error(f"Error during batch embedding: {e}")
            
   # Fallback: embed one by one
            logger.info("Falling back to single-text embedding...")
            embeddings = []
            for i, text in enumerate(tqdm(texts, desc="Embedding texts")):
                if i % 1000 == 0:
                    logger.info(f"Processed {i:,} texts")
                embedding = self.embed_text(text, show_progress=False)
                embeddings.append(embedding)
            
            return np.array(embeddings)
    
    def embed_chunks(self, 
                    chunks: List[Dict[str, Any]], 
                    batch_size: int = 32,
                    save_every: Optional[int] = None,
                    checkpoint_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Embed all text chunks and add embeddings to chunk dictionaries
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for embedding
            save_every: Save checkpoint every N chunks (optional)
            checkpoint_path: Path to save checkpoints (optional)
            
        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return []
        
        logger.info(f"Embedding {len(chunks):,} chunks with model: {self.model_name}")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()  # Convert to list for JSON serialization
        
        # Analyze embedding quality
        self.analyze_embeddings(embeddings)
        
        # Save checkpoint if requested
        if save_every and checkpoint_path and len(chunks) >= save_every:
            self.save_checkpoint(chunks[:save_every], checkpoint_path)
        
        return chunks
    
    def analyze_embeddings(self, embeddings: np.ndarray):
        """Analyze and log embedding statistics"""
        if len(embeddings) == 0:
            return
        
        print("\n" + "="*70)
        print("EMBEDDING QUALITY ANALYSIS")
        print("="*70)
        
        print(f"Total embeddings: {len(embeddings):,}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Model used: {self.model_name}")
        
        # Calculate statistics
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        
        print(f"\n📊 EMBEDDING STATISTICS:")
        print(f"  Mean norm:          {np.mean(embedding_norms):.4f}")
        print(f"  Std of norms:       {np.std(embedding_norms):.4f}")
        print(f"  Min norm:           {np.min(embedding_norms):.4f}")
        print(f"  Max norm:           {np.max(embedding_norms):.4f}")
        print(f"  Median norm:        {np.median(embedding_norms):.4f}")
        
        # Check for zero vectors (empty texts)
        zero_vectors = np.sum(np.all(embeddings == 0, axis=1))
        if zero_vectors > 0:
            print(f"  ⚠️  Zero vectors:     {zero_vectors} (empty texts)")
        
        # Check for NaN or Inf values
        nan_count = np.sum(np.isnan(embeddings))
        inf_count = np.sum(np.isinf(embeddings))
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ❌ NaN values:        {nan_count}")
            print(f"  ❌ Inf values:        {inf_count}")
        else:
            print(f"  ✓ No NaN/Inf values")
        
        # Sample similarity analysis
        if len(embeddings) >= 5:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate similarities between first 5 embeddings
            sample_size = min(5, len(embeddings))
            sample_embeddings = embeddings[:sample_size]
            similarities = cosine_similarity(sample_embeddings)
            
            print(f"\n🔍 SAMPLE COSINE SIMILARITIES (first {sample_size} embeddings):")
            print("    " + " ".join([f"{i+1:>6}" for i in range(sample_size)]))
            for i in range(sample_size):
                row = [f"{sim:6.3f}" for sim in similarities[i]]
                print(f"{i+1:2}: " + " ".join(row))
            
            # Calculate self-similarity (diagonal should be 1.0)
            self_similarity = np.diag(similarities)
            if not np.allclose(self_similarity, 1.0, atol=1e-5):
                print(f"  ⚠️  Self-similarity issues: {self_similarity}")
        
        # Distribution analysis
        print(f"\n📈 EMBEDDING DISTRIBUTION:")
        flat_embeddings = embeddings.flatten()
        print(f"  Mean value:         {np.mean(flat_embeddings):.6f}")
        print(f"  Std deviation:      {np.std(flat_embeddings):.6f}")
        print(f"  Min value:          {np.min(flat_embeddings):.6f}")
        print(f"  Max value:          {np.max(flat_embeddings):.6f}")
        
        # Quality assessment
        print(f"\n✅ EMBEDDING QUALITY ASSESSMENT:")
        
        quality_issues = []
        
        if zero_vectors / len(embeddings) > 0.01:
            quality_issues.append(f"Too many zero vectors ({zero_vectors/len(embeddings)*100:.1f}%)")
        
        if nan_count > 0 or inf_count > 0:
            quality_issues.append("NaN or Inf values detected")
        
        if np.std(embedding_norms) < 0.1:
            quality_issues.append("Very low variance in embedding norms")
        
        if quality_issues:
            print("  ⚠️  Potential issues detected:")
            for issue in quality_issues:
                print(f"    • {issue}")
        else:
            print("  ✓ Good embedding quality!")
        
        print("\n" + "="*70)
    
    def save_checkpoint(self, 
                       chunks: List[Dict[str, Any]], 
                       checkpoint_path: str):
        """Save embedding checkpoint"""
        import os
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Save chunks with embeddings
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Saved checkpoint with {len(chunks):,} chunks to {checkpoint_path}")
    
    def test_model_capabilities(self, 
                               test_texts: Optional[List[str]] = None,
                               test_cases: Optional[List[Dict[str, str]]] = None):
        """
        Test the embedding model with sample texts
        
        Args:
            test_texts: List of texts to test
            test_cases: List of dicts with 'text' and 'description'
        """
        if test_texts is None and test_cases is None:
            # Default test cases for financial complaints
            test_cases = [
                {
                    'text': "I have issues with my credit card billing and unauthorized charges.",
                    'description': "Credit card billing complaint"
                },
                {
                    'text': "The bank charged me unexpected fees on my savings account.",
                    'description': "Savings account fees complaint"
                },
                {
                    'text': "My personal loan application was rejected without proper explanation.",
                    'description': "Loan application rejection"
                },
                {
                    'text': "Money transfer failed but funds were deducted from my account.",
                    'description': "Money transfer failure"
                },
                {
                    'text': "Customer service was unhelpful when I reported fraudulent activity.",
                    'description': "Customer service complaint"
                }
            ]
            test_texts = [case['text'] for case in test_cases]
        elif test_texts is not None and test_cases is None:
            test_cases = [{'text': text, 'description': f"Test {i+1}"} 
                         for i, text in enumerate(test_texts)]
        
        print("\n" + "="*70)
        print("EMBEDDING MODEL CAPABILITY TEST")
        print("="*70)
        
        print(f"\nModel: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Test cases: {len(test_cases)}")
        
        # Generate embeddings
        embeddings = self.embed_texts(test_texts, batch_size=len(test_texts))
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        print("\n📝 TEST TEXTS:")
        for i, case in enumerate(test_cases):
            print(f"\n{i+1}. {case['description']}:")
            print(f"   Text: {case['text'][:80]}...")
        
        print("\n🔗 COSINE SIMILARITY MATRIX:")
        print("   " + " ".join([f"{i+1:>8}" for i in range(len(test_cases))]))
        
        for i in range(len(test_cases)):
            row = [f"{sim:8.3f}" for sim in similarities[i]]
            print(f"{i+1:2}: " + " ".join(row))
        
        print("\n🎯 SEMANTIC ANALYSIS:")
        
        # Analyze expected similarities
        expected_similarities = [
            (0, 1, 0.6, "Both about banking/financial issues"),
            (0, 2, 0.5, "Credit card vs loan - related financial products"),
            (0, 3, 0.3, "Credit card vs money transfer - less related"),
            (2, 3, 0.4, "Loan vs money transfer - financial services"),
        ]
        
        for i, j, expected, reason in expected_similarities:
            if i < len(test_cases) and j < len(test_cases):
                actual = similarities[i, j]
                diff = abs(actual - expected)
                symbol = "✓" if diff < 0.2 else "⚠" if diff < 0.3 else "✗"
                print(f"  {symbol} Texts {i+1}-{j+1}: Expected ~{expected:.2f}, "
                      f"Got {actual:.3f} ({reason})")
        
        print("\n" + "="*70)
        
        return embeddings, similarities

def run_embedding_pipeline(chunks_path: str = '../data/sampled/complaint_chunks.json',
                          output_path: str = '../data/sampled/embedded_chunks.json',
                          model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                          batch_size: int = 64):
    """Complete embedding pipeline"""
    import json
    import os
    
    logger.info("="*70)
    logger.info("STARTING EMBEDDING PIPELINE")
    logger.info("="*70)
    
    try:
        # Load chunks
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks not found at {chunks_path}. Run chunking first.")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks):,} chunks from {chunks_path}")
        
        # Initialize embedder
        embedder = ComplaintEmbedder(model_name=model_name)
        
        # Test model first
        print("\nTesting model capabilities...")
        embedder.test_model_capabilities()
        
        # Embed all chunks
        print(f"\nStarting batch embedding of {len(chunks):,} chunks...")
        embedded_chunks = embedder.embed_chunks(
            chunks,
            batch_size=batch_size,
            save_every=5000,
            checkpoint_path=output_path.replace('.json', '_checkpoint.json')
        )
        
        # Save embedded chunks
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Saved {len(embedded_chunks):,} embedded chunks to {output_path}")
        
        # Also save embeddings separately as numpy array
        embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks])
        embeddings_path = output_path.replace('.json', '_embeddings.npy')
        np.save(embeddings_path, embeddings)
        logger.info(f"✓ Saved embeddings array to {embeddings_path}")
        
        # Save metadata separately
        metadata = []
        for chunk in embedded_chunks:
            metadata.append({k: v for k, v in chunk.items() if k != 'embedding'})
        
        metadata_path = output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved metadata to {metadata_path}")
        
        return embedded_chunks
        
    except Exception as e:
        logger.error(f"Embedding pipeline failed: {e}")
        raise

if __name__ == "__main__":
    print("Testing Complaint Embedder...")
    
    # Test with small sample
    embedder = ComplaintEmbedder()
    
    # Test model capabilities
    embedder.test_model_capabilities()
    
    # Test single embedding
    test_text = "Credit card billing dispute with unauthorized charges"
    embedding = embedder.embed_text(test_text)
    print(f"\nSingle embedding shape: {embedding.shape}")
    print(f"Sample embedding (first 5 values): {embedding[:5]}")
    
    print("\n" + "="*70)
    print("READY FOR TASK 2C: VECTOR STORE INDEXING")
    print("="*70)