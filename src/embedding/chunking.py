from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintChunker:
    """Handle text chunking for complaint narratives"""
    
    def __init__(self, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50,
                 separators: Optional[List[str]] = None):
        """
        Initialize chunker with specified parameters
        
        Args:
            chunk_size: Maximum size of each chunk in characters (default: 500 as per pre-built)
            chunk_overlap: Overlap between chunks in characters (default: 50 as per pre-built)
            separators: List of separators for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use default separators if none provided
        if separators is None:
            separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            keep_separator=True
        )
        
        logger.info(f"Initialized chunker with size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_complaint(self, 
                       complaint_text: str, 
                       metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single complaint narrative with metadata
        
        Args:
            complaint_text: The complaint narrative text
            metadata: Dictionary of metadata for this complaint
            
        Returns:
            List of chunks with metadata
        """
        if not complaint_text or not isinstance(complaint_text, str):
            logger.warning(f"Invalid complaint text for complaint {metadata.get('complaint_id', 'unknown')}")
            return []
        
        # Clean text if needed
        text = complaint_text.strip()
        if not text:
            return []
        
        # Split text into chunks
        try:
            chunks = self.text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []
        
        # Prepare chunk documents with metadata
        chunk_documents = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = {
                'text': chunk_text,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_id': f"{metadata.get('complaint_id', 'unknown')}_chunk_{i}",
                **metadata  # Include all original metadata
            }
            chunk_documents.append(chunk_doc)
        
        if len(chunks) > 1:
            logger.debug(f"Created {len(chunks)} chunks for complaint {metadata.get('complaint_id', 'unknown')}")
        
        return chunk_documents
    
    def chunk_dataframe(self, 
                       df: pd.DataFrame, 
                       text_column: str = 'consumer_complaint_narrative',
                       batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Chunk all complaints in a dataframe with progress tracking
        
        Args:
            df: Dataframe containing complaints
            text_column: Name of the column containing narrative text
            batch_size: Number of complaints to process before logging progress
            
        Returns:
            List of all chunks from all complaints
        """
        all_chunks = []
        total_complaints = len(df)
        
        logger.info(f"Starting chunking of {total_complaints:,} complaints")
        logger.info(f"Chunk parameters: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # Required metadata fields
        required_metadata_fields = [
            'complaint_id', 'product_category', 'date_received',
            'issue', 'sub_issue', 'company', 'state'
        ]
        
        for idx, row in df.iterrows():
            # Extract metadata from all available fields
            metadata = {}
            for field in required_metadata_fields:
                if field in row:
                    metadata[field] = row[field]
                else:
                    metadata[field] = 'unknown'
            
            # Add additional fields if present
            additional_fields = ['original_product', 'narrative_length', 'cleaned_length']
            for field in additional_fields:
                if field in row:
                    metadata[field] = row[field]
            
            # Get complaint text
            if text_column not in row:
                logger.warning(f"Text column '{text_column}' not found in row {idx}")
                continue
                
            complaint_text = row[text_column]
            
            # Skip if text is empty or NaN
            if pd.isna(complaint_text) or not str(complaint_text).strip():
                continue
            
            # Chunk the complaint
            chunks = self.chunk_complaint(str(complaint_text), metadata)
            all_chunks.extend(chunks)
            
            # Log progress
            if (idx + 1) % batch_size == 0:
                chunks_so_far = len(all_chunks)
                logger.info(f"Processed {idx + 1:,}/{total_complaints:,} complaints "
                           f"({(idx + 1)/total_complaints*100:.1f}%) - "
                           f"Created {chunks_so_far:,} chunks so far")
        
        logger.info(f"✓ Chunking complete. Created {len(all_chunks):,} total chunks "
                   f"from {total_complaints:,} complaints")
        
        # Calculate and display statistics
        self.analyze_chunking_results(df, all_chunks)
        
        return all_chunks
    
    def analyze_chunking_results(self, 
                               original_df: pd.DataFrame, 
                               chunks: List[Dict[str, Any]]):
        """Analyze and log chunking statistics"""
        if not chunks:
            logger.warning("No chunks created")
            return
        
        # Convert to dataframe for analysis
        chunks_df = pd.DataFrame(chunks)
        
        # Calculate statistics
        total_complaints = len(original_df)
        total_chunks = len(chunks)
        
        avg_chunks_per_complaint = total_chunks / total_complaints if total_complaints > 0 else 0
        avg_chunk_length = chunks_df['text'].apply(len).mean() if len(chunks_df) > 0 else 0
        
        # Distribution of chunks per complaint
        if 'complaint_id' in chunks_df.columns:
            chunks_per_complaint = chunks_df.groupby('complaint_id').size()
            chunks_per_stats = {
                'min': chunks_per_complaint.min(),
                'max': chunks_per_complaint.max(),
                'median': chunks_per_complaint.median(),
                'mean': chunks_per_complaint.mean()
            }
        else:
            chunks_per_stats = None
        
        # Chunk length distribution
        chunk_lengths = chunks_df['text'].apply(len)
        length_distribution = {
            'very_short': (chunk_lengths < 100).sum(),
            'short': ((chunk_lengths >= 100) & (chunk_lengths < 300)).sum(),
            'medium': ((chunk_lengths >= 300) & (chunk_lengths <= self.chunk_size)).sum(),
            'long': (chunk_lengths > self.chunk_size).sum()
        }
        
        print("\n" + "="*70)
        print("CHUNKING ANALYSIS REPORT")
        print("="*70)
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Total complaints processed: {total_complaints:,}")
        print(f"  Total chunks created:      {total_chunks:,}")
        print(f"  Average chunks per complaint: {avg_chunks_per_complaint:.2f}")
        print(f"  Average chunk length:       {avg_chunk_length:.0f} characters")
        
        print(f"\n⚙️  CHUNKING PARAMETERS:")
        print(f"  Chunk size:    {self.chunk_size} characters")
        print(f"  Chunk overlap: {self.chunk_overlap} characters")
        
        if chunks_per_stats:
            print(f"\n📈 CHUNKS PER COMPLAINT DISTRIBUTION:")
            print(f"  Minimum:  {chunks_per_stats['min']}")
            print(f"  Maximum:  {chunks_per_stats['max']}")
            print(f"  Median:   {chunks_per_stats['median']:.1f}")
            print(f"  Mean:     {chunks_per_stats['mean']:.2f}")
        
        print(f"\n📏 CHUNK LENGTH DISTRIBUTION:")
        print(f"  Very short (<100 chars):   {length_distribution['very_short']:,} "
              f"({length_distribution['very_short']/total_chunks*100:.1f}%)")
        print(f"  Short (100-299 chars):     {length_distribution['short']:,} "
              f"({length_distribution['short']/total_chunks*100:.1f}%)")
        print(f"  Medium (300-{self.chunk_size} chars): {length_distribution['medium']:,} "
              f"({length_distribution['medium']/total_chunks*100:.1f}%)")
        print(f"  Long (>500 chars):         {length_distribution['long']:,} "
              f"({length_distribution['long']/total_chunks*100:.1f}%)")
        
        # Show chunk examples
        print(f"\n🔍 CHUNK EXAMPLES:")
        if len(chunks) >= 3:
            for i in range(min(3, len(chunks))):
                chunk = chunks[i]
                chunk_text = chunk['text']
                preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                print(f"\n  Chunk {i+1} (Complaint: {chunk.get('complaint_id', 'unknown')}, "
                      f"Index: {chunk.get('chunk_index', 'N/A')}/{chunk.get('total_chunks', 'N/A')}):")
                print(f"  {preview}")
        
        print(f"\n✅ CHUNKING QUALITY ASSESSMENT:")
        
        # Quality checks
        quality_issues = []
        
        if avg_chunks_per_complaint < 1.1:
            quality_issues.append("Very few chunks per complaint - may need smaller chunk size")
        
        if length_distribution['very_short'] / total_chunks > 0.1:
            quality_issues.append("Too many very short chunks (<100 chars)")
        
        if length_distribution['long'] > 0:
            quality_issues.append(f"Some chunks exceed target size of {self.chunk_size} chars")
        
        if quality_issues:
            print("  ⚠️  Areas for improvement:")
            for issue in quality_issues:
                print(f"    • {issue}")
        else:
            print("  ✓ Good chunking results!")
        
        print("\n" + "="*70)
    
    def experiment_with_parameters(self, 
                                 sample_texts: List[str],
                                 complaint_ids: List[str] = None):
        """
        Experiment with different chunking parameters to find optimal settings
        
        Args:
            sample_texts: List of sample complaint texts to test
            complaint_ids: Optional list of complaint IDs for reference
        """
        if complaint_ids is None:
            complaint_ids = [f"sample_{i}" for i in range(len(sample_texts))]
        
        print("\n" + "="*70)
        print("CHUNKING PARAMETER EXPERIMENT")
        print("="*70)
        
        # Different parameter combinations to try
        parameter_combinations = [
            {'chunk_size': 300, 'chunk_overlap': 30},
            {'chunk_size': 400, 'chunk_overlap': 40},
            {'chunk_size': 500, 'chunk_overlap': 50},  # Default from pre-built
            {'chunk_size': 600, 'chunk_overlap': 60},
            {'chunk_size': 500, 'chunk_overlap': 100},
        ]
        
        results = []
        
        for params in parameter_combinations:
            # Create chunker with these parameters
            chunker = ComplaintChunker(
                chunk_size=params['chunk_size'],
                chunk_overlap=params['chunk_overlap']
            )
            
            # Test on sample texts
            total_chunks = 0
            chunk_lengths = []
            
            for i, text in enumerate(sample_texts):
                metadata = {'complaint_id': complaint_ids[i]}
                chunks = chunker.chunk_complaint(text, metadata)
                total_chunks += len(chunks)
                chunk_lengths.extend([len(chunk['text']) for chunk in chunks])
            
            avg_chunks_per_text = total_chunks / len(sample_texts) if sample_texts else 0
            avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
            
            # Calculate coverage (percentage of chunks in optimal 300-500 char range)
            optimal_chunks = sum(1 for length in chunk_lengths 
                               if 300 <= length <= 500)
            coverage = optimal_chunks / len(chunk_lengths) * 100 if chunk_lengths else 0
            
            results.append({
                'chunk_size': params['chunk_size'],
                'chunk_overlap': params['chunk_overlap'],
                'avg_chunks_per_text': round(avg_chunks_per_text, 2),
                'avg_chunk_length': round(avg_chunk_length),
                'total_chunks': total_chunks,
                'optimal_coverage': round(coverage, 1)
            })
        
        # Display results
        results_df = pd.DataFrame(results)
        
        print("\n📊 EXPERIMENT RESULTS:")
        print(results_df.to_string(index=False))
        
        # Recommendation
        print("\n" + "-"*70)
        print("🎯 RECOMMENDATION:")
        print("-"*70)
        
        # Find best parameters based on coverage and chunk count
        best_idx = results_df['optimal_coverage'].idxmax()
        best_params = results_df.loc[best_idx]
        
        print(f"\nRecommended parameters based on experiment:")
        print(f"  • Chunk size:    {int(best_params['chunk_size'])} characters")
        print(f"  • Chunk overlap: {int(best_params['chunk_overlap'])} characters")
        print(f"\nRationale:")
        print(f"  • Achieves {best_params['optimal_coverage']}% optimal coverage (300-500 chars)")
        print(f"  • Creates {best_params['avg_chunks_per_text']:.2f} chunks per complaint on average")
        print(f"  • Matches pre-built vector store parameters (500/50) for consistency")
        
        print("\n" + "="*70)
        
        return results_df

# Helper function to run chunking pipeline
def run_chunking_pipeline(sample_path: str = '../data/sampled/complaints_sample.csv',
                         output_path: str = '../data/sampled/complaint_chunks.json'):
    """Complete chunking pipeline"""
    import json
    import os
    
    logger.info("="*70)
    logger.info("STARTING CHUNKING PIPELINE")
    logger.info("="*70)
    
    try:
        # Load sampled data
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample data not found at {sample_path}. Run sampling first.")
        
        df = pd.read_csv(sample_path)
        logger.info(f"Loaded {len(df):,} sampled complaints")
        
        # Initialize chunker with recommended parameters
        # Using 500/50 to match pre-built vector store
        chunker = ComplaintChunker(chunk_size=500, chunk_overlap=50)
        
        # Chunk all complaints
        chunks = chunker.chunk_dataframe(df, text_column='consumer_complaint_narrative')
        
        # Save chunks to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Saved {len(chunks):,} chunks to {output_path}")
        
        # Also save as CSV for easier inspection
        csv_path = output_path.replace('.json', '.csv')
        chunks_df = pd.DataFrame(chunks)
        chunks_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"✓ Also saved as CSV: {csv_path}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Chunking pipeline failed: {e}")
        raise

if __name__ == "__main__":
    print("Testing Complaint Chunker...")
    
    # Test with sample text
    sample_text = """I am writing to complain about my credit card account with XYZ Bank. 
    On January 15th, I was charged an unauthorized fee of $50 for a service I never requested. 
    When I called customer service, they were unhelpful and refused to remove the charge. 
    This is the third time this has happened this year. I want the charge removed immediately 
    and compensation for my time dealing with this issue. The lack of proper customer service 
    is unacceptable for a financial institution of your size."""
    
    metadata = {
        'complaint_id': 'TEST_001',
        'product_category': 'Credit Cards',
        'date_received': '2024-01-20'
    }
    
    chunker = ComplaintChunker()
    chunks = chunker.chunk_complaint(sample_text, metadata)
    
    print(f"\nCreated {len(chunks)} chunks from sample text:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk['text'])} chars):")
        print(chunk['text'])