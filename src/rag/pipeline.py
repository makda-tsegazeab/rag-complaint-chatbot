from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

from src.rag.retriever import RAGRetriever, create_retriever
from src.rag.generator import RAGGenerator, create_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation"""
    
    def __init__(self, 
                 retriever: Optional[RAGRetriever] = None,
                 generator: Optional[RAGGenerator] = None,
                 top_k: int = 5):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: RAGRetriever instance (will create if None)
            generator: RAGGenerator instance (will create if None)
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        
        # Initialize components
        self.retriever = retriever or create_retriever()
        self.generator = generator or create_generator()
        
        logger.info(f"Initialized RAG Pipeline with top_k={top_k}")
        logger.info(f"  Retriever: {self.retriever.__class__.__name__}")
        logger.info(f"  Generator: {self.generator.model_name}")
    
    def process_query(self, 
                     query: str,
                     filter_criteria: Optional[Dict[str, Any]] = None,
                     prompt_type: str = "rag",
                     additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query: User query
            filter_criteria: Optional filters for retrieval
            prompt_type: Type of prompt to use
            additional_params: Additional parameters for generation
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        start_time = datetime.now()
        
        logger.info(f"Processing query: '{query[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieve_start = datetime.now()
            retrieved_docs = self.retriever.retrieve(
                query=query,
                k=self.top_k,
                filter_criteria=filter_criteria
            )
            retrieve_time = (datetime.now() - retrieve_start).total_seconds()
            
            # Step 2: Generate answer
            generate_start = datetime.now()
            response = self.generator.generate_answer(
                question=query,
                retrieved_docs=retrieved_docs,
                prompt_type=prompt_type,
                additional_params=additional_params
            )
            generate_time = (datetime.now() - generate_start).total_seconds()
            
            # Step 3: Compile final response
            total_time = (datetime.now() - start_time).total_seconds()
            
            final_response = {
                "query": query,
                "answer": response.get("answer", ""),
                "retrieval_metrics": {
                    "num_documents_retrieved": len(retrieved_docs),
                    "retrieval_time_seconds": retrieve_time,
                    "top_k": self.top_k
                },
                "generation_metrics": {
                    "generation_time_seconds": generate_time,
                    "model": self.generator.model_name,
                    "prompt_type": prompt_type
                },
                "pipeline_metrics": {
                    "total_time_seconds": total_time,
                    "timestamp": datetime.now().isoformat()
                },
                "sources": []
            }
            
         # Add source information
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs[:5], 1):  # Limit to top 5
                    source = {
                        "rank": i,
                        "score": doc.get("score", 0.0),
                        "text_preview": doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text'],
                        "metadata": {
                            "product_category": doc.get('metadata', {}).get('product_category', 'Unknown'),
                            "issue": doc.get('metadata', {}).get('issue', 'Unknown'),
                            "company": doc.get('metadata', {}).get('company', 'Unknown'),
                            "date_received": doc.get('metadata', {}).get('date_received', 'Unknown')
                        }
                    }
                    final_response["sources"].append(source)
            
            logger.info(f"✓ Query processed in {total_time:.2f}s")
            logger.info(f"  Retrieved: {len(retrieved_docs)} docs")
            logger.info(f"  Answer length: {len(final_response['answer'])} chars")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            error_response = {
                "query": query,
                "answer": f"Error processing your query: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            return error_response
    
    def batch_process(self, 
                     queries: List[str],
                     filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in sequence
        
        Args:
            queries: List of queries to process
            filter_criteria: Optional filters for all queries
            
        Returns:
            List of responses
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        
        responses = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: '{query[:50]}...'")
            
            response = self.process_query(query, filter_criteria)
            responses.append(response)
            
            # Add query index to response
            response["query_index"] = i
        
        logger.info(f"✓ Batch processing complete")
        
        return responses
    
    def analyze_trends(self, 
                      product_category: str,
                      time_period: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze trends for a specific product category
        
        Args:
            product_category: Product to analyze
            time_period: Optional time period filter
            
        Returns:
            Trend analysis
        """
        logger.info(f"Analyzing trends for {product_category}")
        
        # Construct query based on parameters
        if time_period:
            query = f"What are the recent trends and issues with {product_category}?"
        else:
            query = f"What are the main issues and trends with {product_category}?"
        
        # Set filter criteria
        filter_criteria = {"product_category": product_category}
        
        # Process with summary prompt
        additional_params = {
            "product_category": product_category,
            "time_period": time_period or "all time"
        }
        
        response = self.process_query(
            query=query,
            filter_criteria=filter_criteria,
            prompt_type="summary",
            additional_params=additional_params
        )
        
        # Add trend-specific metadata
        response["analysis_type"] = "trend_analysis"
        response["product_category"] = product_category
        response["time_period"] = time_period or "all"
        
        return response
    
    def compare_products(self, 
                        product_a: str, 
                        product_b: str) -> Dict[str, Any]:
        """
        Compare complaints between two product categories
        
        Args:
            product_a: First product category
            product_b: Second product category
            
        Returns:
            Comparison analysis
        """
        logger.info(f"Comparing {product_a} vs {product_b}")
        
        # Retrieve documents for each product
        docs_a = self.retriever.retrieve(
            query=f"issues with {product_a}",
            k=3,
            filter_criteria={"product_category": product_a}
        )
        
        docs_b = self.retriever.retrieve(
            query=f"issues with {product_b}",
            k=3,
            filter_criteria={"product_category": product_b}
        )
        
        # Format contexts
        from src.rag.generator import RAGGenerator
        temp_generator = RAGGenerator()  # For formatting
        
        context_a = temp_generator.format_context(docs_a, max_chars=1500)
        context_b = temp_generator.format_context(docs_b, max_chars=1500)
        
        # Generate comparison
        additional_params = {
            "context_a": context_a,
            "context_b": context_b,
            "product_a": product_a,
            "product_b": product_b
        }
        
        response = self.process_query(
            query=f"Compare issues between {product_a} and {product_b}",
            prompt_type="comparison",
            additional_params=additional_params
        )
        
        # Add comparison metadata
        response["comparison"] = {
            "product_a": product_a,
            "product_b": product_b,
            "num_docs_a": len(docs_a),
            "num_docs_b": len(docs_b)
        }
        
        return response
    
    def evaluate_query(self, 
                      query: str,
                      expected_answer: Optional[str] = None,
                      filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a query with optional expected answer for testing
        
        Args:
            query: Test query
            expected_answer: Optional expected answer for evaluation
            filter_criteria: Optional filters
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating query: '{query}'")
        
        # Process the query
        response = self.process_query(query, filter_criteria)
        
        # Prepare evaluation
        evaluation = {
            "query": query,
            "response": response,
            "evaluation": {
                "has_answer": bool(response.get("answer") and len(response.get("answer", "")) > 10),
                "answer_length": len(response.get("answer", "")),
                "num_sources": len(response.get("sources", [])),
                "processing_time": response.get("pipeline_metrics", {}).get("total_time_seconds", 0)
            }
        }
        
        # Compare with expected answer if provided
        if expected_answer:
            # Simple keyword-based evaluation
            answer = response.get("answer", "").lower()
            expected = expected_answer.lower()
            
            # Check for key terms
            key_terms = expected_answer.split()[:10]  # First 10 words as key terms
            matches = sum(1 for term in key_terms if term.lower() in answer)
            
            evaluation["evaluation"]["expected_answer_match"] = {
                "key_terms_matched": matches,
                "total_key_terms": len(key_terms),
                "match_percentage": (matches / len(key_terms) * 100) if key_terms else 0
            }
        
        return evaluation
    
    def save_response(self, 
                     response: Dict[str, Any], 
                     filepath: str):
        """Save response to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Response saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving response: {e}")
    
    def interactive_mode(self):
        """Run interactive mode for testing"""
        print("\n" + "="*70)
        print("RAG PIPELINE INTERACTIVE MODE")
        print("="*70)
        print("\nType your questions about customer complaints.")
        print("Type 'exit' to quit, 'filter' to set filters, 'help' for commands.")
        
        current_filters = None
        
        while True:
            print("\n" + "-"*50)
            user_input = input("\nQuestion: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  exit          - Quit interactive mode")
                print("  filter        - Set filter criteria")
                print("  clear filter  - Clear current filters")
                print("  show filter   - Show current filters")
                print("  trends        - Analyze trends for a product")
                print("  compare       - Compare two products")
                print("  Any question  - Ask about customer complaints")
            
            elif user_input.lower() == 'filter':
                print("\nSet filter criteria (leave empty to skip):")
                product = input("Product category (Credit Cards/Personal Loans/etc): ").strip()
                issue = input("Issue category: ").strip()
                company = input("Company: ").strip()
                
                current_filters = {}
                if product:
                    current_filters['product_category'] = product
                if issue:
                    current_filters['issue'] = issue
                if company:
                    current_filters['company'] = company
                
                print(f"✓ Filters set: {current_filters}")
            
            elif user_input.lower() == 'clear filter':
                current_filters = None
                print("✓ Filters cleared")
            
            elif user_input.lower() == 'show filter':
                print(f"Current filters: {current_filters}")
            
            elif user_input.lower().startswith('trends'):
                parts = user_input.split()
                if len(parts) >= 2:
                    product = ' '.join(parts[1:])
                    response = self.analyze_trends(product)
                    print(f"\n📈 TREND ANALYSIS for {product}:")
                    print(f"\n{response['answer']}")
                else:
                    print("Usage: trends <product_category>")
            
            elif user_input.lower().startswith('compare'):
                parts = user_input.split()
                if len(parts) >= 3:
                    product_a = parts[1]
                    product_b = parts[2]
                    response = self.compare_products(product_a, product_b)
                    print(f"\n🔄 COMPARISON: {product_a} vs {product_b}:")
                    print(f"\n{response['answer']}")
                else:
                    print("Usage: compare <product1> <product2>")
            
            else:
                # Process as a regular query
                print(f"\nProcessing query with filters: {current_filters}")
                
                response = self.process_query(
                    query=user_input,
                    filter_criteria=current_filters
                )
                
                print(f"\n💡 ANSWER:")
                print(response['answer'])
                
                if response.get('sources'):
                    print(f"\n📋 SOURCES (top {len(response['sources'])}):")
                    for i, source in enumerate(response['sources'][:3], 1):
                        print(f"\n  Source {i} (score: {source['score']:.3f}):")
                        print(f"    Product: {source['metadata']['product_category']}")
                        print(f"    Issue: {source['metadata']['issue']}")
                        print(f"    Preview: {source['text_preview']}")
                
                print(f"\n⏱️  Processed in {response['pipeline_metrics']['total_time_seconds']:.2f}s")

# Helper function for easy initialization
def create_pipeline(top_k: int = 5) -> RAGPipeline:
    """Create a complete RAG pipeline with sensible defaults"""
    logger.info(f"Creating RAG Pipeline with top_k={top_k}")
    
    retriever = create_retriever()
    generator = create_generator()
    
    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=top_k
    )

if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    
    # Create pipeline
    pipeline = create_pipeline(top_k=5)
    
    # Test with sample queries
    test_queries = [
        "What are common credit card complaints?",
        "Tell me about savings account issues",
        "Why do customers complain about money transfers?",
        "What problems do people have with loans?"
    ]
    
    print("\n" + "="*70)
    print("RAG PIPELINE TEST")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}: {query}")
        print('='*50)
        
        response = pipeline.process_query(query)
        
        print(f"\nANSWER:")
        print(response['answer'][:500] + "..." if len(response['answer']) > 500 else response['answer'])
        
        print(f"\nMETRICS:")
        print(f"  Time: {response['pipeline_metrics']['total_time_seconds']:.2f}s")
        print(f"  Sources: {len(response['sources'])}")
        if response['sources']:
            print(f"  Top source score: {response['sources'][0]['score']:.3f}")
    
    print("\n" + "="*70)
    print("PIPELINE TEST COMPLETE")
    print("="*70)
    
    # Start interactive mode
    print("\nStarting interactive mode...")
    pipeline.interactive_mode()