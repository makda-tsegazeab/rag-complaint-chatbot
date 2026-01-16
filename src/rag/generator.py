from typing import List, Dict, Any, Optional
import logging
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGGenerator:
    """Generate answers using retrieved context and LLM"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 use_huggingface: bool = True,
                 temperature: float = 0.7,
                 max_new_tokens: int = 500):
        """
        Initialize RAG generator
        
        Args:
            model_name: Name of the LLM to use
            use_huggingface: Whether to use HuggingFace models (True) or OpenAI API (False)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.use_huggingface = use_huggingface
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"Initializing RAG Generator...")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Max tokens: {max_new_tokens}")
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize prompts
        self.prompts = self._initialize_prompts()
        
        logger.info("✓ RAG Generator initialized successfully")
    
    def _initialize_llm(self):
        """Initialize the language model"""
        if not self.use_huggingface:
            # For OpenAI or other API-based models
            try:
                from langchain.llms import OpenAI
                import os
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found, falling back to HuggingFace")
                    self.use_huggingface = True
                    return self._initialize_huggingface_llm()
                
                logger.info("Using OpenAI GPT model")
                return OpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens
                )
            except ImportError:
                logger.warning("OpenAI not available, falling back to HuggingFace")
                self.use_huggingface = True
                return self._initialize_huggingface_llm()
        else:
            # Use HuggingFace models
            return self._initialize_huggingface_llm()
    
    def _initialize_huggingface_llm(self):
        """Initialize a HuggingFace language model"""
        logger.info(f"Loading HuggingFace model: {self.model_name}")
        
        try:
            # Try to use a smaller, faster model for testing
            # You can change this to any model from HuggingFace
            if self.model_name == "microsoft/DialoGPT-medium":
                # This is a smaller model good for testing
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                # Create text generation pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Wrap in LangChain
                llm = HuggingFacePipeline(pipeline=pipe)
                
            else:
                # For other models, use a generic approach
                pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True
                )
                llm = HuggingFacePipeline(pipeline=pipe)
            
            logger.info(f"✓ HuggingFace model loaded: {self.model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            logger.info("Falling back to a mock LLM for testing")
            return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """Create a mock LLM for testing when real models fail"""
        from langchain.llms.fake import FakeListLLM
        
        logger.warning("Using mock LLM for testing. Install transformers/torch for real model.")
        
        # Mock responses for testing
        responses = [
            "Based on the complaints, customers are experiencing billing issues with credit cards.",
            "The main issue appears to be unauthorized charges and poor customer service.",
            "Customers report problems with money transfers not going through.",
            "Loan applications are being rejected without proper explanation.",
            "Savings accounts have issues with unexpected fees and withdrawal limits."
        ]
        
        return FakeListLLM(responses=responses)
    
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize prompt templates for different types of queries"""
        
        # Main RAG prompt template
        rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful financial analyst assistant for CrediTrust Financial. 
Your task is to answer questions about customer complaints using ONLY the provided context.
If the context doesn't contain relevant information to answer the question, say "I don't have enough information based on the available complaints."

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided complaint excerpts
2. Be concise and factual
3. Summarize patterns and trends you see
4. Mention specific product categories when relevant
5. Do not make up information not in the context

CONTEXT (customer complaint excerpts):
{context}

QUESTION: {question}

ANALYSIS (based only on above context):"""
        )
        
        # Summarization prompt for trend analysis
        summary_prompt = PromptTemplate(
            input_variables=["context", "time_period", "product_category"],
            template="""You are a financial analyst summarizing customer complaint trends.

Based on the following complaint excerpts from {time_period} about {product_category}, 
identify the main issues, their frequency, and any patterns.

COMPLAINT EXCERPTS:
{context}

TREND ANALYSIS SUMMARY:"""
        )
        
        # Comparison prompt for different products
        comparison_prompt = PromptTemplate(
            input_variables=["context_a", "context_b", "product_a", "product_b"],
            template="""Compare customer complaints between {product_a} and {product_b}.

{product_a} COMPLAINTS:
{context_a}

{product_b} COMPLAINTS:
{context_b}

COMPARISON ANALYSIS (similarities and differences):"""
        )
        
        # Simple Q&A prompt
        simple_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
        )
        
        return {
            "rag": rag_prompt,
            "summary": summary_prompt,
            "comparison": comparison_prompt,
            "simple": simple_prompt
        }
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]], max_chars: int = 3000) -> str:
        """
        Format retrieved documents into a context string
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            max_chars: Maximum characters for context
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant complaints found."
        
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Format each document
            text = doc['text']
            metadata = doc.get('metadata', {})
            
            # Add metadata info
            product = metadata.get('product_category', 'Unknown product')
            issue = metadata.get('issue', 'Unknown issue')
            
            formatted_doc = f"[Document {i} - {product} - {issue}]: {text}"
            
            # Check if adding this would exceed max length
            if total_chars + len(formatted_doc) > max_chars:
                # Add partial if we have space
                remaining = max_chars - total_chars - 100  # Leave room for ellipsis
                if remaining > 100:  # Only add if we have meaningful content
                    formatted_doc = formatted_doc[:remaining] + "..."
                    context_parts.append(formatted_doc)
                break
            
            context_parts.append(formatted_doc)
            total_chars += len(formatted_doc)
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, 
                       question: str, 
                       retrieved_docs: List[Dict[str, Any]],
                       prompt_type: str = "rag",
                       additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an answer using retrieved context
        
        Args:
            question: User question
            retrieved_docs: Retrieved documents from vector store
            prompt_type: Type of prompt to use ("rag", "summary", "comparison", "simple")
            additional_params: Additional parameters for specialized prompts
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating answer for question: '{question[:50]}...'")
        logger.info(f"Using {len(retrieved_docs)} retrieved documents")
        logger.info(f"Prompt type: {prompt_type}")
        
        # Select appropriate prompt
        if prompt_type not in self.prompts:
            logger.warning(f"Unknown prompt type '{prompt_type}', using 'rag'")
            prompt_type = "rag"
        
        prompt_template = self.prompts[prompt_type]
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Prepare inputs based on prompt type
        if prompt_type == "rag":
            inputs = {
                "context": context,
                "question": question
            }
        elif prompt_type == "summary":
            inputs = {
                "context": context,
                "time_period": additional_params.get("time_period", "recent") if additional_params else "recent",
                "product_category": additional_params.get("product_category", "various products") if additional_params else "various products"
            }
        elif prompt_type == "comparison":
            # This requires splitting context
            if additional_params and "context_a" in additional_params and "context_b" in additional_params:
                inputs = {
                    "context_a": additional_params["context_a"],
                    "context_b": additional_params["context_b"],
                    "product_a": additional_params.get("product_a", "Product A"),
                    "product_b": additional_params.get("product_b", "Product B")
                }
            else:
                # Fallback to regular RAG
                logger.warning("Missing context splits for comparison, using regular RAG")
                inputs = {"context": context, "question": question}
        else:  # simple
            inputs = {
                "context": context,
                "question": question
            }
        
        try:
            # Create chain and generate
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            answer = chain.run(**inputs)
            
            # Clean up answer
            answer = answer.strip()
            
            # Prepare response
            response = {
                "answer": answer,
                "question": question,
                "num_sources": len(retrieved_docs),
                "prompt_type": prompt_type,
                "context_length": len(context),
                "model": self.model_name
            }
            
            # Add source information
            if retrieved_docs:
                sources = []
                for doc in retrieved_docs[:3]:  # Include top 3 sources
                    source_info = {
                        "text_preview": doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text'],
                        "product": doc.get('metadata', {}).get('product_category', 'Unknown'),
                        "issue": doc.get('metadata', {}).get('issue', 'Unknown'),
                        "score": doc.get('score', 0.0)
                    }
                    sources.append(source_info)
                response["sources"] = sources
            
            logger.info(f"✓ Answer generated ({len(answer)} characters)")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            
            # Return error response
            return {
                "answer": f"I encountered an error while generating an answer: {str(e)}",
                "question": question,
                "num_sources": len(retrieved_docs),
                "error": str(e),
                "model": self.model_name
            }
    
    def test_generation(self, 
                       retriever = None,
                       test_questions: Optional[List[str]] = None):
        """Test the generation system with sample questions"""
        
        if test_questions is None:
            test_questions = [
                "What are the main issues with credit cards?",
                "Tell me about problems with savings accounts",
                "Why are customers complaining about money transfers?",
                "What issues do people have with loan applications?"
            ]
        
        print("\n" + "="*70)
        print("GENERATION SYSTEM TEST")
        print("="*70)
        
        print(f"\n🤖 Model: {self.model_name}")
        print(f"📝 Test questions: {len(test_questions)}")
        
        # Create a mock retriever if none provided
        if retriever is None:
            from src.rag.retriever import create_retriever
            try:
                retriever = create_retriever()
            except:
                print("⚠️  Could not create retriever, using mock retrieval")
                # Create mock retrieved docs
                mock_docs = [
                    {
                        'text': 'Customer complains about unauthorized credit card charges.',
                        'metadata': {'product_category': 'Credit Cards', 'issue': 'Unauthorized charges'},
                        'score': 0.85
                    },
                    {
                        'text': 'Issues with credit card billing and late fees.',
                        'metadata': {'product_category': 'Credit Cards', 'issue': 'Billing problems'},
                        'score': 0.78
                    }
                ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*50}")
            print(f"QUESTION {i}: {question}")
            print('='*50)
            
            try:
                # Retrieve relevant documents
                if retriever and hasattr(retriever, 'retrieve'):
                    retrieved_docs = retriever.retrieve(question, k=3)
                else:
                    retrieved_docs = mock_docs
                
                print(f"📚 Retrieved {len(retrieved_docs)} documents")
                
                # Generate answer
                response = self.generate_answer(question, retrieved_docs)
                
                # Display answer
                print(f"\n💡 ANSWER:")
                print(response['answer'])
                
                # Display sources
                if 'sources' in response:
                    print(f"\n📋 SOURCES (top {len(response['sources'])}):")
                    for j, source in enumerate(response['sources'], 1):
                        print(f"\n  Source {j}:")
                        print(f"    Product: {source['product']}")
                        print(f"    Issue: {source['issue']}")
                        print(f"    Score: {source['score']:.3f}")
                        print(f"    Text: {source['text_preview']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "="*70)
        print("GENERATION TEST COMPLETE")
        print("="*70)

# Helper function for easy initialization
def create_generator(model_name: str = None,
                    use_huggingface: bool = True) -> RAGGenerator:
    """Create a generator with sensible defaults"""
    
    if model_name is None:
        # Use a smaller model for testing if available
        try:
            # Try to load a small model
            model_name = "microsoft/DialoGPT-medium"
        except:
            # Fallback to any available model
            model_name = "gpt2"  # Very small model
    
    logger.info(f"Creating generator with model: {model_name}")
    
    return RAGGenerator(
        model_name=model_name,
        use_huggingface=use_huggingface,
        temperature=0.7,
        max_new_tokens=300
    )

if __name__ == "__main__":
    print("Testing RAG Generator...")
    
    # Create generator
    generator = create_generator()
    
    # Test generation
    generator.test_generation()