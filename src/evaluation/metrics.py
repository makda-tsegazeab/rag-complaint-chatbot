from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluate RAG pipeline performance"""
    
    def __init__(self, pipeline):
        """
        Initialize evaluator with a RAG pipeline
        
        Args:
            pipeline: RAGPipeline instance to evaluate
        """
        self.pipeline = pipeline
        logger.info("Initialized RAG Evaluator")
    
    def create_test_questions(self) -> List[Dict[str, Any]]:
        """
        Create representative test questions for evaluation
        
        Returns:
            List of test questions with metadata
        """
        test_questions = [
            {
                "question": "What are the most common complaints about credit cards?",
                "category": "Credit Cards",
                "expected_topics": ["billing", "fees", "interest rates", "unauthorized charges"],
                "difficulty": "easy"
            },
            {
                "question": "Why are customers unhappy with savings accounts?",
                "category": "Savings Accounts",
                "expected_topics": ["fees", "withdrawal", "interest", "service"],
                "difficulty": "easy"
            },
            {
                "question": "What issues do people report with personal loans?",
                "category": "Personal Loans",
                "expected_topics": ["application", "approval", "terms", "repayment"],
                "difficulty": "medium"
            },
            {
                "question": "What problems occur with money transfers?",
                "category": "Money Transfers",
                "expected_topics": ["failed transfers", "delays", "fees", "customer service"],
                "difficulty": "medium"
            },
            {
                "question": "Compare complaints between credit cards and savings accounts",
                "category": "Comparison",
                "expected_topics": ["differences", "similarities", "trends"],
                "difficulty": "hard"
            },
            {
                "question": "What are recent trends in customer complaints?",
                "category": "Trend Analysis",
                "expected_topics": ["patterns", "frequency", "emerging issues"],
                "difficulty": "hard"
            },
            {
                "question": "How do complaints differ between different companies?",
                "category": "Company Comparison",
                "expected_topics": ["company-specific", "service quality", "resolution"],
                "difficulty": "hard"
            },
            {
                "question": "What are the main billing-related complaints?",
                "category": "Specific Issue",
                "expected_topics": ["billing errors", "late fees", "statements"],
                "difficulty": "medium"
            },
            {
                "question": "How do customers describe poor customer service experiences?",
                "category": "Customer Service",
                "expected_topics": ["response time", "helpfulness", "resolution"],
                "difficulty": "medium"
            },
            {
                "question": "What fraudulent activities do customers report?",
                "category": "Fraud",
                "expected_topics": ["unauthorized transactions", "identity theft", "security"],
                "difficulty": "hard"
            }
        ]
        
        logger.info(f"Created {len(test_questions)} test questions")
        return test_questions
    
    def evaluate_question(self, 
                         test_question: Dict[str, Any],
                         save_response: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single test question
        
        Args:
            test_question: Test question dictionary
            save_response: Whether to save the response to file
            
        Returns:
            Evaluation results
        """
        question = test_question["question"]
        category = test_question["category"]
        
        logger.info(f"Evaluating question: '{question[:50]}...'")
        
        # Process the question
        start_time = datetime.now()
        response = self.pipeline.process_query(question)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract answer and sources
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        
        # Calculate evaluation metrics
        evaluation = {
            "question": question,
            "category": category,
            "difficulty": test_question.get("difficulty", "unknown"),
            "processing_time_seconds": processing_time,
            "answer_length": len(answer),
            "num_sources": len(sources),
            "source_scores": [s.get("score", 0) for s in sources],
            "has_answer": len(answer) > 20,  # Basic check for non-empty answer
            "answer_quality": self._rate_answer_quality(answer, test_question),
            "source_relevance": self._calculate_source_relevance(sources, test_question),
            "response_time_rating": self._rate_response_time(processing_time)
        }
        
        # Calculate overall score
        evaluation["overall_score"] = self._calculate_overall_score(evaluation)
        
        # Add full response if requested
        if save_response:
            response_file = f"evaluation_response_{category.lower().replace(' ', '_')}.json"
            self.pipeline.save_response(response, response_file)
            evaluation["response_file"] = response_file
        
        logger.info(f"✓ Evaluation complete - Score: {evaluation['overall_score']}/10")
        
        return evaluation
    
    def _rate_answer_quality(self, answer: str, test_question: Dict[str, Any]) -> float:
        """Rate answer quality on scale 1-10"""
        if not answer or len(answer) < 20:
            return 1.0
        
        score = 5.0  # Start with average
        
        # Check for expected topics
        expected_topics = test_question.get("expected_topics", [])
        if expected_topics:
            answer_lower = answer.lower()
            matches = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
            topic_score = (matches / len(expected_topics)) * 4  # Up to 4 points
            
            score += topic_score
        
        # Check answer length (neither too short nor too long)
        if 100 <= len(answer) <= 500:
            score += 1.0
        elif len(answer) > 500:
            score += 0.5  # Slightly long but okay
        
        # Check for structure (paragraphs, bullet points)
        if "\n" in answer or "- " in answer or "* " in answer:
            score += 0.5
        
        # Check for citations or references to sources
        if "according to" in answer.lower() or "based on" in answer.lower():
            score += 0.5
        
        # Cap at 10
        return min(score, 10.0)
    
    def _calculate_source_relevance(self, 
                                  sources: List[Dict[str, Any]], 
                                  test_question: Dict[str, Any]) -> float:
        """Calculate average source relevance score"""
        if not sources:
            return 0.0
        
        # Get average score
        scores = [s.get("score", 0) for s in sources]
        avg_score = sum(scores) / len(scores)
        
        # Convert to 0-10 scale (assuming scores are 0-1)
        return avg_score * 10
    
    def _rate_response_time(self, processing_time: float) -> float:
        """Rate response time on scale 1-10"""
        # Ideal: under 5 seconds, acceptable: under 10 seconds
        if processing_time <= 5:
            return 10.0
        elif processing_time <= 10:
            return 8.0
        elif processing_time <= 20:
            return 6.0
        elif processing_time <= 30:
            return 4.0
        else:
            return 2.0
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall score from components"""
        weights = {
            "answer_quality": 0.5,
            "source_relevance": 0.3,
            "response_time_rating": 0.2
        }
        
        overall_score = (
            evaluation["answer_quality"] * weights["answer_quality"] +
            evaluation["source_relevance"] * weights["source_relevance"] +
            evaluation["response_time_rating"] * weights["response_time_rating"]
        )
        
        return round(overall_score, 2)
    
    def run_evaluation(self, 
                      num_questions: Optional[int] = None,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation
        
        Args:
            num_questions: Number of questions to evaluate (None for all)
            save_results: Whether to save results to file
            
        Returns:
            Complete evaluation results
        """
        logger.info("="*70)
        logger.info("STARTING RAG PIPELINE EVALUATION")
        logger.info("="*70)
        
        # Get test questions
        test_questions = self.create_test_questions()
        
        if num_questions:
            test_questions = test_questions[:num_questions]
        
        logger.info(f"Evaluating {len(test_questions)} questions...")
        
        # Evaluate each question
        evaluations = []
        for i, test_question in enumerate(test_questions, 1):
            logger.info(f"  Question {i}/{len(test_questions)}: {test_question['category']}")
            
            evaluation = self.evaluate_question(test_question, save_response=True)
            evaluations.append(evaluation)
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(evaluations)
        
        # Print evaluation report
        self._print_evaluation_report(evaluations, summary)
        
        # Save results if requested
        if save_results:
            results = {
                "evaluations": evaluations,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "pipeline_info": {
                    "top_k": self.pipeline.top_k,
                    "model": self.pipeline.generator.model_name
                }
            }
            
            results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Evaluation results saved to {results_file}")
        
        logger.info("="*70)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*70)
        
        return {
            "evaluations": evaluations,
            "summary": summary
        }
    
    def _calculate_summary_statistics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluations"""
        df = pd.DataFrame(evaluations)
        
        summary = {
            "total_questions": len(evaluations),
            "average_overall_score": df["overall_score"].mean(),
            "average_answer_quality": df["answer_quality"].mean(),
            "average_source_relevance": df["source_relevance"].mean(),
            "average_response_time": df["processing_time_seconds"].mean(),
            "average_answer_length": df["answer_length"].mean(),
            "success_rate": (df["has_answer"].sum() / len(df)) * 100,
            "by_difficulty": {},
            "by_category": {}
        }
        
        # Group by difficulty
        if "difficulty" in df.columns:
            for difficulty in df["difficulty"].unique():
                diff_df = df[df["difficulty"] == difficulty]
                summary["by_difficulty"][difficulty] = {
                    "count": len(diff_df),
                    "average_score": diff_df["overall_score"].mean(),
                    "success_rate": (diff_df["has_answer"].sum() / len(diff_df)) * 100
                }
        
        # Group by category
        if "category" in df.columns:
            for category in df["category"].unique():
                cat_df = df[df["category"] == category]
                summary["by_category"][category] = {
                    "count": len(cat_df),
                    "average_score": cat_df["overall_score"].mean(),
                    "average_time": cat_df["processing_time_seconds"].mean()
                }
        
        return summary
    
    def _print_evaluation_report(self, 
                               evaluations: List[Dict[str, Any]],
                               summary: Dict[str, Any]):
        """Print comprehensive evaluation report"""
        print("\n" + "="*80)
        print("RAG PIPELINE EVALUATION REPORT")
        print("="*80)
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Total questions evaluated: {summary['total_questions']}")
        print(f"  Overall average score:     {summary['average_overall_score']:.2f}/10")
        print(f"  Answer quality:            {summary['average_answer_quality']:.2f}/10")
        print(f"  Source relevance:          {summary['average_source_relevance']:.2f}/10")
        print(f"  Average response time:     {summary['average_response_time']:.2f}s")
        print(f"  Success rate:              {summary['success_rate']:.1f}%")
        
        # By difficulty
        print(f"\n🎯 PERFORMANCE BY DIFFICULTY:")
        for difficulty, stats in summary.get("by_difficulty", {}).items():
            print(f"  {difficulty.capitalize():10} - Score: {stats['average_score']:.2f}/10, "
                  f"Success: {stats['success_rate']:.1f}% ({stats['count']} questions)")
        
        # By category
        print(f"\n📁 PERFORMANCE BY CATEGORY:")
        for category, stats in summary.get("by_category", {}).items():
            print(f"  {category:20} - Score: {stats['average_score']:.2f}/10, "
                  f"Time: {stats['average_time']:.2f}s ({stats['count']} questions)")
        
        # Detailed results table
        print(f"\n" + "="*80)
        print("DETAILED QUESTION EVALUATION")
        print("="*80)
        
        df = pd.DataFrame(evaluations)
        display_columns = ["category", "difficulty", "overall_score", 
                          "processing_time_seconds", "answer_length", "num_sources"]
        
        if all(col in df.columns for col in display_columns):
            display_df = df[display_columns].copy()
            display_df.columns = ["Category", "Difficulty", "Score", "Time (s)", "Answer Length", "Sources"]
            display_df["Score"] = display_df["Score"].round(2)
            display_df["Time (s)"] = display_df["Time (s)"].round(2)
            
            print("\n" + display_df.to_string(index=False))
        
        # Top performing questions
        print(f"\n" + "="*80)
        print("TOP PERFORMING QUESTIONS")
        print("="*80)
        
        if len(evaluations) >= 3:
            top_3 = sorted(evaluations, key=lambda x: x["overall_score"], reverse=True)[:3]
            for i, eval_item in enumerate(top_3, 1):
                print(f"\n{i}. Score: {eval_item['overall_score']}/10")
                print(f"   Question: {eval_item['question'][:60]}...")
                print(f"   Category: {eval_item['category']}")
                print(f"   Time: {eval_item['processing_time_seconds']:.2f}s")
                print(f"   Sources: {eval_item['num_sources']}")
        
        # Recommendations for improvement
        print(f"\n" + "="*80)
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print("="*80)
        
        recommendations = []
        
        if summary['average_response_time'] > 10:
            recommendations.append("• Response time is high (>10s). Consider optimizing retrieval or using a faster LLM.")
        
        if summary['average_source_relevance'] < 7:
            recommendations.append("• Source relevance could be improved. Fine-tune embedding model or adjust retrieval parameters.")
        
        if summary['success_rate'] < 80:
            recommendations.append("• Success rate is low. Improve prompt engineering or expand training data coverage.")
        
        if summary['average_answer_quality'] < 7:
            recommendations.append("• Answer quality needs improvement. Enhance prompt templates and context formatting.")
        
        if recommendations:
            for rec in recommendations:
                print(rec)
        else:
            print("✓ Pipeline performance is good! Consider these optional improvements:")
            print("  • Add more diverse test questions")
            print("  • Implement user feedback collection")
            print("  • Add answer fact-checking")
        
        print("\n" + "="*80)

# Helper function for easy evaluation
def evaluate_pipeline(pipeline=None, num_questions: int = 8):
    """Run evaluation on a pipeline"""
    if pipeline is None:
        from src.rag.pipeline import create_pipeline
        pipeline = create_pipeline()
    
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run_evaluation(num_questions=num_questions, save_results=True)
    
    return results

if __name__ == "__main__":
    print("Testing RAG Evaluator...")
    
    # Create pipeline and evaluator
    from src.rag.pipeline import create_pipeline
    pipeline = create_pipeline()
    
    evaluator = RAGEvaluator(pipeline)
    
    # Run evaluation
    results = evaluator.run_evaluation(num_questions=5, save_results=True)
    
    print("\n✅ Evaluation complete!")