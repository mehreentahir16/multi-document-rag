from typing import Dict, Any
import json
import time
from datetime import datetime
from src.rag_system import RAGSystem, RAGResponse
from .test_queries import TEST_QUERIES


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.results = []
    
    def evaluate_retrieval(self, response: RAGResponse, expected_doc: str) -> Dict[str, Any]:
        """Evaluate retrieval quality"""
        metrics = {
            'num_sources': len(response.sources),
            'avg_relevance': 0.0,
            'max_relevance': 0.0,
            'min_relevance': 0.0,
            'correct_doc_retrieved': False
        }
        
        if response.sources:
            scores = [s.score for s in response.sources]
            metrics['avg_relevance'] = sum(scores) / len(scores)
            metrics['max_relevance'] = max(scores)
            metrics['min_relevance'] = min(scores)
            
            # Check if expected document was retrieved
            if expected_doc != "Multiple":
                retrieved_docs = [s.source for s in response.sources]
                metrics['correct_doc_retrieved'] = expected_doc in retrieved_docs
        
        return metrics
    
    def evaluate_answer_quality(self, response: RAGResponse) -> Dict[str, Any]:
        """Evaluate answer quality"""
        answer = response.answer
        
        metrics = {
            'answer_length': len(answer),
            'has_citations': '[Source' in answer,
            'num_citations': answer.count('[Source'),
            'acknowledges_limitation': any(phrase in answer.lower() for phrase in [
                'cannot', 'unable', 'does not', 'do not have', 'not available'
            ])
        }
        
        return metrics
    
    def run_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """Run complete evaluation"""
        print("\n" + "="*70)
        print("📊 RAG SYSTEM EVALUATION")
        print("="*70 + "\n")
        
        print(f"Running {len(TEST_QUERIES)} test queries...\n")
        
        all_results = []
        total_time = 0
        
        for i, test_query in enumerate(TEST_QUERIES, 1):
            print(f"\n{'='*70}")
            print(f"TEST {i}/{len(TEST_QUERIES)}: {test_query.category}")
            print(f"Query: {test_query.query}")
            print("="*70)
            
            # Execute query
            start_time = time.time()
            try:
                response = self.rag_system.query(test_query.query, top_k=4)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # Evaluate
                retrieval_metrics = self.evaluate_retrieval(response, test_query.expected_doc)
                answer_metrics = self.evaluate_answer_quality(response)
                
                # Store results
                result = {
                    'test_id': i,
                    'query': test_query.query,
                    'category': test_query.category,
                    'expected_doc': test_query.expected_doc,
                    'description': test_query.description,
                    'response_time': elapsed_time,
                    'retrieval_metrics': retrieval_metrics,
                    'answer_metrics': answer_metrics,
                    'answer': response.answer,
                    'sources': [
                        {
                            'source': s.source,
                            'score': s.score,
                            'citation': s.get_citation()
                        } for s in response.sources
                    ]
                }
                
                all_results.append(result)
                
                # Print summary
                print(f"\nResponse Time: {elapsed_time:.2f}s")
                print(f"Sources Retrieved: {retrieval_metrics['num_sources']}")
                print(f"Avg Relevance: {retrieval_metrics['avg_relevance']:.3f}")
                print(f"Correct Doc: {'Yes' if retrieval_metrics['correct_doc_retrieved'] else 'No'}")
                print(f"Has Citations: {'Yes' if answer_metrics['has_citations'] else 'No'}")
                print(f"\nAnswer Preview: {response.answer[:200]}...")
                
            except Exception as e:
                print(f"\nError: {e}")
                all_results.append({
                    'test_id': i,
                    'query': test_query.query,
                    'error': str(e)
                })
        
        # Calculate aggregate metrics
        successful_tests = [r for r in all_results if 'error' not in r]
        
        aggregate_metrics = {
            'total_tests': len(TEST_QUERIES),
            'successful_tests': len(successful_tests),
            'failed_tests': len(TEST_QUERIES) - len(successful_tests),
            'avg_response_time': total_time / len(successful_tests) if successful_tests else 0,
            'avg_sources_retrieved': sum(r['retrieval_metrics']['num_sources'] for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            'avg_relevance': sum(r['retrieval_metrics']['avg_relevance'] for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            'correct_doc_rate': sum(1 for r in successful_tests if r['retrieval_metrics']['correct_doc_retrieved']) / len(successful_tests) if successful_tests else 0,
            'citation_rate': sum(1 for r in successful_tests if r['answer_metrics']['has_citations']) / len(successful_tests) if successful_tests else 0,
        }
        
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'aggregate_metrics': aggregate_metrics,
            'individual_results': all_results
        }
        
        # Print summary
        self._print_summary(aggregate_metrics)
        
        # Save results
        if save_results:
            filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {filename}")
        
        return evaluation_report
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"\nTests Completed: {metrics['successful_tests']}/{metrics['total_tests']}")
        print(f"Tests Failed: {metrics['failed_tests']}")
        print(f"\nAverage Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"Average Sources Retrieved: {metrics['avg_sources_retrieved']:.1f}")
        print(f"Average Relevance Score: {metrics['avg_relevance']:.3f}")
        print(f"Correct Document Rate: {metrics['correct_doc_rate']*100:.1f}%")
        print(f"Citation Rate: {metrics['citation_rate']*100:.1f}%")
        print("="*70)
