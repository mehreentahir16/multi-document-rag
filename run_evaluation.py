from eval.rag_evaluator import RAGEvaluator

def run_full_evaluation():
    """Run complete evaluation and generate report"""
    
    print("\n RAG EVALUATION PIPELINE \n")
    
    try:
        evaluator = RAGEvaluator()
        results = evaluator.run_evaluation(save_results=True)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print("\nKey Findings:")
        
        metrics = results['aggregate_metrics']
        
        # Performance assessment
        if metrics['avg_response_time'] < 5:
            print("Response time: Excellent (< 5s)")
        elif metrics['avg_response_time'] < 10:
            print("Response time: Good (5-10s)")
        else:
            print("Response time: Needs optimization (> 10s)")
        
        if metrics['avg_relevance'] > 0.7:
            print("Retrieval quality: Excellent (> 0.7)")
        elif metrics['avg_relevance'] > 0.5:
            print("Retrieval quality: Good (0.5-0.7)")
        else:
            print("Retrieval quality: Needs improvement (< 0.5)")
        
        if metrics['correct_doc_rate'] > 0.8:
            print("Document targeting: Excellent (> 80%)")
        elif metrics['correct_doc_rate'] > 0.6:
            print("Document targeting: Good (60-80%)")
        else:
            print("Document targeting: Needs improvement (< 60%)")
        
        if metrics['citation_rate'] > 0.9:
            print("Citation usage: Excellent (> 90%)")
        elif metrics['citation_rate'] > 0.7:
            print("Citation usage: Good (70-90%)")
        else:
            print("Citation usage: Needs improvement (< 70%)")
        
        return True
        
    except Exception as e:
        print(f"\nEVALUATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_full_evaluation()
    
    if success:
        print("\nEvaluation completed successfully!")
    else:
        print("\nFix errors and try again.")