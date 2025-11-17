"""
Chunk Quality Analyzer
Evaluates chunking quality before embedding
"""
from collections import Counter
from typing import List, Dict, Any
from .chunking import ProcessedChunk

class ChunkQualityAnalyzer:
    """Analyze and evaluate chunk quality"""
    
    def __init__(self, chunks: List[ProcessedChunk]):
        self.chunks = chunks
    
    def analyze_all(self) -> Dict[str, Any]:
        """Run all quality checks"""
        print("\n" + "="*70)
        print("🔍 CHUNK QUALITY ANALYSIS")
        print("="*70 + "\n")
        
        results = {
            'size_distribution': self._analyze_size_distribution(),
            'semantic_coherence': self._analyze_semantic_coherence(),
            'metadata_completeness': self._analyze_metadata(),
            'overlap_effectiveness': self._analyze_overlap(),
            'content_diversity': self._analyze_content_diversity(),
            'recommendations': []
        }
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Print report
        self._print_report(results)
        
        return results
    
    def _analyze_size_distribution(self) -> Dict[str, Any]:
        """Check if chunk sizes are in optimal range BY DOCUMENT TYPE"""
        print("📏 1. CHUNK SIZE DISTRIBUTION (Per Document Type)")
        print("-" * 70)
        
        # Define optimal ranges per document type
        optimal_ranges = {
            'pdf': (1000, 1800),      # Technical papers - larger context
            'docx': (900, 1500),     # Legal docs - standard
            'excel': (200, 600)      # Tabular - smaller
        }
        
        # Group by document type
        by_type = {}
        for chunk in self.chunks:
            doc_type = chunk.metadata['doc_type']
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(chunk)
        
        # Analyze each type separately
        type_results = {}
        overall_issues = []
        
        for doc_type, type_chunks in by_type.items():
            sizes = [c.metadata['char_count'] for c in type_chunks]
            avg_size = sum(sizes) / len(sizes)
            min_size = min(sizes)
            max_size = max(sizes)
            
            # Get optimal range for this type
            opt_min, opt_max = optimal_ranges.get(doc_type, (400, 1200))
            
            # Count chunks in ranges
            too_small = [c for c in type_chunks if c.metadata['char_count'] < opt_min * 0.5]
            too_large = [c for c in type_chunks if c.metadata['char_count'] > opt_max * 1.2]
            optimal = [c for c in type_chunks if opt_min <= c.metadata['char_count'] <= opt_max]
            
            optimal_pct = len(optimal) / len(type_chunks) * 100
            
            # Print per-type analysis
            print(f"\n  {doc_type.upper()} ({len(type_chunks)} chunks):")
            print(f"    Target range: {opt_min}-{opt_max} chars")
            print(f"    Average: {avg_size:.0f} chars")
            print(f"    Actual range: {min_size}-{max_size} chars")
            print(f"    In target range: {len(optimal)}/{len(type_chunks)} ({optimal_pct:.1f}%)")
            
            # Check for issues in this type
            type_issues = []
            if len(too_small) > len(type_chunks) * 0.1:
                type_issues.append(f"Too many small chunks ({len(too_small)})")
            if len(too_large) > len(type_chunks) * 0.1:
                type_issues.append(f"Too many large chunks ({len(too_large)})")
            if optimal_pct < 60:
                type_issues.append(f"Only {optimal_pct:.0f}% in target range")
            
            type_status = "✅ Good" if not type_issues else "⚠️  Warning"
            print(f"    Status: {type_status}")
            
            if type_issues:
                for issue in type_issues:
                    print(f"      - {issue}")
                overall_issues.extend([f"{doc_type}: {issue}" for issue in type_issues])
            
            type_results[doc_type] = {
                'avg_size': avg_size,
                'min_size': min_size,
                'max_size': max_size,
                'optimal_percentage': optimal_pct,
                'too_small': len(too_small),
                'too_large': len(too_large),
                'status': 'good' if not type_issues else 'warning'
            }
        
        # Overall assessment
        all_good = all(r['status'] == 'good' for r in type_results.values())
        status = "✅ GOOD" if all_good else "⚠️  NEEDS ATTENTION"
        
        print(f"\n  Overall Status: {status}")
        if overall_issues:
            print(f"  Issues found:")
            for issue in overall_issues:
                print(f"    - {issue}")
        print()
        
        return {
            'by_type': type_results,
            'overall_status': 'good' if all_good else 'warning',
            'issues': overall_issues
        }
    
    def _analyze_semantic_coherence(self) -> Dict[str, Any]:
        """Check if chunks break in logical places"""
        print("🧩 2. SEMANTIC COHERENCE")
        print("-" * 70)
        
        # Check for common coherence issues
        broken_sentences = 0
        broken_paragraphs = 0
        good_breaks = 0
        
        for chunk in self.chunks:
            text = chunk.text
            
            # Check if chunk starts/ends mid-sentence
            starts_lowercase = text[0].islower() if text else False
            ends_incomplete = not text.rstrip().endswith(('.', '!', '?', '\n')) if text else True
            
            if starts_lowercase or ends_incomplete:
                broken_sentences += 1
            else:
                good_breaks += 1
            
            # Check for paragraph breaks
            if '\n\n' in text:
                broken_paragraphs += 1
        
        coherence_score = good_breaks / len(self.chunks) * 100
        
        print(f"  Chunks with clean breaks: {good_breaks}/{len(self.chunks)} ({coherence_score:.1f}%)")
        print(f"  Chunks with broken sentences: {broken_sentences}")
        print(f"  Chunks with paragraph breaks: {broken_paragraphs}")
        
        # Sample problematic chunks
        problematic = [c for c in self.chunks[:10] if c.text and (c.text[0].islower() or not c.text.rstrip().endswith(('.', '!', '?', '\n')))]
        
        if problematic:
            print(f"\n  Sample Problematic Chunks:")
            for i, chunk in enumerate(problematic[:2], 1):
                preview = chunk.text[:100].replace('\n', ' ')
                print(f"    {i}. {chunk.chunk_id}: '{preview}...'")
        
        status = "✅ GOOD" if coherence_score > 70 else "⚠️  NEEDS ATTENTION"
        print(f"\n  Status: {status}")
        print()
        
        return {
            'coherence_score': coherence_score,
            'good_breaks': good_breaks,
            'broken_sentences': broken_sentences,
            'status': 'good' if coherence_score > 70 else 'warning'
        }
    
    def _analyze_metadata(self) -> Dict[str, Any]:
        """Check metadata completeness"""
        print("📋 3. METADATA COMPLETENESS")
        print("-" * 70)
        
        required_fields = ['source', 'doc_name', 'doc_type', 'chunk_index', 'char_count']
        
        missing_metadata = []
        for chunk in self.chunks:
            for field in required_fields:
                if field not in chunk.metadata:
                    missing_metadata.append((chunk.chunk_id, field))
        
        # Check for citation-useful metadata (page, section, row)
        with_citation_info = 0
        for chunk in self.chunks:
            if any(k in chunk.metadata for k in ['page', 'section', 'row']):
                with_citation_info += 1
        
        citation_percentage = with_citation_info / len(self.chunks) * 100
        
        print(f"  Required fields present: {len(missing_metadata) == 0}")
        print(f"  Chunks with citation info: {with_citation_info}/{len(self.chunks)} ({citation_percentage:.1f}%)")
        
        if missing_metadata:
            print(f"\n  ⚠️  Missing metadata:")
            for chunk_id, field in missing_metadata[:5]:
                print(f"    - {chunk_id}: missing '{field}'")
        
        status = "✅ GOOD" if not missing_metadata else "❌ ISSUES"
        print(f"\n  Status: {status}")
        print()
        
        return {
            'complete': len(missing_metadata) == 0,
            'citation_percentage': citation_percentage,
            'missing_count': len(missing_metadata),
            'status': 'good' if not missing_metadata else 'error'
        }
    
    def _analyze_overlap(self) -> Dict[str, Any]:
        """Analyze if overlap strategy is working"""
        print("🔄 4. OVERLAP EFFECTIVENESS")
        print("-" * 70)
        
        # Group chunks by source
        by_source = {}
        for chunk in self.chunks:
            source = chunk.metadata['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)
        
        # Check for repeated content between consecutive chunks
        overlap_detected = 0
        total_comparisons = 0
        
        for source, source_chunks in by_source.items():
            sorted_chunks = sorted(source_chunks, key=lambda c: c.metadata['chunk_index'])
            
            for i in range(len(sorted_chunks) - 1):
                total_comparisons += 1
                chunk1 = sorted_chunks[i].text[-100:]  # Last 100 chars
                chunk2 = sorted_chunks[i+1].text[:100]  # First 100 chars
                
                # Simple overlap check (not perfect but indicative)
                if any(word in chunk2.split() for word in chunk1.split()[-10:]):
                    overlap_detected += 1
        
        overlap_rate = (overlap_detected / total_comparisons * 100) if total_comparisons > 0 else 0
        
        print(f"  Consecutive chunks compared: {total_comparisons}")
        print(f"  Overlaps detected: {overlap_detected} ({overlap_rate:.1f}%)")
        print(f"  Expected overlap rate: ~30-50%")
        
        status = "✅ GOOD" if 20 <= overlap_rate <= 60 else "⚠️  NEEDS ATTENTION"
        print(f"\n  Status: {status}")
        if overlap_rate < 20:
            print(f"    - Overlap might be too low (context loss)")
        elif overlap_rate > 60:
            print(f"    - Overlap might be too high (redundancy)")
        print()
        
        return {
            'overlap_rate': overlap_rate,
            'comparisons': total_comparisons,
            'status': 'good' if 20 <= overlap_rate <= 60 else 'warning'
        }
    
    def _analyze_content_diversity(self) -> Dict[str, Any]:
        """Check if chunks capture diverse content"""
        print("🎨 5. CONTENT DIVERSITY")
        print("-" * 70)
        
        # Count chunks per document
        by_doc = Counter(c.metadata['doc_name'] for c in self.chunks)
        
        # Check if distribution is too skewed
        total = len(self.chunks)
        distribution = {doc: count/total*100 for doc, count in by_doc.items()}
        
        print(f"  Chunks per document:")
        for doc, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            count = by_doc[doc]
            print(f"    {doc}: {count} chunks ({percentage:.1f}%)")
        
        # Check if any document dominates
        max_percentage = max(distribution.values())
        min_percentage = min(distribution.values())
        
        issues = []
        if max_percentage > 60:
            issues.append(f"One document dominates ({max_percentage:.0f}%)")
        if min_percentage < 5:
            issues.append(f"Some documents under-represented (<5%)")
        
        status = "✅ GOOD" if not issues else "⚠️  NEEDS ATTENTION"
        print(f"\n  Status: {status}")
        if issues:
            for issue in issues:
                print(f"    - {issue}")
        print()
        
        return {
            'distribution': distribution,
            'most_represented': max(distribution, key=distribution.get),
            'least_represented': min(distribution, key=distribution.get),
            'issues': issues,
            'status': 'good' if not issues else 'warning'
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations per document type"""
        recommendations = []
        
        # Size recommendations per document type
        if 'by_type' in results['size_distribution']:
            for doc_type, type_results in results['size_distribution']['by_type'].items():
                if type_results['status'] == 'warning':
                    if type_results['optimal_percentage'] < 60:
                        config_var = {
                            'pdf': 'PDF_CHUNK_SIZE',
                            'docx': 'DOCX_CHUNK_SIZE',
                            'excel': 'EXCEL_CHUNK_SIZE'
                        }.get(doc_type, 'CHUNK_SIZE')
                        
                        recommendations.append(
                            f"⚙️  {doc_type.upper()}: Adjust {config_var} in config.py "
                            f"({type_results['optimal_percentage']:.0f}% in target range)"
                        )
        
        # Coherence recommendations
        if results['semantic_coherence']['coherence_score'] < 70:
            recommendations.append(
                "⚙️  Consider adjusting separators in RecursiveCharacterTextSplitter for cleaner breaks"
            )
        
        # Overlap recommendations
        overlap = results['overlap_effectiveness']['overlap_rate']
        if overlap < 20:
            recommendations.append(
                "⚙️  Increase CHUNK_OVERLAP in config.py to preserve more context"
            )
        elif overlap > 60:
            recommendations.append(
                "⚙️  Decrease CHUNK_OVERLAP in config.py to reduce redundancy"
            )
        
        # Diversity recommendations
        if results['content_diversity']['issues']:
            recommendations.append(
                "📊 Consider document-specific chunking parameters if one doc is too dominant"
            )
        
        # If everything looks good
        if not recommendations:
            recommendations.append("✅ Chunking quality looks good! Ready for embedding.")
        
        return recommendations
    
    def _print_report(self, results: Dict[str, Any]):
        """Print final assessment report"""
        print("="*70)
        print("📊 QUALITY ASSESSMENT SUMMARY")
        print("="*70 + "\n")
        
        # Status indicators
        if 'by_type' in results['size_distribution']:
            size_status = results['size_distribution']['overall_status']
        else:
            size_status = results['size_distribution']['status']
            
        all_statuses = [
            size_status,
            results['semantic_coherence']['status'],
            results['metadata_completeness']['status'],
            results['overlap_effectiveness']['status'],
            results['content_diversity']['status']
        ]
        
        good_count = all_statuses.count('good')
        warning_count = all_statuses.count('warning')
        error_count = all_statuses.count('error')
        
        print(f"✅ Good: {good_count}/5")
        print(f"⚠️  Warnings: {warning_count}/5")
        print(f"❌ Errors: {error_count}/5")
        
        # Overall assessment
        print("\n" + "="*70)
        if error_count > 0:
            print("⚠️  OVERALL: NEEDS FIXES before embedding")
        elif warning_count > 2:
            print("⚠️  OVERALL: ACCEPTABLE but could be improved")
        else:
            print("✅ OVERALL: GOOD QUALITY - Ready for embedding!")
        
        # Recommendations
        if results['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("="*70 + "\n")

def test_chunk_quality():
    """Test chunk quality analysis"""
    from data_loading_and_chunking_pipeline import DocumentPipeline
    
    print("\n🧪 TESTING CHUNK QUALITY ANALYSIS\n")
    
    # Process documents
    pipeline = DocumentPipeline()
    chunks = pipeline.process_documents()
    
    # Analyze quality
    analyzer = ChunkQualityAnalyzer(chunks)
    results = analyzer.analyze_all()
    
    # Additional insights
    print("\n💡 WHAT TO LOOK FOR (Per Document Type):")
    print("-" * 70)
    print("✅ Good chunking has:")
    print("  PDF (Technical papers):")
    print("    - 60%+ chunks in 800-1400 char range")
    print("    - Preserves complex technical context")
    print("  DOCX (Legal docs):")
    print("    - 60%+ chunks in 700-1200 char range")
    print("    - Maintains legal section coherence")
    print("  Excel (Tabular):")
    print("    - 60%+ chunks in 200-600 char range")
    print("    - Each row/record well-represented")
    print()
    print("  All types:")
    print("    - 70%+ chunks with clean sentence breaks")
    print("    - Complete metadata for citations")
    print("    - 30-50% overlap rate between consecutive chunks")
    print("    - Balanced distribution across documents")
    print()
    print("⚠️  Problems to watch for:")
    print("  - Chunks way outside target range for their type")
    print("  - Chunks breaking mid-sentence - confuses embeddings")
    print("  - No overlap - loses context between chunks")
    print("  - One document dominating - others under-represented")
    print("="*70)
    
    return results


if __name__ == "__main__":
    test_chunk_quality()