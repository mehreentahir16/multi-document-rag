from dataclasses import dataclass

@dataclass
class TestQuery:
    """Test query with metadata"""
    query: str
    category: str
    expected_doc: str
    description: str

# Test suite 
TEST_QUERIES = [
    # EU AI Act queries
    TestQuery(
        query="What are the prohibited AI practices according to the EU AI Act?",
        category="EU AI Act",
        expected_doc="EU AI Act Doc.docx",
        description="Tests retrieval of specific regulatory content and legal terminology"
    ),
    TestQuery(
        query="What are high-risk AI systems under the EU AI Act?",
        category="EU AI Act",
        expected_doc="EU AI Act Doc.docx",
        description="Tests understanding of classification and categorization in legal text"
    ),
    
    # Transformer paper queries
    TestQuery(
        query="Explain the self-attention mechanism in transformers",
        category="Transformers",
        expected_doc="Attention_is_all_you_need.pdf",
        description="Tests retrieval of technical concepts and ability to explain complex mechanisms"
    ),
    TestQuery(
        query="How do multi-head attention layers work in the transformer architecture?",
        category="Transformers",
        expected_doc="Attention_is_all_you_need.pdf",
        description="Tests the ability to understand deep technical details and architectural aspects"
    ),
    
    # DeepSeek paper queries
    TestQuery(
        query="What is the key idea presented in DeepSeek-R1?",
        category="DeepSeek",
        expected_doc="Deepseek-r1.pdf",
        description="Tests the ability to identify key contributions in research papers"
    ),
    
    # Inflation data queries
    TestQuery(
        query="What was the average inflation rate in 1950?",
        category="Inflation Data",
        expected_doc="Inflation_Calculator.xlsx",
        description="Tests retrieval of specific numerical data from tabular sources"
    ),
    TestQuery(
        query="Calculate the inflation adjusted value of $25 in 2020 as compared to 2015",
        category="Inflation Data",
        expected_doc="Inflation_Calculator.xlsx",
        description="Tests the ability to retrieve multi-year data from Excel data and perform calculation on it"
    ),
    
    # Cross-document query
    TestQuery(
        query="How might transformer-based AI systems be regulated under the EU AI Act?",
        category="Cross-document",
        expected_doc="Multiple",
        description="Tests the ability to synthesize information across multiple documents"
    ),
]