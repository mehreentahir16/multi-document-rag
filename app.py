import time
import streamlit as st
from datetime import datetime

from src.config import Config
from src.rag_system import RAGSystem

# Page configuration
st.set_page_config(
    page_title="RAG Challenge - Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat-style interface
st.markdown("""
<style>
    /* Chat container styling */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Question bubble */
    .question-bubble {
        background-color: rgba(30, 136, 229, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    
    /* Answer bubble */
    .answer-bubble {
        background-color: rgba(100, 200, 100, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    
    /* Compact source styling */
    .source-compact {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: rgba(100, 100, 100, 0.05);
        border-radius: 5px;
    }
    
    /* Chat message */
    .chat-message {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Timestamp */
    .timestamp {
        font-size: 0.75rem;
        color: #999;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system (cached to avoid reloading)"""
    return RAGSystem()


def display_chat_message(msg_data, index):
    """Display a single chat message (question + answer + sources)"""
    query = msg_data['query']
    response = msg_data['response']
    timestamp = msg_data.get('timestamp', '')
    
    # Create a container for this message
    with st.container():
        # Question
        st.markdown(f"""
        <div class="question-bubble">
            <strong>🙋 You asked:</strong><br>
            {query}
            <div class="timestamp">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Answer
        st.markdown(f"""
        <div class="answer-bubble">
            <strong>🤖 Answer:</strong><br>
            {response.answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Sources (compact, collapsible)
        with st.expander(f"📚 View {len(response.sources)} Sources", expanded=False):
            for i, source in enumerate(response.sources, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {source.get_citation()}**")
                with col2:
                    st.markdown(f"*Score: {source.score:.3f}*")
                
                # Show preview
                preview = source.text[:200] + "..." if len(source.text) > 200 else source.text
                st.text(preview)
                
                # Full content in nested expander
                with st.expander(f"View full content #{i}"):
                    st.text(source.text)
                    st.json({k: v for k, v in source.metadata.items() if k != 'text'})
                
                st.markdown("---")


def execute_query(query_text, top_k, doc_filter):
    """Execute RAG query and add to message history"""
    if not query_text or not query_text.strip():
        return
    
    with st.spinner("🔍 Searching documents and generating answer..."):
        try:
            # Build filter
            filter_dict = None
            if doc_filter != "All Documents":
                filter_dict = {"source": doc_filter}
            
            # Query RAG system
            start_time = time.time()
            response = st.session_state.rag_system.query(
                query=query_text.strip(),
                top_k=top_k,
                filter_dict=filter_dict
            )
            elapsed_time = time.time() - start_time
            
            # Get timestamp
            timestamp = datetime.now().strftime("%I:%M %p")
            
            # Save to history
            st.session_state.messages.append({
                'query': query_text.strip(),
                'response': response,
                'time': elapsed_time,
                'timestamp': timestamp
            })
            
            # Clear any pending query
            if 'pending_query' in st.session_state:
                del st.session_state.pending_query
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 RAG Document Q&A System</div>', unsafe_allow_html=True)
    st.markdown("*Ask questions about EU AI Act, Transformers, DeepSeek-R1, and Inflation Data*")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    if 'rag_system' not in st.session_state:
        with st.spinner("🔄 Initializing RAG system..."):
            st.session_state.rag_system = initialize_rag_system()
    
    if 'example_clicked' not in st.session_state:
        st.session_state.example_clicked = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System info
        with st.expander("📊 System Info", expanded=False):
            st.code(f"""
Model: {Config.MODEL_NAME}
Embeddings: {Config.EMBEDDING_MODEL}
Index: {Config.PINECONE_INDEX_NAME}
            """)
        
        # Settings
        st.subheader("🔧 Query Settings")
        top_k = st.slider("Sources to retrieve", 1, 8, Config.TOP_K, 
                         help="Number of document chunks to retrieve")
        
        # Document filter
        st.subheader("📁 Document Filter")
        doc_filter = st.selectbox(
            "Filter by document",
            ["All Documents", "EU AI Act Doc.docx", "Attention_is_all_you_need.pdf", 
             "Deepseek-r1.pdf", "Inflation_Calculator.xlsx"],
            help="Limit search to specific document"
        )
        
        # Example queries - Individual clickable buttons
        st.subheader("💡 Example Questions")
        
        example_questions = [
            "What are prohibited AI practices in the EU AI Act?",
            "Explain the self-attention mechanism in transformers",
            "What is DeepSeek-R1's key innovation?",
            "Show me inflation data for 2020",
        ]
        
        for i, question in enumerate(example_questions):
            # Create unique key for each button
            button_key = f"example_btn_{i}"
            
            # When clicked, execute the query immediately
            if st.button(question, key=button_key, use_container_width=True):
                # Execute query directly
                success = execute_query(question, top_k, doc_filter)
                if success:
                    st.rerun()
        
        # Clear history
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.input_key = st.session_state.get('input_key', 0) + 1
            st.rerun()
    
    # Main content area
    st.markdown("---")
    
    # Chat input at the top 
    st.subheader("💬 Ask a Question")
    
    # Input form with unique key that changes on submit to avoid errors
    with st.form(key=f"query_form_{st.session_state.input_key}", clear_on_submit=True):
        query = st.text_input(
            "Your question:",
            placeholder="e.g., What are the key prohibitions in the EU AI Act?",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit_button = st.form_submit_button("🚀 Ask", use_container_width=True, type="primary")
        with col2:
            clear_button = st.form_submit_button("Clear", use_container_width=True)
    
    # Process manual query submission
    if submit_button and query.strip():
        success = execute_query(query, top_k, doc_filter)
        if success:
            st.session_state.input_key += 1
            st.rerun()
    
    if clear_button:
        st.session_state.input_key += 1
        st.rerun()
    
    # Chat history
    st.markdown("---")
    
    if st.session_state.messages:
        st.subheader("💬 Conversation")
        
        # Create chat container
        chat_container = st.container()
        
        with chat_container:
            # Display messages in reverse order (newest first)
            for idx, msg in enumerate(reversed(st.session_state.messages)):
                display_chat_message(msg, len(st.session_state.messages) - idx - 1)
    
    else:
        # Welcome message
        st.info("👋 **Welcome to the RAG Document Q&A System!**")
        st.markdown("""
        This system can answer questions about:
        - 📄 **EU AI Act** - Legal regulations for AI systems
        - 📄 **Attention is All You Need** - Technical paper explaining Transformer architecture 
        - 📄 **DeepSeek-R1** - Latest AI research paper
        - 📊 **Inflation Calculator** - Historical CPI data (1913-2022)
        
        **Try asking a question above, or click an example question in the sidebar!** 👈
        """)


if __name__ == "__main__":
    main()