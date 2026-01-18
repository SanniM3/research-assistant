"""
Streamlit web interface for the Research Assistant.

Run with: streamlit run app.py
"""
import streamlit as st
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Must be the first Streamlit command
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
from src.graph.workflow import ResearchWorkflow
from src.models.state import ResearchState
from src.config.settings import get_settings


# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-running {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
    }
    .status-complete {
        background-color: #d4edda;
        border: 1px solid #28a745;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "research_state" not in st.session_state:
        st.session_state.research_state = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "final_report" not in st.session_state:
        st.session_state.final_report = None
    if "progress_log" not in st.session_state:
        st.session_state.progress_log = []
    if "workflow" not in st.session_state:
        st.session_state.workflow = None


def extract_state_value(state, key, default=None):
    """Safely extract a value from state (handles both dict and object)."""
    if state is None:
        return default
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def run_research(topic: str, constraints: str, max_iterations: int, use_streaming: bool = False):
    """Run the research workflow with progress updates."""
    st.session_state.is_running = True
    st.session_state.progress_log = []
    st.session_state.final_report = None
    
    # Create workflow
    workflow = ResearchWorkflow()
    thread_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    workflow.set_thread_id(thread_id)
    st.session_state.workflow = workflow
    
    # Create initial state
    initial_state = ResearchState(
        topic=topic,
        user_constraints=constraints if constraints else None,
        max_iterations=max_iterations,
        phase="init",
    )
    
    try:
        if use_streaming:
            # Run with streaming (shows progress but may be less stable)
            for state_update in workflow.graph.stream(initial_state, config=workflow.config):
                if state_update is None:
                    continue
                    
                for node_name, node_state in state_update.items():
                    # Skip if node_state is None
                    if node_state is None:
                        continue
                    
                    # Safely extract phase
                    phase = extract_state_value(node_state, 'phase', 'unknown')
                    
                    # Log progress
                    log_entry = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "node": node_name,
                        "phase": phase,
                    }
                    
                    # Safely extract metrics
                    papers_ingested = extract_state_value(node_state, 'papers_ingested', {})
                    if papers_ingested:
                        log_entry["papers"] = len(papers_ingested) if isinstance(papers_ingested, dict) else 0
                    
                    claims = extract_state_value(node_state, 'claims', {})
                    if claims:
                        log_entry["claims"] = len(claims) if isinstance(claims, dict) else 0
                    
                    st.session_state.progress_log.append(log_entry)
                    st.session_state.research_state = node_state
                    
                    # Check for final report
                    final_report = extract_state_value(node_state, 'final_report', None)
                    if final_report:
                        st.session_state.final_report = final_report
            
            # If streaming finished but no final report, try to get it from the final state
            if not st.session_state.final_report and st.session_state.research_state:
                final_report = extract_state_value(st.session_state.research_state, 'final_report', None)
                if final_report:
                    st.session_state.final_report = final_report
        else:
            # Run without streaming (more stable)
            final_state = workflow.graph.invoke(initial_state, config=workflow.config)
            
            st.session_state.research_state = final_state
            
            # Extract final report
            if isinstance(final_state, dict):
                st.session_state.final_report = final_state.get('final_report', '')
                papers = final_state.get('papers_ingested', {})
                claims = final_state.get('claims', {})
            else:
                st.session_state.final_report = getattr(final_state, 'final_report', '')
                papers = getattr(final_state, 'papers_ingested', {})
                claims = getattr(final_state, 'claims', {})
            
            # Log final state
            st.session_state.progress_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "node": "complete",
                "phase": "complete",
                "papers": len(papers) if papers else 0,
                "claims": len(claims) if claims else 0,
            })
        
        st.session_state.is_running = False
        return True
        
    except Exception as e:
        st.session_state.is_running = False
        import traceback
        st.error(f"Research failed: {str(e)}")
        st.code(traceback.format_exc())
        return False


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key status
        st.subheader("API Status")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        
        if openai_key and len(openai_key) > 10:
            st.success("✓ OpenAI API Key configured")
        else:
            st.error("✗ OpenAI API Key missing")
            st.text_input("Enter OpenAI API Key:", type="password", key="openai_input")
        
        if tavily_key and len(tavily_key) > 10:
            st.success("✓ Tavily API Key configured")
        else:
            st.warning("⚠ Tavily API Key missing (web search disabled)")
        
        st.divider()
        
        # Research settings
        st.subheader("Research Settings")
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of research iterations"
        )
        
        use_streaming = st.checkbox(
            "Enable streaming updates",
            value=False,
            help="Show real-time progress (may be less stable)"
        )
        
        st.divider()
        
        # Info
        st.subheader("About")
        st.markdown("""
        This research assistant uses multiple AI agents to:
        - Search academic databases (arXiv)
        - Analyze and extract claims
        - Synthesize survey papers
        - Ensure all citations are grounded
        
        [View Documentation](https://github.com/SanniM3/research-assistant)
        """)
        
        return max_iterations, use_streaming


def render_progress():
    """Render progress information."""
    if st.session_state.progress_log:
        st.subheader("📊 Progress")
        
        # Current phase
        latest = st.session_state.progress_log[-1]
        phase_colors = {
            "init": "🔵",
            "planning": "🟡",
            "search_planning": "🟡",
            "retrieval": "🟠",
            "triage": "🟠",
            "ingestion": "🟠",
            "extraction": "🟣",
            "kb_update": "🟣",
            "synthesis": "🔴",
            "verification": "🔴",
            "gap_scoring": "🟤",
            "review": "⚫",
            "finalize": "🟢",
            "complete": "✅",
        }
        
        phase = latest.get("phase", "unknown")
        emoji = phase_colors.get(phase, "⚪")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Phase", f"{emoji} {phase}")
        with col2:
            papers = latest.get("papers", 0)
            st.metric("Papers Ingested", papers)
        with col3:
            claims = latest.get("claims", 0)
            st.metric("Claims Extracted", claims)
        
        # Progress log
        with st.expander("View detailed log", expanded=False):
            for entry in reversed(st.session_state.progress_log[-20:]):
                st.text(f"[{entry['time']}] {entry['node']}: {entry.get('phase', '')}")


def render_results():
    """Render research results."""
    if st.session_state.final_report:
        st.subheader("📄 Research Report")
        
        # Download button
        col1, col2 = st.columns([3, 1])
        with col2:
            st.download_button(
                label="📥 Download Report",
                data=st.session_state.final_report,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        # Display report
        st.markdown(st.session_state.final_report)
        
    elif st.session_state.research_state and not st.session_state.is_running:
        st.info("Research completed but no report was generated. Check the logs for details.")


def main():
    """Main application."""
    init_session_state()
    
    # Header
    st.title("📚 Academic Research Assistant")
    st.markdown("*AI-powered literature review with grounded citations*")
    
    # Sidebar
    max_iterations, use_streaming = render_sidebar()
    
    # Main content
    tab1, tab2 = st.tabs(["🔬 New Research", "📖 Results"])
    
    with tab1:
        st.subheader("Start New Research")
        
        # Input form
        with st.form("research_form"):
            topic = st.text_input(
                "Research Topic",
                placeholder="e.g., Transformer architectures for natural language processing",
                help="Enter the topic you want to research"
            )
            
            constraints = st.text_area(
                "Additional Constraints (optional)",
                placeholder="e.g., Focus on efficiency improvements, include papers from 2020-2024",
                help="Any specific requirements or focus areas"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submitted = st.form_submit_button(
                    "🚀 Start Research",
                    disabled=st.session_state.is_running,
                    type="primary"
                )
        
        # Run research
        if submitted and topic:
            with st.spinner("Running research... This may take several minutes."):
                progress_placeholder = st.empty()
                
                # Run in a way that allows UI updates
                success = run_research(topic, constraints, max_iterations, use_streaming)
                
                if success:
                    st.success("✅ Research completed!")
                    st.balloons()
        
        # Show progress while running
        if st.session_state.is_running:
            st.warning("🔄 Research in progress...")
            render_progress()
            time.sleep(1)
            st.rerun()
        elif st.session_state.progress_log:
            render_progress()
    
    with tab2:
        render_results()
        
        if not st.session_state.final_report and not st.session_state.is_running:
            st.info("👈 Start a new research project to see results here.")


if __name__ == "__main__":
    main()
