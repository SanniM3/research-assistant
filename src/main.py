"""Main entry point for the research assistant."""
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

from .graph.workflow import ResearchWorkflow
from .config.settings import get_settings
from .utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description="Multilingual Multi-Agent Academic Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main "transformer architectures in NLP"
  python -m src.main "quantum computing algorithms" --max-iterations 3
  python -m src.main "climate change impacts" --output report.md
        """
    )
    
    parser.add_argument(
        "topic",
        type=str,
        help="Research topic to investigate"
    )
    
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="Additional constraints or focus areas"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum research iterations (default: 5)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: report_<timestamp>.md)"
    )
    
    parser.add_argument(
        "--output-language",
        type=str,
        default="en",
        help="Output language code (default: en)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream progress updates"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger("main")
    
    logger.info(f"Starting research on: {args.topic}")
    
    # Create workflow
    workflow = ResearchWorkflow()
    
    # Set unique thread ID
    thread_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    workflow.set_thread_id(thread_id)
    
    try:
        if args.stream:
            # Run with streaming
            logger.info("Running with streaming updates...")
            final_state = None
            for state_update in workflow.run_with_streaming(
                topic=args.topic,
                constraints=args.constraints,
                max_iterations=args.max_iterations,
                output_language=args.output_language,
            ):
                # Log progress
                if isinstance(state_update, dict):
                    for node_name, node_state in state_update.items():
                        if hasattr(node_state, 'phase'):
                            logger.info(f"[{node_name}] Phase: {node_state.phase}, Iteration: {getattr(node_state, 'iteration', 0)}")
                        final_state = node_state
        else:
            # Run synchronously
            final_state = workflow.run(
                topic=args.topic,
                constraints=args.constraints,
                max_iterations=args.max_iterations,
                output_language=args.output_language,
            )
        
        # Get the final report
        if hasattr(final_state, 'final_report'):
            report = final_state.final_report
        elif isinstance(final_state, dict) and 'final_report' in final_state:
            report = final_state['final_report']
        else:
            report = "Report generation incomplete."
        
        # Determine output path
        output_path = args.output
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"report_{timestamp}.md"
        
        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Report written to: {output_path}")
        
        # Print summary
        if hasattr(final_state, 'papers_ingested'):
            papers_count = len(final_state.papers_ingested) if isinstance(final_state.papers_ingested, dict) else 0
        elif isinstance(final_state, dict):
            papers_count = len(final_state.get('papers_ingested', {}))
        else:
            papers_count = 0
            
        print(f"\n{'='*60}")
        print(f"Research Complete!")
        print(f"{'='*60}")
        print(f"Topic: {args.topic}")
        print(f"Papers reviewed: {papers_count}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        logger.warning("Research interrupted by user")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        raise


class ResearchAssistant:
    """
    High-level API for the research assistant.
    
    Usage:
        assistant = ResearchAssistant()
        result = assistant.research("transformer architectures")
        print(result.final_report)
    """
    
    def __init__(self, max_iterations: int = 5, output_language: str = "en"):
        """
        Initialize the research assistant.
        
        Args:
            max_iterations: Default maximum iterations
            output_language: Default output language
        """
        self.workflow = ResearchWorkflow()
        self.max_iterations = max_iterations
        self.output_language = output_language
        self.logger = get_logger("assistant")
    
    def research(self, topic: str, constraints: str = None, 
                 max_iterations: int = None, output_language: str = None):
        """
        Conduct research on a topic.
        
        Args:
            topic: Research topic
            constraints: Optional constraints
            max_iterations: Override default max iterations
            output_language: Override default output language
        
        Returns:
            ResearchState with completed survey
        """
        self.logger.info(f"Starting research: {topic}")
        
        # Set unique thread ID
        thread_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workflow.set_thread_id(thread_id)
        
        result = self.workflow.run(
            topic=topic,
            constraints=constraints,
            max_iterations=max_iterations or self.max_iterations,
            output_language=output_language or self.output_language,
        )
        
        self.logger.info("Research complete")
        return result
    
    def research_streaming(self, topic: str, **kwargs):
        """
        Conduct research with streaming updates.
        
        Yields progress updates during research.
        """
        thread_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workflow.set_thread_id(thread_id)
        
        for state in self.workflow.run_with_streaming(
            topic=topic,
            **kwargs
        ):
            yield state


if __name__ == "__main__":
    main()
