import os
from dotenv import load_dotenv
from src.agents import create_research_graph, ResearchState

load_dotenv()

class ResearchAssistant:
    def __init__(self):
        self.graph = create_research_graph()
        self.config = {"configurable": {"thread_id": "1"}}
    
    def research(self, question: str) -> str:
        """
        Conduct research on a given question and return either a short answer
        or a detailed report based on the question's complexity.
        """
        # Initialize the state
        state = ResearchState(question=question)
        
        # Run the research workflow
        self.graph.invoke(state, config=self.config)
        final_state = self.graph.get_state(thread_id=self.config["configurable"]["thread_id"])
        # print(f"Final state: {final_state}")    
        # Return appropriate output based on answer format
        if final_state.answer_format == "short":
            return final_state.short_answer
        else:
            return final_state.report

def main():
    assistant = ResearchAssistant()
    questions = [
        "What is the current state of quantum computing?",  # Likely to get a long report
        # "What is the capital of France?",  # Likely to get a short answer
        # "How does climate change affect coral reefs?"  # Likely to get a long report
    ]
    
    for question in questions:
        print(f"\nResearching: {question}")
        result = assistant.research(question)
        print("\nResult:")
        print(result)
        print("\n" + "="*80)

if __name__ == "__main__":
    main()