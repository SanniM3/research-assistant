"""Orchestrator agent - handles initialization and topic clarification."""
from typing import Dict, Any
from langchain_core.messages import AIMessage

from ..models.state import ResearchState
from .base import get_llm, create_agent_message


def orchestrator_node(state: ResearchState) -> Dict[str, Any]:
    """
    Orchestrator node - validates the topic and decides whether to proceed
    or request clarification.

    This node is the graph entry point (START -> orchestrator).  It runs
    exactly once at the beginning of a workflow invocation.  All subsequent
    phase transitions and iteration decisions are handled by the individual
    agent nodes (gap_scorer, reviewer) and the graph's conditional edges.
    """
    from ..storage.registry import derive_corpus_id

    llm = get_llm(role="orchestrator")

    state.log_action("orchestrator", "processing", {
        "phase": state.phase, "iteration": state.iteration,
    })

    if state.phase != "init":
        return {}

    corpus_id = state.corpus_id or derive_corpus_id(state.topic)

    if not state.topic or len(state.topic.strip()) < 10:
        prompt = f"""The user has provided a research topic: "{state.topic}"

Is this topic clear enough to proceed with academic research? Consider:
1. Is it specific enough to define a scope?
2. Are there obvious ambiguities that need clarification?

If the topic is clear, respond with: PROCEED
If clarification is needed, respond with: CLARIFY: [your question]"""

        messages = create_agent_message("orchestrator", prompt)
        response = llm.invoke(messages)

        if "CLARIFY:" in response.content:
            return {
                "messages": [AIMessage(
                    content=response.content.replace("CLARIFY:", "").strip()
                )],
                "phase": "clarify",
                "corpus_id": corpus_id,
            }

    return {
        "phase": "planning",
        "iteration": 0,
        "corpus_id": corpus_id,
    }
