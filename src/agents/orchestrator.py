"""Orchestrator agent - coordinates the research workflow."""
from typing import Dict, Any
from langchain_core.messages import AIMessage

from ..models.state import ResearchState
from .base import get_llm, create_agent_message


def orchestrator_node(state: ResearchState) -> Dict[str, Any]:
    """
    Orchestrator node - manages global state and coordinates workflow.
    
    Responsibilities:
    - Initialize research state
    - Determine current phase
    - Decide on iteration continuation
    - Handle user clarification requests
    """
    llm = get_llm()
    
    state.log_action("orchestrator", "processing", {"phase": state.phase, "iteration": state.iteration})
    
    # Initial phase - validate topic and constraints
    if state.phase == "init":
        # Check if topic needs clarification
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
                # Need clarification from user
                return {
                    "messages": [AIMessage(content=response.content.replace("CLARIFY:", "").strip())],
                    "phase": "clarify",
                }
            
            # Ready to proceed to planning
            return {
                "phase": "planning",
                "iteration": 0,
            }
    
    # Check for stopping conditions
    if state.phase == "research" and state.iteration > 0:
        if state.should_stop():
            return {
                "phase": "synthesis",
            }
        
        # Check marginal gain
        if state.iteration >= 2:
            recent_claims = len([c for c in state.claims.values() 
                               if c.created_at.timestamp() > (state.audit_log[-1].get("timestamp", 0) if state.audit_log else 0)])
            
            if recent_claims < state.acceptance_criteria.min_claims_per_section:
                # Low marginal gain, consider stopping
                prompt = f"""Current research state:
- Iteration: {state.iteration}
- Papers ingested: {len(state.papers_ingested)}
- Total claims: {len(state.claims)}
- New claims this iteration: {recent_claims}
- Open blockers: {len(state.get_open_issues())}

Coverage scores:
{state.coverage_scores.model_dump_json(indent=2)}

Should we continue iterating or proceed to synthesis?
Respond with: CONTINUE or SYNTHESIZE"""

                messages = create_agent_message("orchestrator", prompt)
                response = llm.invoke(messages)
                
                if "SYNTHESIZE" in response.content:
                    return {"phase": "synthesis"}
    
    # Default: continue current phase
    return {}


def determine_next_phase(state: ResearchState) -> str:
    """Determine the next phase based on current state."""
    phase_order = [
        "init",
        "planning", 
        "search_planning",
        "retrieval",
        "triage",
        "ingestion",
        "extraction",
        "kb_update",
        "synthesis",
        "verification",
        "gap_scoring",
        "review",
        "revision",
        "finalize",
    ]
    
    current_idx = phase_order.index(state.phase) if state.phase in phase_order else 0
    
    # Handle loops
    if state.phase == "gap_scoring":
        if state.should_stop():
            return "review"
        else:
            return "search_planning"  # Loop back for more research
    
    if state.phase == "revision":
        if state.has_blocking_issues():
            return "search_planning"  # Need more research
        else:
            return "finalize"
    
    # Default: proceed to next phase
    if current_idx < len(phase_order) - 1:
        return phase_order[current_idx + 1]
    
    return "finalize"
