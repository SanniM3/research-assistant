"""Triage agent - screens papers for relevance."""
import json
from typing import Dict, Any, List

from ..models.state import ResearchState
from ..models.paper import Paper, PaperMetadata
from .base import get_llm, create_agent_message, parse_llm_json


def triage_node(state: ResearchState) -> Dict[str, Any]:
    """
    Triage node - screens candidate papers and selects for ingestion.
    
    Responsibilities:
    - Read titles and abstracts
    - Assess relevance to research topic
    - Tag papers with metadata
    - Select papers for full-text ingestion
    """
    llm = get_llm(role="triage")
    
    state.log_action("triage", "screening_papers", {"count": len(state.candidate_papers)})
    
    # Get papers not yet triaged
    already_selected = set(state.selected_papers)
    already_ingested = set(state.kb().papers_map().keys())
    papers_to_triage = [
        p for p in state.candidate_papers 
        if p.paper_id not in already_selected and p.paper_id not in already_ingested
    ]
    
    if not papers_to_triage:
        return {"phase": "ingestion"}
    
    # Batch papers for triage (process in groups)
    batch_size = 10
    selected_ids = list(already_selected)
    
    for i in range(0, len(papers_to_triage), batch_size):
        batch = papers_to_triage[i:i + batch_size]
        
        # Create triage prompt
        papers_info = []
        for p in batch:
            papers_info.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors[:3] if p.authors else [],
                "year": p.year,
                "abstract": (p.abstract or "")[:500],
            })
        
        prompt = f"""Screen these papers for relevance to the research topic.

RESEARCH TOPIC: {state.topic}
SCOPE: {state.scope}

PAPERS TO SCREEN:
{json.dumps(papers_info, indent=2)}

For each paper, decide whether to:
- INGEST: Full-text analysis needed - highly relevant
- SKIP: Not relevant enough for full analysis

Output JSON format:
{{
    "decisions": [
        {{
            "paper_id": "id",
            "decision": "INGEST" or "SKIP",
            "rationale": "brief reason",
            "relevance_score": 0.0 to 1.0,
            "tags": {{
                "method_type": "type or null",
                "domain": "domain or null",
                "task": "task or null",
                "is_survey": true/false,
                "is_seminal": true/false
            }}
        }}
    ]
}}

Selection criteria:
1. High relevance to research questions
2. Preference for recent papers (last 3 years) and seminal works
3. Coverage of different methods/approaches
4. Include surveys and benchmark papers
5. Consider venue quality if apparent

Output ONLY valid JSON."""

        messages = create_agent_message("triage", prompt)
        response = llm.invoke(messages)
        
        triage_results = parse_llm_json(
            response.content, fallback=None, agent="triage"
        )

        if triage_results and isinstance(triage_results, dict):
            for decision in triage_results.get("decisions", []):
                paper_id = decision.get("paper_id")
                if decision.get("decision") == "INGEST":
                    selected_ids.append(paper_id)

                    paper = next((p for p in batch if p.paper_id == paper_id), None)
                    if paper:
                        tags = decision.get("tags", {})
                        paper.metadata = PaperMetadata(
                            domain=tags.get("domain"),
                            task=tags.get("task"),
                            method_type=tags.get("method_type"),
                            is_seminal=tags.get("is_seminal", False),
                        )
        else:
            state.log_action("triage", "batch_parse_error_fallback", {
                "batch_start": i,
                "batch_size": len(batch),
            })
            for p in batch:
                selected_ids.append(p.paper_id)
    
    # Prune candidate_papers to only keep those that were selected.
    # Rejected papers are dead weight in the state (ingestion only needs
    # selected ones).  This significantly reduces state serialisation size.
    selected_set = set(selected_ids)
    pruned_candidates = [
        p for p in state.candidate_papers if p.paper_id in selected_set
    ]

    return {
        "selected_papers": selected_ids,
        "candidate_papers": pruned_candidates,
        "phase": "ingestion",
    }


