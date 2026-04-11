"""Verifier agent - ensures grounding of all claims."""
import json
import re
from typing import Dict, Any, List
import uuid

from ..models.state import ResearchState
from ..models.issue import Issue, IssueSeverity, IssueCategory, IssueStatus
from .base import get_llm, create_agent_message, parse_llm_json


def verifier_node(state: ResearchState) -> Dict[str, Any]:
    """
    Verifier node - checks grounding and identifies issues.
    
    Responsibilities:
    - Check every claim has evidence
    - Flag unsupported statements
    - Identify contradictions
    - Create Issues for problems
    """
    llm = get_llm()
    
    state.log_action("verifier", "starting", {"sections": len(state.draft_sections)})
    
    issues = list(state.issues)  # Copy existing issues
    
    for section_id, content in state.draft_sections.items():
        section = state.get_section_by_id(section_id)
        section_title = section.title if section else section_id
        
        # Check for grounding issues
        section_issues = verify_section_grounding(
            section_id=section_id,
            section_title=section_title,
            content=content,
            claims=state.claims,
            chunks=state.chunks,
            papers=state.papers_ingested,
            llm=llm
        )
        
        issues.extend(section_issues)
        
        state.log_action("verifier", "section_verified", {
            "section_id": section_id,
            "issues_found": len(section_issues),
        })
    
    # Check for claim-level issues
    claim_issues = verify_claims(state.claims, state.chunks)
    issues.extend(claim_issues)
    
    # Deduplicate issues
    issues = deduplicate_issues(issues)
    
    return {
        "issues": issues,
        "phase": "verification",
    }


def verify_section_grounding(section_id: str, section_title: str, content: str,
                             claims: Dict, chunks: Dict, papers: Dict, llm) -> List[Issue]:
    """Verify grounding of a section's content."""
    issues = []
    
    # Extract citation patterns
    citation_pattern = r'\[@([^\]]+)\]'
    citations = re.findall(citation_pattern, content)
    
    # Check for factual statements without citations
    prompt = f"""Analyze this survey section for grounding issues.

SECTION: {section_title}

CONTENT:
{content[:5000]}

CITATIONS FOUND: {citations[:20] if citations else "None"}

AVAILABLE PAPERS: {list(papers.keys())[:20]}

Identify issues in JSON format:
{{
    "issues": [
        {{
            "type": "unsupported_claim|missing_citation|weak_evidence|potential_contradiction",
            "severity": "blocker|major|minor",
            "description": "Description of the issue",
            "location": "Quote or description of where in the text",
            "suggested_action": "How to fix this"
        }}
    ],
    "citation_validity": {{
        "valid_citations": ["list of valid citations"],
        "invalid_citations": ["citations that don't match known papers"]
    }},
    "overall_grounding_score": 0.0 to 1.0
}}

Check for:
1. Factual claims without any citation
2. Strong claims ("first", "best", "state-of-the-art") without evidence
3. Statistics or numbers without sources
4. Potentially contradictory statements

Output ONLY valid JSON."""

    messages = create_agent_message("verifier", prompt)
    response = llm.invoke(messages)
    
    data = parse_llm_json(response.content, fallback=None, agent="verifier")

    if data and isinstance(data, dict):
        for issue_data in data.get("issues", []):
            issue_type = issue_data.get("type", "unsupported_claim")

            category_map = {
                "unsupported_claim": IssueCategory.UNSUPPORTED_CLAIM,
                "missing_citation": IssueCategory.MISSING_CITATION,
                "weak_evidence": IssueCategory.WEAK_EVIDENCE,
                "potential_contradiction": IssueCategory.CONTRADICTION,
            }
            category = category_map.get(issue_type, IssueCategory.UNSUPPORTED_CLAIM)

            severity_map = {
                "blocker": IssueSeverity.BLOCKER,
                "major": IssueSeverity.MAJOR,
                "minor": IssueSeverity.MINOR,
            }
            severity = severity_map.get(
                issue_data.get("severity", "minor"),
                IssueSeverity.MINOR
            )

            issue = Issue(
                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                severity=severity,
                category=category,
                description=issue_data.get("description", ""),
                linked_section=section_id,
                status=IssueStatus.OPEN,
                created_by="verifier_agent",
            )
            issues.append(issue)

        for invalid in data.get("citation_validity", {}).get("invalid_citations", []):
            issue = Issue(
                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                severity=IssueSeverity.MAJOR,
                category=IssueCategory.MISSING_CITATION,
                description=f"Invalid citation reference: {invalid}",
                linked_section=section_id,
                status=IssueStatus.OPEN,
                created_by="verifier_agent",
            )
            issues.append(issue)
    else:
        issues.extend(basic_grounding_check(section_id, content, papers))
    
    return issues


def verify_claims(claims: Dict, chunks: Dict) -> List[Issue]:
    """Verify claim-level evidence."""
    issues = []
    
    for claim_id, claim in claims.items():
        if not claim.has_sufficient_evidence():
            issue = Issue(
                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                severity=IssueSeverity.MAJOR,
                category=IssueCategory.UNSUPPORTED_CLAIM,
                description=f"Claim lacks evidence: {claim.text[:100]}...",
                linked_claim_ids=[claim_id],
                status=IssueStatus.OPEN,
                created_by="verifier_agent",
            )
            issues.append(issue)
        else:
            # Verify evidence chunks exist
            for evidence in claim.evidence:
                if evidence.chunk_id not in chunks:
                    issue = Issue(
                        issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                        severity=IssueSeverity.MINOR,
                        category=IssueCategory.MISSING_CITATION,
                        description=f"Evidence chunk not found: {evidence.chunk_id}",
                        linked_claim_ids=[claim_id],
                        status=IssueStatus.OPEN,
                        created_by="verifier_agent",
                    )
                    issues.append(issue)
    
    return issues


def basic_grounding_check(section_id: str, content: str, papers: Dict) -> List[Issue]:
    """Basic rule-based grounding checks."""
    issues = []
    
    # Check for strong claims without citations
    strong_claims = [
        "state-of-the-art", "state of the art", "sota",
        "first", "novel", "best", "outperforms all",
        "solves", "guarantees", "proves",
    ]
    
    sentences = content.split(".")
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        for claim_word in strong_claims:
            if claim_word in sentence_lower:
                # Check if sentence has citation
                if "[@" not in sentence:
                    issue = Issue(
                        issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                        severity=IssueSeverity.MAJOR,
                        category=IssueCategory.UNSUPPORTED_CLAIM,
                        description=f"Strong claim '{claim_word}' without citation: {sentence[:100]}...",
                        linked_section=section_id,
                        status=IssueStatus.OPEN,
                        created_by="verifier_agent",
                    )
                    issues.append(issue)
                    break
    
    return issues


def deduplicate_issues(issues: List[Issue]) -> List[Issue]:
    """Remove duplicate issues based on description similarity."""
    seen_descriptions = set()
    unique_issues = []
    
    for issue in issues:
        # Normalize description
        normalized = issue.description.lower().strip()[:100]
        
        if normalized not in seen_descriptions:
            seen_descriptions.add(normalized)
            unique_issues.append(issue)
    
    return unique_issues
