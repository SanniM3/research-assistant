"""Planner agent - creates research plan with preliminary questions and scope."""
import json
from typing import Dict, Any, List
from langchain_core.messages import AIMessage

from ..models.state import (
    ResearchState, OutlineSection, AcceptanceCriteria,
    ResearchQuestion, QuestionStatus,
)
from .base import get_llm, create_agent_message, parse_llm_json


def planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Planner node - creates research scope, preliminary questions, and a
    preliminary outline.

    The outline produced here is intentionally tentative: it gives the
    search planner enough structure to generate targeted queries, but will
    be refined after research completes and we know what findings, groups,
    and comparisons actually emerged.
    """
    llm = get_llm(role="planner")

    state.log_action("planner", "creating_plan", {"topic": state.topic})

    prompt = f"""Create a research plan for the following topic:

TOPIC: {state.topic}
USER CONSTRAINTS: {state.user_constraints or "None specified"}
OUTPUT LANGUAGE: {state.output_language}

Provide your plan in the following JSON format:
{{
    "scope": "A clear 2-3 sentence definition of the research scope",
    "research_questions": [
        "Question 1 — the most important question the survey should answer",
        "Question 2 — another key question",
        "Question 3 — ..."
    ],
    "preliminary_outline": [
        {{
            "section_id": "sec_1",
            "title": "Introduction",
            "description": "What this section covers"
        }},
        {{
            "section_id": "sec_2",
            "title": "Background",
            "description": "Foundational concepts"
        }}
    ],
    "key_concepts": ["concept1", "concept2"],
    "expected_methods": ["method1", "method2"],
    "expected_datasets": ["dataset1", "dataset2"],
    "acceptance_criteria": {{
        "min_papers": 15,
        "min_answered_questions_ratio": 0.7,
        "max_open_follow_ups": 3,
        "max_open_majors": 2
    }}
}}

Guidelines:
- Research questions should be specific, answerable through literature.
- Keep them to 4-7 questions; the system will add follow-ups during research.
- The preliminary outline is a rough skeleton (5-8 sections). It WILL be
  revised after research based on actual findings.
- Acceptance criteria control when the system stops researching.

Output ONLY valid JSON, no additional text."""

    messages = create_agent_message("planner", prompt)
    response = llm.invoke(messages)

    plan = parse_llm_json(response.content, fallback=None, agent="planner")

    if plan and isinstance(plan, dict):
        questions = []
        for i, q_text in enumerate(plan.get("research_questions", [])):
            questions.append(ResearchQuestion(
                question_id=f"rq_{i+1}",
                text=q_text,
                status=QuestionStatus.OPEN,
                iteration_created=0,
            ))

        outline = []
        for i, sec_data in enumerate(plan.get("preliminary_outline", plan.get("outline", []))):
            section = OutlineSection(
                section_id=sec_data.get("section_id", f"sec_{i+1}"),
                title=sec_data.get("title", f"Section {i+1}"),
                description=sec_data.get("description", ""),
                order=i,
                required_elements=sec_data.get("required_elements", []),
                min_claims=sec_data.get("min_claims", 3),
            )
            outline.append(section)

        crit = plan.get("acceptance_criteria", {})
        criteria = AcceptanceCriteria(
            min_papers=crit.get("min_papers", 15),
            min_answered_questions_ratio=crit.get("min_answered_questions_ratio", 0.7),
            max_open_follow_ups=crit.get("max_open_follow_ups", 3),
            max_open_majors=crit.get("max_open_majors", 2),
            require_seminal_papers=crit.get("require_seminal_papers", True),
            require_recent_papers=crit.get("require_recent_papers", True),
        )

        return {
            "scope": plan.get("scope", ""),
            "research_questions": questions,
            "outline": outline,
            "outline_finalized": False,
            "acceptance_criteria": criteria,
            "phase": "search_planning",
        }

    state.log_action("planner", "json_parse_error_using_defaults", {})

    default_questions = [
        ResearchQuestion(
            question_id="rq_1",
            text=f"What are the main approaches to {state.topic}?",
            status=QuestionStatus.OPEN,
            iteration_created=0,
        )
    ]
    default_outline = create_default_outline()

    return {
        "scope": f"Research survey on: {state.topic}",
        "research_questions": default_questions,
        "outline": default_outline,
        "outline_finalized": False,
        "acceptance_criteria": AcceptanceCriteria(),
        "phase": "search_planning",
    }


def create_default_outline() -> List[OutlineSection]:
    """Create a default preliminary survey outline."""
    sections = [
        ("Introduction", "Overview and motivation", ["scope", "motivation"]),
        ("Background", "Foundational concepts", ["definitions", "formulation"]),
        ("Methods", "Survey of approaches", ["taxonomy", "methods"]),
        ("Experiments", "Evaluation and results", ["datasets", "results"]),
        ("Discussion", "Analysis and trends", ["trends", "insights"]),
        ("Conclusion", "Summary and future work", ["summary", "future"]),
    ]

    outline = []
    for i, (title, desc, elements) in enumerate(sections):
        outline.append(OutlineSection(
            section_id=f"sec_{i+1}",
            title=title,
            description=desc,
            order=i,
            required_elements=elements,
            min_claims=3,
        ))

    return outline
