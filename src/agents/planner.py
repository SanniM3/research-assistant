"""Planner agent - creates research plan and outline."""
import json
from typing import Dict, Any, List
from langchain_core.messages import AIMessage

from ..models.state import ResearchState, OutlineSection, AcceptanceCriteria
from .base import get_llm, create_agent_message


def planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Planner node - creates research plan, outline, and acceptance criteria.
    
    Responsibilities:
    - Define research scope
    - Generate research questions
    - Create survey outline with sections
    - Set acceptance criteria
    """
    llm = get_llm()
    
    state.log_action("planner", "creating_plan", {"topic": state.topic})
    
    prompt = f"""Create a comprehensive research plan for the following topic:

TOPIC: {state.topic}
USER CONSTRAINTS: {state.user_constraints or "None specified"}
OUTPUT LANGUAGE: {state.output_language}

Provide your plan in the following JSON format:
{{
    "scope": "A clear 2-3 sentence definition of the research scope",
    "research_questions": [
        "Question 1...",
        "Question 2...",
        "Question 3..."
    ],
    "outline": [
        {{
            "section_id": "sec_1",
            "title": "Introduction",
            "description": "What this section covers",
            "required_elements": ["scope_definition", "motivation", "contributions"],
            "min_claims": 3
        }},
        {{
            "section_id": "sec_2", 
            "title": "Background",
            "description": "Foundational concepts and definitions",
            "required_elements": ["definitions", "problem_formulation"],
            "min_claims": 5
        }},
        {{
            "section_id": "sec_3",
            "title": "Taxonomy of Approaches",
            "description": "Classification of methods in the field",
            "required_elements": ["taxonomy", "category_descriptions"],
            "min_claims": 8
        }},
        {{
            "section_id": "sec_4",
            "title": "Methods and Techniques",
            "description": "Detailed review of major approaches",
            "required_elements": ["method_descriptions", "comparisons"],
            "min_claims": 10
        }},
        {{
            "section_id": "sec_5",
            "title": "Datasets and Benchmarks",
            "description": "Evaluation resources in the field",
            "required_elements": ["dataset_descriptions", "benchmark_table"],
            "min_claims": 5
        }},
        {{
            "section_id": "sec_6",
            "title": "Experimental Results and Analysis",
            "description": "Comparative analysis of approaches",
            "required_elements": ["comparison_table", "analysis"],
            "min_claims": 8
        }},
        {{
            "section_id": "sec_7",
            "title": "Discussion",
            "description": "Trends, insights, and disagreements",
            "required_elements": ["trends", "contradictions", "insights"],
            "min_claims": 5
        }},
        {{
            "section_id": "sec_8",
            "title": "Open Problems and Future Directions",
            "description": "Identified gaps and research opportunities",
            "required_elements": ["open_problems", "future_directions"],
            "min_claims": 5
        }},
        {{
            "section_id": "sec_9",
            "title": "Conclusion",
            "description": "Summary and key takeaways",
            "required_elements": ["summary", "recommendations"],
            "min_claims": 3
        }}
    ],
    "acceptance_criteria": {{
        "min_papers": 15,
        "min_claims_per_section": 3,
        "taxonomy_coverage": 0.7,
        "benchmark_coverage": 0.6,
        "require_seminal_papers": true,
        "require_recent_papers": true,
        "max_open_blockers": 0,
        "max_open_majors": 2
    }},
    "key_concepts": ["concept1", "concept2", "concept3"],
    "expected_methods": ["method1", "method2"],
    "expected_datasets": ["dataset1", "dataset2"]
}}

Customize this structure appropriately for the specific topic. Add or remove sections as needed.
Output ONLY valid JSON, no additional text."""

    messages = create_agent_message("planner", prompt)
    response = llm.invoke(messages)
    
    try:
        # Parse the JSON response
        plan = json.loads(response.content)
        
        # Create outline sections
        outline = []
        for i, sec_data in enumerate(plan.get("outline", [])):
            section = OutlineSection(
                section_id=sec_data.get("section_id", f"sec_{i+1}"),
                title=sec_data.get("title", f"Section {i+1}"),
                description=sec_data.get("description", ""),
                order=i,
                required_elements=sec_data.get("required_elements", []),
                min_claims=sec_data.get("min_claims", 3),
            )
            outline.append(section)
        
        # Create acceptance criteria
        criteria_data = plan.get("acceptance_criteria", {})
        criteria = AcceptanceCriteria(
            min_papers=criteria_data.get("min_papers", 15),
            min_claims_per_section=criteria_data.get("min_claims_per_section", 3),
            taxonomy_coverage=criteria_data.get("taxonomy_coverage", 0.7),
            benchmark_coverage=criteria_data.get("benchmark_coverage", 0.6),
            require_seminal_papers=criteria_data.get("require_seminal_papers", True),
            require_recent_papers=criteria_data.get("require_recent_papers", True),
            max_open_blockers=criteria_data.get("max_open_blockers", 0),
            max_open_majors=criteria_data.get("max_open_majors", 2),
        )
        
        return {
            "scope": plan.get("scope", ""),
            "research_questions": plan.get("research_questions", []),
            "outline": outline,
            "acceptance_criteria": criteria,
            "phase": "search_planning",
        }
        
    except json.JSONDecodeError as e:
        # Handle JSON parsing error - try to extract key info
        state.log_action("planner", "json_parse_error", {"error": str(e)})
        
        # Create minimal default outline
        default_outline = create_default_outline()
        
        return {
            "scope": f"Research survey on: {state.topic}",
            "research_questions": [f"What are the main approaches to {state.topic}?"],
            "outline": default_outline,
            "acceptance_criteria": AcceptanceCriteria(),
            "phase": "search_planning",
        }


def create_default_outline() -> List[OutlineSection]:
    """Create a default survey outline."""
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
