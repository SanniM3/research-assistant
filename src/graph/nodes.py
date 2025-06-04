import operator
from typing import Annotated
from langgraph.constants import Send
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string

from .types import (
    ResearcherState,
    ResearchGraphState,
    ResearchState,
    SearchQuery,
    Researcher,
    ResearcherList
)
from src.prompts import (
    CREATE_RESEARCHERS_PROMPT,
    QUESTION_INSTRUCTIONS,
    ANSWER_INSTRUCTIONS,
    SEARCH_INSTRUCTIONS,
    REPORT_WRITER_PROMPT
)
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o', temperature=0)
tavily_search = TavilySearchResults(max_results=5)
# ─────────────────────────────────────────────────────────────────────────────
# TEAM CREATION NODES
# ─────────────────────────────────────────────────────────────────────────────

def create_researchers(state: ResearcherState):
    topic = state['topic']
    draft = state['draft']
    human_feedback = state.get('feedback', '')

    system_message = CREATE_RESEARCHERS_PROMPT.format(
        topic=topic,
        draft=draft,
        human_feedback=human_feedback
    )

    structured_llm = llm.with_structured_output(ResearcherList)
    response = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Create the research team")
    ])

    return {"researchers": response.researchers}


def human_feedback(state: ResearcherState):
    """Interrupt node for manual review of proposed researchers."""
    pass




# ─────────────────────────────────────────────────────────────────────────────
# LITERATURE REVIEW NODES
# ─────────────────────────────────────────────────────────────────────────────

def generate_question(state: ResearchState):
    researcher = state["researcher"]
    messages = state["messages"]
    prompt = QUESTION_INSTRUCTIONS.format(researcher=researcher)

    question = llm.invoke([SystemMessage(content=prompt)] + messages)
    return {"messages": [question]}


def search_web(state: ResearchState):
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"])
    docs = tavily_search.invoke(search_query.search_query)

    context = "\n\n---\n\n".join(
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in docs
    )
    return {"context": [context], "search_queries": [search_query]}


def search_wikipedia(state: ResearchState):
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SEARCH_INSTRUCTIONS] + state["messages"])

    docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
    context = "\n\n---\n\n".join(
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
        for doc in docs
    )
    return {"context": [context], "search_queries": [search_query]}


def generate_answer(state: ResearchState):
    researcher = state["researcher"]
    messages = state["messages"]
    context = state["context"]

    prompt = ANSWER_INSTRUCTIONS.format(researcher=researcher, context=context)
    answer = llm.invoke([SystemMessage(content=prompt)] + messages)
    return {"messages": [answer]}

def run_all_researchers(state: ResearchGraphState):
    """Kick off research cycles in parallel or return to team creation if feedback was given."""
    feedback = state.get("feedback")
    if feedback:
        return "create_researchers"

    return [
        Send("conduct_research", {
            "researcher": researcher,
            "messages": [HumanMessage(content="Start extensive literature review")]
        })
        for researcher in state["researchers"]
    ]


def save_findings(state: ResearchState):
    findings = get_buffer_string(state["messages"])
    return {"findings": [findings]}


def continue_search(state: ResearchState):
    num_searches = len(state["search_queries"])
    max_turns = state.get("max_research_turns", 2)

    if num_searches >= max_turns:
        return "save_findings"

    last_question = state["messages"][-2]
    if "Literature review completed" in last_question.content:
        return "save_findings"

    return "internal_lit_review"


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITING NODE
# ─────────────────────────────────────────────────────────────────────────────

def write_report(state: ResearchGraphState):
    topic = state["topic"]
    findings = state["findings"]

    formatted = "\n\n".join(str(f) for f in findings)
    prompt = REPORT_WRITER_PROMPT.format(topic=topic, context=formatted)

    report = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Write an in-depth literature review paper based upon these findings.")
    ])

    return {"content": report.content}
