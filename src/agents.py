from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from .tools import RESEARCH_TOOLS

# State Models
class ResearchState(BaseModel):
    question: str
    answer_format: Literal["short", "long"] = "short"
    findings: List = Field(default_factory=list)
    evaluation: str = ""
    report: str = ""
    short_answer: str = ""
    refinement_count: int = 0
    is_satisfactory: bool = False
    messages: list = Field(default_factory=list)
    depth: int = 0
    max_depth: int = 1
    is_sufficient: bool = False

    

# Node: Manager determines answer format
def manager_node(state: ResearchState) -> ResearchState:
    print(f"Manager node called")
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    response = llm.invoke([
        SystemMessage(content="You are a research manager agent that coordinates deep research tasks. Your role is to: 1. Analyze the user's question 2. Determine if a short answer or detailed report is needed. Return either 'short' or 'long' as your answer."),
        HumanMessage(content=state.question)
    ])
    print(f"Manager response: {response}")
    state.answer_format = "long" if "long" in response.content.lower() else "short"
    print(state.answer_format)
    return state

# Node: Short answer extraction
def short_answer_node(state: ResearchState) -> ResearchState:
    print(f"Short answer node called")
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    response = llm.invoke([
        SystemMessage(content="You are an expert at extracting concise, accurate answers from research findings. Your role is to: 1. Analyze the research findings 2. Extract the most relevant information 3. Formulate a clear, concise answer 4. Ensure accuracy and completeness"),
        HumanMessage(content=f"Based on these findings: {state.messages}\nExtract a concise answer to: {state.question}")
    ])
    state.short_answer = response.content
    return state

# Node: Report writer
def report_writer_node(state: ResearchState) -> ResearchState:
    print(f"Report writer node called")
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    response = llm.invoke([
        SystemMessage(content="You are a report writer that creates well-structured research reports. Your role is to: 1. Organize findings into logical sections 2. Create clear and concise summaries 3. Highlight key insights and evidence 4. Maintain academic rigor and clarity 5. Analyze the reviewer's evaluation and improve the report accordingly"),
        HumanMessage(content=f"Findings: {state.messages}.\n\n Reviewer's evaluation: {state.evaluation}")
    ])
    print(f"Report writer response: {response}")
    state.report = response.content
    return state

# Node: Report refiner
def report_refiner_node(state: ResearchState) -> ResearchState:
    print(f"Report refiner node called")
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    response = llm.invoke([
        SystemMessage(content="You are a report refiner that improves research reports. Your role is to: 1. Critically evaluate report quality as it applies to the user's question 2. Suggest specific improvements 3. Ensure clarity and coherence 4. Maintain academic standards 5. If the report has reached a high quality, return 'acceptable' only, otherwise return the suggestions for improvement"),
        HumanMessage(content=f"User's question: {state.question}\n\n Report: {state.report}")
    ])
    state.evaluation = response.content
    state.is_satisfactory = "acceptable" in response.content.lower()
    state.refinement_count += 1
    return state

# Deep research subgraph nodes

def llm_with_tools_node(state: ResearchState) -> ResearchState:
    print(f"LLM with tools node called")
    llm = ChatOpenAI(model="gpt-4-turbo-preview").bind_tools(RESEARCH_TOOLS)
    if not state.messages:
        state.messages.append(SystemMessage(content="You are a research agent. Use tools to answer the question deeply. Question: {state.question}"))
    response = llm.invoke(state.messages)
    state.messages.append(response)
    print(f"LLM with tools response: {response}")
    return state

def aggregate_findings_node(state: ResearchState) -> ResearchState:
    print(f"Aggregate findings node called")
    findings = []
    for msg in state.messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_call_outputs'):
            findings.extend(msg.tool_call_outputs)
    state.messages = findings
    return state

def review_findings_node(state: ResearchState) -> ResearchState:
    print(f"Review findings node called")
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    review_prompt = [
        SystemMessage(content="You are a research reviewer. Review the findings for a user question and decide if more research is needed. If sufficient, reply with 'sufficient'. Otherwise, suggest follow-up queries."),
        HumanMessage(content=f"User question: {state.question}\n\n Findings so far: {state.messages}")
    ]
    response = llm.invoke(review_prompt)
    state.messages.append(response)
    state.depth += 1
    if hasattr(response, "content") and "sufficient" in str(response.content).lower():
        state.is_sufficient = True
    elif state.depth >= state.max_depth:
        state.is_sufficient = True
    else:
        state.is_sufficient = False
    return state

def should_continue_or_return(state: ResearchState):
    return END if state.is_sufficient else "llm_with_tools"

def create_deep_research_subgraph():
    builder = StateGraph(ResearchState)
    builder.add_node('llm_with_tools', llm_with_tools_node)
    builder.add_node('tools', ToolNode(RESEARCH_TOOLS))
    builder.add_node('aggregate_findings', aggregate_findings_node)
    builder.add_node('review_findings', review_findings_node)
    builder.add_edge(START, 'llm_with_tools')
    builder.add_conditional_edges('llm_with_tools', tools_condition)
    builder.add_edge('tools', 'aggregate_findings')
    builder.add_edge('aggregate_findings', 'review_findings')
    builder.add_conditional_edges('review_findings', should_continue_or_return)
    # builder.add_edge('llm_with_tools', END)
    return builder.compile(checkpointer=MemorySaver())

def route_after_research(state: ResearchState) -> str:
    return "get_short_answer" if state.answer_format == "short" else "write_report"

def should_continue_refinement(state: ResearchState) -> str:
    if state.is_satisfactory or state.refinement_count >= 1:
        return END
    return "write_report"

def create_research_graph() -> StateGraph:
    deep_research = create_deep_research_subgraph()
    workflow = StateGraph(ResearchState)
    workflow.add_node("manager", manager_node)
    workflow.add_node("research", deep_research)
    workflow.add_node("get_short_answer", short_answer_node)
    workflow.add_node("write_report", report_writer_node)
    workflow.add_node("refine_report", report_refiner_node)
    workflow.add_edge(START, "manager")
    workflow.add_edge("manager", "research")
    workflow.add_conditional_edges("research", route_after_research)
    workflow.add_edge("get_short_answer", END)
    workflow.add_edge("write_report", "refine_report")
    workflow.add_conditional_edges("refine_report", should_continue_refinement)
    return workflow.compile(checkpointer=MemorySaver()) 
