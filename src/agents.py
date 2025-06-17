from typing import Dict, Any, List, Tuple, Literal
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from .tools import RESEARCH_TOOLS

# State Models
class ResearchState(BaseModel):
    question: str
    answer_format: Literal["short", "long"] = "short"  # Default to short format
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    current_query: str = ""
    evaluation: str = ""
    report: str = ""
    short_answer: str = ""
    refinement_count: int = 0
    is_satisfactory: bool = False

# Agent Prompts
MANAGER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research manager agent that coordinates deep research tasks.
    Your role is to:
    1. Analyze the user's question
    2. Determine if a short answer or detailed report is needed
    3. Coordinate with other agents to get the best results
    4. Ensure the final output meets the user's needs
    
    Return either "short" or "long" as your answer."""),
    ("user", "{question}")
])

TOOL_CALLING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research agent that conducts deep research using various tools.
    Your role is to:
    1. Generate effective search queries
    2. Use appropriate tools to gather information
    3. Evaluate the quality and relevance of findings
    4. Iterate until you have comprehensive results"""),
    ("user", "{question}")
])

SHORT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting concise, accurate answers from research findings.
    Your role is to:
    1. Analyze the research findings
    2. Extract the most relevant information
    3. Formulate a clear, concise answer
    4. Ensure accuracy and completeness"""),
    ("user", "Based on these findings: {findings}\nExtract a concise answer to: {question}")
])

REPORT_WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a report writer that creates well-structured research reports.
    Your role is to:
    1. Organize findings into logical sections
    2. Create clear and concise summaries
    3. Highlight key insights and evidence
    4. Maintain academic rigor and clarity"""),
    ("user", "{findings}")
])

REPORT_REFINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a report refiner that improves research reports.
    Your role is to:
    1. Critically evaluate report quality
    2. Suggest specific improvements
    3. Ensure clarity and coherence
    4. Maintain academic standards"""),
    ("user", "{report}")
])

# Agent Classes
class ManagerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            prompt=MANAGER_PROMPT,
            tools=[]
        )
        self.executor = AgentExecutor(agent=self.agent, tools=[])
    
    def determine_output_type(self, state: ResearchState) -> ResearchState:
        """Determine if a short answer or detailed report is needed"""
        response = self.executor.invoke({"question": state.question})
        state.answer_format = "long" if "long" in response["output"].lower() else "short"
        return state

class ToolCallingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            prompt=TOOL_CALLING_PROMPT,
            tools=RESEARCH_TOOLS
        )
        self.executor = AgentExecutor(agent=self.agent, tools=RESEARCH_TOOLS)
    
    def research(self, state: ResearchState) -> ResearchState:
        """Conduct research and return findings"""
        response = self.executor.invoke({"question": state.question})
        state.findings = response["findings"]
        return state

class ShortAnswerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            prompt=SHORT_ANSWER_PROMPT,
            tools=[]
        )
        self.executor = AgentExecutor(agent=self.agent, tools=[])
    
    def extract_answer(self, state: ResearchState) -> ResearchState:
        """Extract a concise answer from findings"""
        response = self.executor.invoke({
            "findings": state.findings,
            "question": state.question
        })
        state.short_answer = response["output"]
        return state

class ReportWriterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            prompt=REPORT_WRITER_PROMPT,
            tools=[]
        )
        self.executor = AgentExecutor(agent=self.agent, tools=[])
    
    def write_report(self, state: ResearchState) -> ResearchState:
        """Create a structured report from findings"""
        response = self.executor.invoke({"findings": state.findings})
        state.report = response["output"]
        return state

class ReportRefinerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            prompt=REPORT_REFINER_PROMPT,
            tools=[]
        )
        self.executor = AgentExecutor(agent=self.agent, tools=[])
    
    def refine_report(self, state: ResearchState) -> ResearchState:
        """Refine the report and determine if it's satisfactory"""
        response = self.executor.invoke({"report": state.report})
        state.report = response["output"]
        state.is_satisfactory = "satisfactory" in response["output"].lower()
        state.refinement_count += 1
        return state

# Graph Functions
def route_after_research(state: ResearchState) -> str:
    """Route to either short answer or report writer based on answer_format"""
    return "get_short_answer" if state.answer_format == "short" else "write_report"

def should_continue_refinement(state: ResearchState) -> str:
    """Determine if report refinement should continue"""
    if state.is_satisfactory or state.refinement_count >= 3:
        return "end"
    return "refine"

def create_research_graph() -> StateGraph:
    # Initialize agents
    manager = ManagerAgent()
    tool_caller = ToolCallingAgent()
    short_answer = ShortAnswerAgent()
    report_writer = ReportWriterAgent()
    report_refiner = ReportRefinerAgent()
    
    # Create graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("manager", manager.determine_output_type)
    workflow.add_node("research", tool_caller.research)
    workflow.add_node("get_short_answer", short_answer.extract_answer)
    workflow.add_node("write_report", report_writer.write_report)
    workflow.add_node("refine_report", report_refiner.refine_report)
    
    # Add edges
    workflow.add_edge("manager", "research")
    workflow.add_edge("research", route_after_research)
    workflow.add_edge("get_short_answer", "end")
    workflow.add_edge("write_report", "refine_report")
    workflow.add_edge("refine_report", should_continue_refinement)
    
    # Set entry point
    workflow.set_entry_point("manager")
    
    return workflow.compile() 