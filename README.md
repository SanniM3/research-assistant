# Multi-Agent Research Assistant

A multi-agent system built with LangGraph for conducting deep research across various domains. The system uses a hierarchical manager-worker architecture to coordinate research tasks and generate comprehensive reports.

## Features

- **Hierarchical Agent Architecture**
  - Manager Agent (Orchestrator)
  - ToolCallingAgent (Research Worker)
  - ReportWriterAgent
  - ReportRefinerAgent

- **Comprehensive Research Tools**
  - ArxivTool for academic papers
  - PubMedTool for medical literature
  - TavilyTool for web search
  - VisualQATool for image analysis
  - FileReaderTool for PDFs and web pages

- **Workflow**
  - Automatic determination of output type (single answer vs. detailed report)
  - Report refinement loop
  - Quality control mechanisms

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SanniM3/research-assistant.git
cd research-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
ENTREZ_EMAIL=your_email@example.com
```

## Usage

```bash
python run.py
```

## Project Structure

```
research-assistant/
├── src/
│   ├── __init__.py
│   ├── main.py          # Main application entry point
│   ├── agents.py        # Agent definitions and graph
│   └── tools.py         # Tools implementation
├── run.py               
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
└── .env               # Environment variables (create this)
```

## To-do
1. Manager should understand input and request clarification/limit scope for user (add human interrupt node here)
2. Manager should combine question and human feedback to define subquestions that need to be answered.
3. Generate and applicable search query. Then iterate over results (giving `process logs` to the user) and 
4. Then at each aggregate findings node, verify that all initial subquestions have been answered and use then to generate a summary relative to the initial question.
5. The review findings looks for missing answers to initial subquestions, it also looks for follow up questions based on the findings. If there are none, return `research completed`. Consider adding paper review instructions to this prompt (modify maybe ACL instructions for survey papers), then set a score threshold which is what we use to review findings. Alternatively, I can make the report writing a new subgraph that has review and the interative process.
6. [Maybe start with this] - While reading each paper, add the paper to a knowledge base, (I can still do summarisation to find out research gaps or additional searches), but during report writing, actual details are cited from the proper knowledge base to avoid hallucination. Alternatively, to avoid keeping track of too much context, while reading each paper and search result, add them to a knowledge base, then get use that to answer the user query. Then identify gaps from this answer to create another search. Then repeat. That way, we may not need to keep track of previous searches and papers (maybe?)
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangGraph for the graph-based agent framework
- OpenAI for the language models
- Various API providers for research tools
