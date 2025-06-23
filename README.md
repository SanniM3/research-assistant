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
git clone https://github.com/yourusername/research-assistant.git
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

## Agent Roles

### Manager Agent
- Receives user questions
- Determines output type (single answer vs. report)
- Coordinates with other agents
- Ensures quality of final output

### ToolCallingAgent
- Conducts deep research using various tools
- Generates and refines search queries
- Evaluates research quality
- Returns consolidated findings

### ReportWriterAgent
- Creates structured reports from findings
- Organizes content logically
- Highlights key insights
- Maintains academic rigor

### ReportRefinerAgent
- Critically evaluates reports
- Suggests improvements
- Ensures clarity and coherence
- Maintains quality standards


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangGraph for the graph-based agent framework
- OpenAI for the language models
- Various API providers for research tools
