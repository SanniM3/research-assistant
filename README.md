# LangGraph Research Assistant

This project is a multi-agent research assistant built using [LangGraph](https://docs.langchain.com/langgraph/). It automates the literature review process by coordinating a team of specialized AI researchers. 

---

## Project Structure

```
repo-root/
│
├── main.py                   # CLI script to run the research assistant
├── requirements.txt          
├── samples/literature_review.md      # Output report file 
│
├── src/
│   ├── graph/
│   │   ├── __init__.py       
│   │   ├── builder.py        # Assembles the full graph
│   │   ├── nodes.py          # Contains the node functions and tools
│   │   ├── types.py          # Contains data schemas for researchers and state
│   │
│   ├── prompts/
│   │   ├── create_researchers.txt      # Prompt template to create research agents
│   │   ├── search_instructions.txt     # Prompt template to write search queries
│   │   ├── answer_instructions.txt     # Prompt template to answer questions with citations
│   │   ├── report_writer.txt           # Prompt template to compile findings into a paper
```

---

## How to Run

### 1. Install Dependencies

Ensure you're using Python 3.10+ and run:

```bash
pip install -r requirements.txt
```

You also need to set your OpenAI_API_KEY, TAVILY_API_KEY and LANGCHAIN_API_KEY in a `.env` file:

### 2. Run the Assistant

From your terminal, run:

```bash
python main.py
```

You will be prompted to enter:
- A research topic
- An initial draft or background (e.g., survey description)

Then, a team of AI researchers will be created. You can iteratively provide feedback on the team design before proceeding. Once finalized, each researcher runs their literature review process, and the findings are compiled into a final report.

---

## Output

The completed literature review is written to `samples/literature_review.md`


## To-Do

- [ ] **Add arXiv Search Retriever**  to fetch full papers (not just abstracts) for more grounding.

- [ ] **Clarification Node** that asks clarifying questions from the user if the agent is unsure about the direction of the research.

- [ ] **Memory of Prior Feedback** - persist all previous human feedback during researcher creation to avoid repeating the same mistakes in refinement.

- [ ] **Chain of Verification** - Incorporate a verification step to ensure all answers and citations in the literature review are consistent, accurate, and properly referenced.
- [ ]  **Memory**
- [ ]  **Translate tool** - Can be the llm itself


---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
