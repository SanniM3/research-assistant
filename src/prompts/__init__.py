from pathlib import Path

PROMPT_DIR = Path(__file__).parent

CREATE_RESEARCHERS_PROMPT = (PROMPT_DIR / "create_researchers.txt").read_text()
QUESTION_INSTRUCTIONS = (PROMPT_DIR / "question_instructions.txt").read_text()
ANSWER_INSTRUCTIONS = (PROMPT_DIR / "answer_instructions.txt").read_text()
SEARCH_INSTRUCTIONS = (PROMPT_DIR / "search_instructions.txt").read_text()
REPORT_WRITER_PROMPT = (PROMPT_DIR / "report_writer.txt").read_text()
