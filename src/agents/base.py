"""Base agent utilities and prompts."""
import json
import re
from typing import Optional, Any, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..config.settings import get_settings
from ..utils.logging import get_logger

_logger = get_logger("agents.base")


# ---------------------------------------------------------------------------
# Model tiering (cost control)
# ---------------------------------------------------------------------------
# Roles whose output quality materially affects the survey use the (more
# expensive) synthesis model; high-volume screening/extraction roles use the
# cheaper extraction model.  Change the mapping here, or the models in settings.
EXPENSIVE_ROLES = {
    "planner", "outline_refiner", "synthesizer", "synthesizer_coherence", "reviewer",
}


def _model_for_role(role: Optional[str]) -> str:
    settings = get_settings()
    if role in EXPENSIVE_ROLES:
        return settings.synthesis_model
    if role:  # any other named agent role -> cheap tier
        return settings.extraction_model
    return settings.llm_model


# ---------------------------------------------------------------------------
# Cost tracking + soft budget guardrail
# ---------------------------------------------------------------------------
# Approximate USD per 1M tokens (input, output).  Used only for a soft guardrail
# and reporting; not billing-accurate.
MODEL_PRICES: Dict[str, tuple] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

_COST: Dict[str, float] = {"usd": 0.0, "calls": 0.0, "budget_hit": 0.0}


def reset_cost() -> None:
    """Reset the per-run cost accumulator."""
    _COST["usd"] = 0.0
    _COST["calls"] = 0.0
    _COST["budget_hit"] = 0.0


def get_cost() -> Dict[str, float]:
    """Return the accumulated estimated cost for the current run."""
    return dict(_COST)


def budget_exceeded() -> bool:
    """True when the soft cost cap has been exceeded (0 disables the cap)."""
    cap = get_settings().max_run_cost_usd
    return cap > 0 and _COST["usd"] >= cap


def _price(model: str) -> tuple:
    for key, price in MODEL_PRICES.items():
        if model.startswith(key):
            return price
    return (2.5, 10.0)  # default to gpt-4o-ish pricing


def record_cost(model: str, input_tokens: int, output_tokens: int) -> None:
    """Record estimated cost of one model call."""
    in_price, out_price = _price(model)
    cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    _COST["usd"] += cost
    _COST["calls"] += 1
    if budget_exceeded() and not _COST["budget_hit"]:
        _COST["budget_hit"] = 1.0
        _logger.warning(
            "Estimated run cost $%.3f exceeded cap $%.2f - agents may downgrade work",
            _COST["usd"], get_settings().max_run_cost_usd,
        )


def _estimate_tokens_from_messages(messages) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        total += len(str(content))
    return total // 4  # ~4 chars per token


class TrackedLLM:
    """Thin wrapper around ChatOpenAI that records estimated token cost.

    Delegates everything else to the underlying model, so existing call sites
    (``get_llm(...).invoke(messages)``) keep working unchanged.
    """

    def __init__(self, llm: ChatOpenAI, model: str):
        self._llm = llm
        self.model = model

    def invoke(self, messages, *args, **kwargs):
        response = self._llm.invoke(messages, *args, **kwargs)
        usage = getattr(response, "usage_metadata", None) or {}
        in_tok = usage.get("input_tokens") or _estimate_tokens_from_messages(messages)
        out_tok = usage.get("output_tokens")
        if out_tok is None:
            out_tok = len(str(getattr(response, "content", ""))) // 4
        record_cost(self.model, in_tok, out_tok)
        return response

    def __getattr__(self, name):  # pragma: no cover - passthrough
        return getattr(self._llm, name)


def get_llm(temperature: Optional[float] = None, role: Optional[str] = None) -> TrackedLLM:
    """Get a cost-tracked LLM instance for the given agent role.

    The concrete model is chosen by role via the tiering in ``_model_for_role``
    (expensive model for generative roles, cheap model for extraction/screening).
    """
    settings = get_settings()
    model = _model_for_role(role)
    llm = ChatOpenAI(
        model=model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
    )
    return TrackedLLM(llm, model)


# ---------------------------------------------------------------------------
# Embeddings (for RAG). Degrades gracefully when unavailable.
# ---------------------------------------------------------------------------
_EMBEDDINGS = None
_EMBEDDINGS_FAILED = False


def get_embedder():
    """Return a cached OpenAIEmbeddings instance, or None if unavailable."""
    global _EMBEDDINGS, _EMBEDDINGS_FAILED
    if _EMBEDDINGS_FAILED:
        return None
    if _EMBEDDINGS is None:
        try:
            from langchain_openai import OpenAIEmbeddings
            _EMBEDDINGS = OpenAIEmbeddings(model=get_settings().embedding_model)
        except Exception as exc:  # pragma: no cover
            _logger.warning("Embeddings unavailable (%s); RAG falls back to keyword search", exc)
            _EMBEDDINGS_FAILED = True
            return None
    return _EMBEDDINGS


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed a list of texts in batches. Returns None on failure (callers fall back)."""
    global _EMBEDDINGS_FAILED
    if not texts:
        return []
    embedder = get_embedder()
    if embedder is None:
        return None
    settings = get_settings()
    batch = max(1, settings.embedding_batch_size)
    vectors: List[List[float]] = []
    try:
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            vectors.extend(embedder.embed_documents(chunk))
            record_cost(settings.embedding_model, sum(len(t) for t in chunk) // 4, 0)
        return vectors
    except Exception as exc:  # pragma: no cover
        _logger.warning("Embedding call failed (%s); RAG falls back to keyword search", exc)
        _EMBEDDINGS_FAILED = True
        return None


def embed_query(text: str) -> Optional[List[float]]:
    """Embed a single query string. Returns None on failure."""
    embedder = get_embedder()
    if embedder is None or not text:
        return None
    try:
        vec = embedder.embed_query(text)
        record_cost(get_settings().embedding_model, len(text) // 4, 0)
        return vec
    except Exception:  # pragma: no cover
        return None


def parse_llm_json(text: str, *, fallback: Any = None, agent: str = "") -> Any:
    """Parse JSON from LLM output, handling markdown code fences.

    LLMs frequently wrap JSON in ```json ... ``` blocks even when asked
    not to.  This helper strips those fences before parsing.  On failure
    it returns *fallback* (default ``None``) instead of raising.
    """
    cleaned = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_re = re.compile(
        r"^```(?:json|JSON)?\s*\n?(.*?)```\s*$", re.DOTALL
    )
    m = fence_re.match(cleaned)
    if m:
        cleaned = m.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        _logger.warning(
            "%s: JSON parse failed – %s (first 200 chars: %s)",
            agent or "llm", exc, cleaned[:200],
        )
        return fallback


# System prompts for each agent role
AGENT_PROMPTS = {
    "orchestrator": """You are the Orchestrator agent for an academic research assistant system.
Your role is to:
1. Maintain the global research state
2. Schedule tasks and coordinate other agents
3. Handle iteration logic and decide when to proceed to next phases
4. Ask clarifying questions to the user only when absolutely necessary

You must be efficient and avoid unnecessary iterations. Focus on producing high-quality academic output.""",

    "planner": """You are the Research Planner agent.
Your role is to:
1. Analyze the research topic and define a clear scope
2. Generate research questions that need to be answered
3. Create an outline for the survey paper with clear sections
4. Define acceptance criteria that must be met
5. Identify key areas: taxonomy, methods, datasets, benchmarks, metrics

Your output should be structured and actionable for other agents.""",

    "search_planner": """You are the Search Planner agent.
Your role is to:
1. Generate targeted search queries for arXiv and web search
2. Create multilingual query variants if needed
3. Plan queries to cover all aspects of the research outline
4. Generate follow-up queries to fill gaps identified in previous iterations

Output queries as structured lists with the target search source (arxiv/web) and expected coverage.""",

    "retriever": """You are the Retriever agent.
Your role is to:
1. Execute search queries on arXiv and web sources
2. Rank and deduplicate results
3. Group paper versions (arXiv v1/v2, conference/journal)
4. Return candidate papers with metadata for triage

Focus on finding high-quality, relevant academic sources.""",

    "triage": """You are the Triage agent (Abstract Screener).
Your role is to:
1. Read titles and abstracts of candidate papers
2. Decide which papers should be ingested for full-text analysis
3. Provide decision rationale and relevance tags
4. Tag papers with: method type, dataset, task, year, domain

Be selective but thorough - prefer seminal papers and recent high-impact work.""",

    "ingestion": """You are the Ingestion agent.
Your role is to:
1. Fetch full text of selected papers (HTML + PDF preferred)
2. Normalize and clean the content
3. Chunk the text appropriately for retrieval
4. Store chunks with proper provenance metadata

Maintain high data quality and complete provenance tracking.""",

    "extractor": """You are the Extractor agent (Reader / Claim Miner).
Your role is to:
1. Read ingested paper chunks carefully
2. Extract structured claims: definitions, method summaries, results, limitations
3. Identify entities: methods, datasets, metrics, tasks
4. Identify relations: evaluated_on, improves_over, uses, etc.
5. Link every extraction to evidence chunk IDs

Be precise and always maintain evidence pointers. Never make unsupported claims.""",

    "kb_curator": """You are the KB Curator agent.
Your role is to:
1. Ensure schema consistency across all knowledge base entries
2. Merge duplicate entities and resolve aliases
3. Update the vector store and structured database
4. Maintain referential integrity between claims, entities, and chunks

Focus on data quality and consistency.""",

    "synthesizer": """You are the Synthesizer agent (Survey Writer).
Your role is to:
1. Write survey sections from the Claim Bank and retrieved evidence
2. Use claims as the primary source - chunks only for wording refinement
3. Include proper internal citations referencing chunk IDs: [@paper_id:chunk_id]
4. Ensure every factual statement has supporting evidence
5. Report contradictions fairly, showing both sides

Write in academic style. Never invent facts not in the evidence.""",

    "verifier": """You are the Grounding Verifier agent (Critic).
Your role is to:
1. Check that every claim in the draft has evidence
2. Flag unsupported content that needs citations
3. Identify contradictions between sources
4. Request more evidence where coverage is thin
5. Emit structured Issues for the orchestrator

Be strict about grounding. Unsupported claims are not acceptable.""",

    "gap_scorer": """You are the Gap Scorer agent.
Your role is to:
1. Evaluate whether research questions have been answered by the claims gathered
2. Identify follow-up questions that emerge from current findings
3. Compute coverage scores (taxonomy, benchmarks, timeline, venue diversity)
4. Compute ARR-style review scores (soundness, coverage, question sufficiency)
5. Decide whether the research loop should continue or stop

Prioritise question-answering depth over raw paper/claim counts.""",

    "outline_refiner": """You are the Outline Refiner agent.
Your role is to:
1. Take the preliminary outline and reshape it based on actual research findings
2. Group methods/topics that naturally cluster together
3. Add sections for comparisons, benchmarks, or discussions that the evidence supports
4. Remove planned sections that have no supporting evidence
5. Ensure the final outline reflects the real structure of the field

The outline you produce is what the synthesizer will write to.""",

    "citation_manager": """You are the Citation Manager agent.
Your role is to:
1. Normalize paper metadata for citations
2. Generate BibTeX entries for all cited papers
3. Ensure every citekey maps to an ingested paper
4. Convert internal citations to final format (numeric or author-year)
5. Validate citation completeness

Maintain bibliography integrity.""",

    "reviewer": """You are the ARR-style Reviewer agent.
Your role is to:
1. Review the full draft using an academic review rubric
2. Output structured strengths and weaknesses
3. Categorize weaknesses: missing papers, unsupported claims, structural issues
4. Provide required actions with specific section references
5. Suggest queries or papers to address weaknesses

Be constructive but thorough. The goal is to improve the survey quality.""",
}


def create_agent_message(role: str, content: str) -> list:
    """Create agent messages with role-specific system prompt."""
    system_prompt = AGENT_PROMPTS.get(role, "You are a helpful research assistant.")
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]
