# Multilingual Multi-Agent Academic Research Assistant

**LangGraph + LangChain | Iterative literature review → grounded, correctly cited survey papers | Dynamic KB + RAG**

## 1) Purpose

Build a multilingual, multi-agent system that mimics the academic literature review process and produces **survey-style academic reports** with **strict grounding and traceable citations**. The system iteratively searches, ingests, extracts, synthesizes, critiques, and revises until it meets explicit completion criteria.

## 2) Goals and non-goals

### Goals

* **Survey-quality synthesis**: taxonomy, comparisons, trends, gaps, future directions.
* **Grounded writing**: every factual claim must trace to ingested evidence (chunk-level provenance).
* **Dynamic KB**: during research, construct and evolve a knowledge base of:

  * raw chunks for RAG
  * structured “Claim Bank” + entities/relations for synthesis and verification
* **Iterative gap-filling loop** driven by coverage scoring and reviewer feedback.
* **Multilingual retrieval**: search and ingest sources across languages; normalize structured knowledge into a canonical language for reasoning and writing (default: English, configurable).
* **Auditability**: reproducible trail of queries, decisions, and citations.

### Non-goals (initially)

* Perfect PDF table parsing in all cases (we provide a strategy, but not guaranteed).
* Fully automated peer-review correctness (reviewer is a tool for improvement, not formal validation).
* Replacing human judgment for final submission readiness.

## 3) System overview

### High-level pipeline

1. **Scope**: clarify (if needed), define research scope, outline, acceptance criteria.
2. **Search**: generate targeted queries (web + arXiv), retrieve, deduplicate, rank.
3. **Triage**: abstract screening → choose full-text ingestion.
4. **Ingest**: fetch HTML + PDF (preferred), chunk, store with provenance.
5. **Extract**: produce structured claims/entities/relations + evidence pointers.
6. **Synthesize**: write section drafts from claim bank + RAG chunks.
7. **Verify**: enforce grounding; flag missing citations, contradictions, thin coverage.
8. **Gap score**: compute coverage metrics; plan next iteration tasks.
9. **Stop or iterate**: repeat 2–8 until completion thresholds met.
10. **Assemble paper**: compile sections, tables, bib, limitations, future work.
11. **ARR-style review**: reviewer rubric generates actionable critiques.
12. **Revision loop**: convert critiques → new retrieval/extraction tasks → revise draft.
13. **Finalize**: formatting, citation style, export.

## 4) Core design principles (must-follow)

1. **No phantom citations**: citations can only reference ingested sources.
2. **Provenance everywhere**: every chunk and claim carries stable IDs and source anchors.
3. **Write from claims**: synthesizer uses structured Claim Bank first; raw chunks only to refine wording or add context.
4. **Verifier controls the loop**: iteration continues until grounding + coverage criteria are satisfied.
5. **Separation of concerns**: retrieval, extraction, synthesis, verification, and bibliography management are distinct roles.



## 5) Agents and responsibilities

### A. Orchestrator / Manager

* Maintains global state, schedules tasks, handles iteration logic.
* Decides when to ask the user clarifying questions (only when required).

**Inputs**: user topic, current state
**Outputs**: scope, outline, tasks, stop/iterate decisions

### B. Planner (Research Director)

* Produces research plan: subquestions, taxonomy targets, search strategy, success metrics.
* Produces “acceptance criteria” checklist to enable stopping.

### C. Search Planner

* Builds tool-specific query sets (web + arXiv), including multilingual variants.
* Produces query templates for gap-driven iterations.

### D. Retriever (Web + arXiv)

* Executes searches; returns ranked candidates.
* Deduplicates by DOI/arXiv ID/title similarity; groups versions (arXiv v1/v2, conference/journal).

### E. Triage / Abstract Screener

* Reads titles/abstracts and decides whether to ingest full text.
* Must output decision + rationale + tags (method, dataset, task, year, domain).

### F. Ingestion Agent

* Fetches **HTML + PDF** (preferred) and normalizes content.
* Chunks text and stores in chunk store + vector store with provenance.

### G. Extractor (Reader / Claim Miner)

* Extracts structured knowledge:

  * claims (definitions, results, limitations)
  * entities (methods, datasets, metrics)
  * relations (evaluated_on, improves_over, uses, assumes)
* Links each extracted item to evidence chunk IDs.

### H. KB Curator

* Ensures schema consistency, merges duplicates, resolves aliases.
* Updates vector store and structured DB.

### I. Synthesizer (Survey Writer)

* Writes survey sections from claim bank and retrieved evidence.
* Must output drafts with internal citations referencing chunk IDs.

### J. Grounding Verifier (Critic)

* Enforces: each claim has evidence; identifies unsupported content.
* Flags contradictions; requests more evidence.
* Emits structured “Issues” for the orchestrator.

### K. Gap Scorer

* Computes coverage scores vs outline acceptance criteria:

  * taxonomy completeness
  * benchmark coverage
  * timeline coverage (seminal + recent)
  * venue/source diversity
* Emits prioritized next tasks.

### L. Citation Manager

* Normalizes metadata; generates BibTeX/CSL entries.
* Ensures every citekey maps to an ingested paper ID.

### M. ARR-style Reviewer

* Reviews full draft using rubric; outputs strengths, weaknesses, and required changes.
* Produces actionable retrieval/extraction tasks for missing citations or thin areas.



## 6) Data model (knowledge base)

### 6.1 Paper record

**Paper**

* `paper_id` (stable internal ID; prefer DOI or arXiv ID)
* `title`, `authors`, `year`, `venue` (if known)
* `doi`, `arxiv_id`, `url_list`
* `language`
* `version_group_id` (to group arXiv versions / extensions)
* `retrieved_at` (timestamp)
* `license/usage_notes` (optional)
* `metadata_confidence` (low/medium/high)

### 6.2 Chunk record (grounding unit)

**Chunk**

* `chunk_id` (stable, deterministic if possible via hash)
* `paper_id`
* `source_type`: `pdf_text | arxiv_html | web_html`
* `section_path`: e.g., `Introduction > Related Work`
* `page_span` (if from PDF)
* `paragraph_span` (or DOM path for HTML)
* `text`
* `hash` (dedup)
* `created_at`

> **Rule:** citations in the paper ultimately point to `paper_id` and optionally `chunk_id` anchors (internally). Externally, citations render to paper-level references.

### 6.3 Claim Bank (structured assertions)

**Claim**

* `claim_id`
* `type`: `definition | method_summary | empirical_result | theoretical_result | limitation | comparison | open_problem`
* `text` (canonical language)
* `normalized_form` (optional: templates like “X improves Y on Z by Δ”)
* `entities`: list of entity IDs involved
* `evidence`: list of `(chunk_id, quote_span_start, quote_span_end)` or `(chunk_id, snippet_hash)`
* `confidence`: `low | medium | high`
* `notes`: evaluation context, assumptions
* `extracted_by` (agent/version)

### 6.4 Entities + relations (optional but recommended)

**Entity**

* `entity_id`
* `type`: `method | dataset | metric | task | domain | benchmark | framework`
* `name`, `aliases`
* `description` (grounded where possible)
* `evidence_chunks`

**Relation**

* `relation_id`
* `subject_entity_id`
* `predicate`: `evaluated_on | improves_over | uses | assumes | similar_to | contradicts`
* `object_entity_id`
* `evidence_chunks`

### 6.5 Issues (drives iteration)

**Issue**

* `issue_id`
* `severity`: `blocker | major | minor`
* `category`: `missing_citation | unsupported_claim | contradiction | thin_coverage | taxonomy_gap | benchmark_gap`
* `description`
* `linked_section`
* `suggested_queries`
* `status`: `open | in_progress | resolved`



## 7) Retrieval and ingestion strategy

### 7.1 Tooling

* **arXiv search**: query + filters (category, date range) → list papers
* **Web search**: for surveys, seminal works, datasets, benchmarks, non-arXiv venues, blogs/docs *only when appropriate*
* Optional: semantic scholar / crossref for metadata (if available)

### 7.2 Dedup and versioning

* Prefer DOI match; else arXiv ID; else normalized title similarity.
* Group:

  * arXiv versions
  * conference → journal extension
* Maintain `version_group_id` to avoid double-counting.

### 7.3 Full-text ingestion policy

* Abstract triage selects candidates for full ingestion.
* Ingest **HTML + PDF** when possible.
* If PDF extraction is poor, fall back to HTML; mark chunk quality.

### 7.4 Chunking

* Chunk by section boundaries, then paragraph-ish windows.
* Include headings in chunk metadata for better retrieval.
* Avoid mixing unrelated sections in one chunk.



## 8) Synthesis and grounding

### 8.1 “Write from claims” workflow

1. For each outline section:

   * retrieve relevant **Claim** objects
   * retrieve top supporting chunks per claim (for wording + context)
2. Writer produces section draft with internal citations referencing claim evidence.

### 8.2 Citation rules

* Every factual statement maps to ≥1 evidence chunk.
* Strong claims (“SOTA”, “first”, “best”) require:

  * multiple sources, or
  * explicit qualifiers (“in X setting”, “as reported by…”).
* If evidence is missing: writer must say “not found in current sources” and create an Issue.

### 8.3 Contradictions

* If contradictory claims exist, the survey must:

  * report both sides
  * explain plausible reasons (dataset, metric, setup, assumptions)
  * avoid “averaging” unless sources justify.



## 9) Iteration logic (LangGraph)

### 9.1 State schema (conceptual)

* `topic`
* `user_constraints` (optional)
* `scope`
* `outline`
* `acceptance_criteria`
* `iteration`
* `queries_run[]`
* `candidate_papers[]`
* `selected_papers[]`
* `papers_ingested[]`
* `chunks[]` (or references)
* `claims[]` (or references)
* `entities[]`, `relations[]`
* `draft_sections{section: text}`
* `issues[]`
* `coverage_scores`
* `bib_entries`

### 9.2 Graph stages (recommended nodes)

1. ClarifyScope (conditional)
2. PlanOutline
3. SearchPlanner
4. Retrieve
5. TriageAbstracts
6. IngestFullText
7. ExtractToClaims
8. KBUpdate
9. SynthesizeSection(s)
10. GroundingVerifier
11. GapScorer
12. StopOrIterate (conditional edge)
13. AssembleSurvey
14. ARRReviewer
15. RevisionPlanner
16. RevisionLoop → back to Retrieve/Ingest/Extract/Synthesize as needed
17. Finalize

### 9.3 Stop conditions (must be explicit)

Stop the research loop when ALL are true:

* All outline sections meet minimum grounded content:

  * ≥K claims per section (configurable)
  * and ≥1 evidence chunk per claim
* Open Issues:

  * 0 blockers
  * majors below threshold
* Coverage:

  * taxonomy coverage above threshold
  * benchmark coverage above threshold
  * timeline coverage includes seminal + recent
* Marginal gain:

  * last iteration added <X new unique claims OR no new high-priority issues resolved by searching



## 10) ARR-style reviewer rubric (outputs must be actionable)

Reviewer outputs:

* **Strengths** (e.g., comprehensive taxonomy, clear comparisons)
* **Weaknesses** categorized:

  * missing seminal papers
  * missing benchmarks/datasets
  * unsupported claims / weak evidence
  * unclear taxonomy/structure
  * missing limitations/failure modes
  * missing future directions grounded in cited gaps
* **Required actions**: each with:

  * section
  * what to add/fix
  * suggested queries or paper targets
  * severity

RevisionPlanner converts these into tasks and routes back to retrieval/extraction/synthesis.



## 11) Quality gates and acceptance tests

### 11.1 Grounding tests

* **No-citation detector**: any sentence with factual markers must reference evidence.
* **Citation integrity**: every citekey resolves to an ingested paper record.
* **Claim-evidence trace**: each claim in the Claim Bank has ≥1 chunk evidence.

### 11.2 Coverage tests

* Outline sections have:

  * taxonomy
  * comparison (table or structured comparison)
  * datasets/benchmarks (if relevant)
  * limitations
  * open problems/future work (derived from explicit gaps)

### 11.3 Reproducibility (audit trail)

* Log every query + timestamp + top results + selection rationale.
* Log every ingestion event and chunk counts.
* Log extraction summary: claims per paper, entity merges.



## 12) Implementation plan (phased)

### Phase 0: Foundations

* Define schemas (Paper/Chunk/Claim/Issue).
* Implement storage:

  * vector store for chunks
  * structured DB tables for paper/claim/entity/relation/issues
* Implement provenance discipline and IDs.

### Phase 1: Minimal working loop

* Nodes: Plan → Retrieve(arXiv) → Ingest → Chunk → Synthesize → Verify → Iterate
* Grounding Verifier blocks unsupported content.

### Phase 2: Claim Bank + gap scoring

* Add ExtractToClaims and KB Curator.
* Switch synthesis to claim-first workflow.
* Add Gap Scorer with objective metrics + stop criteria.

### Phase 3: Reviewer + revision loop

* Add ARR-style reviewer rubric.
* RevisionPlanner issues tasks and triggers re-search + rewrite.

### Phase 4: Multilingual

* Multilingual query expansion + ingestion.
* Canonicalization layer for claims/entities into chosen output language.



## 13) Risks and mitigations

### Risk: PDF extraction quality

* Mitigation: ingest HTML + PDF; store quality flags; allow manual fallback to HTML.

### Risk: Hallucinated synthesis

* Mitigation: claim-first writing + strict verifier + “no new facts in polish pass”.

### Risk: Infinite loops

* Mitigation: explicit stop criteria + marginal gain threshold + iteration cap (soft cap with escalation to “needs human decision”).

### Risk: Over-reliance on arXiv

* Mitigation: coverage scoring includes venue diversity + targeted web search for peer-reviewed anchors.



## 14) Output format (survey structure template)

Typical survey sections:

1. Abstract
2. Introduction + scope
3. Background / problem formulation
4. Taxonomy of approaches
5. Method categories (subsections)
6. Benchmarks/datasets/metrics
7. Comparative analysis (tables)
8. Discussion: trends + disagreements
9. Open problems + future directions (grounded in gaps)
10. Limitations of existing research + of this survey (methodological)
11. Conclusion
12. References (BibTeX/CSL)



## 15) Operational notes

* Keep internal citations as `[@paper_id:chunk_id]` during drafting.
* Convert to target style at finalization (numeric or author-year).
* Maintain a “Do Not Claim” list for overstated phrases unless supported:

  * “state-of-the-art”, “first”, “solves”, “guarantees”, “proves” (requires explicit evidence).