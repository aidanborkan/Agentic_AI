# Agentic_AI
Applications of Artificial Intelligence in Bioinformatics


# Agentic Proteomics

Agentic Proteomics is a Streamlit application for accession-guided structural proteomics analysis.  
The app combines:

- a dropdown-first UI for selecting a protein accession from a dataset
- retrieval-augmented generation (RAG) using accession-specific rows from `VNMX_LiP_DA.csv`
- a LangGraph ReAct workflow for deciding when to call structure tools
- live structure-source queries against:
  - PDB
  - AlphaFold DB
  - AlphaFill
  - SWISS-MODEL / 3D-Beacons
- deterministic ranking logic to choose the best candidate structure

## Project goal

Given a protein accession from the dataset, the app:

1. retrieves accession-specific experimental context from the CSV
2. asks an LLM to reason over that context
3. allows the LLM to call structure tools when needed
4. collects returned structure candidates
5. ranks the candidates using explicit code-based scoring
6. returns the best available structure source with an explanation

## Current workflow

The app uses a LangGraph state machine with the following major steps:

- `load_accession_context`
- `react_model`
- `tool_node`
- `collect_candidates`
- `validate_and_rank`
- `finalize_answer`

The ReAct loop continues until the model stops requesting tools.

## Architecture

### UI
- Streamlit app
- accession selected from a dropdown built from `PG.ProteinAccessions`

### RAG
- accession-specific rows retrieved from `VNMX_LiP_DA.csv`
- selected columns are converted to compact markdown and passed to the model

### ReAct agent
- local Ollama model via `ChatOllama`
- tools bound to the model:
  - `query_pdb`
  - `query_alphafold`
  - `query_alphafill`
  - `query_swiss_model`

### Ranking
Candidate structures are scored deterministically using:
- coverage
- resolution
- confidence
- ligand/cofactor context
- source-specific priority

## Graph state

The graph state tracks:

- `accession`
- `rag_context`
- `messages`
- `structure_candidates`
- `best_candidate`
- `final_answer`
- `errors`

## Example graph flow

```text
START
  ↓
load_accession_context
  ↓
react_model
  ├── if tool calls → tool_node → collect_candidates → react_model
  └── if no tool calls → validate_and_rank → finalize_answer → END
