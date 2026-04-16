#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py

import streamlit as st
st.set_page_config(page_title="Agentic Proteomics", layout="wide")

# =========================
# Imports and environment
# =========================

from dotenv import load_dotenv
load_dotenv()  # Reads LANGSMITH_* and any other keys from your .env

import json
import requests
import pandas as pd
import streamlit as st

from typing import TypedDict, List, Dict, Any, Optional, Literal

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


#!/usr/bin/env python
# coding: utf-8

import os
import json
from typing import TypedDict, List, Dict, Any, Optional, Literal

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama




# =========================
# Environment
# =========================

load_dotenv()

DATA_PATH = "VNMX_LiP_DA.csv"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")
SHOW_DEBUG = True


# =========================
# Dataset loading
# =========================

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the CSV once and cache it.
    """
    return pd.read_csv(path)


df = load_dataset(DATA_PATH)


# =========================
# RAG helpers
# =========================

def get_accessions() -> List[str]:
    """
    Build dropdown options from the accession column.
    """
    if "PG.ProteinAccessions" not in df.columns:
        return []

    vals = (
        df["PG.ProteinAccessions"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return sorted(vals)


def retrieve_accession_rows(accession: str, max_rows: int = 12) -> pd.DataFrame:
    """
    Return accession-specific rows from the dataset.
    """
    sub = df[df["PG.ProteinAccessions"].astype(str).str.strip() == accession].copy()
    return sub.head(max_rows)


def retrieve_accession_context(accession: str, max_rows: int = 12) -> str:
    """
    Convert accession-specific rows into compact markdown for the LLM.
    """
    sub = retrieve_accession_rows(accession, max_rows=max_rows)

    if sub.empty:
        return f"No dataset rows found for accession {accession}."

    preferred_cols = [
        "PG.ProteinAccessions",
        "PEP.StrippedSequence",
        "start",
        "end",
        "coverage",
        "uniprot_length",
        "uniprot_pdb",
        "diff",
        "adj_pval",
        "comparison",
        "uniprot_go_f",
    ]

    existing_cols = [c for c in preferred_cols if c in sub.columns]
    if not existing_cols:
        return sub.to_markdown(index=False)

    return sub[existing_cols].to_markdown(index=False)


# =========================
# HTTP helper
# =========================

def safe_get_json(
    url: str,
    method: str = "GET",
    json_body: Optional[dict] = None,
    timeout: int = 30,
) -> dict | list:
    """
    Small helper for API calls. Raises on HTTP errors.
    """
    headers = {"Accept": "application/json"}

    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=timeout)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
    else:
        raise ValueError(f"Unsupported method: {method}")

    resp.raise_for_status()
    return resp.json()


# =========================
# Structure tools
# =========================

def query_pdb(accession: str) -> dict:
    """
    Query the RCSB PDB Search API for experimental structures mapped to a UniProt accession.
    Always returns a plain dict.
    """
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    query_body = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": 10},
        },
    }

    try:
        result = safe_get_json(search_url, method="POST", json_body=query_body)
        hits = result.get("result_set", [])

        candidates = []
        for hit in hits:
            entry_id = hit.get("identifier")
            if not entry_id:
                continue

            entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
            entry_data = safe_get_json(entry_url)

            exptl = entry_data.get("exptl", [])
            method = exptl[0].get("method") if exptl else None

            resolution = None
            res_list = entry_data.get("rcsb_entry_info", {}).get("resolution_combined")
            if res_list:
                resolution = res_list[0]

            candidates.append(
                {
                    "source": "pdb",
                    "structure_id": entry_id,
                    "method": method,
                    "resolution": resolution,
                    "coverage": None,
                    "confidence": None,
                    "has_ligand_context": False,
                    "url": f"https://www.rcsb.org/structure/{entry_id}",
                }
            )

        return {
            "source": "pdb",
            "accession": accession,
            "candidates": candidates,
        }

    except Exception as e:
        return {
            "source": "pdb",
            "accession": accession,
            "candidates": [],
            "error": str(e),
        }


def query_alphafold(accession: str) -> dict:
    """
    Query AlphaFold DB for a UniProt accession.
    Always returns a plain dict.
    """
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"

    try:
        result = safe_get_json(url)

        candidates = []
        for item in result:
            candidates.append(
                {
                    "source": "alphafold",
                    "structure_id": item.get("entryId") or accession,
                    "method": "AlphaFold DB prediction",
                    "resolution": None,
                    "coverage": 1.0,
                    "confidence": item.get("globalMetricValue"),
                    "has_ligand_context": False,
                    "url": item.get("pdbUrl") or item.get("bcifUrl") or item.get("modelUrl"),
                    "pae_url": item.get("paeDocUrl") or item.get("paeImageUrl"),
                }
            )

        return {
            "source": "alphafold",
            "accession": accession,
            "candidates": candidates,
        }

    except Exception as e:
        return {
            "source": "alphafold",
            "accession": accession,
            "candidates": [],
            "error": str(e),
        }


def query_alphafill(accession: str) -> dict:
    """
    Query AlphaFill using a 3D-Beacon-style endpoint.
    Always returns a plain dict.
    """
    url = f"https://alphafill.eu/v1/aff/3d-beacon/{accession}"

    try:
        result = safe_get_json(url)
        raw_models = result if isinstance(result, list) else result.get("structures", [])

        candidates = []
        for item in raw_models:
            candidates.append(
                {
                    "source": "alphafill",
                    "structure_id": item.get("id") or item.get("model_id") or accession,
                    "method": "AlphaFill enriched model",
                    "resolution": None,
                    "coverage": item.get("coverage"),
                    "confidence": item.get("confidence_score"),
                    "has_ligand_context": True,
                    "url": item.get("url") or item.get("model_url"),
                    "metadata_url": item.get("json_url"),
                }
            )

        return {
            "source": "alphafill",
            "accession": accession,
            "candidates": candidates,
        }

    except Exception as e:
        return {
            "source": "alphafill",
            "accession": accession,
            "candidates": [],
            "error": str(e),
        }


def query_swiss_model(accession: str) -> dict:
    """
    Query a 3D-Beacons-style endpoint and keep only SWISS-MODEL entries.
    Always returns a plain dict.
    """
    url = f"https://www.ebi.ac.uk/pdbe/pdbe-kb/3dbeacons/api/uniprot/{accession}.json"

    try:
        result = safe_get_json(url)

        all_models = (
            result.get("structures")
            or result.get("models")
            or result.get("entries")
            or []
        )

        candidates = []
        for item in all_models:
            provider = str(
                item.get("provider")
                or item.get("model_provider")
                or item.get("source")
                or ""
            ).lower()

            if "swiss" not in provider:
                continue

            candidates.append(
                {
                    "source": "swiss_model",
                    "structure_id": item.get("id") or item.get("model_id") or accession,
                    "method": "SWISS-MODEL homology model",
                    "resolution": None,
                    "coverage": item.get("coverage") or item.get("sequence_coverage"),
                    "confidence": item.get("qmean") or item.get("model_score"),
                    "has_ligand_context": False,
                    "url": item.get("url") or item.get("model_url"),
                }
            )

        return {
            "source": "swiss_model",
            "accession": accession,
            "candidates": candidates,
        }

    except Exception as e:
        return {
            "source": "swiss_model",
            "accession": accession,
            "candidates": [],
            "error": str(e),
        }


TOOL_REGISTRY = {
    "query_pdb": query_pdb,
    "query_alphafold": query_alphafold,
    "query_alphafill": query_alphafill,
    "query_swiss_model": query_swiss_model,
}


# =========================
# Scoring
# =========================

def safe_float(x):
    """
    Convert values to float when possible.
    """
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def score_candidate(candidate: dict) -> float:
    """
    Deterministic ranking logic.
    """
    source = candidate.get("source")
    coverage = safe_float(candidate.get("coverage")) or 0.0
    resolution = safe_float(candidate.get("resolution"))
    confidence = safe_float(candidate.get("confidence")) or 0.0
    has_ligand_context = bool(candidate.get("has_ligand_context", False))

    score = 0.0
    score += coverage * 50

    if source == "pdb":
        score += 40
        if resolution is not None:
            score += max(0, 10 - (resolution * 2))

    elif source == "alphafill":
        score += 30
        if has_ligand_context:
            score += 10
        score += confidence / 10

    elif source == "swiss_model":
        score += 20
        score += confidence / 10

    elif source == "alphafold":
        score += 15
        score += confidence / 10

    return score


def choose_best_candidate(candidates: List[dict]) -> Optional[dict]:
    """
    Pick the top-scoring candidate.
    """
    if not candidates:
        return None

    ranked = sorted(candidates, key=score_candidate, reverse=True)
    return ranked[0]


# =========================
# Graph state
# =========================

class GraphState(TypedDict, total=False):
    accession: str
    rag_context: str

    # ReAct planning output
    plan: Dict[str, Any]

    # Raw model outputs for debugging
    messages: List[Dict[str, Any]]

    structure_candidates: List[Dict[str, Any]]
    best_candidate: Optional[Dict[str, Any]]
    final_answer: str
    errors: List[str]


# =========================
# LLM
# =========================

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


# =========================
# Graph nodes
# =========================

def load_accession_context(state: GraphState) -> GraphState:
    """
    RAG node: retrieve accession-specific CSV context.
    """
    accession = state["accession"]
    context = retrieve_accession_context(accession)

    return {
        **state,
        "rag_context": context,
        "messages": state.get("messages", []),
        "errors": state.get("errors", []),
        "structure_candidates": state.get("structure_candidates", []),
    }


def react_plan(state: GraphState) -> GraphState:
    """
    ReAct planning node.

    The model is asked to emit strict JSON with:
    - thought
    - tools
    - done

    This preserves a ReAct-style structure without relying on native tool-calling.
    """
    accession = state["accession"]
    rag_context = state["rag_context"]
    errors = list(state.get("errors", []))
    messages = list(state.get("messages", []))

    prompt = f"""
You are a structural proteomics assistant.

Selected accession: {accession}

Retrieved accession-specific dataset context:
{rag_context}

Available tools:
- query_pdb
- query_alphafold
- query_alphafill
- query_swiss_model

Return STRICT JSON only with this schema:
{{
  "thought": "brief reasoning",
  "tools": ["tool_name_1", "tool_name_2"],
  "done": false
}}

Rules:
- Return valid JSON only
- Do not include markdown
- Use only the tool names listed above
- If enough information is already available, return:
  {{
    "thought": "...",
    "tools": [],
    "done": true
  }}
- For an initial lookup, it is reasonable to request all relevant structure tools
"""

    raw_response = llm.invoke(prompt)

    parsed_plan = None
    raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    try:
        parsed_plan = json.loads(raw_text)
    except Exception as e:
        errors.append(f"Plan JSON parse failed: {e}")
        parsed_plan = {
            "thought": "Failed to parse model plan; falling back to all tools.",
            "tools": list(TOOL_REGISTRY.keys()),
            "done": False,
        }

    if not isinstance(parsed_plan, dict):
        errors.append(f"Plan was not a dict: {type(parsed_plan)}")
        parsed_plan = {
            "thought": "Non-dict plan; falling back to all tools.",
            "tools": list(TOOL_REGISTRY.keys()),
            "done": False,
        }

    planned_tools = parsed_plan.get("tools", [])
    if not isinstance(planned_tools, list):
        errors.append("Plan tools field was not a list; falling back to all tools.")
        planned_tools = list(TOOL_REGISTRY.keys())

    # Keep only valid tool names
    planned_tools = [t for t in planned_tools if t in TOOL_REGISTRY]

    # If model returned nothing useful, fall back to all tools
    if not planned_tools and not parsed_plan.get("done", False):
        planned_tools = list(TOOL_REGISTRY.keys())

    parsed_plan["tools"] = planned_tools

    messages.append(
        {
            "node": "react_plan",
            "raw_model_output": raw_text,
            "parsed_plan": parsed_plan,
        }
    )

    return {
        **state,
        "plan": parsed_plan,
        "messages": messages,
        "errors": errors,
    }


def route_after_plan(state: GraphState) -> Literal["execute_tools", "validate_and_rank"]:
    """
    Conditional routing after planning.
    """
    plan = state.get("plan", {})
    tools = plan.get("tools", []) if isinstance(plan, dict) else []
    done = bool(plan.get("done", False)) if isinstance(plan, dict) else False

    if tools and not done:
        return "execute_tools"

    return "validate_and_rank"


def execute_tools(state: GraphState) -> GraphState:
    """
    Act node: execute the planned tools deterministically.
    """
    accession = state["accession"]
    plan = state.get("plan", {})
    selected_tools = plan.get("tools", [])
    errors = list(state.get("errors", []))
    messages = list(state.get("messages", []))
    candidates = list(state.get("structure_candidates", []))

    for tool_name in selected_tools:
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            errors.append(f"Unknown tool in plan: {tool_name}")
            continue

        try:
            payload = fn(accession)

            messages.append(
                {
                    "node": "execute_tools",
                    "tool_name": tool_name,
                    "tool_payload": payload,
                }
            )

            if isinstance(payload, dict):
                new_candidates = payload.get("candidates", [])
                if isinstance(new_candidates, list):
                    candidates.extend(new_candidates)

                if payload.get("error"):
                    errors.append(f"{payload.get('source', tool_name)}: {payload['error']}")
            else:
                errors.append(f"{tool_name} returned unexpected type: {type(payload)}")

        except Exception as e:
            errors.append(f"{tool_name}: {e}")

    # Deduplicate structure candidates
    deduped = []
    seen = set()
    for c in candidates:
        key = (c.get("source"), c.get("structure_id"))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return {
        **state,
        "structure_candidates": deduped,
        "messages": messages,
        "errors": errors,
    }


def validate_and_rank(state: GraphState) -> GraphState:
    """
    Choose the best candidate using deterministic scoring.
    """
    best = choose_best_candidate(state.get("structure_candidates", []))

    return {
        **state,
        "best_candidate": best,
    }


def finalize_answer(state: GraphState) -> GraphState:
    """
    Use the LLM to explain the best structure source, but only after
    retrieval and ranking are finished.
    """
    accession = state["accession"]
    best = state.get("best_candidate")
    rag_context = state.get("rag_context", "")
    errors = state.get("errors", [])

    if not best:
        return {
            **state,
            "final_answer": (
                f"I could not identify a strong structure candidate for accession {accession}. "
                f"I checked the available sources and either no candidates were returned or the results were incomplete. "
                f"Errors/gaps: {errors if errors else 'No explicit API errors, but no usable candidates were identified.'}"
            ),
        }

    prompt = f"""
You are a structural proteomics assistant.

Accession: {accession}

Dataset context:
{rag_context}

Best candidate:
{json.dumps(best, indent=2)}

Write a concise explanation of why this is the best current structure source.
Mention source, structure ID, and key evidence such as resolution, confidence,
coverage, and ligand/cofactor context when available.
"""

    try:
        response = llm.invoke(prompt)
        explanation = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        explanation = (
            f"For accession {accession}, the best current structure source is "
            f"{best.get('source')} with structure {best.get('structure_id')}. "
            f"Additional explanation generation failed: {e}"
        )

    return {
        **state,
        "final_answer": explanation,
    }


# =========================
# Build graph
# =========================

builder = StateGraph(GraphState)

builder.add_node("load_accession_context", load_accession_context)
builder.add_node("react_plan", react_plan)
builder.add_node("execute_tools", execute_tools)
builder.add_node("validate_and_rank", validate_and_rank)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "load_accession_context")
builder.add_edge("load_accession_context", "react_plan")

builder.add_conditional_edges(
    "react_plan",
    route_after_plan,
    {
        "execute_tools": "execute_tools",
        "validate_and_rank": "validate_and_rank",
    },
)

builder.add_edge("execute_tools", "validate_and_rank")
builder.add_edge("validate_and_rank", "finalize_answer")
builder.add_edge("finalize_answer", END)

graph = builder.compile()


# =========================
# Streamlit UI
# =========================

st.title("Agentic Proteomics")
st.write("Start by selecting a protein accession from the dataset.")

accessions = get_accessions()

if not accessions:
    st.error("No accessions found. Check that VNMX_LiP_DA.csv exists and contains PG.ProteinAccessions.")
    st.stop()

selected_accession = st.selectbox(
    "Select a protein accession",
    options=accessions,
)

if st.button("Analyze structure sources"):
    with st.spinner("Running RAG + ReAct structure analysis..."):
        result = graph.invoke(
            {
                "accession": selected_accession,
                "messages": [],
                "errors": [],
                "structure_candidates": [],
            }
        )

    st.subheader("Final answer")
    st.markdown(result.get("final_answer", "No final answer produced."))

    with st.expander("Retrieved accession context (RAG)"):
        st.code(result.get("rag_context", ""))

    with st.expander("Best candidate"):
        best_candidate = result.get("best_candidate")
        if isinstance(best_candidate, (dict, list)):
            st.json(best_candidate)
        else:
            st.write(best_candidate)

    with st.expander("All structure candidates"):
        st.json(result.get("structure_candidates", []))

    if SHOW_DEBUG:
        with st.expander("Errors / debug"):
            st.write("Errors:")
            st.json(result.get("errors", []))

            st.write("Plan:")
            st.write(result.get("plan"))

            st.write("Best candidate raw type:")
            st.write(type(result.get("best_candidate")))

            st.write("Best candidate raw value:")
            st.write(result.get("best_candidate"))

            st.write("Number of structure candidates:")
            st.write(len(result.get("structure_candidates", [])))

            st.write("Messages:")
            for i, msg in enumerate(result.get("messages", [])):
                st.write(f"Message {i}:")
                st.write(msg)

