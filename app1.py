import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

DEFAULT_KB_PATH = APP_DIR / "sony_customer_care_rag_document.txt"
ANSWER_LOG_PATH = APP_DIR / "answers.txt"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "with",
}


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def split_into_sections(document: str) -> List[str]:
    parts = re.split(r"\n(?=Section: )", document.strip())
    return [part.strip() for part in parts if part.strip()]


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def retrieve_rag_context(
    query: str, document: str, top_k: int = 4, min_score: int = 3
) -> Tuple[str, bool]:
    sections = split_into_sections(document)
    if not sections:
        return "No local Sony customer-care knowledge document was found.", False

    query_terms = set(tokenize(query))
    scored_sections: List[Tuple[int, str]] = []

    for section in sections:
        section_terms = tokenize(section)
        section_term_set = set(section_terms)
        overlap = len(query_terms.intersection(section_term_set))
        title_bonus = 4 if query.lower() in section[:160].lower() else 0
        intent_bonus = sum(2 for term in query_terms if f" {term}" in section[:300].lower())
        scored_sections.append((overlap + title_bonus + intent_bonus, section))

    best = sorted(scored_sections, key=lambda item: item[0], reverse=True)[:top_k]
    selected = [section for score, section in best if score >= min_score]
    if not selected:
        return "No relevant local Sony customer-care RAG context was found.", False

    return "\n\n---\n\n".join(selected), True


def require_environment(include_serper: bool = False) -> List[str]:
    required_keys = ["OPENAI_API_KEY"]
    if include_serper:
        required_keys.append("SERPER_API_KEY")
    return [key for key in required_keys if not os.getenv(key)]


def task_output_text(task) -> str:
    output = getattr(task, "output", None)
    raw = getattr(output, "raw", None)
    if raw:
        return str(raw).strip()
    return str(output).strip() if output else ""


def build_and_run_crew(query: str, rag_context: str, use_web_search: bool) -> Dict[str, str]:
    from crewai import Agent, Crew, Process, Task

    search_tools = []
    if use_web_search:
        from crewai_tools import SerperDevTool

        search_tools = [SerperDevTool(n_results=5)]

    assistant = Agent(
        role="Assistant",
        goal="Answer Sony customer-care queries directly using the supplied support context when relevant.",
        backstory=(
            "You are a careful Sony customer-care assistant. You give concise, practical "
            "answers, ask for missing model or warranty details, and avoid promising repair "
            "or warranty outcomes."
        ),
        llm=DEFAULT_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    web_search_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Search the web for current Sony support information when local RAG context "
            "is not enough."
        ),
        backstory=(
            "You are a web research support specialist. You prefer official Sony pages, "
            "mention when details should be verified live, and avoid invented contact details. "
            "When the local RAG document already has relevant information, you do not search."
        ),
        tools=search_tools,
        llm=DEFAULT_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal="Write the customer query, final answer, and web-search status into a text record.",
        backstory=(
            "You are a precise support record keeper. You preserve the customer's query, "
            "the final answer, and whether web search was used in a clean text format."
        ),
        llm=DEFAULT_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    assistant_task = Task(
        description=(
            "Customer query:\n{query}\n\n"
            "Relevant local Sony customer-care RAG context:\n{rag_context}\n\n"
            "Answer the query directly. Use the local context only when it is relevant. "
            "If details are missing, ask for the exact details needed. Keep the answer "
            "customer-friendly and avoid claiming warranty approval."
        ),
        expected_output=(
            "A direct customer-care answer for the query, with practical next steps and "
            "missing details to request if needed."
        ),
        agent=assistant,
    )

    if use_web_search:
        web_description = (
            "The local Sony customer-care RAG document did not contain enough relevant "
            "information for this query:\n{query}\n\n"
            "Search the web for current Sony customer-care information. Prefer official "
            "Sony pages or clearly reliable sources. Summarize what you find and include "
            "important caveats about region, warranty, or verification."
        )
        web_expected_output = (
            "A web-informed answer that cites the kind of sources used and gives current "
            "support guidance."
        )
    else:
        web_description = (
            "Do not search the web and do not use any tools. The local Sony customer-care "
            "RAG document already contains relevant information for this query:\n{query}\n\n"
            "Return exactly this sentence: Web search skipped because the local RAG document "
            "had relevant information."
        )
        web_expected_output = "A one-sentence note that web search was skipped."

    web_task = Task(
        description=web_description,
        expected_output=web_expected_output,
        agent=web_search_assistant,
        context=[assistant_task],
    )

    entry_task = Task(
        description=(
            "Create a text support record for this query:\n{query}\n\n"
            "Use the previous task outputs as the Assistant Answer and Web Search status. "
            "If web search was skipped, record that clearly. Write the record with these "
            "labels: Query, Answer Source, Assistant Answer, Web Search Status or Answer, "
            "Follow-up Notes. This text will be saved to:\n"
            "{answer_file}"
        ),
        expected_output=(
            "A complete text record containing the query, answer source, final answer, "
            "web-search status or answer, and follow-up notes."
        ),
        agent=entry_agent,
        context=[assistant_task, web_task],
        output_file=str(ANSWER_LOG_PATH),
    )

    crew = Crew(
        agents=[assistant, web_search_assistant, entry_agent],
        tasks=[assistant_task, web_task, entry_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff(
        inputs={
            "query": query,
            "rag_context": rag_context,
            "answer_file": str(ANSWER_LOG_PATH),
        }
    )

    return {
        "assistant_answer": task_output_text(assistant_task),
        "web_answer": task_output_text(web_task),
        "entry_record": task_output_text(entry_task) or str(crew_output),
        "answer_file": str(ANSWER_LOG_PATH),
        "used_web_search": str(use_web_search),
    }


st.set_page_config(page_title="Sony Customer Care CrewAI", layout="wide")
st.title("Sony Customer Care Multi-Agent Support")

st.caption(
    "RAG first. Web search runs only when the local Sony document does not have relevant context."
)

missing_keys = require_environment()
if missing_keys:
    st.warning(
        "Missing environment variables: "
        + ", ".join(missing_keys)
        + ". Set them before running the crew."
    )

kb_text = read_text_file(DEFAULT_KB_PATH)
if kb_text:
    st.sidebar.success(f"RAG document loaded: {DEFAULT_KB_PATH.name}")
else:
    st.sidebar.error(f"RAG document not found: {DEFAULT_KB_PATH.name}")

st.sidebar.write(f"Model: {DEFAULT_MODEL}")
st.sidebar.write(f"Answer log: {ANSWER_LOG_PATH.name}")

query = st.text_area(
    "Enter the customer query or task",
    placeholder="Example: My Sony Bravia TV is not turning on. What should I do?",
    height=120,
)

run_clicked = st.button("Run support crew", type="primary", disabled=not query.strip())

if run_clicked:
    if missing_keys:
        st.error("Add the missing API keys, then rerun the app.")
        st.stop()

    rag_context, rag_found = retrieve_rag_context(query, kb_text)
    use_web_search = not rag_found

    web_missing_keys = require_environment(include_serper=use_web_search)
    if web_missing_keys:
        st.error(
            "This query needs web search, but these environment variables are missing: "
            + ", ".join(web_missing_keys)
        )
        st.stop()

    if rag_found:
        st.success("Relevant RAG context found. Web search will be skipped.")
    else:
        st.info("No strong RAG match found. Running web search fallback.")

    with st.expander("Retrieved RAG context", expanded=False):
        st.text(rag_context)

    try:
        spinner_text = (
            "Running RAG answer flow..."
            if rag_found
            else "Running web-search fallback flow..."
        )
        with st.spinner(spinner_text):
            result = build_and_run_crew(
                query=query.strip(),
                rag_context=rag_context,
                use_web_search=use_web_search,
            )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    final_answer = result["web_answer"] if use_web_search else result["assistant_answer"]
    final_source = "Web Search" if use_web_search else "Local RAG Document"

    st.subheader("Final Answer")
    st.caption(f"Source: {final_source}")
    st.write(final_answer)

    with st.expander("Saved Entry", expanded=False):
        st.text(result["entry_record"])

    answer_path = Path(result["answer_file"])
    if answer_path.exists():
        st.download_button(
            "Download answers.txt",
            data=answer_path.read_text(encoding="utf-8", errors="ignore"),
            file_name="answers.txt",
            mime="text/plain",
        )
