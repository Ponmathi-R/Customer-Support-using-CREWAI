'''provide both answers in a text record with the labels: Query, Assistant Answer, Web Search Answer, Follow-up Notes. 
Save this record to answers.txt.'''
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


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def split_into_sections(document: str) -> List[str]:
    parts = re.split(r"\n(?=Section: )", document.strip())
    return [part.strip() for part in parts if part.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve_rag_context(query: str, document: str, top_k: int = 4) -> str:
    sections = split_into_sections(document)
    if not sections:
        return "No local Sony customer-care knowledge document was found."

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
    selected = [section for score, section in best if score > 0]
    if not selected:
        selected = [sections[0]]

    return "\n\n---\n\n".join(selected)


def require_environment() -> List[str]:
    required_keys = ["OPENAI_API_KEY", "SERPER_API_KEY"]
    return [key for key in required_keys if not os.getenv(key)]


def task_output_text(task) -> str:
    output = getattr(task, "output", None)
    raw = getattr(output, "raw", None)
    if raw:
        return str(raw).strip()
    return str(output).strip() if output else ""


def build_and_run_crew(query: str, rag_context: str) -> Dict[str, str]:
    from crewai import Agent, Crew, Process, Task
    from crewai_tools import SerperDevTool

    search_tool = SerperDevTool(n_results=5)

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
        goal="Search the web for current Sony support information and answer from the results.",
        backstory=(
            "You are a web research support specialist. You prefer official Sony pages, "
            "mention when details should be verified live, and avoid invented contact details."
        ),
        tools=[search_tool],
        llm=DEFAULT_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal="Write the customer query and both answers into a text record.",
        backstory=(
            "You are a precise support record keeper. You preserve the customer's query, "
            "the direct assistant answer, and the web-search answer in a clean text format."
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

    web_task = Task(
        description=(
            "Search the web for current Sony customer-care information related to this query:\n"
            "{query}\n\n"
            "Prefer official Sony pages or clearly reliable sources. Summarize what you find "
            "and include any important caveats about region, warranty, or verification."
        ),
        expected_output=(
            "A web-informed answer that cites the kind of sources used and gives current "
            "support guidance."
        ),
        agent=web_search_assistant,
        context=[assistant_task],
    )

    entry_task = Task(
        description=(
            "Create a text support record for this query:\n{query}\n\n"
            "Use the previous task outputs as Answer 1 from the Assistant and Answer 2 from "
            "the Web Search Assistant. Write the record with these labels: Query, Assistant "
            "Answer, Web Search Answer, Follow-up Notes. This text will be saved to:\n"
            "{answer_file}"
        ),
        expected_output=(
            "A complete text record containing the query, answer 1, answer 2, and follow-up notes."
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
    }


st.set_page_config(page_title="Sony Customer Care CrewAI", layout="wide")
st.title("Sony Customer Care Multi-Agent Support")

st.caption(
    "Three-agent sequential CrewAI flow: Assistant, Web Search Assistant, Entry Agent."
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

    rag_context = retrieve_rag_context(query, kb_text)

    with st.expander("Retrieved RAG context", expanded=False):
        st.text(rag_context)

    try:
        with st.spinner("Running the sequential CrewAI agents..."):
            result = build_and_run_crew(query=query.strip(), rag_context=rag_context)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    left, right = st.columns(2)

    with left:
        st.subheader("Assistant Answer")
        st.write(result["assistant_answer"])

    with right:
        st.subheader("Web Search Answer")
        st.write(result["web_answer"])

    st.subheader("Saved Entry")
    st.text(result["entry_record"])

    answer_path = Path(result["answer_file"])
    if answer_path.exists():
        st.download_button(
            "Download answers.txt",
            data=answer_path.read_text(encoding="utf-8", errors="ignore"),
            file_name="answers.txt",
            mime="text/plain",
        )
