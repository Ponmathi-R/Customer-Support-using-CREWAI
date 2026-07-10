import html
import importlib.util
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

DEFAULT_KB_PATH = APP_DIR / "sony_customer_care_rag_document.txt"
ANSWER_LOG_PATH = APP_DIR / "answers.txt"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
GOOGLE_WORKSHEET_NAME = os.getenv("GOOGLE_WORKSHEET_NAME", "CrewAI Logs")
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
    "sony",
    "customer",
    "care",
    "support",
    "help",
    "please",
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


def extract_urls(text: str) -> List[str]:
    urls = re.findall(r"https?://[^\s)\]>\"']+", text or "")
    cleaned = [url.rstrip(".,;:") for url in urls]
    return list(dict.fromkeys(cleaned))


def google_credentials_path() -> Path | None:
    raw_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not raw_path:
        return None

    path = Path(raw_path)
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def google_sheet_status() -> Tuple[str, str, str]:
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    credentials_path = google_credentials_path()

    if importlib.util.find_spec("gspread") is None:
        return "Install dependency", "Run: pip install -r requirements.txt", "warn"
    if sheet_id and credentials_path and credentials_path.exists():
        return "Configured", f"Worksheet: {GOOGLE_WORKSHEET_NAME}", "good"
    if sheet_id and credentials_path:
        return "Needs credentials", f"Missing file: {credentials_path.name}", "warn"
    if sheet_id:
        return "Partial", "Sheet ID found, credentials not configured.", "warn"
    return "Not configured", "TXT logging is active for this build.", "idle"


def append_google_sheet(
    query: str,
    source: str,
    answer: str,
    web_status: str,
    references: List[str],
    entry_record: str,
) -> Tuple[bool, str]:
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    credentials_path = google_credentials_path()

    if not sheet_id:
        return False, "Google Sheet ID is not configured."
    if not credentials_path or not credentials_path.exists():
        return False, "Google service-account credentials file is missing."

    try:
        import gspread
    except ModuleNotFoundError:
        return False, "Google Sheet save skipped: install dependencies with pip install -r requirements.txt"

    try:
        client = gspread.service_account(filename=str(credentials_path))
        spreadsheet = client.open_by_key(sheet_id)
        try:
            worksheet = spreadsheet.worksheet(GOOGLE_WORKSHEET_NAME)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=GOOGLE_WORKSHEET_NAME,
                rows=1000,
                cols=8,
            )

        headers = [
            "Timestamp",
            "Query",
            "Answer Source",
            "Final Answer",
            "Web Status",
            "References",
            "Entry Record",
            "Model",
        ]
        if worksheet.row_values(1) != headers:
            worksheet.update("A1:H1", [headers])

        worksheet.append_row(
            [
                datetime.now().isoformat(timespec="seconds"),
                query,
                source,
                answer,
                web_status,
                "\n".join(references),
                entry_record,
                DEFAULT_MODEL,
            ],
            value_input_option="USER_ENTERED",
        )
        return True, f"Saved to Google Sheet worksheet '{GOOGLE_WORKSHEET_NAME}'."
    except Exception as exc:
        return False, f"Google Sheet save failed: {exc}"


def render_theme() -> None:
    st.markdown(
        """
        <style>
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 18px 60px rgba(124, 58, 237, 0.18); }
            50% { box-shadow: 0 22px 80px rgba(14, 165, 233, 0.25); }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(255, 128, 191, 0.22), transparent 30%),
                radial-gradient(circle at 85% 8%, rgba(16, 185, 129, 0.18), transparent 28%),
                linear-gradient(120deg, #26104f, #0f766e, #7c2d12, #1d4ed8);
            background-size: 180% 180%;
            animation: gradientShift 16s ease infinite;
            color: #f8fafc;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.35rem;
            padding-bottom: 6rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.68);
            backdrop-filter: blur(18px);
            border-right: 1px solid rgba(255, 255, 255, 0.16);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .hero-panel, .glass-card, .timeline-card, .loading-card {
            background: rgba(255, 255, 255, 0.13);
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 18px;
            box-shadow: 0 18px 60px rgba(15, 23, 42, 0.24);
            backdrop-filter: blur(18px);
        }

        .hero-panel {
            padding: 1.35rem 1.45rem;
            margin-bottom: 1rem;
            animation: pulseGlow 5s ease-in-out infinite;
        }

        .hero-title {
            font-size: 2.2rem;
            line-height: 1.05;
            font-weight: 800;
            margin: 0.15rem 0 0.45rem;
        }

        .hero-copy {
            color: rgba(248, 250, 252, 0.82);
            margin: 0;
            max-width: 780px;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }

        .pill {
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.20);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            color: #ffffff;
            font-size: 0.86rem;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem;
        }

        .glass-card {
            padding: 0.9rem 1rem;
        }

        .card-label {
            color: rgba(248, 250, 252, 0.72);
            font-size: 0.78rem;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .card-value {
            color: #ffffff;
            font-size: 1.06rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .card-detail {
            color: rgba(248, 250, 252, 0.76);
            font-size: 0.86rem;
        }

        .timeline-card {
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
            border-left: 4px solid rgba(255, 255, 255, 0.45);
        }

        .timeline-card.good { border-left-color: #22c55e; }
        .timeline-card.warn { border-left-color: #facc15; }
        .timeline-card.idle { border-left-color: #94a3b8; }
        .timeline-card.run { border-left-color: #38bdf8; }

        .timeline-title {
            color: #ffffff;
            font-weight: 760;
            margin-bottom: 0.15rem;
        }

        .timeline-detail {
            color: rgba(248, 250, 252, 0.76);
            font-size: 0.88rem;
        }

        .loading-card {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.85rem 1rem;
            color: #ffffff;
            margin: 0.8rem 0;
        }

        .loader {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.32);
            border-top-color: #ffffff;
            border-radius: 999px;
            animation: spin 0.8s linear infinite;
        }

        .source-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.32rem 0.65rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.35);
            color: #ecfdf5;
            margin-bottom: 0.7rem;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.20);
            border-radius: 16px;
            padding: 0.45rem;
            backdrop-filter: blur(14px);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            color: #ffffff;
        }

        .stTextArea textarea, .stTextInput input,
        [data-testid="stChatInput"] textarea,
        [data-baseweb="textarea"] textarea {
            background: rgba(255, 255, 255, 0.96) !important;
            color: #111827 !important;
            caret-color: #111827 !important;
            border: 1px solid rgba(15, 23, 42, 0.22) !important;
        }

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder,
        [data-testid="stChatInput"] textarea::placeholder,
        [data-baseweb="textarea"] textarea::placeholder {
            color: #64748b !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def glass_card(label: str, value: str, detail: str) -> str:
    return (
        '<div class="glass-card">'
        f'<div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-detail">{html.escape(detail)}</div>'
        "</div>"
    )


def timeline_card(title: str, status: str, detail: str, tone: str) -> str:
    return (
        f'<div class="timeline-card {html.escape(tone)}">'
        f'<div class="timeline-title">{html.escape(title)} - {html.escape(status)}</div>'
        f'<div class="timeline-detail">{html.escape(detail)}</div>'
        "</div>"
    )


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am your Sony customer-care assistant. Ask me about warranty, "
                    "service booking, repair status, troubleshooting, or support contacts."
                ),
            }
        ]
    if "timeline" not in st.session_state:
        st.session_state.timeline = [
            ("RAG Retriever", "Ready", "Waiting for a customer query.", "idle"),
            ("Assistant", "Ready", "Will answer from local Sony context first.", "idle"),
            ("Web Search Assistant", "Standby", "Runs only if RAG has no strong match.", "idle"),
            ("Entry Agent", "Ready", "Saves the support record to answers.txt.", "idle"),
        ]
    if "last_run" not in st.session_state:
        st.session_state.last_run = None


st.set_page_config(page_title="Sony Customer Care CrewAI", layout="wide")
render_theme()
initialize_session()

st.markdown(
    """
    <div class="hero-panel">
        <div class="card-label">Sony Customer Care CrewAI</div>
        <div class="hero-title">🤖 Glassy RAG-first support desk</div>
        <p class="hero-copy">
            Ask a Sony support question. The assistant checks the local customer-care
            document first, then uses web search only when the document does not have
            enough relevant context.
        </p>
        <div class="pill-row">
            <span class="pill">📚 Agent timeline</span>
            <span class="pill">🌐 Web fallback</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kb_text = read_text_file(DEFAULT_KB_PATH)
missing_keys = require_environment()
sheet_status, sheet_detail, sheet_tone = google_sheet_status()
last_run = st.session_state.last_run

openai_status = "Ready" if "OPENAI_API_KEY" not in missing_keys else "Missing"
serper_status = "Ready" if os.getenv("SERPER_API_KEY") else "Fallback key missing"
rag_status = "Loaded" if kb_text else "Missing"
last_source = last_run["source"] if last_run else "Waiting"

st.markdown(
    '<div class="status-grid">'
    + glass_card("📄 RAG document", rag_status, DEFAULT_KB_PATH.name)
    + glass_card("🧠 Model", DEFAULT_MODEL, f"OpenAI API: {openai_status}")
    + glass_card("🌐 Web search", serper_status, "Used only when RAG has no strong match")
    + glass_card("✅ Google Sheet status", sheet_status, sheet_detail)
    + "</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 📊 Agent status panel")
st.sidebar.markdown(glass_card("Assistant", "Ready", "Answers from local context first"), unsafe_allow_html=True)
st.sidebar.markdown(glass_card("Web Search Assistant", "Conditional", "Runs only for RAG fallback"), unsafe_allow_html=True)
st.sidebar.markdown(glass_card("Entry Agent", "Ready", "Saves answers.txt"), unsafe_allow_html=True)
st.sidebar.markdown(glass_card("Last answer source", last_source, "Chat history is kept in this session"), unsafe_allow_html=True)

if missing_keys:
    st.sidebar.warning("Missing: " + ", ".join(missing_keys))

if st.sidebar.button("Clear chat history"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Chat cleared. Ask me a Sony customer-care question and I will check "
                "the source document before web search."
            ),
        }
    ]
    st.session_state.last_run = None
    st.session_state.timeline = [
        ("RAG Retriever", "Ready", "Waiting for a customer query.", "idle"),
        ("Assistant", "Ready", "Will answer from local Sony context first.", "idle"),
        ("Web Search Assistant", "Standby", "Runs only if RAG has no strong match.", "idle"),
        ("Entry Agent", "Ready", "Saves the support record to answers.txt.", "idle"),
    ]
    st.rerun()

chat_tab, timeline_tab, source_tab, web_tab = st.tabs(
    ["🤖 Chat", "📚 Agent timeline", "📄 Source document", "🌐 Web references"]
)

with chat_tab:
    st.markdown(f'<span class="source-chip">Current mode: {html.escape(last_source)}</span>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if last_run:
        answer_path = Path(last_run["answer_file"])
        with st.expander("Saved Entry", expanded=False):
            if last_run.get("sheet_saved"):
                st.success(last_run.get("sheet_message", "Saved to Google Sheet."))
            else:
                st.warning(last_run.get("sheet_message", "Google Sheet was not updated."))
            st.text(last_run["entry_record"])
        if answer_path.exists():
            st.download_button(
                "Download answers.txt",
                data=answer_path.read_text(encoding="utf-8", errors="ignore"),
                file_name="answers.txt",
                mime="text/plain",
            )

with timeline_tab:
    for title, status, detail, tone in st.session_state.timeline:
        st.markdown(timeline_card(title, status, detail, tone), unsafe_allow_html=True)

with source_tab:
    if not kb_text:
        st.error(f"Source document not found: {DEFAULT_KB_PATH.name}")
    else:
        sections = split_into_sections(kb_text)
        filter_text = st.text_input(
            "Filter source document",
            placeholder="Try warranty, repair, headphones, service center...",
        )
        filtered_sections = [
            (index, section)
            for index, section in enumerate(sections)
            if not filter_text or filter_text.lower() in section.lower()
        ]
        if filtered_sections:
            labels = [
                f"{index + 1}. {section.splitlines()[0][:90]}"
                for index, section in filtered_sections
            ]
            selected_label = st.selectbox("Document sections", labels)
            selected_index = labels.index(selected_label)
            st.text_area(
                "Source content",
                value=filtered_sections[selected_index][1],
                height=440,
            )
        else:
            st.info("No source sections matched that filter.")

with web_tab:
    if not last_run:
        st.info("Run a chat query to see web-search status and references here.")
    elif last_run["used_web_search"]:
        st.success("Web search was used because RAG did not have a strong match.")
        st.markdown("#### Web answer")
        st.write(last_run["web_answer"])
        references = last_run["references"]
        if references:
            st.markdown("#### Extracted links")
            for ref in references:
                st.markdown(f"- [{ref}]({ref})")
        else:
            st.info("No direct URLs were found in the CrewAI web answer.")
    else:
        st.success("Web search skipped because the local RAG document had relevant information.")
        with st.expander("Retrieved RAG context", expanded=False):
            st.text(last_run["rag_context"])

prompt = st.chat_input("Ask Sony customer care anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if missing_keys:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "I need `OPENAI_API_KEY` in the `.env` file before I can run the agents.",
            }
        )
        st.rerun()

    rag_context, rag_found = retrieve_rag_context(prompt, kb_text)
    use_web_search = not rag_found
    web_missing_keys = require_environment(include_serper=use_web_search)

    if web_missing_keys:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "This query needs web search, but these environment variables are missing: "
                    + ", ".join(web_missing_keys)
                ),
            }
        )
        st.rerun()

    loading_placeholder = st.empty()
    loading_label = (
        "⚡ Retrieving Sony document context and running the Assistant..."
        if rag_found
        else "⚡ RAG match was weak, running web-search fallback..."
    )
    loading_placeholder.markdown(
        f'<div class="loading-card"><span class="loader"></span><span>{html.escape(loading_label)}</span></div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("CrewAI agents are working..."):
            result = build_and_run_crew(
                query=prompt.strip(),
                rag_context=rag_context,
                use_web_search=use_web_search,
            )
    except Exception as exc:
        loading_placeholder.empty()
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Agent run failed: `{exc}`"}
        )
        st.rerun()

    loading_placeholder.empty()

    final_answer = result["web_answer"] if use_web_search else result["assistant_answer"]
    final_source = "Web Search" if use_web_search else "Local RAG Document"
    references = extract_urls(result["web_answer"])
    web_status = (
        "Web search used because RAG did not have a strong match."
        if use_web_search
        else "Web search skipped because local RAG had relevant information."
    )
    sheet_saved, sheet_message = append_google_sheet(
        query=prompt.strip(),
        source=final_source,
        answer=final_answer,
        web_status=web_status,
        references=references,
        entry_record=result["entry_record"],
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"**Source:** {final_source}\n\n{final_answer}\n\n"
                f"**Google Sheet:** {sheet_message}"
            ),
        }
    )

    st.session_state.last_run = {
        "source": final_source,
        "rag_context": rag_context,
        "rag_found": rag_found,
        "used_web_search": use_web_search,
        "assistant_answer": result["assistant_answer"],
        "web_answer": result["web_answer"],
        "references": references,
        "entry_record": result["entry_record"],
        "answer_file": result["answer_file"],
        "sheet_saved": sheet_saved,
        "sheet_message": sheet_message,
    }

    if rag_found:
        st.session_state.timeline = [
            ("RAG Retriever", "Matched", "Relevant Sony document sections were retrieved.", "good"),
            ("Assistant", "Completed", "Final answer came from the local RAG context.", "good"),
            ("Web Search Assistant", "Skipped", "No web call was needed for this query.", "idle"),
            ("Entry Agent", "Saved", "Support record written to answers.txt.", "good"),
            (
                "Google Sheet",
                "Saved" if sheet_saved else "Skipped",
                sheet_message,
                "good" if sheet_saved else "warn",
            ),
        ]
    else:
        st.session_state.timeline = [
            ("RAG Retriever", "No strong match", "Local document did not meet the relevance threshold.", "warn"),
            ("Assistant", "Checked", "Local context was insufficient for the final answer.", "warn"),
            ("Web Search Assistant", "Completed", "Serper web search fallback produced the final answer.", "good"),
            ("Entry Agent", "Saved", "Support record written to answers.txt.", "good"),
            (
                "Google Sheet",
                "Saved" if sheet_saved else "Skipped",
                sheet_message,
                "good" if sheet_saved else "warn",
            ),
        ]

    st.rerun()
