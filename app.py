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
SAFETY_REFUSAL_MESSAGE = (
    "I cannot help with instructions to harm or kill a person. If there is an "
    "immediate safety risk, contact local emergency services or a trusted person "
    "right now. I can help with safe, non-harmful Sony customer-care questions."
)
UNSAFE_INTENT_PATTERNS = [
    (
        r"\b(how|ways|steps|guide|method|methods|plan|plans|best way|easy way)\b"
        r".*\b(kill|murder|assassinate|stab|shoot|strangle|poison|torture|harm)\b"
        r".*\b(human|humans|person|persons|someone|people|man|woman|child|children|victim|victims)\b",
        "violent harm instructions",
    ),
    (
        r"\b(kill|murder|assassinate|stab|shoot|strangle|poison|torture|harm)\b"
        r".*\b(human|humans|person|persons|someone|people|man|woman|child|children|victim|victims)\b",
        "violent harm request",
    ),
    (
        r"\b(human|humans|person|persons|someone|people|man|woman|child|children|victim|victims)\b"
        r".*\b(kill|murder|assassinate|stab|shoot|strangle|poison|torture|harm)\b",
        "violent harm request",
    ),
    (
        r"\b(make|build|create|construct|assemble|use)\b"
        r".*\b(bomb|explosive|poison|weapon)\b",
        "weapon or poison instructions",
    ),
    (
        r"\b(hide|dispose|bury|destroy)\b.*\b(body|corpse|evidence)\b",
        "evading detection after harm",
    ),
    (
        r"\b(kill myself|suicide|self harm|self-harm)\b",
        "self-harm request",
    ),
]


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


def normalize_for_guardrail(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def run_safety_guardrail(text: str) -> Tuple[bool, str, List[str]]:
    normalized = normalize_for_guardrail(text)
    matched_reasons = [
        reason
        for pattern, reason in UNSAFE_INTENT_PATTERNS
        if re.search(pattern, normalized)
    ]
    if matched_reasons:
        return False, SAFETY_REFUSAL_MESSAGE, matched_reasons
    return True, "", []


def build_blocked_memory(query: str, reasons: List[str]) -> Dict[str, object]:
    reason_text = ", ".join(reasons) if reasons else "unsafe request"
    return {
        "source": "Input Guardrail",
        "rag_context": "Guardrail blocked before RAG retrieval.",
        "rag_found": False,
        "used_web_search": False,
        "greeting": "",
        "rag_answer": "",
        "web_answer": "",
        "references": [],
        "entry_record": (
            f"Query: {query}\n"
            f"Guardrail: blocked before agent execution\n"
            f"Reason: {reason_text}\n"
            f"Response: {SAFETY_REFUSAL_MESSAGE}"
        ),
        "answer_file": "",
        "execution_memory": {
            "input_guardrail": "blocked",
            "matched_reasons": reasons,
            "agent_execution": "not_started",
            "agent_1_greeting": "not_run",
            "agent_2_rag_output": "not_run",
            "agent_3_web_output": "not_run",
            "agent_4_entry_output": "not_run",
        },
        "sheet_saved": False,
        "sheet_message": "Skipped because input guardrail blocked the request.",
    }


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


def persist_answer_log(entry_record: str) -> None:
    ANSWER_LOG_PATH.write_text(entry_record, encoding="utf-8")


def build_and_run_crew(query: str, rag_context: str, rag_found: bool) -> Dict[str, object]:
    from crewai import Agent, Crew, Process, Task
    from crewai_tools import SerperDevTool

    greeting_agent = Agent(
        role="Greeting Agent",
        goal="Greet the customer warmly and acknowledge their support request.",
        backstory=(
            "You are the first point of contact for Sony customer care. You greet the "
            "customer professionally, acknowledge the request, and briefly say the support "
            "team will check the best available source."
        ),
        llm=DEFAULT_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    rag_search_agent = Agent(
        role="RAG Search Agent",
        goal="Answer Sony customer-care queries using the supplied local RAG context.",
        backstory=(
            "You are a careful Sony customer-care RAG specialist. You use only the provided "
            "local support context when it is relevant. You give concise, practical answers, "
            "ask for missing model or warranty details, and avoid promising repair or "
            "warranty outcomes."
        ),
        llm=DEFAULT_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal="Write Agent 1 output and the selected Agent 2 or Agent 3 output into a text record.",
        backstory=(
            "You are a precise support record keeper. You receive execution memory from the "
            "previous agents, preserve the greeting, preserve either the RAG answer or the "
            "web-search answer, and save everything in a clean text format."
        ),
        llm=DEFAULT_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    greeting_task = Task(
        description=(
            "Customer query:\n{query}\n\n"
            "Greet the customer in one or two friendly sentences. Acknowledge their request "
            "without answering the support question yet."
        ),
        expected_output=(
            "A short greeting that acknowledges the customer's Sony support request."
        ),
        agent=greeting_agent,
    )

    rag_task = Task(
        description=(
            "Customer query:\n{query}\n\n"
            "Relevant local Sony customer-care RAG context:\n{rag_context}\n\n"
            "RAG match status: {rag_status}\n\n"
            "If RAG match status is FOUND, answer the customer using only the local RAG "
            "context. If RAG match status is NOT_FOUND, do not invent an answer; return "
            "exactly: This information is not present in the local Sony customer-care document. "
            "If details are missing from a valid RAG answer, ask for the exact details "
            "needed. Keep the answer customer-friendly and avoid claiming warranty approval."
        ),
        expected_output=(
            "A RAG-based answer when local context is relevant, otherwise a clear no-match note."
        ),
        agent=rag_search_agent,
        context=[greeting_task],
    )

    web_search_assistant = Agent(
        role="Web Search Assistant",
        goal="Search the web for current Sony information and answer from the results.",
        backstory=(
            "You are a web research support specialist. You prefer official Sony pages, "
            "mention when details should be verified live, and avoid invented contact details."
        ),
        tools=[SerperDevTool(n_results=5)],
        llm=DEFAULT_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    web_task = Task(
        description=(
            "Search the web for current Sony information related to this query:\n"
            "{query}\n\n"
            "Prefer official Sony pages or clearly reliable sources. Summarize what you "
            "find and include important caveats about region, warranty, or verification."
        ),
        expected_output=(
            "A web-informed answer that cites the kind of sources used and gives current "
            "support guidance."
        ),
        agent=web_search_assistant,
        context=[greeting_task, rag_task],
    )

    entry_task = Task(
        description=(
            "Create a text support record for this query:\n{query}\n\n"
            "Execution memory is available through the previous task outputs:\n"
            "- Agent 1 output: Greeting Agent output\n"
            "- Agent 2 output: RAG Search Agent output\n"
            "- Agent 3 output: Web Search Assistant output\n\n"
            "Write Agent 1 output, Agent 2 RAG output, and Agent 3 web-search output. "
            "If Agent 2 says the information is not present in the local document, record "
            "that clearly and still preserve Agent 3's web answer. Write the record with "
            "these labels: Query, Agent 1 Greeting, Agent 2 RAG Output, Agent 3 Web Search "
            "Output, Follow-up Notes. "
            "This text will be saved to:\n"
            "{answer_file}"
        ),
        expected_output=(
            "A complete text record containing Agent 1 output, Agent 2 RAG output, "
            "Agent 3 web-search output, and follow-up notes."
        ),
        agent=entry_agent,
        context=[greeting_task, rag_task, web_task],
    )

    crew = Crew(
        agents=[greeting_agent, rag_search_agent, web_search_assistant, entry_agent],
        tasks=[greeting_task, rag_task, web_task, entry_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff(
        inputs={
            "query": query,
            "rag_context": rag_context,
            "rag_status": "FOUND" if rag_found else "NOT_FOUND",
            "answer_file": str(ANSWER_LOG_PATH),
        }
    )

    web_answer = task_output_text(web_task)
    execution_memory = {
        "agent_1_greeting": task_output_text(greeting_task),
        "agent_2_rag_output": task_output_text(rag_task),
        "agent_3_web_output": web_answer,
        "agent_3_was_executed": True,
        "rag_found": rag_found,
        "shown_outputs": ["Local RAG Document", "Web Search"],
    }

    return {
        "greeting": execution_memory["agent_1_greeting"],
        "rag_answer": execution_memory["agent_2_rag_output"],
        "web_answer": web_answer,
        "entry_record": task_output_text(entry_task) or str(crew_output),
        "answer_file": str(ANSWER_LOG_PATH),
        "used_web_search": "True",
        "execution_memory": execution_memory,
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

        .agent-output-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 0.75rem 0 1rem;
            overflow: hidden;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.24);
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(16px);
        }

        .agent-output-table th {
            width: 50%;
            padding: 0.85rem 1rem;
            text-align: left;
            color: #ffffff;
            background: rgba(15, 23, 42, 0.42);
            border-bottom: 1px solid rgba(255, 255, 255, 0.22);
        }

        .agent-output-table td {
            vertical-align: top;
            padding: 1rem;
            color: #f8fafc;
            line-height: 1.55;
            white-space: pre-wrap;
            border-right: 1px solid rgba(255, 255, 255, 0.18);
        }

        .agent-output-table td:last-child,
        .agent-output-table th:last-child {
            border-right: 0;
        }

        .agent-greeting-box {
            margin: 0.8rem 0 0.5rem;
            padding: 0.85rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.13);
            border: 1px solid rgba(255, 255, 255, 0.22);
            color: #f8fafc;
            white-space: pre-wrap;
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


def agent_output_table(rag_output: str, web_output: str) -> str:
    rag_text = html.escape((rag_output or "No Agent 2 output available.").strip())
    web_text = html.escape((web_output or "No Agent 3 output available.").strip())
    return (
        '<table class="agent-output-table">'
        "<thead><tr>"
        "<th>Agent 2: PDF / Local Document Output</th>"
        "<th>Agent 3: Web Search Output</th>"
        "</tr></thead>"
        "<tbody><tr>"
        f"<td>{rag_text}</td>"
        f"<td>{web_text}</td>"
        "</tr></tbody>"
        "</table>"
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
            ("Agent 1: Greeting Agent", "Ready", "Will greet and acknowledge the customer.", "idle"),
            ("Agent 2: RAG Search Agent", "Ready", "Will answer from the local document or say not present.", "idle"),
            ("Agent 3: Web Search Agent", "Ready", "Will run web search for every safe query.", "idle"),
            ("Agent 4: Entry Agent", "Ready", "Receives and saves Agent 1, Agent 2, and Agent 3 outputs.", "idle"),
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
            document and web search for every safe query, then shows both outputs side by side.
        </p>
        <div class="pill-row">
            <span class="pill">💜 Glassmorphism</span>
            <span class="pill">🌈 Animated gradients</span>
            <span class="pill">📚 Agent timeline</span>
            <span class="pill">🌐 Web search</span>
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
serper_status = "Ready" if os.getenv("SERPER_API_KEY") else "Search key missing"
rag_status = "Loaded" if kb_text else "Missing"
last_source = last_run["source"] if last_run else "Waiting"

st.markdown(
    '<div class="status-grid">'
    + glass_card("📄 RAG document", rag_status, DEFAULT_KB_PATH.name)
    + glass_card("🧠 Model", DEFAULT_MODEL, f"OpenAI API: {openai_status}")
    + glass_card("🌐 Web search", serper_status, "Runs for every safe query")
    + glass_card("✅ Google Sheet status", sheet_status, sheet_detail)
    + "</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 📊 Agent status panel")
st.sidebar.markdown(glass_card("Agent 1", "Greeting", "Greets and acknowledges the customer"), unsafe_allow_html=True)
st.sidebar.markdown(glass_card("Agent 2", "RAG Search", "Shows local document answer or not-present message"), unsafe_allow_html=True)
st.sidebar.markdown(glass_card("Agent 3", "Web Search", "Runs for every safe query"), unsafe_allow_html=True)
st.sidebar.markdown(glass_card("Agent 4", "Entry", "Saves Agent 1, Agent 2, and Agent 3 outputs"), unsafe_allow_html=True)
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
        ("Agent 1: Greeting Agent", "Ready", "Will greet and acknowledge the customer.", "idle"),
        ("Agent 2: RAG Search Agent", "Ready", "Will answer from the local document or say not present.", "idle"),
        ("Agent 3: Web Search Agent", "Ready", "Will run web search for every safe query.", "idle"),
        ("Agent 4: Entry Agent", "Ready", "Receives and saves Agent 1, Agent 2, and Agent 3 outputs.", "idle"),
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
        answer_file_value = last_run.get("answer_file", "")
        answer_path = Path(answer_file_value) if answer_file_value else None
        if last_run.get("source") != "Input Guardrail":
            st.markdown("#### Agent Outputs")
            st.markdown(
                '<div class="agent-greeting-box">'
                '<strong>Agent 1: Greeting Output</strong><br>'
                f'{html.escape(last_run.get("greeting", "No Agent 1 output available.")).strip()}'
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                agent_output_table(
                    last_run.get("rag_answer", ""),
                    last_run.get("web_answer", ""),
                ),
                unsafe_allow_html=True,
            )
        with st.expander("Saved Entry", expanded=False):
            if last_run.get("sheet_saved"):
                st.success(last_run.get("sheet_message", "Saved to Google Sheet."))
            else:
                st.warning(last_run.get("sheet_message", "Google Sheet was not updated."))
            st.text(last_run["entry_record"])
        with st.expander("Execution Memory Sent to Agent 4", expanded=False):
            st.json(last_run.get("execution_memory", {}))
        if answer_path and answer_path.exists():
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
        st.success("Web search was used for this safe query.")
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
        st.warning("Web search was not started because the input guardrail blocked the request.")

prompt = st.chat_input("Ask Sony customer care anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    input_safe, input_guardrail_message, input_reasons = run_safety_guardrail(prompt)
    if not input_safe:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "**Input Guardrail Blocked**\n\n"
                    f"{input_guardrail_message}\n\n"
                    "No RAG retrieval, agent execution, web search, file logging, or "
                    "Google Sheet logging was started."
                ),
            }
        )
        st.session_state.last_run = build_blocked_memory(prompt, input_reasons)
        st.session_state.timeline = [
            (
                "Input Guardrail",
                "Blocked",
                "Request stopped before RAG, CrewAI agents, Serper, and logging.",
                "warn",
            ),
            ("Agent 1: Greeting Agent", "Not started", "Blocked before agent execution.", "idle"),
            ("Agent 2: RAG Search Agent", "Not started", "Blocked before RAG execution.", "idle"),
            ("Agent 3: Web Search Agent", "Not started", "Blocked before web search.", "idle"),
            ("Agent 4: Entry Agent", "Not started", "Blocked before persistence.", "idle"),
        ]
        st.rerun()

    if missing_keys:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "I need `OPENAI_API_KEY` in the `.env` file before I can run the agents.",
            }
        )
        st.rerun()

    rag_context, rag_found = retrieve_rag_context(prompt, kb_text)
    use_web_search = True
    web_missing_keys = require_environment(include_serper=True)

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
        "⚡ Retrieving Sony document context and running RAG plus web search..."
        if rag_found
        else "⚡ Local document has no strong match, running RAG not-present flow plus web search..."
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
                rag_found=rag_found,
            )
    except Exception as exc:
        loading_placeholder.empty()
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Agent run failed: `{exc}`"}
        )
        st.rerun()

    loading_placeholder.empty()

    final_answer = (
        "### RAG / Local Document Answer\n"
        f"{result['rag_answer']}\n\n"
        "### Web Search Answer\n"
        f"{result['web_answer']}"
    )
    final_source = "Local RAG Document + Web Search"
    references = extract_urls(result["web_answer"])
    web_status = "Web search used for this safe query. RAG output was also generated."
    output_guardrail_text = "\n\n".join(
        [
            str(final_answer),
            str(result["entry_record"]),
            str(result["execution_memory"]),
        ]
    )
    output_safe, output_guardrail_message, output_reasons = run_safety_guardrail(
        output_guardrail_text
    )
    if output_safe:
        persist_answer_log(str(result["entry_record"]))
        sheet_saved, sheet_message = append_google_sheet(
            query=prompt.strip(),
            source=final_source,
            answer=final_answer,
            web_status=web_status,
            references=references,
            entry_record=result["entry_record"],
        )
    else:
        final_answer = output_guardrail_message
        final_source = "Output Guardrail"
        web_status = (
            "Output guardrail blocked the generated answer after agent execution."
        )
        references = []
        result["entry_record"] = (
            f"Query: {prompt.strip()}\n"
            "Guardrail: output blocked after agent execution\n"
            f"Reason: {', '.join(output_reasons)}\n"
            f"Response: {output_guardrail_message}"
        )
        result["execution_memory"]["output_guardrail"] = "blocked"
        result["execution_memory"]["output_guardrail_reasons"] = output_reasons
        persist_answer_log(str(result["entry_record"]))
        sheet_saved = False
        sheet_message = (
            "Google Sheet skipped because output guardrail blocked the generated answer. "
            "A sanitized local guardrail record was saved."
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"**Agent 1: Greeting Output**\n\n{result['greeting']}\n\n"
                f"**Sources:** {final_source}\n\n{final_answer}\n\n"
                "The same Agent 2 and Agent 3 outputs are shown below in a side-by-side table.\n\n"
                f"**Google Sheet:** {sheet_message}"
            ),
        }
    )

    st.session_state.last_run = {
        "source": final_source,
        "rag_context": rag_context,
        "rag_found": rag_found,
        "used_web_search": True,
        "greeting": result["greeting"],
        "rag_answer": result["rag_answer"],
        "web_answer": result["web_answer"],
        "references": references,
        "entry_record": result["entry_record"],
        "answer_file": result["answer_file"],
        "execution_memory": result["execution_memory"],
        "input_guardrail": "passed",
        "output_guardrail": "passed" if output_safe else "blocked",
        "output_guardrail_reasons": output_reasons,
        "sheet_saved": sheet_saved,
        "sheet_message": sheet_message,
    }

    if not output_safe:
        st.session_state.timeline = [
            ("Input Guardrail", "Passed", "Request was allowed into the agent flow.", "good"),
            ("Agent 1: Greeting Agent", "Completed", "Greeting saved in execution memory.", "good"),
            (
                "Agent 2: RAG Search Agent",
                "Completed",
                "Local document answer or not-present message completed before output validation.",
                "good" if rag_found else "warn",
            ),
            (
                "Agent 3: Web Search Agent",
                "Completed",
                "Web search completed before output validation.",
                "good",
            ),
            (
                "Output Guardrail",
                "Blocked",
                "Generated content was replaced before display and external logging.",
                "warn",
            ),
            ("Agent 4: Entry Agent", "Sanitized", "Unsafe output was not persisted externally.", "warn"),
        ]
    elif rag_found:
        st.session_state.timeline = [
            ("Input Guardrail", "Passed", "Request was allowed into the agent flow.", "good"),
            ("Agent 1: Greeting Agent", "Completed", "Greeting saved in execution memory.", "good"),
            ("Agent 2: RAG Search Agent", "Matched", "Relevant Sony document sections were used.", "good"),
            ("Agent 3: Web Search Agent", "Completed", "Web search answer was generated too.", "good"),
            ("Agent 4: Entry Agent", "Saved", "Agent 1, Agent 2, and Agent 3 outputs saved to answers.txt.", "good"),
            ("Output Guardrail", "Passed", "Final answer passed post-agent validation.", "good"),
            (
                "Google Sheet",
                "Saved" if sheet_saved else "Skipped",
                sheet_message,
                "good" if sheet_saved else "warn",
            ),
        ]
    else:
        st.session_state.timeline = [
            ("Input Guardrail", "Passed", "Request was allowed into the agent flow.", "good"),
            ("Agent 1: Greeting Agent", "Completed", "Greeting saved in execution memory.", "good"),
            ("Agent 2: RAG Search Agent", "Not present", "Local document did not contain enough relevant content.", "warn"),
            ("Agent 3: Web Search Agent", "Completed", "Serper web search answer was generated.", "good"),
            ("Agent 4: Entry Agent", "Saved", "Agent 1, Agent 2, and Agent 3 outputs saved to answers.txt.", "good"),
            ("Output Guardrail", "Passed", "Final answer passed post-agent validation.", "good"),
            (
                "Google Sheet",
                "Saved" if sheet_saved else "Skipped",
                sheet_message,
                "good" if sheet_saved else "warn",
            ),
        ]

    st.rerun()
