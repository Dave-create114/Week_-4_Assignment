"""
LAGOS STATE PUBLIC SERVICE RULES AI ASSISTANT

An interactive AI assistant that helps Lagos State
civil servants and prospective public officers understand
the Lagos State Public Service Rules.

Run with:
streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import streamlit as st

from rag.config import DATA_FILE_PATHS
from rag.pipeline import (
    Pipeline,
    ask_assistant,
    build_pipeline,
    stream_answer,
)

try:
    from rag.config import ConfigError  # type: ignore[attr-defined]
except ImportError:
    class ConfigError(Exception):
        """Raised when required configuration or knowledge files are missing."""


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & CHATGPT-STYLE THEME POLISH
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lagos State Public Servant AI Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Center main conversation like ChatGPT */
        .block-container {
            max-width: 820px;
            padding-top: 1.25rem;
            padding-bottom: 6rem;
        }

        /* Subtle user message bubble styling */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background-color: rgba(128, 128, 128, 0.08);
            border-radius: 16px;
            padding: 0.65rem 1rem;
            margin-left: auto;
            max-width: 85%;
        }

        /* Cleaner assistant message spacing */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background-color: transparent;
            padding: 0.5rem 0.25rem;
        }

        /* Tighten top header spacing */
        div[data-testid="stHorizontalBlock"]:first-of-type {
            border-bottom: 1px solid rgba(128, 128, 128, 0.16);
            padding-bottom: 0.65rem;
            margin-bottom: 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SESSION STATE & CONSTANTS
# ---------------------------------------------------------------------------

st.session_state.setdefault("messages", [])

ASSISTANT_AVATAR = "📘"

SUGGESTIONS = {
    ":blue[:material/person_add:] Appointment": (
        "What are the requirements for appointment into the "
        "Lagos State Public Service?"
    ),
    ":green[:material/verified:] Confirmation": (
        "What are the conditions and procedures for confirmation "
        "of appointment?"
    ),
    ":orange[:material/trending_up:] Promotion": (
        "What are the requirements and procedures for promotion "
        "in the Lagos State Public Service?"
    ),
    ":violet[:material/event_available:] Leave": (
        "What types of leave are available to public officers "
        "and what are the conditions for taking them?"
    ),
    ":red[:material/gavel:] Discipline": (
        "What constitutes misconduct and what disciplinary "
        "procedures apply to public officers?"
    ),
    ":blue[:material/work_history:] Retirement": (
        "What are the rules governing retirement from the "
        "Lagos State Public Service?"
    ),
}

WELCOME_TITLE = "How can I help with Lagos State Public Service Rules?"
WELCOME_SUBTITLE = (
    "Ask about recruitment, appointment, confirmation, promotion, leave, "
    "discipline, or retirement. Answers are grounded in official service "
    "documents—always verify formal administrative actions against current "
    "rules and circulars."
)


# ---------------------------------------------------------------------------
# PATH & FINGERPRINT HELPERS
# ---------------------------------------------------------------------------

def iter_data_paths(raw_paths: Path | str | Iterable[Path | str]) -> list[Path]:
    """Normalize DATA_FILE_PATHS (single path or iterable) into a list of Paths."""
    if isinstance(raw_paths, (Path, str)):
        return [Path(raw_paths)]
    return [Path(p) for p in raw_paths]


def get_missing_files(paths: list[Path]) -> list[Path]:
    """Return paths that do not exist on disk."""
    return [p for p in paths if not p.exists()]


def document_file_hash(paths: list[Path]) -> str:
    """
    Generate a combined fingerprint across all knowledge files using
    modification time and file size so cache invalidates if any file changes.
    """
    parts: list[str] = []
    for path in paths:
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_mtime_ns}-{stat.st_size}")
    return "|".join(parts)


def new_chat() -> None:
    """Clear the current conversation."""
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# BUILD RAG PIPELINE
# ---------------------------------------------------------------------------

@st.cache_resource(max_entries=2)
def get_pipeline(file_hash: str) -> Pipeline:
    """Build and cache the RAG pipeline."""
    return build_pipeline()


data_paths = iter_data_paths(DATA_FILE_PATHS)
missing_files = get_missing_files(data_paths)


# ---------------------------------------------------------------------------
# SIDEBAR (CHATGPT-STYLE NAVIGATION & KNOWLEDGE STATUS)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📘 Public Servant AI")
    st.caption("Lagos State Government • PSR Assistant")

    st.button(
        ":material/add: New chat",
        on_click=new_chat,
        use_container_width=True,
    )

    st.divider()

    st.markdown("**Core Rule Areas**")
    st.markdown(
        """
- Recruitment & Appointment
- Confirmation of Appointment
- Promotion & Advancement
- Leave & Absence
- Discipline & Misconduct
- Retirement & Pension
- General Service Regulations
        """
    )

    st.divider()

    st.markdown("**Knowledge Base**")
    if not missing_files and data_paths:
        st.success(f"{len(data_paths)} document(s) ready")
        for p in data_paths:
            st.caption(f"• `{p.name}`")
    else:
        st.error("Missing knowledge file(s)")
        for p in missing_files:
            st.caption(f"• `{p}`")

    st.divider()
    st.caption(
        "AI-generated answers are for guidance; verify official decisions "
        "against applicable Lagos State Public Service Rules and circulars."
    )


# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------

head_left, head_right = st.columns([0.82, 0.18], vertical_alignment="center")
with head_left:
    st.markdown("#### :material/account_balance: Lagos Public Service AI")
with head_right:
    st.button("New chat", on_click=new_chat, use_container_width=True)


# ---------------------------------------------------------------------------
# INITIALIZE PIPELINE
# ---------------------------------------------------------------------------

pipeline: Pipeline | None = None

try:
    if not data_paths or missing_files:
        formatted = ", ".join(str(p) for p in (missing_files or data_paths))
        raise ConfigError(f"Knowledge document(s) not found: {formatted}")

    with st.spinner("Initializing Public Service AI Assistant..."):
        pipeline = get_pipeline(document_file_hash(data_paths))

except ConfigError as e:
    st.error(str(e), icon=":material/key_off:")
    st.info(
        "Check `DATA_FILE_PATHS` in `rag/config.py` and confirm your data "
        "files are committed to your repository."
    )
    st.stop()

except Exception as e:
    st.error(f"{e.__class__.__name__}: {e}", icon=":material/error:")
    st.stop()


# ---------------------------------------------------------------------------
# CHAT RENDERING & RESPONSE GENERATION
# ---------------------------------------------------------------------------

def render_message(message: dict) -> None:
    """Display an existing conversation message."""
    avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_reply(
    active_pipeline: Pipeline,
    question: str,
    history: list[dict],
) -> str:
    """Stream an answer from the RAG pipeline with fallback to non-streaming."""
    box = st.empty()
    parts: list[str] = []

    with st.status(":shimmer[Searching Public Service documents...]", expanded=False) as status:
        for piece in stream_answer(active_pipeline, question, history):
            parts.append(piece)
            box.markdown("".join(parts))

        status.update(label="Search complete", state="complete")

    answer = "".join(parts).strip()

    if not answer:
        answer = ask_assistant(active_pipeline, question, history)
        box.markdown(answer)

    return answer


# ---------------------------------------------------------------------------
# EMPTY STATE (WELCOME + STARTER PROMPTS) OR HISTORY
# ---------------------------------------------------------------------------

prompt: str | None = None

if not st.session_state.messages:
    st.markdown(f"### {WELCOME_TITLE}")
    st.caption(WELCOME_SUBTITLE)

    selected = st.pills(
        "Suggested questions",
        list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
    )
    if selected:
        prompt = SUGGESTIONS[selected]
else:
    for message in st.session_state.messages:
        render_message(message)


# ---------------------------------------------------------------------------
# CHAT INPUT (PINNED AT BOTTOM)
# ---------------------------------------------------------------------------

user_prompt = st.chat_input("Message Lagos Public Service AI...")
if user_prompt:
    prompt = user_prompt


# ---------------------------------------------------------------------------
# HANDLE NEW TURN
# ---------------------------------------------------------------------------

if prompt and pipeline is not None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            answer = generate_reply(pipeline, prompt, history)
        except Exception as e:
            st.error(
                f"The assistant encountered an error: {e.__class__.__name__}: {e}",
                icon=":material/error:",
            )
            answer = None

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})