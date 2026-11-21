"""
LAGOS STATE PUBLIC SERVICE RULES AI ASSISTANT

An interactive AI assistant that helps Lagos State
civil servants and prospective public officers understand
the Lagos State Public Service Rules.

The assistant retrieves information from official
service documents and provides grounded answers.

Core areas:
- Recruitment and Appointment
- Confirmation of Appointment
- Promotion and Advancement
- Leave and Absence
- Discipline and Misconduct
- Retirement and Pension
- General Public Service Regulations

Run with:
streamlit run streamlit_app.py
"""

import streamlit as st

from rag.config import DATA_FILE_PATHS
from rag.pipeline import (
    Pipeline,
    ask_assistant,
    build_pipeline,
    stream_answer,
)


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lagos State Public Servant AI Assistant",
    page_icon="📘",
    layout="centered",
)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

st.session_state.setdefault("messages", [])

ASSISTANT_AVATAR = "📘"


# ---------------------------------------------------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# WELCOME MESSAGE
# ---------------------------------------------------------------------------

WELCOME = (
    "Welcome to the **Lagos State Public Service AI Assistant**.\n\n"
    "Ask me questions about recruitment, appointment, confirmation, "
    "promotion, leave, discipline, retirement and other public service "
    "regulations.\n\n"
    "My answers are generated using the official documents available "
    "in my knowledge base. Always verify important administrative "
    "decisions against applicable official rules and circulars."
)


# ---------------------------------------------------------------------------
# NEW CHAT FUNCTION
# ---------------------------------------------------------------------------

def new_chat() -> None:
    """Clear the current conversation."""
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# DOCUMENT FINGERPRINT
# ---------------------------------------------------------------------------

def document_file_hash() -> str:
    """
    Generate a document fingerprint using its
    modification time and file size.
    """

    return (
        f"{DATA_FILE_PATHS.stat().st_mtime_ns}-"
        f"{DATA_FILE_PATHS.stat().st_size}"
    )


# ---------------------------------------------------------------------------
# BUILD RAG PIPELINE
# ---------------------------------------------------------------------------

@st.cache_resource(max_entries=2)
def get_pipeline(file_hash: str) -> Pipeline:
    """
    Build and cache the RAG pipeline.

    The pipeline is reused instead of rebuilding
    the document embeddings on every interaction.
    """

    return build_pipeline()


# ---------------------------------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------------------------------

head_left, head_right = st.columns(
    [0.85, 0.15],
    vertical_alignment="center",
)

head_left.markdown(
    "## :material/account_balance: Lagos Public Service AI"
)

head_right.button(
    "New chat",
    on_click=new_chat,
)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:

    st.title("📘 Public Servant AI")

    st.caption("Lagos State Government")

    st.divider()

    st.subheader("Knowledge Areas")

    st.markdown("""
    - Recruitment and Appointment
    - Confirmation of Appointment
    - Promotion and Advancement
    - Leave and Absence
    - Discipline and Misconduct
    - Retirement and Pension
    - Public Service Regulations
    """)

    st.divider()

    st.subheader("Knowledge Base")

    if DATA_FILE_PATHS.exists():
        st.success("Knowledge document detected")
        st.caption(f"Source: {DATA_FILE_PATHS.name}")
    else:
        st.error("Knowledge document not found")

    st.divider()

    st.caption(
        "Designed to support access to Lagos State "
        "Public Service information."
    )

    st.caption(
        "AI-generated responses should be verified "
        "against applicable official documents."
    )


# ---------------------------------------------------------------------------
# INITIALIZE PIPELINE
# ---------------------------------------------------------------------------

try:

    with st.spinner("Initializing Public Service AI Assistant..."):

        if not DATA_FILE_PATHS.exists():
            st.error(
                f"Knowledge document not found: {DATA_FILE_PATHS}",
                icon=":material/error:",
            )
            st.stop()

        pipeline = get_pipeline(document_file_hash())

except ConfigError as e:

    st.error(
        str(e),
        icon=":material/key_off:",
    )

    st.stop()

except Exception as e:

    st.error(
        f"{e.__class__.__name__}: {e}",
        icon=":material/error:",
    )

    st.stop()


# ---------------------------------------------------------------------------
# CHAT MESSAGE DISPLAY
# ---------------------------------------------------------------------------

def render_message(message: dict) -> None:
    """Display an existing conversation message."""

    avatar = (
        ASSISTANT_AVATAR
        if message["role"] == "assistant"
        else None
    )

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):

        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# GENERATE AI RESPONSE
# ---------------------------------------------------------------------------

def generate_reply(
    pipeline: Pipeline,
    question: str,
    history: list[dict],
) -> str:

    """
    Generate an answer using the RAG pipeline.

    Stream the response when supported.
    Fall back to a normal response if streaming
    does not return any content.
    """

    box = st.empty()

    parts = []

    with st.status(
        ":shimmer[Searching Public Service documents...]",
        expanded=False,
    ) as status:

        for piece in stream_answer(
            pipeline,
            question,
            history,
        ):

            parts.append(piece)

            box.markdown("".join(parts))

        answer = "".join(parts)

        status.update(
            label="Document search completed",
            state="complete",
        )

    if not answer:

        answer = ask_assistant(
            pipeline,
            question,
            history,
        )

        box.markdown(answer)

    return answer


# ---------------------------------------------------------------------------
# WELCOME SCREEN AND SUGGESTIONS
# ---------------------------------------------------------------------------

if not st.session_state.messages:

    st.markdown("### Welcome to Lagos State Public Service AI")

    st.markdown(WELCOME)

    st.markdown("### What would you like to know?")

    suggestions = st.container()

    selected = suggestions.pills(
        "Suggested questions",
        list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
    )

    if selected:

        prompt = SUGGESTIONS[selected]

    else:

        prompt = None

else:

    prompt = None


# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------

user_prompt = st.chat_input(
    "Ask a question about Lagos State Public Service Rules..."
)


if user_prompt:

    prompt = user_prompt


# ---------------------------------------------------------------------------
# DISPLAY PREVIOUS CONVERSATION
# ---------------------------------------------------------------------------

for message in st.session_state.messages:

    render_message(message)


# ---------------------------------------------------------------------------
# PROCESS USER QUESTION
# ---------------------------------------------------------------------------

if prompt:

    # Save the user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # Build conversation history
    history = [
        {
            "role": message["role"],
            "content": message["content"],
        }

        for message in st.session_state.messages[:-1]
    ]

    # Generate assistant response
    with st.chat_message(
        "assistant",
        avatar=ASSISTANT_AVATAR,
    ):

        try:

            with st.spinner(
                "Reviewing relevant Public Service Rules..."
            ):

                answer = generate_reply(
                    pipeline,
                    prompt,
                    history,
                )

        except Exception as e:

            st.error(
                f"The assistant encountered an error: "
                f"{e.__class__.__name__}: {e}",
                icon=":material/error:",
            )

            answer = None

    # Save successful response
    if answer:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )