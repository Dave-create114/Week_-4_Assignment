#Import and Initialization
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

import os 
import requests
import sys
import warnings

#getting standard out put from windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#silence noise from langchain
warnings.filterwarnings("ignore",category=DeprecationWarning, module="langchain_community")

#load env
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
JINA_KEY = os.getenv("JINA_API_KEY")

print("GROQ_API_KEY loaded:", bool(GROQ_KEY))
print("JINA_API_KEY loaded:", bool(JINA_KEY))

if not GROQ_KEY:
    raise SystemExit("Missing GROQ_API_KEY in .env")

if not JINA_KEY:
    raise SystemExit("Missing JINA_API_KEY in .env")
 
 #Stage 1: Data Ingestion
DATA_FILE_PATH = os.path.join("data", "PublicServiceRules.pdf")

if not os.path.exists(DATA_FILE_PATH):
    raise FileNotFoundError(
        f"PDF file not found: {DATA_FILE_PATH}"
    )

loader = PyPDFLoader(DATA_FILE_PATH)

documents = loader.load()

if not documents:
    raise ValueError("No pages were extracted from the PDF.")

print(
    f"Loaded {len(documents)} pages. "
    f"{len(documents[0].page_content)} characters on the first page."
)

#stage 2: chunking
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size = 800, 
    chunk_overlap=100
    )
chunks =text_splitter.split_documents(documents)

if not chunks:
    raise ValueError("No text chunks were generated.")

print(f"Split into{len(chunks)} chunks")


#stage 3 embedding

# STAGE 3: JINA EMBEDDINGS

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v5-text-small"


class JinaEmbeddings(Embeddings):

    def __init__(self, api_key: str, model: str = JINA_MODEL):
        self.model = model
        self.session = requests.Session()

        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity"
        })

    def _embed(
        self,
        texts: list[str],
        task: str
    ) -> list[list[float]]:

        all_embeddings = []
        batch_size = 64

        for start in range(0, len(texts), batch_size):

            batch = texts[start:start + batch_size]

            response = self.session.post(
                JINA_API_URL,
                json={
                    "model": self.model,
                    "task": task,
                    "normalized": True,
                    "input": batch
                },
                timeout=60
            )

            response.raise_for_status()

            data = response.json()["data"]

            if len(data) != len(batch):
                raise RuntimeError(
                    "Unexpected number of embeddings returned."
                )

            batch_embeddings = [
                item["embedding"]
                for item in sorted(
                    data,
                    key=lambda item: item["index"]
                )
            ]

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        return self._embed(
            texts,
            task="retrieval.passage"
        )

    def embed_query(
        self,
        text: str
    ) -> list[float]:

        return self._embed(
            [text],
            task="retrieval.query"
        )[0]


embeddings_model = JinaEmbeddings(
    api_key=JINA_KEY
)

print("Jina embeddings ready:", embeddings_model.model)

# STAGE 4: FAISS VECTOR DATABASE

vector_store = FAISS.from_documents(
    chunks,
    embeddings_model
)

print(
    f"Indexed {vector_store.index.ntotal} chunks in FAISS."
)

#stage 5
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
print("LLM ready: ", llm.model_name)

#---------------------------------------------------------------------------
# STAGE 6 — THE TOOL
# ---------------------------------------------------------------------------

retriever =vector_store.as_retriever(
    search_kwargs={"k": 3}
)

def search_policy(question: str) -> str:
    """"
    Search the Lagos State Public Service Rules for information
    concerning recruitment, appointment, promotion, confirmation,
    leave, discipline, retirement, termination, salary and other
    public service conditions of employment.
    """
    matching_chunks = retriever.invoke(question)

    if not matching_chunks:
        return "No relevant information was retrieved."

    results = []

    for chunk in matching_chunks:

        page = chunk.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page, int):
            page += 1

        results.append(
            f"Source: Lagos State Public Service Rules\n"
            f"Page: {page}\n"
            f"Content:\n{chunk.page_content}"
        )

    return "\n\n".join(results)

# ---------------------------------------------------------------------------
# STAGE 7 — THE AGENT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT= """
You are an official HR and Policy Assistant specializing
exclusively in the Lagos State Public Service Rules.

Your responsibility is to answer questions accurately using
ONLY information retrieved from the provided document.

CORE RULES:

1. STRICT GROUNDING:
Use only the information retrieved through the search_policy
tool. Never invent rules, timelines, leave days, salary grades,
or disciplinary procedures.

2. MANDATORY RETRIEVAL:
For every substantive question about Lagos State Public
Service Rules, you MUST call search_policy before answering.

3. SOURCE ATTRIBUTION:
Whenever possible, identify the exact Rule number,
Chapter, Section and page number.

4. MISSING INFORMATION:
If the retrieved excerpts do not contain sufficient
information, clearly state:

"I cannot find information regarding this in the provided
Lagos State Public Service Rules document. Please consult
the Lagos State Ministry of Establishments and Training
or your departmental HR unit for official clarification."

5. OBJECTIVITY:
Do not provide personal legal advice or unsupported
interpretations.

6. ANSWER FORMAT:
Start with a direct answer.
Use bullet points for conditions and procedures.
Keep answers clear, concise and professional.

7. FOLLOW-UP QUESTIONS:
Use conversation history to understand follow-up questions,
but always retrieve supporting information from the document.
"""


psr_assistant = create_agent(
    model=llm,
    tools=[search_policy],
    system_prompt=SYSTEM_PROMPT
)

print("\nPublic Service Rules AI Assistant is ready.")

#STAGE 8: QUESTION-ANSWER HELPER
def ask_assistant(
    question: str,
    history: list | None = None
) -> str:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("-" * 60)

    messages = list(history or [])

    messages.append({
        "role": "user",
        "content": question
    })

    response = psr_assistant.invoke({
        "messages": messages
    })

    answer_message = next(
        (
            message
            for message in reversed(response["messages"])
            if getattr(message, "type", None) == "ai"
            and not getattr(message, "tool_calls", None)
        ),
        None
    )

    if answer_message is None:
        raise RuntimeError("No final AI response was generated.")

    answer = answer_message.content

    if not isinstance(answer, str):
        raise RuntimeError("Unexpected response format.")

    print("\nANSWER:")
    print(answer)

    return answer



# ============================================================
# STAGE 9: INTERACTIVE CHAT LOOP
# ============================================================

def chat_loop():

    print("\nWelcome to Lagos State Public Service Rules AI.")
    print("Ask questions about the Public Service Rules.")
    print("Type 'quit' to exit.")
    print("Type 'reset' to clear conversation history.\n")

    history = []

    while True:

        try:
            question = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        if question.lower() == "reset":
            history.clear()
            print("Conversation history cleared.\n")
            continue

        try:

            answer = ask_assistant(
                question,
                history
            )

            history.append({
                "role": "user",
                "content": question
            })

            history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            print(
                f"\n[ERROR] {e.__class__.__name__}: {e}\n"
            )


# ============================================================
# STAGE 10: APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Initial retrieval and answer-generation tests

    ask_assistant(
        "What are the conditions for recruitment into "
        "the Lagos State Public Service?"
    )

    ask_assistant(
        "What are the provisions governing promotion "
        "of public officers?"
    )

    ask_assistant(
        "What are the rules governing annual leave?"
    )

    # Start interactive chatbot

    chat_loop()
