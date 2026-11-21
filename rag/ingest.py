"""STAGES 1-2: data ingestion and chunking.

Loads the raw public service rule documents from disk and splits them into
 Retrieval-friendly chunks.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_FILE_PATH


def load_documents(
    path: Path = DATA_FILE_PATH,
) -> list[Document]:
    """STAGE 1: read the policy PDF file into LangChain documents."""
    if not path.exists():
        raise FileNotFoundError(
            f"Policy file not found at {path}."
            "Expected PublicServiceRules.pdf inside the data/ folder inside the project folder."
        )
    loader = PyPDFLoader(str(path))
    return loader.load()


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """STAGE 2: split documents into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)
