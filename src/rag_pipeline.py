from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from .config import (CHROMA_PERSIST_DIR, CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR,
                     EMBEDDING_MODEL_NAME, MAX_CONTEXT_DOCS)

try:
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except ImportError:
    logging.warning("langchain is not installed; using a local fallback for document splitting.")
    _HAS_LANGCHAIN = False

    class Document:
        def __init__(self, page_content: str, metadata: dict[str, str] | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators

        def split_text(self, text: str) -> list[str]:
            if len(text) <= self.chunk_size:
                return [text]

            chunks: list[str] = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += self.chunk_size - self.chunk_overlap
            return chunks

        def split_documents(self, documents: list[Document]) -> list[Document]:
            chunked: list[Document] = []
            for document in documents:
                for chunk in self.split_text(document.page_content):
                    chunked.append(Document(page_content=chunk, metadata=document.metadata))
            return chunked

_HAS_CHROMADB = False
_HAS_GEMINI_EMBEDDINGS = False
_HAS_SKLEARN = False

try:
    import chromadb  # noqa: F401
    _HAS_CHROMADB = True
except ImportError:
    logging.warning("chromadb is not installed; falling back to a local search engine.")

try:
    from langchain.embeddings import GoogleGeminiEmbeddings
    from langchain.vectorstores import Chroma
    _HAS_GEMINI_EMBEDDINGS = True
except ImportError:
    logging.warning("GoogleGeminiEmbeddings / Chroma are unavailable; using TF-IDF fallback.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    logging.warning("scikit-learn is not installed; using simple keyword retrieval as a fallback.")

try:
    import PyPDF2  # noqa: F401
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False
    logging.warning("PyPDF2 is not installed; PDF support will be limited.")


class RagPipeline:
    def __init__(self):
        self.documents: list[Document] = []
        self.vectorstore = None
        self._vectorizer = None
        self._document_matrix = None
        self._load_documents()
        self._build_index()

    def _load_documents(self) -> None:
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

        documents: list[Document] = []
        for path in sorted(DATA_DIR.iterdir()):
            if path.is_dir():
                continue
            suffix = path.suffix.lower()
            if suffix in {".txt", ".md"}:
                content = path.read_text(encoding="utf-8", errors="ignore")
                documents.append(Document(page_content=content, metadata={"source": path.name}))
            elif suffix == ".pdf":
                content = self._load_pdf(path)
                if content:
                    documents.append(Document(page_content=content, metadata={"source": path.name}))

        if not documents:
            raise RuntimeError("No documents were loaded for the RAG pipeline.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " "]
        )
        self.documents = splitter.split_documents(documents)

    def _load_pdf(self, path: Path) -> str:
        if not _HAS_PYPDF2:
            logging.warning("Unable to parse PDF without PyPDF2: %s", path)
            return ""

        text = []
        try:
            with path.open("rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text.append(page_text)
        except Exception as exc:
            logging.warning("Failed to load PDF %s: %s", path, exc)
        return "\n\n".join(text)

    def _build_index(self) -> None:
        if _HAS_CHROMADB and _HAS_GEMINI_EMBEDDINGS:
            try:
                embeddings = GoogleGeminiEmbeddings(model=EMBEDDING_MODEL_NAME)
                self.vectorstore = Chroma.from_documents(
                    documents=self.documents,
                    embedding=embeddings,
                    persist_directory=str(CHROMA_PERSIST_DIR),
                )
                self.vectorstore.persist()
                return
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to build Chroma vectorstore: %s", exc)

        if _HAS_SKLEARN:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._document_matrix = self._vectorizer.fit_transform(
                [doc.page_content for doc in self.documents]
            )
        else:
            self._vectorizer = None
            self._document_matrix = None

    def retrieve(self, query: str, k: int = MAX_CONTEXT_DOCS) -> list[Document]:
        if self.vectorstore is not None:
            return self.vectorstore.similarity_search(query, k=k)

        if self._vectorizer is not None and self._document_matrix is not None:
            return self._retrieve_with_tfidf(query, k)

        return self._retrieve_with_keyword_match(query, k)

    def _retrieve_with_tfidf(self, query: str, k: int) -> list[Document]:
        query_matrix = self._vectorizer.transform([query])
        scores = cosine_similarity(query_matrix, self._document_matrix)[0]
        ranked_indices = scores.argsort()[::-1][:k]
        return [self.documents[i] for i in ranked_indices if scores[i] > 0]

    def _retrieve_with_keyword_match(self, query: str, k: int) -> list[Document]:
        normalized_query = query.lower()
        scored: list[tuple[int, int]] = []
        for idx, document in enumerate(self.documents):
            score = sum(1 for token in normalized_query.split() if token in document.page_content.lower())
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [self.documents[idx] for idx, score in scored[:k] if score > 0]

    def get_context(self, query: str, k: int = MAX_CONTEXT_DOCS) -> str:
        results = self.retrieve(query, k=k)
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content.strip()}"
            for doc in results
        )


if __name__ == "__main__":
    pipeline = RagPipeline()
    print(pipeline.get_context("How do I submit a help ticket?"))
