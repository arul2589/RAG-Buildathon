"""Single-file Playwright scraper + LangChain RAG + FastAPI + Streamlit app.

Commands:
    python app.py scrape       # one-time scrape into local data files
    uvicorn app:api --reload   # start FastAPI
    streamlit run app.py       # start Streamlit UI (in another terminal)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import streamlit as st
from starlette.routing import Mount


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = DATA_DIR / "vector_db"
VECTOR_DB_PATH = VECTOR_DIR / "vectors.sqlite3"
HTML_PATH = RAW_DIR / "schemes.html"
TEXT_PATH = PROCESSED_DIR / "schemes.txt"
METADATA_PATH = PROCESSED_DIR / "scrape_metadata.json"
SOURCE_URL = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html#sls"

# Prefer this project's .env over any stale OPENAI_API_KEY set in Windows.
load_dotenv(ROOT / ".env", override=True)


# ---------------------------------------------------------------------------
# One-time Playwright scrape. The app never scrapes automatically.
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _state_schemes_section(text: str) -> str:
    start = text.find("State Level Schemes")
    if start < 0:
        raise RuntimeError("Could not find 'State Level Schemes' on the page.")
    end = text.find("Central Schemes", start + len("State Level Schemes"))
    if end < 0:
        raise RuntimeError("Could not find 'Central Schemes' to mark the section boundary.")
    section = text[start:end].strip()
    if len(section) < 500:
        raise RuntimeError("The extracted section looks too short; refusing to save it.")
    return section


def scrape_once(refresh: bool = False) -> None:
    """Use Chromium via Playwright and save HTML plus state-scheme text locally."""
    if not refresh and (HTML_PATH.exists() or TEXT_PATH.exists()):
        raise FileExistsError(
            "A local scrape already exists; no website request was made. "
            "Use `python app.py scrape --refresh` only to intentionally replace it."
        )

    # Import only for this explicit scraping command, so normal app startup never
    # needs Playwright or makes a network request.
    from playwright.sync_api import sync_playwright

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(SOURCE_URL, wait_until="domcontentloaded", timeout=60_000)
        page.locator("body").wait_for()
        page.wait_for_timeout(800)
        html = page.content()
        visible_text = page.locator("body").inner_text()
        final_url = page.url
        browser.close()

    section = _state_schemes_section(_clean_text(visible_text))
    HTML_PATH.write_text(html, encoding="utf-8")
    TEXT_PATH.write_text(section + "\n", encoding="utf-8")
    metadata = {
        "source_url": SOURCE_URL,
        "final_url": final_url,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "State Level Schemes section on the TNAU paddy page",
        "raw_html": "data/raw/schemes.html",
        "processed_text": "data/processed/schemes.txt",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved rendered page: {HTML_PATH}")
    print(f"Saved extracted text: {TEXT_PATH}")
    print(f"Extracted {len(section):,} characters. The app will use these local files.")


# ---------------------------------------------------------------------------
# Local LangChain retrieval and OpenAI answer generation
# ---------------------------------------------------------------------------

class TNAURAG:
    def __init__(self) -> None:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing. Add it to the .env file.")

        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            # Chunks are small, so send raw text directly and avoid importing a
            # tokenizer package that is unnecessary for these inputs.
            check_embedding_ctx_length=False,
        )
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer only from the supplied excerpts from the locally saved "
                    "TNAU paddy schemes page. If the answer is absent, say you could "
                    "not find it in the saved page. Do not guess or add current policy "
                    "details. State scheme names, amounts, eligibility, and contact "
                    "roles as they appear in the excerpts.",
                ),
                ("human", "Question: {question}\n\nLocal excerpts:\n{context}"),
            ]
        )
        self._ensure_local_index()

    def _ensure_local_index(self) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if not TEXT_PATH.exists():
            raise FileNotFoundError(
                "Local source text is missing. Run `python app.py scrape` once first."
            )
        source_text = TEXT_PATH.read_text(encoding="utf-8").strip()
        if not source_text:
            raise ValueError("The locally saved source text is empty.")

        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        # SQLite is the persistent local vector store. It uses the Python standard
        # library, so this project does not need Chroma or its orjson dependency.
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(VECTOR_DB_PATH) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS index_metadata "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS documents "
                "(id INTEGER PRIMARY KEY, content TEXT NOT NULL, "
                "metadata TEXT NOT NULL, embedding TEXT NOT NULL)"
            )
            row = connection.execute(
                "SELECT value FROM index_metadata WHERE key = 'source_sha256'"
            ).fetchone()
            if row and row[0] == digest:
                return

        # Re-index from the local source text only if absent or changed; never scrape.
        splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=120)
        documents = splitter.create_documents(
            [source_text],
            metadatas=[{"source": SOURCE_URL, "title": "TNAU Paddy Schemes"}],
        )
        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        with sqlite3.connect(VECTOR_DB_PATH) as connection:
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM index_metadata")
            connection.executemany(
                "INSERT INTO documents (content, metadata, embedding) VALUES (?, ?, ?)",
                [
                    (
                        doc.page_content,
                        json.dumps(doc.metadata, ensure_ascii=False),
                        json.dumps(vector),
                    )
                    for doc, vector in zip(documents, vectors)
                ],
            )
            connection.execute(
                "INSERT INTO index_metadata (key, value) VALUES ('source_sha256', ?)",
                (digest,),
            )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def ask(self, question: str) -> dict[str, Any]:
        query_vector = self.embeddings.embed_query(question)
        with sqlite3.connect(VECTOR_DB_PATH) as connection:
            rows = connection.execute(
                "SELECT content, metadata, embedding FROM documents"
            ).fetchall()
        scored = [
            (
                self._cosine_similarity(query_vector, json.loads(vector_json)),
                content,
                json.loads(metadata_json),
            )
            for content, metadata_json, vector_json in rows
        ]
        top_documents = sorted(scored, key=lambda item: item[0], reverse=True)[:4]
        if not top_documents:
            return {
                "answer": "I could not find relevant information in the locally saved TNAU schemes page.",
                "sources": [],
            }
        context = "\n\n---\n\n".join(item[1] for item in top_documents)
        response = self.llm.invoke(
            self.prompt.format_messages(question=question, context=context)
        )
        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer)
        return {
            "answer": answer,
            "sources": [{"title": "TNAU Paddy Schemes", "url": SOURCE_URL}],
        }


# ---------------------------------------------------------------------------
# FastAPI service (start with: uvicorn app:api --reload)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application):
    application.state.rag = TNAURAG()
    yield
    application.state.rag = None


api = FastAPI(
    title="TNAU Paddy Schemes RAG API",
    description="Answers questions from locally saved TNAU schemes data.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@api.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ready" if getattr(api.state, "rag", None) else "starting",
        "source": "local saved TNAU files",
    }


@api.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    rag = getattr(api.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG service is not ready.")
    try:
        return rag.ask(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Streamlit UI (start with: streamlit run app.py)
# ---------------------------------------------------------------------------

def run_streamlit_ui() -> None:
    import requests
    import streamlit as st

    st.set_page_config(
    page_title="தமிழ்நாடு வேளாண்மைப்திட்ட உதவியாளர்",
    page_icon="🌾",
    )

    st.markdown(
    """
    <h1 style="font-size: 1.5rem; line-height: 1.25;">
        🌾 தமிழ்நாடு வேளாண்மைப் திட்ட உதவியாளர்
    </h1>
    """,
    unsafe_allow_html=True,
    )
    st.caption("Ask questions about the schemes in your locally saved TNAU page.")
    st.info(
        "The website is scraped only when you run the scrape command. "
        "Questions are answered using the local vector database."
    )

    with st.form("question_form"):
        question = st.text_input(
            "Your question", placeholder="What subsidy is available for paddy seeds?"
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
            return
        api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
        try:
            with st.spinner("Searching the local schemes data…"):
                result = requests.post(
                    f"{api_url}/ask", json={"question": question}, timeout=120
                )
                result.raise_for_status()
                data = result.json()
            st.markdown("### Answer")
            st.write(data["answer"])
            if data.get("sources"):
                st.markdown("### Source")
                for source in data["sources"]:
                    st.markdown(f"[{source['title']}]({source['url']})")
        except requests.RequestException as exc:
            st.error(
                "Could not reach FastAPI. Start it in another terminal with "
                f"`uvicorn app:api --reload`. Details: {exc}"
            )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "scrape":
    scrape_once(refresh="--refresh" in sys.argv[2:])
else:
    # Streamlit sets a ScriptRunContext while executing `streamlit run app.py`.
    # The context check keeps this same file importable by Uvicorn as `app:api`.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is not None:
            run_streamlit_ui()
    except ImportError:
        pass

# Streamlit UI at /; FastAPI routes under /api/
app = st.App(
    __file__,
    routes=[Mount("/api", app=api)],
)