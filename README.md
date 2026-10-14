# TNAU Paddy Schemes RAG Assistant

A single-file Python app that answers questions about Tamil Nadu Agricultural University (TNAU) paddy schemes. It uses Playwright to save the webpage once, LangChain to split and retrieve local text, OpenAI for embeddings and answers, SQLite for the local vector index, FastAPI for the API, and Streamlit for the chat interface. LangSmith tracing is optional.

The Streamlit heading is displayed in Tamil:

> 🌾 தமிழ்நாடு வேளாண்மைப் பல்கலைக்கழக நெல் திட்ட உதவியாளர்

## How it works

1. Playwright renders the TNAU schemes page and saves a local HTML copy and extracted text.
2. FastAPI reads the saved text, creates embeddings, and stores the local vector index in SQLite.
3. The Streamlit interface sends questions to FastAPI.
4. LangChain retrieves relevant local text and sends the question with those excerpts to the OpenAI chat model.

On the first FastAPI startup, if the local extracted text is missing or empty, the app scrapes the page and saves it. Later starts use the saved local files without visiting TNAU. OpenAI is still used for embeddings and answer generation; the source page and vector index remain local.

## Requirements

- Python 3.11 or later
- An OpenAI API key
- A LangSmith API key only if you want tracing

## Windows installation

Open Command Prompt in the project folder, for example:

```bat
cd /d "D:\AI\RAG bilathon"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

## Configure environment variables

Create `.env` from the example file:

```bat
copy .env.example .env
```

Edit `.env` and set your own keys. Never share or commit this file.

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=tnau-paddy-schemes-rag

# Streamlit-to-FastAPI address
API_URL=http://127.0.0.1:8000
```

Set `LANGSMITH_TRACING=false` if you do not want tracing. When enabled, questions and the retrieved excerpts may be sent to LangSmith as trace data.

## First-time data setup

You do not need to run `python app.py scrape` separately. Start FastAPI as shown below. If `data/processed/schemes.txt` is missing or empty, FastAPI runs Playwright once and saves the rendered HTML, extracted text, and scrape metadata under `data/`. It then builds the SQLite vector index from that saved text. Keep the `data/` folder on your computer; `.gitignore` excludes it from GitHub.

To deliberately refresh the local copy later, run:

```bat
python app.py scrape --refresh
```

Normal app starts reuse the local text and vector index. The page is scraped again only if the extracted text is missing or empty, or when you deliberately run the refresh command.

## Run the app with one command

The FastAPI backend and Streamlit interface run as separate services. To start both together, save the following as `start_app.bat` in the same folder as `app.py`:

```bat
@echo off
cd /d "%~dp0"

start "FastAPI" cmd /k "call .venv\Scripts\activate.bat && uvicorn app:api --reload --port 8000"
timeout /t 3 /nobreak >nul

call .venv\Scripts\activate.bat
python -m streamlit run app.py --server.port 8503
```



It opens FastAPI in one window and Streamlit in another. Keep both windows open while using the app. Close both windows before starting it again.

Open the Streamlit **Local URL**, usually `http://localhost:8503`.

FastAPI endpoints:

- Health check: `http://127.0.0.1:8000/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`

Example questions:

- What subsidy is available for paddy seed distribution?
- Which districts are mentioned for the rice scheme?
- Who should a farmer contact about agricultural machinery subsidies?

## Keep generated files and secrets out of GitHub

The root `.gitignore` should contain:

```gitignore
.env
.env.*
!.env.example
data/
```

If Git already tracks the local data folder, untrack it without deleting your local files:

```bat
git rm -r --cached data
```

Then commit the `.gitignore` change. If `.env` or `data/` was already pushed, this removes it from the latest repository version; previously committed history may still contain it.

## Troubleshooting

### Streamlit says it cannot reach FastAPI

Confirm the FastAPI command window is still open and reports startup complete. The API should respond at `http://127.0.0.1:8000/health`. Then submit the question again in Streamlit.

### First startup cannot scrape the page

Check that Playwright Chromium is installed with `python -m playwright install chromium` and that the computer has access to the TNAU page. The app needs that access only when it has no saved extracted text. Once saved, normal startup uses local files.

### OpenAI key or authentication error

Check that `.env` is in the project folder and that `OPENAI_API_KEY` is valid. Do not paste the key into screenshots, chat, or GitHub. A ChatGPT subscription is separate from OpenAI API billing.

### The FastAPI root URL returns `{"detail":"Not Found"}`

That is expected because no `/` route is defined. Use `/health` or `/docs` instead.

## Source

TNAU Paddy Schemes page: <https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html#sls>


Then run this single command from Command Prompt:

```bat
start_app.bat
```