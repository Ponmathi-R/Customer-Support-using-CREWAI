# Sony Customer Care CrewAI RAG App

This project is a Streamlit customer-support app built with CrewAI. It uses a
local Sony customer-care document first, then falls back to web search only when
the local document does not contain a strong match.

 
# Multi-Agent Architecture

┌──────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│                                                              │
│  • Customer chat input                                       │
│  • Chat history                                              │
│  • Agent execution timeline                                  │
│  • Source document viewer                                    │
│  • Web references                                            │
│  • Google Sheet status                                       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   QUERY PROCESSING LAYER                     │
│                                                              │
│  Customer Query                                              │
│       │                                                      │
│       ▼                                                      │
│  Local Knowledge Retrieval                                   │
│       │                                                      │
│       ▼                                                      │
│  Relevance Decision                                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│ AGENT 1: ASSISTANT       │    │ AGENT 2: WEB SEARCH         │
│                          │    │ ASSISTANT                    │
│ • Uses local RAG context │    │                              │
│ • Answers support query  │    │ • Uses Serper API           │
│ • Requests missing info  │    │ • Searches current details  │
│ • Avoids false promises  │    │ • Prefers official sources  │
└─────────────┬────────────┘    └──────────────┬───────────────┘
              │                                │
              └──────────────┬─────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                   FINAL ANSWER SELECTION                     │
│                                                              │
│  RAG answer when local context is relevant                   │
│  Web answer when local context is insufficient               │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    AGENT 3: ENTRY AGENT                      │
│                                                              │
│  • Records customer query                                    │
│  • Records answer source                                     │
│  • Records final answer                                      │
│  • Records web-search status                                 │
│  • Records follow-up notes                                   │
└──────────────────────┬───────────────────────┬───────────────┘
                       │                       │
                       ▼                       ▼
          ┌──────────────────────┐   ┌────────────────────────┐
          │ answers.txt          │   │ Google Sheets          │
          │                      │   │                        │
          │ Local support log    │   │ Cloud support log      │
          └──────────────────────┘   └────────────────────────┘


## Features

- Streamlit chat interface
- Glassmorphism frontend with animated gradients
- CrewAI sequential agent flow
- RAG-first answer flow using `sony_customer_care_rag_document.txt`
- Web search fallback using Serper
- Agent execution timeline
- Agent status panel
- Chat history
- Source document viewer
- Web search references tab
- Text log saved to `answers.txt`
- Optional Google Sheet logging

## Project Files

```text
app2.py
requirements.txt
sony_customer_care_rag_document.txt
.env.example
.gitignore
README.md
```

## Setup

Create and activate a virtual environment.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install dependencies.

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the same folder as `app2.py`.

```env
OPENAI_API_KEY=your-openai-key
SERPER_API_KEY=your-serper-key
OPENAI_MODEL_NAME=gpt-4o-mini

GOOGLE_SHEET_ID=your-google-sheet-id
GOOGLE_APPLICATION_CREDENTIALS=service-account.json
GOOGLE_WORKSHEET_NAME=CrewAI Logs
```

`SERPER_API_KEY` is required only when the app needs web search fallback.

## Google Sheet Setup

1. Create a Google Cloud service account.
2. Enable Google Sheets API for the project.
3. Download the service-account JSON key.
4. Save it beside `app2.py` as `service-account.json`, or use an absolute path in
   `GOOGLE_APPLICATION_CREDENTIALS`.
5. Open the JSON file and copy the service-account email.
6. Share your Google Sheet with that email as an editor.
7. Put the spreadsheet ID in `GOOGLE_SHEET_ID`.

The app creates the worksheet automatically if it does not already exist.

## Run

```powershell
streamlit run app2.py
```

## How The App Works

1. The user asks a question in the chat UI.
2. The app searches the local Sony customer-care document.
3. If relevant RAG context is found, the Assistant answers from the local
   document and web search is skipped.
4. If no strong RAG match is found, the Web Search Assistant uses Serper.
5. The Entry Agent saves the support record to `answers.txt`.
6. If Google Sheet variables are configured, the same record is appended to the
   configured worksheet.
 


 
