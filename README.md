# Sony Customer Care CrewAI RAG App

This project is a Streamlit customer-support app built with CrewAI. It uses a
local Sony customer-care document first, then falls back to web search only when
the local document does not contain a strong match.

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
app.py
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

Create a `.env` file in the same folder as `app.py`.

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
4. Save it beside `app.py` as `service-account.json`, or use an absolute path in
   `GOOGLE_APPLICATION_CREDENTIALS`.
5. Open the JSON file and copy the service-account email.
6. Share your Google Sheet with that email as an editor.
7. Put the spreadsheet ID in `GOOGLE_SHEET_ID`.

The app creates the worksheet automatically if it does not already exist.

## Run

```powershell
streamlit run app.py
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

## Notes

- Do not commit `.env`.
- Do not commit `service-account.json`.
- Do not commit `answers.txt` if it contains user queries or API output.
- The local RAG matcher ignores generic words such as `sony`, `customer`,
  `care`, and `support`, so general company-history questions use web search
  instead of incorrectly matching the source document.

flowchart TD

    A[Customer Opens Streamlit Application] --> B[Customer Enters Sony Support Query]

    B --> C[RAG Retriever]

    C --> D[Load Local Knowledge Base]
    D --> E[Split Document into Sections]
    E --> F[Tokenize Customer Query]
    F --> G[Calculate Relevance Score]

    G --> H{Relevant RAG Context Found?}

    H -- Yes --> I[Assistant Agent]
    I --> J[Generate Answer from Local Knowledge Base]

    H -- No --> K[Web Search Assistant]
    K --> L[Search Using Serper API]
    L --> M[Prefer Official Sony Support Sources]
    M --> N[Generate Web-Based Answer]

    J --> O[Select Final Answer]
    N --> O

    O --> P[Entry Agent]

    P --> Q[Create Support Record]
    Q --> R[Save Record to answers.txt]

    Q --> S{Google Sheets Configured?}

    S -- Yes --> T[Authenticate Using Service Account]
    T --> U[Append Query and Answer to Google Sheet]

    S -- No --> V[Skip Google Sheet Logging]

    R --> W[Update Agent Timeline]
    U --> W
    V --> W

    W --> X[Display Final Answer in Streamlit]
    X --> Y[Show Source, References and Save Status]


Multi-Agent Architecture

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

RAG Decision Architecture

flowchart LR

    A[Customer Query] --> B[Tokenize Query]

    C[Knowledge Base TXT File] --> D[Split into Sections]
    D --> E[Tokenize Sections]

    B --> F[Compare Query Terms]
    E --> F

    F --> G[Calculate Score]

    G --> H{Score Above Threshold?}

    H -- Yes --> I[Return Top Relevant Sections]
    I --> J[Assistant Agent Answer]

    H -- No --> K[Return No Strong Match]
    K --> L[Serper Web Search]
    L --> M[Web Search Assistant Answer]

CrewAI Sequential Workflow

sequenceDiagram

    participant U as Customer
    participant UI as Streamlit UI
    participant R as RAG Retriever
    participant A as Assistant Agent
    participant W as Web Search Agent
    participant E as Entry Agent
    participant T as answers.txt
    participant G as Google Sheets

    U->>UI: Submit customer-care query
    UI->>R: Retrieve relevant local context

    alt Relevant context found
        R->>A: Send query and RAG context
        A->>A: Generate local support answer
        A->>W: Pass task context
        W->>W: Skip web search
    else No relevant context found
        R->>A: Send query with insufficient context
        A->>W: Request web fallback
        W->>W: Search using Serper API
        W->>W: Generate current web-based answer
    end

    A->>E: Pass assistant output
    W->>E: Pass web-search output
    E->>T: Save support record
    E->>G: Append support log
    E->>UI: Return final record and save status
    UI->>U: Display answer and source

Simple End-to-End Flow


Customer Query
      │
      ▼
Streamlit Frontend
      │
      ▼
Local RAG Retrieval
      │
      ▼
Is Relevant Context Available?
      │
      ├── Yes ──► Assistant Agent ──► Local RAG Answer
      │
      └── No ───► Web Search Agent ──► Serper API Answer
                                              │
                         ┌────────────────────┘
                         ▼
                  Final Answer
                         │
                         ▼
                    Entry Agent
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         answers.txt          Google Sheets
              │                     │
              └──────────┬──────────┘
                         ▼
             Display Result in Streamlit    