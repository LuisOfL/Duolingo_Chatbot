# Duolingo : Support Data RAG System

Hello and welcome to the project! This repository contains a **RAG (Retrieval-Augmented Generation)** based solution specifically designed to empower and streamline support in **Duolingo Thrive**.
![Alt text if image fails to load](img/duolingo.png)
---

## 🚀 Project Overview

The goal of this project is to connect a language model with Duolingo Thrive's support data and knowledge base. Thanks to the RAG architecture, the system is capable of searching for accurate information within help documents and generating contextual, precise, and fast responses for users or support agents.

### Key Features
- **Semantic Search:** Retrieval of relevant support fragments using vector embeddings.
- **Contextual Generation:** Precise answers based exclusively on official Duolingo Thrive documentation.
- **Efficiency:** Reduced response times for support inquiries.

---

## 🏗️ Architecture and Workflow

The RAG system workflow operates in the following steps:

1. **Data Ingestion and Processing:** Duolingo Thrive support documents (PDFs, Markdown, FAQs) are cleaned and split into chunks.
2. **Vectorization (Embeddings):** Each chunk is converted into a numerical vector using an embedding model and stored in a vector database.
3. **Retrieval:** When a user makes a query, the system searches for the vector fragments most semantically relevant to the question.
4. **Generation:** The user query along with the retrieved context are sent to the LLM to draft a coherent and well-founded response.


[ User Query ]
        │
        ▼
[ Vector Searcher ] ──( Retrieves context )──> [ Vector Database ]
        │
        ▼
[ LLM (Generation) ] <──( Context + Question )
        │
        ▼
[ Final Support Response ]
```

---

