


## Smart Assistant for Research Summarization (RAG + Gemini Flash 2.0)

This is a locally running intelligent assistant that summarizes research PDFs and allows interactive Q&A or challenge-style quizzes using **Retrieval-Augmented Generation (RAG)**. It integrates:
- **Gemini Flash 2.0 API** for accurate generation
- **FAISS + Sentence Transformers** for document chunk indexing and semantic search
- **Streamlit** for an intuitive web interface

---

## Features

-  Upload PDF/TXT documents
-  Auto-generated document summary
-  Ask questions with answers strictly derived from document content (RAG)
-  "Challenge Me" mode for auto-generated MCQ-style quizzes with strict answer evaluation
-  Highlights source context
-  Fully local embedding & indexing using FAISS
-  Clean, intuitive UI (Streamlit)

---

## Architecture
+-------------------+         +--------------------------+         +-----------------------+
|    Streamlit UI   |  --->   |  FAISS Vector Store +    |  --->   | Gemini Flash 2.0 API  |
| (Upload, Ask, Eval)|         | Sentence-Transformer RAG |         |     (Summary / QA)    |
+-------------------+         +--------------------------+         +-----------------------+


## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

**Required Packages:**

* `keras==2.11.0` and `tensorflow==2.11.0` (compatible with `transformers`)
* `sentence-transformers`, `faiss-cpu` for vector-based chunk search
* `streamlit` for frontend
* `google-generativeai` for Gemini API access
* `python-dotenv` for loading API keys from `.env`
* `PyMuPDF` for fast PDF parsing

---

## API Key Setup

Create a `.env` file in the root folder with the following content:

```
API_KEY=your_gemini_api_key_here
```

---

## Run the App

```
streamlit run app.py
```

---

## How It Works

### Upload

* Accepts `.pdf` and `.txt` files
* Text is chunked (with overlap) and embedded using `all-MiniLM-L6-v2`
* Chunks are indexed into a FAISS vector store

### Ask Anything Mode

* User question is embedded and top relevant chunks are retrieved
* Strict prompt sent to Gemini:

  * low hallucination
  * No outside knowledge
  * Answer only if found in context

### Challenge Me Mode

* Auto-generates 3 quiz questions from chunks
* User submits answers
* Gemini evaluates correctness strictly using chunks
* Feedback includes evaluation, justification, and the correct answer

---

## Limitations

* Gemini Flash 2.0 API has **quota limits** (watch out for 429 errors)
* Context window is limited to **top-k chunks** (default: 5)
* Only tested on English language PDFs

---

## Sample Use Cases

* Research paper summarization
* Educational tools for testing comprehension
* Business report Q\&A
* Personal note review assistant

---

## Folder Structure

```
├── app.py
├── rag_utils.py
├── utils.py
├── requirements.txt
├── .env
└── README.md
```

---

## License

MIT License. Free for academic and non-commercial use.

---

## Author

Built by **Kartik Dwivedi** as a GenAI project using Google's Gemini.


