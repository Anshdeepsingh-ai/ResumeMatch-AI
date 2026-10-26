# ðŸ“„ ResumeMatch AI

> **Intelligent Semantic Resume & Job Description Matching Platform**<br>
> *Course:* CSE472 Deep Learning for NLP<br>
> *Milestone:* Day 1 â€” Project Foundation & Architecture<br>
> *Current Branch:* `feature/project-foundation`

---

## ðŸ“– Overview

**ResumeMatch AI** is a Deep Learning-powered Natural Language Processing application designed to intelligently match and rank candidate resumes against job descriptions (JDs).

Traditional keyword-based Applicant Tracking Systems (ATS) fail to recognize synonyms, conceptual equivalence, or nuanced contextual descriptions of work experience. ResumeMatch AI bridges this semantic gap by applying deep contextual text representations, semantic vector similarity, and skill taxonomy matching.

### ðŸŒŸ Planned Capabilities (7-Day Roadmap)
- **Document Extraction (Day 2):** Parse PDF, DOCX, and text resumes preserving semantic layout and sections.
- **Text Preprocessing (Day 3):** Standardize, clean, tokenize, and segment text into experience, education, and skill categories.
- **Deep Contextual NLP (Day 4):** Generate dense sentence and document embeddings using Transformer encoders (e.g., Sentence-BERT).
- **Semantic & Skill Matching (Day 5):** Calculate multidimensional similarity using cosine distance and taxonomy-aware skill matching.
- **Candidate Ranking (Day 6):** Dynamically rank candidates, compute match percentages, and analyze skill gaps.
- **Analytics & Export (Day 6-7):** Interactive visual dashboards, match distribution analytics, and 1-click CSV export.

---

## ðŸ—ï¸ Project Architecture & Directory Layout

The project enforces strict modularity across user interface, configuration, extraction, preprocessing, deep learning, matching, and analytics:

```text
ResumeMatch-AI/
â”œâ”€â”€ .env.example              # Environment configuration template
â”œâ”€â”€ .gitignore                 # Excludes venv, secrets, logs, real resumes, model weights
â”œâ”€â”€ README.md                  # Project documentation & setup instructions
â”œâ”€â”€ requirements.txt           # Minimal production & testing dependencies
â”œâ”€â”€ app.py                     # Main Streamlit application entrypoint
â”œâ”€â”€ config/                    # Centralized settings & configuration
â”‚   â”œâ”€â”€ __init__.py
â”‚   â””â”€â”€ settings.py
â”œâ”€â”€ src/                       # Modular business logic
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ parsers/               # (Day 2) PDF/DOCX document text extractors
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ preprocessing/         # (Day 3) NLP cleaning, tokenization, & segmentation
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ nlp/                   # (Day 4) Deep learning encoders & dense embeddings
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ matching/              # (Day 5) Semantic similarity & skill overlap formulas
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ analytics/             # (Day 6) Candidate ranking & export utilities
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â””â”€â”€ utils/                 # Logging and reusable utilities
â”‚       â”œâ”€â”€ __init__.py
â”‚       â””â”€â”€ logger.py
â”œâ”€â”€ ui/                        # Modular Streamlit frontend components
â”‚   â”œâ”€â”€ __init__.py
â”‚   â””â”€â”€ landing.py
â”œâ”€â”€ tests/                     # Automated unit and smoke test suite
â”‚   â”œâ”€â”€ __init__.py
â”‚   â””â”€â”€ test_app.py
â”œâ”€â”€ docs/                      # Architectural documentation & diagrams
â”‚   â””â”€â”€ architecture.md
â”œâ”€â”€ data/                      # Data storage (git-ignored except sample/)
â”‚   â””â”€â”€ sample/
â”‚       â””â”€â”€ .gitkeep
â””â”€â”€ logs/                      # Runtime log output (git-ignored except .gitkeep)
    â””â”€â”€ .gitkeep
```

Detailed architectural diagrams and pipeline descriptions are documented in [Architecture Documentation](docs/architecture.md).

---

## âš™ï¸ Installation & Setup

### 1. Prerequisites
- **Python:** Python 3.12.14
- **Environment Manager:** Conda or venv
- **Git**

### 2. Activate the Conda Environment (`nlp_env`)

Open a terminal (PowerShell or Bash) and activate the project's designated `nlp_env` environment:

```powershell
# Activate the existing nlp_env conda environment
conda activate nlp_env

# Verify Python version (should report Python 3.12.14)
python --version
```

*(Optional) If setting up from scratch on a new machine:*
```bash
conda create -n nlp_env python=3.12.14 -y
conda activate nlp_env
```

### 3. Install Dependencies

Install the foundation dependencies specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a local `.env` configuration file from the provided `.env.example`:

```powershell
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS Bash
cp .env.example .env
```

You can review or adjust settings in `.env` (e.g., `LOG_LEVEL=INFO`, `DEBUG=true`).

---

## ðŸš€ Running the Application

Launch the Streamlit landing page with:

```bash
streamlit run app.py
```

The Streamlit web server will start and open in your default browser at:
`http://localhost:8501`

---

## ðŸ§ª Running Tests

A comprehensive smoke test suite is included in `tests/test_app.py` verifying centralized configuration, modular package imports, logging, and UI exports.

Run tests using `pytest`:

```bash
# Run the test suite
pytest -v

# Run with full output
pytest -v -s
```

---

## ðŸ”’ Security & Privacy Notice
- Candidate resumes containing Personal Identifiable Information (PII) must **never** be committed to version control.
- All real resumes (`*.pdf`, `*.docx`, `data/resumes/`) are explicitly excluded via `.gitignore`.
- Secrets, credentials, and local `.env` files are strictly git-ignored.

---

## ðŸ“… 7-Day Project Milestone Tracker (CSE472)

| Day | Milestone | Status | Key Deliverable |
| :---: | :--- | :---: | :--- |
| **Day 1** | **Project Foundation & Landing Page** | âœ… **Completed** | Modular structure, settings, logging, smoke tests, Streamlit landing page |
| **Day 2** | Document Extraction Engine | â³ Planned | Multi-format PDF & DOCX text parsing |
| **Day 3** | Text Preprocessing & Cleaning | â³ Planned | Normalization, tokenization, section segmentation |
| **Day 4** | Deep Learning NLP & Embeddings | â³ Planned | Transformer encoders & dense vector representations |
| **Day 5** | Semantic Matching & Skill Alignment | â³ Planned | Cosine similarity & skill taxonomy scoring |
| **Day 6** | Candidate Ranking & Analytics | â³ Planned | Top-K candidate sorting & CSV report export |
| **Day 7** | End-to-End Integration & Polish | â³ Planned | Batch processing pipeline & final UI polish |
