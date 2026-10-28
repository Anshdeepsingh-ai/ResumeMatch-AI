"""
Modular UI Components for ResumeMatch AI Landing Page.
Course: CSE472 Deep Learning for NLP
Milestone: Day 1 Project Foundation
"""

import streamlit as st
from config.settings import settings


def apply_custom_styles() -> None:
    """Inject custom CSS for styling the landing page."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.6rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .feature-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            min-height: 160px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .feature-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1E40AF;
            margin-bottom: 0.5rem;
        }
        .feature-desc {
            font-size: 0.92rem;
            color: #4B5563;
            line-height: 1.45;
        }
        .status-badge-done {
            display: inline-block;
            background-color: #DEF7EC;
            color: #03543F;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            margin-bottom: 0.5rem;
        }
        .status-badge-planned {
            display: inline-block;
            background-color: #E1EFFE;
            color: #1E429F;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            margin-bottom: 0.5rem;
        }
        .pipeline-step {
            background-color: #EFF6FF;
            border-left: 4px solid #2563EB;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            border-radius: 0 6px 6px 0;
            font-size: 0.92rem;
        }
        .pipeline-title {
            font-weight: 700;
            color: #1E3A8A;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the sidebar with project metadata, course info, and runtime status."""
    with st.sidebar:
        st.title("📄 ResumeMatch AI")
        st.caption("Deep Learning for NLP (CSE472)")

        st.divider()

        st.subheader("📌 Project Metadata")
        st.markdown(f"**Application:** `{settings.APP_NAME}`")
        st.markdown(f"**Version:** `{settings.VERSION}`")
        st.markdown(f"**Milestone:** `Day 1 - Foundation`")
        st.markdown(f"**Environment:** `{settings.APP_ENV}`")
        st.markdown(f"**Debug Mode:** `{'Enabled' if settings.DEBUG else 'Disabled'}`")
        st.markdown(f"**Log Level:** `{settings.LOG_LEVEL}`")

        st.divider()

        st.subheader("🧭 Repository Quick Links")
        st.markdown("- [Architecture Docs](file:///docs/architecture.md)")
        st.markdown("- [Configuration Settings](file:///config/settings.py)")
        st.markdown("- [Unit & Smoke Tests](file:///tests/test_app.py)")

        st.divider()
        st.info("💡 **Day 1 Goal Completed**: Centralized architecture, configuration, logging, testing, and UI foundation.")


def render_landing_page(include_sidebar: bool = True) -> None:
    """Render the complete landing page for ResumeMatch AI."""
    apply_custom_styles()
    if include_sidebar:
        render_sidebar()

    # Hero / Header Section
    st.markdown('<div class="main-header">🎯 ResumeMatch AI</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Intelligent Semantic Resume & Job Description Matching Platform | <em>{settings.COURSE}</em></div>',
        unsafe_allow_html=True,
    )

    # Purpose Section
    with st.container():
        st.subheader("💡 Purpose & Motivation")
        st.markdown(
            """
            Traditional applicant tracking systems (ATS) depend heavily on exact keyword searches,
            which frequently penalize qualified candidates who describe their experiences using different
            phrasing or terminology.

            **ResumeMatch AI** solves this limitation by leveraging **Deep Learning for Natural Language Processing (NLP)**.
            By transforming resumes and job descriptions into dense semantic vector representations, the system
            understands context, recognizes conceptual equivalences (e.g., *Deep Learning Engineer* vs. *Neural Network Specialist*),
            and provides objective, transparent matching and ranking for recruitment teams.
            """
        )

    st.divider()

    # Planned Features Section
    st.subheader("🚀 Planned Core Features")
    st.write("ResumeMatch AI is structured as a 7-day deep learning engineering milestone project. Below are the core capabilities being developed:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <span class="status-badge-planned">Day 2 Milestone</span>
                <div class="feature-title">📑 Document Extraction</div>
                <div class="feature-desc">
                    Robust multi-format parser supporting PDF, DOCX, and TXT files. Extracts structured text, contact info, experience, and educational sections.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <span class="status-badge-planned">Day 5 Milestone</span>
                <div class="feature-title">🎯 Skill Matching & Taxonomy</div>
                <div class="feature-desc">
                    Context-aware entity extraction and taxonomy mapping for technical and soft skills, detecting hard requirements and candidate skill gaps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <span class="status-badge-planned">Day 3 Milestone</span>
                <div class="feature-title">🧹 Text Preprocessing</div>
                <div class="feature-desc">
                    Text normalization, noise removal, tokenization, lemmatization, and section-specific segmentation optimized for NLP model inference.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <span class="status-badge-planned">Day 6 Milestone</span>
                <div class="feature-title">🏆 Candidate Ranking</div>
                <div class="feature-desc">
                    Weighted hybrid scoring combining deep semantic similarity, required skill overlap, and experience alignment into ranked candidate leaderboards.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <span class="status-badge-planned">Day 4 Milestone</span>
                <div class="feature-title">🧠 Deep Learning NLP Embeddings</div>
                <div class="feature-desc">
                    Sentence-BERT / Transformer dense contextual embeddings capturing latent semantic relationships beyond simple vocabulary overlap.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <span class="status-badge-planned">Day 6 & 7 Milestone</span>
                <div class="feature-title">📊 Analytics & CSV Export</div>
                <div class="feature-desc">
                    Interactive dashboards, skill coverage heatmaps, match score distribution histograms, and 1-click export of structured evaluation reports to CSV.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # System Architecture Pipeline Preview
    st.subheader("🔄 End-to-End Processing Pipeline")
    st.markdown(
        """
        Below is the conceptual flow of ResumeMatch AI from file ingestion through deep learning inference to final candidate analytics:
        """
    )

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="pipeline-title">1. Ingestion & Document Parsing (src/parsers)</div>
                PDF & DOCX parsing, extraction of text blocks, layout recognition, and sanitization.
            </div>
            <div class="pipeline-step">
                <div class="pipeline-title">2. NLP Cleaning & Segmentation (src/preprocessing)</div>
                Noise elimination, stopword handling, section header detection, and entity isolation.
            </div>
            <div class="pipeline-step">
                <div class="pipeline-title">3. Deep Contextual Vectorization (src/nlp)</div>
                Generation of dense document and sentence embeddings using pre-trained Transformer encoders.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col2:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="pipeline-title">4. Semantic Matching & Skill Alignment (src/matching)</div>
                Cosine distance computation, skill taxonomy overlap, and hybrid multi-factor score calculation.
            </div>
            <div class="pipeline-step">
                <div class="pipeline-title">5. Ranking & Gap Analytics (src/analytics)</div>
                Candidate sorting, skill gap breakdown, matching confidence metrics, and statistics generation.
            </div>
            <div class="pipeline-step">
                <div class="pipeline-title">6. User Interface & Reporting (ui & app.py)</div>
                Interactive Streamlit visualization, candidate comparison, and automated CSV report export.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # 7-Day Roadmap Tracker
    st.subheader("📅 7-Day Project Plan (CSE472)")
    with st.expander("🔍 Click to view 7-Day Milestone Details", expanded=True):
        for item in settings.PLANNED_FEATURES:
            badge_class = "status-badge-done" if item["status"] == "Completed" else "status-badge-planned"
            icon = "✅" if item["status"] == "Completed" else "⏳"
            st.markdown(
                f"""
                **{icon} {item['day']}: {item['name']}**<br>
                <span class="{badge_class}">{item['status']}</span><br>
                {item['description']}
                """,
                unsafe_allow_html=True,
            )
            st.write("")
