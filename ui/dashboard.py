"""
Candidate Ranking and Batch Analytics Dashboard for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 6 Candidate Ranking & Analytics
"""

from typing import List

import pandas as pd
import streamlit as st

from config.settings import settings
from src.analytics.export import export_candidates_to_csv_bytes
from src.analytics.ranking import rank_candidates


def render_ranking_dashboard() -> None:
    """Render the Candidate Ranking and Analytics interactive UI."""
    st.markdown('<div class="main-header">🏆 Candidate Ranking & Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Batch Semantic Matching, Skill Gap Analysis, and Automated Candidate Leaderboards</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "ℹ️ **Evaluation Methodology**: Similarity scores reflect scaled vector cosine similarity (0–100 scale), "
        "not probabilities or calibrated hiring predictions. Technical skill mentions reflect documented terms extracted "
        "from submitted resume files."
    )

    # Input Section
    col_jd, col_upload = st.columns([1, 1], gap="medium")

    with col_jd:
        st.subheader("1. Target Job Description")
        jd_text = st.text_area(
            "Paste target job description:",
            height=240,
            placeholder="Paste target job requirements, qualifications, and role responsibilities here...",
            key="jd_input_text",
        )

    with col_upload:
        st.subheader("2. Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload multiple resumes (PDF, DOCX, TXT):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resume_file_uploader",
            help=f"Supported extensions: {', '.join(settings.SUPPORTED_EXTENSIONS)}. Max file size: {settings.MAX_FILE_SIZE_MB}MB.",
        )

    with st.expander("⚙️ Advanced Scoring & Model Configuration", expanded=False):
        c_weight1, c_weight2, c_method = st.columns(3)
        with c_weight1:
            semantic_weight = st.slider(
                "Semantic Similarity Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(settings.SEMANTIC_WEIGHT),
                step=0.05,
                help="Relative weight of dense contextual text similarity in the overall score.",
            )
        with c_weight2:
            skill_weight = round(1.0 - semantic_weight, 2)
            st.metric("Skill Coverage Weight", f"{skill_weight:.2f}", help="Calculated automatically (1.0 - Semantic Weight).")
        with c_method:
            matching_method = st.selectbox(
                "Embedding Backend",
                options=["auto", "sentence-transformers", "tfidf"],
                index=0,
                help="'auto' uses deep SentenceTransformers if available, with pure-NumPy TF-IDF fallback.",
            )

    st.write("")
    run_ranking = st.button("🚀 Match & Rank Candidates", type="primary", use_container_width=True)

    if run_ranking:
        if not jd_text or not jd_text.strip():
            st.error("⚠️ Please provide a target job description before running candidate ranking.")
            return

        if not uploaded_files:
            st.error("⚠️ Please upload at least one candidate resume (PDF, DOCX, or TXT).")
            return

        with st.spinner("Processing documents, normalizing text, and computing semantic matches..."):
            try:
                batch_result = rank_candidates(
                    resumes=uploaded_files,
                    job_description_text=jd_text,
                    semantic_weight=semantic_weight,
                    skill_weight=skill_weight,
                    matching_method=matching_method,
                )
                st.session_state["ranking_result"] = batch_result
                st.success("✅ Batch candidate evaluation completed successfully!")
            except Exception as exc:
                st.error(f"❌ Batch processing failed: {exc}")
                return

    # Render results if available in session state
    result = st.session_state.get("ranking_result")
    if not result or not result.candidates:
        return

    analytics = result.analytics

    st.divider()

    # Aggregate Metric Cards
    st.subheader("📊 Batch Summary Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Submitted", analytics.total_submitted)
    with m2:
        st.metric("Processed Successfully", analytics.total_processed)
    with m3:
        st.metric("Processing Failures", analytics.total_failed)
    with m4:
        avg_display = f"{analytics.mean_overall_score:.1f}" if analytics.mean_overall_score is not None else "N/A"
        st.metric("Average Overall Score", avg_display)
    with m5:
        med_display = f"{analytics.median_overall_score:.1f}" if analytics.median_overall_score is not None else "N/A"
        st.metric("Median Overall Score", med_display)

    # Failures alert
    if analytics.total_failed > 0:
        with st.expander(f"⚠️ Notice: {analytics.total_failed} candidate file(s) failed extraction", expanded=False):
            failed_list = [c for c in result.candidates if c.status == "failed"]
            for f_cand in failed_list:
                st.markdown(f"- **{f_cand.filename}**: `{f_cand.error_message}`")

    st.divider()

    # Ranked Candidates Leaderboard
    st.subheader("🏆 Candidate Ranking Leaderboard")

    table_data = []
    for c in result.candidates:
        rank_str = str(c.rank) if c.rank is not None else "Failed"
        overall_str = f"{c.overall_score:.1f}" if c.overall_score is not None else "N/A"
        sim_str = f"{c.similarity_score:.1f}" if c.similarity_score is not None else "N/A"
        skill_str = f"{c.skill_coverage_score:.1f}%" if c.skill_coverage_score is not None else "N/A"
        matched_str = ", ".join(c.matched_skills) if c.matched_skills else "(None)"

        table_data.append({
            "Rank": rank_str,
            "Candidate File": c.filename,
            "Overall Score": overall_str,
            "Semantic Sim": sim_str,
            "Skill Coverage": skill_str,
            "Matched Skills": matched_str,
            "Status": "✅ Success" if c.status == "success" else "❌ Failed",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Visualizations: Score Distribution & Skill Gap
    st.divider()
    st.subheader("📈 Batch Visual Analytics")

    v_col1, v_col2 = st.columns(2, gap="large")

    with v_col1:
        st.markdown("**Overall Score Distribution**")
        dist_data = analytics.score_distribution
        if any(dist_data.values()):
            chart_df = pd.DataFrame(
                list(dist_data.items()),
                columns=["Score Range", "Candidate Count"]
            ).set_index("Score Range")
            st.bar_chart(chart_df)
        else:
            st.info("No successful score distribution to display.")

    with v_col2:
        st.markdown("**Skill Gap Analysis**")
        sg_col1, sg_col2 = st.columns(2)
        with sg_col1:
            st.markdown("*Top Matched Skills*")
            if analytics.top_matched_skills:
                for skill, count in analytics.top_matched_skills[:6]:
                    st.markdown(f"- **{skill}**: {count} candidates")
            else:
                st.caption("No matching skills detected.")

        with sg_col2:
            st.markdown("*Top Missing Skills*")
            if analytics.top_missing_skills:
                for skill, count in analytics.top_missing_skills[:6]:
                    st.markdown(f"- **{skill}**: {count} missing")
            else:
                st.caption("No missing skills identified.")

    # Candidate Detail Expander
    st.divider()
    st.subheader("🔍 Individual Candidate Assessment Details")
    successful_cands = [c for c in result.candidates if c.status == "success"]

    if successful_cands:
        cand_names = [f"Rank #{c.rank}: {c.filename} (Score: {c.overall_score:.1f})" for c in successful_cands]
        selected_idx = st.selectbox("Select a candidate to inspect evidence:", range(len(cand_names)), format_func=lambda i: cand_names[i])
        selected_cand = successful_cands[selected_idx]

        st.markdown(f"#### Evaluation for `{selected_cand.filename}`")
        if selected_cand.match_result:
            st.markdown(f"**Explanation:** {selected_cand.match_result.explanation}")

        det_c1, det_c2, det_c3 = st.columns(3)
        with det_c1:
            st.markdown(f"**Method:** `{selected_cand.method}`")
            st.markdown(f"**Fallback Active:** `{'Yes' if selected_cand.is_fallback else 'No'}`")
        with det_c2:
            st.markdown(f"**Matched Skills ({len(selected_cand.matched_skills)}):**")
            st.write(selected_cand.matched_skills or "(None)")
        with det_c3:
            st.markdown(f"**Missing Skills ({len(selected_cand.missing_skills)}):**")
            st.write(selected_cand.missing_skills or "(None)")

    # CSV Download Section
    st.divider()
    st.subheader("📥 Export Evaluation Report")
    st.write("Download the comprehensive candidate ranking report in RFC 4180 compliant CSV format:")

    csv_bytes = export_candidates_to_csv_bytes(result.candidates)
    st.download_button(
        label="📥 Download Candidate Ranking Report (CSV)",
        data=csv_bytes,
        file_name="resumematch_candidate_ranking_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
