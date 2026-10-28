"""
Batch analytics calculations and statistical aggregations for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 6 Candidate Ranking & Analytics
"""

from collections import Counter
from typing import List, Optional

import numpy as np

from src.analytics.models import BatchAnalytics, CandidateResult


def calculate_batch_analytics(
    candidates: List[CandidateResult],
    jd_skills: Optional[List[str]] = None,
) -> BatchAnalytics:
    """
    Compute statistical distributions, skill frequency rankings, and summary metrics
    for a batch of evaluated candidate resumes.

    Args:
        candidates: List of CandidateResult instances.
        jd_skills: Optional list of target job description canonical skills.

    Returns:
        BatchAnalytics containing mean/median scores, failure counts, and skill gap frequencies.
    """
    total_submitted = len(candidates)
    successful = [
        c for c in candidates
        if c.status == "success" and c.overall_score is not None
    ]
    total_processed = len(successful)
    total_failed = total_submitted - total_processed
    success_rate = (
        round((total_processed / total_submitted) * 100.0, 2)
        if total_submitted > 0
        else 0.0
    )

    empty_distribution = {
        "0-20": 0,
        "21-40": 0,
        "41-60": 0,
        "61-80": 0,
        "81-100": 0,
    }

    if total_processed == 0:
        return BatchAnalytics(
            total_submitted=total_submitted,
            total_processed=0,
            total_failed=total_failed,
            success_rate=success_rate,
            mean_overall_score=None,
            median_overall_score=None,
            min_overall_score=None,
            max_overall_score=None,
            std_overall_score=None,
            mean_similarity_score=None,
            mean_skill_coverage_score=None,
            top_matched_skills=[],
            top_missing_skills=[],
            score_distribution=empty_distribution,
        )

    # Statistical score metrics
    overall_scores = [c.overall_score for c in successful if c.overall_score is not None]
    sim_scores = [c.similarity_score for c in successful if c.similarity_score is not None]
    skill_scores = [c.skill_coverage_score for c in successful if c.skill_coverage_score is not None]

    mean_overall = round(float(np.mean(overall_scores)), 2)
    median_overall = round(float(np.median(overall_scores)), 2)
    min_overall = round(float(np.min(overall_scores)), 2)
    max_overall = round(float(np.max(overall_scores)), 2)
    std_overall = (
        round(float(np.std(overall_scores)), 2)
        if len(overall_scores) > 1
        else 0.0
    )

    mean_sim = round(float(np.mean(sim_scores)), 2) if sim_scores else None
    mean_skill = round(float(np.mean(skill_scores)), 2) if skill_scores else None

    # Skill frequency rankings
    matched_pool: List[str] = [
        skill for c in successful for skill in c.matched_skills
    ]
    top_matched = Counter(matched_pool).most_common(10)

    missing_pool: List[str] = [
        skill for c in successful for skill in c.missing_skills
    ]
    top_missing = Counter(missing_pool).most_common(10)

    # Score distribution partitioning
    distribution = dict(empty_distribution)
    for score in overall_scores:
        if score <= 20.0:
            distribution["0-20"] += 1
        elif score <= 40.0:
            distribution["21-40"] += 1
        elif score <= 60.0:
            distribution["41-60"] += 1
        elif score <= 80.0:
            distribution["61-80"] += 1
        else:
            distribution["81-100"] += 1

    return BatchAnalytics(
        total_submitted=total_submitted,
        total_processed=total_processed,
        total_failed=total_failed,
        success_rate=success_rate,
        mean_overall_score=mean_overall,
        median_overall_score=median_overall,
        min_overall_score=min_overall,
        max_overall_score=max_overall,
        std_overall_score=std_overall,
        mean_similarity_score=mean_sim,
        mean_skill_coverage_score=mean_skill,
        top_matched_skills=top_matched,
        top_missing_skills=top_missing,
        score_distribution=distribution,
    )
