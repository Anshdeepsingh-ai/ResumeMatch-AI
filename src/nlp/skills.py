"""
Configurable skill taxonomy and extraction module for ResumeMatch AI.
Extracts technical skills with exact and alias vocabulary matching.
Course: CSE472 Deep Learning for NLP
Milestone: Day 4 NLP Feature Extraction
"""

import re
from typing import Any, Dict, List, Optional

from src.nlp.schemas import SkillMention

# Comprehensive default technical skill taxonomy for recruitment NLP
DEFAULT_SKILL_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # 1. Programming Languages
    "Python": {
        "category": "programming_languages",
        "aliases": ["python3", "python 3"],
        "case_sensitive": False,
    },
    "C++": {
        "category": "programming_languages",
        "aliases": ["cpp"],
        "case_sensitive": False,
    },
    "C#": {
        "category": "programming_languages",
        "aliases": ["c-sharp", "csharp"],
        "case_sensitive": False,
    },
    "C": {
        "category": "programming_languages",
        "aliases": ["c language", "c-lang"],
        "case_sensitive": True,
    },
    "Java": {
        "category": "programming_languages",
        "aliases": ["core java"],
        "case_sensitive": False,
    },
    "JavaScript": {
        "category": "programming_languages",
        "aliases": ["js", "es6", "ecmascript"],
        "case_sensitive": False,
    },
    "TypeScript": {
        "category": "programming_languages",
        "aliases": ["ts"],
        "case_sensitive": False,
    },
    "Go": {
        "category": "programming_languages",
        "aliases": ["golang"],
        "case_sensitive": True,
    },
    "Rust": {
        "category": "programming_languages",
        "aliases": [],
        "case_sensitive": False,
    },
    "SQL": {
        "category": "programming_languages",
        "aliases": ["structured query language"],
        "case_sensitive": False,
    },
    "R": {
        "category": "programming_languages",
        "aliases": ["r-lang", "r language"],
        "case_sensitive": True,
    },
    "Ruby": {
        "category": "programming_languages",
        "aliases": [],
        "case_sensitive": False,
    },
    "PHP": {
        "category": "programming_languages",
        "aliases": [],
        "case_sensitive": False,
    },
    "Swift": {
        "category": "programming_languages",
        "aliases": [],
        "case_sensitive": False,
    },
    "Kotlin": {
        "category": "programming_languages",
        "aliases": [],
        "case_sensitive": False,
    },
    "Scala": {
        "category": "programming_languages",
        "aliases": [],
        "case_sensitive": False,
    },
    "Bash": {
        "category": "programming_languages",
        "aliases": ["shell", "bash scripting"],
        "case_sensitive": False,
    },

    # 2. Frameworks & Libraries
    "PyTorch": {
        "category": "frameworks_and_libraries",
        "aliases": ["torch"],
        "case_sensitive": False,
    },
    "TensorFlow": {
        "category": "frameworks_and_libraries",
        "aliases": ["tf"],
        "case_sensitive": False,
    },
    "Keras": {
        "category": "frameworks_and_libraries",
        "aliases": [],
        "case_sensitive": False,
    },
    "Hugging Face": {
        "category": "frameworks_and_libraries",
        "aliases": ["huggingface", "transformers"],
        "case_sensitive": False,
    },
    "Scikit-Learn": {
        "category": "frameworks_and_libraries",
        "aliases": ["sklearn", "scikit learn"],
        "case_sensitive": False,
    },
    "Pandas": {
        "category": "frameworks_and_libraries",
        "aliases": [],
        "case_sensitive": False,
    },
    "NumPy": {
        "category": "frameworks_and_libraries",
        "aliases": [],
        "case_sensitive": False,
    },
    "FastAPI": {
        "category": "frameworks_and_libraries",
        "aliases": ["fast api"],
        "case_sensitive": False,
    },
    "Flask": {
        "category": "frameworks_and_libraries",
        "aliases": [],
        "case_sensitive": False,
    },
    "Django": {
        "category": "frameworks_and_libraries",
        "aliases": [],
        "case_sensitive": False,
    },
    "React": {
        "category": "frameworks_and_libraries",
        "aliases": ["reactjs", "react.js"],
        "case_sensitive": False,
    },
    "Node.js": {
        "category": "frameworks_and_libraries",
        "aliases": ["nodejs", "node.js", "node js"],
        "case_sensitive": False,
    },
    "Vue.js": {
        "category": "frameworks_and_libraries",
        "aliases": ["vuejs", "vue.js", "vue"],
        "case_sensitive": False,
    },
    ".NET": {
        "category": "frameworks_and_libraries",
        "aliases": ["dotnet", ".net core", "asp.net", "asp.net core"],
        "case_sensitive": False,
    },
    "Spring Boot": {
        "category": "frameworks_and_libraries",
        "aliases": ["spring", "springboot"],
        "case_sensitive": False,
    },

    # 3. Cloud & DevOps
    "AWS": {
        "category": "cloud_and_devops",
        "aliases": ["amazon web services"],
        "case_sensitive": False,
    },
    "GCP": {
        "category": "cloud_and_devops",
        "aliases": ["google cloud", "google cloud platform"],
        "case_sensitive": False,
    },
    "Azure": {
        "category": "cloud_and_devops",
        "aliases": ["microsoft azure"],
        "case_sensitive": False,
    },
    "Docker": {
        "category": "cloud_and_devops",
        "aliases": ["dockerization", "containerization"],
        "case_sensitive": False,
    },
    "Kubernetes": {
        "category": "cloud_and_devops",
        "aliases": ["k8s"],
        "case_sensitive": False,
    },
    "Terraform": {
        "category": "cloud_and_devops",
        "aliases": [],
        "case_sensitive": False,
    },
    "CI/CD": {
        "category": "cloud_and_devops",
        "aliases": ["ci-cd", "continuous integration"],
        "case_sensitive": False,
    },
    "Linux": {
        "category": "cloud_and_devops",
        "aliases": ["ubuntu", "debian", "redhat", "centos"],
        "case_sensitive": False,
    },

    # 4. Databases & Data Engineering
    "PostgreSQL": {
        "category": "databases_and_data_engineering",
        "aliases": ["postgres", "psql"],
        "case_sensitive": False,
    },
    "MySQL": {
        "category": "databases_and_data_engineering",
        "aliases": [],
        "case_sensitive": False,
    },
    "MongoDB": {
        "category": "databases_and_data_engineering",
        "aliases": ["mongo"],
        "case_sensitive": False,
    },
    "Redis": {
        "category": "databases_and_data_engineering",
        "aliases": [],
        "case_sensitive": False,
    },
    "Elasticsearch": {
        "category": "databases_and_data_engineering",
        "aliases": ["elastic search"],
        "case_sensitive": False,
    },
    "Apache Spark": {
        "category": "databases_and_data_engineering",
        "aliases": ["spark", "pyspark"],
        "case_sensitive": False,
    },
    "Apache Kafka": {
        "category": "databases_and_data_engineering",
        "aliases": ["kafka"],
        "case_sensitive": False,
    },
    "Snowflake": {
        "category": "databases_and_data_engineering",
        "aliases": [],
        "case_sensitive": False,
    },

    # 5. AI, NLP & Machine Learning Concepts
    "Natural Language Processing": {
        "category": "ai_and_nlp",
        "aliases": ["nlp"],
        "case_sensitive": False,
    },
    "Machine Learning": {
        "category": "ai_and_nlp",
        "aliases": ["ml"],
        "case_sensitive": False,
    },
    "Deep Learning": {
        "category": "ai_and_nlp",
        "aliases": ["dl"],
        "case_sensitive": False,
    },
    "Computer Vision": {
        "category": "ai_and_nlp",
        "aliases": ["cv"],
        "case_sensitive": False,
    },
    "Large Language Models": {
        "category": "ai_and_nlp",
        "aliases": ["llm", "llms"],
        "case_sensitive": False,
    },
    "BERT": {
        "category": "ai_and_nlp",
        "aliases": ["sentence-bert", "sbert"],
        "case_sensitive": False,
    },
    "Retrieval-Augmented Generation": {
        "category": "ai_and_nlp",
        "aliases": ["rag"],
        "case_sensitive": False,
    },
    "Semantic Search": {
        "category": "ai_and_nlp",
        "aliases": ["vector search", "dense retrieval"],
        "case_sensitive": False,
    },

    # 6. Developer Tools
    "Git": {
        "category": "developer_tools",
        "aliases": ["github", "gitlab"],
        "case_sensitive": False,
    },
}


class SkillExtractor:
    """
    Configurable technical skill extraction engine.
    Matches verified canonical skills and aliases without heuristic hallucinations.
    """

    def __init__(self, taxonomy: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize SkillExtractor with a taxonomy.

        Args:
            taxonomy: Optional custom dictionary mapping canonical skill names to
                      {'category': str, 'aliases': List[str], 'case_sensitive': bool}.
                      If None, uses DEFAULT_SKILL_TAXONOMY.
        """
        self.taxonomy = taxonomy if taxonomy is not None else DEFAULT_SKILL_TAXONOMY
        self._compiled_patterns: Dict[str, Dict[str, Any]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for each canonical skill and its aliases."""
        for canon, data in self.taxonomy.items():
            category = data.get("category", "general")
            aliases = data.get("aliases", [])
            case_sens = data.get("case_sensitive", False)
            flags = 0 if case_sens else re.IGNORECASE

            # Compile canonical pattern with boundary protection for single-letter languages
            if canon == "C":
                boundary_after = r"(?![A-Za-z0-9_+#])"
                boundary_before = r"(?<![A-Za-z0-9_])"
            elif canon == "R":
                boundary_after = r"(?![A-Za-z0-9_&])"
                boundary_before = r"(?<![A-Za-z0-9_&])"
            else:
                boundary_before = r"(?<![A-Za-z0-9_])"
                boundary_after = r"(?![A-Za-z0-9_])"

            p_canon = re.compile(
                boundary_before + re.escape(canon) + boundary_after,
                flags=flags,
            )

            # Compile alias patterns
            alias_patterns = []
            for alias in aliases:
                p_al = re.compile(
                    boundary_before + re.escape(alias) + boundary_after,
                    flags=flags,
                )
                alias_patterns.append((alias, p_al))

            self._compiled_patterns[canon] = {
                "category": category,
                "canonical_pattern": p_canon,
                "alias_patterns": alias_patterns,
            }

    def extract_skills(self, text: str) -> List[SkillMention]:
        """
        Extract verified technical skills present in the text.

        Distinguishes exact canonical matches from alias matches.
        Does not perform statistical guesswork or hallucinate inferred skills.

        Args:
            text: Input document text.

        Returns:
            List of detected SkillMention objects.
        """
        if not text or text.isspace():
            return []

        mentions: List[SkillMention] = []

        for canon, compiled in self._compiled_patterns.items():
            category = compiled["category"]
            p_canon = compiled["canonical_pattern"]
            canon_matches = p_canon.findall(text)
            canon_count = len(canon_matches)

            alias_count = 0
            first_matched_alias = None
            for alias_name, p_alias in compiled["alias_patterns"]:
                al_matches = p_alias.findall(text)
                if al_matches:
                    alias_count += len(al_matches)
                    if first_matched_alias is None:
                        first_matched_alias = alias_name

            total_count = canon_count + alias_count

            if total_count > 0:
                # If canonical name was matched at least once, match_type is exact
                if canon_count > 0:
                    match_type = "exact"
                    surface_name = canon_matches[0]
                else:
                    match_type = "alias"
                    surface_name = first_matched_alias or canon

                mentions.append(
                    SkillMention(
                        name=surface_name,
                        canonical_name=canon,
                        category=category,
                        match_type=match_type,
                        count=total_count,
                    )
                )

        # Sort alphabetically by canonical name for deterministic output
        mentions.sort(key=lambda s: s.canonical_name.lower())
        return mentions
