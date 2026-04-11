"""Data models for the research assistant."""
from .paper import Paper, PaperMetadata
from .chunk import Chunk, ChunkMetadata
from .claim import Claim, ClaimType, Evidence
from .entity import Entity, EntityType
from .relation import Relation, RelationType
from .issue import Issue, IssueSeverity, IssueCategory, IssueStatus
from .state import (
    ResearchState, OutlineSection, AcceptanceCriteria, CoverageScores,
    ResearchQuestion, QuestionStatus, ReviewScores,
)

__all__ = [
    "Paper", "PaperMetadata",
    "Chunk", "ChunkMetadata", 
    "Claim", "ClaimType", "Evidence",
    "Entity", "EntityType",
    "Relation", "RelationType",
    "Issue", "IssueSeverity", "IssueCategory", "IssueStatus",
    "ResearchState", "OutlineSection", "AcceptanceCriteria", "CoverageScores",
    "ResearchQuestion", "QuestionStatus", "ReviewScores",
]
