"""Relation model for knowledge graph edges."""
from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Types of relations between entities."""
    EVALUATED_ON = "evaluated_on"
    IMPROVES_OVER = "improves_over"
    USES = "uses"
    ASSUMES = "assumes"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    BASED_ON = "based_on"
    APPLIED_TO = "applied_to"
    PART_OF = "part_of"
    COMPARES_WITH = "compares_with"


class Relation(BaseModel):
    """
    Relation representing an edge in the knowledge graph.
    
    Relations connect entities and capture structured knowledge
    about how methods, datasets, metrics, etc. relate to each other.
    
    Attributes:
        relation_id: Unique identifier
        subject_entity_id: Source entity ID
        predicate: Type of relationship
        object_entity_id: Target entity ID
        evidence_chunks: Chunk IDs supporting this relation
        confidence: Confidence score (0-1)
        attributes: Additional structured attributes (e.g., improvement delta)
        paper_id: Paper where relation was found
        created_at: Timestamp of creation
    """
    relation_id: str
    subject_entity_id: str
    predicate: RelationType
    object_entity_id: str
    evidence_chunks: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    attributes: dict = Field(default_factory=dict)
    paper_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_triple(self) -> tuple:
        """Return as (subject, predicate, object) triple."""
        return (self.subject_entity_id, self.predicate.value, self.object_entity_id)
    
    def to_natural_language(self, subject_name: str, object_name: str) -> str:
        """Convert relation to natural language description."""
        predicate_templates = {
            RelationType.EVALUATED_ON: "{subject} was evaluated on {object}",
            RelationType.IMPROVES_OVER: "{subject} improves over {object}",
            RelationType.USES: "{subject} uses {object}",
            RelationType.ASSUMES: "{subject} assumes {object}",
            RelationType.SIMILAR_TO: "{subject} is similar to {object}",
            RelationType.CONTRADICTS: "{subject} contradicts {object}",
            RelationType.EXTENDS: "{subject} extends {object}",
            RelationType.BASED_ON: "{subject} is based on {object}",
            RelationType.APPLIED_TO: "{subject} is applied to {object}",
            RelationType.PART_OF: "{subject} is part of {object}",
            RelationType.COMPARES_WITH: "{subject} compares with {object}",
        }
        template = predicate_templates.get(self.predicate, "{subject} relates to {object}")
        return template.format(subject=subject_name, object=object_name)
