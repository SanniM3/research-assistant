"""Entity model for knowledge graph nodes."""
from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    TASK = "task"
    DOMAIN = "domain"
    BENCHMARK = "benchmark"
    FRAMEWORK = "framework"
    MODEL = "model"
    TECHNIQUE = "technique"


class Entity(BaseModel):
    """
    Entity representing a concept in the knowledge graph.
    
    Entities are linked to claims and relations to form a structured
    knowledge representation of the research domain.
    
    Attributes:
        entity_id: Unique identifier
        entity_type: Category of entity
        name: Canonical name
        aliases: Alternative names/spellings
        description: Grounded description from sources
        evidence_chunks: Chunk IDs supporting this entity
        paper_ids: Papers mentioning this entity
        attributes: Additional structured attributes
        created_at: Timestamp of creation
        merged_from: IDs of entities merged into this one
    """
    entity_id: str
    entity_type: EntityType
    name: str
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    evidence_chunks: List[str] = Field(default_factory=list)
    paper_ids: List[str] = Field(default_factory=list)
    attributes: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    merged_from: List[str] = Field(default_factory=list)
    
    def add_alias(self, alias: str) -> None:
        """Add an alternative name if not already present."""
        normalized = alias.lower().strip()
        if normalized not in [a.lower() for a in self.aliases]:
            self.aliases.append(alias)
    
    def matches_name(self, query: str) -> bool:
        """Check if query matches name or any alias."""
        query_lower = query.lower().strip()
        if self.name.lower() == query_lower:
            return True
        return any(alias.lower() == query_lower for alias in self.aliases)
    
    def merge_with(self, other: "Entity") -> None:
        """Merge another entity into this one."""
        # Add other's name as alias if different
        if other.name.lower() != self.name.lower():
            self.add_alias(other.name)
        
        # Merge aliases
        for alias in other.aliases:
            self.add_alias(alias)
        
        # Merge evidence and paper references
        self.evidence_chunks = list(set(self.evidence_chunks + other.evidence_chunks))
        self.paper_ids = list(set(self.paper_ids + other.paper_ids))
        
        # Track merge history
        self.merged_from.append(other.entity_id)
        self.merged_from.extend(other.merged_from)
        
        # Merge description if current is empty
        if not self.description and other.description:
            self.description = other.description
