"""Base storage abstractions."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StorageBackend(ABC, Generic[T]):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save(self, item: T) -> str:
        """Save an item and return its ID."""
        pass
    
    @abstractmethod
    def get(self, item_id: str) -> Optional[T]:
        """Get an item by ID."""
        pass
    
    @abstractmethod
    def get_all(self) -> List[T]:
        """Get all items."""
        pass
    
    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Delete an item by ID."""
        pass
    
    @abstractmethod
    def query(self, filters: Dict[str, Any]) -> List[T]:
        """Query items by filters."""
        pass
    
    @abstractmethod
    def update(self, item_id: str, updates: Dict[str, Any]) -> Optional[T]:
        """Update an item."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Count total items."""
        pass


class InMemoryStorage(StorageBackend[T]):
    """In-memory storage implementation for development/testing."""
    
    def __init__(self):
        self._store: Dict[str, T] = {}
        self._id_field: str = "id"
    
    def set_id_field(self, field_name: str) -> None:
        """Set the field name used as ID."""
        self._id_field = field_name
    
    def _get_id(self, item: T) -> str:
        """Extract ID from item."""
        return getattr(item, self._id_field)
    
    def save(self, item: T) -> str:
        """Save an item and return its ID."""
        item_id = self._get_id(item)
        self._store[item_id] = item
        return item_id
    
    def get(self, item_id: str) -> Optional[T]:
        """Get an item by ID."""
        return self._store.get(item_id)
    
    def get_all(self) -> List[T]:
        """Get all items."""
        return list(self._store.values())
    
    def delete(self, item_id: str) -> bool:
        """Delete an item by ID."""
        if item_id in self._store:
            del self._store[item_id]
            return True
        return False
    
    def query(self, filters: Dict[str, Any]) -> List[T]:
        """Query items by filters."""
        results = []
        for item in self._store.values():
            match = True
            for key, value in filters.items():
                item_value = getattr(item, key, None)
                if item_value != value:
                    match = False
                    break
            if match:
                results.append(item)
        return results
    
    def update(self, item_id: str, updates: Dict[str, Any]) -> Optional[T]:
        """Update an item."""
        if item_id not in self._store:
            return None
        item = self._store[item_id]
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)
        return item
    
    def count(self) -> int:
        """Count total items."""
        return len(self._store)
    
    def clear(self) -> None:
        """Clear all items."""
        self._store.clear()
