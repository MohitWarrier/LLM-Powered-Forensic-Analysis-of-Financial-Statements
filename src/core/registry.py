from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class Feature:
    """A registered feature in the application."""
    id: str
    name: str
    icon: str
    service: Any
    page_renderer: Callable
    enabled: bool = True


class FeatureRegistry:
    """Central registry of all application features."""

    def __init__(self):
        self._features: dict[str, Feature] = {}

    def register(self, feature: Feature):
        self._features[feature.id] = feature

    def get(self, feature_id: str) -> Feature:
        return self._features[feature_id]

    def get_enabled(self) -> list[Feature]:
        return [f for f in self._features.values() if f.enabled]

    def list_ids(self) -> list[str]:
        return list(self._features.keys())
