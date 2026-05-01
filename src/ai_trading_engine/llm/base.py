from __future__ import annotations

from abc import ABC, abstractmethod


class LlmClient(ABC):
    @abstractmethod
    def decide(self, system_prompt: str, context: str) -> str:
        """Return model output as text, expected to contain JSON decision."""
        raise NotImplementedError
