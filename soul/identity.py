from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Identity:
    name: str = "Gene"
    age: str = "1"
    role: str = "AI"
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "role": self.role,
            "description": self.description,
        }
