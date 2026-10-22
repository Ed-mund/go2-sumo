"""Immutable checkpoint pool and bounded, prioritized opponent sampling."""

import hashlib
import json
from pathlib import Path

from sumo_campaign import write_json
from tactics_contract import CONTRACT


class League:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"contract": CONTRACT, "members": []}
        if self.data["contract"] != CONTRACT:
            raise ValueError("Incompatible opponent observation contract")

    def add(self, checkpoint, style, role, forced_skill=-1):
        checkpoint = Path(checkpoint).resolve()
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        key = f"{digest}:{role}:{forced_skill}"
        if any(member["id"] == key for member in self.data["members"]):
            return
        self.data["members"].append(
            {
                "id": key,
                "checkpoint": str(checkpoint),
                "sha256": digest,
                "style": style,
                "role": role,
                "forced_skill": forced_skill,
                "games": 0,
                "learner_score": 0.0,
            }
        )
        # Keep baseline/style coverage, with bounded metadata/cache. Files are never deleted here.
        members = self.data["members"]
        if len(members) > 24:
            first = {}
            for i, member in enumerate(members):
                first.setdefault(member["style"], i)
            removable = [i for i in range(len(members) - 8) if i not in first.values()]
            if removable:
                members.pop(removable[0])
        write_json(self.path, self.data)

    def sample(self, rng, count=3, exclude=(), require_style=None):
        members = [m for m in self.data["members"] if m["checkpoint"] not in exclude]
        chosen = []
        while members and len(chosen) < count:
            # Smoothed win rate: favor opponents we cannot beat, retain a uniform floor.
            weights = [0.2 + (1 - (m["learner_score"] + 1) / (m["games"] + 2)) ** 2 for m in members]
            matching = [i for i, m in enumerate(members) if m["style"] == require_style]
            index = (
                matching[0] if matching and not chosen else rng.choices(range(len(members)), weights=weights, k=1)[0]
            )
            member = members.pop(index)
            actual = hashlib.sha256(Path(member["checkpoint"]).read_bytes()).hexdigest()
            if actual != member["sha256"]:
                raise ValueError("Historical opponent checkpoint changed")
            chosen.append(member)
        return chosen

    def record(self, results):
        for member in self.data["members"]:
            row = results.get(member["id"], {})
            member["games"] += row.get("games", 0)
            member["learner_score"] += row.get("score", 0.0)
        write_json(self.path, self.data)
