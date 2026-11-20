"""Finite, acknowledged dialogue control; no generation or transport calls."""

import copy


class DialogueController:
    def __init__(self, memory, reserve=3.):
        if reserve < 3:
            raise ValueError("Delivery reserve cannot be below three seconds")
        self.memory, self.reserve = memory, float(reserve)

    def should_generate(self, now):
        if not self.memory.can_chat(now, self.reserve):
            return False, "quota_time_pending_or_phase"
        if self.memory.current.accepted == 0:
            return True, "opening"
        if not self.memory.relevant_update():
            return False, "no_new_relevant_evidence"
        return True, "new_evidence"

    def admissible(self, candidate, now):
        if candidate.get("delivery") == "terminate":
            return True, None
        if not self.memory.can_chat(now, self.reserve):
            return False, "quota_time_pending_or_phase"
        if candidate["recipients"] and self.memory.current.private_accepted:
            return False, "private_coordination_cap"
        accepted = [record for record in self.memory.deliveries.values()
                    if record["owner"] == self.memory.current.owner and record["status"] == "accepted"]
        if any(record["canonical_text"] == candidate["text"] and record["recipients"] == candidate["recipients"]
               for record in accepted):
            return False, "duplicate_canonical_claim"
        if self.memory.current.accepted >= 2:
            commitments = [record["target"] for record in accepted if record["intent"] == "coordinate_vote"]
            if candidate["intent"] != "coordinate_vote" or commitments and commitments[-1] == candidate["target"]:
                return False, "third_chat_requires_changed_commitment"
        return True, None

    def candidates(self, candidates, now):
        result = copy.deepcopy(candidates)
        for candidate in result:
            candidate["available"], candidate["unavailable_reason"] = self.admissible(candidate, now)
        return result
