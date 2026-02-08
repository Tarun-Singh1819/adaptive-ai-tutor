from core.state import LTMState
from typing import Dict,Any

def ImportanceScorerNode(state: LTMState) -> Dict[str, Any]:
    text = state.get("current_memory")

    if not text:
        return {"importance": None}

    text = text.lower()
    score = 0.5

    identity_keywords = ["name", "background", "student", "professional", "experience", "identity"]
    if any(k in text for k in identity_keywords):
        score = max(score, 0.95)

    skill_keywords = ["learning", "studying", "beginner", "intermediate", "advanced", "progressing", "understands", "struggles"]
    if any(k in text for k in skill_keywords):
        score = max(score, 0.9)

    goal_keywords = ["goal", "aim", "wants to", "planning to", "building", "working on", "project"]
    if any(k in text for k in goal_keywords):
        score = max(score, 0.85)

    preference_keywords = ["likes", "prefers", "enjoys", "comfortable with", "doesn't like"]
    if any(k in text for k in preference_keywords):
        score = max(score, 0.75)

    transient_keywords = ["today", "now", "currently", "right now", "feels", "tired", "hungry"]
    if any(k in text for k in transient_keywords):
        score = min(score, 0.4)

    score = max(0.0, min(score, 1.0))
    return {"importance": score}
