"""Controller SUS."""
import pandas as pd

from models import db
from models import sus as S


def submit(respondent, answers):
    s = S.score(answers)
    db.add_sus(respondent.strip() or "-", answers, s)
    return s


def summary():
    rows = db.list_sus()
    scores = [r["score"] for r in rows]
    mean = sum(scores) / len(scores) if scores else None
    return rows, mean


def table(rows):
    return pd.DataFrame([{"id": r["id"], "respondent": r["respondent"], "time": r["created_at"],
                          **{f"Q{i + 1}": a for i, a in enumerate(r["answers"])}, "SUS": r["score"]}
                         for r in rows])


def delete(sus_id):
    db.delete_sus(sus_id)