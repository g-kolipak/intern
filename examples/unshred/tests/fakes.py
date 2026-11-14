"""A fake ask() for tests: it answers from the answer key, no network."""
from unshred.drex import CallResult


class FakeAsk:
    """Stands in for drex.ask / baseline.ask and records every request."""

    def __init__(self, shr, hit=0.95, miss=0.05):
        self.key = shr.answer_key
        self.id_at = {pos: pid for pid, pos in self.key.items()}
        by_id = {p.piece_id: p for p in shr.pieces}
        height = len(shr.pieces[0].lines)
        self.page = [
            "".join(by_id[self.id_at[(r, c)]].lines[j]
                    for c in range(shr.cols)).rstrip()
            for r in range(shr.rows) for j in range(height)]
        self.last = max(i for i, line in enumerate(self.page) if line)
        self.blank = {p.piece_id for p in shr.pieces
                      if not "".join(p.lines).strip()}
        self.hit, self.miss = hit, miss
        self.requests = []

    def _line(self, text):
        return self.page.index(text) if text in self.page else -99

    def _noul(self, yes):
        return {"type": "noul", "noul": self.hit if yes else self.miss}

    def _choice(self, spec, pick):
        labels = list(spec["criteria"])
        rest = (1 - self.hit) / max(1, len(labels) - 1)
        return {"type": "choice", "choice": pick, "probabilities": {
            lab: (self.hit if lab == pick else rest) for lab in labels}}

    def __call__(self, state, questions, **kwargs):
        self.requests.append(
            {"state": state, "questions": questions, "kwargs": kwargs})
        answers = {}
        if "left_piece" in state:
            pick = questions.get("right_pick")
            cands = (set(pick["criteria"]) - {"none"} if pick else
                     {q[6:] for q in questions
                      if q.startswith("right_") and q != "right_edge"})
            rest = set(self.key) - cands  # A, plus blanks left out
            (a,) = (rest - self.blank) or rest
            r, c = self.key[a]
            true = self.id_at.get((r, c + 1))
            for q, spec in questions.items():
                if q == "right_edge":  # blank to the right looks like an edge
                    answers[q] = self._noul(true is None or true in self.blank)
                elif q == "right_pick":
                    answers[q] = self._choice(
                        spec, true if true in spec["criteria"] else "none")
                else:
                    answers[q] = self._noul(q[6:] == true)
        else:
            for q, spec in questions.items():
                quoted = spec["instructions"].split("\n")[1:-1]
                idx = [self._line(t) for t in quoted]
                if q.startswith("bottom_"):
                    answers[q] = self._noul(idx[-1] == self.last)
                elif q.startswith("below_pick_"):
                    pick = next(
                        (lab for lab, text in spec["criteria"].items()
                         if lab != "none"
                         and self._line(text.split("\n")[0]) == idx[-1] + 1),
                        "none")
                    answers[q] = self._choice(spec, pick)
                else:
                    answers[q] = self._noul(all(
                        b == a + 1 for a, b in zip(idx, idx[1:])))
        body = {"answers": answers, "usage": {"input_tokens": 100}}
        return CallResult(body=body, latency_ms=12.5, input_tokens=100,
                          request_id=f"r{len(self.requests)}",
                          model=kwargs.get("model", ""))


def words(n):
    """Text whose wrapped lines are all different."""
    return " ".join(f"word{i:03d}" for i in range(n))
