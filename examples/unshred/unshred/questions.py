"""Question builders shared by the Drex matcher and the baseline.

Requests are built from piece text only: the answer key never gets here.
Every question quotes its own candidate text, because Drex judges each
question on its own and never sees the question ids.

Two strategies ask the same thing in two shapes:
  noul    one yes/no question per candidate (the project default)
  choice  one pick-one question per piece, one option per candidate
"""

from .shredder import Piece

NONE = "none"
CONTEXT = (
    "A printed page was cut by a shredder into a grid of rectangular "
    "pieces. Each piece holds a few short lines of text. Cuts fall in the "
    "middle of words, so a line that is cut off at the right side of one "
    "piece continues on the same line of the piece that sat on its right."
)
STRIPS = (
    " Here the pieces are already joined side by side into full-width "
    "strips. Text runs to the end of a line and continues at the start of "
    "the next line."
)
SEAM = {
    "true": "Across the join the letters form real words or numbers and "
            "the phrases read naturally on most lines.",
    "false": "Across the join most lines show broken words, nonsense "
             "letter runs, or unrelated phrases pushed together.",
}
FLOW = {
    "true": "The sentence or list runs on naturally from the upper strip "
            "into the lower strip.",
    "false": "The text jumps: the lower strip does not continue what the "
             "upper strip was saying.",
}


def join_right(a: Piece, b: Piece) -> str:
    """A's lines followed directly by B's lines, line by line."""
    n = max(len(a.lines), len(b.lines))
    return "\n".join(
        (a.lines[i] if i < len(a.lines) else "")
        + (b.lines[i] if i < len(b.lines) else "")
        for i in range(n))


def stack(top, bottom) -> str:
    """The lines of `top` directly above the lines of `bottom`."""
    return "\n".join(list(top) + list(bottom))


def show(lines) -> str:
    """One bracketed line per row, so spaces at the ends stay visible."""
    return "\n".join(f"[{line}]" for line in lines)


def right_request(a: Piece, others: list[Piece],
                  strategy: str = "noul") -> tuple[dict, dict]:
    """One request about piece A: its right neighbor and the right edge."""
    state = {"context": CONTEXT, "left_piece": show(a.lines)}
    questions = {"right_edge": {
        "type": "noul",
        "instructions": (
            "left_piece is shown one line per row inside square brackets, "
            "so spaces at the ends are visible. Is left_piece on the right "
            "edge of the page, so that nothing follows it on the right?"),
        "criteria": {
            "true": "Most lines end in blank padding or at the end of a "
                    "word: nothing continues to the right.",
            "false": "Most lines are cut off mid-word or mid-phrase at the "
                     "right bracket: the text continues on another piece.",
        },
    }}
    if strategy == "choice":
        criteria = {o.piece_id: join_right(a, o) for o in others}
        criteria[NONE] = "No option fits: left_piece is on the right edge."
        questions["right_pick"] = {
            "type": "choice",
            "instructions": (
                "Each option shows left_piece glued, line by line, to one "
                "candidate piece on its right. Which option is one "
                "continuous natural passage, where the letters on both "
                "sides of the join form correct words on most lines?"),
            "criteria": criteria,
        }
        return state, questions
    for o in others:
        questions[f"right_{o.piece_id}"] = {
            "type": "noul",
            "instructions": (
                "A candidate piece was glued to the right side of "
                "left_piece. Joined line by line, the result is:\n"
                f"{join_right(a, o)}\n"
                "Read across the middle of each line, where the two pieces "
                "meet. Is this joined text one continuous natural passage "
                "there?"),
            "criteria": SEAM,
        }
    return state, questions


def below_request(strips: dict[str, list[str]], strategy: str = "noul",
                  k: int = 2) -> tuple[dict, dict]:
    """One request about full-width strips: which strip follows which.

    strips maps a key to the lines of one rebuilt row. A strip whose last
    line is blank is the end of the page and gets no questions.
    """
    state = {"context": CONTEXT + STRIPS}
    questions: dict[str, dict] = {}
    for x, top in strips.items():
        if not top or not top[-1].strip():
            continue
        tail = stack(top[-k:], [])
        questions[f"bottom_{x}"] = {
            "type": "noul",
            "instructions": (
                "A strip of the page ends with these lines:\n"
                f"{tail}\n"
                "Is this the last strip of the page, so that the document "
                "ends here and no text follows below it?"),
        }
        heads = {y: stack(b[:k], []) for y, b in strips.items() if y != x}
        if strategy == "choice" and heads:
            criteria = dict(heads)
            criteria[NONE] = "No option fits: the page ends with strip X."
            questions[f"below_pick_{x}"] = {
                "type": "choice",
                "instructions": (
                    "Strip X of the page ends with these lines:\n"
                    f"{tail}\n"
                    "Each option shows the first lines of another strip. "
                    "Which option continues the text of strip X directly, "
                    "as the next lines of the same passage?"),
                "criteria": criteria,
            }
            continue
        for y, head in heads.items():
            questions[f"below_{x}__{y}"] = {
                "type": "noul",
                "instructions": (
                    "Strip X was stacked directly on top of strip Y. The "
                    "last lines of X followed by the first lines of Y read:\n"
                    f"{tail}\n{head}\n"
                    "Is this stacked text one continuous natural passage, "
                    "where the end of strip X runs straight on into the "
                    "start of strip Y?"),
                "criteria": FLOW,
            }
    return state, questions
