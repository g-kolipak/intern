"""Tests for unshred.demo: the data file behind web/tear.html."""
import json

from unshred import demo
from unshred.assembler import assemble

META = {p: {"cx": 0.1 * i, "cy": 0.5, "skew": 10.0, "w": 100 + i, "h": 80}
        for i, p in enumerate(["p1", "p2", "p3", "p4"], 1)}
TURNS = {p: {"degrees": 90, "confidence": 0.9} for p in META}


def _assembly():
    right = {a: {b: 0.1 for b in META if b != a} for a in META}
    right["p1"]["p2"] = 0.9
    return assemble(right, {a: {} for a in META}, 4, 4, 0.6)


def test_groups_splits_linked_and_loose():
    linked, loose = demo.groups(_assembly())
    assert linked == [{"p1": (0, 0), "p2": (0, 1)}]
    assert sorted(loose) == ["p3", "p4"]


def test_build_has_what_the_page_reads():
    data = demo.build("shot", (4000, 3000), META, TURNS, _assembly(),
                      {"calls": 10}, 0.6)
    assert data["name"] == "shot" and data["photo"] == "outline.jpg"
    assert data["photo_size"] == [4000, 3000] and data["threshold"] == 0.6
    assert data["links"] == [{"a": "p1", "b": "p2", "direction": "right",
                              "score": 0.9}]
    assert sorted(data["loose"]) == ["p3", "p4"]
    by_id = {p["id"]: p for p in data["pieces"]}
    assert by_id["p1"]["placed"] and by_id["p1"]["score"] == 0.9
    assert not by_id["p3"]["placed"] and by_id["p3"]["score"] is None
    assert by_id["p1"]["start"] == -80.0          # skew 10 minus turn 90
    assert by_id["p1"]["src"] == "upright/p1.png"
    assert by_id["p2"]["x"] > by_id["p1"]["x"]     # p2 sits right of p1
    assert by_id["p3"]["y"] > by_id["p1"]["y"]     # loose row is below
    assert data["canvas"][0] > 0 and data["usage"] == {"calls": 10}


def test_write_makes_a_script_the_page_can_load(tmp_path):
    data = demo.build("shot", (4000, 3000), META, TURNS, _assembly(), None,
                      0.6)
    path = demo.write(tmp_path, data)
    text = path.read_text()
    assert path.name == "demo.js" and text.startswith("window.DEMO = ")
    assert json.loads(text[len("window.DEMO = "):].rstrip(";\n")) == data
