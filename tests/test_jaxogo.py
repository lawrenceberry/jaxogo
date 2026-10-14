import itertools
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as MplPath

from jaxogo import jaxogo
from jaxogo._font import GLYPHS
from jaxogo._logo import DEFAULT_PALETTES, _face_colors, _layout

# The JAX logo, from
# https://github.com/jax-ml/jax/blob/ebf7c1b6127d6df3a64a4d969297ca70e126a341/images/jax_logo.svg
REFERENCE = Path(__file__).parent / "data" / "jax_logo.svg"
SVG = "{http://www.w3.org/2000/svg}"


def _reference_polygons():
    text = REFERENCE.read_text()
    fills = dict(re.findall(r"\.(cls-\d)\{fill:(#[0-9a-f]{6});\}", text))
    for el in ET.fromstring(text).iter(f"{SVG}polygon"):
        pts = np.array(el.get("points").split(), dtype=float).reshape(-1, 2)
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        yield fills[el.get("class")], pts


def _rendered_paths(svg):
    """(fill, path) for every filled path, in drawing order."""
    out = []
    for el in ET.fromstring(svg).iter(f"{SVG}path"):
        fill = re.search(r"fill: (#[0-9a-f]{6})", el.get("style", ""))
        if fill and el.get("d"):
            nums = np.array(re.findall(r"-?[\d.]+", el.get("d")), dtype=float)
            out.append((fill.group(1), MplPath(nums.reshape(-1, 2))))
    return out


def _fill_at(paths, point):
    for fill, path in reversed(paths):
        if path.contains_point(point):
            return fill
    return None


def test_jax_matches_reference_logo():
    svg = jaxogo("JAX")
    root = ET.fromstring(svg)
    assert [float(v) for v in root.get("viewBox").split()] == pytest.approx(
        [0, 0, 451, 260.81], abs=0.01
    )
    paths = _rendered_paths(svg)
    checked = 0
    for fill, pts in _reference_polygons():
        # Sample each triangle of the reference polygon near its centroid.
        for i in range(1, len(pts) - 1):
            centroid = pts[[0, i, i + 1]].mean(axis=0)
            assert _fill_at(paths, centroid) == fill, (fill, centroid)
            checked += 1
    assert checked > 50


def test_jax_has_no_extra_ink():
    paths = _rendered_paths(jaxogo("JAX"))
    reference = [MplPath(pts) for _, pts in _reference_polygons()]
    offsets = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
    for x in np.arange(5, 451, 10.0):
        for y in np.arange(5, 260, 10.0):
            hits = [
                any(p.contains_point((x + dx, y + dy)) for p in reference)
                for dx, dy in offsets
            ]
            inside, near = all(hits), any(hits)
            drawn = _fill_at(paths, (x, y)) is not None
            if inside:
                assert drawn, (x, y)
            elif not near:
                assert not drawn, (x, y)


def test_ligatures_share_strokes():
    joined = _layout("JAX", ligatures=True)
    apart = _layout("JAX", ligatures=False)
    width = lambda placed: max(x for p in placed for x, _, _ in p.voxels)
    assert width(joined) == 5
    assert width(apart) > width(joined)
    j, a, x = (p.voxels for p in joined)
    assert {v for v in j if v[0] == 2} <= a  # the J's stem is the A's left leg
    assert {v for v in a if v[2] != 0} <= x  # the A's right leg is the X's tower


def test_writes_file_and_is_deterministic(tmp_path):
    out = tmp_path / "logo.svg"
    svg = jaxogo("Hello, world".replace(",", ""), out)
    assert out.read_text() == svg
    assert jaxogo("Hello world") == svg


@pytest.mark.parametrize("char", sorted(GLYPHS))
def test_every_glyph_renders(char):
    assert "<svg" in jaxogo(char)


def test_glyphs_are_five_rows_tall_and_grounded():
    for char, voxels in GLYPHS.items():
        rows = {y + z for _, y, z in voxels}
        assert max(rows) <= 4 and min(rows) >= 0, char


def test_custom_palette():
    svg = jaxogo(
        "JAX", palettes=[("#111111", "#222222", "#333333")], edge_color="#000000"
    )
    fills = set(re.findall(r"fill: (#[0-9a-f]{6})", svg))
    assert fills == {"#111111", "#222222", "#333333"}


@pytest.mark.parametrize("text", ["", "   ", "J@X"])
def test_rejects_bad_text(text):
    with pytest.raises(ValueError):
        jaxogo(text)


@pytest.mark.parametrize("text", ["JAX", "MODAX"])
def test_neighbouring_letters_share_a_stroke(text):
    placed = _layout(text, ligatures=True)
    for a, b in itertools.pairwise(placed):
        assert len(a.voxels & b.voxels) >= 3, (a.char, b.char)


def test_round_letters_leave_out_their_corners():
    left, right = min, max
    corners = {
        "B": [right],
        "D": [right],
        "C": [left],
        "G": [left],
        "Q": [left],
        "O": [left, right],
    }
    for char, sides in corners.items():
        voxels = GLYPHS[char]
        for side in sides:
            x = side(vx for vx, _, _ in voxels)
            assert {(x, 0, 0), (x, 4, 0)}.isdisjoint(voxels), char
    xs = [x for x, _, _ in GLYPHS["U"]]
    assert {(min(xs), 0, 0), (max(xs), 0, 0)}.isdisjoint(GLYPHS["U"])


def test_corner_cubes_take_the_colour_of_the_letter_that_needs_them():
    placed = _layout("MODAX", ligatures=True)
    _, _, d, a, _ = placed
    leg = {v for v in a.voxels if v[0] == min(x for x, _, _ in a.voxels)}
    top, bottom = max(leg, key=lambda v: v[1]), min(leg, key=lambda v: v[1])
    assert leg - d.voxels == {top, bottom}
    colors = _face_colors(placed, DEFAULT_PALETTES)
    a_palette = DEFAULT_PALETTES[3 % len(DEFAULT_PALETTES)]
    assert colors[(top, "top")] == a_palette[0]
    assert colors[(bottom, "front")] == a_palette[1]
