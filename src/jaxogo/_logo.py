"""Lay out voxel glyphs and render them with matplotlib's mplot3d."""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ._font import GLYPHS, JOINING_GLYPHS, SPACE_WIDTH, Voxel

# (top, front, side) face colours of each letter, cycled along the string.
# These are the colours of the J, A and X of the JAX logo.
DEFAULT_PALETTES: tuple[tuple[str, str, str], ...] = (
    ("#5e97f6", "#2a56c6", "#3367d6"),  # blue
    ("#26a69a", "#00695c", "#00796b"),  # teal
    ("#ea80fc", "#6a1b9a", "#9c27b0"),  # purple
)
EDGE_COLOR = "#dce0df"

# Size of a cube edge in the output SVG, in points; the JAX logo uses 50.
CUBE_SIZE = 50.0

# The camera looks along (-1, 1, -1), so each cube shows its top (+z),
# front (-y) and right (+x) faces. Rolling the isometric view by 30 degrees
# makes the x axis horizontal, the y axis rise at 60 degrees and the z axis
# at 120 degrees, which is the projection used by the JAX logo.
ELEVATION = float(np.degrees(np.arctan(1 / np.sqrt(2))))
AZIMUTH = -45.0
ROLL = 30.0

# Gap between letters on screen, in half-cubes: letters that cannot share a
# stroke touch, and letters that are kept apart leave a gap.
TOUCH_GAP = -2
LETTER_GAP = 2

# Visible faces: name -> (outward normal, corners relative to the voxel).
_FACES: dict[str, tuple[Voxel, tuple[Voxel, ...]]] = {
    "top": ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    "front": ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    "side": ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
}


def _screen_cell(v: Voxel) -> tuple[int, int]:
    """Where a voxel lands on screen, in half-cube columns and rows.

    Voxels that differ by a multiple of (1, -1, 1) share a screen cell; the one
    with the larger ``x - y + z`` is in front.
    """
    x, y, z = v
    return 2 * x + y - z, y + z


@dataclass
class _Placed:
    char: str
    voxels: frozenset[Voxel]


def _shift(voxels: frozenset[Voxel], dx: int) -> frozenset[Voxel]:
    return frozenset((x + dx, y, z) for x, y, z in voxels)


def _column(voxels: frozenset[Voxel], x: int) -> set[tuple[int, int]]:
    return {(vy, vz) for vx, vy, vz in voxels if vx == x}


def _try_join(
    placed: Sequence[_Placed], glyph: frozenset[Voxel]
) -> frozenset[Voxel] | None:
    """Overlap ``glyph`` with the previous letter so that they share a stroke.

    The JAX logo shares strokes between its letters: the stem of the J is the
    left leg of the A, and the right leg of the A is the tower of the X. The
    letters are overlapped by one voxel column, which works when the last
    column of the first is contained in the first column of the second (or
    vice versa) and nothing else in the two letters collides on screen.
    """
    prev = placed[-1].voxels
    last = max(x for x, _, _ in prev)
    dx = last - min(x for x, _, _ in glyph)
    moved = _shift(glyph, dx)
    a, b = _column(prev, last), _column(moved, last)
    if len(a & b) < 3 or not (a <= b or b <= a):
        return None
    occupied: dict[tuple[int, int], Voxel] = {}
    for p in placed:
        for v in p.voxels:
            occupied[_screen_cell(v)] = v
    for v in moved - prev:
        hit = occupied.get(_screen_cell(v))
        if hit is not None and hit != v:
            return None
    return moved


def _min_gap_shift(placed: Sequence[_Placed], glyph: frozenset[Voxel], gap: int) -> int:
    """Smallest x shift that keeps ``glyph`` ``gap`` half-cubes clear of every placed letter."""
    # Two cube hexagons whose rows differ by dr overlap when their columns
    # differ by less than 4 - dr, so ask for gap more than that.
    left: dict[int, int] = {}
    for p in placed:
        for v in p.voxels:
            h, r = _screen_cell(v)
            left[r] = max(left.get(r, h), h)
    need = -(10**9)
    for v in glyph:
        h, r = _screen_cell(v)
        for dr in (-2, -1, 0, 1, 2):
            if r + dr in left:
                need = max(need, left[r + dr] + 4 - abs(dr) + gap - h)
    return -(-need // 2)  # moving one voxel in x moves two columns on screen


def _layout(text: str, ligatures: bool) -> list[_Placed]:
    """Place each letter of ``text``.

    With ``ligatures``, neighbouring letters share a stroke where they can and
    otherwise touch; only spaces separate them. Without, letters are spaced.
    """
    # (glyph key, number of spaces before it)
    tokens: list[tuple[str, int]] = []
    spaces = 0
    for char in text:
        if char.isspace():
            spaces += 1
            continue
        key = char.upper()
        if key not in GLYPHS:
            raise ValueError(f"jaxogo has no glyph for {char!r}")
        tokens.append((key, spaces))
        spaces = 0
    if not tokens:
        raise ValueError("text must contain at least one visible character")

    def joins(i: int) -> bool:
        """Whether token i is joined to token i - 1."""
        return ligatures and 0 < i < len(tokens) and not tokens[i][1]

    def forms(key: str, side: str) -> tuple[frozenset[Voxel], ...]:
        """The glyph and its forms for joining a neighbour on ``side``."""
        return (GLYPHS[key], *(f for s, f in JOINING_GLYPHS.get(key, ()) if s == side))

    placed: list[_Placed] = []
    for i, (key, spaces) in enumerate(tokens):
        candidates = {GLYPHS[key]}
        if joins(i):
            candidates.update(forms(key, "left"))
        if joins(i + 1):
            candidates.update(forms(key, "right"))
        # Score each form by the voxels it shares with its neighbours; the
        # normal glyph wins ties.
        best, best_score = None, -1
        for form in sorted(candidates, key=lambda f: f != GLYPHS[key]):
            score = 0
            if not placed:
                moved = _shift(form, -min(x for x, _, _ in form))
            elif joins(i):
                moved = _try_join(placed, form)
                if moved is None:
                    moved = _shift(form, _min_gap_shift(placed, form, TOUCH_GAP))
                else:
                    score = len(moved & placed[-1].voxels)
            else:
                gap = LETTER_GAP + 4 * SPACE_WIDTH * spaces
                moved = _shift(form, _min_gap_shift(placed, form, gap))
            if joins(i + 1):
                ahead = [*placed, _Placed(key, moved)]
                nxt = [_try_join(ahead, f) for f in forms(tokens[i + 1][0], "left")]
                score += max((len(n & moved) for n in nxt if n is not None), default=0)
            if score > best_score:
                best, best_score = moved, score
        placed.append(_Placed(key, best))
    return placed


def _face_colors(
    placed: Sequence[_Placed], palettes: Sequence[Sequence[str]]
) -> dict[tuple[Voxel, str], str]:
    """Colour every exposed top/front/side face of every voxel.

    A stroke shared by two joined letters is split between them as in the JAX
    logo: the left letter colours the faces on the left of the stroke on
    screen and the right letter colours the rest. A ``/`` stroke in the ground
    plane mostly shows its tops (left) and sides (right); a ``\\`` tower mostly
    shows its fronts (left) and sides (right). A tower that rises above the
    ground plane also gives its tops to the right letter, as the X does in the
    JAX logo; one that hangs below it leaves them in the left letter's outline.
    """
    owners: dict[Voxel, list[int]] = {}
    for i, p in enumerate(placed):
        for v in p.voxels:
            owners.setdefault(v, []).append(i)
    tops_right = {
        (min(idx), max(idx))
        for (_, _, z), idx in owners.items()
        if z > 0 and len(idx) > 1
    }
    filled = set(owners)
    colors = {}
    for v, idx in owners.items():
        left, right = min(idx), max(idx)
        for face, ((nx, ny, nz), _) in _FACES.items():
            if (v[0] + nx, v[1] + ny, v[2] + nz) in filled:
                continue
            if face == "side":
                i = right
            elif face == "front":
                i = left
            else:
                i = right if (left, right) in tops_right else left
            top, front, side = palettes[i % len(palettes)]
            colors[(v, face)] = {"top": top, "front": front, "side": side}[face]
    return colors


def _render_svg(colors: dict[tuple[Voxel, str], str], edge_color: str) -> str:
    polys, facecolors = [], []
    for ((x, y, z), face), color in colors.items():
        corners = _FACES[face][1]
        polys.append([(x + cx, y + cy, z + cz) for cx, cy, cz in corners])
        facecolors.append(color)
    pts = np.array(polys, dtype=float).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    fig = Figure(dpi=72)  # SVG output is in points, so make pixels points
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1), projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEVATION, azim=AZIMUTH, roll=ROLL)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    ax.set_axis_off()
    fig.patch.set_alpha(0)

    # Every face of a voxel is the same distance from the camera, so sorting
    # faces by their mean depth draws farther cubes first.
    coll = Poly3DCollection(
        polys,
        facecolors=facecolors,
        edgecolors=edge_color,
        linewidths=1.0,
        zsort="average",
    )
    coll.set_joinstyle("round")
    ax.add_collection3d(coll)

    # mplot3d fits the scene to the axes, so measure a cube edge on screen and
    # resize the figure until it is CUBE_SIZE points long.
    for _ in range(2):
        fig.canvas.draw()
        to_px = ax.transData.transform
        proj = ax.get_proj()
        ex, ey, _ = proj3d.proj_transform(
            np.array([0.0, 1.0]), np.zeros(2), np.zeros(2), proj
        )
        edge = np.ptp(to_px(np.column_stack([ex, ey])), axis=0)
        scale = CUBE_SIZE / float(np.hypot(*edge))
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * scale, h * scale)
    fig.canvas.draw()

    verts = np.concatenate([path.vertices for path in coll.get_paths()])
    px = ax.transData.transform(verts)
    pad = 0.5  # half the stroke width
    bbox = Bbox.from_extents(*(px.min(axis=0) - pad), *(px.max(axis=0) + pad))

    buf = io.StringIO()
    with matplotlib.rc_context({"svg.hashsalt": "jaxogo", "svg.fonttype": "none"}):
        fig.savefig(
            buf,
            format="svg",
            bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()),
            pad_inches=0,
            transparent=True,
            metadata={"Date": None},
        )
    return buf.getvalue()


def jaxogo(
    text: str,
    path: str | os.PathLike[str] | None = None,
    *,
    palettes: Sequence[Sequence[str]] = DEFAULT_PALETTES,
    edge_color: str = EDGE_COLOR,
    ligatures: bool = True,
) -> str:
    """Render ``text`` as voxel letters in the style of the JAX logo.

    Each letter is built from cubes, drawn in isometric projection by
    matplotlib's mplot3d and coloured with the next palette in ``palettes``.
    ``jaxogo("JAX")`` reproduces the JAX logo.

    Args:
        text: Letters, digits, spaces and ``-_.!?`` to draw (case-insensitive).
        path: If given, the SVG is also written to this file.
        palettes: ``(top, front, side)`` face colours, one per letter, cycled.
        edge_color: Colour of the cube outlines.
        ligatures: Join neighbouring letters, like the J, A and X of the JAX
            logo: they share a stroke where they fit together and otherwise
            touch. Only spaces separate letters. If False, letters are spaced.

    Returns:
        The SVG document as a string.
    """
    if not palettes or any(len(p) != 3 for p in palettes):
        raise ValueError(
            "palettes must be a non-empty sequence of (top, front, side) colours"
        )
    placed = _layout(text, ligatures)
    svg = _render_svg(_face_colors(placed, palettes), edge_color)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
    return svg
