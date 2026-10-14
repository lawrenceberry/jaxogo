"""A five-row voxel font in the style of the JAX logo.

Glyphs are drawn as ASCII bitmaps, top row first. Column ``x`` and row ``y``
of a bitmap become the voxel ``(x, y, 0)``, so the glyph lies flat in the
``z = 0`` plane. The isometric camera used by the renderer shears that plane:
bitmap columns appear as ``/`` strokes and bitmap anti-diagonals appear as
``\\`` strokes, just like the stem of the J in the JAX logo.

A ``\\`` stroke can also be drawn as a vertical tower of voxels instead of a
staircase in the plane, which is how the JAX logo draws the ``\\`` of its X.
Mark the tower's cells with ``\\`` and the one cell where the tower passes
through the ``z = 0`` plane with ``*``. A voxel ``(x, y, z)`` lands on the same
spot on screen as ``(x - z, y + z, 0)``, so each ``\\`` cell is lifted onto the
tower standing on the ``*`` that shares its anti-diagonal.

Bitmap diagonals that rise to the right are drawn almost flat and their cubes
only touch along an edge, so glyphs mostly avoid them. The round letters are
the exception: B and D leave out their right corners, C, G and Q their left
corners, U its bottom corners and O all four, even though those corners then
only touch along an edge.

Characters:
    ``#``  voxel in the z = 0 plane
    ``*``  voxel in the z = 0 plane that anchors a tower
    ``+``  empty cell that anchors a tower
    ``\\``  voxel lifted onto the tower anchored on its anti-diagonal
    ``.``  empty

Trailing whitespace is ignored; it lets a raw-string row end in ``\\``.
"""

from __future__ import annotations

Voxel = tuple[int, int, int]

GLYPH_ROWS = 5

_BITMAPS: dict[str, tuple[str, ...]] = {
    "A": (
        r"#\....",
        r"#.\...",
        r"#..*..",
        r"####\.",
        r"#....\ ",
    ),
    "B": ("###.", "#..#", "####", "#..#", "###."),
    "C": (".###", "#...", "#...", "#...", ".###"),
    "D": ("###.", "#..#", "#..#", "#..#", "###."),
    "E": ("####", "#...", "###.", "#...", "####"),
    "F": ("####", "#...", "###.", "#...", "#..."),
    "G": (".###", "#...", "#.##", "#..#", ".###"),
    "H": ("#..#", "#..#", "####", "#..#", "#..#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", "###"),
    "K": ("#..#", "#..#", "###.", "#.#.", "#..#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": (
        r"#\.#*....",
        r"#.\#.\...",
        r"#..*..\..",
        r"#......\.",
        r"#.......\ ",
    ),
    "N": (
        r"#\...#",
        r"#.\..#",
        r"#..\.#",
        r"#...\#",
        r"#....*",
    ),
    "O": (".##.", "#..#", "#..#", "#..#", ".##."),
    "P": ("###.", "#..#", "####", "#...", "#..."),
    "Q": (".##..", "#..#.", "#..#.", "#.##.", ".####"),
    "R": ("###.", "#..#", "####", "#.#.", "#..#"),
    "S": ("####", "#...", ".##.", "...#", "####"),
    "T": ("#####", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#..#", "#..#", "#..#", "#..#", ".##."),
    "V": (
        r"\...#",
        r".\..#",
        r"..\.#",
        r"...\#",
        r"....*",
    ),
    "W": (
        r"\.....#",
        r".\....#",
        r"..\.#.#",
        r"...\#\#",
        r"....*.*",
    ),
    "X": (
        r"\.#..",
        r".\#..",
        r"..*..",
        r"..#\.",
        r"..#.\ ",
    ),
    "Y": (
        r"\.#",
        r".\#",
        r"..*",
        r"..#",
        r"..#",
    ),
    "Z": ("###..", "..#..", "..#..", "..#..", "..###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": ("##.", ".#.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", ".##", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", "..#", "..#", "..#"),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
    "-": ("...", "...", "###", "...", "..."),
    "_": ("...", "...", "...", "...", "###"),
    ".": (".", ".", ".", ".", "#"),
    "!": ("#", "#", "#", ".", "#"),
    "?": ("###", "..#", ".##", "...", ".#."),
}

# Extra forms of a glyph for joining a neighbour, keyed by the side ("left" or
# "right") that they join on. A form is used when it shares more voxels with
# that neighbour than the normal glyph does.
_JOINING_BITMAPS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    # An O whose left side is a hanging tower, so it can share the last
    # stroke of an M. Like the normal O it leaves out its corners, so the top
    # and bottom of the tower belong to the M alone.
    "O": (
        (
            "left",
            (
                r"+####.",
                r".\...#",
                r"..\..#",
                r"...\.#",
                r"......",
            ),
        ),
    ),
}

# Horizontal space (in voxels) taken by a blank character.
SPACE_WIDTH = 2


def _parse(char: str, rows: tuple[str, ...]) -> frozenset[Voxel]:
    if len(rows) != GLYPH_ROWS:
        raise ValueError(f"glyph {char!r} must have {GLYPH_ROWS} rows")
    cells: dict[tuple[int, int], str] = {}
    for i, row in enumerate(rows):
        y = GLYPH_ROWS - 1 - i
        for x, c in enumerate(row.rstrip()):
            if c != ".":
                cells[(x, y)] = c

    anchors = {x + y: (x, y) for (x, y), c in cells.items() if c in "*+"}
    voxels = set()
    for (x, y), c in cells.items():
        if c in "#*":
            voxels.add((x, y, 0))
        elif c == "+":
            pass
        elif c == "\\":
            if x + y not in anchors:
                raise ValueError(f"glyph {char!r}: tower cell {(x, y)} has no anchor")
            ax, ay = anchors[x + y]
            voxels.add((ax, ay, y - ay))
        else:
            raise ValueError(f"glyph {char!r}: unknown cell {c!r}")
    return frozenset(voxels)


GLYPHS: dict[str, frozenset[Voxel]] = {
    c: _parse(c, rows) for c, rows in _BITMAPS.items()
}
JOINING_GLYPHS: dict[str, tuple[tuple[str, frozenset[Voxel]], ...]] = {
    c: tuple((side, _parse(c, rows)) for side, rows in forms)
    for c, forms in _JOINING_BITMAPS.items()
}
