"""Generate JAX-style logos for a few names, including JAX-based libraries.

Run with ``uv run examples/generate_examples.py``; the logos are written next
to this script as ``<name>.svg``.
"""

from pathlib import Path

from jaxogo import jaxogo

NAMES = [
    "jaxogo",
    "hello world",
    "modax",
    "diffrax",
    "flax",
    "optax",
    "equinox",
    "haiku",
    "numpyro",
    "chex",
    "orbax",
    "brax",
    "blackjax",
]

here = Path(__file__).parent
for name in NAMES:
    out = here / f"{name.replace(' ', '_')}.svg"
    jaxogo(name, out)
    print(f"Wrote {out}")
