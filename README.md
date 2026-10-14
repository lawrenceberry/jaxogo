<p align="center">
  <img alt="jaxogo" src="https://raw.githubusercontent.com/lawrenceberry/jaxogo/main/examples/jaxogo.svg" width="600">
</p>

<p align="center">
  <a href="https://github.com/lawrenceberry/jaxogo/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="#install">Install</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#development">Development</a> •
  <a href="#license">License</a> •
  <a href="#acknowledgement">Acknowledgement</a>
</p>

`jaxogo` turns any string into an SVG logo in the style of the [JAX](https://github.com/jax-ml/jax)
logo: letters built from cubes, drawn in isometric projection with matplotlib's `mplot3d`.
`jaxogo("JAX")` reproduces the JAX logo exactly.

## Install

```bash
pip install git+https://github.com/lawrenceberry/jaxogo.git
```

## Quickstart

```python
from jaxogo import jaxogo

svg = jaxogo("jaxogo", "jaxogo.svg")  # returns the SVG text and writes it to jaxogo.svg
```

That draws the logo at the top of this page. Neighbouring letters are joined, sharing a stroke
where they fit together as the J, A and X of the JAX logo do; a space keeps words apart.

Options:

* `palettes`: one `(top, front, side)` colour triple per letter, cycled along the string.
  The default is the blue, teal and purple of the JAX logo.
* `edge_color`: the colour of the cube outlines.
* `ligatures=False`: space the letters apart instead of joining them.

```python
jaxogo("hello world", "hello_world.svg")
jaxogo("numpy", "numpy.svg", palettes=[("#f7c948", "#b8860b", "#daa520")])
```

`uv run examples/generate_examples.py` writes logos for `jaxogo`, `hello world`, `modax` and a
dozen JAX-based libraries (Flax, Optax, Equinox, Diffrax, …) into [`examples/`](https://github.com/lawrenceberry/jaxogo/tree/main/examples).

## Development

```bash
uv run pytest
uv run pre-commit install  # once: run ruff linting and formatting on every commit
```

The tests compare `jaxogo("JAX")` polygon by polygon against the original logo in
`tests/data/jax_logo.svg`.

The JAX logo is a real 3D voxel model seen from an isometric camera rolled by 30°, so the
x axis is horizontal, the y axis rises at 60° and the z axis at 120°. Its J lies flat on the
ground; its X is a bar along y crossed by a tower along z; and its A has no strokes of its
own apart from the crossbar, borrowing the J's stem as its left leg and the X's tower as its right.

* `src/jaxogo/_font.py` is a five-row voxel font. Glyphs are ASCII bitmaps in the ground
  plane, and `\` strokes can be lifted into towers like the X in the logo.
* `src/jaxogo/_logo.py` lays out the glyphs, joins neighbouring letters, colours each
  letter's top, front and side faces with a cycled palette, and renders the exposed faces as
  a `Poly3DCollection` with an orthographic `mplot3d` camera.

## License

jaxogo is released under the [MIT License](https://github.com/lawrenceberry/jaxogo/blob/main/LICENSE).

## Acknowledgement

The style of jaxogo comes from the [JAX logo](https://github.com/jax-ml/jax/blob/main/images/jax_logo.svg),
whose design belongs to the JAX authors. jaxogo recreates that design with its own code but does not
own it, and is an independent project that is not affiliated with or endorsed by the JAX project or
Google. The copy of the logo in `tests/data/jax_logo.svg`, used to test that `jaxogo("JAX")`
matches it, comes from the [JAX repository](https://github.com/jax-ml/jax) and is used under its
[Apache License 2.0](https://github.com/jax-ml/jax/blob/main/LICENSE).
