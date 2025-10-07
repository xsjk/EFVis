# Electrode Field Visualizations

Two complementary scripts render the same electrode setup:

- **`matplotlib_visualization.py`** — parallel frame renderer that uses `joblib` to generate PNGs then merging them with `ffmpeg`.
- **`manimgl_visualization.py`** — ManimGL renderer that writes the animation in a single process with flashing streamlines.

The embedded clip produced with the Matplotlib pipeline typically finishes in about a minute while the ManimGL renderer can take over an hour.

## Electrode setup

The animation tracks a pair of charged plates that begin as a classic parallel-plate capacitor. The upper plate is positively charged, the lower plate is negatively charged. As the plates rotate away from one another, the electric field dissipates to the surrounding space.

### Note on the 2D visualization of electric fields

Although the visualization is presented in 2D, it represents a projection of a field derived from a 3D model. The plates are conceptualized as infinite strips in 3D space, specifically at $ ([-a, a], \pm d/2, (-\infty, \infty)) $. This modeling choice is not arbitrary but a physical imperative.

The formula of electric field produced by a point charge is governed by the dimensionality in Gauss's law ($\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$). In 3D space, it follows **1/r²**, while in true 2D space, it follows **1/r**.

Attempting to force a **1/r²** field model in purely 2D simulations violates Gauss's theorem for 2D space. This inconsistency disrupts the fundamental relationship between field lines and charge. To maintain proportional field line density while keeping lines directed from positive to negative charges, the density near charges would need to diverge unnaturally.

Therefore I perform a 3D physical simulation to leverage the square-law field, and then project the result onto 2D for visualization.

For a single charged plate located at $y = 0$, $x \in [-a, a]$, ($z \in [-\infty, \infty]$ is omitted) with constant charge density $\sigma$, the electric field experienced by a unit point charge at $(x, y, 0)$ is:

$$
\mathbf{F} = \frac{\sigma}{2\pi\epsilon_0} \left[ \frac{1}{2}  \ln \left( \frac{(x + a)^2 + y^2}{(x - a)^2 + y^2} \right), \arctan\left( \frac{a - x}{y} \right) + \arctan\left( \frac{a + x}{y} \right), 0 \right]
$$

The corresponding electric potential (up to an additive constant) is:

$$
\phi = \frac{\sigma}{4\pi\epsilon_0} \left( (x+a)\ln((x+a)^2+y^2) - (x-a)\ln((x-a)^2+y^2) + 2y\left(\arctan\frac{x+a}{y} - \arctan\frac{x-a}{y}\right) \right)
$$

For the two-plate system, these solutions are superimposed with appropriate coordinate shifts and sign changes.

## Quick start

```shell
uv sync
```

### Matplotlib Render

```shell
uv run matplotlib_visualization.py
```

### ManimGL Render

```shell
# Live preview
uv run manimgl manimgl_visualization.py -c "#FFFFFF" ElectrodeStreamLines

# Render
uv run manimgl manimgl_visualization.py ElectrodeStreamLines -c "#FFFFFF" -w -r 3840x2160 --fps 60 --file_name animation --open
```
