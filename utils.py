from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

type PointArray = NDArray[np.float64]
type PointLike = Sequence[float] | PointArray
type FieldFunction = Callable[[PointArray], PointArray]


@dataclass(slots=True)
class InductorGeometry:
    coil: PointArray
    bottom_lead: PointArray
    top_lead: PointArray


def _as_array(point: PointLike) -> PointArray:
    return np.asarray(point, dtype=float)


def line_charge_fields(
    p0: PointLike,
    p1: PointLike,
) -> tuple[FieldFunction, FieldFunction]:
    p0_arr, p1_arr = _as_array(p0), _as_array(p1)
    center = (p0_arr + p1_arr) / 2
    chord = p1_arr - p0_arr
    a = np.linalg.norm(chord) / 2
    angle = np.arctan2(chord[1], chord[0])
    rotation = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)],
    ])

    def transform(points: PointArray) -> PointArray:
        return (points - center) @ rotation.T

    def field_at_point(points: PointArray) -> PointArray:
        pts = transform(points)
        x, y = pts[..., 0], pts[..., 1]
        ex = (np.log((x + a) ** 2 + y**2) - np.log((x - a) ** 2 + y**2)) / 2
        ey = -(np.arctan2(y, x + a) - np.arctan2(y, x - a))
        e = np.stack([ex, ey], axis=-1)
        return e @ rotation

    def potential_at_point(points: PointArray) -> PointArray:
        pts = transform(points)
        x, y = pts[..., 0], pts[..., 1]
        v = ((x - a) * np.log((x - a) ** 2 + y**2) - (x + a) * np.log((x + a) ** 2 + y**2)) / 2
        v += y * (np.arctan2(y, x + a) - np.arctan2(y, x - a))
        return v

    return field_at_point, potential_at_point


def rotate_point(p: PointLike, center: PointLike, angle: float) -> PointArray:
    point = _as_array(p)
    center_arr = _as_array(center)
    shift = point - center_arr
    cosine, sine = np.cos(angle), np.sin(angle)
    rotated = np.array([cosine * shift[0] - sine * shift[1], sine * shift[0] + cosine * shift[1]])
    return rotated + center_arr


def unit(vector: PointLike) -> PointArray:
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return np.zeros_like(vec)
    return vec / norm


def rot90(vector: PointLike) -> PointArray:
    x, y = np.asarray(vector, dtype=float)
    return np.array([-y, x], dtype=float)


def capsule_outline(
    p0: PointLike,
    p1: PointLike,
    eps: float = 0.15,
    n_side: int = 200,
    n_arc: int = 64,
) -> tuple[PointArray, PointArray, PointArray]:
    p0_arr, p1_arr = _as_array(p0), _as_array(p1)
    tangent = unit(p1_arr - p0_arr)
    normal = rot90(tangent)
    up0, up1 = p0_arr + eps * normal, p1_arr + eps * normal
    dn0, dn1 = p0_arr - eps * normal, p1_arr - eps * normal

    up = np.linspace(up0, up1, n_side)
    dn = np.linspace(dn1, dn0, n_side)
    theta_top = np.linspace(0, np.pi, n_arc)
    theta_bottom = np.linspace(np.pi, 2 * np.pi, n_arc)
    arc_top = p1_arr + eps * (np.cos(theta_top)[:, None] * normal - np.sin(theta_top)[:, None] * tangent)
    arc_bottom = p0_arr + eps * (np.cos(theta_bottom)[:, None] * normal - np.sin(theta_bottom)[:, None] * tangent)

    polyline = np.vstack([up, arc_top, dn, arc_bottom])
    normals = np.vstack([
        np.repeat(normal[None, :], len(up), axis=0),
        unit(arc_top - p1_arr),
        np.repeat(-normal[None, :], len(dn), axis=0),
        unit(arc_bottom - p0_arr),
    ])
    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0, append=polyline[:1]), axis=1)

    n = n_side // 2
    polyline = np.roll(polyline, -n, axis=0)
    normals = np.roll(normals, -n, axis=0)
    segment_lengths = np.roll(segment_lengths, -n, axis=0)

    return polyline, normals, segment_lengths


def inductor_geometry(
    *,
    x0: float,
    y0: float,
    coils: int = 5,
    radius: float = 0.6,
    gap: float = 0.0,
    lead: float = 3.0,
    samples: int = 800,
) -> InductorGeometry:
    if samples < 2:
        raise ValueError("samples must be at least 2 to define a polyline.")

    pitch = 2.0 * radius + gap
    height = coils * pitch
    y_start = y0 - height / 2.0

    u = np.linspace(0.0, coils * np.pi, samples, dtype=float)
    turns = np.floor(u / np.pi)
    phase = u - turns * np.pi - np.pi / 2.0
    yc = y_start + (turns + 0.5) * pitch

    xs = np.full_like(u, x0, dtype=float) + radius * np.cos(phase)
    ys = yc + radius * np.sin(phase)

    coil = np.stack([xs, ys], axis=-1)
    bottom = np.array([[xs[0], ys[0]], [x0, ys[0] - lead]], dtype=float)
    top = np.array([[xs[-1], ys[-1]], [x0, ys[-1] + lead]], dtype=float)

    return InductorGeometry(coil=coil, bottom_lead=bottom, top_lead=top)


def streamline_seeds(
    p0: PointLike,
    p1: PointLike,
    field_func: FieldFunction,
    *,
    eps: float = 0.15,
    n_side: int = 400,
    n_arc: int = 0,
    n_seeds: int = 20,
) -> PointArray:
    polyline, normals, _segment_lengths = capsule_outline(p0, p1, eps=eps, n_side=n_side, n_arc=n_arc)
    field_values = field_func(polyline)
    flux_density = np.einsum("ij,ij->i", field_values, normals)
    total_flux = float(flux_density.sum())
    assert not np.isclose(total_flux, 0)
    cdf = np.cumsum(flux_density) / total_flux
    targets = (np.arange(n_seeds) + 0.5) / n_seeds
    indices = np.searchsorted(cdf, targets, side="right")
    offset = 1e-3 * eps
    return polyline[indices] + offset * normals[indices]


def field_integrator(
    field_fn: FieldFunction,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ds: float = 0.05,
    max_steps: int = 10000,
) -> Callable[[float, float, str], tuple[PointArray, str]]:
    def integrate(x0: float, y0: float, direction: str = "both") -> tuple[PointArray, str]:
        if direction == "forward":
            dt = ds
        elif direction == "backward":
            dt = -ds
        else:
            traj_forward, reason_forward = integrate(x0, y0, direction="forward")
            traj_backward, _ = integrate(x0, y0, direction="backward")
            combined = np.vstack([traj_backward[::-1], traj_forward[1:]])
            return combined, reason_forward

        trajectory = [np.array([x0, y0], dtype=float)]
        x, y = float(x0), float(y0)
        prev_direction: PointArray | None = None
        for _ in range(max_steps):
            if not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]):
                return np.array(trajectory), "out_of_bounds"

            field_value = field_fn(np.array([[x, y]], dtype=float))[0]
            speed = np.linalg.norm(field_value)
            if speed < 1e-9:
                return np.array(trajectory), "stalled"

            direction_vec = field_value / speed
            if prev_direction is not None and np.dot(prev_direction, direction_vec) < 0:
                return np.array(trajectory), "stalled"

            x += direction_vec[0] * dt
            y += direction_vec[1] * dt
            trajectory.append(np.array([x, y], dtype=float))
            prev_direction = direction_vec

        return np.array(trajectory), "max_steps"

    return integrate


__all__ = [
    "PointArray",
    "PointLike",
    "FieldFunction",
    "InductorGeometry",
    "capsule_outline",
    "line_charge_fields",
    "streamline_seeds",
    "inductor_geometry",
    "field_integrator",
    "rotate_point",
    "rot90",
    "unit",
]
