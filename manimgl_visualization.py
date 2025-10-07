import os
import sys
from typing import Any, Callable, Sequence

import numpy as np
from manimlib import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from utils import (
    PointArray,
    field_integrator,
    inductor_geometry,
    line_charge_fields,
    rotate_point,
    streamline_seeds,
)

STREAM_LINE_COLOR = GREY_E
STREAM_LINE_STROKE_WIDTH = 2.0
STREAM_LINE_OPACITY = 0.3

EQUIPOTENTIAL_COLOR = BLUE_E
EQUIPOTENTIAL_STROKE_WIDTH = 1.4
EQUIPOTENTIAL_OPACITY = 0.4
EQUIPOTENTIAL_DASH_LENGTH = 0.6
EQUIPOTENTIAL_GAP_LENGTH = 0.4

FLASH_COLOR = MAROON_E
FLASH_WIDTH_SCALE = 1.5
FLASH_TIME_WIDTH = 0.8

ARROW_COLOR = GREY_E
ARROW_OPACITY = 0.5
ARROW_TIP_SCALE = 1.0

CIRCUIT_COLOR = BLACK
CIRCUIT_LINEWIDTH = 4.0
CONDUCTOR_LINEWIDTH = 8.0


def compute_virtual_time(points: PointArray) -> float:
    if len(points) < 2:
        return 4.0
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    path_len = lengths.sum()
    return np.clip(path_len / 3.5, 4.0, 12.0)


def set_inductor_geometry(
    group: VGroup,
    axes: CoordinateSystem,
    *,
    x0: float,
    y0: float,
    coils: int = 5,
    radius: float = 0.6,
    gap: float = 0.0,
    lead: float = 3.0,
    stroke_width: float = CIRCUIT_LINEWIDTH,
    color: Any = CIRCUIT_COLOR,
    samples: int = 800,
) -> None:
    coil, lead_bottom, lead_top = group

    geometry = inductor_geometry(
        x0=x0,
        y0=y0,
        coils=coils,
        radius=radius,
        gap=gap,
        lead=lead,
        samples=samples,
    )

    coil_points = np.array([axes.c2p(x, y, 0.0) for x, y in geometry.coil])
    coil.set_points_as_corners(coil_points)
    coil.set_stroke(color=color, width=stroke_width)

    bottom_pts = [axes.c2p(pt[0], pt[1], 0.0) for pt in geometry.bottom_lead]
    top_pts = [axes.c2p(pt[0], pt[1], 0.0) for pt in geometry.top_lead]

    lead_bottom.put_start_and_end_on(*bottom_pts)
    lead_top.put_start_and_end_on(*top_pts)
    lead_bottom.set_stroke(color=color, width=stroke_width)
    lead_top.set_stroke(color=color, width=stroke_width)


def create_inductor(axes: CoordinateSystem, **kwargs: Any) -> VGroup:
    coil = VMobject()
    lead_bottom = Line(ORIGIN, ORIGIN)
    lead_top = Line(ORIGIN, ORIGIN)
    group = VGroup(coil, lead_bottom, lead_top)
    set_inductor_geometry(group, axes, **kwargs)
    return group


from typing import TypedDict


class LayoutDict(TypedDict):
    top: tuple[PointArray, PointArray]
    bottom: tuple[PointArray, PointArray]
    bridge_top: tuple[PointArray, PointArray, PointArray]
    bridge_bottom: tuple[PointArray, PointArray, PointArray]
    inductor_anchor: PointArray


def generate_streamline_configuration(
    deg: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_seeds: int = 20,
) -> tuple[list[tuple[PointArray, float]], LayoutDict, Callable[[PointArray], PointArray]]:
    qa, qb = 1.0, -1.0
    theta = np.deg2rad(deg)

    p0a, p1a, p2a, p3a, p4a = np.array([
        [-2, 1],
        [2, 1],
        [0, 1],
        [0, 6],
        [5, 6],
    ])
    p0b, p1b, p2b, p3b, p4b = np.array([
        [-2, -1],
        [2, -1],
        [0, -1],
        [0, -6],
        [5, -6],
    ])

    p0a, p1a, p2a = [rotate_point(p, p3a, -theta) for p in (p0a, p1a, p2a)]
    p0b, p1b, p2b = [rotate_point(p, p3b, theta) for p in (p0b, p1b, p2b)]

    field_a, potential_a = line_charge_fields(p0a.tolist(), p1a.tolist())
    field_b, potential_b = line_charge_fields(p0b.tolist(), p1b.tolist())

    def field_total(points: PointArray) -> PointArray:
        return qa * field_a(points) + qb * field_b(points)

    def potential_total(points: PointArray) -> PointArray:
        return qa * potential_a(points) + qb * potential_b(points)

    seeds = streamline_seeds(p0a, p1a, field_total, eps=0.001, n_side=1200, n_seeds=n_seeds)

    integrate = field_integrator(field_total, xlim=xlim, ylim=ylim)
    polyline_list: list[PointArray] = []

    for seed in seeds:
        trajectory, state = integrate(seed[0], seed[1], "both")
        if trajectory.shape[0] < 2:
            continue

        if state == "stalled":
            mirrored = trajectory[::-1].copy()
            mirrored[:, 1] = -mirrored[:, 1]
            trajectory = (trajectory + mirrored) / 2
            upper = lower = trajectory
        else:
            upper = trajectory
            lower = trajectory[::-1].copy()
            lower[:, 1] = -lower[:, 1]

        polyline_list.append(upper)
        polyline_list.append(lower)

    line_data = [(line, compute_virtual_time(line)) for line in polyline_list]
    layout = LayoutDict({
        "top": (p0a, p1a),
        "bottom": (p0b, p1b),
        "bridge_top": (p2a, p3a, p4a),
        "bridge_bottom": (p2b, p3b, p4b),
        "inductor_anchor": p4a,
    })
    return line_data, layout, potential_total


def create_stream_group(
    axes: CoordinateSystem,
    line_data: list[tuple[PointArray, float]],
    *,
    color: Any = STREAM_LINE_COLOR,
    stroke_width: float = STREAM_LINE_STROKE_WIDTH,
    opacity: float = STREAM_LINE_OPACITY,
) -> VGroup:
    group = VGroup()
    for points, virtual_time in line_data:
        mob = VMobject()
        coords = np.array([axes.c2p(x, y, 0.0) for x, y in points])
        mob.set_points_as_corners(coords)
        mob.set_stroke(color=color, width=stroke_width, opacity=opacity)
        setattr(mob, "virtual_time", virtual_time)
        group.add(mob)
    return group


def update_stream_group(
    axes: CoordinateSystem,
    group: VGroup,
    line_data: list[tuple[PointArray, float]],
    *,
    color: Any = STREAM_LINE_COLOR,
    stroke_width: float = STREAM_LINE_STROKE_WIDTH,
    opacity: float = STREAM_LINE_OPACITY,
) -> None:
    assert len(group) == len(line_data)
    for mob, (points, virtual_time) in zip(group, line_data):
        coords = np.array([axes.c2p(x, y, 0.0) for x, y in points])
        mob.set_points_as_corners(coords)
        mob.set_stroke(color=color, width=stroke_width, opacity=opacity)
        setattr(mob, "virtual_time", virtual_time)


_MARCHING_SQUARES_CASES: dict[int, tuple[tuple[int, int], ...]] = {
    0: (),
    1: ((3, 0),),
    2: ((0, 1),),
    3: ((3, 1),),
    4: ((1, 2),),
    5: ((3, 2), (0, 1)),
    6: ((0, 2),),
    7: ((3, 2),),
    8: ((2, 3),),
    9: ((0, 2),),
    10: ((3, 0), (1, 2)),
    11: ((1, 2),),
    12: ((3, 1),),
    13: ((0, 1),),
    14: ((3, 0),),
    15: (),
}

_EDGE_CORNERS: dict[int, tuple[int, int]] = {
    0: (0, 1),
    1: (1, 2),
    2: (2, 3),
    3: (3, 0),
}


def _interpolate_edge(
    edge: int,
    corners: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    values: tuple[float, float, float, float],
    level: float,
) -> np.ndarray:
    i0, i1 = _EDGE_CORNERS[edge]
    p0, p1 = corners[i0], corners[i1]
    v0, v1 = values[i0], values[i1]
    if np.isclose(v0, v1):
        t = 0.5
    else:
        t = (level - v0) / (v1 - v0)
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - t) * p0 + t * p1


def _marching_squares(
    grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    level: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rows, cols = grid.shape
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = grid[i, j]
            v1 = grid[i, j + 1]
            v2 = grid[i + 1, j + 1]
            v3 = grid[i + 1, j]
            idx = ((v0 >= level) << 0) | ((v1 >= level) << 1) | ((v2 >= level) << 2) | ((v3 >= level) << 3)
            cases = _MARCHING_SQUARES_CASES[idx]
            if not cases:
                continue
            corners = (
                np.array([xs[j], ys[i]], dtype=float),
                np.array([xs[j + 1], ys[i]], dtype=float),
                np.array([xs[j + 1], ys[i + 1]], dtype=float),
                np.array([xs[j], ys[i + 1]], dtype=float),
            )
            values = (v0, v1, v2, v3)
            for edge0, edge1 in cases:
                p_start = _interpolate_edge(edge0, corners, values, level)
                p_end = _interpolate_edge(edge1, corners, values, level)
                segments.append((p_start, p_end))
    return segments


def _connect_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    tol: float = 1e-3,
) -> list[np.ndarray]:
    if not segments:
        return []
    unused = segments.copy()
    polylines: list[np.ndarray] = []

    def pop_matching(point: np.ndarray) -> tuple[bool, np.ndarray | None]:
        for idx, (start, end) in enumerate(unused):
            if np.linalg.norm(start - point) <= tol:
                unused.pop(idx)
                return True, end
            if np.linalg.norm(end - point) <= tol:
                unused.pop(idx)
                return True, start
        return False, None

    while unused:
        start_seg = unused.pop()
        poly = [start_seg[0], start_seg[1]]
        extended = True
        while extended:
            extended = False
            matched, next_point = pop_matching(poly[-1])
            if matched and next_point is not None:
                poly.append(next_point)
                extended = True
                continue
            matched, prev_point = pop_matching(poly[0])
            if matched and prev_point is not None:
                poly.insert(0, prev_point)
                extended = True
        polylines.append(np.array(poly, dtype=float))

    return polylines


def _build_dashed_segments(
    coords: np.ndarray,
    *,
    dash_length: float,
    gap_length: float,
) -> VGroup:
    segments_group = VGroup()
    if len(coords) < 2:
        return segments_group

    dash_on = True
    length_left = dash_length

    for idx in range(len(coords) - 1):
        start = coords[idx]
        end = coords[idx + 1]
        vec = end - start
        seg_len = np.linalg.norm(vec)
        if seg_len < 1e-8:
            continue
        direction = vec / seg_len
        pos = start.copy()
        remaining = seg_len
        while remaining > 1e-8:
            step = min(length_left, remaining)
            next_pos = pos + direction * step
            if dash_on:
                line = Line(pos, next_pos)
                segments_group.add(line)
            pos = next_pos
            remaining -= step
            length_left -= step
            if length_left <= 1e-8:
                dash_on = not dash_on
                length_left = dash_length if dash_on else gap_length

    return segments_group


def generate_equipotential_curves(
    potential_fn: Callable[[PointArray], PointArray],
    levels: Sequence[float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    grid_size: int = 160,
) -> list[tuple[PointArray, float]]:
    xs = np.linspace(xlim[0], xlim[1], grid_size, dtype=float)
    ys = np.linspace(ylim[0], ylim[1], grid_size, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    values = potential_fn(points).reshape(ys.size, xs.size)

    curves: list[tuple[PointArray, float]] = []
    for level in levels:
        segments = _marching_squares(values, xs, ys, level)
        polylines = _connect_segments(segments)
        for poly in polylines:
            if poly.shape[0] < 2:
                continue
            curves.append((poly, level))
    return curves


def create_equipotential_group(
    axes: CoordinateSystem,
    curve_data: list[tuple[PointArray, float]],
    *,
    stroke_width: float = EQUIPOTENTIAL_STROKE_WIDTH,
    opacity: float = EQUIPOTENTIAL_OPACITY,
    dash_length: float = EQUIPOTENTIAL_DASH_LENGTH,
    gap_length: float = EQUIPOTENTIAL_GAP_LENGTH,
    color: Any = EQUIPOTENTIAL_COLOR,
) -> VGroup:
    group = VGroup()
    update_equipotential_group(
        axes,
        group,
        curve_data,
        stroke_width=stroke_width,
        opacity=opacity,
        dash_length=dash_length,
        gap_length=gap_length,
        color=color,
    )
    return group


def update_equipotential_group(
    axes: CoordinateSystem,
    group: VGroup,
    curve_data: list[tuple[PointArray, float]],
    *,
    stroke_width: float = EQUIPOTENTIAL_STROKE_WIDTH,
    opacity: float = EQUIPOTENTIAL_OPACITY,
    dash_length: float = EQUIPOTENTIAL_DASH_LENGTH,
    gap_length: float = EQUIPOTENTIAL_GAP_LENGTH,
    color: Any = EQUIPOTENTIAL_COLOR,
) -> None:
    while len(group) < len(curve_data):
        group.add(VGroup())
    while len(group) > len(curve_data):
        group.remove(group[-1])

    for mob, (points, value) in zip(group, curve_data):
        coords = np.array([axes.c2p(x, y, 0.0) for x, y in points])
        dashed_segments = _build_dashed_segments(
            coords,
            dash_length=dash_length,
            gap_length=gap_length,
        )
        mob.become(dashed_segments)
        for segment in mob:
            segment.set_stroke(color=color, width=stroke_width, opacity=opacity)
        setattr(mob, "potential_level", value)


class FlashingStreamLines(VGroup):
    def __init__(
        self,
        base_group: VGroup,
        *,
        color: Any = FLASH_COLOR,
        width_scale: float = FLASH_WIDTH_SCALE,
        time_width: float = FLASH_TIME_WIDTH,
        rate_func: Callable[[float], float] = linear,
    ) -> None:
        super().__init__()
        self.base_group = base_group
        self.color = Color(color)
        self.width_scale = width_scale
        self.time_width = time_width
        self.rate_func = rate_func
        self.entries: list[dict[str, Any]] = []
        self.sync()
        self.add_updater(self._tick)

    def sync(self) -> None:
        for entry in self.entries:
            entry["flash"].clear_updaters()
        self.clear()
        self.entries.clear()
        for line in self.base_group:
            self.entries.append(self._make_entry(line))

    def _make_entry(self, line: VMobject) -> dict[str, Any]:
        flash = line.copy()
        flash.match_style(line)
        flash.set_stroke(
            color=self.color,
            width=float(line.get_stroke_width() * self.width_scale),
            opacity=1.0,
        )

        def follow(mob: Mobject, _dt: float = 0.0) -> None:
            mob.set_points(line.get_points())

        flash.add_updater(follow)

        run_time = max(getattr(line, "virtual_time", 6.0), 1e-3)
        anim = VShowPassingFlash(
            flash,
            time_width=self.time_width,
            taper_width=0.0,
            run_time=run_time,
            rate_func=self.rate_func,
            remover=False,
        )
        anim.begin()
        entry = {"anim": anim, "flash": anim.mobject, "time": np.random.uniform(-run_time, 0.0)}
        self.add(entry["flash"])
        return entry

    def _tick(self, _mob: Mobject, dt: float) -> None:
        for entry in self.entries:
            anim: VShowPassingFlash = entry["anim"]
            run_time = anim.run_time
            if run_time <= 1e-6:
                continue
            entry["time"] += dt
            phase = max(entry["time"], 0.0) % run_time
            anim.update(phase / run_time)


class StreamLineArrows(VGroup):
    def __init__(
        self,
        base_group: VGroup,
        *,
        color: Any = ARROW_COLOR,
        opacity: float = ARROW_OPACITY,
        tip_scale: float = ARROW_TIP_SCALE,
    ) -> None:
        super().__init__()
        self.base_group = base_group
        self.color = Color(color)
        self.opacity = opacity
        self.tip_scale = tip_scale
        self.sync()

    def sync(self) -> None:
        for tip in list(self):
            tip.clear_updaters()
        self.clear()
        for line in self.base_group:
            tip = ArrowTip(fill_color=self.color, fill_opacity=self.opacity)
            tip.scale(self.tip_scale)
            tip.add_updater(self._build_updater(line))
            self.add(tip)

    def _build_updater(self, line: VMobject) -> Callable[[Mobject, float], None]:
        def updater(mob: Mobject, _dt: float = 0.0) -> None:
            alpha = 0.5
            p_mid = line.point_from_proportion(alpha)
            forward = line.point_from_proportion(min(alpha + 1e-3, 1.0))
            backward = line.point_from_proportion(max(alpha - 1e-3, 0.0))
            direction = np.asarray(forward) - np.asarray(backward)
            angle = np.arctan2(direction[1], direction[0])
            template = ArrowTip(fill_color=self.color, fill_opacity=self.opacity)
            template.scale(self.tip_scale)
            template.rotate(angle)
            template.move_to(p_mid)
            mob.become(template)

        return updater


class ElectrodeStreamLines(Scene):
    xlim = (-30.0, 30.0)
    ylim = (-16.875, 16.875)

    def construct(self) -> None:
        axes = NumberPlane(x_range=(*self.xlim, 3), y_range=(*self.ylim, 3))
        axes.add_coordinate_labels(font_size=16)

        frame = self.camera.frame
        frame.set_width(self.xlim[1] - self.xlim[0])
        frame.set_height(self.ylim[1] - self.ylim[0])
        frame.move_to(axes.c2p((self.xlim[0] + self.xlim[1]) / 2, (self.ylim[0] + self.ylim[1]) / 2, 0.0))

        deg_tracker = ValueTracker(0.0)

        def current_angle() -> float:
            return deg_tracker.get_value()  # type: ignore

        potential_levels = np.linspace(-14.0, 14.0, 21).tolist()

        line_data, layout, potential_fn = generate_streamline_configuration(current_angle(), self.xlim, self.ylim)
        stream_lines = create_stream_group(axes, line_data)
        stream_lines.set_z_index(30)

        equipotential_data = generate_equipotential_curves(potential_fn, potential_levels, self.xlim, self.ylim)
        equipotential = create_equipotential_group(axes, equipotential_data)
        equipotential.set_z_index(20)

        arrows = StreamLineArrows(stream_lines)
        arrows.set_z_index(100)

        flashing = FlashingStreamLines(stream_lines)
        flashing.set_z_index(40)

        top_segment = VMobject()
        bottom_segment = VMobject()
        top_connector = VMobject()
        bottom_connector = VMobject()
        inductor = create_inductor(axes, x0=layout["inductor_anchor"][0], y0=0.0, coils=5, radius=0.6, gap=0.0, lead=3.0)

        for mob in (top_segment, bottom_segment, top_connector, bottom_connector, inductor):
            mob.set_z_index(90)
        for mob in (top_segment, bottom_segment):
            mob.set_stroke(color=CIRCUIT_COLOR, width=CONDUCTOR_LINEWIDTH)
        for mob in (top_connector, bottom_connector, inductor):
            mob.set_stroke(color=CIRCUIT_COLOR, width=CIRCUIT_LINEWIDTH)

        def apply_layout(data: LayoutDict) -> None:
            top_segment.set_points_as_corners([axes.c2p(*pt, 0) for pt in data["top"]])
            bottom_segment.set_points_as_corners([axes.c2p(*pt, 0) for pt in data["bottom"]])
            top_connector.set_points_as_corners([axes.c2p(*pt, 0) for pt in data["bridge_top"]])
            bottom_connector.set_points_as_corners([axes.c2p(*pt, 0) for pt in data["bridge_bottom"]])
            set_inductor_geometry(inductor, axes, x0=data["inductor_anchor"][0], y0=0.0, coils=5, radius=0.6, gap=0.0, lead=3.0, stroke_width=CIRCUIT_LINEWIDTH, color=CIRCUIT_COLOR)

        apply_layout(layout)

        controller = VGroup()
        state = {"deg": current_angle()}

        def controller_update(_mob: Mobject, _dt: float = 0.0) -> None:
            deg = current_angle()
            if np.isclose(deg, state["deg"], atol=1e-4):
                return
            new_lines, new_layout, new_potential = generate_streamline_configuration(deg, self.xlim, self.ylim)
            update_stream_group(axes, stream_lines, new_lines)
            arrows.sync()
            equip_data = generate_equipotential_curves(new_potential, potential_levels, self.xlim, self.ylim)
            update_equipotential_group(axes, equipotential, equip_data)
            apply_layout(new_layout)
            state["deg"] = deg

        controller.add_updater(controller_update)

        self.add(
            equipotential,
            stream_lines,
            arrows,
            flashing,
            top_segment,
            bottom_segment,
            top_connector,
            bottom_connector,
            inductor,
            controller,
        )

        self.wait(5)
        self.play(deg_tracker.animate.set_value(180.0), run_time=20.0, rate_func=smooth)
        self.wait(10)
