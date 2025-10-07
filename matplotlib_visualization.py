import os
from contextlib import contextmanager

import joblib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import collections as mcollections
from matplotlib import patches as mpatches
from tqdm import tqdm

from utils import (
    field_integrator,
    inductor_geometry,
    line_charge_fields,
    rotate_point,
    streamline_seeds,
)


def smooth(t: float) -> float:
    s = 1 - t
    return (t**3) * (10 * s * s + 5 * s * t + t * t)


xlim = (-30.0, 30.0)
ylim = (-16.875, 16.875)
X, Y = np.meshgrid(np.linspace(*xlim, 1000), np.linspace(*ylim, 1000))
Pgrid = np.stack([X, Y], axis=-1)
qa, qb = 1.0, -1.0


@contextmanager
def tqdm_joblib(pbar: tqdm):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback, joblib.parallel.BatchCompletionCallBack = joblib.parallel.BatchCompletionCallBack, TqdmBatchCompletionCallback
    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        pbar.close()


FIELD_LINEWIDTH = 1.5
CIRCUIT_LINEWIDTH = 3.0
PLATE_LINEWIDTH = 6.0

FIELD_COLOR = "#343434"
CIRCUIT_COLOR = "#000000"

POTENTIAL_CMAP = "RdBu_r"


def render_frame(theta: float, save_path: str):
    p0a, p1a, p2a, p3a, p4a = (-2, 1), (2, 1), (0, 1), (0, 6), (5, 6)
    p0b, p1b, p2b, p3b, p4b = (-2, -1), (2, -1), (0, -1), (0, -6), (5, -6)
    p0a, p1a, p2a = map(lambda p: rotate_point(p, p3a, -theta), [p0a, p1a, p2a])
    p0b, p1b, p2b = map(lambda p: rotate_point(p, p3b, theta), [p0b, p1b, p2b])
    fa, ga = line_charge_fields(p0a, p1a)
    fb, gb = line_charge_fields(p0b, p1b)

    def field_total(P):
        return qa * fa(P) + qb * fb(P)

    def potential_total(P):
        return qa * ga(P) + qb * gb(P)

    seeds_pos = streamline_seeds(p0a, p1a, field_total, eps=0.001, n_side=2000, n_seeds=20)

    def field_difference(P):
        return fa(P) - fb(P)

    integrate = field_integrator(field_difference, xlim=xlim, ylim=ylim)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.gca()
    streamlines = []
    for p in seeds_pos:
        trajectory, state = integrate(p[0], p[1], "both")
        if state == "out_of_bounds":
            t1, t2 = trajectory, trajectory[::-1].copy()
            t2[:, 1] = -t2[:, 1]
            streamlines.append(t1)
            streamlines.append(t2)
        elif state == "stalled":
            trajectory_inv = trajectory[::-1].copy()
            trajectory_inv[:, 1] = -trajectory_inv[:, 1]
            trajectory = (trajectory_inv + trajectory) / 2
            streamlines.append(trajectory)
        else:
            raise RuntimeError("Unexpected state:", state)

    ax.contourf(X, Y, potential_total(Pgrid), levels=np.linspace(-14, 14, 31), cmap=POTENTIAL_CMAP, alpha=0.7)
    ax.add_collection(mcollections.LineCollection(streamlines, color=FIELD_COLOR, linewidth=FIELD_LINEWIDTH, zorder=2))
    ax.add_collection(mcollections.PatchCollection((mpatches.FancyArrowPatch(posA=line[loc := len(line) // 2], posB=line[loc + 1]) for line in streamlines), zorder=3, facecolor=FIELD_COLOR))

    ax.plot([p0a[0], p1a[0]], [p0a[1], p1a[1]], color=CIRCUIT_COLOR, linewidth=PLATE_LINEWIDTH)
    ax.plot([p0b[0], p1b[0]], [p0b[1], p1b[1]], color=CIRCUIT_COLOR, linewidth=PLATE_LINEWIDTH)
    ax.plot([p2a[0], p3a[0], p4a[0]], [p2a[1], p3a[1], p4a[1]], color=CIRCUIT_COLOR, linewidth=CIRCUIT_LINEWIDTH)

    geom = inductor_geometry(x0=p4a[0], y0=0.0, coils=5, radius=3 / 5, gap=0.0, lead=3.0, samples=4000)
    ax.plot(geom.coil[:, 0], geom.coil[:, 1], linewidth=CIRCUIT_LINEWIDTH, color=CIRCUIT_COLOR)
    ax.plot(geom.bottom_lead[:, 0], geom.bottom_lead[:, 1], linewidth=CIRCUIT_LINEWIDTH, color=CIRCUIT_COLOR)
    ax.plot(geom.top_lead[:, 0], geom.top_lead[:, 1], linewidth=CIRCUIT_LINEWIDTH, color=CIRCUIT_COLOR)
    ax.plot([p2b[0], p3b[0], p4b[0]][::-1], [p2b[1], p3b[1], p4b[1]][::-1], color=CIRCUIT_COLOR, linewidth=CIRCUIT_LINEWIDTH)
    ax.set_aspect("equal", "box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


if __name__ == "__main__":
    output_video_path = "videos/animation_matplotlib.mp4"

    n_frames = 400
    n_jobs = joblib.cpu_count(only_physical_cores=True)

    os.makedirs("frames", exist_ok=True)
    with tqdm_joblib(tqdm(total=n_frames + 1)):
        Parallel(n_jobs=n_jobs, prefer="processes")(delayed(render_frame)(theta=smooth(i / n_frames) * np.pi, save_path=f"frames/{i:03d}.png") for i in range(n_frames + 1))

    ffmpeg_command = f'ffmpeg -y -r 60 -i frames/%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx265 -pix_fmt yuv420p {output_video_path}'
    os.system(ffmpeg_command)
