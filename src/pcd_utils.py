"""Minimal PCD helpers for AprilTag 3D segmentation."""

from __future__ import annotations

import struct
from typing import Sequence

import numpy as np

_TYPE_TO_STRUCT = {
    ("F", 4): "f",
    ("U", 4): "I",
    ("I", 4): "i",
}


def read_pcd_xyz(data: bytes) -> np.ndarray:
    """Parse x/y/z from a PCD blob (ascii or binary)."""
    marker = b"DATA "
    idx = data.find(marker)
    if idx < 0:
        raise ValueError("invalid PCD: missing DATA section")

    header = data[:idx].decode("utf-8", errors="replace")
    rest = data[idx + len(marker) :]
    mode_line, _, payload = rest.partition(b"\n")
    mode = mode_line.strip().decode("ascii").lower()

    meta: dict[str, list[str]] = {}
    for line in header.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            meta[parts[0].upper()] = parts[1:]

    fields = meta.get("FIELDS", ["x", "y", "z"])
    sizes = [int(v) for v in meta.get("SIZE", ["4"] * len(fields))]
    types = meta.get("TYPE", ["F"] * len(fields))
    counts = [int(v) for v in meta.get("COUNT", ["1"] * len(fields))]
    n_points = int(meta.get("POINTS", ["0"])[0])

    try:
        x_idx = fields.index("x")
        y_idx = fields.index("y")
        z_idx = fields.index("z")
    except ValueError as e:
        raise ValueError("PCD must contain x, y, z fields") from e

    if mode == "ascii":
        points = np.empty((n_points, 3), dtype=np.float64)
        rows = payload.decode("utf-8", errors="replace").splitlines()
        if len(rows) < n_points:
            raise ValueError("PCD ascii payload shorter than POINTS count")
        for i, row in enumerate(rows[:n_points]):
            cols = row.split()
            points[i, 0] = float(cols[x_idx])
            points[i, 1] = float(cols[y_idx])
            points[i, 2] = float(cols[z_idx])
        return points

    if mode != "binary":
        raise ValueError(f"unsupported PCD DATA mode: {mode!r}")

    point_step = sum(s * c for s, c in zip(sizes, counts))
    needed = n_points * point_step
    if len(payload) < needed:
        raise ValueError("PCD binary payload shorter than expected")

    def _field_offset(field_index: int) -> int:
        off = 0
        for i in range(field_index):
            off += sizes[i] * counts[i]
        return off

    fmt_parts: list[str] = []
    for size, typ, count in zip(sizes, types, counts):
        key = (typ, size)
        if key not in _TYPE_TO_STRUCT:
            raise ValueError(f"unsupported PCD field type {typ} size {size}")
        fmt_parts.append(_TYPE_TO_STRUCT[key] * count)
    point_fmt = "<" + "".join(fmt_parts)

    points = np.empty((n_points, 3), dtype=np.float64)
    x_off = _field_offset(x_idx)
    y_off = _field_offset(y_idx)
    z_off = _field_offset(z_idx)

    for i in range(n_points):
        chunk = payload[i * point_step : (i + 1) * point_step]
        values = struct.unpack(point_fmt, chunk)
        points[i, 0] = values[sum(counts[:x_idx])]
        points[i, 1] = values[sum(counts[:y_idx])]
        points[i, 2] = values[sum(counts[:z_idx])]
    return points


def write_pcd_xyz(points: np.ndarray) -> bytes:
    """Encode an Nx3 xyz array as an ascii PCD blob."""
    if points.size == 0:
        points = np.zeros((0, 3), dtype=np.float64)
    n = int(points.shape[0])
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    if n == 0:
        return header.encode("ascii")
    lines = [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points]
    return (header + "\n".join(lines)).encode("ascii")


def filter_points_in_bbox(
    points: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray:
    """Keep 3D points whose projection falls inside the 2D bbox."""
    if points.size == 0:
        return points.reshape(0, 3)

    fx, fy, cx, cy = intrinsics
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    valid = zs > 1e-6
    u = fx * xs / zs + cx
    v = fy * ys / zs + cy
    mask = (
        valid
        & (u >= x_min)
        & (u <= x_max)
        & (v >= y_min)
        & (v <= y_max)
    )
    return points[mask]


def segment_geometry(points: np.ndarray, label: str):
    """Build Geometry proto fields from an xyz segment."""
    from viam.proto.common import Geometry, Pose, RectangularPrism, Vector3

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    dims = maxs - mins
    return Geometry(
        center=Pose(x=float(center[0]), y=float(center[1]), z=float(center[2])),
        box=RectangularPrism(
            dims_mm=Vector3(
                x=max(float(dims[0]), 1.0),
                y=max(float(dims[1]), 1.0),
                z=max(float(dims[2]), 1.0),
            )
        ),
        label=label,
    )
