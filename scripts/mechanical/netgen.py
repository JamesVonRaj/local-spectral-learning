"""Periodic random-Delaunay network construction.

This module is a compact implementation built from NumPy and SciPy public
APIs.  It replaces the previously vendored upstream source while preserving
the network-generation behavior used for the publication data.

The algorithm tiles points near the periodic boundaries, computes a Delaunay
triangulation of that extended point cloud, keeps edges that meet the central
box, wraps the retained vertices, merges periodic copies, and evaluates edge
lengths with the minimum-image convention.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
from scipy.spatial import Delaunay


def _inside_box(point: np.ndarray, box: np.ndarray) -> bool:
    return bool(np.all(point >= 0.0) and np.all(point <= box))


def _segment_meets_box(
    start: np.ndarray, end: np.ndarray, box: np.ndarray
) -> bool:
    """Return whether a closed segment intersects the central periodic box."""
    if _inside_box(start, box) or _inside_box(end, box):
        return True

    delta = end - start
    for axis, length in enumerate(box):
        if abs(delta[axis]) < 1.0e-14:
            continue
        for boundary in (0.0, length):
            fraction = (boundary - start[axis]) / delta[axis]
            if 0.0 <= fraction <= 1.0:
                point = start + fraction * delta
                if _inside_box(point, box):
                    return True
    return False


def _periodic_replicas(
    points: np.ndarray, box: np.ndarray, buffer_distance: float
) -> np.ndarray:
    """Tile only the point copies that lie near the central box."""
    replicas = [points]
    for offset in product((-1, 0, 1), repeat=points.shape[1]):
        if not any(offset):
            continue
        shifted = points + np.asarray(offset, dtype=float) * box
        keep = np.all(
            (shifted >= -buffer_distance)
            & (shifted <= box + buffer_distance),
            axis=1,
        )
        replicas.append(shifted[keep])
    return np.vstack(replicas)


def _delaunay_edges(points: np.ndarray) -> np.ndarray:
    simplices = Delaunay(points).simplices
    pairs = np.vstack(
        [simplices[:, pair] for pair in combinations(range(points.shape[1] + 1), 2)]
    )
    return np.unique(np.sort(pairs, axis=1), axis=0)


def _merge_periodic_copies(
    points: np.ndarray,
    edges: np.ndarray,
    box: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge wrapped copies while retaining the first representative."""
    count = len(points)
    parent = np.arange(count)
    rank = np.zeros(count, dtype=int)

    def find(index: int) -> int:
        if parent[index] != index:
            parent[index] = find(int(parent[index]))
        return int(parent[index])

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    for left, right in combinations(range(count), 2):
        displacement = points[right] - points[left]
        displacement -= box * np.round(displacement / box)
        if np.linalg.norm(displacement) < tolerance:
            union(left, right)

    representatives = np.array([find(index) for index in range(count)])
    unique_representatives = np.unique(representatives)
    representative_to_new = {
        representative: index
        for index, representative in enumerate(unique_representatives)
    }
    old_to_new = np.array(
        [representative_to_new[representative] for representative in representatives]
    )

    remapped = old_to_new[edges]
    remapped = remapped[remapped[:, 0] != remapped[:, 1]]
    remapped = np.array(
        list(set(map(tuple, np.sort(remapped, axis=1)))), dtype=int
    )
    return points[unique_representatives], remapped


def periodic_delaunay(
    points: np.ndarray,
    box_dimensions: np.ndarray,
    *,
    buffer_distance: float = 3.0,
    merge_tolerance: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a periodic Delaunay graph.

    Returns the wrapped node coordinates and an ``(M, 3)`` array containing
    the two integer endpoint indices and the minimum-image edge length.
    """
    points = np.asarray(points, dtype=float)
    box = np.asarray(box_dimensions, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    if box.shape != (points.shape[1],) or np.any(box <= 0.0):
        raise ValueError("box_dimensions must contain one positive length per axis")

    tiled_points = _periodic_replicas(points, box, buffer_distance)
    tiled_edges = _delaunay_edges(tiled_points)

    retained = set(
        np.flatnonzero(np.all((tiled_points >= 0.0) & (tiled_points <= box), axis=1))
    )
    for left, right in tiled_edges:
        if _segment_meets_box(tiled_points[left], tiled_points[right], box):
            retained.update((int(left), int(right)))

    retained_indices = np.array(sorted(retained), dtype=int)
    tiled_to_retained = {
        old_index: new_index
        for new_index, old_index in enumerate(retained_indices)
    }
    retained_edges = np.array(
        [
            (tiled_to_retained[left], tiled_to_retained[right])
            for left, right in tiled_edges
            if left in tiled_to_retained and right in tiled_to_retained
        ],
        dtype=int,
    )
    wrapped_points = np.mod(tiled_points[retained_indices], box)
    merged_points, merged_edges = _merge_periodic_copies(
        wrapped_points, retained_edges, box, merge_tolerance
    )

    displacement = merged_points[merged_edges[:, 1]] - merged_points[merged_edges[:, 0]]
    displacement -= box * np.round(displacement / box)
    lengths = np.linalg.norm(displacement, axis=1)
    return merged_points, np.column_stack((merged_edges, lengths))


class GraphGenerator:
    """Compatibility wrapper for the publication's network factory."""

    def __init__(self, points: np.ndarray, box_dimensions: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self.box_dimensions = np.asarray(box_dimensions, dtype=float)

    def periodic_delaunay_tessellation(self) -> tuple[np.ndarray, np.ndarray]:
        return periodic_delaunay(self.points, self.box_dimensions)
