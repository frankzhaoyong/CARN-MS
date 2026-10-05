#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAD-Aware Rule-Gated and Network-Guided Mesh Simplification (CARN-MS)
=====================================================================
A hybrid CAD mesh decimation algorithm that decouples:
1. Deterministic Geometry Kernel (Area-weighted Quadrics + Analytic Projections)
2. Topology Gatekeeper (Link Condition + 1-Ring Normal Inversion Barrier)
3. Neural Priority Engine (EdgeDecisionNet: Local proxy-damage prediction)

Authors: Antigravity Pair Programmer
"""

from __future__ import annotations

import argparse
import copy
import heapq
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

# Global constants & Experiment Configurations
CACHE_SCHEMA_VERSION = "c5_d_v3_consistent"
METRIC_SCHEMA_VERSION = "c5_d_arc_midpoint_v3"
METHOD_NAME = "CAD-Aware Rule-Gated and Network-Guided Mesh Simplification"
METHOD_ACRONYM = "CARN-MS"


def file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "trimesh": trimesh.__version__, "networkx": nx.__version__}


def experiment_fingerprint(paths, config) -> str:
    payload = {"files": {str(Path(p).resolve()): file_digest(p) for p in paths},
               "config": config, "versions": environment_versions()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


EPSILON = 1e-12
DIHEDRAL_FEATURE_THRESH_DEG = 30.0
NORMAL_INVERSION_MIN_DOT = 0.50  # cos(60 deg) strict barrier against triangle folding
FEATURE_NAMES_V10 = (
    "log_edge_len",
    "normal_dot",
    "dihedral_max",
    "feature_proximity",
    "axial_alignment",
    "curvature_jump",
    "ligament_factor",
    "local_sliver_ratio",
    "area_density",
    "norm_qem_cost",
)

# Standardized Sample Count and Seeds Configuration
DEFAULT_METRIC_SAMPLES = 5000
DEFAULT_METRIC_SEED_COUNT = 3
DEFAULT_BASE_SEED = 42
DEFAULT_CPU_FRACTION = 0.70  # Control CPU resource usage to 70%


def configure_resource_limits(cpu_fraction: float = DEFAULT_CPU_FRACTION) -> int:
    """Cap PyTorch CPU threads and parallel math libraries to 70% of available CPU cores."""
    total_cpus = os.cpu_count() or 1
    target_threads = max(1, int(total_cpus * float(cpu_fraction)))
    torch.set_num_threads(target_threads)
    os.environ["OMP_NUM_THREADS"] = str(target_threads)
    os.environ["MKL_NUM_THREADS"] = str(target_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(target_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(target_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(target_threads)
    return target_threads


def get_train_device() -> str:
    """Fixed device for training: MPS (Apple Silicon GPU) if available, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_infer_device() -> str:
    """Fixed device for inference (simplification queue): CPU."""
    return "cpu"


def get_eval_device() -> str:
    """Fixed device for metric evaluation: CPU."""
    return "cpu"



# ============================================================================
# 1. Deterministic Geometry Kernel & Metric Utilities
# ============================================================================

def load_triangle_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load one triangle mesh, concatenating scenes if needed."""
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(loaded)!r}: {path}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Empty mesh: {path}")
    return loaded


def normalize_mesh_pair(original: trimesh.Trimesh, simplified: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Normalize mesh pair using original bounding box max extent."""
    verts = np.asarray(original.vertices, dtype=np.float64)
    bbox_center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    max_extent = float((verts.max(axis=0) - verts.min(axis=0)).max())
    if max_extent <= EPSILON:
        raise ValueError("Mesh has near-zero bounding-box extent.")
    scale = 1.0 / max_extent
    orig_norm = original.copy()
    simp_norm = simplified.copy()
    orig_norm.vertices = (np.asarray(orig_norm.vertices, dtype=np.float64) - bbox_center) * scale
    simp_norm.vertices = (np.asarray(simp_norm.vertices, dtype=np.float64) - bbox_center) * scale
    return orig_norm, simp_norm


def extract_feature_segments(mesh: trimesh.Trimesh, dihedral_thresh: float = DIHEDRAL_FEATURE_THRESH_DEG) -> np.ndarray:
    """Extract high-dihedral feature segments and boundary segments from a mesh."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(lengths, EPSILON)

    edge_faces: Dict[Tuple[int, int], List[int]] = {}
    for face_id, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_faces.setdefault(edge, []).append(face_id)

    feature_edges = []
    if not 0.0 < float(dihedral_thresh) < 180.0:
        raise ValueError("dihedral_thresh is in degrees and must be in (0, 180)")
    thresh_rad = np.deg2rad(float(dihedral_thresh))
    cos_thresh = np.cos(thresh_rad)
    for edge, incident in edge_faces.items():
        if len(incident) == 1:
            # Boundary edges are essential geometric boundary features in CAD
            feature_edges.append(edge)
        elif len(incident) == 2:
            n0 = face_normals[incident[0]]
            n1 = face_normals[incident[1]]
            dot = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
            if dot < cos_thresh:  # Angle > thresh_rad
                feature_edges.append(edge)

    if not feature_edges:
        return np.empty((0, 2, 3), dtype=np.float64)
    return vertices[np.asarray(feature_edges, dtype=np.int64)]


def point_to_segment_set_distance(points: np.ndarray, segments: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Exact point-to-segment distance query with bounded memory chunking."""
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    segments = np.asarray(segments, dtype=np.float64).reshape((-1, 2, 3))
    if not len(points):
        return np.empty(0, dtype=np.float64)
    if not len(segments):
        return np.full(len(points), float("inf"), dtype=np.float64)

    a = segments[:, 0, :]
    b = segments[:, 1, :]
    ab = b - a
    ab_len_sq = np.sum(ab ** 2, axis=1)
    ab_len_sq = np.maximum(ab_len_sq, EPSILON)

    min_distances = np.full(len(points), float("inf"), dtype=np.float64)
    for i in range(0, len(points), chunk_size):
        pts = points[i : i + chunk_size]
        ap = pts[:, None, :] - a[None, :, :]
        t = np.sum(ap * ab[None, :, :], axis=2) / ab_len_sq[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        dists_sq = np.sum((pts[:, None, :] - closest) ** 2, axis=2)
        min_distances[i : i + chunk_size] = np.sqrt(np.min(dists_sq, axis=1))

    return min_distances


def evaluate_terminal_metrics(
    original: trimesh.Trimesh,
    simplified: trimesh.Trimesh,
    num_samples: int = DEFAULT_METRIC_SAMPLES,
    base_seed: int = DEFAULT_BASE_SEED,
    seed_count: int = DEFAULT_METRIC_SEED_COUNT,
    dihedral_thresh: float = DIHEDRAL_FEATURE_THRESH_DEG,
    feature_samples: int = 8,
) -> Dict[str, Any]:
    """
    Standardized terminal metrics evaluation:
    - 5000 surface points x 3 seeds (seeds 42, 43, 44 by default)
    - All distance metrics normalized by original BBox diagonal
    - Dimensionless Curvature Error (CE) via unit normal consistency
    - Arc-length weighted Symmetric Feature Dihedral Preservation Error (Symmetric FDPE_RMS)
    - Max Hausdorff Distance (HD_max, Auxiliary Metric)
    """
    if num_samples <= 0 or seed_count <= 0 or feature_samples <= 0:
        raise ValueError("Sample and seed counts must be positive")
    orig_norm, simp_norm = normalize_mesh_pair(original, simplified)
    bbox_diag = float(np.linalg.norm(orig_norm.bounds[1] - orig_norm.bounds[0]))
    if bbox_diag <= EPSILON:
        bbox_diag = 1.0

    cd_list = []
    hd_max_list = []
    ce_list = []

    for seed_offset in range(max(1, seed_count)):
        seed = base_seed + seed_offset
        p_orig, f_orig = trimesh.sample.sample_surface(orig_norm, num_samples, seed=seed)
        p_simp, f_simp = trimesh.sample.sample_surface(simp_norm, num_samples, seed=seed + 1000)

        tree_simp = scipy.spatial.cKDTree(p_simp)
        tree_orig = scipy.spatial.cKDTree(p_orig)

        dist_to_simp, match_to_simp = tree_simp.query(p_orig)
        dist_to_orig, match_to_orig = tree_orig.query(p_simp)

        # Distance metrics normalized by bbox_diag (CD normalized by bbox_diag^2 for dimensionless squared distance)
        cd_val = float(np.mean(dist_to_simp ** 2) + np.mean(dist_to_orig ** 2)) / (bbox_diag ** 2)
        hd_val = float(max(np.max(dist_to_simp), np.max(dist_to_orig))) / bbox_diag

        # Curvature error (CE) via normal consistency on sampled points (Dimensionless, not divided by bbox_diag)
        fn_orig = orig_norm.face_normals[f_orig]
        fn_simp = simp_norm.face_normals[f_simp]
        fn_matched_simp = fn_simp[match_to_simp]
        fn_matched_orig = fn_orig[match_to_orig]

        dot1 = np.clip(np.sum(fn_orig * fn_matched_simp, axis=1), -1.0, 1.0)
        dot2 = np.clip(np.sum(fn_simp * fn_matched_orig, axis=1), -1.0, 1.0)
        ce_val = float(np.mean(1.0 - dot1) + np.mean(1.0 - dot2))

        cd_list.append(cd_val)
        hd_max_list.append(hd_val)
        ce_list.append(ce_val)

    # Symmetric Feature Dihedral Preservation Error (Symmetric FDPE_RMS) on CAD sharp creases
    ref_segs = extract_feature_segments(orig_norm, dihedral_thresh=dihedral_thresh)
    simp_segs = extract_feature_segments(simp_norm, dihedral_thresh=dihedral_thresh)

    feature_curve_count = len(ref_segs)
    simplified_feature_curve_count = len(simp_segs)

    ref_total_len = float(np.sum(np.linalg.norm(ref_segs[:, 1, :] - ref_segs[:, 0, :], axis=1))) if feature_curve_count > 0 else 0.0
    simp_total_len = float(np.sum(np.linalg.norm(simp_segs[:, 1, :] - simp_segs[:, 0, :], axis=1))) if simplified_feature_curve_count > 0 else 0.0

    if feature_curve_count > 0 and simplified_feature_curve_count > 0:
        k_samples = int(feature_samples)
        # Midpoint sampling rule: (i + 0.5) / k prevents double-counting shared junction vertices
        alpha = (np.arange(k_samples, dtype=np.float64) + 0.5) / float(k_samples)
        ref_pts = ((1.0 - alpha[None, :, None]) * ref_segs[:, :1, :] + alpha[None, :, None] * ref_segs[:, 1:, :]).reshape((-1, 3))
        simp_pts = ((1.0 - alpha[None, :, None]) * simp_segs[:, :1, :] + alpha[None, :, None] * simp_segs[:, 1:, :]).reshape((-1, 3))

        # Arc-length weights: each sample point is weighted by its parent segment length / k_samples
        ref_seg_lens = np.linalg.norm(ref_segs[:, 1, :] - ref_segs[:, 0, :], axis=1)
        simp_seg_lens = np.linalg.norm(simp_segs[:, 1, :] - simp_segs[:, 0, :], axis=1)
        w_ref = np.repeat(ref_seg_lens / k_samples, k_samples)
        w_simp = np.repeat(simp_seg_lens / k_samples, k_samples)

        fwd_dists = point_to_segment_set_distance(ref_pts, simp_segs)
        rev_dists = point_to_segment_set_distance(simp_pts, ref_segs)

        sum_w_ref = float(np.sum(w_ref))
        sum_w_simp = float(np.sum(w_simp))

        fdpe_fwd = float(np.sqrt(np.sum(w_ref * (fwd_dists ** 2)) / max(sum_w_ref, EPSILON)) / bbox_diag)
        fdpe_rev = float(np.sqrt(np.sum(w_simp * (rev_dists ** 2)) / max(sum_w_simp, EPSILON)) / bbox_diag)
        fdpe_rms = float(0.5 * (fdpe_fwd + fdpe_rev))
        has_feature_curves = True
    elif feature_curve_count > 0 and simplified_feature_curve_count == 0:
        fdpe_fwd = 1.0 / bbox_diag
        fdpe_rev = 1.0 / bbox_diag
        fdpe_rms = 1.0 / bbox_diag
        has_feature_curves = True
    elif feature_curve_count == 0 and simplified_feature_curve_count > 0:
        fdpe_fwd = 1.0 / bbox_diag
        fdpe_rev = 1.0 / bbox_diag
        fdpe_rms = 1.0 / bbox_diag
        has_feature_curves = False
    else:
        fdpe_fwd = 0.0
        fdpe_rev = 0.0
        fdpe_rms = 0.0
        has_feature_curves = False

    return {
        "cd": float(np.mean(cd_list)),
        "cd_std": float(np.std(cd_list)) if len(cd_list) > 1 else 0.0,
        "ce": float(np.mean(ce_list)),
        "ce_std": float(np.std(ce_list)) if len(ce_list) > 1 else 0.0,
        "fdpe_rms": float(fdpe_rms),
        "fdpe_rms_std": None,  # deterministic quadrature, not a random replicate
        "feature_status": ("matched" if feature_curve_count and simplified_feature_curve_count
                           else "missing" if feature_curve_count else "spurious" if simplified_feature_curve_count else "not_applicable"),
        "metric_schema": METRIC_SCHEMA_VERSION,
        "fdpe_fwd": float(fdpe_fwd),
        "fdpe_rev": float(fdpe_rev),
        "hd_max": float(np.mean(hd_max_list)),
        "hd_max_std": float(np.std(hd_max_list)) if len(hd_max_list) > 1 else 0.0,
        "has_feature_curves": has_feature_curves,
        "feature_curve_count": feature_curve_count,
        "simplified_feature_curve_count": simplified_feature_curve_count,
        "ref_total_feature_len": ref_total_len,
        "simp_total_feature_len": simp_total_len,
        "bbox_diag": float(bbox_diag),
        "num_samples": num_samples,
        "seed_count": seed_count,
        "seeds": [base_seed + i for i in range(max(1, seed_count))],
    }


# ============================================================================
# 2. Quadric Error Metric (QEM) Kernel with Boundary Protection
# ============================================================================

class QuadricKernel:
    """Area-weighted Quadric representation with boundary/crease protection."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, boundary_weight: float = 50.0):
        self.num_verts = len(vertices)
        self.Q = [np.zeros((4, 4), dtype=np.float64) for _ in range(self.num_verts)]
        self._init_quadrics(vertices, faces, boundary_weight)

    def _init_quadrics(self, vertices: np.ndarray, faces: np.ndarray, boundary_weight: float) -> None:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(cross, axis=1)
        areas = lengths * 0.5
        unit_normals = cross / np.maximum(lengths[:, None], EPSILON)

        # 1. Area-weighted face quadrics
        for f_idx, (a, b, c) in enumerate(faces):
            n = unit_normals[f_idx]
            area = areas[f_idx]
            d = -float(np.dot(n, vertices[a]))
            p = np.array([n[0], n[1], n[2], d], dtype=np.float64).reshape((4, 1))
            Kp = area * np.matmul(p, p.T)
            self.Q[a] += Kp
            self.Q[b] += Kp
            self.Q[c] += Kp

        # 2. Boundary and high-dihedral crease constraint quadrics
        edge_faces: Dict[Tuple[int, int], List[int]] = {}
        for f_idx, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = (min(a, b), max(a, b))
                edge_faces.setdefault(edge, []).append(f_idx)

        cos_30 = np.cos(np.deg2rad(DIHEDRAL_FEATURE_THRESH_DEG))
        for (a, b), inc_faces in edge_faces.items():
            is_crease = False
            if len(inc_faces) == 1:
                is_crease = True
            elif len(inc_faces) == 2:
                n0 = unit_normals[inc_faces[0]]
                n1 = unit_normals[inc_faces[1]]
                if np.dot(n0, n1) < cos_30:
                    is_crease = True

            if is_crease:
                p_a = vertices[a]
                p_b = vertices[b]
                edge_vec = p_b - p_a
                edge_len = float(np.linalg.norm(edge_vec))
                if edge_len <= EPSILON:
                    continue
                edge_dir = edge_vec / edge_len
                n_face = unit_normals[inc_faces[0]]
                # Plane perpendicular to face passing through edge
                n_bound = np.cross(edge_dir, n_face)
                n_bound_len = np.linalg.norm(n_bound)
                if n_bound_len > EPSILON:
                    n_bound /= n_bound_len
                    d_bound = -float(np.dot(n_bound, p_a))
                    p_bnd = np.array([n_bound[0], n_bound[1], n_bound[2], d_bound], dtype=np.float64).reshape((4, 1))
                    K_bnd = (boundary_weight * edge_len) * np.matmul(p_bnd, p_bnd.T)
                    self.Q[a] += K_bnd
                    self.Q[b] += K_bnd

    def solve_optimal_position(self, u: int, v: int, pos_u: np.ndarray, pos_v: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve for optimal contraction position v_opt and associated quadric cost."""
        Q_sum = self.Q[u] + self.Q[v]
        A = Q_sum[:3, :3]
        b = -Q_sum[:3, 3]

        # Check conditioning of 3x3 quadric matrix
        solved = False
        v_opt = np.zeros(3, dtype=np.float64)
        det = float(np.linalg.det(A))
        if np.isfinite(det) and det > 1e-10:
            cond = float(np.linalg.cond(A))
            if np.isfinite(cond) and cond < 1e10:
                try:
                    v_opt = np.linalg.solve(A, b)
                    # Plausibility check: optimal position should stay within reasonable neighborhood
                    mid = 0.5 * (pos_u + pos_v)
                    edge_len = float(np.linalg.norm(pos_v - pos_u))
                    if np.linalg.norm(v_opt - mid) <= 3.0 * max(edge_len, 1e-6):
                        solved = True
                except np.linalg.LinAlgError:
                    solved = False

        if not solved:
            # Fallback evaluation among u, v, and midpoint
            candidates = [pos_u, pos_v, 0.5 * (pos_u + pos_v)]
            best_c = float("inf")
            best_pos = pos_u
            for cand in candidates:
                p_homo = np.append(cand, 1.0).reshape((4, 1))
                cost = float(np.matmul(np.matmul(p_homo.T, Q_sum), p_homo).item())
                if cost < best_c:
                    best_c = cost
                    best_pos = cand
            v_opt = best_pos
            cost_opt = max(best_c, 0.0)
        else:
            p_homo = np.append(v_opt, 1.0).reshape((4, 1))
            cost_opt = max(float(np.matmul(np.matmul(p_homo.T, Q_sum), p_homo).item()), 0.0)

        return v_opt, cost_opt


# ============================================================================
# 3. Topology & Normal Inversion Gatekeeper
# ============================================================================

def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product for the gate's float64 3-vectors, without axis dispatch."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], dtype=np.float64)


class TopologyGatekeeper:
    """Strictly enforces 2-manifold Link Condition and 1-ring normal anti-inversion."""

    @staticmethod
    def check_link_condition(
        u: int, v: int,
        neighbors_u: Set[int],
        neighbors_v: Set[int],
        shared_faces_uv: List[int],
        faces: np.ndarray,
        boundary_vertices: Optional[Set[int]] = None,
        incident_faces_u: Optional[Set[int]] = None,
        incident_faces_v: Optional[Set[int]] = None,
    ) -> bool:
        r"""Link condition for manifold mesh: Link(u) \cap Link(v) == shared 1-ring vertices."""
        # 1. Non-manifold edge guard: Edge must have exactly 1 (boundary) or 2 (interior) shared faces
        num_shared = len(shared_faces_uv)
        if num_shared < 1 or num_shared > 2:
            return False

        # 2. Component annihilation guard: Collapsing must not annihilate all faces of an isolated component
        if incident_faces_u is not None and incident_faces_v is not None:
            surviving_faces = (incident_faces_u | incident_faces_v) - set(shared_faces_uv)
            if not surviving_faces:
                return False

        # 3. Boundary pinch barrier: If both u and v are on the boundary, edge (u, v) must be a boundary edge (len(shared) == 1).
        # Collapsing an interior edge connecting two boundary points collapses and pinches the boundary loop.
        if boundary_vertices is not None:
            if (u in boundary_vertices) and (v in boundary_vertices):
                if num_shared > 1:
                    return False

        common_neighbors = neighbors_u.intersection(neighbors_v)
        # In a valid edge collapse, common neighbors must exactly equal vertices opposite to edge (u, v) in shared faces
        expected_common = set()
        for f_idx in shared_faces_uv:
            face = faces[f_idx]
            for vert in face:
                if vert != u and vert != v:
                    expected_common.add(int(vert))
        if common_neighbors != expected_common:
            return False
        if incident_faces_u is None or incident_faces_v is None:
            incident_faces_u = set(np.flatnonzero(np.any(faces == u, axis=1)))
            incident_faces_v = set(np.flatnonzero(np.any(faces == v, axis=1)))
        # Vertex links contain edges too; the vertex-only test admits a tetrahedron collapse.
        link_u = {tuple(sorted(int(w) for w in faces[f] if w != u)) for f in incident_faces_u}
        link_v = {tuple(sorted(int(w) for w in faces[f] if w != v)) for f in incident_faces_v}
        if link_u & link_v:
            return False
        surviving = (incident_faces_u | incident_faces_v) - set(shared_faces_uv)
        signatures = [tuple(sorted(u if int(w) == v else int(w) for w in faces[f])) for f in surviving]
        return bool(signatures) and len(signatures) == len(set(signatures))

    @staticmethod
    def check_normal_inversion(
        u: int, v: int, v_opt: np.ndarray,
        incident_faces_u: Set[int],
        incident_faces_v: Set[int],
        shared_faces_uv: Set[int],
        vertices: np.ndarray,
        faces: np.ndarray,
        min_cos_angle: float = NORMAL_INVERSION_MIN_DOT,
    ) -> bool:
        """
        Check that collapsing (u, v) -> v_opt does not flip or degenerate any surviving 1-ring triangle.
        This is a local barrier; it does not test global self-intersections.
        """
        surviving_faces = (incident_faces_u | incident_faces_v) - shared_faces_uv
        for f_idx in surviving_faces:
            face = faces[f_idx]
            # Old normal
            p0 = vertices[face[0]]
            p1 = vertices[face[1]]
            p2 = vertices[face[2]]
            old_cross = _cross3(p1 - p0, p2 - p0)
            old_len = np.linalg.norm(old_cross)
            if old_len <= EPSILON:
                continue
            old_normal = old_cross / old_len

            # New triangle points with u/v replaced by v_opt
            np0 = v_opt if (face[0] == u or face[0] == v) else p0
            np1 = v_opt if (face[1] == u or face[1] == v) else p1
            np2 = v_opt if (face[2] == u or face[2] == v) else p2

            new_cross = _cross3(np1 - np0, np2 - np0)
            new_len = np.linalg.norm(new_cross)
            if new_len <= EPSILON:
                return False  # Degenerates to a line or point
            new_normal = new_cross / new_len

            # Anti-flip threshold (strict barrier against fold-over into holes)
            if np.dot(old_normal, new_normal) < min_cos_angle:
                return False

        return True


# ============================================================================
# 4. 10D Invariant CAD Edge Feature Extractor
# ============================================================================

class CADEdgeFeatureExtractor:
    """Extracts 10D scale- and SE(3)-invariant geometric features for each edge."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
        self.faces = np.array(mesh.faces, dtype=np.int64, copy=True)
        self.bbox_diag = max(float(np.linalg.norm(np.ptp(self.vertices, axis=0))), EPSILON)

        # Precompute face areas and normals
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(cross, axis=1)
        self.face_areas = np.array(lengths * 0.5, dtype=np.float64, copy=True)
        self.face_normals = np.array(cross / np.maximum(lengths[:, None], EPSILON), dtype=np.float64, copy=True)

        # Vertex normals computed by area-weighted averaging across faces (consistent with update_local_geometry)
        self.vertex_normals = np.zeros((len(self.vertices), 3), dtype=np.float64)
        for f_idx, (a, b, c) in enumerate(self.faces):
            fn = self.face_normals[f_idx] * self.face_areas[f_idx]
            self.vertex_normals[a] += fn
            self.vertex_normals[b] += fn
            self.vertex_normals[c] += fn
        v_lengths = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
        self.vertex_normals = np.where(v_lengths > EPSILON, self.vertex_normals / np.maximum(v_lengths, EPSILON), np.array([0.0, 0.0, 1.0]))

        self.median_edge_len = max(float(np.median(mesh.edges_unique_length)), 1e-6)
        self.mean_face_area = max(float(np.mean(self.face_areas)), 1e-8)

        self.G_sharp = nx.Graph()
        edge_faces = {}
        for f, face in enumerate(self.faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_faces.setdefault(tuple(sorted((int(a), int(b)))), set()).add(f)
        self.vertex_to_loop = {}
        self._refresh_sharp_edges(edge_faces, set(edge_faces))

    def _refresh_sharp_edges(self, edge_faces, edges):
        cos_thresh = np.cos(np.deg2rad(DIHEDRAL_FEATURE_THRESH_DEG))
        for edge in sorted(edges):
            incident = sorted(edge_faces.get(edge, ()))
            sharp = len(incident) == 1 or (len(incident) == 2 and
                np.dot(self.face_normals[incident[0]], self.face_normals[incident[1]]) < cos_thresh)
            if sharp:
                self.G_sharp.add_edge(*edge)
            elif self.G_sharp.has_edge(*edge):
                self.G_sharp.remove_edge(*edge)
        self.G_sharp.remove_nodes_from(list(nx.isolates(self.G_sharp)))
        self.sharp_vertices = set(self.G_sharp.nodes())
        self.connected_feature_loops = sorted(
            list(nx.connected_components(self.G_sharp)),
            key=lambda c: min(c) if c else -1
        )
        self.vertex_to_loop = {
            int(v): l_idx
            for l_idx, loop in enumerate(self.connected_feature_loops)
            for v in loop
        }

    def update_local_geometry(
        self,
        affected_faces: Set[int],
        affected_verts: Set[int],
        vertex_faces: List[Set[int]],
        edge_shared_faces: Optional[Dict[Tuple[int, int], Set[int]]] = None,
        v_merged: Optional[int] = None,
        u_surviving: Optional[int] = None,
    ) -> Tuple[bool, Set[int]]:
        """Dynamically update face areas, face normals, vertex normals, and sharp feature graph for affected 1-ring entities."""
        for f_idx in affected_faces:
            if f_idx >= len(self.faces):
                continue
            fa = self.faces[f_idx]
            p0 = self.vertices[fa[0]]
            p1 = self.vertices[fa[1]]
            p2 = self.vertices[fa[2]]
            cross = np.cross(p1 - p0, p2 - p0)
            length = np.linalg.norm(cross)
            self.face_areas[f_idx] = length * 0.5
            if length > EPSILON:
                self.face_normals[f_idx] = cross / length
            else:
                self.face_normals[f_idx] = np.array([0.0, 0.0, 1.0])

        for v_idx in affected_verts:
            if v_idx >= len(vertex_faces):
                continue
            v_fs = sorted(vertex_faces[v_idx])
            if not v_fs:
                continue
            # Area-weighted vertex normal
            weighted_norm = np.zeros(3, dtype=np.float64)
            for f_idx in v_fs:
                if f_idx < len(self.face_normals):
                    weighted_norm += self.face_normals[f_idx] * self.face_areas[f_idx]
            norm_len = np.linalg.norm(weighted_norm)
            self.vertex_normals[v_idx] = weighted_norm / norm_len if norm_len > EPSILON else [0.0, 0.0, 1.0]

        # Graph splits/merges can change ligament scores far outside the geometric one-ring.
        before = {frozenset(c - {v_merged}) for c in self.connected_feature_loops if c - {v_merged}}
        if edge_shared_faces is not None:
            dirty = {e for e in self.G_sharp.edges() if set(e) & affected_verts}
            dirty = {tuple(sorted(e)) for e in dirty}
            for w in affected_verts:
                for f in vertex_faces[w]:
                    fa = self.faces[f]
                    dirty.update(tuple(sorted((int(a), int(b)))) for a, b in
                                 ((fa[0], fa[1]), (fa[1], fa[2]), (fa[2], fa[0])))
            self._refresh_sharp_edges(edge_shared_faces, dirty)
        after = {frozenset(c) for c in self.connected_feature_loops}
        graph_changed = (before != after)
        changed_verts = set()
        if graph_changed:
            len_before = len(before)
            len_after = len(after)
            if (len_before >= 2 and len_after < 2) or (len_before < 2 and len_after >= 2):
                changed_verts = set(self.sharp_vertices)
            else:
                for c in (before ^ after):
                    changed_verts.update(c)
        return graph_changed, changed_verts

    def extract_features(
        self,
        u: int,
        v: int,
        edge_len: float,
        qem_cost: float,
        incident_faces_u: Set[int],
        incident_faces_v: Set[int],
        shared_faces: Set[int],
    ) -> np.ndarray:
        """Compute the 10-dimensional invariant feature vector for edge (u, v)."""
        feat = np.zeros(10, dtype=np.float32)

        # 0. log_edge_len (relative to median scale)
        feat[0] = np.log(max(edge_len / self.median_edge_len, 1e-6))

        # 1. normal_dot
        nu = self.vertex_normals[u]
        nv = self.vertex_normals[v]
        feat[1] = float(np.clip(np.dot(nu, nv), -1.0, 1.0))

        # 2. dihedral_max across edge
        if len(shared_faces) >= 2:
            s_list = list(shared_faces)[:2]
            n0 = self.face_normals[s_list[0]]
            n1 = self.face_normals[s_list[1]]
            feat[2] = float(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0)) / np.pi)
        else:
            feat[2] = 0.0

        # 3. feature_proximity
        is_u_feat = u in self.sharp_vertices
        is_v_feat = v in self.sharp_vertices
        if is_u_feat or is_v_feat:
            feat[3] = 1.0
        else:
            feat[3] = 0.0

        # 4. axial_alignment (estimated cylinder generatrix alignment)
        edge_dir = (self.vertices[v] - self.vertices[u]) / max(edge_len, EPSILON)
        # Approximate tangent of minimum curvature using cross of vertex normals
        cross_n = np.cross(nu, nv)
        cross_len = np.linalg.norm(cross_n)
        if cross_len > 1e-4:
            axis_dir = cross_n / cross_len
            feat[4] = float(abs(np.dot(edge_dir, axis_dir)))
        else:
            feat[4] = 0.0

        # 5. curvature_jump
        feat[5] = float(abs(feat[1] - 1.0))

        # 6. ligament_factor (cross-hole narrow bridge indicator)
        if len(self.connected_feature_loops) >= 2:
            u_loop = self.vertex_to_loop.get(u, -1)
            v_loop = self.vertex_to_loop.get(v, -1)
            feat[6] = 1.0 if (u_loop >= 0 and v_loop >= 0 and u_loop != v_loop) else 0.0
        else:
            feat[6] = 0.0

        # 7. local_sliver_ratio
        all_faces = incident_faces_u | incident_faces_v
        max_aspect = 1.0
        for f_idx in all_faces:
            fa = self.faces[f_idx]
            pa, pb, pc = self.vertices[fa[0]], self.vertices[fa[1]], self.vertices[fa[2]]
            ea = np.linalg.norm(pb - pa)
            eb = np.linalg.norm(pc - pb)
            ec = np.linalg.norm(pa - pc)
            perim = ea + eb + ec
            area = self.face_areas[f_idx]
            asp = (max(ea, eb, ec) * perim) / max(4.0 * np.sqrt(3.0) * area, 1e-12)
            max_aspect = max(max_aspect, asp)
        feat[7] = float(np.clip(np.log1p(max_aspect), 0.0, 10.0))

        # 8. area_density
        area_sum = sum(self.face_areas[f_idx] for f_idx in all_faces)
        feat[8] = float(np.log(max(area_sum / self.mean_face_area, 1e-4)))

        # 9. norm_qem_cost
        feat[9] = float(np.log1p(qem_cost / max(self.median_edge_len ** 2, 1e-8)))

        return feat


# ============================================================================
# 5. Neural Priority Decision Brain (EdgeDecisionNet)
# ============================================================================

class EdgeDecisionNet(nn.Module):
    """
    Lightweight, high-throughput Multi-Head CAD Damage Predictor.
    Predicts normalized damage for CD, CE, and FDPE under edge collapse.
    """

    def __init__(self, in_dim: int = 10, hidden: int = 128, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            for _ in range(num_layers)
        ])

        # Three dedicated heads for terminal metrics
        self.head_cd = nn.Linear(hidden, 1)
        self.head_ce = nn.Linear(hidden, 1)
        self.head_fdpe = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (pred_cd, pred_ce, pred_fdpe, composite_risk)
        """
        h = self.input_layer(x)
        for block in self.blocks:
            h = h + block(h)  # Residual connection

        pred_cd = torch.sigmoid(self.head_cd(h))
        pred_ce = torch.sigmoid(self.head_ce(h))
        pred_fdpe = torch.sigmoid(self.head_fdpe(h))

        # Composite learned risk: heavy on CAD feature preservation
        composite_risk = 0.20 * pred_cd + 0.40 * pred_ce + 0.40 * pred_fdpe
        return pred_cd, pred_ce, pred_fdpe, composite_risk


# ============================================================================
# 6. Dynamic Priority Queue Decimator
# ============================================================================

def validate_triangle_mesh(vertices, faces):
    """Reject unsupported input; do not silently repair or change the benchmark mesh."""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite 3D coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh must contain valid triangle indices")
    canonical = np.sort(faces, axis=1)
    if np.any(np.diff(canonical, axis=1) == 0) or len(np.unique(canonical, axis=0)) != len(faces):
        raise ValueError("Repeated vertex or duplicate triangle")
    span = np.ptp(vertices, axis=0).max()
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Mesh has zero extent")
    p = ((vertices - vertices.min(axis=0)) / span)[faces]
    if np.any(np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1) <= EPSILON):
        raise ValueError("Mesh contains degenerate triangles")
    links = [dict() for _ in vertices]
    edges = {}
    for a, b, c in faces:
        for v, x, y in ((a,b,c), (b,c,a), (c,a,b)):
            links[v].setdefault(int(x), set()).add(int(y))
            links[v].setdefault(int(y), set()).add(int(x))
            key = tuple(sorted((int(v), int(x))))
            edges.setdefault(key, []).append(v < x)
    if any(len(inc) > 2 for inc in edges.values()):
        raise ValueError("Non-manifold edge")
    if any(len(inc) == 2 and inc[0] == inc[1] for inc in edges.values()):
        raise ValueError("Inconsistent face winding")
    for link in links:
        if not link:
            raise ValueError("Unreferenced vertex")
        seen, stack = set(), [next(iter(link))]
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(link[v] - seen)
        degrees = [len(n) for n in link.values()]
        if len(seen) != len(link) or any(d not in (1, 2) for d in degrees) or degrees.count(1) not in (0, 2):
            raise ValueError("Non-manifold vertex link")


class NeuroDecimator:
    """
    Decimation engine driven by Neural Priority and Gatekeeper.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        net: Optional[EdgeDecisionNet] = None,
        device: str = "cpu",
        use_gatekeeper: bool = True,
    ):
        validate_triangle_mesh(mesh.vertices, mesh.faces)
        self.mesh = mesh.copy()
        raw_verts = np.array(self.mesh.vertices, dtype=np.float64, copy=True)
        self.bbox_center = (raw_verts.min(axis=0) + raw_verts.max(axis=0)) / 2.0
        self.max_extent = float((raw_verts.max(axis=0) - raw_verts.min(axis=0)).max())
        self.scale = 1.0 / self.max_extent

        # Canonical normalized coordinate space [-0.5, 0.5]^3
        self.vertices = (raw_verts - self.bbox_center) * self.scale
        self.faces = np.array(self.mesh.faces, dtype=np.int64, copy=True)
        self.net = net
        self.device = torch.device(device)
        if self.net is not None:
            self.net.to(self.device).eval()
        self.use_gatekeeper = use_gatekeeper

        # Kernels
        self.quadrics = QuadricKernel(self.vertices, self.faces)
        self.gatekeeper = TopologyGatekeeper()

        # Build dynamic adjacency
        self._build_topology_structures()

        # Feature extractor operating in canonical normalized space
        norm_mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        self.feature_extractor = CADEdgeFeatureExtractor(norm_mesh)
        self.feature_extractor.vertices = self.vertices  # Share mutable vertex buffer
        self.feature_extractor.faces = self.faces        # Share mutable face buffer!

        # State tracking for continuous advance
        # Order ties by canonical edge ID; version only invalidates old entries.
        self.heap: Optional[List[Tuple[float, int, int, int, np.ndarray]]] = None
        self.edge_version: Dict[Tuple[int, int], int] = {}
        self.collapsed_count = 0
        self.target_reached = False

    def _build_topology_structures(self) -> None:
        self.num_verts = len(self.vertices)
        self.vertex_neighbors: List[Set[int]] = [set() for _ in range(self.num_verts)]
        self.vertex_faces: List[Set[int]] = [set() for _ in range(self.num_verts)]
        self.edge_shared_faces: Dict[Tuple[int, int], Set[int]] = {}
        self.active_faces: Set[int] = set(range(len(self.faces)))
        self.active_vertices: Set[int] = set(range(self.num_verts))

        for f_idx, (a, b, c) in enumerate(self.faces):
            self.vertex_faces[a].add(f_idx)
            self.vertex_faces[b].add(f_idx)
            self.vertex_faces[c].add(f_idx)

            for u, v in ((a, b), (b, c), (c, a)):
                self.vertex_neighbors[u].add(v)
                self.vertex_neighbors[v].add(u)
                edge_key = (min(u, v), max(u, v))
                self.edge_shared_faces.setdefault(edge_key, set()).add(f_idx)

        # Identify boundary vertices
        self.boundary_vertices: Set[int] = set()
        for edge_key, f_set in self.edge_shared_faces.items():
            if len(f_set) == 1:
                self.boundary_vertices.add(edge_key[0])
                self.boundary_vertices.add(edge_key[1])

    def _evaluate_candidate_edges_batched(
        self, edges: List[Tuple[int, int]], batch_size: int = 4096
    ) -> List[Tuple[int, int, float, np.ndarray, float]]:
        """
        Evaluate multiple candidate edges in vectorized batches.
        Returns list of (u, v, priority, v_opt, qem_cost) for valid edges.
        """
        results = []
        if not edges:
            return results

        # 1. Pre-filter by shared faces and topological gatekeeper (fast early rejection)
        legal_candidates = []
        for u, v in edges:
            edge_key = (min(u, v), max(u, v))
            shared = self.edge_shared_faces.get(edge_key, set())
            if len(shared) == 0:
                continue

            # 1a. Topological link condition check (short-circuit rejection before solving quadrics)
            if self.use_gatekeeper:
                valid_link = self.gatekeeper.check_link_condition(
                    u, v,
                    self.vertex_neighbors[u],
                    self.vertex_neighbors[v],
                    list(shared),
                    self.faces,
                    boundary_vertices=self.boundary_vertices,
                    incident_faces_u=self.vertex_faces[u],
                    incident_faces_v=self.vertex_faces[v],
                )
                if not valid_link:
                    continue

            # 1b. Geometric optimal position via quadrics
            pos_u = self.vertices[u]
            pos_v = self.vertices[v]
            v_opt, qem_cost = self.quadrics.solve_optimal_position(u, v, pos_u, pos_v)
            if not np.all(np.isfinite(v_opt)):
                continue

            # 1c. 1-ring normal inversion gatekeeper
            if self.use_gatekeeper:
                valid_normals = self.gatekeeper.check_normal_inversion(
                    u, v, v_opt,
                    self.vertex_faces[u],
                    self.vertex_faces[v],
                    shared,
                    self.vertices,
                    self.faces,
                )
                if not valid_normals:
                    continue

            legal_candidates.append((u, v, v_opt, qem_cost, shared, pos_u, pos_v))

        if not legal_candidates:
            return results

        # 4. Neural Priority Scoring vs Pure QEM Priority
        if self.net is None:
            for u, v, v_opt, qem_cost, _, _, _ in legal_candidates:
                results.append((u, v, float(qem_cost), v_opt, float(qem_cost)))
            return results

        # Process neural evaluations in chunks of batch_size
        for i in range(0, len(legal_candidates), batch_size):
            chunk = legal_candidates[i : i + batch_size]
            chunk_feats = []
            chunk_qem_costs = []
            for u, v, v_opt, qem_cost, shared, pos_u, pos_v in chunk:
                edge_len = float(np.linalg.norm(pos_v - pos_u))
                feat = self.feature_extractor.extract_features(
                    u, v, edge_len, qem_cost,
                    self.vertex_faces[u],
                    self.vertex_faces[v],
                    shared,
                )
                chunk_feats.append(feat)
                chunk_qem_costs.append(qem_cost)

            feat_tensor = torch.from_numpy(np.asarray(chunk_feats, dtype=np.float32)).to(self.device)
            with torch.no_grad():
                _, _, _, comp_risk = self.net(feat_tensor)
                neural_risks = np.atleast_1d(comp_risk.squeeze(-1).cpu().numpy())

            q_arr = np.asarray(chunk_qem_costs, dtype=np.float64)
            alphas = np.clip(1.0 - (q_arr / 1e-4), 0.3, 1.0)
            norm_qems = np.clip(np.log1p(q_arr * 1e4) / 10.0, 0.0, 1.0)
            priorities = alphas * neural_risks + (1.0 - alphas) * norm_qems

            for (u, v, v_opt, qem_cost, _, _, _), p in zip(chunk, priorities):
                results.append((u, v, float(p), v_opt, float(qem_cost)))

        return results

    def _evaluate_candidate_edge(self, u: int, v: int) -> Optional[Tuple[float, np.ndarray, float]]:
        """Evaluate edge validity, optimal position, and neural priority score."""
        res = self._evaluate_candidate_edges_batched([(u, v)])
        if not res:
            return None
        _, _, priority, v_opt, qem_cost = res[0]
        return priority, v_opt, qem_cost

    def _init_heap(self) -> None:
        """Initialize the candidate edge priority heap."""
        self.heap = []
        self.edge_version = {}
        all_edges = set()
        for u in range(self.num_verts):
            for v in self.vertex_neighbors[u]:
                if u < v:
                    all_edges.add((u, v))

        sorted_edges = sorted(all_edges)
        for u, v in sorted_edges:
            self.edge_version[(u, v)] = 0

        eval_results = self._evaluate_candidate_edges_batched(sorted_edges)
        for u, v, priority, v_opt, _ in eval_results:
            heapq.heappush(self.heap, (priority, u, v, 0, v_opt))

    def advance_to_vertex_count(self, target_v: int, verbose: bool = True) -> trimesh.Trimesh:
        """Advance decimation continuously until vertex count reaches target_v."""
        if not isinstance(target_v, (int, np.integer)) or target_v < 1 or target_v > self.num_verts:
            raise ValueError("target_v must be an integer in [1, original vertex count]")
        target_v = max(min(4, self.num_verts), target_v)
        current_v = len(self.active_vertices)
        if current_v < target_v:
            raise ValueError("Cannot increase the vertex budget of an already simplified state")
        if current_v == target_v:
            self.target_reached = True
            return self._rebuild_mesh()

        if self.heap is None:
            self._init_heap()

        if verbose:
            print(f"[{METHOD_ACRONYM}] Target V: {target_v} | Current V: {current_v}")

        start_time = time.time()
        collapsed_this_round = 0

        while current_v > target_v and self.heap:
            priority, u, v, ver, v_opt = heapq.heappop(self.heap)

            # Check if vertices are still active
            if u not in self.active_vertices or v not in self.active_vertices:
                continue
            if ver != self.edge_version.get((min(u, v), max(u, v)), -1):
                continue

            # Re-verify gatekeeper at execution time
            edge_key = (min(u, v), max(u, v))
            shared = self.edge_shared_faces.get(edge_key, set())
            if not shared:
                continue

            if self.use_gatekeeper:
                valid_link = self.gatekeeper.check_link_condition(
                    u, v,
                    self.vertex_neighbors[u],
                    self.vertex_neighbors[v],
                    list(shared),
                    self.faces,
                    boundary_vertices=self.boundary_vertices,
                    incident_faces_u=self.vertex_faces[u],
                    incident_faces_v=self.vertex_faces[v],
                )
                if not valid_link:
                    continue

                valid_normals = self.gatekeeper.check_normal_inversion(
                    u, v, v_opt,
                    self.vertex_faces[u],
                    self.vertex_faces[v],
                    shared,
                    self.vertices,
                    self.faces,
                )
                if not valid_normals:
                    continue

            # Entities affected by this collapse
            affected_faces = (self.vertex_faces[u] | self.vertex_faces[v]) - shared
            affected_verts = {u, v}
            for f in self.vertex_faces[u] | self.vertex_faces[v]:
                for w in self.faces[f]:
                    affected_verts.add(w)

            # EXECUTE COLLAPSE: merge v into u at v_opt
            self.vertices[u] = v_opt
            self.active_vertices.remove(v)
            self.quadrics.Q[u] += self.quadrics.Q[v]

            # Remove shared faces
            shared_list = list(shared)
            for f in shared_list:
                if f in self.active_faces:
                    self.active_faces.remove(f)
                fa = self.faces[f]
                for w in fa:
                    self.vertex_faces[w].discard(f)
                for e_a, e_b in ((fa[0], fa[1]), (fa[1], fa[2]), (fa[2], fa[0])):
                    e_k = (min(e_a, e_b), max(e_a, e_b))
                    if e_k in self.edge_shared_faces:
                        self.edge_shared_faces[e_k].discard(f)
                        if not self.edge_shared_faces[e_k]:
                            self.edge_shared_faces.pop(e_k, None)

            # Update surviving faces of v to u
            for f in list(self.vertex_faces[v]):
                fa = self.faces[f]
                for e_a, e_b in ((fa[0], fa[1]), (fa[1], fa[2]), (fa[2], fa[0])):
                    e_k = (min(e_a, e_b), max(e_a, e_b))
                    if e_k in self.edge_shared_faces:
                        self.edge_shared_faces[e_k].discard(f)
                        if not self.edge_shared_faces[e_k]:
                            self.edge_shared_faces.pop(e_k, None)
                fa[fa == v] = u
                self.vertex_faces[u].add(f)
                for e_a, e_b in ((fa[0], fa[1]), (fa[1], fa[2]), (fa[2], fa[0])):
                    e_k = (min(e_a, e_b), max(e_a, e_b))
                    self.edge_shared_faces.setdefault(e_k, set()).add(f)
            self.vertex_faces[v].clear()

            # Update neighbor relationships
            self.vertex_neighbors[u].discard(v)
            for nbr in list(self.vertex_neighbors[v]):
                self.vertex_neighbors[nbr].discard(v)
                if nbr != u and nbr in self.active_vertices:
                    self.vertex_neighbors[nbr].add(u)
                    self.vertex_neighbors[u].add(nbr)
            self.vertex_neighbors[v].clear()

            self.edge_shared_faces.pop(edge_key, None)

            # Rebuild touched neighbor sets from surviving faces, removing unsupported edges.
            for w in affected_verts:
                if w in self.active_vertices:
                    self.vertex_neighbors[w] = {int(a) for f in self.vertex_faces[w] for a in self.faces[f] if a != w}
            # Update boundary vertices for affected verts
            self.boundary_vertices.discard(v)
            for w in affected_verts:
                if w in self.active_vertices:
                    is_bnd = any(
                        len(self.edge_shared_faces.get((min(w, nbr), max(w, nbr)), ())) == 1
                        for nbr in self.vertex_neighbors[w]
                    )
                    if is_bnd:
                        self.boundary_vertices.add(w)
                    else:
                        self.boundary_vertices.discard(w)

            # Dynamic local geometry update (including sharp graph)
            graph_changed, changed_verts = self.feature_extractor.update_local_geometry(
                affected_faces, affected_verts, self.vertex_faces,
                edge_shared_faces=self.edge_shared_faces,
                v_merged=v, u_surviving=u
            )

            current_v -= 1
            self.collapsed_count += 1
            collapsed_this_round += 1

            # Refresh priority for all edges incident to ANY affected vertex
            affected_edges = set()
            for w in affected_verts:
                if w in self.active_vertices:
                    for nbr in self.vertex_neighbors[w]:
                        if nbr in self.active_vertices:
                            affected_edges.add((min(w, nbr), max(w, nbr)))

            if graph_changed and self.net is not None:
                for cv in changed_verts:
                    if cv in self.active_vertices:
                        for nbr in self.vertex_neighbors[cv]:
                            if nbr in self.active_vertices:
                                affected_edges.add((min(cv, nbr), max(cv, nbr)))

            sorted_affected = sorted(affected_edges)
            for e_u, e_v in sorted_affected:
                v_ver = self.edge_version.get((e_u, e_v), 0) + 1
                self.edge_version[(e_u, e_v)] = v_ver

            eval_results = self._evaluate_candidate_edges_batched(sorted_affected)
            for e_u, e_v, p, n_vopt, _ in eval_results:
                v_ver = self.edge_version[(e_u, e_v)]
                heapq.heappush(self.heap, (p, e_u, e_v, v_ver, n_vopt))

            if verbose and collapsed_this_round % 500 == 0:
                elapsed = time.time() - start_time
                print(f"  -> Collapsed {self.collapsed_count} edges | Remaining V: {current_v} | Elapsed: {elapsed:.2f}s")

        self.target_reached = (current_v <= target_v)
        if not self.target_reached and verbose:
            print(f"[{METHOD_ACRONYM} WARNING] Heap exhausted before target reached: V={current_v} > target={target_v}")

        rebuilt = self._rebuild_mesh()
        if verbose:
            total_elapsed = time.time() - start_time
            print(f"[{METHOD_ACRONYM}] Step complete: {len(rebuilt.vertices)} verts, {len(rebuilt.faces)} faces in {total_elapsed:.2f}s")
        return rebuilt

    def simplify(self, target_ratio: float, verbose: bool = True) -> trimesh.Trimesh:
        """Run complete decimation loop down to target ratio (relative to original mesh size)."""
        if not (0.0 < target_ratio <= 1.0):
            raise ValueError(f"target_ratio must be in (0.0, 1.0], got {target_ratio}")
        target_v = min(self.num_verts, max(4, int(np.ceil(target_ratio * self.num_verts))))
        return self.advance_to_vertex_count(target_v, verbose=verbose)

    def _rebuild_mesh(self, world: bool = True) -> trimesh.Trimesh:
        surviving_f_indices = sorted(self.active_faces)
        surviving_faces = self.faces[surviving_f_indices]
        old_to_new = {}
        new_vertices = []
        for new_idx, old_idx in enumerate(sorted(self.active_vertices)):
            old_to_new[old_idx] = new_idx
            new_vertices.append(self.vertices[old_idx])

        remapped_faces = []
        for face in surviving_faces:
            remapped_faces.append([old_to_new[face[0]], old_to_new[face[1]], old_to_new[face[2]]])

        # Restore original world coordinates
        norm_verts = np.asarray(new_vertices, dtype=np.float64)
        world_verts = (norm_verts / self.scale) + self.bbox_center if world else norm_verts

        simplified_mesh = trimesh.Trimesh(
            vertices=world_verts,
            faces=np.asarray(remapped_faces, dtype=np.int64),
            process=False,
        )
        return simplified_mesh


# ============================================================================
# 7. Training & Horizon Rollout Dataset Generator
# ============================================================================

def generate_training_sample(
    mesh: trimesh.Trimesh,
    net: Optional[EdgeDecisionNet] = None,
    samples_per_state: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ground-truth regression and ranking targets using counterfactual rollouts.
    Returns (features, target_damages).
    """
    extractor = CADEdgeFeatureExtractor(mesh)
    quadrics = QuadricKernel(mesh.vertices, mesh.faces)
    gatekeeper = TopologyGatekeeper()

    # Find valid candidate edges
    valid_edges = []
    v_neighbors = [set() for _ in range(len(mesh.vertices))]
    v_faces = [set() for _ in range(len(mesh.vertices))]
    e_faces: Dict[Tuple[int, int], Set[int]] = {}

    for f_idx, (a, b, c) in enumerate(mesh.faces):
        v_faces[a].add(f_idx)
        v_faces[b].add(f_idx)
        v_faces[c].add(f_idx)
        for u, v in ((a, b), (b, c), (c, a)):
            v_neighbors[u].add(v)
            v_neighbors[v].add(u)
            e_faces.setdefault((min(u, v), max(u, v)), set()).add(f_idx)

    for (u, v), shared in e_faces.items():
        if not gatekeeper.check_link_condition(u, v, v_neighbors[u], v_neighbors[v], list(shared), mesh.faces):
            continue
        v_opt, cost = quadrics.solve_optimal_position(u, v, mesh.vertices[u], mesh.vertices[v])
        if not gatekeeper.check_normal_inversion(u, v, v_opt, v_faces[u], v_faces[v], shared, mesh.vertices, mesh.faces):
            continue
        valid_edges.append((u, v, v_opt, cost, shared))

    if len(valid_edges) == 0:
        return np.empty((0, 10), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Subsample candidates to evaluate
    if len(valid_edges) > samples_per_state:
        indices = np.random.choice(len(valid_edges), samples_per_state, replace=False)
        selected_candidates = [valid_edges[i] for i in indices]
    else:
        selected_candidates = valid_edges

    features_list = []
    targets_list = []

    for u, v, v_opt, cost, shared in selected_candidates:
        edge_len = float(np.linalg.norm(mesh.vertices[v] - mesh.vertices[u]))
        feat = extractor.extract_features(u, v, edge_len, cost, v_faces[u], v_faces[v], shared)

        # Counterfactual damage computation
        # 1. Feature damage: how much feature vertices move
        is_u_feat = u in extractor.sharp_vertices
        is_v_feat = v in extractor.sharp_vertices
        if is_u_feat or is_v_feat:
            fdpe_target = min(1.0, edge_len / max(extractor.median_edge_len, 1e-6))
        else:
            fdpe_target = 0.05

        # 2. Curvature damage
        nu = mesh.vertex_normals[u]
        nv = mesh.vertex_normals[v]
        ce_target = float(np.clip(1.0 - np.dot(nu, nv), 0.0, 1.0))

        # 3. CD damage (local point movement)
        cd_target = float(np.clip(cost * 100.0, 0.0, 1.0))

        features_list.append(feat)
        targets_list.append([cd_target, ce_target, fdpe_target])

    return np.asarray(features_list, dtype=np.float32), np.asarray(targets_list, dtype=np.float32)


def train_edge_decision_net(
    mesh_paths: List[str],
    checkpoint_out: str,
    epochs: int = 300,
    patience: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    cpu_fraction: float = DEFAULT_CPU_FRACTION,
    device: Optional[str] = None,
    val_split: float = 0.20,
    num_samples: int = DEFAULT_METRIC_SAMPLES,
    seed_count: int = DEFAULT_METRIC_SEED_COUNT,
) -> EdgeDecisionNet:
    """
    Train EdgeDecisionNet on multiple CAD meshes with multi-task loss and early stopping.
    - Training device fixed to: MPS (if available) / CPU.
    - CPU resource usage capped at 70%.
    - Max 300 Epochs with Early Stopping Patience = 30.
    """
    allocated_threads = configure_resource_limits(cpu_fraction)
    if device is None:
        device = get_train_device()

    print(f"\n" + "=" * 65)
    print(f"[{METHOD_ACRONYM} Training Setup]")
    print(f"  • Fixed Training Device:   {device.upper()}")
    print(f"  • Max Training Epochs:     {epochs}")
    print(f"  • Early Stopping Patience: {patience}")
    print(f"  • CPU Thread Cap (70%):    {allocated_threads} threads")
    print(f"  • Metric Samples x Seeds:  {num_samples} points x {seed_count} seeds")
    print("=" * 65)

    print(f"[Train] Collecting dataset from {len(mesh_paths)} models...")
    all_feats = []
    all_targets = []

    for path in mesh_paths:
        try:
            m = load_triangle_mesh(path)
            feats, targets = generate_training_sample(m, samples_per_state=128)
            if len(feats) > 0:
                all_feats.append(feats)
                all_targets.append(targets)
        except Exception as e:
            print(f"[Train] Skipped {path}: {e}")

    if not all_feats:
        raise RuntimeError("No valid training samples generated.")

    X = np.concatenate(all_feats, axis=0)
    Y = np.concatenate(all_targets, axis=0)
    total_samples = len(X)
    print(f"[Train] Dataset ready: {total_samples} edge samples.")

    # Partition train / validation split deterministically
    np.random.seed(DEFAULT_BASE_SEED)
    indices = np.random.permutation(total_samples)
    val_size = max(1, int(total_samples * val_split))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train_t = torch.from_numpy(X[train_idx]).to(device)
    Y_train_t = torch.from_numpy(Y[train_idx]).to(device)
    X_val_t = torch.from_numpy(X[val_idx]).to(device)
    Y_val_t = torch.from_numpy(Y[val_idx]).to(device)

    net = EdgeDecisionNet(in_dim=10, hidden=128, num_layers=3, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    criterion_reg = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = copy.deepcopy(net.state_dict())

    num_train = len(train_idx)
    print(f"[Train] Starting optimization: {num_train} train, {val_size} val samples on {device.upper()}...")

    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(num_train)
        train_loss = 0.0
        num_batches = 0

        for i in range(0, num_train, batch_size):
            b_idx = perm[i : i + batch_size]
            bx = X_train_t[b_idx]
            by = Y_train_t[b_idx]

            optimizer.zero_grad()
            p_cd, p_ce, p_fdpe, comp_risk = net(bx)

            # 1. Multi-task regression loss aligned with (CD, CE, FDPE)
            loss_cd = criterion_reg(p_cd, by[:, 0:1])
            loss_ce = criterion_reg(p_ce, by[:, 1:2])
            loss_fdpe = criterion_reg(p_fdpe, by[:, 2:3])
            loss_reg = 0.20 * loss_cd + 0.40 * loss_ce + 0.40 * loss_fdpe

            # 2. Pairwise ranking loss on composite score
            if len(bx) > 1:
                idx_a = torch.arange(len(bx) - 1)
                idx_b = idx_a + 1
                diff_true = (0.20 * by[idx_a, 0] + 0.40 * by[idx_a, 1] + 0.40 * by[idx_a, 2]) - \
                            (0.20 * by[idx_b, 0] + 0.40 * by[idx_b, 1] + 0.40 * by[idx_b, 2])
                diff_pred = comp_risk[idx_a, 0] - comp_risk[idx_b, 0]
                y_rank = torch.sign(diff_true)
                loss_rank = F.margin_ranking_loss(diff_pred, torch.zeros_like(diff_pred), y_rank, margin=0.05)
            else:
                loss_rank = torch.tensor(0.0, device=device)

            loss = loss_reg + 1.5 * loss_rank
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        # Validation Step
        net.eval()
        with torch.no_grad():
            v_cd, v_ce, v_fdpe, v_comp = net(X_val_t)
            v_loss_cd = criterion_reg(v_cd, Y_val_t[:, 0:1])
            v_loss_ce = criterion_reg(v_ce, Y_val_t[:, 1:2])
            v_loss_fdpe = criterion_reg(v_fdpe, Y_val_t[:, 2:3])
            val_loss = float(0.20 * v_loss_cd + 0.40 * v_loss_ce + 0.40 * v_loss_fdpe)

        # Early Stopping check
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = copy.deepcopy(net.state_dict())
            is_best = True
        else:
            patience_counter += 1
            is_best = False

        if epoch % 10 == 0 or epoch == 1 or is_best or patience_counter >= patience:
            marker = " ★ (Best)" if is_best else ""
            print(f"  Epoch {epoch:3d}/{epochs:3d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} (Patience: {patience_counter}/{patience}){marker}")

        if patience_counter >= patience:
            print(f"\n[EarlyStopping] Reached patience threshold ({patience}). Early stopping triggered at epoch {epoch}!")
            print(f"[EarlyStopping] Restoring best model weights from epoch {best_epoch} (Val Loss: {best_val_loss:.5f}).")
            break

    # Restore best checkpoint
    net.load_state_dict(best_state_dict)
    out_p = Path(checkpoint_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(out_p))
    print(f"[Train] Best checkpoint saved successfully to: {checkpoint_out}")
    return net


# ============================================================================
# 8. Command-Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} ({METHOD_ACRONYM})")
    parser.add_argument("--mode", type=str, default="simplify", choices=["simplify", "evaluate", "train", "benchmark"],
                        help="Operation mode.")
    parser.add_argument("-i", "--input", type=str, default="", help="Input OBJ mesh path.")
    parser.add_argument("-o", "--output", type=str, default="", help="Output simplified OBJ path.")
    parser.add_argument("-r", "--ratio", type=float, default=0.5, help="Simplification target ratio (0 < r <= 1.0).")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/carn_ms_net.pth", help="Model checkpoint path; simplify falls back to the bundled carn_ms_pretrained.pth.")
    parser.add_argument("--no_gatekeeper", action="store_true", help="Disable topology/inversion gatekeeper.")
    parser.add_argument("--device", type=str, default="", help="Device override (cpu, mps, cuda). Default: train on mps, simplify/evaluate on cpu.")
    parser.add_argument("--train_models", type=str, nargs="+", default=[], help="Model paths for training.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300).")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30).")
    parser.add_argument("--metric_samples", "--num_samples", dest="metric_samples", type=int, default=DEFAULT_METRIC_SAMPLES,
                        help="Number of surface points for evaluation (default: 5000).")
    parser.add_argument("--metric_seed_count", "--seed_count", dest="metric_seed_count", type=int, default=DEFAULT_METRIC_SEED_COUNT,
                        help="Number of evaluation seeds (default: 3).")
    parser.add_argument("--cpu_fraction", type=float, default=DEFAULT_CPU_FRACTION,
                        help="CPU usage ratio cap (default: 0.70 for 70%% resource limit).")
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.mode in ("train", "benchmark"):
        print("[Development entry only] Freeze experiments use cache_c5d_dataset.py -> train_c5d_generalized.py -> benchmark_c5d_suite.py; built-in benchmark uses net=None.")

    # 1. Enforce 70% CPU resource limit globally
    allocated_threads = configure_resource_limits(args.cpu_fraction)

    if args.mode == "simplify":
        if not args.input or not args.output:
            print("Error: --input and --output are required for simplify mode.", file=sys.stderr)
            sys.exit(1)

        mesh = load_triangle_mesh(args.input)
        net = None
        # Device for inference (simplification priority & geometry updates) is fixed to CPU
        infer_device = args.device if args.device else get_infer_device()
        ckpt_path = args.checkpoint
        default_ckpt = "./checkpoints/carn_ms_net.pth"
        explicit_checkpoint = any(a == "--checkpoint" or a.startswith("--checkpoint=") for a in sys.argv[1:])
        if explicit_checkpoint and not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Specified checkpoint file does not exist: {ckpt_path}")
        if ckpt_path == default_ckpt:
            for fallback in [str(Path(__file__).resolve().with_name("carn_ms_pretrained.pth"))]:
                if not os.path.exists(ckpt_path) and os.path.exists(fallback):
                    ckpt_path = fallback
                    break
        else:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Specified checkpoint file does not exist: {ckpt_path}")

        if os.path.exists(ckpt_path):
            print(f"[{METHOD_ACRONYM}] Loading neural priority model: {ckpt_path} (Inference Device: {infer_device.upper()})")
            net = EdgeDecisionNet(in_dim=10, hidden=128, num_layers=3)
            net.load_state_dict(torch.load(ckpt_path, map_location=infer_device))
        else:
            print(f"[{METHOD_ACRONYM}] No default checkpoint found. Running the rule-only area-weighted QEM + gatekeeper variant.")

        # Algorithm runtime includes CARN-MS state construction, initial candidate
        # preparation, neural/rule-guided collapse execution, and mesh rebuilding.
        # Output serialization and terminal metric evaluation are excluded.
        algorithm_start = time.perf_counter()
        decimator = NeuroDecimator(
            mesh=mesh,
            net=net,
            device=infer_device,
            use_gatekeeper=not args.no_gatekeeper,
        )
        simplified = decimator.simplify(target_ratio=args.ratio)
        algorithm_runtime = time.perf_counter() - algorithm_start
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        simplified.export(str(out_p))
        print(f"[{METHOD_ACRONYM}] Exported simplified mesh: {args.output}")

        # Automatically evaluate metrics: 5000 samples x 3 seeds on CPU
        eval_device = get_eval_device()
        print(f"\n[{METHOD_ACRONYM}] Evaluating terminal metrics ({args.metric_samples} points x {args.metric_seed_count} seeds on {eval_device.upper()})...")
        metrics = evaluate_terminal_metrics(
            mesh, simplified,
            num_samples=args.metric_samples,
            seed_count=args.metric_seed_count,
        )
        print("\n" + "=" * 60)
        print(f"📊 [{METHOD_ACRONYM} Quality Report: {args.metric_samples} pts x {args.metric_seed_count} seeds (Device: CPU)]")
        print("=" * 60)
        print(f"  • CD (Chamfer Distance):           {metrics['cd']:.8f} ± {metrics['cd_std']:.8f}")
        print(f"  • CE (Curvature/Normal Error):     {metrics['ce']:.8f} ± {metrics['ce_std']:.8f}")
        print(f"  • FDPE_RMS (Feature Preservation): {metrics['fdpe_rms']:.8f} (deterministic; {metrics['feature_status']})")
        print(f"  • HD_max (Max Hausdorff, Aux):     {metrics['hd_max']:.8f} ± {metrics['hd_max_std']:.8f}")
        print(f"  • Algorithm Runtime:               {algorithm_runtime:.4f} s")
        print("=" * 60)

        # Structured one-line outputs for log parsing.
        print(f"[Metrics] CD={metrics['cd']:.9g}")
        print(f"[Metrics] CE={metrics['ce']:.9g}")
        print(f"[Metrics] S_FDPE={metrics['fdpe_rms']:.9g}")
        print(f"[Metrics] HD_max={metrics['hd_max']:.9g}")
        print(f"[Metrics] Runtime={algorithm_runtime:.6f} s")

    elif args.mode == "evaluate":
        if not args.input or not args.output:
            print("Error: --input (original) and --output (simplified) are required for evaluate mode.", file=sys.stderr)
            sys.exit(1)
        orig = load_triangle_mesh(args.input)
        simp = load_triangle_mesh(args.output)
        metrics = evaluate_terminal_metrics(
            orig, simp,
            num_samples=args.metric_samples,
            seed_count=args.metric_seed_count,
        )
        print(json.dumps(metrics, indent=2))

    elif args.mode == "train":
        models = args.train_models
        if not models:
            models = [f"{i}.obj" for i in range(1, 6) if os.path.exists(f"{i}.obj")]

        train_device = args.device if args.device else get_train_device()
        train_edge_decision_net(
            mesh_paths=models,
            checkpoint_out=args.checkpoint,
            epochs=args.epochs,
            patience=args.patience,
            cpu_fraction=args.cpu_fraction,
            device=train_device,
            num_samples=args.metric_samples,
            seed_count=args.metric_seed_count,
        )

    elif args.mode == "benchmark":
        print(f"[Benchmark] Benchmarking {METHOD_ACRONYM} across models (5000 pts x 3 seeds on CPU)...")
        models = [f"{i}.obj" for i in range(1, 6) if os.path.exists(f"{i}.obj")]
        for m_path in models:
            orig = load_triangle_mesh(m_path)
            decimator = NeuroDecimator(mesh=orig, net=None, device="cpu", use_gatekeeper=True)
            simp = decimator.simplify(target_ratio=0.5, verbose=False)
            metrics = evaluate_terminal_metrics(orig, simp, num_samples=args.metric_samples, seed_count=args.metric_seed_count)
            print(f"Model {m_path}: V={len(simp.vertices)}, CD={metrics['cd']:.6f}±{metrics['cd_std']:.6f}, CE={metrics['ce']:.6f}±{metrics['ce_std']:.6f}, FDPE={metrics['fdpe_rms']:.6f}")


if __name__ == "__main__":
    main()
