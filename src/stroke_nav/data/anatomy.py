"""Query layer over a centerline graph: neighbours, geodesic distances, routes and target specs.

Target / start specs accepted everywhere (env, scripts):
    1234                          node id
    "left_carotid@120"            branch name + arc length in mm along that branch
    "left_carotid@-1"             last node of the branch (negative = from the distal end)
    "xyz:-120.5,110.2,-200.0"     nearest centerline node to a 3D point (mm)
    "random:left_carotid@0-60|right_carotid@0-60"
                                  sample a node per episode from one of the '|'-separated ranges
    "file:data/targets/x.json"    {"target_node": ..., "start_node": ...} written by pick_target.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from stroke_nav.data.centerline import CenterlineGraph, processed_path


@dataclass(frozen=True)
class RangeSpec:
    branch: str
    lo: float
    hi: float


class Anatomy:
    def __init__(self, graph: CenterlineGraph):
        self.graph = graph
        self.nodes = graph.nodes
        self.radius = graph.radius
        self.branch_id = graph.branch_id
        self.branch_names = graph.branch_names
        self.arc = graph.arc
        self.meta = graph.meta
        n = len(self.nodes)
        self.neighbors: list[list[int]] = [[] for _ in range(n)]
        for a, b in graph.edges:
            self.neighbors[int(a)].append(int(b))
            self.neighbors[int(b)].append(int(a))
        self.degree = np.array([len(x) for x in self.neighbors])
        w = np.linalg.norm(self.nodes[graph.edges[:, 0]] - self.nodes[graph.edges[:, 1]], axis=1)
        w = np.maximum(w, 1e-3)
        self._csr = csr_matrix(
            (np.concatenate([w, w]), (np.concatenate([graph.edges[:, 0], graph.edges[:, 1]]),
                                       np.concatenate([graph.edges[:, 1], graph.edges[:, 0]]))),
            shape=(n, n),
        )
        self._tree = cKDTree(self.nodes)
        self.bbox_min = self.nodes.min(0)
        self.bbox_max = self.nodes.max(0)
        self.center = 0.5 * (self.bbox_min + self.bbox_max)
        self.half_extent = 0.5 * float(np.max(self.bbox_max - self.bbox_min))
        self._dist_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model: str, variant: str | None = None, processed_dir: str | Path = "data/processed") -> "Anatomy":
        """Load ``data/processed/{model}[_{variant}].npz``; if variant is None, use the model's default
        variant (reference variant for 0248) or the only/first match."""
        from stroke_nav.data.fetch import default_variant

        variant = variant or default_variant(model)
        path = processed_path(model, variant, processed_dir)
        if not path.exists():
            matches = sorted(Path(processed_dir).glob(f"{model}*.npz"))
            if variant is None and matches:
                path = matches[0]
            else:
                raise FileNotFoundError(
                    f"{path} not found. Run: python scripts/build_anatomy.py --model {model}"
                    + (f" --variant {variant}" if variant else "")
                )
        return cls(CenterlineGraph.load(path))

    @property
    def name(self) -> str:
        v = self.meta.get("variant") or (self.meta.get("surface") or "")[:-4]
        return f"{self.meta.get('model', '?')}" + (f"/{v}" if v else "")

    # ------------------------------------------------------------------ basic queries
    def branch_nodes(self, branch: str) -> np.ndarray:
        if branch not in self.branch_names:
            raise KeyError(f"unknown branch {branch!r}; available: {self.branch_names}")
        idx = np.where(self.branch_id == self.branch_names.index(branch))[0]
        return idx[np.argsort(self.arc[idx])]

    def branch_of(self, node: int) -> str:
        return self.branch_names[int(self.branch_id[node])]

    def nearest_node(self, xyz) -> int:
        return int(self._tree.query(np.asarray(xyz, dtype=np.float64))[1])

    def distances_from(self, node: int) -> tuple[np.ndarray, np.ndarray]:
        """Geodesic distance (mm) from ``node`` to every node, and Dijkstra predecessors."""
        node = int(node)
        if node not in self._dist_cache:
            d, pred = dijkstra(self._csr, directed=False, indices=node, return_predecessors=True)
            if len(self._dist_cache) > 256:
                self._dist_cache.clear()
            self._dist_cache[node] = (d, pred)
        return self._dist_cache[node]

    def route(self, start: int, target: int) -> list[int]:
        """Shortest node path start -> target (inclusive)."""
        _, pred = self.distances_from(target)
        if not np.isfinite(self.distances_from(target)[0][start]):
            raise ValueError(f"target {target} is not reachable from start {start}")
        path = [int(start)]
        while path[-1] != target:
            path.append(int(pred[path[-1]]))
        return path

    def forward_direction(self, u: int, v: int, lookahead: int = 5) -> np.ndarray:
        """Unit direction from node u into the vessel that starts with edge u->v, looking ``lookahead`` nodes ahead."""
        return _forward_direction(self, int(u), int(v), int(lookahead))

    def walk_from(self, u: int, v: int, steps: int) -> int:
        """Node reached after ``steps`` nodes starting with edge u->v, continuing along the same
        branch where possible (else the straightest neighbour)."""
        prev, cur = int(u), int(v)
        for _ in range(steps - 1):
            nxt = [w for w in self.neighbors[cur] if w != prev]
            if not nxt:
                break
            same = [w for w in nxt if self.branch_id[w] == self.branch_id[cur]]
            d0 = self.nodes[cur] - self.nodes[prev]
            w = max(same or nxt, key=lambda w: float(np.dot(self.nodes[w] - self.nodes[cur], d0)))
            prev, cur = cur, w
        return cur

    def junctions(self) -> np.ndarray:
        return np.where(self.degree >= 3)[0]

    def leaves(self) -> np.ndarray:
        return np.where(self.degree == 1)[0]

    # ------------------------------------------------------------------ start / target specs
    def default_start(self) -> int:
        """Leaf of the root branch farthest from the centroid (for 0248: the distal descending aorta)."""
        root = self.meta.get("root", self.branch_names[0])
        nodes = self.branch_nodes(root)
        ends = [int(nodes[0]), int(nodes[-1])]
        ends = [e for e in ends if self.degree[e] == 1] or ends
        centroid = self.nodes.mean(0)
        return max(ends, key=lambda e: float(np.linalg.norm(self.nodes[e] - centroid)))

    def default_occlusion_target(self, min_lumen_mm: float = 1.0) -> int:
        """Proximal entrance node (radius >= ``min_lumen_mm``) of the primary occlusion or stenosis.

        Checks recorded 100 % occlusions in ``meta['occlusions']`` first, then the carotid stenosis
        region (60-160 mm along ``left_carotid`` / ``right_carotid``), and falls back to the tightest
        interior constriction across non-root branches.
        """
        occs = self.meta.get("occlusions") or []
        if occs:
            br = occs[0]["branch"]
            from_mm = float(occs[0]["from_mm"])
            nodes = self.branch_nodes(br)
            arcs = self.arc[nodes] - self.arc[nodes].min()
            idx = int(np.argmin(np.abs(arcs - from_mm)))
            while idx > 0 and self.radius[nodes[idx]] < min_lumen_mm:
                idx -= 1
            return int(nodes[idx])

        carotids = [b for b in ("left_carotid", "right_carotid") if b in self.branch_names]
        candidates = []
        for br in (carotids or [b for b in self.branch_names if b != self.meta.get("root")]):
            nodes = self.branch_nodes(br)
            if len(nodes) < 5:
                continue
            arcs = self.arc[nodes] - self.arc[nodes].min()
            rads = self.radius[nodes]
            mask = (arcs >= 60.0) & (arcs <= 160.0) if br in carotids and arcs.max() >= 120.0 else np.ones(len(nodes), dtype=bool)
            if not mask.any():
                mask = np.ones(len(nodes), dtype=bool)
            sub_idx = np.where(mask)[0]
            min_pos = int(sub_idx[int(np.argmin(rads[sub_idx]))])
            candidates.append((float(rads[min_pos]), br, nodes, min_pos))

        if candidates:
            _, _, nodes, idx = min(candidates, key=lambda item: item[0])
            while idx > 0 and self.radius[nodes[idx]] < min_lumen_mm:
                idx -= 1
            return int(nodes[idx])

        nodes = self.branch_nodes(self.branch_names[-1])
        return int(nodes[max(0, len(nodes) - 5)])

    def resolve(self, spec, rng: np.random.Generator | None = None) -> int:
        """Turn a node/target spec into a node id (see module docstring)."""
        if isinstance(spec, (int, np.integer)):
            return int(spec)
        if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 3:
            return self.nearest_node(spec)
        s = str(spec).strip()
        if s.lower() in ("occlusion", "stenosis"):
            return self.default_occlusion_target()
        if s.lstrip("-").isdigit():
            return int(s)
        if s.startswith("xyz:"):
            return self.nearest_node([float(v) for v in s[4:].split(",")])
        if s.startswith("file:"):
            data = json.loads(Path(s[5:]).read_text())
            return int(data["target_node"])
        if s.startswith("random:"):
            nodes = self.sample_pool(s)
            rng = rng or np.random.default_rng()
            return int(rng.choice(nodes))
        branch, _, where = s.partition("@")
        nodes = self.branch_nodes(branch)
        if not where:
            return int(nodes[-1])
        a = float(where)
        arcs = self.arc[nodes] - self.arc[nodes].min()
        if a < 0:
            a = arcs.max() + a + 1.0
        return int(nodes[int(np.argmin(np.abs(arcs - a)))])

    def sample_pool(self, spec: str) -> np.ndarray:
        """All candidate nodes of a 'random:' spec."""
        return _sample_pool(self, spec)

    def is_random(self, spec) -> bool:
        return isinstance(spec, str) and spec.startswith("random:")


def parse_range(part: str) -> RangeSpec:
    m = re.fullmatch(r"\s*([A-Za-z0-9_]+)\s*(?:@\s*(-?[\d.]+)\s*-\s*(-?[\d.]+))?\s*", part)
    if not m:
        raise ValueError(f"bad range spec {part!r}; expected branch or branch@lo-hi")
    lo = float(m.group(2)) if m.group(2) else 0.0
    hi = float(m.group(3)) if m.group(3) else float("inf")
    return RangeSpec(m.group(1), lo, hi)


def _sample_pool(anat: Anatomy, spec: str) -> np.ndarray:
    pool = []
    for part in spec[len("random:"):].split("|"):
        r = parse_range(part)
        nodes = anat.branch_nodes(r.branch)
        arcs = anat.arc[nodes] - anat.arc[nodes].min()
        total = float(arcs.max())
        lo = total + r.lo + 1.0 if r.lo < 0 else r.lo  # negative = measured from the distal end
        hi = total + r.hi + 1.0 if r.hi < 0 else r.hi
        sel = nodes[(arcs >= lo) & (arcs <= hi)]
        pool.append(sel)
    out = np.unique(np.concatenate(pool)) if pool else np.array([], dtype=int)
    if len(out) == 0:
        raise ValueError(f"spec {spec!r} matches no nodes")
    return out


def _forward_direction(anat: Anatomy, u: int, v: int, lookahead: int) -> np.ndarray:
    key = (u, v, lookahead)
    cache = anat.__dict__.setdefault("_fwd_cache", {})
    if key in cache:
        return cache[key]
    prev, cur = u, v
    last_dir = anat.nodes[v] - anat.nodes[u]
    for _ in range(lookahead - 1):
        nxt = [w for w in anat.neighbors[cur] if w != prev]
        if not nxt:
            break
        # continue along the same branch if possible, else the straightest neighbour
        same = [w for w in nxt if anat.branch_id[w] == anat.branch_id[cur]]
        cand = same or nxt
        d0 = anat.nodes[cur] - anat.nodes[prev]
        w = max(cand, key=lambda w: float(np.dot(anat.nodes[w] - anat.nodes[cur], d0)))
        prev, cur = cur, w
    d = anat.nodes[cur] - anat.nodes[u]
    if np.linalg.norm(d) < 1e-9:
        d = last_dir
    d = d / max(np.linalg.norm(d), 1e-9)
    cache[key] = d
    return d
