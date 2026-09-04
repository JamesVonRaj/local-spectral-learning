"""Network generators for the Bloch band-gap sweep.

Spatial (periodic, Euclidean edge lengths):
    rand-del : uniform random points (Poisson) + periodic Delaunay (~6-coord)

Non-spatial (no positions; edge 'lengths' are drawn L_e ~ Uniform(0.5, 2.0)
as quenched per-edge disorder, independent of any geometry):
    er : Erdos-Renyi G(N=size^2, m=3*size^2)     (N=400, M=1200)
    ba : Barabasi-Albert (m=3 preferential)      (N=400, M~1191)
    ws : Watts-Strogatz (k=6, p=0.1)             (N=400, M=1200)

`make_network(topology, size, seed)` returns:
    pos     : (N, 2)      node positions; zeros for non-spatial
    edges   : (M, 2) int
    lengths : (M,)        Euclidean for spatial, Uniform(0.5, 2.0) for non-spatial
    box     : (2,)        bounding box; [1,1] placeholder for non-spatial

The random-Delaunay generator in ``mechanical.netgen`` is an independent,
compact NumPy/SciPy implementation. The lattice/Gabriel topologies of an
earlier exploratory sweep (hu-del, lat-del, rand-gab) are not part of the
publication pipeline.
"""
import numpy as np

from mechanical.netgen import GraphGenerator

SPATIAL = ('rand-del',)
NONSPATIAL = ('er', 'ba', 'ws')
TOPOLOGIES = SPATIAL + NONSPATIAL


def _make_spatial(topology, size, seed):
    if topology != 'rand-del':
        raise NotImplementedError(
            f"spatial topology '{topology}' is not available in this "
            "standalone repository; only 'rand-del' is supported"
        )
    # Reproduces LatticeGenerator(size, C=1).generate_random(): seed the global
    # RNG, then draw size^2 uniform points in the [0, size]^2 box.
    np.random.seed(seed)
    box = np.array([float(size), float(size)])
    pts = np.random.uniform(low=0.0, high=box, size=(size * size, 2))

    gg = GraphGenerator(pts, box)
    _, edges_raw = gg.periodic_delaunay_tessellation()

    edges = edges_raw[:, :2].astype(np.int64)
    lengths = edges_raw[:, 2].astype(np.float64)
    return pts, edges, lengths, box


def _make_nonspatial(topology, size, seed):
    """Abstract graph + quenched random per-edge 'lengths' in [0.5, 2.0]."""
    import networkx as nx
    N = size * size  # 400 for size=20, matches spatial run

    # Connectivity: ER at m=1200/N=400 sits at the connectivity threshold;
    # retry with offset seeds if a realization is disconnected.
    def _pick(make_fn):
        for offset in range(0, 10000, 1):
            try_seed = seed + offset * 10000
            G = make_fn(try_seed)
            if nx.is_connected(G):
                return G, try_seed
        raise RuntimeError(
            f'{topology}: no connected graph within 10k seed offsets of {seed}')

    if topology == 'er':
        G, _ = _pick(lambda s: nx.gnm_random_graph(n=N, m=1200, seed=s))
    elif topology == 'ba':
        G, _ = _pick(lambda s: nx.barabasi_albert_graph(n=N, m=3, seed=s))
    elif topology == 'ws':
        G, _ = _pick(lambda s: nx.watts_strogatz_graph(n=N, k=6, p=0.1, seed=s))
    else:
        raise ValueError(f'unknown non-spatial topology: {topology}')

    edges = np.array(sorted(G.edges()), dtype=np.int64)

    # Lengths drawn with an rng keyed *only* on the requested seed so that
    # two generators with the same seed share a reproducible length draw
    # conditional on their edge count.
    rng = np.random.RandomState(seed)
    lengths = rng.uniform(0.5, 2.0, size=len(edges)).astype(np.float64)

    # Placeholder pos/box so downstream npz schema is unchanged.
    pos = np.zeros((N, 2), dtype=np.float64)
    box = np.array([1.0, 1.0], dtype=np.float64)
    return pos, edges, lengths, box


def make_network(topology, size, seed):
    if topology in SPATIAL:
        return _make_spatial(topology, size, seed)
    if topology in NONSPATIAL:
        return _make_nonspatial(topology, size, seed)
    raise ValueError(f'unknown topology: {topology}')


if __name__ == '__main__':
    for topo in TOPOLOGIES:
        pos, edges, lengths, box = make_network(topo, size=20, seed=0)
        print(f'{topo:10s}  N={len(pos):4d}  M={len(edges):5d}  '
              f'L=[{lengths.min():.3f},{lengths.max():.3f}]  '
              f'mean={lengths.mean():.3f}  coord={2*len(edges)/len(pos):.2f}')
