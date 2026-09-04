"""Network generation for the mechanical band-gap pipeline.

Generates a (pos, edges, lengths, box) network and saves it to NPZ.

Usage:
    # Generate a new network
    PYTHONPATH=scripts python -m mechanical.generate \\
        --topology rand-del --size 20 --seed 0 --output network.npz

    # Use JSON configuration
    PYTHONPATH=scripts python -m mechanical.generate --config config.json
"""
import argparse
import os

import numpy as np

from mechanical.io import load_mechanical_config
from mechanical.topology import make_network


def generate_from_config(config):
    """Generate (or load preset) a network and return a network dict.

    The dict has keys ``topology``, ``net_seed``, ``size``, ``pos``,
    ``edges``, ``lengths``, ``box``.  When ``network.preset`` is set, the
    listed NPZ is loaded instead and ``topology`` / ``size`` / ``net_seed``
    come from that file.
    """
    net = config['network']

    if net.get('preset'):
        path = net['preset']
        print(f'Loading network from preset: {path}')
        npz = np.load(path, allow_pickle=True)
        return {
            'topology': str(npz['topology']) if 'topology' in npz.files else 'preset',
            'net_seed': int(npz['net_seed']) if 'net_seed' in npz.files else 0,
            'size': int(npz['size']) if 'size' in npz.files else 0,
            'pos': npz['pos'], 'edges': npz['edges'],
            'lengths': npz['lengths'], 'box': npz['box'],
        }

    topology = net['topology']
    size = int(net.get('size', 20))
    seed = int(net.get('seed', 0))
    print(f'Generating {topology} network: size={size}, seed={seed}')

    pos, edges, lengths, box = make_network(topology, size=size, seed=seed)
    print(f'  -> N={len(pos)}, M={len(edges)}')
    return {
        'topology': topology, 'net_seed': seed, 'size': size,
        'pos': pos, 'edges': edges, 'lengths': lengths, 'box': box,
    }


def main():
    p = argparse.ArgumentParser(
        description='Generate a mechanical network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Topologies:
  rand-del   Random Delaunay        (~6-coordinated, spatial)
  hu-del     Perturbed-lattice      (~6-coordinated, density-regular)
  rand-gab   Random Gabriel         (~4-coordinated, spatial)
  lat-del    Perturbed-lattice      (matches the slide reference)
  er         Erdos-Renyi            (non-spatial, m=3*size^2)
  ba         Barabasi-Albert        (non-spatial, m=3 preferential)
  ws         Watts-Strogatz         (non-spatial, k=6, p=0.1)
""")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--topology', '-t', type=str)
    src.add_argument('--preset', '-p', type=str,
                     help='Load network from existing NPZ')
    src.add_argument('--config', '-c', type=str)

    p.add_argument('--size', '-s', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', '-o', type=str, default=None)
    args = p.parse_args()

    if args.config:
        config = load_mechanical_config(args.config)
    elif args.preset:
        config = {'network': {'preset': args.preset}}
    else:
        config = {'network': {'topology': args.topology,
                              'size': args.size, 'seed': args.seed}}

    network = generate_from_config(config)

    out = args.output
    if out is None:
        out = (config.get('paths', {}).get('data_file')
               or f'../outputs/mechanical/data/network_{network["topology"]}.npz')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez_compressed(
        out,
        topology=network['topology'],
        net_seed=np.int64(network['net_seed']),
        size=np.int64(network['size']),
        pos=network['pos'].astype(np.float64),
        edges=network['edges'].astype(np.int64),
        lengths=network['lengths'].astype(np.float64),
        box=np.asarray(network['box']).astype(np.float64),
    )
    print(f'Saved network to: {out}')


if __name__ == '__main__':
    main()
