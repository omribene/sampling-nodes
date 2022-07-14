import argparse
import os
import sys
from pathlib import Path
from random import choice
from time import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from globals import datasets
from GraphUtils import GraphUtils as gu
from previous_exp.rejection import RejectionSampling
from previous_exp.mh import MetropolisHastingsSampling
from previous_exp.max_deg import MaxDegreeSampling

parser = argparse.ArgumentParser()

parser.add_argument('--output_path', default=None)
parser.add_argument('--method', choices=['rej', 'mh', 'md'], help='which sampling method to use: '
                                                                  'rej -- rejection sampling (Alg. 1), mh - Metropolis-Hastings, md - Max Degree',
                    default='rej')
parser.add_argument('--init_steps', default=10000)
parser.add_argument('--total_steps', default=1000000)
parser.add_argument('--sample_interval', default=1000)
parser.add_argument('--exp_num', type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    sampling_method, sampling = None, None

    start = time()
    path, sep, title, directed = datasets[0]
    g, in_degrees = gu.load_graph(path, directed=directed)
    g.to_undirected()
    initial_node = choice(g.vs)
    min_deg = min(g.degree())
    max_deg = max(g.degree())
    if args.method == 'rej':
        sampling_method = 'rej'
        sampling = RejectionSampling(initial_node=initial_node, min_deg=min_deg)
    elif args.method == 'mh':
        sampling_method = 'mh'
        sampling = MetropolisHastingsSampling(initial_node=initial_node, min_deg=min_deg)
    elif args.method == 'md':
        sampling_method = 'md'
        sampling = MaxDegreeSampling(initial_node=initial_node, max_deg=max_deg)

    # init walk
    sampling.random_walk(args.init_steps)
    sampled_nodes = []
    cum_queries = []
    for i in tqdm(range(int((args.total_steps - args.init_steps) / args.sample_interval))):
        sampling.random_walk(steps=args.sample_interval)
        v = sampling.get_node()
        if v:
            sampled_nodes.append(v.index)
            cum_queries.append(sampling.query_count)
    elapsed = (time() - start)
    print(
        f"{title}-{sampling_method}-exp-{args.exp_num}: Done. Elapsed time: {elapsed:.2f}s. Number of sampled nodes: {len(sampled_nodes)}. Number of queries made: {sampling.query_count}")
    results = pd.DataFrame({"nodes": sampled_nodes, "cum_queries": cum_queries})
    output_path = Path(args.output_path).joinpath(
        f"{sampling_method}_{title}_init-{args.init_steps}_interval-{args.sample_interval}_total-{args.total_steps}_{args.exp_num}.csv")
    print("Given output path: ", args.output_path)
    print("outputting results to the file: ", output_path)
    results.to_csv(output_path, header=False, index=False)
