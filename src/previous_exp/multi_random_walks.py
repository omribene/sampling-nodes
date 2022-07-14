import argparse
import json
import logging
import os
import sys
from multiprocessing import cpu_count, freeze_support, Pool, RLock
from pathlib import Path
from time import time
from typing import NewType

import igraph as ig
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm, trange

path = Path(os.getcwd())
sys.path.insert(0, str(path / "src"))
sys.path.insert(0, str(path.parent / "src/previous_exp"))

from utils.globals import datasets
from utils.graph_utils import GraphUtils as gu
from previous_exp.rejection import RejectionSampling
from previous_exp.mh import MetropolisHastingsSampling
from previous_exp.max_deg import MaxDegreeSampling

Graph = NewType("Graph", ig.GraphBase)

parser = argparse.ArgumentParser()

parser.add_argument('--config', default=None,
                    help="An experiment configuration file. If specified, overrides the other argument parameter.")
parser.add_argument('--output_path', default=None)
parser.add_argument('--method', choices=['rej', 'mh', 'mh+', 'md'], help='which sampling method to use: '
                                                                         'rej -- rejection sampling (Alg. 1), mh - Metropolis-Hastings, md - Max Degree',
                    default='rej')
parser.add_argument('--interval_length', type=int, default=100)
parser.add_argument('--num_intervals', type=int, default=300)
parser.add_argument('--exp_num', type=int, default=-1)
parser.add_argument('--min_exp_idx', default=0, type=int)
parser.add_argument('--cpu_count', default=0, type=int)
parser.add_argument('--min_dataset_idx', type=int)
parser.add_argument('--max_dataset_idx', type=int)
parser.add_argument('--n_steps_min', default=1, type=int)
parser.add_argument('--n_steps_max', default=1, type=int)
parser.add_argument('--n_init_nodes', default=1, type=int)
parser.add_argument('--starting_node', type=int, default=-1)

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')


def run_experiment(g: Graph, init_node_idx: int, experiment_index: int, method: str, title: str, total_intervals: int,
                   interval_length: int = 100, n_steps_min: int = 1, n_steps_max: int = 1):
    initial_node = g.vs[init_node_idx]
    min_deg = min(g.degree())
    max_deg = max(g.degree())
    sampler = None
    if method == 'rej':
        sampler = RejectionSampling(initial_node=initial_node, min_deg=min_deg, n_steps_min=n_steps_min,
                                    n_steps_max=n_steps_max)
    elif method == 'mh':
        sampler = MetropolisHastingsSampling(initial_node=initial_node, min_deg=min_deg, plus=False)
    elif method == 'mh+':
        sampler = MetropolisHastingsSampling(initial_node=initial_node, min_deg=min_deg, plus=True)
    elif method == 'md':
        sampler = MaxDegreeSampling(initial_node=initial_node, max_deg=max_deg)

    samples = []

    last_accepted = sampler.current_node.index
    step_counter = 0
    for _ in trange(total_intervals):
        for _ in range(interval_length):
            sampler.random_step()
            node = sampler.get_node()
            node_idx = node.index if node is not None else None
            if node_idx is not None:
                last_accepted = node_idx

        step_counter += interval_length
        samples.append((step_counter, int(last_accepted), sampler.query_count))

    simulation_df = pd.DataFrame(samples, columns=['step', 'node', 'queries'])
    return simulation_df


def run_batch(g: Graph, dataset_idx: int, init_node_idx: int, experiment_index_range: tuple, method: str,
              output_path: str,
              total_intervals: int, interval_length: int, n_steps_min: int, n_steps_max: int, batch_id: int):
    path, sep, title, directed = datasets[dataset_idx]
    start_idx, end_idx = experiment_index_range
    file_path = Path(output_path)

    tqdm_text = "Batch #" + "{}".format(batch_id).zfill(3)

    if not file_path.exists():
        file_path.mkdir(parents=True)

    # Combining tqdm with Pool was done is inspired by the src in this post:
    # https://leimao.github.io/blog/Python-tqdm-Multiprocessing/

    with tqdm(total=end_idx - start_idx, desc=tqdm_text, position=batch_id + 1) as pbar:
        for exp_idx in range(start_idx, end_idx):
            experiment_df = run_experiment(g, init_node_idx, exp_idx, method, title,
                                           total_intervals, interval_length, n_steps_min, n_steps_max)
            experiment_df["exp"] = exp_idx
            with open(file_path / f"{title}-{method}-exp_{start_idx}-{end_idx}-init_node_{init_node_idx}.tsv",
                      'a') as f:
                experiment_df.to_csv(f, sep="\t", index=False, header=f.tell() == 0)
            pbar.update()


def get_rand_nodes(graph_vcount, n_rand_nodes, seed) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(range(graph_vcount), size=n_rand_nodes).astype(int)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config is not None:
        p = Path(args.config)
        with p.open('r') as f:
            conf = yaml.full_load(f)
    else:
        conf = {}
        for arg in vars(args):
            conf[arg] = getattr(args, arg)

    print(f'Running with the following configuration: {conf}')

    logging.info("start")
    num_cpus = conf['cpu_count'] if conf['cpu_count'] > 0 else cpu_count()
    freeze_support()  # For Windows support
    pool = Pool(processes=num_cpus, initargs=(RLock(),), initializer=tqdm.set_lock)
    out_path = Path(conf['output_path'])
    out_path.mkdir(exist_ok=True)
    for dataset_idx in range(conf['min_dataset_idx'], conf['max_dataset_idx'] + 1):
        start = time()
        dataset = datasets[dataset_idx]
        path, sep, title, directed = dataset
        g, in_degrees = gu.load_graph(str(path), directed=directed)
        g.to_undirected()
        g = g.components().giant()
        exp_num = conf['exp_num'] if conf['exp_num'] > 0 else g.vcount()
        batch_size = np.ceil(exp_num / num_cpus).astype(int)
        experiment_idx_ranges = [(i, i + batch_size) for i in
                                 range(conf['min_exp_idx'], conf['min_exp_idx'] + exp_num, batch_size)]

        starting_nodes = get_rand_nodes(g.vcount(), n_rand_nodes=conf['n_init_nodes'], seed=dataset_idx)
        if conf['starting_node'] >= 0:
            starting_nodes = [conf['starting_node']]

        params = {'title': title, 'method': conf['method'], 'starting_nodes': [int(v) for v in starting_nodes],
                  'min_exp_idx': conf['min_exp_idx'], 'exp_num': exp_num,
                  'interval_length': conf['interval_length'], 'num_intervals': conf['num_intervals'],
                  'n_steps_min': conf['n_steps_min'], 'n_steps_max': conf['n_steps_max']}

        with open(out_path / f'{title}-{conf["method"]}-params.json', 'w') as f:
            json.dump(params, f, indent=4)

        for init_node_idx in starting_nodes:
            # print(f"Starting run for {title}, starting node {init_node_idx}")
            if num_cpus == 1:
                print("Running walks on single cpu")
                run_batch(g, dataset_idx, init_node_idx, experiment_idx_ranges[0], conf['method'], conf['output_path'],
                          conf['num_intervals'], conf['interval_length'], conf['n_steps_min'], conf['n_steps_max'],
                          0)
                continue
            pool.starmap(run_batch,
                         ((g, dataset_idx, init_node_idx, experiment_idx_range, conf['method'], conf['output_path'],
                           conf['num_intervals'], conf['interval_length'], conf['n_steps_min'], conf['n_steps_max'],
                           batch_id)
                          for batch_id, experiment_idx_range in enumerate(experiment_idx_ranges)))
            print("\n" * (len(experiment_idx_ranges) + 1))

        logging.info(f"Done with {title}. Elapsed time: {time() - start}s")
