import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from globals import all_num_nodes
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir')
parser.add_argument('--info_path')
parser.add_argument('--output_dir', default=None)  # if output_path is not supplied, set it to results_path.
parser.add_argument('--with_empirical', dest='with_emp', action='store_true')
parser.add_argument('--no_empirical', dest='with_emp', action='store_false')
parser.set_defaults(with_emp=True)
parser.add_argument('--upto_step', type=int, default=-1)


def load_steps(filename, steps_range):
    df = pd.read_csv(filename, sep="\t")
    return df.loc[df.step.isin(steps_range)]


def get_emp_dist(node_counts: np.ndarray,
                 num_nodes: int,
                 num_exps: int):
    node_counts = np.concatenate([node_counts, np.zeros(num_nodes - len(node_counts))])
    assert len(node_counts) == num_nodes
    ideal = np.ones(num_nodes) / num_nodes
    empirical = node_counts / num_exps
    return np.linalg.norm(empirical - ideal, 1) / 2.


def bincount(node_counts: np.ndarray,
             total_num_nodes: int):
    bc = np.bincount(node_counts).tolist()
    assert bc[0] == 0
    bc[0] = total_num_nodes - sum(bc)
    return bc


if __name__ == "__main__":
    args = parser.parse_args()
    parent_dir = Path.cwd()
    results_dir = args.results_dir
    info_path = args.info_path
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = results_dir
    with_emp = args.with_emp

    full_info_path = parent_dir / info_path
    print(f"Loaded metadata json file from {full_info_path}")
    info = json.load(open(full_info_path))
    cur_graph = info["title"]
    method = info["method"]
    exp_num = info["exp_num"]
    starting_nodes = info["starting_nodes"]
    num_intervals = info["num_intervals"]
    interval_length = info["interval_length"]
    num_nodes = all_num_nodes[cur_graph]

    sampling_results_dir = parent_dir / results_dir
    print(f"Loading raw sampling results from {sampling_results_dir}...")
    init_nodes = set(
        [str(s).split("init_node_")[1].split(".tsv")[0] for s in list(sampling_results_dir.glob(cur_graph + "*.tsv"))])
    for init_node in init_nodes:
        print("Summarizing results for init node: ", init_node)
        all_files = list(sampling_results_dir.glob(cur_graph + f'*_{init_node}.tsv'))

        nodes_df = pd.concat([pd.read_csv(filename, sep="\t",
                                          dtype={'step': np.uint32, 'node': np.uint32, 'queries': np.uint32,
                                                 'exp': np.uint32})
                              for filename in tqdm(all_files)])
        # nodes_df = orig_nodes_df.loc[orig_nodes_df['exp'] < num_nodes]
        max_step = max(nodes_df['step'])
        upto_step = max_step if args.upto_step <= 0 else args.upto_step
        nodes_df = nodes_df[nodes_df['step'] <= upto_step]
        # if max(nodes_df.exp) > num_nodes:
        assert len(set(nodes_df.exp)) <= num_nodes + 100
        if exp_num > num_nodes:
            exp_num = num_nodes
        steps = set(nodes_df.step)
        steps = set([step for step in steps if step <= upto_step])
        df_grouped = nodes_df.groupby('step')
        del nodes_df

        print(f"Processing sampling results...")
        if with_emp:
            counted_query_means = [(step,
                                    df.queries.mean(),
                                    get_emp_dist(df.node.value_counts().values, num_nodes, len(df)),
                                    bincount(df.node.value_counts().values, num_nodes)
                                    ) for (step, df) in tqdm(df_grouped)]
        else:  # no need for get_emp_dist and bincount
            counted_query_means = [(step,
                                    df.queries.mean()
                                    ) for (step, df) in tqdm(df_grouped)]
        del df_grouped

        nodes_str = "_".join([str(node) for node in starting_nodes])
        full_output_path = parent_dir / output_dir / f'{cur_graph}-{method}-exp_num{exp_num}-num_int{num_intervals}-int_len{interval_length}-nodes{nodes_str}-init_node-{init_node}.json'
        if with_emp:
            count_histograms_df = pd.DataFrame(data=counted_query_means,
                                               columns=["steps", "queries", "emp_dist", "counts"])
        else:
            count_histograms_df = pd.DataFrame(data=counted_query_means, columns=["steps", "queries"])
        count_histograms_df.to_json(full_output_path)
        print(f"Saved compressed results pickle to {full_output_path}")
