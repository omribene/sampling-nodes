import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from sampler import Sampler
from src.utils.globals import nums_L0, nums_L0_plus, nums_L1_up, nums_L1_up_plus, nums_L2_reaches, nums_L2_reaches_plus

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--config', default=None,
                    help="An experiment configuration file. If specified, overrides the other argument parameter.")
parser.add_argument('--output_path', default=None)
parser.add_argument('--plus', action='store_true', help="use SampLayer+ ? otherwise (by default) uses SampLayer")
parser.add_argument('--dataset_idx', type=int, help="see globals.py")
parser.add_argument('--eps', type=float, default=0.05, help="target distance from uniformity")
parser.add_argument('--L0_size', type=int, default=-1, help="Number of nodes queried in L0")
parser.add_argument('--L1_num_initial_samples', type=int, default=-1,
                    help="Number of nodes from L1 for L2-size estimation")
parser.add_argument('--L2_num_initial_reaches', type=int, default=-1,
                    help="Number of reached nodes in L2 for L2-size estimation")
parser.add_argument('--num_samples', type=int, help="Number of node samples to generate")
parser.add_argument('--num_additional_L1_nodes_per_sample', type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()

    # reads parameters into `conf`, either from argument line or from config file
    if args.config is not None:
        p = Path(args.config)
        with p.open('r') as f:
            conf = yaml.full_load(f)
    else:
        conf = {}
        for arg in vars(args):
            conf[arg] = getattr(args, arg)

    print(f'Running Samplayer with the following arguments: {conf}')

    # preprocessing: setting up
    sampler = Sampler(dataset_idx=conf['dataset_idx'], plus=conf['plus'])
    dataset_name = sampler.title

    L0_size = conf['L0_size'] if conf['L0_size'] > 0 else (
        nums_L0_plus[dataset_name] if conf['plus'] else nums_L0[dataset_name])
    L1_num_initial_samples = conf['L1_num_initial_samples'] if conf['L1_num_initial_samples'] > 0 else \
        (nums_L1_up_plus[dataset_name] if conf['plus'] else nums_L1_up[dataset_name])
    L2_num_initial_reaches = conf['L2_num_initial_reaches'] if conf['L2_num_initial_reaches'] > 0 else \
        (nums_L2_reaches_plus[dataset_name] if conf['plus'] else nums_L2_reaches[dataset_name])

    # preprocessing: building L0
    logging.info("Generating L0...")
    sampler.generate_L0(L0_size=L0_size)

    # preprocessing: estimating size and reachability in L2
    logging.info("Completing preprocessing...")
    sampler.preprocess_L2(L1_num_samples=L1_num_initial_samples,
                          L2_num_reaches=L2_num_initial_reaches,
                          allowed_error=conf['eps'])

    # sampling
    logging.info("Sampling...")
    node_samples, query_counts = sampler.sample(num_samples=conf['num_samples'],
                                                num_additional_L1_samples=conf['num_additional_L1_nodes_per_sample'],
                                                with_tqdm=True)

    # aggregating and outputting results
    indices = [str(node.index) for node in node_samples]
    results_df = pd.DataFrame(data={'node': indices, 'queries': query_counts})
    if conf['output_path'] is None:
        print("Sampled nodes:")
        print(results_df)
    else:
        results_df.to_csv(conf['output_path'], sep='\t', index=False)
        print("Sampled nodes. Total queries:", query_counts[-1], "\nOutput located at:", Path.cwd() / conf["output_path"])