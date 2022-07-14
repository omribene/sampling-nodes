import logging
from time import time
from typing import NewType, Set, List, Tuple

import igraph as ig
import numpy as np
from autologging import logged, traced
from tqdm.auto import tqdm

from src.utils.graph_utils import GraphUtils as gu
from src.utils.globals import *

logger = logging.getLogger('LayerwiseSample logger')
# logger.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


Graph = NewType("Graph", ig.GraphBase)
Node = NewType("Node", ig.Vertex)
NodesSet = NewType("NodesSet", Set[ig.Vertex])


@logged
@traced
class HighGraphPreprocessing(object):
    """
    Methods for constructing L0, both for SampLayer ("greedy" methods) and SampLayer+ ("absolute" methods).
    """
    def __init__(self, g: Graph):
        self.g = g
        self.vcount = g.vcount()
        self.L0_list = None
        self.L0_set = None
        self.L1_list = None
        self.L1_set = None
        self.L1_degrees = None
        self.L1_probs = None
        self.method = None
        self.first_run = True

    def set_method(self, method):
        # currently defaults to "greedy_high"
        self.method = method

    def initialize_params(self, init_node: Node):
        if self.method == "greedy":
            self.greedy_high_init(init_node)
        elif self.method == "absolute":
            self.absolute_high_init(init_node)
        else:
            raise NotImplementedError("L0 generation method " +
                                      str(self.method) +
                                      " does not exist.")

    def run(self, num_nodes: int):
        if self.first_run:
            num_nodes -= 1
            self.first_run = False
        if self.method == "greedy":
            self.greedy_high(num_nodes)
        elif self.method == "absolute":
            self.absolute_high_run(num_nodes)
        else:
            raise NotImplementedError("L0 generation method " +
                                      str(self.method) +
                                      " does not exist.")

    def get_high_nodes(self, num_nodes: int, init_node: Node):
        self.initialize_params(init_node)
        self.run(num_nodes)

    def greedy_high_init(self, init_node: Node):
        self.__log.info(f"Initializing L0 and L1 with method {self.method}. Init node: {init_node}")
        self.unchosen = np.ones(self.vcount, dtype=np.bool)
        self.candidates = np.zeros(self.vcount, dtype=np.uint32)
        self.starting_idx = init_node.index
        self.unchosen[self.starting_idx] = 0
        neigh_indices = [neigh.index for neigh in init_node.neighbors() if neigh != init_node]
        self.candidates[neigh_indices] += 1

    def sample_high_batch(self, batch_size: int = 1):
        if batch_size == 1:
            return np.random.choice(np.flatnonzero(self.candidates == self.candidates.max()), size=1)
        # want to find the "threshold value" for the batch
        quantile = np.quantile(self.candidates, 1 - batch_size / self.vcount)
        large_locs = np.flatnonzero(self.candidates >= quantile)
        large_vals = self.candidates[large_locs]

        indices_above_thres = large_locs[large_vals > quantile]
        indices_at_thres = large_locs[large_vals == quantile]
        sample_size = batch_size - len(indices_above_thres)
        sampled_indices = np.random.choice(indices_at_thres, size=sample_size, replace=False)
        return np.concatenate([indices_above_thres, sampled_indices])

    def greedy_high_run(self, num_iters: int, batch_size: int = 1):
        for _ in tqdm(range(num_iters)):
            curr_indices = self.sample_high_batch(batch_size)
            self.unchosen[curr_indices] = False
            self.candidates[curr_indices] = 0
            chosen = self.g.vs[list(curr_indices)]
            for curr_node in chosen:
                neigh_indices = [neigh.index for neigh in curr_node.neighbors() if self.unchosen[neigh.index]]
                self.candidates[neigh_indices] += 1

        self.greedy_high_update()

    def greedy_high_update(self):
        l0_indices = np.flatnonzero(self.unchosen == 0)
        self.L0_list = [self.g.vs[idx] for idx in l0_indices]
        self.L0_set = set(self.L0_list)
        self.L1_dict = {int(l0_indices[i]): i for i in range(len(l0_indices))}
        l1_indices = np.flatnonzero(self.candidates > 0)
        self.L1_list = [self.g.vs[idx] for idx in l1_indices]
        self.L1_set = set(self.L1_list)
        self.L1_dict = {int(l1_indices[i]): i for i in range(len(l1_indices))}
        total_n_edges = np.sum(self.candidates)
        self.L1_degrees = [int(self.candidates[i]) for i in l1_indices]
        self.L1_probs = [float(self.candidates[i]) / total_n_edges for i in l1_indices]
        assert abs(sum(self.L1_probs) - 1) < 0.00001
        assert len(self.L0_set.intersection(self.L1_set)) == 0
        for v in self.L1_set:
            assert len(set(v.neighbors()) - self.L0_set) < len(set(v.neighbors()))

    def greedy_high(self, num_nodes: int):
        self.greedy_high_run(num_nodes - 1, batch_size=1)

    def absolute_high_init(self, init_node: Node):
        self.__log.info(f"Initializing L0 and L1 with method {self.method}...")
        self.unvisited = np.ones(self.g.vcount(), dtype=np.bool)
        self.candidates = np.zeros(self.g.vcount(), dtype=np.uint32)
        self.deg_to_L0 = np.zeros(self.g.vcount(), dtype=np.uint32)
        self.degrees = np.array(self.g.degree(range(self.g.vcount()), loops=False), dtype=np.uint32)
        self.starting_idx = init_node.index
        self.unvisited[self.starting_idx] = False
        neigh_indices = [neigh.index for neigh in init_node.neighbors() if neigh != init_node]
        self.candidates[neigh_indices] = self.degrees[neigh_indices] * self.unvisited[neigh_indices]
        self.deg_to_L0[neigh_indices] = 1

    def absolute_high_run(self, num_nodes: int):
        for _ in tqdm(range(1, num_nodes)):
            curr_idx = np.random.choice(np.flatnonzero(self.candidates == self.candidates.max()))
            self.unvisited[curr_idx] = False
            self.candidates[curr_idx] = self.deg_to_L0[curr_idx] = 0
            curr_node = self.g.vs[int(curr_idx)]
            neigh_indices = [neigh.index for neigh in curr_node.neighbors() if self.unvisited[neigh.index]]
            self.candidates[neigh_indices] = self.degrees[neigh_indices]
            self.deg_to_L0[neigh_indices] += 1

        self.absolute_high_update()

    def absolute_high_update(self):
        l0_indices = np.flatnonzero(self.unvisited == 0)
        self.L0_list = [self.g.vs[idx] for idx in l0_indices]
        self.L0_set = set(self.L0_list)
        self.L1_dict = {int(l0_indices[i]): i for i in range(len(l0_indices))}
        l1_indices = np.flatnonzero(self.candidates > 0)
        self.L1_list = [self.g.vs[idx] for idx in l1_indices]
        self.L1_set = set(self.L1_list)
        self.L1_dict = {int(l1_indices[i]): i for i in range(len(l1_indices))}
        self.L1_degrees = [int(self.deg_to_L0[i]) for i in l1_indices]
        total_n_L0_L1_edges = np.sum(self.deg_to_L0)
        self.L1_probs = [float(self.deg_to_L0[i]) / total_n_L0_L1_edges for i in l1_indices]
        total_n_L1_L2_edges = np.sum(self.degrees[self.candidates > 0]) - np.sum(self.deg_to_L0)
        self.L1_L2_probs = [(self.degrees[i].astype(float) - self.deg_to_L0[i]) / total_n_L1_L2_edges for i in
                            l1_indices]
        assert abs(sum(self.L1_probs) - 1) < 0.00001
        assert len(self.L0_set.intersection(self.L1_set)) == 0
        for v in self.L1_set:
            assert len(set(v.neighbors()) - self.L0_set) < len(set(v.neighbors()))

    def absolute_high(self, num_nodes: int, init_node: Node):
        self.absolute_high_init(init_node)
        self.absolute_high_run(num_nodes - 1)


if __name__ == "__main__":
    for dataset in datasets[0:1]:
        path, sep, directed, title = dataset.path, dataset.sep, dataset.directed, dataset.title
        print("Reading graph: ", title)
        g, in_degrees = gu.load_graph(path, directed=directed)
        start = time()
        high_graph_loader = HighGraphPreprocessing(g)
        high_graph_loader.set_method("greedy")
        high_nodes = high_graph_loader.get_high_nodes(500, np.random.choice(g.vs))
        print(f"finished. Elapsed time: {time() - start} seconds")
        print(f"obtained {len(high_graph_loader.L0_set)} nodes in L0, and {len(high_graph_loader.L1_set)} in L1")
        print("degrees:", np.sort(g.degree(high_graph_loader.L0_list))[::-1])
