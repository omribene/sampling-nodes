import random

import numpy as np
from autologging import logged, traced

from high_graph_preprocessing import HighGraphPreprocessing, Graph, logger
from neighbor_manager import NeighborManager


@logged
@traced
class LayerManager:
    """
    Methods for managing and reaching different layers.
    """
    g = None
    high_subgraph = None
    neighbor_manager = None
    num_L1_samples = int(1e5)
    L1_samples = None
    L1_sample_index = 0

    def __init__(self, g: Graph,
                 high_subgraph: HighGraphPreprocessing,
                 neighbor_manager: NeighborManager):
        self.g = g
        self.high_subgraph = high_subgraph
        self.neighbor_manager = neighbor_manager
        self.method = self.high_subgraph.method
        self.load_samples()

    def load_samples(self):
        if self.method is "greedy":
            self.L1_samples = np.random.choice(a=range(len(self.high_subgraph.L1_list)), p=self.high_subgraph.L1_probs,
                                               size=self.num_L1_samples)
        elif self.method is "absolute":
            self.L1_samples = np.random.choice(a=range(len(self.high_subgraph.L1_list)),
                                               p=self.high_subgraph.L1_L2_probs,
                                               size=self.num_L1_samples)
        else:
            raise NotImplementedError()
        self.L1_sample_index = 0

    def sample_from_L1(self):
        try:
            v = self.high_subgraph.L1_list[self.L1_samples[self.L1_sample_index]]
        except IndexError:
            self.load_samples()
            v = self.high_subgraph.L1_list[self.L1_samples[self.L1_sample_index]]
        self.L1_sample_index += 1
        return v

    def reach_layer(self, layer_num: int):
        assert layer_num > 1
        v = self.sample_from_L1()

        if self.method is "greedy":
            for i in range(1, layer_num):
                nbrs_above = list(self.neighbor_manager.neighbors_above(v, i))
                if not nbrs_above:
                    return None
                v = random.choice(nbrs_above)

        elif self.method is "absolute":
            for i in range(1, layer_num):
                nbrs_at_least = list(self.neighbor_manager.neighbors_above(v, i - 1))
                if not nbrs_at_least:
                    return None
                v = random.choice(nbrs_at_least)
                if self.neighbor_manager.check_layer_association(v, i):
                    return None

        else:
            raise NotImplementedError()

        return v

    def get_layers(self, max_dist: int = 10, logging=True):
        """
        return brute force partitioning to layers.
        Not part of the algorithm, used for testing purposes.
        """
        layers = [self.high_subgraph.L0_set, self.high_subgraph.L1_set]
        covered = layers[0].union(layers[1])
        uncovered = set(self.g.vs) - covered
        if logging:
            logger.info("Brute-force calculation of layer partitioning")
        for i in range(2, max_dist):
            new_layer = set()
            for v in uncovered:  # tqdm(uncovered):
                if len(set(v.neighbors()).intersection(covered)) > 0:
                    new_layer.add(v)
            covered = covered.union(new_layer)
            uncovered = uncovered - new_layer
            layers.append(new_layer)
        layers.append(uncovered)
        sizes = [len(l) for l in layers]
        if logging:
            logger.info("Obtained sizes: " + str(sizes))
        assert sum(sizes) == self.g.vcount()
        return layers
