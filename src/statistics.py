import random
from typing import NewType

import igraph as ig
import numpy as np

from component_collector import ComponentCollector
from high_graph_preprocessing import HighGraphPreprocessing
from layer_manager import LayerManager
from neighbor_manager import NeighborManager
from reachability_estimator import ReachabilityEstimator

Graph = NewType("Graph", ig.GraphBase)
from tqdm import tqdm


class Statistics:
    """
    Computes various statistics about the sizes, reachabilities, degrees
    of components in our structural decomposition.
    """
    g = None
    high_subgraph = None
    reachability_estimator = None

    def __init__(self, g: Graph,
                 high_subgraph: HighGraphPreprocessing,
                 neighbor_manager: NeighborManager,
                 layer_manager: LayerManager,
                 reachability_estimator: ReachabilityEstimator,
                 component_collector: ComponentCollector):
        self.g = g
        self.high_subgraph = high_subgraph
        self.neighbor_manager = neighbor_manager
        self.layer_manager = layer_manager
        self.reachability_estimator = reachability_estimator
        self.component_collector = component_collector

    def component_reachability_stats(self, components: list, with_tqdm=False):
        """
        returns reachability scores of all components.
        """
        scores = []
        if with_tqdm:
            components = tqdm(components)
        for comp in components:
            reach = self.reachability_estimator.component_reachability(comp)
            for _ in range(len(comp)):
                scores.append(reach)
        return scores

    def all_component_nodes_and_reachabilities(self):
        """
        returns the list of all components and nodes.
        """
        # This method is never used as part of the algorithm (only for evaluative purposes in our code),
        # so we make sure that the visited nodes do not count toward our query budgets.
        self.neighbor_manager.stop_recording()
        components = self.component_collector.extract_all_connected_components()
        nodes = [v for comp in components for v in comp]
        reachabilities = self.component_reachability_stats(components)
        assert len(nodes) == len(reachabilities)
        self.neighbor_manager.resume_recording()
        return nodes, reachabilities

    def quantiles(self, vals: list, start: float = 0.02, end: float = 0.99, jump: float = 0.02):
        val = list(sorted(vals))
        start = int(start * len(val))
        end = int(end * len(val))
        jump = int(jump * len(val))
        return val[start:end:jump]

    def avg_degrees(self, comp_layer=2):
        """
        Returns avg degree of L1 nodes in the bipartite subgraph with L2, and avg degree of L2 nodes in
        the same bipartite graph.
        """
        # This method is never used as part of the algorithm (only for evaluative purposes in our code),
        # so we make sure that the visited nodes do not count toward our query budgets.
        self.neighbor_manager.stop_recording()
        layers = self.layer_manager.get_layers(comp_layer, logging=True)
        num_edges = sum([len(self.neighbor_manager.neighbors_above(v, comp_layer - 1))
                         for v in layers[comp_layer - 1]])
        self.neighbor_manager.resume_recording()
        return (num_edges / len(layers[comp_layer - 1]), num_edges / len(layers[comp_layer]))

    def check_TV_dist_from_uniformity(self,
                                      L0_size,
                                      L1_size,
                                      actual_L2_size,
                                      estimated_L2_size,
                                      L2_reachabilities,
                                      baseline_reachability,
                                      empirical=False,
                                      num_exps_empirical=None):
        """
        Given a choice of baseline reachability, calculates how close our method is
        (in total variation distance) to producing uniformly radnom nodes.
        """
        actual_total_size = L0_size + L1_size + actual_L2_size
        estimated_total_size = L0_size + L1_size + estimated_L2_size

        estimated_L2_frac = estimated_L2_size / estimated_total_size

        # This is the contribution of layers zero and one
        dist = (L0_size + L1_size) * abs(1. / estimated_total_size - 1. / actual_total_size)

        # Now to the contribution of layer two.
        L2_reachabilities = np.array(L2_reachabilities)
        normed_reachabilities = L2_reachabilities / baseline_reachability
        reaches_after_rejection = np.minimum(normed_reachabilities, 1.)
        sum_reaches = np.sum(reaches_after_rejection)

        relative_reaches = reaches_after_rejection / sum_reaches
        L2_probs = estimated_L2_frac * relative_reaches
        sum_distance_of_component_nodes = float(np.sum(np.abs(L2_probs - 1. / actual_total_size)))
        dist += sum_distance_of_component_nodes
        if not empirical:
            return dist

        if num_exps_empirical is None:
            num_exps_empirical = actual_total_size
        prob_vec = np.concatenate((np.ones(L0_size + L1_size) * 1. / estimated_total_size, L2_probs))
        samples = random.choices(range(actual_total_size), weights=prob_vec.tolist(), k=num_exps_empirical)
        true_emp_dist = self._emp_dist(samples, actual_total_size)
        return true_emp_dist

    @staticmethod
    def _emp_dist(samples, num_nodes):
        """
        Given a collection of samples (nodes in the graph) and the total number of nodes,
        returns empirical distance of sampled nodes from uniformity.
        """
        hist = np.histogram(samples, bins=num_nodes, range=(0, num_nodes))
        hmax = int(max(hist[0]))
        histhist = np.histogram(hist[0], bins=hmax + 1, range=(0, hmax + 1), density=True)
        mean_emp = float(len(samples)) / num_nodes
        emp_dist = sum([abs(i - mean_emp) * histhist[0][i] for i in range(hmax + 1)]) / 2.
        return emp_dist
