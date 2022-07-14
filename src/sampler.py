import random
from typing import NewType, Set

import igraph as ig
import numpy as np
from tqdm import tqdm

from component_collector import ComponentCollector
from src.utils.graph_utils import GraphUtils as gu
from high_graph_preprocessing import HighGraphPreprocessing
from layer_manager import LayerManager
from neighbor_manager import NeighborManager
from reachability_estimator import ReachabilityEstimator
from size_estimation import SizeEstimation
from statistics import Statistics
from src.utils.globals import datasets, nums_L0

Graph = NewType("Graph", ig.GraphBase)
Node = NewType("Node", ig.Vertex)
NodesSet = NewType("NodesSet", Set[ig.Vertex])


class Sampler:
    def __init__(self, dataset_name=None, dataset_idx=None, plus=False):
        """
        Constructor for the sampler object, which includes the main algorithmic components.

        :param dataset_name (str, optional): should match one of the datasets in `globals.py`.
        Note: the dataset should be downloaded from SNAP website: http://snap.stanford.edu/data/index.html
        :param dataset_idx: needs to be specified if dataset_name is None. See `datasets` object in `globals.py`.
        :param plus: If false, runs "standard" SampLayer. If true, runs SampLayer "plus"; see paper for details.
        """

        assert (dataset_idx is not None) != (dataset_name is not None)

        if dataset_idx is not None:
            dataset_name = datasets[dataset_idx].title
            print(dataset_name)

        dataset = None
        for ds in datasets:
            if dataset_name == ds.title:
                dataset = ds
                break
        if dataset is None:
            raise ValueError("No dataset with this name")

        self.title = dataset_name
        self.dataset = dataset
        path, sep, directed = dataset.path, dataset.sep, dataset.directed
        self.layer_num = 2

        print("Reading graph:", self.title, "from", path)
        g, in_degrees = gu.load_graph(path, directed=directed)
        g.to_undirected()
        print("Taking giant component...")
        g = g.components().giant()
        # the actual size will not be used at any point during the algorithm
        # (used only for evaluation).
        self.graph = g
        self.graph_size = g.vcount()
        print("Number of nodes:", self.graph_size)

        # initializing components of algorithm
        self.initialized = True
        self.all_L2_reachabilities = None
        self.high_subgraph = None
        self.L0_size = None
        self.L0_generated = False
        self.L1_size = None
        self.L0_L1_size = None
        self.actual_L2_size = None
        self.neighbor_manager = None
        self.layer_manager = None
        self.layers = None
        self.component_collector = None
        self.reachability_estimator = None
        self.size_estimator = None
        self.statistics = None
        self.up_nodes = None
        self.initial_query_counter = None
        self.reached_nodes = []
        self.node_probs = []
        self.reachabilities = []
        self.query_counters = []
        self.is_accepted = []
        self.L2_preprocessed = False
        self.allowed_error = None

        # determine L0 construction procedure: absolute under the stronger query model (plus = True),
        # greedy under weaker one.
        if plus:
            self.method = "absolute"
        else:
            self.method = "greedy"



    def generate_L0(self, L0_size: int = -1, preloaded=None):
        """
        Runs L0 construction phase (strategy depends on whether it is SampLayer or SampLayer "plus").
        Initializes objects for other parts of the graph.
        :param L0_size size of L0 to construct, if -1 then takes default from src/utils/globals.py
        """
        assert self.initialized

        self.high_subgraph = HighGraphPreprocessing(self.graph)
        self.high_subgraph.set_method(self.method)
        if L0_size <= 0:
            L0_size = nums_L0[self.title]
        self.high_subgraph.get_high_nodes(L0_size, np.random.choice(self.graph.vs))

        self.L0_size = L0_size

        self.L0_generated = True
        self.L1_size = len(self.high_subgraph.L1_set)
        self.L0_L1_size = self.L0_size + self.L1_size
        # note the the actual L2_size will not be used anywhere in the algorithm (it
        # is only used for evaluation purposes)
        self.actual_L2_size = self.graph_size - self.L0_L1_size
        self.neighbor_manager = NeighborManager(self.high_subgraph)
        print("Built L0. Total queries:", self.neighbor_manager.query_counter)

        self.layer_manager = LayerManager(self.graph,
                                          self.high_subgraph,
                                          self.neighbor_manager)
        self.layers = self.layer_manager.get_layers()
        self.component_collector = ComponentCollector(self.graph,
                                                      self.high_subgraph,
                                                      self.neighbor_manager,
                                                      self.layer_manager,
                                                      self.layer_num,
                                                      with_in_layer_edges=False)
        self.reachability_estimator = ReachabilityEstimator(self.high_subgraph,
                                                            self.neighbor_manager,
                                                            self.layer_manager,
                                                            self.component_collector,
                                                            self.layer_num)
        self.size_estimator = SizeEstimation(self.high_subgraph,
                                             self.neighbor_manager,
                                             self.layer_manager,
                                             self.component_collector,
                                             self.reachability_estimator)
        self.statistics = Statistics(self.graph,
                                     self.high_subgraph,
                                     self.neighbor_manager,
                                     self.layer_manager,
                                     self.reachability_estimator,
                                     self.component_collector)

    def preprocess_L2(self, L1_num_samples: int = 100, L2_num_reaches: int = 10, allowed_error=0.05):
        """
        Preprocessing phase: estimating size and reachability of L2.

        :param L1_num_samples: How many (uniform) nodes to sample from L1 in preprocessing.
        :param L2_num_reaches: How many (biased) nodes to sample from L2 in preprocessing.
        :param allowed_error: AKA eps in paper, the distance from uniformity that we allow our alg to have.
        """
        assert self.L0_generated

        self.allowed_error = allowed_error

        nodes_from_L1 = random.sample(self.high_subgraph.L1_list, L1_num_samples)
        self.up_nodes = set(nodes_from_L1)
        self.size_estimator.update_up(nodes_from_L1)
        self.initial_query_counter = self.neighbor_manager.query_counter

        print("Sampled", L1_num_samples, "nodes from L1. Total queries:",
              self.initial_query_counter)

        for _ in range(L2_num_reaches):
            self.L2_reach_step(preprocessing=True)

        self.update_estimates()
        for i in range(len(self.reached_nodes)):
            self.is_accepted[i] = self.is_reached_node_accepted(self.node_probs[i], self.reachabilities[i])

        print("Sampled", L2_num_reaches, "nodes from L2+ without rejection. Total queries:",
              self.neighbor_manager.query_counter)

        self.L2_preprocessed = True

    def L2_reach_step(self, with_updating=True, preprocessing=False):
        """

        :param with_updating: should new generated samples update our L2 size/reachability estimation?
        :param preprocessing: are we during preprocessing right now?
        :return: node from L2 if rejection sampling happened and succeeded, None otherwise
        """
        update_flag = with_updating
        next_node_lst = self.component_collector.sample_component_nodes_no_rejection(1, False)
        self.reached_nodes += next_node_lst
        if update_flag:
            self.size_estimator.update_down(next_node_lst)
        node_reachability = self.reachability_estimator.get_reachability(next_node_lst[0],
                                                                         layer_num=self.layer_num)
        self.reachabilities.append(node_reachability)
        if update_flag:
            self.reachability_estimator.update_observed_reachabilities([node_reachability])
        node_prob = random.random()
        self.node_probs.append(node_prob)
        self.query_counters.append(self.neighbor_manager.query_counter)

        if preprocessing:
            self.is_accepted.append(None)
        else:
            acceptance = self.is_reached_node_accepted(node_prob, node_reachability)
            self.is_accepted.append(acceptance)
            if acceptance:
                return next_node_lst[0]
            else:
                return None

    def is_reached_node_accepted(self, node_prob, node_reachability):
        """
        coin flip for rejection sampling
        """
        return node_prob < self.baseline_reach / node_reachability

    def update_estimates(self):
        self.L2_size_estimation = self.size_estimator.estimate_size(self.L1_size)
        self.graph_size_estimation = self.L2_size_estimation + self.L0_L1_size
        self.estimated_fractions = np.array([self.L0_size, self.L1_size, self.L2_size_estimation],
                                            dtype=np.float) / self.graph_size_estimation
        self.reach_quantile = self.allowed_error / self.estimated_fractions[2]
        self.baseline_reach = self.reachability_estimator.estimate_baseline_reachability(self.reach_quantile)

    def sample(self,
               num_samples: int = 1,
               num_additional_L1_samples: int = 1,
               with_tqdm: bool = False):
        """
        The main sampling procedure, must take place only after preprocessing is complete.

        :param num_samples: number of node samples to generate.
        :param num_additional_L1_samples: Number of additional L1 samples to generate
        per node sample in order to improve size/reachability estimates.
        :return: generated samples, how many queries so far (for each sample)
        """

        assert self.L2_preprocessed

        samples = []
        query_counters = []
        layer_numbers = range(3)
        layer_choices = np.random.choice(layer_numbers, num_samples, p=self.estimated_fractions)

        rng = range(num_samples)
        if with_tqdm:
            rng = tqdm(rng)
        for i in rng:
            if layer_choices[i] == 0:
                samples.append(random.choice(self.high_subgraph.L0_list))
            elif layer_choices[i] == 1:
                samples.append(random.choice(self.high_subgraph.L1_list))
            else:
                samples.append(self.sample_from_components())
            query_counters.append(self.neighbor_manager.query_counter)

        new_L1_samples = [self.sample_new_L1_node() for _ in range(num_additional_L1_samples)]
        new_L1_samples = [samp for samp in new_L1_samples if samp is not None]
        if len(new_L1_samples) > 0:
            self.size_estimator.update_up(new_L1_samples)

        self.update_estimates()

        if num_samples == 1:
            return samples[0], query_counters[0]
        else:
            return samples, query_counters

    def sample_from_components(self):
        """
        Sampling specifically from L2.
        """
        node = None
        while node is None:
            node = self.L2_reach_step()
        return node

    def sample_new_L1_node(self):
        """
        Sample new node from L1, ensure that it was not sampled already.
        """
        if len(self.up_nodes) >= self.L1_size:
            return None
        while True:
            node = random.choice(self.high_subgraph.L1_list)
            if node not in self.up_nodes:
                self.up_nodes.add(node)
                return node
