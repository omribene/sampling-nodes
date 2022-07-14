from typing import NewType

import igraph as ig
import numpy as np

from high_graph_preprocessing import HighGraphPreprocessing

Node = NewType("Node", ig.Vertex)


class NeighborManager:
    """
    Various methods to compute neighbors in the same layer / layer above or below and the like.
    Makes sure to count queries made properly.
    """
    high_subgraph = None
    query_counter = 0
    memory = None
    recording = True

    def __init__(self, high_subgraph: HighGraphPreprocessing):
        self.high_subgraph = high_subgraph
        self.flush()

    def flush(self):
        self.query_counter = len(self.high_subgraph.L0_set)
        self.memory = {v: v.neighbors() for v in self.high_subgraph.L0_set}

    def copy(self):
        neighbor_manager = NeighborManager(self.high_subgraph)
        neighbor_manager.memory = dict(self.memory)
        neighbor_manager.query_counter = self.query_counter
        return neighbor_manager

    def neighbors(self, v: Node):
        if not self.recording:
            return set(v.neighbors()) - set([v])
        if v not in self.memory:
            self.memory[v] = set(v.neighbors()) - set([v])
            self.query_counter += 1
        return self.memory[v]

    def check_layer_association(self, v: Node, layer_num: int):
        if layer_num == 0:
            return v in self.high_subgraph.L0_set
        if layer_num == 1:
            return v in self.high_subgraph.L1_set
        return bool(np.any([self.check_layer_association(u, layer_num - 1) for u in self.neighbors(v)]))

    def get_layer_num(self, v: Node, max_layer: int):
        for i in range(max_layer):
            if self.check_layer_association(v, i):
                return i
        return max_layer

    def neighbors_in_layer(self, v: Node, layer_num: int):
        return set([u for u in self.neighbors(v) if self.check_layer_association(u, layer_num)])

    def neighbors_up_to(self, v: Node, layer_num: int):
        return set.union(*[self.neighbors_in_layer(v, i) for i in range(layer_num + 1)])

    def neighbors_above(self, v: Node, layer_num: int):
        return self.neighbors(v) - self.neighbors_up_to(v, layer_num)

    def stop_recording(self):
        self.recording = False

    def resume_recording(self):
        self.recording = True

    def is_recording(self):
        return self.recording
