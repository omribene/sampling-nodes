import random

import numpy as np
from tqdm.auto import tqdm

from high_graph_preprocessing import HighGraphPreprocessing, Node, Graph, logger
from layer_manager import LayerManager
from neighbor_manager import NeighborManager


class ComponentCollector:
    """
    Methods for computing, reaching, and managing components in L2.
    """
    g = None
    high_subgraph = None
    neighbor_manager = None
    layer_manager = None
    component_layer_num = 2
    with_in_layer_edges = False

    def __init__(self, g: Graph,
                 high_subgraph: HighGraphPreprocessing,
                 neighbor_manager: NeighborManager,
                 layer_manager: LayerManager,
                 component_layer_num: int = 2,
                 with_in_layer_edges: bool = False,
                 extract_all_components=False):
        self.g = g
        self.high_subgraph = high_subgraph
        self.neighbor_manager = neighbor_manager
        self.layer_manager = layer_manager
        self.component_layer_num = component_layer_num
        self.with_in_layer_edges = with_in_layer_edges
        if extract_all_components:
            self.components = self.extract_all_connected_components()
            self.vtx_to_comp = {v: comp for comp in self.components for v in comp}
            self.visited_comps = set()
        else:
            self.components = {}
            self.vtx_to_comp = None


    def component_bfs(self, node: Node):
        """
        Given a node in L2, returns the whole L2 component node belongs to.
        """
        if self.vtx_to_comp:
            comp = set(self.vtx_to_comp[node])
            if comp not in self.visited_comps:
                self.visited_comps.add(comp)
                for v in comp:
                    self.neighbor_manager.neighbors(v)
            return comp
        node_list = [node]
        node_set = set(node_list)
        length = 1
        curr_idx = 0
        while curr_idx < length:
            v = node_list[curr_idx]
            curr_idx += 1
            neighbors_in_comp = self.neighbors_in_component(v)
            for u in neighbors_in_comp:
                if u not in node_set:
                    node_list.append(u)
                    node_set.add(u)
                    length += 1
        return node_set

    def neighbors_in_component(self, node: Node):
        node_layer = self.neighbor_manager.get_layer_num(node, self.component_layer_num + 1)
        assert node_layer >= self.component_layer_num
        if node_layer == self.component_layer_num:
            return self.neighbor_manager.neighbors_above(node, self.component_layer_num)
        else:
            return self.neighbor_manager.neighbors(node)

    def extract_all_connected_components(self, logging=True, return_sizes=False):
        """
        For testing purposes: returns all components in L2 (not part of alg).
        """
        if logging:
            print("Extracting connected components...")

        recording = self.neighbor_manager.is_recording()
        if recording:
            # This method is only used for debugging / running time optimization purposes,
            # and is not actually used by our algorithm. Therefore, it is important to "turn off"
            # query counting while it runs.
            self.neighbor_manager.stop_recording()
        layers = self.layer_manager.get_layers(logging=False)
        nodes = list(np.concatenate([[x for x in layer] \
                                     for layer in layers[self.component_layer_num:]]))
        name_to_node = {node["name"]: node for node in nodes}
        assert len(nodes) == len(name_to_node)

        subgraph = self.g.induced_subgraph(nodes)
        if not self.with_in_layer_edges:
            tuples = [e.tuple for e in subgraph.es]
            to_del = [(u, v) for (u, v) in tuples if \
                      name_to_node[subgraph.vs[u]["name"]] in layers[self.component_layer_num] and
                      name_to_node[subgraph.vs[v]["name"]] in layers[self.component_layer_num]]
            subgraph.delete_edges(to_del)

        subgraph_components = subgraph.components()
        sizes = [len(comp) for comp in subgraph_components]
        sizes.sort(reverse=True)
        logger.info("largest component sizes: " + str(sizes[:30]))

        if not return_sizes:
            original_components = []
            for subg_comp in subgraph_components:
                orig_comp = [name_to_node[subgraph.vs[v]["name"]] for v in subg_comp]
                original_components.append(orig_comp)

        if recording:
            self.neighbor_manager.resume_recording()

        if return_sizes:
            return sizes
        else:
            return original_components

    def reach_component(self):
        node = self.layer_manager.reach_layer(self.component_layer_num)
        if node is None:
            return None
        component = self.component_bfs(node)
        return component

    def get_node_in_component(self):
        component = None
        while component is None:
            component = self.reach_component()
        return random.choice(list(component))

    def sample_components_no_rejection(self, num_samples: int, logging=True):
        components = []
        if logging:
            logger.info("Sampling %d components (Without rejection)" % (num_samples,))
            rng = tqdm(range(num_samples))
        else:
            rng = range(num_samples)
        for _ in rng:
            comp = None
            while comp is None:
                comp = self.reach_component()
            components.append(comp)
        return components

    def sample_component_nodes_no_rejection(self, num_samples: int, logging=True):
        components = self.sample_components_no_rejection(num_samples, logging)
        nodes = [random.choice(list(comp)) for comp in components]
        return nodes

    def num_neighbors_of_component(self, component):
        return sum([len(self.neighbor_manager.neighbors_in_layer(v, self.component_layer_num - 1)) \
                    for v in component])
