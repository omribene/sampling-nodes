import logging
import os
import sys
from typing import NewType

import igraph as ig

sys.path.insert(0, os.getcwd())
print("current directory: ", os.getcwd())
print('loaded globals')

Graph = NewType("Graph", ig.GraphBase)
Node = NewType("Node", ig.Vertex)

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')


class RandomWalker(object):
    def __init__(self, init_node: Node, marking=False):
        self.current_node = init_node
        self.neighbors_dict = {}
        self.query_count = 0
        self.marking = marking
        if marking:
            self.marked = set()

    def mark(self, node: Node):
        if node not in self.marked:
            self.query_count += 1
            self.marked.add(node)

    def neighbors(self, node: Node):
        if node not in self.neighbors_dict:
            neighbors = set(node.neighbors())
            neighbors = list(neighbors - set([node]))
            self.neighbors_dict[node] = neighbors
            assert node not in self.neighbors_dict[node]
            if self.marking:
                self.mark(node)
            else:
                self.query_count += 1
        return self.neighbors_dict[node]

    def random_step(self):
        pass

    def random_walk(self, steps: int = 1000):
        logging.info("Taking a random walk of length " + str(steps))
        for _ in range(steps):
            self.random_step()

    def get_node(self):
        pass
