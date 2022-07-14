import numpy as np

from previous_exp.walker import Node, RandomWalker


class MaxDegreeSampling(RandomWalker):
    def __init__(self, initial_node: Node, max_deg: int):
        super(MaxDegreeSampling, self).__init__(init_node=initial_node)
        self.max_deg = max_deg

    def random_step(self):
        v = self.current_node

        neighbors = self.neighbors(v)
        v_deg = len(neighbors)
        assert v_deg == len(
            neighbors), f"v= {v.index}. v's degree: {v.degree()}, v's neighbors: {[u.index for u in neighbors]}"

        probs = np.ones(v_deg) / self.max_deg
        complement = 1 - v_deg / self.max_deg

        probs = np.append(probs, complement)
        self.current_node = np.random.choice(neighbors + [v], p=probs)

    def get_node(self):
        return self.current_node
