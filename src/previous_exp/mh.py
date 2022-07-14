import numpy as np

from previous_exp.walker import Node, RandomWalker


class MetropolisHastingsSampling(RandomWalker):
    def __init__(self, initial_node: Node, min_deg: int, plus: bool = False):
        marking = not plus
        super(MetropolisHastingsSampling, self).__init__(init_node=initial_node, marking=marking)
        self.min_deg = min_deg
        self.rand_state = np.random.get_state()
        self.plus = plus
        self._calc_wait_time()

    def degree(self, node: Node):
        if not self.plus:
            self.mark(node)
        return node.degree()

    def _calc_wait_time(self):
        # improve running time by optimizing number of coin flips
        v = self.current_node
        neighbors = self.neighbors(v)
        v_deg = len(neighbors)
        neighbor_degs = np.array([self.degree(u) for u in neighbors], dtype=np.int32)
        max_degs = np.maximum(neighbor_degs, v_deg)
        neighbor_probs = 1. / max_degs
        total_prob = sum(neighbor_probs)

        adjusted_probs = neighbor_probs / total_prob
        # fix rare numerical issues
        if sum(adjusted_probs) != 1:
            adjusted_probs[-1] = 1 - sum(adjusted_probs[:-1])
        total_prob = min(total_prob, 1.0)

        self.next_node = np.random.choice(neighbors, p=adjusted_probs)
        self.wait_time = np.random.geometric(p=total_prob)

    def random_step(self):
        self.wait_time -= 1
        if self.wait_time == 0:
            self.current_node = self.next_node
            self._calc_wait_time()

    def get_node(self):
        return self.current_node
