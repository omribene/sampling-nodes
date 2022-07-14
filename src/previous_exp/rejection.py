import argparse
import os
import sys
from random import choice, random, randint

sys.path.insert(0, os.getcwd())

from previous_exp.walker import Node, RandomWalker

parser = argparse.ArgumentParser()

parser.add_argument('--output_path', default=None)
parser.add_argument('--init_steps', default=0)
parser.add_argument('--total_steps', default=1000000)
parser.add_argument('--sample_interval', default=1000)
parser.add_argument('--exp_num', type=int)


class RejectionSampling(RandomWalker):
    def __init__(self, initial_node: Node, min_deg: int, n_steps_min: int, n_steps_max: int):
        super(RejectionSampling, self).__init__(init_node=initial_node)
        self.min_deg = min_deg
        self.n_steps_min = n_steps_min
        self.n_steps_max = n_steps_max

    def random_step(self):
        n_steps = randint(self.n_steps_min, self.n_steps_max)
        for _ in range(n_steps):
            v = self.current_node
            self.current_node = choice(self.neighbors(v))

    def get_node(self):
        p = random()
        if p <= self.min_deg / len(self.neighbors(self.current_node)):
            return self.current_node
        return None
