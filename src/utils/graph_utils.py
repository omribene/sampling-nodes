from operator import itemgetter

import igraph

class GraphUtils:

    @staticmethod
    def load_graph(file_path, directed=True):
        """
        Loads a graph from file. The graph needs to be in edges format: each line contains two IDs,
        one for each node of the edge.
        """
        print(f"Reading edge list in file. Path: {file_path}. Directed: {directed}")
        # assumes file_path contains only edges (no other metadata - please delete it from file)
        with open(file_path, "r") as f:
            g = igraph.Graph.Read_Edgelist(f, directed=directed)
        print("Done reading file")
        for v in g.vs():
            v["name"] = v.index
        print(f"Done loading the graph. Number of nodes: {g.vcount()}, number of edges: {g.ecount()}")

        # sort degrees in descending order of degrees, and then by ascending vertex names
        print("Computing in degrees")
        in_degrees = [(v.index, -v.indegree()) for v in g.vs()]
        in_degrees = sorted(in_degrees, key=lambda x: (x[1], x[0]))
        in_degrees = [(v, -deg) for v, deg in in_degrees]
        return g, in_degrees

