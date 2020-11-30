import numpy as np
import networkx as nx
import collections
import uuid
from os.path import join


class Graph:
    def __init__(self, graph_type, cur_n=0, p=None, m=None, seed=None):
        """
        initialize a graphing problem
        :param graph_type: type of random graph to generate
        :param cur_n: size of graph
        :param p: probability of being connected
        :param m:
        :param seed: seed used to gen graph under
        """
        self.graph_type = graph_type
        self.cur_n = cur_n
        self.seed = seed
        self.m, self.p = "_", "_"
        if m:
            self.m = m
        if p:
            self.p = p
        if graph_type == "prototype":
            pass
        elif graph_type == 'erdos_renyi':
            self.g = nx.erdos_renyi_graph(n=cur_n, p=p, seed=seed)
        elif graph_type == 'powerlaw':
            self.g = nx.powerlaw_cluster_graph(n=cur_n, m=m, p=p, seed=seed)
        elif graph_type == 'barabasi_albert':
            self.g = nx.barabasi_albert_graph(n=cur_n, m=m, seed=seed)
        elif graph_type == 'gnp_random_graph':
            self.g = nx.gnp_random_graph(n=cur_n, p=p, seed=seed)

        self.solution = None

        # power=0.75
        #
        # self.edgedistdict = collections.defaultdict(int)
        # self.nodedistdict = collections.defaultdict(int)
        #
        # for edge in self.g.edges:
        #     self.edgedistdict[tuple(edge[0],edge[1])] = 1.0/float(len(self.g.edges))
        #
        # for node in self.g.nodes:
        #     self.nodedistdict[node]=float(len(nx.neighbors(self.g,node)))**power/float(len(self.g.edges))

    def nodes(self):

        return nx.number_of_nodes(self.g)

    def edges(self):

        return self.g.edges()

    def neighbors(self, node):

        return nx.all_neighbors(self.g, node)

    def average_neighbor_degree(self, node):

        return nx.average_neighbor_degree(self.g, nodes=node)

    def adj(self):

        return nx.adjacency_matrix(self.g)

    def replicate(self, g_type, graph, cur_n, seed, solution, m, p):
        self.graph_type = g_type
        self.g = graph
        self.cur_n = cur_n
        self.seed = seed
        self.solution = solution
        self.m, self.p = "_", "_"
        if m:
            self.m = m
        if p:
            self.p = p

    # def save(self):
    #
    #     nx.write_adjlist(self.g, join("data", f"{self.uniqueid}_{}_{}.adjlist"))
#
