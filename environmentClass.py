import numpy as np
import torch
import pulp
from pathlib import Path
import networkx as nx
from os.path import join, exists
from os import listdir
import graphClass
import random
from itertools import takewhile
from copy import deepcopy

"""
This file contains the definition of the environment
in which the agents are run.
"""
Path("data").mkdir(exist_ok=True)


class Environment:
    def __init__(self, args):
        self.args = args
        self.graphs = {}
        self.approximals = {}
        self.seeds = []
        self.name = args.environment_name
        self.starter_seed = random.randint(0, 100000000)

        # TODO: Check to see if enough graphs with these arguments and if so, load them instead
        self.load()
        self.generate()

        # TODO: load the ones that do and generate the rest
        # TODO: find their solutions first, and then save them
        self.solve_all()
        self.save()

    def generate(self):
        for graph_ in range(len(self.graphs), self.args.graph_nbr):
            seed = self.starter_seed + graph_
            np.random.seed(seed)
            self.seeds.append(seed)

            self.graphs[graph_] = graphClass.Graph(
                graph_type=self.args.graph_type,
                cur_n=self.args.node,
                p=self.args.p,
                m=self.args.m,
                seed=seed)

    def solve_all(self):
        for k in self.graphs:
            self.graphs[k].solution = self.get_optimal_sol(self.graphs[k])

    def approx_all(self):
        for k in self.graphs:
            self.get_approx(self.graphs[k])
        pass

    def reset(self, g):
        self.games = g
        self.current_graph = self.graphs[self.games]
        self.nodes = self.current_graph.nodes()
        self.nbr_of_nodes = 0
        self.edge_add_old = 0
        self.last_reward = 0
        self.observation = torch.zeros(1, self.nodes, 1, dtype=torch.float)

    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        return self.observation

    def act(self, node):

        self.observation[:, node, :] = 1
        reward = self.get_reward(self.observation, node)
        return reward

    def get_reward(self, observation, node):

        if self.name == "MVC":

            new_nbr_nodes = np.sum(observation[0].numpy())

            if new_nbr_nodes - self.nbr_of_nodes > 0:
                reward = -1  # np.round(-1.0/20.0,3)
            else:
                reward = 0

            self.nbr_of_nodes = new_nbr_nodes

            # Minimum vertex set:

            done = True

            edge_add = 0

            for edge in self.current_graph.edges():
                if observation[:, edge[0], :] == 0 and observation[:, edge[1], :] == 0:
                    done = False
                    # break
                else:
                    edge_add += 1

            # reward = ((edge_add - self.edge_add_old) / np.max(
            #   [1, self.graph_init.average_neighbor_degree([node])[node]]) - 10)/100

            self.edge_add_old = edge_add

            return (reward, done)

        elif self.name == "MAXCUT":

            reward = 0
            done = False

            adj = self.current_graph.edges()
            select_node = np.where(self.observation[0, :, 0].numpy() == 1)[0]
            for nodes in adj:
                if ((nodes[0] in select_node) & (nodes[1] not in select_node)) | (
                        (nodes[0] not in select_node) & (nodes[1] in select_node)):
                    reward += 1  # /20.0
            change_reward = reward - self.last_reward
            if change_reward <= 0:
                done = True

            self.last_reward = reward

            return (change_reward, done)

    def get_approx(self, graph=None):
        if graph == None:
            graph = self.current_graph
        approx = self.approximals.get(graph, None)
        if approx:
            return approx

        if self.name == "MVC":
            cover_edge = []
            edges = list(self.current_graph.edges())
            while len(edges) > 0:
                edge = edges[np.random.choice(len(edges))]
                cover_edge.append(edge[0])
                cover_edge.append(edge[1])
                to_remove = []
                for edge_ in edges:
                    if edge_[0] == edge[0] or edge_[0] == edge[1]:
                        to_remove.append(edge_)
                    else:
                        if edge_[1] == edge[1] or edge_[1] == edge[0]:
                            to_remove.append(edge_)
                for i in to_remove:
                    edges.remove(i)
            return len(cover_edge)

        elif self.name == "MAXCUT":
            return 1

        else:
            return 'you pass a wrong environment name'

    def get_optimal_sol(self, graph=None):
        if graph == None:
            graph = self.current_graph
        optimal = graph.solution
        if optimal:
            return optimal

        if self.name == "MVC":

            x = list(range(graph.g.number_of_nodes()))
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMinimize)

            mdl += sum(xv[k] for k in xv)

            for edge in graph.edges():
                mdl += xv[edge[0]] + xv[edge[1]] >= 1, "constraint :" + str(edge)
            mdl.solve()

            # print("Status:", pulp.LpStatus[mdl.status])
            optimal = 0
            for x in xv:
                optimal += xv[x].value()
                # print(xv[x].value())


        elif self.name == "MAXCUT":

            x = list(range(graph.g.number_of_nodes()))
            e = list(graph.edges())
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)
            ev = pulp.LpVariable.dicts('ev', e,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMaximize)

            mdl += sum(ev[k] for k in ev)

            for i in e:
                mdl += ev[i] <= xv[i[0]] + xv[i[1]]

            for i in e:
                mdl += ev[i] <= 2 - (xv[i[0]] + xv[i[1]])

            # pulp.LpSolverDefault.msg = 1
            mdl.solve()

            # print("Status:", pulp.LpStatus[mdl.status])
            optimal = mdl.objective.value()

        graph.solution = optimal
        return optimal

    def save(self):
        """
        Save all graphs that have a solution
        filename has all of the argument information in the title
        :return:
        """
        for _graph in self.graphs.values():
            location = join("data",
                            f"{_graph.graph_type}_{_graph.cur_n}_{_graph.seed}_{_graph.solution}_p_{_graph.p}_m_{_graph.m}.adjlist")
            if _graph.solution and not exists(location):
                nx.write_adjlist(_graph.g, location)

    def load(self):
        """
        List directory
        Load graphs that match the arguments
        Add them to the dictionary and return
        :return:
        """
        proto_graph = graphClass.Graph("prototype", self.args.node)

        files = listdir("data")
        eligible_graphs = filter_filenames(files, self.args)
        for i, filename in enumerate(eligible_graphs):
            graph_copy = deepcopy(proto_graph)
            new_graph = nx.read_adjlist(join("data", filename), nodetype=int)
            g_type, size, seed, optimal, p, m = parse_filename(filename)
            graph_copy.replicate(g_type, new_graph, size, seed, optimal, m, p)
            self.graphs[i] = graph_copy


# TODO: Testing Environment vs Training Environment
# How to share the models? Or just have a separate method...

def filename_filter(filename, args):
    g_type, size, seed, optimal, p, m = parse_filename(filename)
    if g_type == args.graph_type and size == args.node and p == args.p and m == args.m:
        return True


def filter_filenames(filenames, args):
    return list(filter(lambda file: filename_filter(file, args), filenames))


def parse_filename(filename):
    # parameters:  type_size_seed_optimal_p_m
    g_type = "".join(takewhile(lambda x: not x.isnumeric(), filename)).rstrip("_")

    filename = filename.replace(g_type, "").replace(".adjlist", "").lstrip("_")
    params = filename.split("_")

    p = None
    m = None
    size, seed, optimal, *rest = params

    pindex = rest.index("p")
    mindex = rest.index("m")
    if rest[pindex + 1] != "":
        p = float(rest[pindex + 1])
    if rest[mindex + 1] != "":
        m = float(rest[mindex + 1])

    size = int(size)
    seed = int(seed)
    optimal = float(optimal)

    return g_type, size, seed, optimal, p, m
