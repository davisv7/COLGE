import numpy as np
import torch
import pulp
from pathlib import Path
import networkx as nx
from os.path import join
from os import listdir
import graphClass
import random

"""
This file contains the definition of the environment
in which the agents are run.
"""
Path("data").mkdir(exist_ok=True)


class Environment:
    def __init__(self, args):
        self.args = args
        self.graphs = {}
        self.optimals = {}
        self.seeds = []
        self.starter_seed = random.randint(0, 100000000)

        # TODO: Check to see if enough graphs with these arguments and if so, load them instead
        # enumerate training graphs
        # uniqueID_name(type)_optimal
        # f"{_graph.graph_type}_{_graph.cur_n}_{_graph.seed}_p_{_graph.p}_m_{_graph.m}.adjlist"))
        # file_list = listdir("data")

        # args.graph_nbr
        if len(self.graphs) < args.graph_nbr:
            self.generate()

        # TODO: load the ones that do and generate the rest
        # TODO: find their solutions first, and then save them

        self.name = args.environment_name

    def generate(self):
        for graph_ in range(self.args.graph_nbr):
            seed = np.random.seed(self.starter_seed + graph_)
            self.seeds.append(seed)

            self.graphs[graph_] = graphClass.Graph(
                graph_type=self.args.graph_type,
                cur_n=self.args.node,
                p=self.args.p,
                m=self.args.m,
                seed=seed)


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

    def get_approx(self):

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

    def get_optimal_sol(self):
        optimal = self.optimals.get(self.current_graph, None)
        if optimal:
            return optimal

        if self.name == "MVC":

            x = list(range(self.current_graph.g.number_of_nodes()))
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMinimize)

            mdl += sum(xv[k] for k in xv)

            for edge in self.current_graph.edges():
                mdl += xv[edge[0]] + xv[edge[1]] >= 1, "constraint :" + str(edge)
            mdl.solve()

            # print("Status:", pulp.LpStatus[mdl.status])
            optimal = 0
            for x in xv:
                optimal += xv[x].value()
                # print(xv[x].value())
            self.optimals[self.current_graph] = self.current_graph.solution

        elif self.name == "MAXCUT":

            x = list(range(self.current_graph.g.number_of_nodes()))
            e = list(self.current_graph.edges())
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

        self.current_graph.solution = optimal
        return optimal

    def save(self):
        for _graph in self.graphs.values():
            optimal = self.optimals.get(_graph, None)
            if optimal:
                nx.write_adjlist(_graph, join("data",
                                              f"{_graph.graph_type}_{_graph.cur_n}_{_graph.seed}_{self.optimals[_graph]}_p_{_graph.p}_m_{_graph.m}.adjlist"))

    # def load(self):
    # DG = nx.read_adjlist(join("data", f"{uniqueid}.adjlist"))

# TODO: Testing Environment vs Training Environment
# How to share the models? Or just have a separate method...
