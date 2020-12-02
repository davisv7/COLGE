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
from common_utils import *

"""
This file contains the definition of the environment
in which the agents are run.
"""
Path("data").mkdir(exist_ok=True)
TRAIN_PATH = join("data", "train")
TEST_PATH = join("data", "test")
Path(TRAIN_PATH).mkdir(exist_ok=True)
Path(TEST_PATH).mkdir(exist_ok=True)


class Environment:
    def __init__(self, args):
        self.args = args
        self.train_graphs = {}
        self.test_graphs = {}
        self.name = args.environment_name
        self.starter_seed = random.randint(0, 100000000)

        self.load_train()  # load existing training and test graphs
        self.load_test()  # load existing training and test graphs
        self.generate()  # generate remaining training and test graphs

        self.solve_all()  # solve generated graphs
        self.save()  # save all graphs/solutions

    def generate(self):
        for graph_ in range(len(self.train_graphs), self.args.graph_nbr):
            seed = self.starter_seed + graph_
            np.random.seed(seed)

            self.train_graphs[graph_] = graphClass.Graph(
                graph_type=self.args.graph_type,
                cur_n=self.args.node,
                p=self.args.p,
                m=self.args.m,
                seed=seed)

        for graph_ in range(len(self.test_graphs), self.args.test_pool):
            seed = self.starter_seed + self.args.graph_nbr + graph_
            np.random.seed(seed)

            self.test_graphs[graph_] = graphClass.Graph(
                graph_type=self.args.graph_type,
                cur_n=self.args.node,
                p=self.args.p,
                m=self.args.m,
                seed=seed)

    def solve_all(self):
        for k in self.train_graphs:
            self.train_graphs[k].solution = self.get_optimal_sol(self.train_graphs[k])

        for k in self.test_graphs:
            self.test_graphs[k].solution = self.get_optimal_sol(self.test_graphs[k])

    def approx_all(self):
        for k in self.train_graphs:
            self.get_approx(self.train_graphs[k])
        pass

    def reset(self, g, test=False):
        if test:  # dont update self.games, make current graph a test graph
            self.current_graph = self.test_graphs[g]
        else:
            self.games = g
            self.current_graph = self.train_graphs[self.games]

        self.current_graph.approx = self.get_approx()
        self.nodes = self.current_graph.nodes()
        self.nbr_nodes_selected = 0
        self.old_edges_covered = 0
        self.old_nodes_covered = 0
        self.last_reward = 0
        self.observation = torch.zeros(1, self.nodes, 1, dtype=torch.float)
        self.cumulative_reward = 0

    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        return self.observation.clone()

    def act(self, node):

        self.observation[:, node, :] = 1
        reward, done = self.get_reward(self.observation, node)
        return reward, done

    def get_reward(self, observation, node):

        if self.name == "MVC":
            self.nbr_nodes_selected = np.sum(observation[0].numpy())

            # Minimum vertex set:
            done, edges_covered, nodes_covered = self.is_covered(observation)

            # avg_edges_covered = (len(self.current_graph.edges()) / self.args.node)
            # avg_edges_selected = avg_edges_covered * self.nbr_nodes_selected
            # new_edges_covered = edges_covered - self.old_edges_covered
            # new_nodes_covered = nodes_covered - self.old_nodes_covered

            # according to the paper, this is how the rewards are decided, (old number selected - current number selected)
            reward = -1

            # self.old_edges_covered = edges_covered
            # self.old_nodes_covered = nodes_covered
            self.cumulative_reward += reward
            return reward, done

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

    def is_covered(self, observation):
        done = True
        edges_covered = 0
        nodes_covered = set()
        for edge in self.current_graph.edges():
            if observation[:, edge[0], :] == 0 and observation[:, edge[1], :] == 0:
                done = False
                # break
            else:
                nodes_covered.add(edge[0])
                nodes_covered.add(edge[1])
                edges_covered += 1
        return done, edges_covered, len(nodes_covered)

    def get_approx(self, graph=None):  # greedy approx
        if graph == None:
            graph = self.current_graph
        approx = graph.approx
        if approx:
            return approx

        if self.name == "MVC":
            count = 0
            nodes = list(range(self.current_graph.nodes()))
            edges = list(self.current_graph.edges())
            while len(edges) > 0:
                # node that covers the most edges
                best_node = max(nodes, key=lambda n: np.array(edges).flatten().tolist().count(n))
                # filter edges covered by best_node
                edges = list(filter(lambda e: best_node not in e, edges))
                # remove node from node list
                nodes.remove(best_node)
                count += 1
            graph.approx = count
            return count


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
        for _g in self.train_graphs.values():
            location = join(TRAIN_PATH, f"{_g.graph_type}_{_g.cur_n}_{_g.seed}_{_g.solution}_p_{_g.p}_m_{_g.m}.adjlist")
            if _g.solution and not exists(location):
                nx.write_adjlist(_g.g, location)

        for _g in self.test_graphs.values():
            location = join(TEST_PATH, f"{_g.graph_type}_{_g.cur_n}_{_g.seed}_{_g.solution}_p_{_g.p}_m_{_g.m}.adjlist")
            if _g.solution and not exists(location):
                nx.write_adjlist(_g.g, location)

    def load_train(self):
        """
        List directory
        Load graphs that match the arguments
        Add them to the respective dictionaries and return
        :return:
        """
        proto_graph = graphClass.Graph("prototype", self.args.node)
        files = listdir(TRAIN_PATH)
        eligible_graphs = filter_filenames(files, self.args)
        if len(eligible_graphs) > self.args.graph_nbr:
            random.shuffle(eligible_graphs)
        for i, filename in enumerate(eligible_graphs[:self.args.graph_nbr]):
            graph_shell = deepcopy(proto_graph)
            new_graph = nx.read_adjlist(join(TRAIN_PATH, filename), nodetype=int)
            g_type, size, seed, optimal, p, m = parse_filename(filename)
            graph_shell.replicate(g_type, new_graph, size, seed, optimal, m, p)
            self.train_graphs[i] = graph_shell

    def load_test(self):
        """
        List directory
        Load graphs that match the arguments
        Add them to the respective dictionaries and return
        :return:
        """
        proto_graph = graphClass.Graph("prototype", self.args.node)
        files = listdir(TEST_PATH)
        eligible_graphs = filter_filenames(files, self.args)
        if len(eligible_graphs) > self.args.graph_nbr:
            random.shuffle(eligible_graphs)
        for i, filename in enumerate(eligible_graphs[:self.args.test_pool]):
            graph_shell = deepcopy(proto_graph)
            new_graph = nx.read_adjlist(join(TEST_PATH, filename), nodetype=int)
            g_type, size, seed, optimal, p, m = parse_filename(filename)
            graph_shell.replicate(g_type, new_graph, size, seed, optimal, m, p)
            self.test_graphs[i] = graph_shell

# TODO: Testing Environment vs Training Environment
# How to share the models? Or just have a separate method...
