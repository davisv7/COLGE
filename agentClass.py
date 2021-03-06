import numpy as np
import random
import time
import os
import logging
import models
from utils.config import load_model_config

import torch.nn.functional as F
import torch
from collections import deque

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

"""
Contains the definition of the agent that will run in an
environment.
"""


class DQAgent:

    def __init__(self, environment, model, lr, bs, n_step):

        self.train_graphs = environment.train_graphs
        self.test_graphs = environment.test_graphs
        self.embed_dim = 64
        self.model_name = model

        self.alpha = 0.1
        self.gamma = 0.99
        self.lambd = 0.
        self.n_step = n_step

        self.decay = True  # whether or not to decay this round
        self.epsilon_ = 1  # starting epsilon value
        self.epsilon_min = 0.01  # minimum epsilon value
        self.decay_factor = 1  # overwritten by self.calc_convergence to converge in one epoch
        self.memory = deque(maxlen=1)  # same
        self.memory_n = deque(maxlen=1)  # same

        self.t = 1
        self.minibatch_length = bs
        self.games = 0
        # decide model
        if self.model_name == 'S2V_QN_1':
            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_1(**args_init)

        elif self.model_name == 'S2V_QN_2':
            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_2(**args_init)

        elif self.model_name == 'GCN_QN_1':
            args_init = load_model_config()[self.model_name]
            self.model = models.GCN_QN_1(**args_init)

        elif self.model_name == 'LINE_QN':
            args_init = load_model_config()[self.model_name]
            self.model = models.LINE_QN(**args_init)

        elif self.model_name == 'W2V_QN':
            args_init = load_model_config()[self.model_name]
            self.model = models.W2V_QN(G=self.train_graphs[self.games], **args_init)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.reset(self.games)

    """
    p : embedding dimension
    """

    def game_based_params(self, games):
        # take size of graph, number of graphs and when you want to converge by and calc a discount factor
        # size and number graphs determines max number of discounts per epoch
        # calc value x s.t og_epsilon*(x^max_discounts) = min epsilon
        # return x
        max_steps = self.nodes
        max_discounts = max_steps * games
        self.decay_factor = (self.epsilon_min / self.epsilon_) ** (1 / max_discounts)
        self.memory = deque(maxlen=min(10000, max_discounts))
        self.memory_n = deque(maxlen=min(10000, max_discounts))

    def reset(self, g, test=False):
        if test:
            self.decay = False
            self.nodes = self.test_graphs[g].nodes()
            self.adj = self.test_graphs[g].adj()
        else:
            if g == self.games and g != 0:
                self.decay = False
            else:
                self.decay = True
            self.games = g

            if (len(self.memory_n) != 0) and (len(self.memory_n) % 300000 == 0):
                self.memory_n = random.sample(self.memory_n, 120000)

            self.nodes = self.train_graphs[self.games].nodes()
            self.adj = self.train_graphs[self.games].adj()
        self.adj = self.adj.todense()
        self.adj = torch.from_numpy(np.expand_dims(self.adj.astype(int), axis=0))
        self.adj = self.adj.type(torch.FloatTensor)

        self.last_action = 0
        self.last_observation = torch.zeros(1, self.nodes, 1, dtype=torch.float)
        self.last_reward = -0.01
        self.last_done = 0
        self.iter = 1

    def act(self, observation, test=False):

        if self.epsilon_ > np.random.rand() and not test:
            return np.random.choice(np.where(observation.numpy()[0, :, 0] == 0)[0])
        else:
            q_a = self.model(observation, self.adj)
            q_a = q_a.detach().numpy()
            # get index of first unobserved node with max q value
            # return np.where((q_a[0, :, 0] == np.max(q_a[0, :, 0][observation.numpy()[0, :, 0] == 0])))[0][0]
            # get action with best q value, break ties randomly
            maxes = np.flatnonzero((q_a[0, :, 0] == np.max(q_a[0, :, 0][observation.numpy()[0, :, 0] == 0])))
            choice = np.random.choice(maxes)
            return choice

    def reward(self, observation, action, reward, done):

        if len(self.memory_n) > self.minibatch_length + self.n_step:
            (last_observation_tens, action_tens, reward_tens, observation_tens, done_tens, adj_tens) = self.get_sample()
            target = reward_tens + self.gamma * (1 - done_tens) * \
                     torch.max(self.model(observation_tens, adj_tens) + (observation_tens * (-1e5)), dim=1)[0]
            target_f = self.model(last_observation_tens, adj_tens)
            target_p = target_f.clone()
            target_f[range(self.minibatch_length), action_tens, :] = target
            loss = self.criterion(target_p, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(self.t, loss)

        if self.iter > 1:
            self.remember(self.last_observation, self.last_action, self.last_reward, observation.clone(),
                          self.last_done * 1)

        if done & self.iter > self.n_step:
            # if done:
            self.remember_n(False)
            new_observation = observation.clone()
            new_observation[:, action, :] = 1
            self.remember(observation, action, reward, new_observation, done * 1)

        if self.iter > self.n_step:
            self.remember_n(done)
        self.iter += 1
        self.t += 1
        self.last_action = action
        self.last_observation = observation.clone()
        self.last_reward = reward
        self.last_done = done
        if self.decay:
            self.epsilon_ = max(self.epsilon_min, self.epsilon_ * self.decay_factor)

    def get_sample(self):

        minibatch = random.sample(self.memory_n, self.minibatch_length - 1)
        minibatch.append(self.memory_n[-1])
        last_observation_tens = minibatch[0][0]
        action_tens = torch.Tensor([minibatch[0][1]]).type(torch.LongTensor)
        reward_tens = torch.Tensor([[minibatch[0][2]]])
        observation_tens = minibatch[0][3]
        done_tens = torch.Tensor([[minibatch[0][4]]])
        adj_tens = self.train_graphs[minibatch[0][5]].adj().todense()
        adj_tens = torch.from_numpy(np.expand_dims(adj_tens.astype(int), axis=0)).type(torch.FloatTensor)

        for last_observation_, action_, reward_, observation_, done_, games_ in minibatch[-self.minibatch_length + 1:]:
            last_observation_tens = torch.cat((last_observation_tens, last_observation_))
            action_tens = torch.cat((action_tens, torch.Tensor([action_]).type(torch.LongTensor)))
            reward_tens = torch.cat((reward_tens, torch.Tensor([[reward_]])))
            observation_tens = torch.cat((observation_tens, observation_))
            done_tens = torch.cat((done_tens, torch.Tensor([[done_]])))
            adj_ = self.train_graphs[games_].adj().todense()
            adj = torch.from_numpy(np.expand_dims(adj_.astype(int), axis=0)).type(torch.FloatTensor)
            adj_tens = torch.cat((adj_tens, adj))

        return (last_observation_tens, action_tens, reward_tens, observation_tens, done_tens, adj_tens)

    def remember(self, last_observation, last_action, last_reward, observation, done):
        self.memory.append((last_observation, last_action, last_reward, observation, done, self.games))

    def remember_n(self, done):

        if not done:
            step_init = self.memory[-self.n_step]
            cum_reward = step_init[2]
            for step in range(1, self.n_step):
                cum_reward += self.memory[-step][2]
            self.memory_n.append(
                (step_init[0], step_init[1], cum_reward, self.memory[-1][-3], self.memory[-1][-2], self.memory[-1][-1]))

        else:
            for i in range(1, self.n_step):
                step_init = self.memory[-self.n_step + i]
                cum_reward = step_init[2]
                for step in range(1, self.n_step - i):
                    cum_reward += self.memory[-step][2]
                self.memory_n.append(
                    (step_init[0], step_init[1], cum_reward, self.memory[-1][-3], False, self.memory[-1][-1]))

    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), cwd + '/model.pt')


Agent = DQAgent
