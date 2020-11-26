import argparse
import agentClass
import environmentClass
import runner
import logging
import numpy as np
import networkx as nx
import sys

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='MVC',
                    help='Class to use for the environment. Must be in the \'environment\' module')

# Agent Parameters
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str,
                    help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--model', type=str, default='S2V_QN_1', help='model name')
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")

# Graph Parameters
parser.add_argument('--graph_type', metavar='GRAPH', default='erdos_renyi', help='Type of graph to optimize')
parser.add_argument('--graph_nbr', type=int, default='10',
                    help='number of differente graph to generate for the training sample')
parser.add_argument('--node', type=int, metavar='nnode', default=20, help="number of node in generated graphs")
parser.add_argument('--p', type=float, default=None, help="p, parameter in graph degree distribution")
parser.add_argument('--m', type=float, default=None, help="m, parameter in graph degree distribution")

# Simulation Parameters
parser.add_argument('--ngames', type=int, metavar='n', default='500', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
parser.add_argument('--epoch', type=int, metavar='nepoch', default=25, help="number of epochs")
parser.add_argument('--bs', type=int, default=32, help="minibatch size for training")
parser.add_argument('--n_step', type=int, default=3, help="n step in RL")
parser.add_argument('--batch', type=int, metavar='nagent', default=None,
                    help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')


def step(environment, agent):
    observation = environment.observe().clone()
    action = agent.act(observation).copy()
    (reward, done) = environment.act(action)
    agent.reward(observation, action, reward, done)
    return (observation, action, reward, done)


def main():
    args = parser.parse_args()

    ps = ['erdos_renyi', 'powerlaw', 'gnp_random_graph']
    ms = ['powerlaw', 'barabasi_albert']
    if args.graph_type in ps and args.p is None:
        print("You need to specify a p-value")
        return
    if args.graph_type in ms and args.m is None:
        print("You need to specify an m-value")
        return

    logging.info('Loading graph %s' % args.graph_type)

    logging.info('Loading environment %s' % args.environment_name)
    environment = environmentClass.Environment(args)

    logging.info('Loading agent...')
    agent = agentClass.Agent(environment.graphs, args.model, args.lr, args.bs, args.n_step)

    # print("Running a single instance simulation...")
    # nbr_epoch, games, max_iter = args.ngames, args.epoch, args.niter
    # cumul_reward = 0.0
    # list_cumul_reward = []
    # list_optimal_ratio = []
    # list_aprox_ratio = []
    #
    # for epoch_ in range(nbr_epoch):
    #     print(f"{epoch_}/{nbr_epoch} epochs")
    #     for g in range(1, games + 1):
    #         print(f"{g}/{games} games")
    #         for epoch in range(5):
    #             # update to new environment and reset agent
    #             environment.reset(g)
    #             agent.reset(g)
    #
    #             cumul_reward = 0.0
    #
    #             for i in range(1, max_iter + 1):
    #                 (obs, act, rew, done) = step(environment, agent)
    #                 cumul_reward += rew
    #                 if args.verbose:
    #                     print(" ->       observation: {}".format(obs))
    #                     print(" ->            action: {}".format(act))
    #                     print(" ->            reward: {}".format(rew))
    #                     print(" -> cumulative reward: {}".format(cumul_reward))
    #                     if done:
    #                         # solution from baseline algorithm
    #                         approx_sol = environment.get_approx()
    #
    #                         # optimal solution
    #                         optimal_sol = environment.get_optimal_sol()
    #
    #                         # print cumulative reward of one play, it is actually the solution found by the NN algorithm
    #                         print(" ->    Terminal event: cumulative rewards = {}".format(cumul_reward))
    #
    #                         # print optimal solution
    #                         print(" ->    Optimal solution = {}".format(optimal_sol))
    #
    #                         # we add in a list the solution found by the NN algorithm
    #                         list_cumul_reward.append(-cumul_reward)
    #
    #                         # we add in a list the ratio between the NN solution and the optimal solution
    #                         list_optimal_ratio.append(cumul_reward / (optimal_sol))
    #
    #                         # we add in a list the ratio between the NN solution and the baseline solution
    #                         list_aprox_ratio.append(cumul_reward / (approx_sol))
    #
    #                 if done:
    #                     break
    #
    #         np.savetxt('test_' + str(epoch_) + '.out', list_optimal_ratio, delimiter=',')
    #         np.savetxt('test_approx_' + str(epoch_) + '.out', list_aprox_ratio, delimiter=',')
    #
    #     if args.verbose:
    #         print(" <=> Finished game number: {} <=>".format(g))
    #         print("")
    #
    # np.savetxt('test.out', list_cumul_reward, delimiter=',')
    # np.savetxt('opt_set.out', list_optimal_ratio, delimiter=',')
    #
    # agent.save_model()


if __name__ == '__main__':
    main()
