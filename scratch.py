import argparse
import agentClass
import environmentClass
import runner
import logging
import numpy as np
import networkx as nx
import sys
from time import sleep
import matplotlib.pyplot as plt
from statistics import mean

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')

# Agent Parameters
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str,
                    help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--model', type=str, default='S2V_QN_1', help='model name')
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")

# Graph Parameters
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='MVC',
                    help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--graph_type', metavar='GRAPH', default='erdos_renyi', help='Type of graph to optimize')
parser.add_argument('--graph_nbr', type=int, default=1000,
                    help='number of different graphs to generate for the training sample')
parser.add_argument('--node', type=int, metavar='nnode', default=20, help="number of node in generated graphs")
parser.add_argument('--p', type=float, default=None, help="p, parameter in graph degree distribution")
parser.add_argument('--m', type=int, default=None, help="m, parameter in graph degree distribution")

# Simulation Parameters
parser.add_argument('--epoch', type=int, metavar='nepoch', default=1, help="number of epochs")
parser.add_argument('--bs', type=int, default=32, help="minibatch size for training")
parser.add_argument('--n_step', type=int, default=10, help="n step in RL")
parser.add_argument('--batch', type=int, metavar='nagent', default=10,
                    help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')
parser.add_argument('--test_interval', type=int, default=10, help='Test every x rounds')
parser.add_argument('--test_number', type=int, default=10, help='Test for x rounds')
parser.add_argument('--test_pool', type=int, default=100)


def step(environment, agent, test=False):
    observation = environment.observe()
    action = agent.act(observation, test).copy()
    (reward, done) = environment.act(action)
    if not test:
        agent.reward(observation, action, reward, done)
    return observation, action, reward, done


def main():
    args = parser.parse_args()
    mean_approx_ratios = []
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
    agent = agentClass.Agent(environment, args.model, args.lr, args.bs, args.n_step)

    print("Running a single instance simulation...")
    number_of_epochs, number_of_games, max_iter = args.epoch, args.graph_nbr, args.node
    agent.game_based_params(number_of_games)
    reward_opt_approx = []

    for epoch_ in range(number_of_epochs):
        print(f"{epoch_}/{number_of_epochs} epochs")
        for g in range(number_of_games):
            print(f"{g + 1}/{number_of_games} games")
            # Conduct test every x rounds
            test_and_or_train = list(range(1 + ((g + 1) % args.test_interval == 0)))
            for testing in test_and_or_train:
                approx_ratios = []
                rounds = 3 if not testing else args.test_number
                games = g

                for repeat in range(rounds):
                    # reset all variables
                    if testing: g = np.random.randint(0, args.test_pool)
                    environment.reset(g, testing)
                    agent.reset(g, testing)
                    done = False
                    cumul_reward = 0.0
                    iteration = 0

                    ########################## EPISODE #############################
                    while iteration <= max_iter and not done:
                        (obs, act, rew, done) = step(environment, agent, testing)
                        cumul_reward += rew
                        iteration += 1
                        if args.verbose:
                            # verbose_update(obs, act, rew, cumul_reward)
                            pass
                    ################################################################

                    reward, opt, approx = cumul_reward, environment.get_optimal_sol(), environment.get_approx()
                    if testing:
                        approx_ratios.append(iteration / opt)
                    else:
                        # cumulative reward of one play is actually the solution found by the NN algorithm
                        terminal_update(cumul_reward, iteration, opt)
                        reward_opt_approx.append([reward, opt, approx])

                if args.verbose and not testing:
                    print(" <=> Finished game number: {} <=>".format(g + 1))
                    print(" Epsilon: {:.0f}%".format(agent.epsilon_ * 100))
                    print("")

            if testing:
                mean_approx_ratios.append(mean(approx_ratios))
                print(f"Test: {games} avg_approx: {mean(approx_ratios)}\n")

    # np.savetxt('test.out', list_cumul_reward, delimiter=',')
    # np.savetxt('opt_set.out', list_optimal_ratio, delimiter=',')

    agent.save_model()

    np.savetxt("mean_approx_ratios.txt", mean_approx_ratios, fmt='%.6f')


def terminal_update(cumul_reward, iteration, opt):
    # print(" ->    Terminal event: cumulative rewards = {}".format(cumul_reward))
    # print(" ->    Number of nodes picked = {}".format(iteration))
    # print(" ->    Optimal solution = {}\n".format(opt))
    approx_ratio = ((iteration / opt) - 1) * 100
    print_str = f"R: {cumul_reward:.2f}, OUT: {iteration}, OPT: {opt}, APR: {approx_ratio:.2f}%"

    print(print_str)


def verbose_update(obs, act, rew, cumul_rew):
    print(" ->       observation: {}".format(obs))
    print(" ->            action: {}".format(act))
    print(" ->            reward: {}".format(rew))
    print(" -> cumulative reward: {}".format(cumul_rew))
    # sleep(1)


if __name__ == '__main__':
    main()
