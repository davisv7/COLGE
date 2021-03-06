import argparse
import agentClass
import environmentClass
import runner
import logging
import numpy as np
import networkx as nx
import sys

# # 2to3 compatibility
# try:
#     input = raw_input
# except NameError:
#     pass

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='MVC',
                    help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str,
                    help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--graph_type', metavar='GRAPH', default='erdos_renyi', help='Type of graph to optimize')
parser.add_argument('--graph_nbr', type=int, default='1000',
                    help='number of differente graph to generate for the training sample')
parser.add_argument('--model', type=str, default='S2V_QN_1', help='model name')
parser.add_argument('--ngames', type=int, metavar='n', default='500', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
parser.add_argument('--epoch', type=int, metavar='nepoch', default=25, help="number of epochs")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--bs', type=int, default=32, help="minibatch size for training")
parser.add_argument('--n_step', type=int, default=3, help="n step in RL")
parser.add_argument('--node', type=int, metavar='nnode', default=20, help="number of node in generated graphs")
parser.add_argument('--p', default=0.14, help="p, parameter in graph degree distribution")
parser.add_argument('--m', default=4, help="m, parameter in graph degree distribution")
parser.add_argument('--batch', type=int, metavar='nagent', default=None,
                    help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')


def main():
    args = parser.parse_args()
    logging.info('Loading graph %s' % args.graph_type)
    # seed = 125
    # graph_one = graph.Graph(graph_type=args.graph_type, cur_n=20, p=0.15,m=4, seed=seed)

    logging.info('Loading environment %s' % args.environment_name)
    env_class = environmentClass.Environment(args)

    logging.info('Loading agent...')
    agent_class = agentClass.Agent(env_class.graphs, args.model, args.lr, args.bs, args.n_step)

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchRunner(env_class, agent_class, args.batch, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.epoch, args.niter)
        print("Obtained a final average reward of {}".format(final_reward))
        agent_class.save_model()
    else:
        print("Running a single instance simulation...")
        my_runner = runner.Runner(env_class, agent_class, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.epoch, args.niter)
        print("Obtained a final reward of {}".format(final_reward))
        agent_class.save_model()


if __name__ == "__main__":
    main()
