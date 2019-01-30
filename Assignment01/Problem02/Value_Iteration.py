from gridworld import GridworldEnv
import numpy as np
import gym
from gym import wrappers
import itertools
import argparse

"""
Original Grid world environment was taken from link: https://github.com/dennybritz/reinforcement-learning
I modified it with: 
(1) "Terminal states" at being 
	- Upper left with reward 1.0
	- Upper right with reward 10.0
	- Everywhere else reward 0.0
(2) Inserted stochastic action condition:
	- where it takes input "prob_p = P " probability of taking policy action
	- and with probability (1-P) takes random action
"""


def Value_Iteration(env, N, theta=0.0001, discount_factor=0.9):
	def one_step_lookahead(state, V):
		Q = np.zeros(env.nA)
		# for given "action"
		for a in range(env.nA):
			# compute for all possible "next_state"
			for prob, next_state, reward, done in env.P[state][a]:
				Q[a] += prob * (reward + discount_factor * V[next_state])
		return Q

	V = np.zeros(env.nS)  # V_k(s)

	# while True:
	for itr in itertools.count():

		delta = 0
		for s in range(env.nS):
			# one step look-ahead for each possible state
			# it provides Q[state, all possible action]
			Q = one_step_lookahead(s, V)
			# for given "state" get the "Value" for "best possible action"
			# using one-step-lookahead
			V_s = np.max(Q)  # V_k+1(s)
			# convergence critaria
			delta = max(delta, np.abs(V_s - V[s]))
			# Update value-function:
			V[s] = V_s
		# at each iteration cheak if delta for all state is less than expected theta
		if itr % 20 ==0:
			assignment_report(V, N ,itr)
		if delta < theta:
			print('Converged after {} number of iterations'.format(itr))
			break

	# Create a deterministic policy using the optimal value function
	policy = np.zeros([env.nS, env.nA])
	for s in range(env.nS):
		Q = one_step_lookahead(s, V)
		best_action = np.argmax(Q)
		# all other are zero and makes it deterministic policy
		policy[s, best_action] = 1

	return policy, V


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--grid_size', default=5, type=int)
	parser.add_argument('--prob_p', default=0.9, type=float)
	parser.add_argument('--Final_Result', default=False, type=bool)
	args = parser.parse_args()
	return args


def Print_Final_Result(policy, V,N):
	#print("Policy Probability Distribution:")
	#print(policy)
	#print("")

	print("Reshaped Grid- Final Policy (0=up, 1=right, 2=down, 3=left):")
	print(np.around( np.reshape(np.argmax(policy, axis=1), (N,-1)) ))
	print("")

	#print("Value Function:")
	#print(V)
	#print("")

	print("Reshaped Grid- Final Value Function:")
	print(np.around(V.reshape((N,-1)),2))
	print("")


def assignment_report(V,N,itr):
	print("Value Function after {} iteration".format(itr+1))
	print(np.around(V.reshape((N, -1)), 2))
	# here:
	# reshaped value function according to grid shape
	# np.around truncates to 2 decimal
	print("")

def main():
	args = parse_args()
	N = args.grid_size  # N= 4
	prob_p = args.prob_p  # prop_p = 0.9
	Final_Result = args.Final_Result

	from gridworld import GridworldEnv
	env = GridworldEnv(shape=[N, N], prob_p=prob_p)  # default prob_p = 1.0; change the value to make action stochastic
	policy, V = Value_Iteration(env,N)

	if Final_Result == True:
		Print_Final_Result(policy, V, N)


if __name__ == '__main__':
	main()
