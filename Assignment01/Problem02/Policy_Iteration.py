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


def Policy_Evaluation(env, policy, stop_itr=10, theta=0.005, discount_factor=0.9):
	def one_step_lookahead(state, V, policy):
		Q = np.zeros(env.nA)
		# for given "action"
		for a, prob_a in enumerate(policy[state]):
			# compute for all possible "next_state" and "action"
			# prob_nS =  probability of next state
			for prob_nS, next_state, reward, done in env.P[state][a]:
				Q[a] += prob_a * prob_nS * (reward + discount_factor * V[next_state])
		return Q

	V = np.zeros(env.nS)  # V_k(s)

	# while True:
	for itr in itertools.count():

		delta = 0
		for s in range(env.nS):
			# one step look-ahead for each possible state
			# it provides Q[state, all possible action]
			Q = one_step_lookahead(s, V, policy)
			# for given "state" get the "Value" for "best possible action"
			# using one-step-lookahead
			V_s = np.sum(Q)  # V_k+1(s)

			# convergence critaria
			delta = max(delta, np.abs(V_s - V[s]))
			# Update value-function:
			V[s] = V_s
		# if itr % 50 == 0:
		#   print(delta)
		# at each iteration cheak if delta for all state is less than expected theta
		if delta < theta:
			print('Running Policy Iteration; Value function Converged after {} iterations'.format(itr))
			break
		#if itr >= 1000000:
		#	print('Value function has not converged')
		#	break
	return V


#

def Policy_Improvement(env, policy, V, discount_factor=0.9):
	def one_step_lookahead(state, V):
		Q = np.zeros(env.nA)
		# for given "action"
		for a in range(env.nA):
			# compute for all possible "next_state"
			for prob, next_state, reward, done in env.P[state][a]:
				Q[a] += prob * (reward + discount_factor * V[next_state])
		return Q

	policy_stable = True
	for s in range(env.nS):
		# one step look-ahead for each possible state
		# it provides Q[state, all possible action]
		Q = one_step_lookahead(s, V)
		# for given "state" get the "Value" for "best possible action"
		# using one-step-lookahead

		# condition to stop policy improvement
		if np.argmax(policy[s, :]) != np.argmax(Q):  # V_k+1(s)
			policy_stable = False
		policy[s] = np.eye(env.nA)[np.argmax(Q)]

	return policy, policy_stable


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
	print(np.reshape(np.argmax(policy, axis=1), (N,-1)))
	print("")

	#print("Value Function:")
	#print(V)
	#print("")

	print("Reshaped Grid- Final Value Function:")
	print(np.around(V.reshape((N,-1)),2))
	print("")


def assignment_report(V,N):
	print("Value Function:")
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

	# Start with a random policy
	policy = np.ones([env.nS, env.nA]) / env.nA

	epochs = 10
	for i in range(epochs):
		print("Epoch {}".format(i))
		# optimize value function for current policy
		V = Policy_Evaluation(env, policy)

		# Print value function after each Iteration
		assignment_report(V,N)

		# improve current policy
		new_policy, policy_stable = Policy_Improvement(env, policy, V)

		if policy_stable == True:
			print('Policy Iteration Converged after {} iteration'.format(i + 1))
			break
		print('Policy has not converged, updating new policy')
		print('---------------------------------------------')
		policy = new_policy

	if Final_Result == True:
		Print_Final_Result(policy, V, N)


if __name__ == '__main__':
	main()
