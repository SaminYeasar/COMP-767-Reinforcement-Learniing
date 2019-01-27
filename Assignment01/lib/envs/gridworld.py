import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
	"""
	Grid World environment from Sutton's Reinforcement Learning book chapter 4.
	You are an agent on an MxN grid and your goal is to reach the terminal
	state at the top left or the bottom right corner.

	For example, a 4x4 grid looks as follows:

	T  o  o  o
	o  x  o  o
	o  o  o  o
	o  o  o  T

	x is your position and T are the two terminal states.

	You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
	Actions going off the edge leave you in your current state.
	You receive a reward of -1 at each step until you reach a terminal state.
	"""

	metadata = {'render.modes': ['human', 'ansi']}

	def __init__(self, shape=[4,4], prob_p=1.0):

		"""Inserted prob_p ; default value 1.0
			prob_p = probability of taking action according to policy
			It makes the action STOCHASTIC
		"""
		if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
			raise ValueError('shape argument must be a list/tuple of length 2')

		self.shape = shape
		self._p = prob_p
		nS = np.prod(shape)
		nA = 4

		MAX_Y = shape[0]
		MAX_X = shape[1]

		P = {}
		grid = np.arange(nS).reshape(shape)
		it = np.nditer(grid, flags=['multi_index'])

		while not it.finished:
			s = it.iterindex
			y, x = it.multi_index

			P[s] = {a : [] for a in range(nA)}

			"""Original conditions"""
			#is_done = lambda s: s == 0 or s == (nS - 1)
			# reward = 0.0 if is_done(s) else -1.0

			#######################################################
			# START(1) Changing the "terminal states" and "Terminal rewards"
			#######################################################
			"""
			if at state = 0 or state = MAX_X-1 (for 4X4 grid world it would be at state = 3)
			
				T  o  o  T
				o  o  o  o
				o  o  o  o
				x  o  o  0
			
			"""
			# makes is done only for following conditions
			is_done = lambda s: s == 0 or s == MAX_X-1
			#is_done = lambda s: s == MAX_X-1

			if is_done(s) and s == 0:
				reward = 1.0
			elif is_done(s) and s == MAX_X-1:
				reward = 10.0
			else:
				reward = 0.0
			######################################################
			# END (1)
			######################################################

			# We're stuck in a terminal state
			if is_done(s):
				P[s][UP] = [(1.0, s, reward, True)]
				P[s][RIGHT] = [(1.0, s, reward, True)]
				P[s][DOWN] = [(1.0, s, reward, True)]
				P[s][LEFT] = [(1.0, s, reward, True)]
			# Not a terminal state
			else:
				ns_up = s if y == 0 else s - MAX_X
				ns_right = s if x == (MAX_X - 1) else s + 1
				ns_down = s if y == (MAX_Y - 1) else s + MAX_X
				ns_left = s if x == 0 else s - 1

				# Original arguments
				"""
				P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
				P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
				P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
				P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
				"""
				##############################################################
				# START (2) To enable Stochastic Action
				# Inserting the condition
				# to take policy action with probability "prob_p"
				# else take stochastic action
				##############################################################

				def call_action(n_a):
					if n_a == 0:
						return ns_up
					if n_a == 1:
						return ns_right
					if n_a == 2:
						return ns_down
					if n_a == 3:
						return ns_left

				def pick_na(prob_p, ac):
					"""
					arg:
					a = action to be taken from policy
					prob_p = probability of chosing policy action "a"
					return:
					prob = probability of chosing the action
					n_a = next action
					"""
					x = [prob_p if x == ac else ((1 - prob_p) / 3) for x in range(4)]
					n_a = int(np.random.choice(4, 1, p=x))
					prob_p = x[n_a]
					return prob_p, call_action(n_a)

				prob_0, ac_0 = pick_na(prob_p=self._p, ac=0)
				prob_1, ac_1 = pick_na(prob_p=self._p, ac=1)
				prob_2, ac_2 = pick_na(prob_p=self._p, ac=2)
				prob_3, ac_3 = pick_na(prob_p=self._p, ac=3)
				P[s][UP] = [(prob_0,ac_0, reward, is_done(ns_up))]
				P[s][RIGHT] = [(prob_1,ac_1, reward, is_done(ns_right))]
				P[s][DOWN] = [(prob_2,ac_2, reward, is_done(ns_down))]
				P[s][LEFT] = [(prob_3,ac_3, reward, is_done(ns_left))]

				#######################################################
				# END(2)
				#######################################################

			it.iternext()

		# Initial state distribution is uniform
		isd = np.ones(nS) / nS

		# We expose the model of the environment for educational purposes
		# This should not be used in any model-free learning algorithm
		self.P = P

		super(GridworldEnv, self).__init__(nS, nA, P, isd)

	def _render(self, mode='human', close=False):
		if close:
			return

		outfile = StringIO() if mode == 'ansi' else sys.stdout

		grid = np.arange(self.nS).reshape(self.shape)
		it = np.nditer(grid, flags=['multi_index'])
		while not it.finished:
			s = it.iterindex
			y, x = it.multi_index

			if self.s == s:
				output = " x "
			elif s == 0 or s == self.nS - 1:
				output = " T "
			else:
				output = " o "

			if x == 0:
				output = output.lstrip()
			if x == self.shape[1] - 1:
				output = output.rstrip()

			outfile.write(output)

			if x == self.shape[1] - 1:
				outfile.write("\n")

			it.iternext()