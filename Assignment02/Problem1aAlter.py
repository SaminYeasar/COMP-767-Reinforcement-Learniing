import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline

# For animation
from IPython.display import clear_output
from time import sleep


class Agent(object):
	def __init__(self, method, start_alpha=0.3, start_gamma=0.9, start_epsilon=0.5):
		"""method: one of 'q_learning', 'sarsa' or 'expected_sarsa' """
		self.method = method
		self.env = gym.make('Taxi-v2')
		self.n_squares = 25
		self.n_passenger_locs = 5
		self.n_dropoffs = 4
		self.n_actions = self.env.action_space.n
		self.epsilon = start_epsilon
		self.gamma = start_gamma
		self.alpha = start_alpha
		# Set up initial q-table
		self.q = np.zeros(shape=(self.n_squares * self.n_passenger_locs * self.n_dropoffs, self.env.action_space.n))
		# Set up policy pi, init as equiprobable random policy
		self.pi = np.zeros_like(self.q)
		for i in range(self.pi.shape[0]):
			for a in range(self.n_actions):
				self.pi[i, a] = 1 / self.n_actions

	def simulate_episode(self):
		s = self.env.reset()
		done = False
		r_sum = 0
		n_steps = 0
		gam = self.gamma
		while not done:
			n_steps += 1
			# take action from policy
			x = np.random.random()
			a = np.argmax(np.cumsum(self.pi[s, :]) > x)
			# take step
			s_prime, r, done, info = self.env.step(a)
			if self.method == 'q_learning':
				a_prime = np.random.choice(np.where(self.q[s_prime] == max(self.q[s_prime]))[0])
				self.q[s, a] = self.q[s, a] + self.alpha * \
							   (r + gam * self.q[s_prime, a_prime] - self.q[s, a])
			elif self.method == 'sarsa':
				a_prime = np.argmax(np.cumsum(self.pi[s_prime, :]) > np.random.random())
				self.q[s, a] = self.q[s, a] + self.alpha * \
							   (r + gam * self.q[s_prime, a_prime] - self.q[s, a])
			elif self.method == 'expected_sarsa':
				self.q[s, a] = self.q[s, a] + self.alpha * \
							   (r + gam * np.dot(self.pi[s_prime, :], self.q[s_prime, :]) - self.q[s, a])
			else:
				raise Exception("Invalid method provided")
			# update policy
			best_a = np.random.choice(np.where(self.q[s] == max(self.q[s]))[0])
			for i in range(self.n_actions):
				if i == best_a:
					self.pi[s, i] = 1 - (self.n_actions - 1) * (self.epsilon / self.n_actions)
				else:
					self.pi[s, i] = self.epsilon / self.n_actions

			# decay gamma close to the end of the episode
			if n_steps > 185:
				gam *= 0.875
			s = s_prime
			r_sum += r
		return r_sum

		best_a = np.random.choice(np.where(self.q[s] == max(self.q[s]))[0])
		for i in range(self.n_actions):
			if i == best_a:
				self.pi[s, i] = 1 - (self.n_actions - 1) * (self.epsilon / self.n_actions)
			else:
				self.pi[s, i] = self.epsilon / self.n_actions


def train_agent(agent, n_episodes=200001, epsilon_decay=0.99995, alpha_decay=0.99995, print_trace=False):
	r_sums = []
	for ep in range(n_episodes):
		r_sum = agent.simulate_episode()
		# decrease epsilon and learning rate
		agent.epsilon *= epsilon_decay
		agent.alpha *= alpha_decay
		if print_trace:
			if ep % 20000 == 0 and ep > 0:
				print("Episode:", ep, "alpha:", np.round(agent.alpha, 3), "epsilon:", np.round(agent.epsilon, 3))
				print("Last 100 episodes avg reward: ", np.mean(r_sums[ep - 100:ep]))
		r_sums.append(r_sum)
	return r_sums


# Create agents
sarsa_agent = Agent(method='sarsa')
e_sarsa_agent = Agent(method='expected_sarsa')
q_learning_agent = Agent(method='q_learning')

# Train agents
r_sums_sarsa = train_agent(sarsa_agent, print_trace=True)
r_sums_e_sarsa = train_agent(e_sarsa_agent, print_trace=True)
r_sums_q_learning = train_agent(q_learning_agent, print_trace=True)

df = pd.DataFrame({"Sarsa": r_sums_sarsa,
				   "Expected_Sarsa": r_sums_e_sarsa,
				   "Q-Learning": r_sums_q_learning})
df_ma = df.rolling(100, min_periods=100).mean()
df_ma.iloc[1:1000].plot()