import numpy as np
import matplotlib.pyplot as plt
import gym
import time

# def Compute_TDtarget(reward ,gamma ,Q ,next_state ,temp ,algo):
#     output = {}
#     if algo == 'Qlearning':
#         td_target = reward + gamma *np.max(Q[next_state ,:])
#         output['td_target'] = td_target
#
#     if algo == 'SARSA':
#         next_action ,_ = Boltzmann_Exploration (Q[next_state ,:] ,temp)
#         td_target =reward + gamma *Q[next_state ,next_action]
#         output['td_target'] = td_target
#         output['next_action'] = next_action
#
#     if algo == 'Expected_SARSA':
#         next_action ,prob = Boltzmann_Exploration (Q[next_state ,:] ,temp)
#         td_target =reward + gamma* np.sum(prob *Q[next_state ,:])
#         output['td_target'] = td_target
#         output['next_action'] = next_action
#     return output



class Agent(object):

    def __init__(self, env, gamma, temp, alpha, Q):
        self.Q = Q
        self.env = env
        self.temp = temp
        self.alpha = alpha
        self.gamma = gamma

    def Boltzmann_Exploration(self, Q_val):
        prob = np.exp(Q_val / self.temp) / np.sum(np.exp(Q_val / self.temp))
        action = np.random.choice(len(Q_val), 1, p=prob)[0]
        return action, prob

    def Qlearning(self, episodes):
        rewards = []
        for episode in range(episodes):
            # initialize state,S
            state = self.env.reset()
            done = False
            episodic_reward = 0
            # Repeat for each step of episode
            while not done:
                # Choose action: Use Boltzmann_Exploration
                action, _ = self.Boltzmann_Exploration(self.Q[state, :])
                next_state, reward, done, _ = self.env.step(action)
                # compute TD target:
                td_target = reward + self.gamma * np.max(self.Q[next_state, :])
                td_error = td_target - self.Q[state, action]
                # Q update
                self.Q[state, action] = self.Q[state, action] + self.alpha * td_error
                # s'=s
                state = next_state
                episodic_reward += reward

            rewards.append(episodic_reward)
        return rewards, self.Q

    def eval(self, episodes):
        rewards = []
        for episode in range(episodes):
            # initialize state,S
            state = self.env.reset()
            done = False
            episodic_reward = 0
            # Repeat for each step of episode
            while not done:
                self.env.render()
                time.pause(0.5)
                # take greedy action
                action = np.argmax(self.Q[state, :])
                next_state, reward, done, _ = self.env.step(action)
                # s'=s
                state = next_state
                episodic_reward += reward

            rewards.append(episodic_reward)
        return rewards



def run(alpha, temp):
    results = {}
    gamma = 0.99  # disocunt factor
    env = gym.make('Taxi-v2')

    # env=CliffWalkingEnv(),
    mean_train_r = []
    final_eval_r = []


    for run in range(10):
        print("-----------------------------------------------------")
        print('alpha: {} | temp: {} | run: {}'.format(alpha, temp, run))
        Q = np.zeros([env.observation_space.n, env.action_space.n])
        policy = Agent(env, gamma, temp, alpha, Q)
        train_rewards = []
        eval_rewards = []
        for segment in range(100):
            epi_train_r, Q = policy.Qlearning(10)
            epi_eval_r = policy.eval(1)

            train_rewards.append(epi_train_r)
            eval_rewards.append(epi_eval_r)
        # mean train rewards over last 10 episodes:
        mean_train_r.append(np.array(train_rewards)[-1].mean())
        final_eval_r.append(np.array(eval_rewards)[-1][0])

        results['train'] = mean_train_r
        results['eval'] = final_eval_r
        results['Q'] = Q
        print('mean training reward: {} | final evaluation reward:{}'.format(mean_train_r[-1], final_eval_r[-1]))
    return results






def get_data(alphas, temps):
    # alphas = np.linspace(0.1,1,10)


    SARSA_temp = {}

    for temp in temps:
        SARSA_results = []
        for alpha in alphas:
            SARSA_results.append(run(alpha, temp))
        SARSA_temp['{}'.format(temp)] = SARSA_results

    return SARSA_temp

def get_figures(SARSA_temp, alphas, temps, arg='eval'):
    for temp in temps:
        data = []
        for i in range(len(alphas)):
            data.append(np.mean(SARSA_temp['{}'.format(temp)][i]['{}'.format(arg)]))
        plt.plot(data, label='temp = {}'.format(temp))
        plt.legend()
    plt.show()

alphas = [0.3, 0.6, 0.9]
temps = [0.9]
result = get_data(alphas, temps)
get_figures(result, alphas, temps, arg='eval')