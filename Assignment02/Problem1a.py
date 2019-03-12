import numpy as np
import matplotlib.pyplot as plt
import gym
import itertools

def Boltzmann_Exploration (Q_val,temp):
    prob = np.exp(Q_val/temp)/ np.sum(np.exp(Q_val/temp))
    action = np.random.choice(len(Q_val), 1, p = prob)[0]
    #action = np.argmax(prob)
    return action, prob

def Compute_TDtarget(reward ,gamma ,Q ,next_state ,temp ,algo):
    output = {}
    if algo == 'Qlearning':
        td_target = reward + gamma *np.max(Q[next_state ,:])
        output['td_target'] = td_target

    if algo == 'SARSA':
        next_action ,_ = Boltzmann_Exploration (Q[next_state ,:] ,temp)
        td_target =reward + gamma *Q[next_state ,next_action]
        output['td_target'] = td_target
        output['next_action'] = next_action

    if algo == 'Expected_SARSA':
        next_action ,prob = Boltzmann_Exploration (Q[next_state ,:] ,temp)
        td_target =reward + gamma* np.sum(prob *Q[next_state ,:])
        output['td_target'] = td_target
        output['next_action'] = next_action
    return output



def Agent(env, gamma, temp, alpha, Q, algo, arg='train'):
    rewards = []
    # For #num of episodes
    if arg == 'train':
        episodes = 10
    elif arg == 'eval':
        episodes = 1

    for episode in range(episodes):
        # initialize state,S
        state = env.reset()
        done = False
        episodic_reward = 0

        # Repeat for each step of episode
        for timesteps in itertools.count():

            if arg == 'train':
                if algo == 'Qlearning':
                    # Choose action: Use Boltzmann_Exploration
                    action, _ = Boltzmann_Exploration(Q[state, :], temp)
                    # action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(episode+1)))
                    # action = egreedy_policy(Q[state,:])
                    next_state, reward, done, _ = env.step(action)
                    # compute TD target:
                    output = Compute_TDtarget(reward, gamma, Q, next_state, temp, algo='Qlearning')
                    td_target = output['td_target']
                    td_error = td_target - Q[state, action]
                    # Q update
                    Q[state, action] = Q[state, action] + alpha * td_error

                if algo == 'SARSA' or algo == 'Expected_SARSA':
                    # Choose action: Use Boltzmann_Exploration
                    if timesteps == 0: action, _ = Boltzmann_Exploration(Q[state, :], temp)
                    # action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(episode+1)))
                    # action = egreedy_policy(Q[state,:])
                    next_state, reward, done, _ = env.step(action)
                    # compute TD target:
                    output = Compute_TDtarget(reward, gamma, Q, next_state, temp, algo)
                    td_target = output['td_target']
                    td_error = td_target - Q[state, action]
                    # Q update
                    Q[state, action] = Q[state, action] + alpha * td_error
                    action = output['next_action']

            elif arg == 'eval':

                # take greedy action
                action = np.argmax(Q[state, :])
                next_state, reward, done, _ = env.step(action)

            episodic_reward += reward
            state = next_state

            if done == True or timesteps >= 10000:
                break
        rewards.append(episodic_reward)

    return rewards, Q


def run(alpha, temp, algo):
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
        train_rewards = []
        eval_rewards = []
        for segment in range(100):
            epi_train_r, Q = Agent(env, gamma,temp, alpha, Q, algo, arg='train')
            epi_eval_r, _ = Agent(env, gamma, temp, alpha, Q, algo, arg='eval')

            train_rewards.append(epi_train_r)
            eval_rewards.append(epi_eval_r)
        # mean train rewards over last 10 episodes:
        mean_train_r.append(np.array(train_rewards)[-1].mean())
        final_eval_r.append(np.array(eval_rewards)[-1][0])

        results['train'] = mean_train_r
        results['eval'] = final_eval_r
        print('mean training reward: {} | final evaluation reward:{}'.format(mean_train_r[-1], final_eval_r[-1]))
    return results






def get_data(alphas, temps, algo='SARSA'):
    # alphas = np.linspace(0.1,1,10)


    SARSA_temp = {}

    for temp in temps:
        SARSA_results = []
        for alpha in alphas:
            SARSA_results.append(run(alpha, temp, algo=algo))
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
temps = [0.9, 0.6, 0.3]
result = get_data(alphas, temps, algo='Qlearning')
get_figures(result, alphas, temps, arg='eval')