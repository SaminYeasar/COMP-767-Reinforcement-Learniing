
import numpy as np
import random
import numpy
from matplotlib import pyplot
import matplotlib.pyplot as plt
import argparse

np.random.seed(32)




mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1)




class NormalBandit:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def reward(self):
        return np.random.normal(self.mu, self.sigma, 1)[0]



class BanditTrials:

    def __init__(self, bandits, n_trials=10, n_time_steps=100):
        self.n_trials = n_trials
        self.n_time_steps = n_time_steps
        self.total_trial_results = []
        self.bandits = bandits

    def run_action_elimination(self):
        self.run_trials(ActionElimination)

    def run_ucb(self):
        self.run_trials(UCB)


    def run_trials(self, strategy):
        self.total_trial_results = []
        h1 = self.H1([b.mu for b in self.bandits])
        for trial_num in np.arange(0, self.n_trials):
            trial = strategy(self.bandits)
            trial.run_trial(time_steps=self.n_time_steps)
            self.total_trial_results.append(trial.pull_count_per_timestep / h1)
            print("Trial {} of {} complete".format(trial_num + 1, self.n_trials), end='\r')



    def H1(self, true_means):
        """ Hardness of the Trial"""
        optimal_mean = np.max(true_means)
        delta = optimal_mean - true_means
        r = [np.power(x,-2) if x!=0 else x for x in delta]
        return np.sum(r)

    def softmax(self, x):
        #    e_x = np.exp(x - np.max(x))
        e_x = np.exp(x)
        return e_x / np.sum(e_x)

    def results_as_probability(self):
        #fill = \
        #np.where(np.isin(max([len(x) for x in self.total_trial_results]), [len(x) for x in self.total_trial_results]))[
        #    0]
        return [self.softmax(result) for result in np.mean(self.total_trial_results, axis=0)]




class ActionElimination:
    def __init__(self, bandits):
        """
        bandit_means
            a list of means that will be used to build the bandits
        r_k = 1
            is the number of samples per epoch for each arm.
        """
        self.bandits = bandits

        self.bandit_count = len(self.bandits)
        self.k = self.bandit_count

        bandit_means = [b.mu for b in bandits]  # get's the mean of bandits [0,1/5,2/5,3/5,4/5,1]
        self.optimal_bandit = np.argmax(bandit_means)  #

        self.rewards_per_arm = [[] for _x in np.arange(0, self.bandit_count)]
        self.delta = 0.1
        self.active_bandits = np.ones(self.bandit_count)
        self.pull_count_per_timestep = []

    def empirical_mean(self, bandit_index):
        r"""
        Calculate the empirical mean of a given bandit (indexed). When an arm is hasn't been pulled, return -Infinity
        """
        if len(self.rewards_per_arm[bandit_index]) == 0:
            return -np.Inf
        return np.mean(self.rewards_per_arm[bandit_index])

    def active_bandit_indexes(self):
        r"""
        self.active_bandits is |n|. Return only the indexes == 1.
        """
        return np.nonzero(self.active_bandits)[0]

    def estimated_best_bandit_mean(self):
        """ returns a tuple with the best bandit index and the empirical mean"""
        all_empirical_means = [self.empirical_mean(idx) for idx, rewards in enumerate(self.bandits)]
        best_arm_index = np.nanargmax(all_empirical_means)
        return (best_arm_index, all_empirical_means[best_arm_index])

    def arm(self, idx):
        return self.bandits[idx]

    def pull_arm(self, idx):
        return self.arm(idx).reward()

    def drop_arm(self, idx):
        self.active_bandits[idx] = 0

    def C_ik(self, bandit_index):
        k = len(self.rewards_per_arm[bandit_index])
        n = self.bandit_count
        if k == 0:
            return 0

        A = np.power(np.pi , 2 ) / 3
        B = n * np.power(k, 2) / self.delta

        return np.sqrt( np.log( A * B) / k  )

    def stopping_condition_reached(self):
        return False
        return len(self.active_bandit_indexes()) == 1

    def run_trial(self, time_steps=500):
        current_epoch = 0
        active_bandits_for_epoch = self.active_bandit_indexes()
        for step in np.arange(0,time_steps):
            # Stopping Condition
            if self.stopping_condition_reached():
                mean = self.estimated_best_bandit_mean()
                print("Stopping. Best Arm: {}. Found in {} time steps".format(mean[0], step))
                print("Estimated mean: {}. ".format(mean[1]))
                print("Empirical mean: {}. ".format(self.arm(self.optimal_bandit).mu))
                break


            for bandit_index in active_bandits_for_epoch:

                self.rewards_per_arm[bandit_index].append(self.pull_arm(bandit_index))

                reference_arm = self.estimated_best_bandit_mean()
                reference_C_t = self.C_ik(reference_arm[0])

                for bandit_idx in self.active_bandit_indexes():
                    candidate_arm_mean = self.empirical_mean(bandit_idx)
                    candidate_C_t = self.C_ik(bandit_idx)
                    lhs = reference_arm[1] - reference_C_t
                    rhs = candidate_arm_mean + candidate_C_t
                    if lhs >= rhs and rhs > (-np.inf):
    #                    print("Dropping:  {}: {} < {}".format(bandit_idx, lhs, rhs ))
                        self.drop_arm(bandit_idx)

            # calculate P(I_t = i)
            if current_epoch > 0:
                self.pull_count_per_timestep.append([len(self.rewards_per_arm[idx]) for idx, _b in enumerate(self.bandits)])

            # increment epoch and reset the list of available bandits
            if step > 0 and step % (self.k - 1) == 0:
                active_bandits_for_epoch = self.active_bandit_indexes()
                current_epoch += 1




class UCB:
    def __init__(self, bandits):
        """
        bandit_means
            a list of means that will be used to build the bandits
        """
        self.bandits = bandits
        self.bandit_count = len(self.bandits)

        bandit_means = [b.mu for b in bandits ]
        self.optimal_bandit = np.argmax(bandit_means)
        self.rewards_per_arm = [[] for _x in np.arange(0, self.bandit_count)]
        self.delta = 0.1
        self.pull_count_per_timestep = []

    def arm(self, idx):
        return self.bandits[idx]

    def pull_arm(self, idx):
        return self.arm(idx).reward()

    def empirical_mean(self, bandit_index):
        r"""
        Calculate the empirical mean of a given bandit (indexed). When an arm is hasn't been pulled, return -Infinity
        """
        if len(self.rewards_per_arm[bandit_index]) == 0:
            return -np.Inf
        return np.mean(self.rewards_per_arm[bandit_index])

    def all_empirical_means(self):
        return [self.empirical_mean(idx) for idx,rewards in enumerate(self.bandits)]

    def estimated_best_bandit_mean(self):
        """ returns a tuple with the best bandit index and the empirical mean"""
        means = self.all_empirical_means()
        best_arm_index = np.nanargmax(means)
        return (best_arm_index, means[best_arm_index])


    def C_ik(self, bandit_index):
        k = len(self.rewards_per_arm[bandit_index])
        n = self.bandit_count
        if k == 0:
            return 0

        A = np.power(np.pi , 2 ) / 3
        B = n * np.power(k, 2) / self.delta
        return np.sqrt( np.log( A * B) / k  )

    def stopping_condition_reached(self):
        return False

    def print_stopping_condition(self, step):
        mean = self.estimated_best_bandit_mean()
        #print("Stopping. Best Arm: {}. Found in {} time steps".format(mean[0], step))
        #print("Estimated mean: {}. ".format(mean[1]))
        #print("Empirical mean: {}. ".format(self.arm(self.optimal_bandit).mu))

    def best_filtered_bandit_index(self, bandit_indexes):
        results = [mean for idx, mean in enumerate(self.all_empirical_means()) if idx in bandit_indexes]
        return (np.argmax(results), results)

    def run_trial(self, time_steps=500):
        for step in np.arange(0, time_steps):
            """
            # Stopping Condition
            if self.stopping_condition_reached():
                self.print_stopping_condition(step)
                break
            """
            # check to see if we haven't sampled a bandit yet:
            unexplored = np.where(np.isinf(self.all_empirical_means()))[0]

            if len(unexplored) != 0:
                # grab the next one:
                best_bandit_index = unexplored[0]
            else:
                best_bandit_index, results = self.best_filtered_bandit_index(np.arange(0, self.bandit_count))
                best_mean = results[best_bandit_index]

                filtered_indexes = np.nonzero(np.select([results != best_mean], [results]))[0]
                second_best_bandit_index, _ = self.best_filtered_bandit_index(filtered_indexes)
                second_best_mean = results[second_best_bandit_index]


                lhs = best_mean - self.C_ik(best_bandit_index)
                rhs = second_best_mean + self.C_ik(second_best_bandit_index)

                if lhs > rhs:
                    self.print_stopping_condition(step)
                    #break

            self.rewards_per_arm[best_bandit_index].append(self.pull_arm(best_bandit_index))
            self.pull_count_per_timestep.append([len(self.rewards_per_arm[idx]) for idx, _b in enumerate(self.bandits)])





#x = np.arange(10)
#np.nonzero(np.select([x != 4], [x]))[0]







def plot_bandits(results,strategy,which_graph,BANDIT_MEANS):
    #fig, ax = plt.subplots()
    #plt.style.use('seaborn-whitegrid')
    time_steps = len(np.array(results).T[0])

    for i in range(len(results[0])):
        plt.plot(np.arange(0, time_steps), [ts[i] for ts in results[:time_steps]],label = 'mu = {}'.format(BANDIT_MEANS[i]))
    plt.xlabel('P(It = i)')
    plt.ylabel('Number of Pulls (units of H1)')
    plt.legend()
    plt.savefig('{}_{}.svg'.format(strategy,which_graph),format='svg', dpi=1200)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument('--MEAN', default=[1, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 0], type=list)
    #parser.add_argument('--SIGMA', default=1/4, type=int)
    parser.add_argument('--run', default='Book_Example_10arm_Bandit')
    parser.add_argument('--Sampling_Strategy', default='UCB')
    parser.add_argument('--n_trials', default=10, type=int)
    parser.add_argument('--n_time_steps', default=2000, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    Sampling_Strategy = args.Sampling_Strategy
    n_trials = args.n_trials
    n_time_steps = args.n_time_steps

    if args.run == 'Reproduce_Paper_Results':
        BANDIT_MEANS = [1, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 0]
        SIGMA = 1/4
    elif args.run == 'Book_Example_10arm_Bandit':
        SIGMA = 1
        while True:
            mu, sigma = 0, 1  # mean and standard deviation
            BANDIT_MEANS = np.random.normal(mu, sigma, 10)
            if  abs(mu - np.mean(BANDIT_MEANS)) < 0.01:
                break


    # BANDIT_MEANS = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # bandits = [NormalBandit(mean, SIGMA) for mean in BANDIT_MEANS]
    # trials = UCBBanditTrial(bandits)
    # trials.run_trial(time_steps=4000)

    bandits = [NormalBandit(mean, SIGMA) for mean in BANDIT_MEANS]
    trials = BanditTrials(bandits, n_trials= n_trials, n_time_steps=n_time_steps)

    if args.Sampling_Strategy == 'Action_Elimination':
        print('Running Action Elimination')
        trials.run_action_elimination()
    elif args.Sampling_Strategy == 'UCB':
        print('Running UCB')
        trials.run_ucb()

    results = trials.results_as_probability()
    plot_bandits(results, args.Sampling_Strategy, args.run, BANDIT_MEANS)

    #trials = BanditTrials(bandits, n_trials=100, n_time_steps=200)
    #trials.run_ucb_trials()
    #results = trials.results_as_probability()
    #plot_bandits(results)


if __name__ == '__main__':
    main()







