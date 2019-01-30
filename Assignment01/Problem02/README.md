# Markov Decision Processes and dynamic programming 
Assignment 01 Problem 2(a)

Implement and compare empirically the performance of value iteration, policy iteration and
modified policy iteration. Modified policy iteration is a simple variant of policy iteration in
which the evaluation step is only partial (that is, you will make a finite number of backups
to the value function before an improvement step). You can consult the Puterman (1994) textbook for more information. You should implement your code in matrix form. 

Provide your code as well as a summary of the results, which shows, as a function of the number
of updates performed, the true value of the greedy policy computed by your algorithm, from
the bottom left state and the bottom right state. That is, you should take the greedy policy
currently considered by your algorithm and compute its exact value.

# Make the action stochastic:
To test your algorithm, use a grid world in which, at each time step, your actions move in the
desired direction w.p. p and in a random direction w.p. (1 − p). 

# Grid Description:
The grid is empty, of size n × n. There is a positive reward of +10 in the upper right corner and a positive reward of +1 in the upper left corner. All other rewards are 0. The positive-reward states are absorbing (i.e. terminal). If the agent bumps into the edge of the grid as a result of a transition, it stays in the same spot. 

The discount factor is γ = 0.9. You need to test your algorithm with two different values of p (0.9 and 0.7) and with two different sizes of grid (n = 5 and n = 50). Explain what you see in these results.
