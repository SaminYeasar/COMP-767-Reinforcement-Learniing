# Bandit algorithms 
Assignment 01 Problem 1(a)

 Best-arm identification: also sometimes called pure exploration, this is a problem formulation in which the learner is allowed a period of experimentation of a fixed length, after which it
has to act forever according to the policy it identifies as being best. No further learning is
allowed after the ”pure exploration” phase.

For this part of the assignment you will have to read the following paper:
Best-arm identification algorithms for multi-armed bandits in the fixed confidence setting, Kevin Jamieson and Robert Nowak, CISS, 2014 https://people.eecs.berkeley.edu/˜kjamieson/resources/bestArmSurvey.pdf

The paper describes two different classes of algorithms for this problem. Your task is to: 
(a)summarize the main results in the paper; 
(b) Reproduce the results in Figure 1 
(c) Perform the same empirical comparison on the bandit problem provided in the Sutton & Barto book
(which we discussed in class). Do not forget to average your results over multiple independent runs. 
(d) discuss in a short paragraph a concrete application in which you think regret
optimization would be more useful than best arm identification.
