Works on Python Version 3.6.4

# Multi-Armed Bandit

A multi-armed bandit has __k__ possible actions in which we could take, each action produces a reward of scalar value. The problem we want to solve is to maximize the reward with each action we take.  

- __exploitation__ - choose action of highest reward
- __exploration__ - choose random action

We want a fair balance between __exploitation__ and __exploration__  
With probability __epsilon__ choose __exploration__ otherwise __exploitation__, where:
 - epsilon = [0,1]
 - epsilon usually = 0.1
 
#### Types of Multi-Armed Bandits
- __Static__ - rewards of the k possible actions DOES NOT CHANGE over time
- __Dynamic__ - rewards of the k possible actions DOES CHANGE over time
- __Contextual__ - involves multiple bandits. [details below](#associative-search-aka-contextual-bandits)

## Algorithms to Estimate Reward Values

There are 2: __Sample Average__ and __Constant Step Size__   
<code>Q[]</code> is an array that holds the reward estimates of the Multi-Armed Bandit

#### Sample Average Estimate
- different initial q estimate values __does not affect__ performance in sample average
- works better for __Static__ Multi-Armed Bandits
- after each action <code>a</code> taken and the reward <code>r</code> it produced, update <code>Q[]</code> estimates  
  <code>Q[a] = Q[a] + (1/n)(r - Q[a])</code> where:
  - <code>n</code> - number of times action <code>a</code> was taken

#### Constant Step Size Estimate
- different initial q estimate values __does affect__ performance in sample average
- works better for __Dynamic__ Multi-Armed Bandits
- after each action <code>a</code> taken and the reward <code>r</code> it produced, update <code>Q[]</code> estimates  
  <code>Q[a] = Q[a] + α(r - Q[a])</code> where:
  - <code>α</code> - is a scalar number ∈ (0,1]
- __Choosing Good Initial Q Estimates__ - Optimistic Initial Values
  - choosing initial q estimate values greater than all actual rewards encourages exploration temporarily in beginning
  - The result is that all actions are tried several times before the value estimates converge.
  - The system does a fair amount of exploration even if greedy actions are selected all the time   
  ![Constant Step Size - Initial Q Estimates Graph.png](https://raw.githubusercontent.com/TheRealMarcusChiu/machine-learning-master-example/master/reinforcement-learning/multi-arm-bandit/readme/Constant%20Step%20Size%20-%20Initial%20Q%20Estimates%20Graph.png)

## Choosing Actions - Balancing Exploitation & Exploration

#### Epsilon  
As described above , when epsilon = 0.1, the action to take:
- exploit 90% of the time
- explore 10% of the time

#### Upper Confidence Bound (UCB)  
Action to take is based on the following equation:  
<code>action = argmaxₐ [Q(a) + c * root(ln(t)/N(a))]</code>
   
![Upper Confidence Bound - Action Selection Graph.png](https://github.com/TheRealMarcusChiu/machine-learning-master-example/blob/master/reinforcement-learning/multi-arm-bandit/readme/Upper%20Confidence%20Bound%20-%20Action%20Selection%20Graph.png?raw=true)

## Gradient Bandit Algorithms
Instead of estimating action values (like in __Sample Average__ and __Constant Step Size__), we could estimate the __preference__ for each action <code>a</code>, denoted as <code>H(a)</code>  
Larger the __preference__ the more often the action is taken

<code>H[]</code> - is an array of preference estimates for each action <code>a</code>
- initially all preferences are the same (e.g., H[a] = 0, for all a)
- to choose an action, use softmax distribution (Gibbs or Boltzmann distribution):  
<code>π(a)</code> = <code>Pr{A=a}</code> = <code>exp(H[a]) / [summation (exp(H[b])) of each action b of all possible actions]</code>
- after selecting action <code>a</code> and receiving reward <code>r</code>, update __preferences__:  
<code>H[a] = H[a] + α(r - r ⃰)(1 - π(a))</code> and  
<code>H[b] = H[b] - α(r - r ⃰)π(b)</code> for all actions b != a, where:
  - <code>α</code> - step size > 0
  - <code>r ⃰</code> - average of all rewards, including <code>r</code>
  - <code>π(a)</code> - the softmax distribution  

![gradient bandit algorithms.png](https://github.com/TheRealMarcusChiu/machine-learning-master-example/blob/master/reinforcement-learning/multi-arm-bandit/readme/gradient%20bandit%20algorithms.png?raw=true)

## Associative Search (aka Contextual Bandits)

__non-associative tasks__ - tasks in which there is no need to associate different actions with different situations (one situation)
In these tasks, the learner:
 - tries to find a single best action when the task is stationary 
 - or tries to track the best action as it changes over time when the task is non-stationary
 
__associative tasks__ - tasks in which there is an association between actions and situations (multiple situations)
In these tasks, the learner:
 - including the non-associative points
 - learns a policy: a mapping from situations to the actions that are best in those situations.
 
Most general reinforcement problems contain __associative tasks__

__A simple example:__ suppose there are several different k-armed bandit tasks, and that on each step you confront one of these chosen at random. Each bandit has a distinct color.

This is an example of an __associative search__ task, so called because it involves both trial-and-error learning to _search_ for the best actions, and _association_ of these actions with the situations in which they are best

__Associative search__ tasks are intermediate between the __k-armed bandit__ problem and the __full reinforcement learning__ problem. They are like the full reinforcement learning problem in that they involve learning a policy, but like our version of the k-armed bandit problem in that each action affects only the immediate reward. If actions are allowed to affect the _next situation_ as well as the reward, then we have the full reinforcement learning problem.
    
## Summary

![comparison.png](https://github.com/TheRealMarcusChiu/machine-learning-master-example/blob/master/reinforcement-learning/multi-arm-bandit/readme/comparison.png?raw=true)
