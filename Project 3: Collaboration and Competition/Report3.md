[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: ./descarga.png "Visualization"
[image3]: ./Screen-Shot-2018-08-29-at-17.31.26-1024x372.png "DRLA"



# Report: "Project 3 - Collaboration and Competition"

This project involves training train a system of DeepRL agents to demonstrate collaboration or cooperation on a complex task, by solving the Unity Tennis environment.

![alt text][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Deep-learning algorithm

Due to the fact that the environment is totally simetric, it is possible to reform **Deep Deterministic Policy Gradient(DDPG)** algorithm, instead of the multi-agent version **MADDPG** 

The solution is based on both the ddpg-bipedal and ddpg-pendulum examples in the [repository from Udacity](https://github.com/udacity/deep-reinforcement-learning).

DDPG Algorithm is defined in the paper ["Continuous Control With Deep Reinforcement Learning"](https://arxiv.org/pdf/1509.02971.pdf)

DDPG, as described in the paper consists in "adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain" by using an actor-critic framework with a replay buffer

**ddpg_agent.py** needs the addition of the *num_agents* parameter to the **Agent** class, since the environment contained 2 two agents controlling rackets.

Also reviewing comments in slack channels, the reform of the noise process due to the fact that is it possible that the OUNoise  implementation in the ddpg implementation of the pendulum is wrong https://drlnd.slack.com/messages/CBMG84E2Y/convo/CBMG84E2Y-1542647394.111400/
**model.py** is the same as in the ddpg-pendulum example 

Another improvement tried and later discarded was to make target and local networks start off from the same set of weights as stated in https://drlnd.slack.com/messages/CBMG84E2Y/convo/C9KU4GN6S-1551201562.001200/

### DDPG Hyper Parameters
n_episodes (int): maximum number of training episodes = 5000
max_t (int): maximum number of timesteps per episode = 1000


### Agent Hyper Parameters


```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-1              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```
The BATCH_SIZE and BUFFER_SIZE are parameters for the ReplayBuffer class, an "memory" randomly sampled at each step to obtain _experiences_ passed into the learn method with a discount of GAMMA. 
LEARNING_RATE for actor and critic is a parameter to the Adam optimizer and also WEIGHT_DECAY
TAU is a parameter for a _soft update_ of the target and local models. 

### Neural Network. Model Architecture & Parameters.
DDPG algorithm needs four separate networks for both Actor and Critic and as in the Udacity examples we use a model with fully-connected linear layers and ReLu activations. 

The **Actor** is a mapping of state to action values via 3 fully connected **Linear** layers with **relu** activation. The final output layer yields 4 action values with **tanh** activation. 

The **Critic** is a value function, measuring the quality of the actions via 3 fully connected **Linear** layers with **relu** activation with the third layer yielding the single output value.


### Plot of Rewards

![alt text][image2]

Environment solved in 547 episodes. Average Score: 15.01

## Ideas for Future Work
.- Implement **MADDPG** where the actors and critics don't share weights.
.- Try new algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience
![alt text][image3]

["Introducing Reinforcement Learning Coach 0.10.0"](https://www.intel.ai/introducing-reinforcement-learning-coach-0-10-0/#gs.qmrSG6tZ)
.- Explore the sensitivity of the hyperparameters and the architecture (particularly the depth) of the Actor and Critic models.
.- Noise process configuration
.- Methods to control **exploitation vs. exploration dilema** to strike the right balance between having the agent explore the environment to learn how it's shaped and exploit what's learned to achieve the highest possible reward.







