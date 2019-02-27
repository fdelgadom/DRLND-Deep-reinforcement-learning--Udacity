[//]: # (Image References)

[image1]: ./descarga.png "Visualization"

# Report: "Project 2 - Reacher Continuous Control"

This project involves training a DeepRL agent to solve the Reacher Unity Environment, a continuous control problem.

## Deep-learning algorithm
The learning algorithm implemented for this project is  **Deep Deterministic Policy Gradient(DDPG)** based on both the ddpg-bipedal and ddpg-pendulum examples in the [repository from Udacity](https://github.com/udacity/deep-reinforcement-learning).

DDPG Algorithm is defined in the paper ["Continuous Control With Deep Reinforcement Learning"](https://arxiv.org/pdf/1509.02971.pdf).

DDPG, as described in the paper consists in "adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain" by using an actor-critic framework with a replay buffer

After trying to solve 1 agent version of the environment, and after consulting slack channels the consensus was that 20 agents are a far better option.

So **ddpg_agent.py** needs the addition of the *num_agents* parameter to the **Agent** class, since the environment contained n instances of the virtual reacher "arm" that contributed to learning simultaneously via the two Actor and Critic agents, and the rest of modifications needed.

Also reviewing comments in slack channels, the reform of the noise process due to the fact that is it possible that the OUNoise  implementation in the ddpg implementation of the pendulum is wrong https://drlnd.slack.com/messages/CBMG84E2Y/convo/CBMG84E2Y-1542647394.111400/

**model.py** is the same as in the ddpg-pendulum example.

Another improvement is to make target and local networks start off from the same set of weights as stated in https://drlnd.slack.com/messages/CBMG84E2Y/convo/C9KU4GN6S-1551201562.001200/.

### Hyper Parameters
```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```
The BATCH_SIZE and BUFFER_SIZE are parameters for the ReplayBuffer class, an "memory" randomly sampled at each step to obtain _experiences_ passed into the learn method with a discount of GAMMA. 
LEARNING_RATE for actor and critic is a parameter to the Adam optimizer and also WEIGHT_DECAY
TAU is a parameter for a _soft update_ of the target and local models. 

### Neural Network. Model Architecture & Parameters.
DDPG algorithm needs four separate networks for both Actor and Critic and as in the Udacity examples we use a model with fully-connected linear layers and ReLu activations. 

The Actor is a mapping of state to action values via 3 fully connected Linear layers with relu activation. The final output layer yields 4 action values with tanh activation. 

The Critic is a value function, measuring the quality of the actions via 3 fully connected Linear layers with relu activation with the third layer yielding the single output value.


### Plot of Rewards

![alt text][image1]

Environment solved in 247 episodes.

## Ideas for Future Work

1.- Try new algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

2.- Explore the sensitivity of the hyperparameters and the architecture (particularly the depth) of the Actor and Critic models.

3.- Noise process configuration



