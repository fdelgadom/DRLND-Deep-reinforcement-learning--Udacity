# Report: "Project 1 - Navigation"

This project involves training a DeepRL agent to solve the Banana Unity Environment, a navigation problem.

## Deep-learning algorithm
The learning algorithm implemented for this project is a **Deep Q-Network (DQN)** based on the Deep Q Network Lunar Lander example in the [lesson from Udacity](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/dqn_agent.py). DQN Methods are defined in the paper ["Human-level control through deep reinforcement learning"](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/) 

### Hyper Parameters
```python
BATCH_SIZE = 64         # minibatch size
BUFFER_SIZE = 10000     # replay buffer size
GAMMA = 0.99            # discount factor
LEARNING_RATE = 0.0005  # learning rate 
TAU = 0.001             # for soft update of target parameters
UPDATE_EVERY = 4        # how often to update the network
```
The BATCH_SIZE and BUFFER_SIZE are parameters for the ReplayBuffer class, an "memory" randomly sampled at each step to obtain _experiences_ passed into the learn method with a discount of GAMMA. 
LEARNING_RATE is a parameter to the Adam optimizer. 
TAU is a parameter for a _soft update_ of the target and local models. 
UPDATE_EVERY determines the number of steps before learning from a new sample.
### Neural Network. Model Architecture & Parameters.
The Deep Q-Learning algorithm uses two separate networks with identical model architectures.

As in the Udacity exercise we use a model with fully-connected linear layers and ReLu activations. 
The fully-connected layers are the mapping of state to action values with relu activation and dropout to prevent overfitting. 
The input layer is a fully-connected linear layer with 37 inputs (features). These inputs are the 37 signals that define the status of our environment.  
The final output layer is a fully-connected linear layer with a single output for each valid action, in our environment 4 outputs (features) The action selection is made via argmax and using Epsilon-greedy	logic to allow random exploration.

Model:
+ (fc1): Linear(in_features=37, out_features=64, bias=True)
+ (fc2): Linear(in_features=64, out_features=64, bias=True)
+ (fc3): Linear(in_features=64, out_features=4, bias=True)


### Plot of Rewards

Environment solved in 547 episodes. Average Score: 15.01

## Ideas for Future Work

Future ideas for improving the agent's performance.
1. Implement Double DQN, as described by van Hasselt et al in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), "can decouple the selection from the evaluation" by using [Double Q-learning](http://papers.nips.cc/paper/3964-double-q-learning.pdf) in a deep learning network context.
2. Try other new improvements to the original Deep Q-Learning algorithm:
    1. Prioritized Experience Replay
    2. Dueling DQN
    3. Learning from multi-step bootstrap targets (A3C)
    4. Distributional DQN
    5. Noisy DQN
Researchers at Google DeepMind have been testing the performance of an agent that incorporated all these modifications in the Rainbow paper [Rainbow: Combining Improvements in Deep Reinforcement Learning ](https://arxiv.org/pdf/1710.02298.pdf)
