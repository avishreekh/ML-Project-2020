# ML-Project-2020

This project aims to develop Reinforcement Learning agents that solve the given modified CartPole Environments. A detailed description of the problem statement can be found in the `Details` folder. 

## Environments
The environments are modified versions of the classic CartPole-v1 environment. The details of the environments can be found in the `Details` folder.

## Approaches
We tried the following approaches:
1. REINFORCE
2. Deep Q-Network (DQN)
3. Double Deep Q-Network (DDQN)
4. Double Deep Q-Network with soft update (DDQN_with_soft_update)
5. Dueling Double Deep Q-Network (D3QN)

A detailed report of the approaches and results can be found in the `Results` folder. The notebooks containing the code can be found in the `Notebooks` folder. The weights for the networks can be found in the `Weights` folder.

## Results
The best result was obtained using DDQN with soft update. We could achieve a perfect score on all the environments using a neural network with 2 layers with 4 neurons each.

A detailed report of the approaches and results can be found in the `Results` folder.

## Technical specifications
The notebooks are written in Python3 and use the following major libraries/frameworks:
1. Deep Learning framework: Keras, PyTorch
2. Environment providing framework: OpenAI Gym
3. Data processing: Numpy, Pandas, Scikit-learn

## References/Resources
1. Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement
learning. Mach. Learn., 1992
2. V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra M. Riedmiller
Playing Atari with Deep Reinforcement Learning, 2013
3. H. V. Hasselt, A. Guez, D. Silver, Deep Reinforcement Learning with Double Q-learning, 2015
4. [PyTorch's tutorial on REINFORCE](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)
5. [RL tutorials by Python Lessons](https://pylessons.com/CartPole-reinforcement-learning/)
