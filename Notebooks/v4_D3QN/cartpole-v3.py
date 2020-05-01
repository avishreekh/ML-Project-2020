'''
Cartpole - Task 1 | D3QN | v2

RL: D3QN model
We will try D3QN
D3QN: Dueling Double Deep Q Network

agent will use 2 networks
> online network (Q): returns Q values for left and right | action with higher Q is chosen
( essentially the DQN network )
> target network: (see replay function)

based on the premise that Q value comprises of:
> the value of being in the current state (V(s))
> the advantage of choosing an action a in a state s (A(s,a))

Why separate the two components?
> Suppose, the state you're in leads to "DEATH", no matter what the action,
it would be a futile exercise to waste training resources to calculate an action's
advantage

see nn_model for more detail
'''

'''
Task 1: env
'''
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CartPoleEnv(gym.Env):
	metadata = {
			'render.modes': ['human', 'rgb_array'],
			'video.frames_per_second' : 50
	}

	def __init__(self,case=1):
		self.__version__ = "0.2.0"
		print("CartPoleEnv - Version {}, Noise case: {}".format(self.__version__,case))
		self.gravity = 9.8
		self.masscart = 1.0
		self.masspole = 0.4
		self.total_mass = (self.masspole + self.masscart)
		self.length = 0.5
		self.polemass_length = (self.masspole * self.length)
		self.seed()

		self.force_mag = 10.0
		#self.force_mag = 10.0*(1+self.np_random.uniform(low=-0.10, high=0.10))


		self.tau = 0.02  # seconds between state updates
		self.frictioncart = 5e-4 # AA Added cart friction
		self.frictionpole = 2e-6 # AA Added cart friction
		self.gravity_eps = 0.99 # Random scaling for gravity
		self.frictioncart_eps = 0.99 # Random scaling for friction
		self.frictionpole_eps = 0.99 # Random scaling for friction

		# Angle at which to fail the episode
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4

		# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
		high = np.array([
			self.x_threshold * 2,
			np.finfo(np.float32).max,
			self.theta_threshold_radians * 2,
			np.finfo(np.float32).max])

		self.action_space = spaces.Discrete(2) # AA Set discrete states back to 2
		self.observation_space = spaces.Box(-high, high)

		self.viewer = None
		self.state = None

		self.steps_beyond_done = None

	def seed(self, seed=None): # Set appropriate seed value
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		x, x_dot, theta, theta_dot = state
		force = self.force_mag if action==1 else -self.force_mag
		costheta = math.cos(theta)
		sintheta = math.sin(theta)
		temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta - self.frictioncart * (4 + self.frictioncart_eps*np.random.randn()) *np.sign(x_dot)) / self.total_mass # AA Added cart friction
		thetaacc = (self.gravity * (4 + self.gravity_eps*np.random.randn()) * sintheta - costheta* temp - self.frictionpole * (4 + self.frictionpole_eps*np.random.randn()) *theta_dot/self.polemass_length) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass)) # AA Added pole friction
		xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
		noise = 0
		#noise = self.np_random.uniform(low=-0.10, high=0.10)
		x  = (x + self.tau * x_dot)
		x_dot = (x_dot + self.tau * xacc)
		theta = (theta + self.tau * theta_dot)*(1 + noise)
		theta_dot = (theta_dot + self.tau * thetaacc)
		self.state = (x,x_dot,theta,theta_dot)
		done =  x < -self.x_threshold \
				or x > self.x_threshold \
				or theta < -self.theta_threshold_radians \
				or theta > self.theta_threshold_radians
		done = bool(done)

		if not done:
			reward = 1.0
		elif self.steps_beyond_done is None:
			# Pole just fell!
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0

		return np.array(self.state), reward, done, {}

	def reset(self):
		self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		self.steps_beyond_done = None
		return np.array(self.state)

	def render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		screen_width = 600
		screen_height = 400

		world_width = self.x_threshold*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * 1.0
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

		if self.state is None: return None

		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])
		return self.viewer.render(return_rgb_array = mode=='rgb_array')


# general purpose libraries
import os
import random
import pylab
from collections import deque

# DL modules
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add
from keras.optimizers import Adam, RMSprop
from keras.layers import Dropout
from keras import backend as K
# defining the Environment | v1 allows for a max score of 500, can be changed in __init__.py


# playing random games
def play_random_games():
	for episode in range(10):
		env.reset()

		for step in range(500):
			action = env.action_space.sample() # picking an action at random

			# executing the action
			observation, reward, done, info = env.step(action)
			# print
			print(step, observation, reward, done, info, action)
			if done:
				break

# play_random_games()

'''
now, for our neural net models
we will employ keras
'''

def nn_model(inp_shape, action_space, d3qn):
	X_inp = Input(inp_shape)
		# symmetric hidden layers :)
	X = Dense(64, input_shape=inp_shape, activation='relu', kernel_initializer='he_uniform')(X_inp)
	X = Dense(256, input_shape=inp_shape, activation='relu', kernel_initializer='he_uniform')(X)
	X = Dense(64, input_shape=inp_shape, activation='relu', kernel_initializer='he_uniform')(X)
	# PREV LAYERS: 128 | 128

	if d3qn:
		'''
		IMPLEMENTATION: split the layers
		> 1 for value, and 1 for each action
		> aggregate them using mean of the advantages to normalize

		'''

		# V(s)
		value = Dense(1, kernel_initializer='he_uniform')(X)
		value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(value)

		# A(s,a)
		advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
		advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(advantage)

		# combining
		X = Add()([value, advantage])
	else:
		# output layer | binary prediction
		X = Dense(action_space, activation='linear', kernel_initializer='he_uniform')(X)

	# compiling the model
	model = Model(inputs = X_inp, outputs = X, name='cartpole-v3')
	model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=['accuracy'])

	model.summary()
	return model


# agent class
class Agent:
	def __init__(self):
		self.env = CartPoleEnv()
		# self.env = gym.make('CartPole-v1')
		self.env.seed(0) # seeded
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		self.episodes = 1000
		self.memory = deque(maxlen=2000)

		self._max_episode_steps = 500
		self.gamma = 0.99 # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.001 # amount of randomness to always persist
		self.epsilon_decay = 0.999
		self.batch_size = 64
		self.train_start = 1000

		# model parameters | discriminate between DQN, DDQN and D3QN | activation of softupdate
		self.ddqn = True
		self.d3qn = True
		self.soft = True

		# TAU | soft update hyperparameter
		self.TAU = 0.9

		# evaluation parameters
		self.scores, self.episode_list, self.average = [], [], []

		# model definitions
		self.model = nn_model(inp_shape=(self.state_size,),
							action_space = self.action_size,
							d3qn = self.d3qn)
		self.target_model = nn_model(inp_shape=(self.state_size,),
							action_space = self.action_size,
							d3qn = self.d3qn)

	def target_model_refresh(self):
		'''
		assign the same weights as the online network
		after some time interval

		SOFT TARGET UPDATE:
		incremental increase in weights
		'''
		if self.ddqn and not self.soft:
			self.target_model.set_weights(self.model.get_weights())
			return
		if self.ddqn and self.soft:
			online_weights = self.model.get_weights()
			target_weights = self.target_model.get_weights()

			# soft update | TAU = 0.1
			i = 0
			for o_weight, t_weight in zip(online_weights, target_weights):
				t_weight = t_weight * (1-self.TAU) + o_weight * (self.TAU)
				target_weights[i] = t_weight
				i += 1

			# assign new weights to target models
			self.target_model.set_weights(target_weights)
			return

	def recall(self, curr_state, action, reward, next_state, done):
		self.memory.append((curr_state, action, reward, next_state, done))
		if len(self.memory) > self.train_start:
			if self.epsilon > self.epsilon_min:
				self.epsilon = self.epsilon*self.epsilon_decay
		'''
		epsilon is the probability of our agent randomly executing an action
		as opposed to executing the action that was predicted by the Network

		we tend to decrease this value as the efficiency of our model increases
		'''

	def act(self, state):
		if np.random.random() <= self.epsilon:
			# we return a random action | explore
			return random.randrange(self.action_size)
		else:
			# return the predicted action
			return np.argmax(self.model.predict(state))

	def replay(self):
		'''
		used to train the neural net with past experiences
		experiences are picked at random in small buckets
		'''
		if len(self.memory) < self.train_start:
			return

		# random sampling
		bucket = random.sample(self.memory, self.batch_size )

		curr_state = np.zeros((self.batch_size, self.state_size))
		next_state = np.zeros((self.batch_size, self.state_size))
		action, reward, done = [], [], []

		# this should work? if not, use a loop
		# DEBUG: ValueError: too many values to unpack (expected 5)
		for i in range(self.batch_size):
			curr_state[i] = bucket[i][0]
			action.append(bucket[i][1])
			reward.append(bucket[i][2])
			next_state[i] = bucket[i][3]
			done.append(bucket[i][4])

		# predict targets , i.e. the action vector

		# Q-value for curr_state using main network
		target = self.model.predict(curr_state)
		# action predicted for next_state by main network
		target_next = self.model.predict(next_state)
		# Q-value for next_state using target network
		target_value = self.target_model.predict(next_state)

		# UPDATE batch
		for i in range(len(bucket)):
			if done[i]:
				target[i][action[i]] = reward[i]
				# if done, we don't need to update the DQ-Network
			else:
				'''
				DDQN uses two identical neural net models, one of them being
				the simple DQN, the other one is a replica of the last episode
				of the DQN model.

				This is to prevent a certain action from gaining immutable advantage
				upon being trained by the DQN (here: reward gets added to the Q-network)

				Hence, if we use the Q values from the previous episode, we would
				essentially bring down the difference between the output values (actions)

				IMPLEMENTATION:
				find index of the highest Q-value from the main model and use that
				index to obtain the action from the secondary model.
				'''
				if self.ddqn:
					a = np.argmax(target_next[i])
					target[i][action[i]] = reward[i] + self.gamma * (target_value[i][a])
				else:
					target[i][action[i]] = reward[i] + self.gamma*(np.amax(target_next[i]))
				# DEBUG: argmax is the value for which the maximum is obtained
				# DEBUG: amax is the maximum along a given axis
				# accounting for current rewards as well as future
				# discounting future rewards so as to prioritize current rewards

		# train neural network
		self.model.fit(curr_state, target, batch_size = self.batch_size, verbose=0 )

	def load(self, name):
		self.model = load_model(name)

	def save(self, name):
		self.model.save(name)

	def execute(self):
		average_score = []
		for ep in range(self.episodes):
			state = self.env.reset()
			state = np.reshape(state, [1, self.state_size]) # see sentdex's video
			iter = 0

			# DEBUG:  UnboundLocalError: local variable 'done' referenced before assignment
			done = False

			# initializing episode reward
			episode_reward = 0

			while not done:
				# self.env.render() -- lite
				action = self.act(state)
				next_state, reward, done, info = self.env.step(action)
				next_state = np.reshape(next_state, [1, self.state_size])

				# updating episode rewards
				episode_reward += reward

				'''
				CODE: debugging issues
				if not done or iter == self._max_episode_steps - 1:
					# _max_episode_steps has been predefined
					# self.env._max_episode_steps-1, could use this
					reward = reward
				else:

					# reset reward
					reward = -100 # just to reset
				'''

				self.recall(state, action, reward, next_state, done)
				state = next_state

				# proceeding ahead
				iter += 1

				# setting a cap on the score
				if iter == 500:
					done=True

				if done:
					# maintaing a list of all scores | average over last 100 episodes to be extracted from this list
					average_score.append(iter)

					# assign online network's weights to target model after each episode
					self.target_model_refresh()

					# plot
					average = self.PlotModel(iter, ep)

					# info -- print
					print("episode: {}/{}, score: {}, e: {:.2}, average: {}, average ( ~ 100 EPISODES ): {}".format(ep, self.episodes, iter, self.epsilon, average, np.mean(average_score[-100:])))

					# checking for a score of 500
					# if iter == 500:
					#     print("saving cartpole-v2")
					#     self.save("cartpole-v2")
					#     return

					# checking for an average of 500 over the last 100 EPISODES
					if ( np.mean(average_score[-100:]) == 500 ):
						print("saving cartpole-v3")
						self.save("cartpole-v3")
						return

				# it will run successfully upon sufficient increase in memory
				self.replay()

	def test(self):
		'''
		testing out our trained neural net
		'''
		self.load('cartpole-v3')
		for ep in range(100):
			state = self.env.reset()
			state = np.reshape(state, [1, self.state_size])

			# proceeding ahead
			iter = 0

			# DEBUG:  UnboundLocalError: local variable 'done' referenced before assignment
			done = False

			while not done:
				self.env.render()
				action = np.argmax(self.model.predict(state))
				next_state, reward, done, info = self.env.step(action)
				state = np.reshape(next_state, [1, self.state_size])
				iter += 1

				# checking for a score of 480
				if iter == 500:
					done = True

				if done:
					print("episode: {}/{}, score: {}".format(ep, 100, iter))
					break

	# function for plotting a graph (PERFORMANCE GRAPH)
	pylab.figure(figsize=(20,10))
	def PlotModel(self, score, episode):
		# REFER: pylessons
		self.scores.append(score)
		self.episode_list.append(episode)
		self.average.append(sum(self.scores) / len(self.scores))
		pylab.plot(self.episode_list, self.average, 'r')
		pylab.plot(self.episode_list, self.scores, 'b')
		pylab.ylabel('Score', fontsize=18)
		pylab.xlabel('Steps', fontsize=18)
		dqn = 'DQN_'
		dueling = ''
		softupdate = ''
		if self.ddqn:
			dqn = 'DDQN_'
		if self.d3qn:
			dueling = '_Dueling'

		return str(self.average[-1])[:5]

# if __name__ == "__main__":
agent = Agent()
agent.execute()
agent.test()
