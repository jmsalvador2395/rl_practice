import numpy as np
import gym
import random
from gym.utils.play import *
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from utilities import data_point
import math
import datetime
import sys


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

torch.set_printoptions(precision=10)
model_path='../models/'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

class dqn(nn.Module):
	def __init__(self, actions):
		super(dqn, self).__init__()
		layer1 = nn.Sequential(
			nn.Conv2d(4, 16, kernel_size=8, stride=4, padding=(2, 2)),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=4, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		fc1 = nn.Sequential(
			nn.Linear(32*12*18, 256),
			nn.LayerNorm(256),
			nn.ReLU()
		)

		fc2 = nn.Sequential(
			nn.Linear(256, 256),
			nn.LayerNorm(256),
			nn.ReLU()
		)

		fc3 = nn.Linear(256, 4)

		self.model = nn.Sequential(
			layer1,
			layer2,
			nn.Flatten(),
			fc1,
			fc2,
			fc3
		)

	def forward(self, x):
		x=self.model(x)
		return x

	def update_model(self, memories, batch_size, gamma, 
					 trgt_model, device, optimizer):

		self.train()

		#sample minibatch
		minibatch=random.sample(memories, batch_size)

		#split tuples into groups and convert to tensors
		states =		torch.stack([i[0] for i in minibatch]).to(device)
		actions =		   np.array([i[1] for i in minibatch])
		rewards =		torch.stack([i[2] for i in minibatch]).to(device)
		next_states =	torch.stack([i[3] for i in minibatch]).to(device)
		not_done =		torch.stack([i[4] for i in minibatch]).to(device)

		#create predictions
		policy_scores=self(states)
		#print('scores:\n{}'.format(policy_scores))
		policy_scores=policy_scores[range(batch_size), actions]

		#create max(Q vals) from target policy net
		trgt_policy_scores=trgt_model(next_states)
		trgt_qvals=trgt_policy_scores.max(1)[0]

		"""
		#create labels
		y=policy_scores.clone().detach()
		y[range(batch_size), actions] = rewards + gamma*trgt_qvals*not_done
		"""
		y=rewards + gamma*trgt_qvals*not_done
		"""
		print('actions:\n{}'.format(actions))
		print('labels:\n{}'.format(y))
		"""

		#compute loss using Huber Loss
		loss_fn = nn.SmoothL1Loss()
		loss=loss_fn(policy_scores, y)

		#gradient descent step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		self.eval()

'''
local functions
'''

def epsilon_update(epsilon, eps_start, eps_end, eps_decay, step):
	return eps_end + (eps_start - eps_end) * math.exp(-1 * step / eps_decay)

def main(arg0, pre_trained_model=None, eps_start=.9, episodes=20000, batch_size=32):

	eps_start=float(eps_start)
	episodes=int(episodes)
	batch_size=int(batch_size)			#minibatch size for training
	
	gamma=.99				#gamma for MDP

	#epsilon greedy parameters
	epsilon=eps_start
	eps_end=.1
	eps_decay=3e5			#makes it so that decay applies over ~1 million time steps

	#update_steps=10			#update policy after every 
	C=10					#update target model after every C steps
	dtype=torch.float32		#dtype for torch tensors
	total_steps=0			#tracks global time steps

	memory_size=5000		#size of replay memory buffer
	episode_scores=[]

	
	#create gym environment
	#env = gym.make('BreakoutDeterministic-v4', obs_type='grayscale', render_mode='human')
	env = gym.make('LunarLander-v2')

	#get action space
	#action_map=env.get_keys_to_action()
	A=env.action_space.n

	#initialize main network
	policy_net=dqn(A).to(device)
	
	#load pre-trained model if specified
	if pre_trained_model is not None:
		policy_net=torch.load(model_path + pre_trained_model,
									  map_location=torch.device(device))
		print('loaded pre-trained model')
	else:
		policy_net=dqn(A)

	policy_net=policy_net.to(device)
	policy_net.eval()

	#initialize target network
	trgt_policy_net=dqn(A).to(device)
	trgt_policy_net.load_state_dict(policy_net.state_dict())
	trgt_policy_net.eval()

	#initialize optimizer
	optimizer=optim.RMSprop(policy_net.parameters())

	#initialize some variables before getting into the main loop
	replay_memories=[]
	steps_done=0
	gray=np.array([0.299, 0.587, 0.114])

	for ep in range(episodes):
		total_reward=0

		s_builder=data_point((4, 400, 600))				#initialize phi transformation function
		#s_builder.add_frame(env.reset())	
		env.reset()
		frame=env.render(mode='rgb_array')
		frame=np.dot(frame[...,:3], gray)
		s_builder.add_frame(frame)
		s=s_builder.get()
		env.render()

		t=1									#episodic t
		done=False							#tracks when episodes end
		while not done:
			
			#select action using epsilon greedy policy
			if np.random.uniform(0, 1) < epsilon:
				a=np.random.randint(0, A)
			else:
				with torch.no_grad():
					q_vals=policy_net(torch.tensor(s[np.newaxis], dtype=dtype, device=device))
					a=int(torch.argmax(q_vals[0]))
					#print(q_vals)

			#print(a)

			#update epsilon value
			epsilon = epsilon_update(epsilon, eps_start, eps_end, eps_decay, total_steps)

			#take action and collect reward and s'
			frame, r, done, info = env.step(a) 
			#s_builder.add_frame(s_prime_frame)
			frame=env.render(mode='rgb_array')
			frame=np.dot(frame[... , :3], gray)
			s_builder.add_frame(frame)
			s_prime=s_builder.get()
			env.render()

			#append to replay_memories as (s, a, r, s', done)
			replay_memories.append((torch.tensor(s, dtype=dtype),
									a,
									torch.tensor(r, dtype=dtype),
									torch.tensor(s_prime, dtype=dtype),
									torch.tensor(~done)))

			#remove oldest sample to maintain memory size
			if len(replay_memories) > memory_size:
				del replay_memories[:1]

			#perform gradient descent step
			#if len(replay_memories) > batch_size and total_steps % update_steps == 0:
			if len(replay_memories) > batch_size:
				policy_net.update_model(replay_memories, batch_size, gamma, 
										trgt_policy_net, device, optimizer)

			#set target weights to policy net weights every C steps
			if total_steps % C == 0:
				trgt_policy_net.load_state_dict(policy_net.state_dict())

			#increment counters
			total_steps+=1
			t+=1
			total_reward+=r

			#update state
			s=s_prime

			# save model cheeckpoint every 4000 time steps
			if total_steps % 4000  == 0:
				time=datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
				fname=model_path + 'dqn_checkpoint_' + time + '.pth'
				torch.save(policy_net, fname)
				print('model checkpoint saved to {}'.format(fname))

		episode_scores.append(total_reward)
		if len(episode_scores) > 20:
			del episode_scores[:1]


		print('episode: {0}, reward: {1}, epsilon: {2:.2f}, total_time: {3}, ep_length: {4}, avg: {5}'.format(ep, total_reward, epsilon, total_steps, t, np.mean(episode_scores)))
				
				
if __name__ == '__main__':
	main(*sys.argv)


