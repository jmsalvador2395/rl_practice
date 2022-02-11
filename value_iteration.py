import gym
import copy
import numpy as np

def random_policy_eval(env, gamma=.9, theta=.0001, v=None):

	action_space=env.action_space.n
	state_space=env.observation_space.n
	if v==None:
		v=np.zeros(state_space)
	v_prime=np.zeros(state_space)

	iteration=0
	while 1:
		s=env.reset()
		done=True
		delta=0
		for s in range(env.observation_space.n):
			state_actions=0
			for a in range(env.action_space.n):

				psa=env.P[s][a]
				probs=np.array([i[0] for i in psa])
				s_primes=np.array([i[1] for i in psa])
				rewards=np.array([i[2] for i in psa])
				finished=np.array([i[3] for i in psa])

				state_actions+=np.sum(probs*(rewards+v[s_primes]))

			v_prime[s]=state_actions/env.action_space.n
			delta=max(delta, abs(v[s]-v_prime[s]))

		v=copy.deepcopy(v_prime)
		if(delta < theta):
			print("converged")
			return v
		iteration+=1

def policy_eval(env, gamma=.9, theta=.0001, v=None):

	action_space=env.action_space.n
	state_space=env.observation_space.n
	if v is None:
		v=np.zeros(state_space)
	v_prime=np.zeros(state_space)

	print(env.P[0][0])

	iteration=0
	while 1:
		s=env.reset()
		done=True
		delta=0
		for s in range(env.observation_space.n):
			a=pi(env, s, v, gamma)

			psa=env.P[s][a]
			probs=np.array([i[0] for i in psa])
			s_primes=np.array([i[1] for i in psa])
			rewards=np.array([i[2] for i in psa])
			finished=np.array([i[3] for i in psa])

			v_prime[s]=np.sum(probs*(rewards+gamma*v[s_primes]))

			delta=max(delta, abs(v[s]-v_prime[s]))
		v=copy.deepcopy(v_prime)
		if(delta < theta):
			print("converged")
			print(v)
			return v
		iteration+=1



def pi(env, s, v, gamma=.9):
	state_values=np.zeros(env.action_space.n)
	for a in range(env.action_space.n):
		psa=env.P[s][a]

		probs=np.array([i[0] for i in psa])
		s_primes=np.array([i[1] for i in psa])
		rewards=np.array([i[2] for i in psa])
		finished=np.array([i[3] for i in psa])
		
		state_values[a]=np.sum(probs*(rewards+gamma*v[s_primes]))
	
	return np.argmax(state_values)

def policy_iteration(env, gamma=.9, theta=.0001):
	v=np.zeros(env.observation_space.n)

	while True:
		policy_stable=True
		old_v=copy.deepcopy(v)
		v=policy_eval(env, gamma, theta, v)
		for s in range(env.observation_space.n):
			old_a=pi(env, s, old_v, gamma)
			a=pi(env, s, v, gamma)
			if a != old_a:
				policy_stable=False
		if policy_stable:
			print('optimal policy found')
			return v
	
def value_iteration(env, gamma=.9, theta=.0001):

	action_space=env.action_space.n
	state_space=env.observation_space.n

	v=np.zeros(state_space)
	v_prime=np.zeros(state_space)

	iteration=0
	while 1:
		s=env.reset()
		done=True
		delta=0
		for s in range(env.observation_space.n):
			action_values=np.zeros(env.action_space.n)
			for a in range(env.action_space.n):

				psa=env.P[s][a]
				probs=np.array([i[0] for i in psa])
				s_primes=np.array([i[1] for i in psa])
				rewards=np.array([i[2] for i in psa])
				finished=np.array([i[3] for i in psa])
				
				action_values[a]=np.sum(probs*(rewards+gamma*v[s_primes]))
			v_prime[s]=np.max(action_values)

			delta=max(delta, abs(v[s]-v_prime[s]))
		v=copy.deepcopy(v_prime)
		if(delta < theta):
			print("converged")
			print(v)
			return v
		iteration+=1


if __name__ == '__main__':
	gamma=.9
	episodes=100
	wins=0
	env=gym.make("FrozenLake-v1")
	#v=policy_iteration(env)
	v=value_iteration(env,  gamma)
	"""
	v=policy_evaluation(env)
	for i in range(4):
		for j in range(4):
			print(v[i*4+j], end=',')
		print()
	a=pi(env, 8, v)
	"""
	print(v)

	for i in range(episodes):
		state=env.reset()
		time_step=0
		while True:
			time_step+=1
			a=pi(env, state, v, gamma)
			state, reward, done, _ = env.step(a)

			if done:
				print('=======================')
				print('episode {}'.format(i+1))
				print('Finished in {} steps'.format(time_step))
				print('ended in state {}'.format(state))
				print('=======================\n')
				if state == 15:
					wins+=1
				break
	print('{}% win rate'.format(100*wins/episodes))

