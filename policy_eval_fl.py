import gym
import copy
import numpy as np


def main():

    env=gym.make("FrozenLake-v1")
    action_space=env.action_space.n
    state_space=env.observation_space.n
    discount=.9
    theta=.001
    v=np.zeros(state_space)
    v_prime=np.zeros(state_space)

    print("a: {}, s: {}".format(action_space, state_space))
    print(env.P[0][0])

    iteration=0
    while 1:
        #print("iteration: {}".format(iteration))
        s=env.reset()
        done=True
        delta=0
        for s in range(env.observation_space.n):
            v_prime=np.zeros(env.observation_space.n)
            state_actions=[]
            for a in range(env.action_space.n):
                psa=env.P[s][a]
                probs=np.array([i[0] for i in psa])
                s_primes=np.array([i[1] for i in psa])
                rewards=np.array([i[2] for i in psa])
                finished=np.array([i[3] for i in psa])

                #probs, s_primes, rewards, finished = psa
                v_prime[s]+=np.sum(probs*(rewards+v[s_primes]))
            #v_prime[s]=max(state_actions)
            v_prime[s]/=env.action_space.n

            delta=max(delta, abs(v[s]-v_prime[s]))
        v=copy.deepcopy(v_prime)
        if(delta < theta):
            print("converged")
            print(v)
            return
        iteration+=1



            #s_prime, r, done = env.step(a)





if __name__ == '__main__':
    main()
