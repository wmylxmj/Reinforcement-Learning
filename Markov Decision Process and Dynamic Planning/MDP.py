# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:11:19 2019

@author: wmy
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make('FrozenLake-v0')

def value_iteration(env, gamma=0.9, threshold=1e-10, iterations=100000):
    nS = env.observation_space.n
    nA = env.action_space.n
    V = np.zeros(nS)
    for i in range(iterations):
        last_V = np.copy(V)
        for state in range(nS):
            # Q中记录在当前状态中，分别采取各个行为所获得的奖励值
            Q = []
            for action in range(nA):
                # 计算奖励
                next_states_rewards = []
                # 每个p为1条分支
                for p in env.env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = p
                    # 当前奖励 + gamma*未来奖励
                    next_state_reward = (trans_prob*(reward_prob + gamma*V[next_state]))
                    next_states_rewards.append(next_state_reward)
                    pass
                # len(Q) = nA
                # 计算总奖励
                Q.append(np.sum(next_states_rewards))
                pass
            # update V
            # 当前状态价值等于采取最好行为所获得的奖励值
            V[state] = max(Q)
            pass
        # 波动小时退出迭代
        if (np.sum(np.fabs(V - last_V)) <= threshold):
            print('Value-iteration converged at iteration ' + str(i+1) + '.')
            break
        pass
    return V
                
def extract_policy(env, V, gamma=0.9):
    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.zeros(nS)
    for state in range(nS):
        Q = np.zeros(nA)
        for action in range(nA):
            for p in env.env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = p
                # 算各个状态下各个动作的Q值
                Q[action] += (trans_prob*(reward_prob + gamma*V[next_state]))
                pass
            pass
        # 选取Q值最大的动作
        policy[state] = np.argmax(Q)
        pass
    return policy

optimal_value_function = value_iteration(env)
optimal_policy = extract_policy(env, optimal_value_function)
print(optimal_policy)

state = env.reset()
for i in range(100):
    time.sleep(0.1)
    state, reward, done, infos = env.step(int(optimal_policy[state]))
    env.render()
    if done:
        print('Agent done in step ' + str(i+1) +'.')
        break
    pass
    
def compute_value_function(env, policy, gamma=0.9, threshold=1e-10):
    nS = env.observation_space.n
    V = np.zeros(nS)
    while True:
        last_V = np.copy(V)
        for state in range(nS):
            # 在该决策下的值函数
            action = policy[state]
            next_states_rewards = []
            for p in env.env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = p
                next_state_reward = (trans_prob*(reward_prob + gamma*V[next_state]))
                next_states_rewards.append(next_state_reward)
                pass
            V[state] = sum(next_states_rewards)
            pass
        if (np.sum(np.fabs(V - last_V)) <= threshold):
            break
        pass
    return V

def policy_iteration(env, gamma=0.9, iterations=200000):
    nS = env.observation_space.n
    random_policy = np.zeros(nS)
    for i in range(iterations):
        new_value_function = compute_value_function(env, random_policy, gamma=gamma)
        new_policy = extract_policy(env, new_value_function, gamma=gamma)
        if (np.all(random_policy==new_policy)):
            print('Policy-iteration converged at iteration ' + str(i+1) + '.')
            break
        random_policy = new_policy
        pass
    return new_policy
            
optimal_policy = policy_iteration(env)
print(optimal_policy)

state = env.reset()
for i in range(100):
    time.sleep(0.1)
    state, reward, done, infos = env.step(int(optimal_policy[state]))
    env.render()
    if done:
        print('Agent done in step ' + str(i+1) +'.')
        break
    pass
    
    
