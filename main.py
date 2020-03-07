import numpy as np
import tensorflow as tf
import os
import team_6_env
import team_6_agent
import team_6_reward

MAX_EPISODE =  9000

MEMORY_CAPACITY = 10000

EXPLORE = 3

dataset = np.load('./dataset/ards_all_data_ver1.npy')

reward_function_0 = team_6_reward.Reward_Function(0)
print('env build')
env = team_6_env.Env(
    dataset = dataset,
    reward_function = reward_function_0
    )
agent = team_6_agent.DDPG(
    env.action_space,
    env.observation_space,
    [9.],
    )

session = tf.Session()
session.run(tf.global_variables_initializer())

tf.summary.FileWriter("logs/", session.graph)

for i_episode in range(MAX_EPISODE):
    state = env.reset()
    time = 0
    episode_reward_temp = []
    action_s = []
    while True:
        reward = 0
        action = agent.choose_action(state) +  9

        # add randomness to action selection for exploration

        action = np.clip(np.random.normal(action, EXPLORE), 0, 18)
        # print("action")
        # print(action)
        action_s.append(action[0])
        state_next, reward, done = env.step(action)
        # print(state_next)
        # print("reward  ", str(reward))
        # normalize
        agent.store_transition(state, action, reward, state_next)

        if agent.pointer > MEMORY_CAPACITY:
            EXPLORE *= .99995    # decay the action randomness
            agent.learn()
        
        state = state_next
        episode_reward_temp.append(reward)
        if done:
            print('Reward: %i' % sum(episode_reward_temp), '\tExplore: %.2f' % EXPLORE, '\tAVG Action: %i' % (sum(action_s)/ len(action_s)), '\tMax Action: %i' % (max(action_s)), '\tmin Action: %i' % (min(action_s)))
            episode_reward_temp = []
            # print(sum(action_s)/ len(action_s))
            break
agent.save()