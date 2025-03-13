"""Test environments of gym"""
import gym
import torch
import numpy as np

# WARN: The environment CartPole-v0 is out of date. You should consider upgrading to version `v1`.
# env = gym.make('CartPole-v0')
# set render_mode="human" to pop up game window
env = gym.make('CartPole-v1', render_mode="human")
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space)
# print("env.observation_space.high: ", env.observation_space.high)
# print("env.observation_space.low: ", env.observation_space.low)

for i_episode in range(20):
    observation = env.reset(seed=36)
    for t in range(100):
        print("------------------")
        # WARN: You are calling render method without specifying any render mode.
        # You can specify the render_mode at initialization, e.g. gym("CartPole-v1", render_mode="rgb_array")
        # No need to invoke render explicitly
        # env.render()
        # print(observation)
        # state = torch.from_numpy(observation).float().unsqueeze(0)
        # print(state.size())
        action = env.action_space.sample()
        # print(np.shape(action))
        # observation, reward, done, info = env.step(action)
        observation, reward, terminated, truncated, info = env.step(action)
        # [position of cart, velocity of cart, angle of pole, rotation rate of pole].
        # Defined at https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L75
        print("action: ", action, ", observation:", observation, ", reward: ", reward)
        # if done:
        if terminated:
            print("Episode finished after {} time steps".format(t + 1))
            break

env.close()
