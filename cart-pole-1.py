# # TODO comment code
import gym

env = gym.make('CartPole-v1')
env.reset()
totalReward = 0

print(env.observation_space.high)
print(env.observation_space.low)

# for i in range(10000):
#     env.render()
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     print(observation)
#
#     totalReward += reward
#
#     if terminated or truncated:
#         print(totalReward)
#         totalReward = 0
#         env.reset()

env.close()

# from gym import envs
# print(envs.registry.keys())