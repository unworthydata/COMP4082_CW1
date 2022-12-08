# # TODO comment code
import gym

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
env.reset()
totalReward = 0

for i in range(10000):
    env.render()
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    totalReward += reward

    if terminated or truncated:
        print(totalReward)
        totalReward = 0
        env.reset()

env.close()

# from gym import envs
# print(envs.registry.keys())