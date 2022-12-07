import gym 

# print(envs.registry.keys())

env = gym.make('WizardOfWor-v4', render_mode='human')
env.reset()
totalReward = 0

for i in range(10000):
    env.render()
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample())

env.close()
