# TODO comment code
import gym

env = gym.make('CartPole-v1', render_mode='human')
env.reset()
totalReward = 0

for i in range(10000):
    env.render()
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample())

    if observation[3] > 0:
        if observation[0] > 1:
            env.step(0)
        else:
            env.step(1)

    elif observation[3] > -0.5:
        if observation[0] < -1:
            env.step(1)
        else:
            env.step(0)

    totalReward += reward

    if terminated or truncated:
        print(totalReward)
        totalReward = 0
        env.reset()

env.close()
