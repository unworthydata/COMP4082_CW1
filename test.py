import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
total_time_start = time.perf_counter()


def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high -
                  env.observation_space.low) * np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    avg_episode_reward_list = []
    time_list = []
    avg_episode_time_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps) / episodes

    # Run Q learning algorithm
    for i in range(episodes):
        # if i >= episodes - 20:
        #     env = gym.make('MountainCar-v0', render_mode='human')

        tic = time.perf_counter()
        # Initialize parameters
        tot_reward, reward = 0, 0
        state, _ = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        while True:
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, terminated, truncated, _ = env.step(action)

            # reward += abs(10*state2[0]) + abs(100*state2[1])
            reward += reward + energy_stored(state2)
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low) * \
                         np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states
            if terminated:
                Q[state_adj[0], state_adj[1], action] = reward
                break
            # Adjust Q value for current state
            else:
                delta = learning * (reward +
                                    discount * np.max(Q[state2_adj[0],
                                                        state2_adj[1]]) -
                                    Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

            if truncated:
                break

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        # Track rewards and time
        toc = time.perf_counter()
        reward_list.append(tot_reward)
        time_list.append(toc - tic)

        if (i + 1) % 100 == 0:
            avg_episode_reward = np.mean(reward_list)
            avg_episode_reward_list.append(avg_episode_reward)
            reward_list = []

            avg_episode_time = np.mean(time_list)
            avg_episode_time_list.append(avg_episode_time)
            time_list = []

        # Track time taken

        # print results + time every 100 episodes
        if (i + 1) % 100 == 0:
            print(
                f"Episode: {i + 1}\n\tAverage Reward: {avg_episode_reward}\n\tAverage Time Taken: {avg_episode_time:0.7f}s")

    env.close()

    return avg_episode_reward_list, avg_episode_time_list, Q


def energy_stored(observation):
    # Given an action, the mountain car follows the following transition dynamics:
    # velocity(t+1) = velocity(t) + (action - 1) * force - cos(3 * position(t)) * gravity
    # position(t+1) = position(t) + velocity(t+1)
    #
    # where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity
    # set to 0 upon collision with the wall.
    #
    # reward(t+1) = reward(t) + kinetic energy + potential energy

    mass = 1
    gravity = 0.0025
    position = 10 * observation[0]
    velocity = 100 * observation[1]

    # kinetic energy = 0.5 * m * v^2
    # we consider mass = constant of 1
    kinetic = 0.5 * mass * (velocity ** 2)

    # potential energy
    potential = mass * gravity * abs(position)

    return kinetic + potential


EPISODES = 10000
# Run Q-learning algorithm
rewards, times, Q = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES)

total_time_end = time.perf_counter()
total_time = total_time_end - total_time_start
print(
    f"\nTotal time for experiment with {EPISODES} episodes: {total_time:0.7f}\n")

plt.subplot(1, 2, 1)
# Plot Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
# plt.title('Average Reward vs Episodes: before optimization')

plt.subplot(1, 2, 2)
# Plot Time
plt.plot(100 * (np.arange(len(rewards)) + 1), times, color='red')
plt.xlabel('Episodes')
plt.ylabel('Average Time (in seconds)')
# plt.title('Average Time vs Episodes: before optimization')

plt.savefig('results_before.jpg')
plt.close()
