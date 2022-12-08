import os
import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Import and initialize Mountain Car Environment
env = gym.make('CartPole-v1')
state, _ = env.reset()
EPISODES = 500_000


def QLearning(env, learning, discount, epsilon, min_eps, episodes, resolution):
    # start counter
    total_time_start = time.perf_counter()

    # Determine size of discretized state space
    numBins = 20
    obsSpaceSize = len(env.observation_space.high)

    # Get the size of each bucket
    bins = [
        np.linspace(-4.8, 4.8, numBins),
        np.linspace(-4, 4, numBins),
        np.linspace(-.418, .418, numBins),
        np.linspace(-4, 4, numBins)
    ]

    Q = np.random.uniform(low=0, high=1, size=([numBins] * obsSpaceSize + [env.action_space.n]))

    # Initialize variables to track rewards
    reward_list = []
    avg_episode_reward_list = []
    time_list = []
    avg_episode_time_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps) / episodes

    # Run Q learning algorithm
    for i in range(episodes):
        if i >= episodes - 2000:
            env = gym.make("CartPole-v1", render_mode="human")

        tic = time.perf_counter()
        # Initialize parameters
        terminated = truncated = False
        tot_reward, reward = 0, 0
        state, _ = env.reset()

        # Discretize state
        state_adj = normalize_state(state)

        while not terminated:
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, terminated, truncated, _ = env.step(action)
            # reward = reward_updater(state, reward)

            # Discretize state2
            state2_adj = normalize_state(state2)

            # Allow for terminal states
            if terminated:
                Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action] = reward
            # Adjust Q value for current state
            else:
                delta = learning * (reward +
                                    discount * np.max(Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3]]) -
                                    Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action])
                Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action] += delta

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

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

            # Track time taken
            avg_episode_time = np.mean(time_list)
            avg_episode_time_list.append(avg_episode_time)
            time_list = []

        # print results + time every 100 episodes
        if (i + 1) % 100 == 0:
            print(
                f"Episode: {i + 1}\n\tAverage Reward: {avg_episode_reward}\n\tAverage Time Taken: {avg_episode_time:0.7f}s")

    env.close()

    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start

    return avg_episode_reward_list, avg_episode_time_list, Q, total_time


def output(rewards, times, total_time, run, resolution='1x'):
    results = f"""Resolution: {resolution}
    Total time: {total_time:0.7f}s
    Best average reward: {rewards[-1]}
    Best time: {times[-1]}s
    Stability: {stability(rewards)}%"""

    print(results)

    here = os.path.dirname(os.path.realpath(__file__))
    run_directory = os.path.join(here, f"run_{run}")
    subdirectory = os.path.join(run_directory, resolution)
    filepath = os.path.join(subdirectory, f"{resolution}_experiment_results.txt")

    if not os.path.isdir(run_directory):
        os.mkdir(run_directory)
    if not os.path.isdir(subdirectory):
        os.mkdir(subdirectory)

    with open(filepath, 'w') as f:
        print(results, file=f)
        print("\n" + f"Rewards: {rewards}", file=f)
        print("\n" + f"Times: {times}", file=f)

    # Plot Rewards
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig(os.path.join(subdirectory, f"{resolution}_rewards.jpg"))
    plt.close()

    # Plot Time
    plt.plot(100 * (np.arange(len(rewards)) + 1), times, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig(os.path.join(subdirectory, f"{resolution}_times.jpg"))
    plt.close()

    return results


def normalize_state(state):
    np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
    discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(int))


def reward_updater(state, env_reward):
    # the closer the cart is to the center, and the straighter the pole is, the better
    return env_reward - (abs(state[0]) + abs(state[2])) / 2.5


def stability(rewards):
    # using relative error to measure the avg rate of change of rewards per 100 episodes,
    # to understand the stability of the algorithm
    # 1 - |(x2 - x1) - (x3 - x2)| / (x2 - x1))
    stability = 0
    for i in range(1, len(rewards) - 1):
        delta1 = abs(rewards[i] - rewards[i - 1]) / 100
        delta2 = abs(rewards[i + 1] - rewards[i]) / 100
        stability += abs(delta2 - delta1) / abs(delta2)

    # len(rewards) + 2 because we skipped the first and last rewards in the loop
    avg_stability = stability / (len(rewards) + 2)
    # 1 - avg_stability because we want stability/similarity in rate of change, not difference
    return (1 - avg_stability) * 100


for run in range(1, 4):
    # Run Q-learning algorithm at different granularities
    experiment_results = ""
    outcomes = {}

    # rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 0.25)
    # experiment_results += "\n\n" + output(rewards, times, total_time, run, "Quarter")
    # outcomes["Quarter"] = [rewards, times]
    #
    # rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 0.5)
    # experiment_results += "\n\n" + output(rewards, times, total_time, run, "Half")
    # outcomes["Half"] = [rewards, times]

    rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 1)
    experiment_results += "\n\n" + output(rewards, times, total_time, run, "1x")
    outcomes["1x"] = [rewards, times]

    # rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 2)
    # experiment_results += "\n\n" + output(rewards, times, total_time, run, "2x")
    # outcomes["2x"] = [rewards, times]
    #
    # rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 5)
    # experiment_results += "\n\n" + output(rewards, times, total_time, run, "5x")
    # outcomes["5x"] = [rewards, times]
    #
    # rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 10)
    # experiment_results += "\n\n" + output(rewards, times, total_time, run, "10x")
    # outcomes["10x"] = [rewards, times]
    #
    # rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 100)
    # experiment_results += "\n\n" + output(rewards, times, total_time, run, "100x")
    # outcomes["100x"] = [rewards, times]

    # plt.plot(100 * (np.arange(len(rewards)) + 1),
    #          outcomes["Quarter"][0], label="0.25x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1),
    #          outcomes["Half"][0], label="0.5x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["1x"][0], label="1x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["2x"][0], label="2x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["5x"][0], label="5x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["10x"][0], label="10x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1),
    #          outcomes["100x"][0], label="100x")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig("rewards.jpg")
    plt.close()

    # plt.plot(100 * (np.arange(len(rewards)) + 1),
    #          outcomes["Quarter"][1], label="0.25x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1),
    #          outcomes["Half"][1], label="0.5x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["1x"][1], label="1x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["2x"][1], label="2x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["5x"][1], label="5x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["10x"][1], label="10x")
    # plt.plot(100 * (np.arange(len(rewards)) + 1),
    #          outcomes["100x"][1], label="100x")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig("times.jpg")
    plt.close()

    with open("all_experiment_results.txt", 'w') as f:
        print(experiment_results, file=f)
