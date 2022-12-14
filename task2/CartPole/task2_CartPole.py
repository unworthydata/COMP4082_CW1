import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make('CartPole-v1')

EPISODES = 10000  # Number of iterations run
UPDATE_EVERY = 100


# Create bins and Q table
def create_bins_and_q_table(granularity):
    numBins = int(10 * granularity)
    obsSpaceSize = len(env.observation_space.high)

    # Get the size of each bucket
    bins = [
        np.linspace(-4.8, 4.8, numBins),
        np.linspace(-4, 4, numBins),
        np.linspace(-.418, .418, numBins),
        np.linspace(-4, 4, numBins)
    ]

    qTable = np.random.uniform(low=0, high=1, size=([numBins] * obsSpaceSize + [env.action_space.n]))

    return bins, obsSpaceSize, qTable


# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, bins, obsSpaceSize):
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(np.digitize(state[i], bins[i]) - 1)  # -1 will turn bin into index
    return tuple(stateIndex)


def QLearning(env, learning, discount, epsilon, min_eps, episodes, granularity):
    total_time_start = time.perf_counter()
    epsilon_decay_value = (epsilon - min_eps) / episodes
    bins, obsSpaceSize, qTable = create_bins_and_q_table(granularity)

    # Initialize variables to track rewards and times
    reward_list = []
    avg_episode_reward_list = []
    time_list = []
    avg_episode_time_list = []

    for i in range(EPISODES):
        tic = time.perf_counter()
        # if you want to view the last 20 episodes of the simulation
        # if run >= EPISODES - 20:
        #     env = gym.make("CartPole-v1", render_mode="human")

        state, _ = env.reset()
        discreteState = get_discrete_state(state, bins, obsSpaceSize)
        terminated = truncated = False  # has the enviroment finished?
        cnt = 0  # how may movements cart has made
        tot_reward = 0

        while not terminated and not truncated:
            cnt += 1
            # Get action from Q table
            if np.random.random() < 1 - epsilon:
                action = np.argmax(qTable[discreteState])
            else:
                action = np.random.randint(0, env.action_space.n)

            newState, reward, terminated, truncated, _ = env.step(action)  # perform action on enviroment
            reward -= (abs(newState[0]) + abs(newState[2])) / 2.5

            newDiscreteState = get_discrete_state(newState, bins, obsSpaceSize)

            maxFutureQ = np.max(qTable[newDiscreteState])  # estimate of optiomal future value
            currentQ = qTable[discreteState + (action,)]  # old value

            # pole fell over / went out of bounds, negative reward
            if terminated and cnt < 500:
                reward = -375
            else:
                reward *= 10

            # formula to caculate all Q values
            newQ = (1 - learning) * currentQ + learning * (reward + discount * maxFutureQ)
            qTable[discreteState + (action,)] = newQ  # Update qTable with new Q value

            discreteState = newDiscreteState
            tot_reward += reward

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= epsilon_decay_value

        # Track rewards and time
        toc = time.perf_counter()
        reward_list.append(tot_reward)
        time_list.append(toc - tic)

        if (i + 1) % UPDATE_EVERY == 0:
            avg_episode_reward = np.mean(reward_list)
            avg_episode_reward_list.append(avg_episode_reward)
            reward_list = []

            # Track time taken
            avg_episode_time = np.mean(time_list)
            avg_episode_time_list.append(avg_episode_time)
            time_list = []
            
            # print results + time every 100 episodes
            print(f"""Episode: {i + 1}
    Average Reward: {avg_episode_reward}
    Average Time Taken: {avg_episode_time:0.7f}s""")

    env.close()

    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start

    return avg_episode_reward_list, avg_episode_time_list, total_time


def output(rewards, times, total_time, granularity='1x', run="1"):
    results = f"""Granularity: {granularity}
    Total time: {total_time:0.7f}s
    Best average reward: {rewards[-1]}
    Best time: {times[-1]}s
    Stability: {stability(rewards)}%"""

    print(results)

    here = os.path.dirname(os.path.realpath(__file__))
    run_directory = os.path.join(here, f"run_{run}")
    subdirectory = os.path.join(run_directory, granularity)
    filepath = os.path.join(subdirectory, f"{granularity}_experiment_results.txt")

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
    plt.savefig(os.path.join(subdirectory, f"{granularity}_rewards.jpg"))
    plt.close()

    # Plot Time
    plt.plot(100 * (np.arange(len(rewards)) + 1), times, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig(os.path.join(subdirectory, f"{granularity}_times.jpg"))
    plt.close()

    with open(os.path.join(here, f"all_experiment_results.txt"), 'a') as f:
        print(f"\n\nRun: {run}:\n\n", file=f)
        print(f"{results}", file=f)

    return results


def stability(rewards):
    # using relative error to measure the avg rate of change of rewards per 100 episodes,
    # to understand the stability of the algorithm
    # 1 - |(x2 - x1) - (x3 - x2)| / (x2 - x1))
    stability = 0
    for i in range(1, len(rewards) - 1):
        # divide by 100 because average rewards are taken every 100 episodes
        delta1 = abs(rewards[i] - rewards[i - 1]) / 100
        delta2 = abs(rewards[i + 1] - rewards[i]) / 100
        stability += abs(delta2 - delta1) / abs(delta2)

    # len(rewards) + 2 because we skipped the first and last rewards in the loop
    avg_stability = stability / (len(rewards) + 2)
    # 1 - avg_stability because we want stability/similarity in rate of change, not difference
    return avg_stability * 100


def run_experiment(granularity_label, granularity_multiplier, run="1"):
    rewards, times, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, granularity_multiplier)
    output(rewards, times, total_time, granularity_label, run)
    return rewards, times


# set up all granularities here so we can loop over them, instead of repeating code
granularities = {"Quarter": 0.25, "Half": 0.5, "1x": 1, "2x": 2, "5x": 5, "10x": 10}
here = os.path.dirname(os.path.realpath(__file__))

for run in range(1, 4):
    # Run Q-learning algorithm at different granularities
    outcomes = {}

    for granularity in granularities:
        outcomes[granularity] = list(run_experiment(granularity, granularities[granularity], run))

    for granularity in granularities:
        plt.plot(100 * (np.arange(EPISODES / UPDATE_EVERY) + 1),
             outcomes[granularity][0], label=f"{granularity}")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig(os.path.join(here, f"rewards.jpg"))
    plt.close()

    for granularity in granularities:
        plt.plot(100 * (np.arange(EPISODES / UPDATE_EVERY) + 1),
             outcomes[granularity][1], label=f"{granularity}")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig(os.path.join(here, f"times.jpg"))
    plt.close()
