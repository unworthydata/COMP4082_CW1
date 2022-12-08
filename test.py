import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make('CartPole-v1')

# Between 0 and 1, mesue of how much we carre about future reward over immedate reward
EPISODES = 10000  # Number of iterations run
UPDATE_EVERY = 100

# Create bins and Q table
def create_bins_and_q_table(resolution):
    numBins = 10 * resolution
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


def QLearning(env, learning, discount, epsilon, min_eps, episodes, resolution):
    total_time_start = time.perf_counter()
    epsilon_decay_value = (epsilon - min_eps) / episodes
    bins, obsSpaceSize, qTable = create_bins_and_q_table(resolution)

    previousCnt = []  # array of all scores over runs
    metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics recorded for graph

    # Initialize variables to track rewards
    reward_list = []
    avg_episode_reward_list = []
    time_list = []
    avg_episode_time_list = []

    for run in range(EPISODES):
        tic = time.perf_counter()
        # if run >= EPISODES - 20:
        #     env = gym.make("CartPole-v1", render_mode="human")

        state, _ = env.reset()
        discreteState = get_discrete_state(state, bins, obsSpaceSize)
        terminated = truncated = False  # has the enviroment finished?
        cnt = 0  # how may movements cart has made
        tot_reward = 0

        while not terminated:
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
        previousCnt.append(cnt)
        toc = time.perf_counter()
        reward_list.append(tot_reward)
        time_list.append(toc - tic)

        if (run + 1) % 100 == 0:
            avg_episode_reward = np.mean(reward_list)
            avg_episode_reward_list.append(avg_episode_reward)
            reward_list = []

            # Track time taken
            avg_episode_time = np.mean(time_list)
            avg_episode_time_list.append(avg_episode_time)
            time_list = []

        # print results + time every 100 episodes
        if (run + 1) % UPDATE_EVERY == 0:
            print(
                f"Episode: {i + 1}\n\tAverage Reward: {avg_episode_reward}\n\tAverage Time Taken: {avg_episode_time:0.7f}s")

        # Add new metrics for graph
        if run % UPDATE_EVERY == 0:
            latestRuns = previousCnt[-UPDATE_EVERY:]
            averageCnt = sum(latestRuns) / len(latestRuns)
            metrics['ep'].append(run)
            metrics['avg'].append(averageCnt)
            metrics['min'].append(min(latestRuns))
            metrics['max'].append(max(latestRuns))
            print("Run:", run, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns))

    env.close()

    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start

    return avg_episode_reward_list, avg_episode_time_list, qTable, total_time


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

    rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 0.25)
    experiment_results += "\n\n" + output(rewards, times, total_time, run, "Quarter")
    outcomes["Quarter"] = [rewards, times]

    rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 0.5)
    experiment_results += "\n\n" + output(rewards, times, total_time, run, "Half")
    outcomes["Half"] = [rewards, times]

    rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 1)
    experiment_results += "\n\n" + output(rewards, times, total_time, run, "1x")
    outcomes["1x"] = [rewards, times]

    rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 2)
    experiment_results += "\n\n" + output(rewards, times, total_time, run, "2x")
    outcomes["2x"] = [rewards, times]

    rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 5)
    experiment_results += "\n\n" + output(rewards, times, total_time, run, "5x")
    outcomes["5x"] = [rewards, times]

    rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 10)
    experiment_results += "\n\n" + output(rewards, times, total_time, run, "10x")
    outcomes["10x"] = [rewards, times]

    rewards, times, Q, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, 100)
    experiment_results += "\n\n" + output(rewards, times, total_time, run, "100x")
    outcomes["100x"] = [rewards, times]

    plt.plot(100 * (np.arange(len(rewards)) + 1),
             outcomes["Quarter"][0], label="0.25x")
    plt.plot(100 * (np.arange(len(rewards)) + 1),
             outcomes["Half"][0], label="0.5x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["1x"][0], label="1x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["2x"][0], label="2x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["5x"][0], label="5x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["10x"][0], label="10x")
    plt.plot(100 * (np.arange(len(rewards)) + 1),
             outcomes["100x"][0], label="100x")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig("rewards.jpg")
    plt.close()

    plt.plot(100 * (np.arange(len(rewards)) + 1),
             outcomes["Quarter"][1], label="0.25x")
    plt.plot(100 * (np.arange(len(rewards)) + 1),
             outcomes["Half"][1], label="0.5x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["1x"][1], label="1x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["2x"][1], label="2x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["5x"][1], label="5x")
    plt.plot(100 * (np.arange(len(rewards)) + 1), outcomes["10x"][1], label="10x")
    plt.plot(100 * (np.arange(len(rewards)) + 1),
             outcomes["100x"][1], label="100x")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig("times.jpg")
    plt.close()

    with open("known/all_experiment_results.txt", 'w') as f:
        print(experiment_results, file=f)