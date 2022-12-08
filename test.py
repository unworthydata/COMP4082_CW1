import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

# How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded
LEARNING_RATE = 0.2
# Between 0 and 1, mesue of how much we carre about future reward over immedate reward
DISCOUNT = 0.9
EPISODES = 10000  # Number of iterations run
UPDATE_EVERY = 100  # How oftern the current progress is recorded

# Exploration settings
epsilon = 0.8  # not a constant, going to be decayed
epsilon_decay_value = epsilon / EPISODES


# Create bins and Q table
def create_bins_and_q_table():
    numBins = 100
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


bins, obsSpaceSize, qTable = create_bins_and_q_table()

previousCnt = []  # array of all scores over runs
metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics recorded for graph

for run in range(EPISODES):
    # if run >= EPISODES - 20:
    #     env = gym.make("CartPole-v1", render_mode="human")

    state, _ = env.reset()
    discreteState = get_discrete_state(state, bins, obsSpaceSize)
    terminated = truncated = False  # has the enviroment finished?
    cnt = 0  # how may movements cart has made

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
        newQ = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)
        qTable[discreteState + (action,)] = newQ  # Update qTable with new Q value

        discreteState = newDiscreteState

    previousCnt.append(cnt)

    # Decaying is being done every run if run number is within decaying range
    epsilon -= epsilon_decay_value

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

# Plot graph
plt.plot(metrics['ep'], metrics['avg'], label="average rewards")
plt.plot(metrics['ep'], metrics['min'], label="min rewards")
plt.plot(metrics['ep'], metrics['max'], label="max rewards")
plt.legend(loc=4)
plt.show()
