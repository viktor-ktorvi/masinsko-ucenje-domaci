import numpy as np
import pandas as pd

from scipy import interpolate
from tqdm import tqdm

from matplotlib import pyplot as plt

from ucenje_podsticajem.agent import AgentQ
from ucenje_podsticajem.grid_environment import GridEnvironment


def simulate(grid_environment, agent, num_episodes=10, decrease_lr=False):
    rewards = []  # record the reward and epsilon values
    epsilon = []

    v_values_dict = {}  # record the V values for the individual states
    for state in grid_environment.valid_states:
        v_values_dict[state] = []

    # loop episodes; each episode starts at the starting state and ends at a terminal state
    for episode in tqdm(range(num_episodes)):

        grid_environment.reset()  # go back to the starting state
        agent.memoryReset()  # erase the previous state and action knowledge
        observation, reward, terminate = grid_environment.step()  # initial step

        epoch = 0  # play the grid game
        while not terminate:
            v_values_dict[observation].append((episode, agent.getMaxQ(observation)))  # record V value of current state

            action = agent.act(observation)
            observation, reward, terminate = grid_environment.step(action)

            # learning rate strategy
            if decrease_lr:
                learning_rate = np.log(epoch + 1) / (epoch + 1) * agent.init_learning_rate
            else:
                learning_rate = agent.init_learning_rate
            agent.observe(observation, reward, learning_rate)

            epoch += 1

        # collect data
        rewards.append(reward)
        epsilon.append(agent.epsilon)

    return rewards, epsilon, v_values_dict


if __name__ == '__main__':
    num_episodes = 10000
    learning_rate = 0.01
    init_epsilon = 0.98
    decrease_lr = True

    # init env and agent
    grid_environment = GridEnvironment()
    agent = AgentQ(grid_environment.height, grid_environment.width, init_epsilon=init_epsilon, init_learning_rate=learning_rate)
    agent.train()

    rewards, epsilon, v_values_dict = simulate(grid_environment, agent, num_episodes=num_episodes, decrease_lr=decrease_lr)


    def running_average(x, window_size):
        return np.convolve(x, np.ones(window_size) / window_size, mode='valid')


    fig, ax = plt.subplots(2, 1, sharex='col')

    ax[0].set_title('Nagrada')
    ax[0].plot(rewards, label='nagrada')
    ax[0].plot(running_average(rewards, 20), label='srednja nagrada')
    ax[0].set_ylabel('R')
    ax[0].legend()

    ax[1].set_title('Stopa istraživanja')
    ax[1].plot(epsilon)
    ax[1].set_ylabel('$\epsilon$')

    plt.xlabel('epizoda')

    plt.figure()
    full_episode_ids = np.arange(num_episodes)

    for state in v_values_dict:
        if not v_values_dict[state]:  # if the state hasn't been visited
            continue

        # a state can be visited multiple times per episode. We average the values of those visits
        episode_ids_with_duplicates = [tuple_pair[0] for tuple_pair in v_values_dict[state]]
        associated_v_values = [tuple_pair[1] for tuple_pair in v_values_dict[state]]

        # average based on grouping by the same index
        v_values_averaged_per_episode = pd.Series(associated_v_values).groupby(episode_ids_with_duplicates).mean().to_numpy()

        unique_episode_ids = np.unique(episode_ids_with_duplicates)  # all the episodes that this state was explored in

        # a state can remain unvisited in an episode, so we interpolate in those cases
        interpolation_function = interpolate.interp1d(unique_episode_ids, v_values_averaged_per_episode, kind='previous', fill_value="extrapolate")

        plt.plot(interpolation_function(full_episode_ids), label='s = ({:d}, {:d})'.format(state.row, state.column))

    plt.xlabel('epizoda')
    plt.ylabel('$V_{epizoda}(s)$')
    plt.legend()
    plt.show()
