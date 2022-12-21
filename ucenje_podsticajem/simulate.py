import numpy as np
import pandas as pd

from scipy import interpolate
from tqdm import tqdm

from matplotlib import pyplot as plt

from ucenje_podsticajem.agent import AgentQ
from ucenje_podsticajem.grid_environment import GridEnvironment


def simulate(grid_environment, agent, num_episodes=10, decrease_lr=False, time_penalty_weight=0.1):
    """
    Simulate the agent playing in the environment. The agent can be in both train and eval mode.
    :param grid_environment: GridEnvironment
    :param agent: AgentQ
    :param num_episodes: int
    :param decrease_lr: boolean
    :return: tuple(List[int/float], List[float], List[int],Dict); reward after each episode, epsilon after each episode,
    list of how long it took to converge, a dictionary of value functions per state, per episode
    """
    rewards = []  # record the reward and epsilon values
    epsilon = []
    epochs = []

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
            agent.observe(observation, reward - epoch * time_penalty_weight, learning_rate)

            epoch += 1

        # collect data
        rewards.append(reward)
        epsilon.append(agent.epsilon)
        epochs.append(epoch)

    return rewards, epsilon, epochs, v_values_dict


def plot_value_functions(v_values_dict, num_episodes):
    r"""
    Plot the value function $V_t(s) = \max_a \ Q_t(s, a)$ for every state provided in the v_values_dict.
    If a state has been visited multiple times per episode, then average the visits. If a state hasn't been visited in an episode,
    then interpolate.
    :param v_values_dict: dict; key = State2D; value = tuple(int, float) - episode id and V value.
    :param num_episodes: int
    :return:
    """
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


if __name__ == '__main__':
    num_episodes = 1000
    learning_rate = 0.05
    init_epsilon = 0.97
    decrease_lr = True
    time_penalty_weight = 0.0

    # init env and agent
    grid_environment = GridEnvironment()
    agent = AgentQ(grid_environment.height, grid_environment.width, init_epsilon=init_epsilon, init_learning_rate=learning_rate)
    agent.train()

    rewards, epsilon, steps, v_values_dict = simulate(grid_environment, agent, num_episodes=num_episodes, decrease_lr=decrease_lr, time_penalty_weight=time_penalty_weight)


    def running_average(x, window_size):
        return np.convolve(x, np.ones(window_size) / window_size, mode='valid')


    fig, ax = plt.subplots(3, 1, sharex='col')

    ax[0].set_title('Nagrada')
    ax[0].plot(rewards, label='nagrada', alpha=0.6)
    ax[0].plot(running_average(rewards, 20), label='srednja nagrada')
    ax[0].set_ylabel('R')
    ax[0].legend()

    ax[1].set_title('Stopa istraživanja')
    ax[1].plot(epsilon)
    ax[1].set_ylabel('$\epsilon$')

    ax[2].set_title('Broj koraka')
    ax[2].plot(steps, label='koraci')
    ax[2].plot(running_average(steps, 15), label='srednja vrednost koraka')
    ax[2].set_ylabel('#')
    ax[2].legend()


    plt.xlabel('epizoda')

    plot_value_functions(v_values_dict, num_episodes)

    # evaluate the trained model
    agent.eval()
    rewards_test, _, steps_test, _ = simulate(grid_environment, agent, num_episodes=100)

    print('Srednja vrednost nagrade pri testiranju = {:2.2f}'.format(np.mean(rewards_test)))
    print('Srednji broj koraka da se partija zavrsi = {:2.2f}'.format(np.mean(steps_test)))

    plt.figure()  # plot the reward over the evaluation episodes
    plt.plot(rewards_test)
    plt.xlabel('epizoda')
    plt.ylabel('R')

    # print an episode
    grid_environment.reset()
    agent.memoryReset()
    observation, _, terminate = grid_environment.step()  # initial step

    step = 0
    grid_environment.printMap(step=step)
    while not terminate:
        action = agent.act(observation)
        observation, _, terminate = grid_environment.step(action)
        print('True action = {:s}'.format(action.name))

        step += 1
        grid_environment.printMap(step=step)

    plt.show()
