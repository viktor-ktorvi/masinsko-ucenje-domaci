import numpy as np

from matplotlib import pyplot as plt
from tqdm import tqdm

from ucenje_podsticajem.agent import AgentQ
from ucenje_podsticajem.grid_environment import GridEnvironment

if __name__ == '__main__':
    # init env and agent
    grid_environment = GridEnvironment()
    agent = AgentQ(grid_environment.height, grid_environment.width, epsilon=0.95)
    agent.train()

    num_episodes = 1000

    # loop episodes; each episode starts at the starting state and ends at a terminal state
    rewards = []
    epsilon = []
    for episode in tqdm(range(num_episodes)):

        grid_environment.reset()  # go back to the starting state
        agent.memoryReset()  # erase the previous state and action knowledge
        observation, reward, terminate = grid_environment.step()  # initial step

        epoch = 0  # play the grid game
        while not terminate:
            action = agent.act(observation)
            observation, reward, terminate = grid_environment.step(action)
            agent.observe(observation, reward, epoch)

            epoch += 1

        # collect data
        rewards.append(reward)
        epsilon.append(agent.epsilon)


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

    plt.xlabel('epozoda')
    plt.show()
