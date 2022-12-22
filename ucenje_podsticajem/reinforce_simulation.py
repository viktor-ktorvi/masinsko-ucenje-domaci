from tqdm import tqdm

from matplotlib import pyplot as plt

from ucenje_podsticajem.agent import AgentREINFORCE
from ucenje_podsticajem.grid_environment import GridEnvironment
from utils.utils import running_average

if __name__ == '__main__':
    num_episodes = 1000
    learning_rate = 0.5
    gamma = 0.9

    # init env and agent
    grid_environment = GridEnvironment()
    agent = AgentREINFORCE(num_episodes=10000, grid_environment=grid_environment, hidden_layer_sizes=(20,))

    rewards = []

    for episode in tqdm(range(num_episodes)):
        grid_environment.reset()  # go back to the starting state
        observation, reward, terminate = grid_environment.step()  # initial step

        episode_observations = []
        episode_actions = []
        episode_rewards = []

        while not terminate:
            episode_observations.append(observation)

            action = agent.act(observation)

            observation, reward, terminate = grid_environment.step(action)

            episode_actions.append(action)
            episode_rewards.append(reward)

        agent.observe(episode_observations, episode_actions, episode_rewards, learning_rate, gamma)

        rewards.append(reward)

    plt.figure()
    plt.plot(rewards, label='nagrada')
    plt.plot(running_average(rewards, 40), label='srednja nagrada')
    plt.xlabel('epizoda')
    plt.ylabel('R')
    plt.legend()

    plt.show()
