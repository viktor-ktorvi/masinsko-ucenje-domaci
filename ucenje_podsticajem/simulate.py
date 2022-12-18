from ucenje_podsticajem.agent import AgentQ
from ucenje_podsticajem.grid_environment import GridEnvironment

if __name__ == '__main__':
    grid_environment = GridEnvironment()
    agent = AgentQ(grid_environment.height, grid_environment.width)
    agent.train()

    for episode in range(70):

        grid_environment.reset()
        agent.reset()
        observation, reward, terminate = grid_environment.step()
        epoch = 0

        while not terminate:
            action = agent.act(observation)
            observation, reward, terminate = grid_environment.step(action)
            agent.observe(observation, reward, epoch)

            # print(epoch, end=' ')
            epoch += 1

