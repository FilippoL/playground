'''An example to show how to set up an pommerman game programmatically'''
import pyglet
import numpy as np

import pommerman
from pommerman import agents


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        env.render()
        img = env._viewer.get_buffer().get_texture().get_image_data()
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            # data = img.get_data("RGB", img.width * 3)
            # arr = np.frombuffer(data, dtype=np.uint8)
            # reshaped_array = arr.reshape(img.width, img.height, 3)

        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
