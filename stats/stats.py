import json
import os

with open('stats/20200123-181808_RandomAgent.json', 'r') as f:
    stats_dict = json.load(f)

rewards_list = stats_dict['rewards_per_episode']
num_wins = len([1 for reward in rewards_list if reward >= 1])

print(num_wins)
