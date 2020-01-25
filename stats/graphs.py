import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json


# Opening the JSON files and getting the data
with open('stats/20200123-181808_RandomAgent.json', 'r') as f:
    stats_dict_random = json.load(f)

with open('stats/20200123-191335_SimpleAgent.json', 'r') as f:
    stats_dict_simple = json.load(f)

with open('stats/combined_OurAgent.json', 'r') as f:
    stats_dict_our = json.load(f)

frames_list_random = stats_dict_random['frames_per_episode']
frames_list_simple = stats_dict_simple['frames_per_episode']
frames_list_our = stats_dict_our['frames_per_episode']

# Plotting
hist_rand = plt.hist(frames_list_random, histtype='stepfilled', edgecolor='black', bins=range(0, 800, 10),
                     facecolor='blue', alpha=0.3)
hist_simp = plt.hist(frames_list_our, histtype='stepfilled', edgecolor='black', bins=range(0, 800, 10),
                     facecolor='green', alpha=0.3)

# plt.xlim((0, 800))
plt.title('Frame count distribution')
plt.xlabel('Frame count')
plt.ylabel('Frequency')

plt.legend(("Random Agent", "Our Agent"))

plt.show()
