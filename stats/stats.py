import json
import os
import statistics
from scipy.stats import ks_2samp
from statsmodels.stats import weightstats as stests

# Opening the JSON files and getting the data
with open('stats/20200123-181808_RandomAgent.json', 'r') as f:
    stats_dict_random = json.load(f)

with open('stats/20200123-191335_SimpleAgent.json', 'r') as f:
    stats_dict_simple = json.load(f)

with open('stats/20200124-161546_OurAgent.json', 'r') as f:
    stats_dict_our = json.load(f)

frames_list_random = stats_dict_random['frames_per_episode']
frames_list_simple = stats_dict_simple['frames_per_episode']
frames_list_our = stats_dict_our['frames_per_episode']

rewards_list_random = stats_dict_random['rewards_per_episode']
rewards_list_simple = stats_dict_simple['rewards_per_episode']
rewards_list_our = stats_dict_our['rewards_per_episode']

# Calculating the Z test statistic

rewards_list_random_win = [1 if reward >=
                           1 else 0 for reward in rewards_list_random]
rewards_list_simple_win = [1 if reward >=
                           1 else 0 for reward in rewards_list_simple]
rewards_list_our_win = [1 if reward >=
                        1 else 0 for reward in rewards_list_our]

z_test_random_simple = stests.ztest(
    x1=rewards_list_random_win, x2=rewards_list_simple_win, alternative='two-sided')
z_test_random_our = stests.ztest(
    x1=rewards_list_random_win, x2=rewards_list_our_win, alternative='two-sided')

print("Z-test for RandomAgent/SimpleAgent (statistic, p-value):")
print(z_test_random_simple)
print("Z-test for RandomAgent/Our Agent (statistic, p-value):")
print(z_test_random_our)

# Calculating the KS statistic
ks_test_random_simple = ks_2samp(frames_list_random, frames_list_simple)
ks_test_random_our = ks_2samp(frames_list_random, frames_list_our)

print("\n")
print(ks_test_random_simple)
print(ks_test_random_our)
print("\n")

# Calculating number of wins
num_wins_random = len([1 for reward in rewards_list_random if reward >= 1])
num_wins_simple = len([1 for reward in rewards_list_simple if reward >= 1])
num_wins_our = len([1 for reward in rewards_list_our if reward >= 1])

print(
    f"Random Agent: {num_wins_random} wins in {len(rewards_list_random)} episodes. {num_wins_random/len(rewards_list_random)*100}%")
print(
    f"Simple Agent: {num_wins_simple} wins in {len(rewards_list_simple)} episodes. {num_wins_simple/len(rewards_list_simple)*100}%")
print(
    f"Our Agent: {num_wins_our} wins in {len(rewards_list_our)} episodes. {num_wins_our/len(rewards_list_our)*100}%")
print("\n")

# Printing the descriptive stats
print("-- DESCRIPTIVE STATISTICS ON FRAMES --")
print()
print(">>> RandomAgent")
print(f"Mean: {statistics.mean(frames_list_random)}, SD: {statistics.stdev(frames_list_random)}, Median: {statistics.median(frames_list_random)}, Mode:{statistics.mode(frames_list_random)}")
print()
print(">>> SimpleAgent")
print(f"Mean: {statistics.mean(frames_list_simple)}, SD: {statistics.stdev(frames_list_simple)}, Median: {statistics.median(frames_list_simple)}, Mode:{statistics.mode(frames_list_simple)}")
print()
print(">>> Our Agent")
print(f"Mean: {statistics.mean(frames_list_our)}, SD: {statistics.stdev(frames_list_our)}, Median: {statistics.median(frames_list_our)}, Mode:{statistics.mode(frames_list_our)}")
