import math
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Ads_CTR_Optimisation.csv')

N = 10000
d = 10
print(N, d)
ads_selected=[]
numbers_of_selection=[0]*d
sum_of_rewards=[0]*d
total_reward=0

for i in range(0,N):
    ads=0
    max_upper_bound=0
    for j in range(0,d):
        if (numbers_of_selection[j]>0):
            average_reward=sum_of_rewards[j]/numbers_of_selection[j]
            delta_i=math.sqrt(3/2 *math.log(i+1)/numbers_of_selection[j])
            upper_bound=average_reward+delta_i
        else:
            upper_bound=1e400
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ads=j
    ads_selected.append(ads)
    numbers_of_selection[ads] = numbers_of_selection[ads] + 1
    reward = dataset.values[i, ads]
    sum_of_rewards[ads] = sum_of_rewards[ads] + reward
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.savefig('ucb_histogram.png')
print("Histogram saved as ucb_histogram.png")