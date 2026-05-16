import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset=pd.read_csv('../Ads_CTR_Optimisation.csv')
print(dataset.head())

N=10000
d=10
ads_selected=[]
numbers_of_rewards_1=[0]*d
numbers_of_rewards_0=[0]*d
total_reward=0

for n in range(0, N):
    ad=0
    max_random_beta=0
    for i in range(0,d):
        random_beta=random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1)
        if random_beta>max_random_beta:
            max_random_beta=random_beta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        numbers_of_rewards_1[ad]+=1
    else:
        numbers_of_rewards_0[ad]+=1
    total_reward+=reward
print("Total Reward: ", total_reward)
    
    
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.savefig('thompson_sampling_histogram.png')
print("Histogram saved as thompson_sampling_histogram.png")