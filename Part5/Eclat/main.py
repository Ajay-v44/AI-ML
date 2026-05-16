import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('../Apriori/Market_Basket_Optimisation.csv',header=None)
print(dataset.head())

transactions=[]
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
print(transactions)


from apyori import apriori
rules=apriori(transactions=transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
rules_list=list(rules)
# print(rules_list)

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(rules_list), columns = ['Product 1', 'Product 2', 'Support'])

resultsinDataFrame.nlargest(n = 10, columns = 'Support')