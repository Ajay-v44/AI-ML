import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


dataset = pd.read_csv('../kMeansClustering/Mall_Customers.csv')


X=dataset.iloc[:,[3,4]].values

dendrogram= dendrogram(linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
plt.savefig('dendrogram.png')
print("Dendrogram saved to dendrogram.png")
plt.clf()

from sklearn.cluster import AgglomerativeClustering
hierarchicalClustering=AgglomerativeClustering(n_clusters=5,linkage='ward')
y_hc=hierarchicalClustering.fit_predict(X)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='Cluster 5')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('clusters.png')
print("Plot saved to clusters.png")