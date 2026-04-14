import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn.cluster import KMeans

# Load the diamonds dataset from Seaborn as a global variable
df_diamonds = sns.load_dataset('diamonds')
df_diamonds_num = df_diamonds[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']]
print(df_diamonds_num.head())

def kmeans(X, k):
    """ Run the scikit learn kmeans model on the data X and return
    the results.

    Args:
        X (np.array): the data to cluster
        k (int): the number of clusters

    Returns:
        tuple: a np.array of the centroids followed by an np.array of
        the cluster indices for each point in X
    """
    # Create the KMeans model
    km = KMeans(n_clusters=k)
    km.fit(X)

    # Return the centroids and the centroid labels corresponding to each point
    return km.cluster_centers_, km.labels_

def kmeans_diamonds(n, k):
    pass


if __name__ == '__main__':
    import random

    # Testing
    X = []
    for i in range(10):
        X.append([random.randint(0, 100), random.randint(0, 100)])

    X = np.array(X)

    print(kmeans(X, 3))