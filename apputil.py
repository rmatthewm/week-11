import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn.cluster import KMeans

# Load the diamonds dataset from Seaborn as a global variable
df_diamonds = sns.load_dataset('diamonds')
df_diamonds_num = df_diamonds[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']]

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
    """ Runs kmeans with k clusters on the first n rows of data from
    the diamonds dataset

    Args:
        n (int): the number of rows to use
        k (int): the number of clusters

    Returns:
        tuple: a np.array of the centroids followed by an np.array of
        the cluster indices for each point in X
    """
    # Get just the first n rows
    X = df_diamonds_num.iloc[:n]
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    # The total time across the runs
    total_time = 0

    # Repeat the kmeans algorithm, keeping track of the total time
    for i in range(n_iter):
        # Using perf_counter() instead of time() because it's technically
        # designed for timing runtime performance, although it usually
        # doesn't matter
        start = time.perf_counter()
        kmeans_diamonds(n, k)
        end = time.perf_counter()

        # Calculate the elapsed time
        total_time += end - start

    # Return the average time per run
    return total_time / n_iter


if __name__ == '__main__':
    import random

    # Testing
    X = []
    for i in range(10):
        X.append([random.randint(0, 100), random.randint(0, 100)])

    X = np.array(X)

    print(kmeans(X, 3))

    print(kmeans_timer(1000, 5))