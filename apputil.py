import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    # Create the KMeans model
    km = KMeans(n_clusters=k)
    km.fit(X)

    centroids = km.cluster_centers_

    return centroids



if __name__ == '__main__':
    import random

    # Testing
    X = []
    for i in range(10):
        X.append([random.randint(0, 100), random.randint(0, 100)])

    X = np.array(X)

    print(kmeans(X, 3))