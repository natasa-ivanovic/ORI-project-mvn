from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import sqrt


def optimal_number_of_clusters(wcss):
    """https://jtemporal.com/kmeans-and-elbow-method/"""
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 2


def number_of_clusters(data, plot=False, min_size=2, max_size=20):
    distortions = []
    for i in range(min_size, max_size):
        km = KMeans(
            n_clusters=i, init='k-means++',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(data)
        distortions.append(km.inertia_)
        # print(str(i) + ". iteration change compared to previous: " + str(distortions[i-3] - km.inertia_))
    num_clusters = optimal_number_of_clusters(distortions)
    print("Optimal cluster number is: " + str(num_clusters))
    if plot:
        plt.plot(range(min_size, max_size), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()
    return num_clusters


if __name__ == '__main__':
    input_data = read_csv("./credit_card_data.csv")
    # pre-processing
    input_data = input_data.drop('CUST_ID', 1)
    input_data = input_data.dropna()

    # determining number of clusters
    # cluster_num = number_of_clusters(data=input_data, plot=True)
    cluster_num = 8
    # do K Means fitting
    km = KMeans(n_clusters=cluster_num, init="k-means++",
                n_init=10, max_iter=300,
                tol=1e-04, random_state=0)
    clusters = km.fit_predict(input_data)

