from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

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
            tol=1e-04, random_state=101
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


def separate_clusters(init_data, cluster_number, plot=False):
    print("Starting separating clusters")
    clustered_data = []
    for i in range(cluster_number):
        data = init_data[init_data['CLUSTER'] == i]
        clustered_data.append(data)
    if plot:
        print("Plotting PSA")
        pca = PCA(n_components=2)

        pca_res = pd.DataFrame(data=pca.fit_transform(init_data),
                               columns=['principal component 1', 'principal component 2'])
        print(pca.explained_variance_)
        print(pca.explained_variance_ratio_)
        pca_res = pd.concat([pca_res, init_data[["CLUSTER"]]], axis = 1)
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = [0, 1, 2, 3, 4, 5, 6]
        colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
        for target, color in zip(targets,colors):
            values = pca_res['CLUSTER'] == target
            ax.scatter(pca_res.loc[values, 'principal component 1']
                       , pca_res.loc[values, 'principal component 2']
                       , c = color
                       , s = 50)
        ax.legend(targets)
        ax.grid()

        # print("Plotting TSNE graph")
        # tsne = TSNE(n_components=2)
        #
        # tsne_res = pd.DataFrame(data=tsne.fit_transform(init_data),
        #                        columns=['x axis', 'y axis'])
        # tsne_res = pd.concat([tsne_res, init_data[["CLUSTER"]]], axis = 1)
        # fig = plt.figure(figsize = (8,8))
        # ax = fig.add_subplot(1,1,1)
        # ax.set_xlabel('X axis', fontsize = 15)
        # ax.set_ylabel('Y axis', fontsize = 15)
        # ax.set_title('TSNE reduction', fontsize = 20)
        # targets = [0, 1, 2, 3, 4, 5, 6]
        # colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
        # for target, color in zip(targets,colors):
        #     values = tsne_res['CLUSTER'] == target
        #     ax.scatter(tsne_res.loc[values, 'x axis']
        #                , tsne_res.loc[values, 'y axis']
        #                , c = color
        #                , s = 50)
        # ax.legend(targets)
        # ax.grid()
    return clustered_data


def preprocess(input_data, best_cols, all_cols):
    # remove customer id
    input_data.drop('CUST_ID', 1, inplace=True)
    # remove columns which we didn't define as important
    removal = [col for col in all_cols if col not in best_cols]
    for col in removal:
        input_data.drop(col, 1, inplace=True)
    # print number of missing values per column
    print(input_data.isna().sum())
    # credit_limit has 1 missing and minimum payments 313
    # check the mean and median
    if "MINIMUM_PAYMENTS" in best_cols:
        print("Mean of minimum payments: ", input_data['MINIMUM_PAYMENTS'].mean())
        print("Median of minimum payments: ", input_data['MINIMUM_PAYMENTS'].median())
    if "CREDIT_LIMIT" in best_cols:
        print("Mean of credit limit: ", input_data['CREDIT_LIMIT'].mean())
        print("Median of credit limit: ", input_data['CREDIT_LIMIT'].median())
    # we will use median because of mean's drawbacks (large values skew it too much)
    if "MINIMUM_PAYMENTS" in best_cols:
        input_data['MINIMUM_PAYMENTS'].fillna(input_data['MINIMUM_PAYMENTS'].median(), inplace=True)
    if "CREDIT_LIMIT" in best_cols:
        input_data['CREDIT_LIMIT'].fillna(input_data['CREDIT_LIMIT'].median(), inplace=True)
    # normalize data
    original_data = input_data.copy()
    input_data = (input_data - input_data.mean())/input_data.std()
    # input_data['TENURE'] = tenure
    return input_data, original_data

def cluster_data(columns):
    input_data = read_csv("./credit_card_data.csv")
    # list of all columns
    all_cols = ["BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
                  "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
                  "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT",
                  "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"]
    # pre-processing
    input_data, original_data = preprocess(input_data, columns, all_cols)
    # determining number of clusters
    cluster_num = number_of_clusters(data=input_data, plot=True)
    # number we got
    # cluster_num = 7
    # do K Means fitting
    km = KMeans(n_clusters=cluster_num, init="k-means++",
                n_init=10, max_iter=300,
                tol=1e-04, random_state=101)
    clusters = km.fit_predict(input_data)
    # selects and labels data with cluster information
    original_data['CLUSTER'] = clusters
    # separates clusters into a list of dataframes
    clustered_data = separate_clusters(original_data, cluster_num, plot=True)
    columns.append("CLUSTER")
    sns.pairplot(original_data, hue="CLUSTER")
    plt.show()
    # summary for each cluster
    for cluster in range(cluster_num):
        print("Cluster number " + str(cluster))
        clustered_data[cluster].describe().to_csv('cluster_' + str(cluster) + ".csv")
    # summary for all variables
    original_data.describe().to_csv('total.csv')
    # visualizing box plots
    cols = list(original_data.columns)
    for col in cols:
        if col == "CLUSTER":
            continue
        data = original_data[[col, "CLUSTER"]]
        sns.boxplot(x="CLUSTER", y=col, data=data, hue="CLUSTER")
        plt.show()

def attempt_1():
    best_cols = ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
    cluster_data(best_cols)

def attempt_2():
    best_cols = ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES_FREQUENCY",
                 "PURCHASES_INSTALLMENTS_FREQUENCY", "PRC_FULL_PAYMENT"]
    cluster_data(best_cols)


if __name__ == '__main__':
    # svi pokusaji su dokumentovani u svojim folderima
    # attempt_1()
    attempt_2()
