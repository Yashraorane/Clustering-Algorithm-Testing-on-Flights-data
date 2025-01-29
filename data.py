from matplotlib import pyplot as plt #for plotting the reduced data
import numpy as np #for numerical operations
import pandas as pd #for data manipulation
from sklearn.preprocessing import StandardScaler #for standardizing the dataset
from sklearn.cluster import KMeans, DBSCAN,SpectralClustering #for clustering the dataset
from sklearn.decomposition import PCA #for reducing the dimensionality of the dataset
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score #for evaluating the clustering
from sklearn.metrics.pairwise import rbf_kernel #for kernel algorithm
from sklearn.metrics.pairwise import polynomial_kernel #for polynomial algorithm
import matplotlib #for plotting the reduced data
matplotlib.use('Agg')



#Determining the best cluster count, silhouette score, calinski harabasz score, davies bouldin score (helper function)
def evaluate_cluster(pca_result, labels, best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count, cluster_count):
    #Getting the metrics
    silhouette = silhouette_score(pca_result, labels)
    # print(f'silhouette',silhouette)
    calinski_harabasz = calinski_harabasz_score(pca_result, labels)
    # print(f'calinski_harabasz',calinski_harabasz)
    davies_bouldin = davies_bouldin_score(pca_result, labels)
    # print(f'davies_bouldin',davies_bouldin)

    #Using tuple to store the current metric and the best metric
    current_metric = (silhouette, calinski_harabasz, -davies_bouldin)
    #print(f'current_metric',current_metric)
    best_metric = (best_silhouette, best_calinski_harabasz, -best_davies_bouldin)
    #print(f'best_metric',best_metric)

    #Checking if the current metric is greater than the best metric,if it is then we will update the best metric,
    #for daives bouldin score we are using negative value, as the lower the value the better the clustering
    if current_metric > best_metric:
        best_silhouette = silhouette
        best_calinski_harabasz = calinski_harabasz
        best_davies_bouldin = davies_bouldin
        best_clusters_count = cluster_count

    return best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count

#Flitering the dataset based on the zone, calendaryear and season (helper function)
def filtering_data(dataSet,zone,calendaryear,season):
    print('flitering_data function called where data is filtered based on zone, calendaryear and season')
    #Filtering the dataset based on the zone, calendaryear and season
    dataSet=dataSet[dataSet['Zone']==zone]  
    print('number of rows and columns after filtering zone:',dataSet.shape)
    #calendaryear made int
    calendaryear=int(calendaryear)
    dataSet=dataSet[dataSet['CalendarYear']==calendaryear]
    print('number of rows and columns after filtering calendaryear:',dataSet.shape)
    dataSet=dataSet[dataSet['Season']==season]
    print('number of rows and columns after filtering season:',dataSet.shape)
    return dataSet

#Determining reduce data with pca (helper function)
def perform_pca(dataSet, columns, n_comps=2, rand=42):
    print('perform pca function to reduce the dimensionality of the dataset')
    #Getting the clean dataset by using the columns
    clean_dataSet = dataSet[columns]
    print(f'Number of rows and columns use for PCA:',clean_dataSet.shape)
    #Standardizing the dataset
    scalar = StandardScaler()
    #Fitting the scalar to the clean dataset
    dataSet_Scaled = scalar.fit_transform(clean_dataSet)
    print(f'dataSet_Scaled',dataSet_Scaled.shape)
    #Reducing the dimensionality of the dataset to 2 and 
    #using the random state of 42 for reproducibility
    pca = PCA(n_components=n_comps, random_state=rand)
    #Fitting the PCA to the scaled dataset
    reduce = pca.fit_transform(dataSet_Scaled)
    # print(f'PCA explained variance ratio:',pca.explained_variance_ratio_)
    # print(f'Reduced data info:',reduce.shape)

    # #Plotting the reduced data
    # plt.figure(figsize=(10, 10))
    # plt.scatter(reduce[:, 0], reduce[:, 1], c='blue',edgecolors='k',s=50)
    # plt.title('Reduced Data')
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Second Principal Component')
    # plt.grid(True)
    # plt.savefig('C:/Users/Aanand/Desktop/COMP702/src/static/images/ReducedData.png')

    return reduce, clean_dataSet

#Trial implement of dunn index (not successful hence not used in the final code)
# def calculate_dunn_index(data, clusters):
#     intra_dist =[]
#     inter_dist =[]
#     unique_clusters = set(clusters)-{-1} #Excluding noise points labeled as -1
#     for cluster in unique_clusters:
#         cluster_data = data[clusters==cluster]
#         if len(cluster_data)>1: #only consider cluster with more than one point
#             intra_dist.append(np.mean(pairwise_distances(cluster_data)))
    
#     for i in unique_clusters:
#         for j in unique_clusters:
#             if i<j: # ensure that we do not calculate the same distance twice
#                 inter_cluster_data_i =data[clusters==i]
#                 inter_cluster_data_j =data[clusters==j]
#                 if len(inter_cluster_data_i)>0 and len(inter_cluster_data_j)>0:
#                     inter_dist.append(np.mean(pairwise_distances(inter_cluster_data_i,inter_cluster_data_j)))

#     if intra_dist and inter_dist:
#         return np.min(inter_dist)/np.max(intra_dist)

#Declaring the clustering function
#DBScan Algorithm
def dbscan(pca_result,dataSet):
    #Declaring the best scores
    best_silhouette = -np.inf
    best_calinski_harabasz = -np.inf
    best_davies_bouldin = np.inf
    best_eps=0.1 #eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other
    best_minsample=5 #The number of samples in a neighborhood for a point to be considered as a core point
    #Note: Thereis no concept of "best cluster count" in DBSCAN, as the number of clusters is determined by the eps and min_samples parameters
    # best_clusters_count = 1

    for eps in np.arange(0.1, 0.5, 0.1):
    #Looping through the cluster count from 3 to 7
        for minsamples in range(5, 10):
            #Declaring the standard algorithm
            dbscan = DBSCAN(eps=eps, min_samples=minsamples)
            labels = dbscan.fit_predict(pca_result)

            if len(set(labels))<=1:
                continue

            #Getting the metrics
            silhouette = silhouette_score(pca_result, labels)
            calinski_harabasz = calinski_harabasz_score(pca_result, labels)
            davies_bouldin = davies_bouldin_score(pca_result, labels)

            #Checking if the silhouette score is greater than the best silhouette score
            if silhouette > best_silhouette:
                temp_best = True
            elif silhouette == best_silhouette and davies_bouldin < best_davies_bouldin:
                temp_best = True
            elif silhouette == best_silhouette and davies_bouldin == best_davies_bouldin and calinski_harabasz > best_calinski_harabasz:
                temp_best = True
            else:
                temp_best = False

            #Checking if the temp_best is true
            if temp_best:
                best_silhouette = silhouette
                best_calinski_harabasz = calinski_harabasz
                best_davies_bouldin = davies_bouldin
                # best_clusters_count = cluster_count
                best_eps=eps
                best_minsample=minsamples

    print(f'Best Silhouette Score: {best_silhouette}', f'Best Calinski Harabasz: {best_calinski_harabasz}', f'Best Davies Bouldin: {best_davies_bouldin}', f'Best Eps: {best_eps}', f'Best Minsample: {best_minsample}')

    #Declaring the final DBscan algorithm
    final_dbscan = DBSCAN(eps=best_eps, min_samples=best_minsample)
    final_labels = final_dbscan.fit_predict(pca_result)

    #Adding the cluster labels to the dataset
    dataSet_copy = dataSet.copy()
    dataSet_copy['Cluster_Tags'] = final_labels
    mean_cls = dataSet_copy.groupby('Cluster_Tags').mean()
    print(mean_cls)

    return final_labels

#Kernel algorithm
def kernel(pca_result,dataSet):

    #Note: For experimental purpose we are taking only 50 sub clusters, and then selecting 10 random clusters from, and then taking the indexes of the selected clusters
    #and then taking the data of the selected clusters. This is done to reduce the computation time, and to get the data of the selected clusters
    np.random.seed(42)
    sub_cluster =50
    km=KMeans(n_clusters=sub_cluster, random_state=42)
    #get the cluster labels which will be used to select the random clusters
    sub_cls_label=km.fit_predict(pca_result) 
    no_select_cls=10
    select_cls=np.random.choice(sub_cluster,no_select_cls,replace=False) #selecting 10 random clusters
    experiment_index=np.where(np.isin(sub_cls_label,select_cls))[0] #get the index of the selected clusters
    experiment_pca_result=pca_result[experiment_index, :] #get the data of the selected clusters
    test_data=dataSet.iloc[experiment_index].copy() #get the data of the selected clusters

    scalar = StandardScaler()
    experiment_pca_result=scalar.fit_transform(experiment_pca_result)
    affinity_matrix=rbf_kernel(experiment_pca_result, gamma=0.001)

    #Declaring the best scores
    best_silhouette = -np.inf
    best_calinski_harabasz = -np.inf
    best_davies_bouldin = np.inf
    best_clusters_count = 1

    #Looping through the cluster count from 3 to 7
    for cluster_count in range(4, 9):
        #Declaring the standard algorithm
        kernel = SpectralClustering(n_clusters=cluster_count, affinity='precomputed',assign_labels='kmeans', random_state=0)
        #getting the labels for evaluating metrics
        labels = kernel.fit_predict(affinity_matrix)
        #Calling the evaluate_cluster function to get the best scores
        best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count = evaluate_cluster(experiment_pca_result, labels,
                    best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count, cluster_count)

    print(f'Best Silhouette Score: {best_silhouette}', f'Best Calinski Harabasz: {best_calinski_harabasz}', f'Best Davies Bouldin: {best_davies_bouldin}', f'Best Clusters Count: {best_clusters_count}')

    #Declaring the final Kernel algorithm
    final_kernel = SpectralClustering(n_clusters=best_clusters_count, affinity='precomputed',assign_labels='kmeans', random_state=0)
    final_labels = final_kernel.fit_predict(affinity_matrix)

    #Adding the cluster labels to the dataset
    # dataSet_copy = dataSet.copy()
    test_data['Cluster_Tags'] = final_labels
    #Getting the mean of the cluster tags
    mean_cls = test_data.groupby('Cluster_Tags').mean()
    print(mean_cls)

    return final_labels

#Normalized algorithm
def normalized(pca_result,dataSet):
    #Declaring the best scores
    best_silhouette = -np.inf
    best_calinski_harabasz = -np.inf
    best_davies_bouldin = np.inf
    best_clusters_count = 1

    #Looping through the cluster count from 3 to 7
    for cluster_count in range(3, 8):
        #Declaring the standard algorithm, PCA does not provide labels, so we are using the nearest neighbors to get the labels.
        normalized = SpectralClustering(n_clusters=cluster_count, affinity='nearest_neighbors',eigen_tol="auto", eigen_solver='arpack' ,  n_init=20, random_state=0)
        #Getting the labels, which will be used to evaluate the metrics
        labels = normalized.fit_predict(pca_result)
        #Calling the evaluate_cluster function to get the best scores
        best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count = evaluate_cluster(pca_result, labels,
                    best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count, cluster_count)       

    print(f'Best Silhouette Score: {best_silhouette}', f'Best Calinski Harabasz: {best_calinski_harabasz}', f'Best Davies Bouldin: {best_davies_bouldin}', f'Best Clusters Count: {best_clusters_count}')

    #Declaring the final Nornaliszed algorithm
    final_normalized = SpectralClustering(n_clusters=best_clusters_count, affinity='nearest_neighbors',eigen_solver="arpack", eigen_tol="auto", random_state=0)
    #Getting the final labels used for clustering airport data
    final_labels = final_normalized.fit_predict(pca_result)

    #Adding the cluster labels to the dataset
    dataSet_copy = dataSet.copy()
    dataSet_copy['Cluster_Tags'] = final_labels
    #Taking the mean of all the features based on the cluster tags
    mean_cls = dataSet_copy.groupby('Cluster_Tags').mean()
    print(mean_cls)

    return final_labels

#Standard algorithm
def standardized(pca_result,dataSet):
    #Declaring the best scores as negative infinity and positive infinity
    best_silhouette = -np.inf
    best_calinski_harabasz = -np.inf
    best_davies_bouldin = np.inf
    best_clusters_count = 1

    #Looping through the cluster count from 3 to 7
    for cluster_count in range(3, 8):
        #Declaring the standard algorithm, PCA does not provide labels, so we are using the nearest neighbors to get the labels.
        standard = SpectralClustering(n_clusters=cluster_count, affinity='nearest_neighbors',n_init=25,eigen_tol=0.0, random_state=0)
        #Getting the labels, which will be used to evaluate the metrics
        labels = standard.fit_predict(pca_result)
        #Calling the evaluate_cluster function to get the best scores
        best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count = evaluate_cluster(pca_result, labels,
                    best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count, cluster_count)

    print(f'Best Silhouette Score: {best_silhouette}', f'Best Calinski Harabasz: {best_calinski_harabasz}', f'Best Davies Bouldin: {best_davies_bouldin}', f'Best Clusters Count: {best_clusters_count}')

    #Declaring the final standard algorithm
    final_standard = SpectralClustering(n_clusters=best_clusters_count, affinity='nearest_neighbors', eigen_tol=0.0, random_state=0)
    #Getting the final labels used for clustering airport data
    final_labels = final_standard.fit_predict(pca_result)

    #Adding the cluster labels to the dataset
    dataSet_copy = dataSet.copy()
    dataSet_copy['Cluster_Tags'] = final_labels
    #Getting the mean of all the features based on the cluster tags
    mean_cls = dataSet_copy.groupby('Cluster_Tags').mean()
    print(mean_cls)

    return final_labels

#KMeans algorithm
def kMeans(pca_result,dataSet):
    #Declaring the best scores as negative infinity and positive infinity
    best_silhouette = -np.inf
    best_calinski_harabasz = -np.inf
    best_davies_bouldin = np.inf
    best_clusters_count = 1
    # print(f'KMeans')
    #Looping through the cluster count from 3 to 7 to get the best cluster count
    for cluster_count in range(3, 8):
        #Declaring the KMeans algorithm
        kmeans = KMeans(n_clusters=cluster_count, random_state=42)
        #Getting the labels, PCA does not provide labels, so we are using the K-means to get the labels.
        labels = kmeans.fit_predict(pca_result)
        # print(f'KMeans-labels',labels)
        #Calling the evaluate_cluster function to get the best scores
        best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count = evaluate_cluster(pca_result, labels,
                    best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count, cluster_count)

    print(f'Best Silhouette Score: {best_silhouette}', f'Best Calinski Harabasz: {best_calinski_harabasz}', f'Best Davies Bouldin: {best_davies_bouldin}', f'Best Clusters Count: {best_clusters_count}')

    #Declaring the final KMeans algorithm
    final_kmeans = KMeans(n_clusters=best_clusters_count, random_state=42)
    #Getting the final labels used for clustering airport data
    final_labels = final_kmeans.fit_predict(pca_result)

    #Adding the cluster labels to the dataset
    dataSet_copy = dataSet.copy()
    dataSet_copy['Cluster_Tags'] = final_labels
    #Getting the mean of all the features based on the cluster tags
    mean_cls = dataSet_copy.groupby('Cluster_Tags').mean()
    print(mean_cls)

    return final_labels

#Polynomial algorithm
def polynomial(pca_result,dataSet):

    #Note: For experimental purpose we are taking only 50 sub clusters, and then selecting 10 random clusters from, and then taking the indexes of the selected clusters
    #and then taking the data of the selected clusters. This is done to reduce the computation time, and to get the data of the selected clusters
    np.random.seed(42)
    sub_cluster =50
    km=KMeans(n_clusters=sub_cluster, random_state=42)
    sub_cls_label=km.fit_predict(pca_result)
    no_select_cls=10
    select_cls=np.random.choice(sub_cluster,no_select_cls,replace=False) #selecting 10 random clusters
    experiment_index=np.where(np.isin(sub_cls_label,select_cls))[0] #get the index of the selected clusters
    experiment_pca_result=pca_result[experiment_index, :] #get the data of the selected clusters
    test_data=dataSet.iloc[experiment_index].copy() #get the data of the selected clusters

    scalar = StandardScaler()
    experiment_pca_result=scalar.fit_transform(experiment_pca_result)
    # for polynomial kernel we need to set the coef0 and degree
    affinity_matrix=polynomial_kernel(experiment_pca_result, coef0=1, degree=3)
    #Print the entire matrix
    # for i in range(len(affinity_matrix)):
    #     print(affinity_matrix[i])
    

    #Declaring the best scores as negative infinity and positive infinity
    best_silhouette = -np.inf
    best_calinski_harabasz = -np.inf
    best_davies_bouldin = np.inf
    best_clusters_count = 1

    #Looping through the cluster count from 3 to 7
    for cluster_count in range(4, 9):
        #Declaring the standard algorithm
        poly = SpectralClustering(n_clusters=cluster_count, affinity='precomputed',assign_labels='discretize',n_init=10, random_state=0)
        # print(f'Polynomial Kernel',poly)
        #Getting the labels, PCA does not provide labels, so we are using the K-means to get the labels.
        labels = poly.fit_predict(affinity_matrix)
        # print(f'Polynomial Kernel-labels',labels)
        #Calling the evaluate_cluster function to get the best scores
        best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count = evaluate_cluster(experiment_pca_result, labels,
                    best_silhouette, best_calinski_harabasz, best_davies_bouldin, best_clusters_count, cluster_count)

    print(f'Best Silhouette Score: {best_silhouette}', f'Best Calinski Harabasz: {best_calinski_harabasz}', f'Best Davies Bouldin: {best_davies_bouldin}', f'Best Clusters Count: {best_clusters_count}')

    #Declaring the final polynomial algorithm
    final_poly = SpectralClustering(n_clusters=best_clusters_count, affinity='precomputed',assign_labels='discretize',n_init=10, random_state=0)
    #Getting the final labels used for clustering airport data
    final_labels = final_poly.fit_predict(affinity_matrix)

    #Adding the cluster labels to the dataset
    # dataSet_copy = dataSet.copy()
    test_data['Cluster_Tags'] = final_labels
    #Getting the mean of all the features based on the cluster tags
    mean_cls = test_data.groupby('Cluster_Tags').mean()
    print(mean_cls)

    return  final_labels



#Declaring Region map libraries by using Origin state colunm
REGION_MAP={
        'WA':'West','OR':'West','CA':'West','NV':'West','ID':'West','AZ':'West','UT':'West','MT':'West','WY':'West','AK':'West','NM':'West','CO':'West','HI':'West',
        'GA':'South','FL':'South','SC':'South','TX':'South','OK':'South','AR':'South','NC':'South','LA':'South','MS':'South','AL':'South','VA':'South','WV':'South','TN':'South','KY':'South','DE':'South','MD':'South','DC':'South',
        'ND':'Midwest','WI':'Midwest','IL':'Midwest','IN':'Midwest','SD':'Midwest','NE':'Midwest','KS':'Midwest','MN':'Midwest','MI':'Midwest','OH':'Midwest','IA':'Midwest','MO':'Midwest',
        'PA':'Northeast','NY':'Northeast','NJ':'Northeast','CT':'Northeast','RI':'Northeast','VT':'Northeast','MA':'Northeast','ME':'Northeast','NH':'Northeast'
}
    
#Declaring Season map libraries by using Flight_date column
SEASON_MAP={
        12:'Winter',1:'Winter',2:'Winter',
        3:'Spring',4:'Spring',5:'Spring',
        6:'Summer',7:'Summer',8:'Summer',
        9:'Fall',10:'Fall',11:'Fall'
}

#Decalring fine tune model function
def fine_tune_model(file_path,zone,calendaryear,season,algorithm):
    print('fine_tune_model function called where data is loaded and cleaned')
    #Reading the csv file using pandas library
    dataSet = pd.read_csv(file_path)
    print('Number of rows and columns while loading data:',dataSet.shape)
    #Part1
    #We need to split the Origin_city column to get the city and region, cause each record column has 2 information combined for example "Manhattan, KS"
    dataSet[['Origin_city','Origin_region']]=dataSet['Origin_city'].str.split(', ',expand=True)
    #Similarly we need to split the Destination_city column to get the city and region, cause each record column has 2 information combined for example "Manhattan, KS"
    dataSet[['Destination_city','Destination_region']]=dataSet['Destination_city'].str.split(', ',expand=True)
    
    dataSet[['Origin_airport','Destination_airport','Origin_city','Origin_region','Destination_city','Destination_region']]


    #Part2
    #Changing the Flight_date column to datetime format
    dataSet['Flight_date'] = pd.to_datetime(dataSet['Fly_date'], format='%Y-%m-%d')
    # dataSet.head(1)
    #Adding the CalendarYear column to the dataset
    dataSet['CalendarYear']=dataSet['Flight_date'].dt.year
    #Creating dataframe
    dataSet = pd.DataFrame(dataSet)
    #Dropping the columns that are not needed
    dataSet.drop(columns=['Fly_date'],inplace=True)
    #reorodering the columns
    new_col = ['Origin_airport','Destination_airport','Origin_city','Origin_region',
                       'Destination_city','Destination_region','Passengers','Seats','Flights','Distance',
                       'Origin_population','Destination_population','Org_airport_lat','Org_airport_long',
                       'Dest_airport_lat','Dest_airport_long','Flight_date','CalendarYear']
    dataSet = dataSet[new_col]


    #Part3
    #Checking for the null values and dropping the rows with null values
    # nullrows= dataSet[dataSet.isna().any(axis=1)]
    # nullrows
    dataSet=dataSet.dropna()
    #Checking for the 0 values in the Passengers and Seats columns
    # noseatpassenger=dataSet[(dataSet['Passengers']!=0) | (dataSet['Seats']!=0)]
    dataSet.drop(index=dataSet[(dataSet['Passengers']==0) | (dataSet['Seats']==0)].index,inplace=True)
    # #Checking for the 0 values in the Flights column
    # noflights=dataSet[dataSet['Flights']==0]
    dataSet=dataSet[dataSet['Flights']!=0]
    # #Checking for the 0 values in the Flights column
    # noflights=dataSet[dataSet['Flights']==0]


    #Part4
    #Getting the unique values of the Origin_region column
    dataSet['Origin_region'].unique()
    #Adding the zone column to the dataset by mapping the Origin_region column to the REGION_MAP dictionary
    dataSet['Zone'] = dataSet['Origin_region'].map(REGION_MAP)
    # print(dataSet)
    #Adding the Season column to the dataset by mapping the Flight_date column to the SEASON_MAP dictionary
    dataSet['Season']=dataSet['Flight_date'].dt.month.map(SEASON_MAP)
    # print(dataSet)
    print('Number of rows and columns after cleaning data:',dataSet.shape)


    #Part5
    #Filtering the dataset based on the zone, calendaryear and season
    dataSet=filtering_data(dataSet,zone,calendaryear,season)
    print('Number of rows and columns after filtering data:',dataSet.shape)
    # print(dataSet.describe())
    #For testing purpose taking only first 50 records
    # dataSet=dataSet.head(20)


    #Part6
    #Getting the columns that we want to use for the PCA
    columns = ['Passengers','Seats','Flights','Distance','Origin_population',
               'Org_airport_lat','Org_airport_long','Dest_airport_lat','Dest_airport_long']
    #Getting the reduce data with pca
    reduce, clean_dataSet = perform_pca(dataSet, columns)
    print(f'After performing PCA reduced data',reduce.shape)


    #part7
    #Checking the algorithm to be used which is passed as an argument
    ALGO_MAP = {
        'kmeans': kMeans,
        'standard': standardized,
        'normalized': normalized,
        'dbscan': dbscan,
        'kernel': kernel,
        'polynomial': polynomial,
    }
    #Getting the called clustering function
    called_clustering_function = ALGO_MAP.get(algorithm)
    #Checking if the function is not found
    if called_clustering_function is None:
        raise ValueError("function not found")
    #Getting the clustering labels for reduce data by evaluating best cluster count
    clustering_labels=called_clustering_function(reduce, clean_dataSet)

    #Returning the dataset, clustering labels, metrics
    return dataSet, reduce, clustering_labels
