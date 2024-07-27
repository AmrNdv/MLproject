import pandas as pd
import numpy as np
from part_2_nadav import handle_manual,handle_NaNs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score, confusion_matrix
import seaborn as sns


if __name__ == '__main__':
    PrePrepareData = True
    PrepareData = True
    DO_PCA = False
    DO_Kmeans = True
    DO_Kmeans_performanceAnalysis = False
    DO_Kmeans_performanceAnalysis_manyClusters = True
    ## Data loading:
    path_to_file = r'E:\t2\machine\project\part_1\pythonProject\Xy_train.csv'

    if PrePrepareData:
        ## Data Preperation part:
        NormalizationMethod = 'MinMaxScaler'# MinMaxScalar or StandardScaler

        # Data analysis in case it is not yet prepared
        df = pd.read_csv(path_to_file, encoding='ISO-8859-1')
        categorials = ['Location', 'WindGustDir', 'WindDir3pm', 'WindDir9am',
                       'Cloud9am', 'Cloud3pm', 'CloudsinJakarta', 'RainToday', 'RainTomorrow']
        df = handle_manual(df)
        df = handle_NaNs(df,categorials)

    if PrepareData:
        # Create dummy variables
        df = pd.get_dummies(df,columns =['WindGustDir','WindDir9am','WindDir3pm','Location'])

        # Create X and Y vectors:
        X = df.drop('RainTomorrow', axis=1)
        Y = df['RainTomorrow']

        # Normalize Data:
        if NormalizationMethod == 'MinMaxScaler':
            minmax_scaler = MinMaxScaler()
            X_normalized = minmax_scaler.fit_transform(X)
            Y_normalized = Y
        elif NormalizationMethod == 'StandardScaler':
            standard_scaler = StandardScaler()
            X_normalized = standard_scaler.fit_transform(X)
            Y_normalized = Y


    # First Part - Do Principal Components Analysis on our Dataset.
    # Present a graph of the principal components, which correspond to the first 2 eigenvectors of the data matrix.

    if DO_PCA:
        pca = PCA(n_components=2)
        pca.fit(X)
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_.sum())
        df_pca = pca.transform(X)
        df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
        df_pca['RainTommorow'] = Y
        # Transform the data and create a DataFrame
        df_pca = pca.transform(X)
        df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
        df_pca['RainTomorrow'] = Y

        # Plot the PCA results with different colors for RainTomorrow categories
        colors = {0: 'blue', 1: 'red'}
        plt.figure(figsize=(10, 8))

        for category in [0,1]:
            subset = df_pca[df_pca['RainTomorrow'] == category]
            plt.scatter(subset['PC1'], subset['PC2'], c=colors[category], label=f'RainTomorrow = {category}', alpha=0.6)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Dataset with RainTomorrow Categories')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Second Part - Do Principal Components Analysis on our Dataset.

    if DO_Kmeans:
        # Perform KMeans clustering
        n_classes = 4
        kmeans = KMeans(n_clusters=n_classes, max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        df['cluster'] = kmeans.predict(X)

        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['cluster'] = df['cluster']
        df_pca['label'] = Y.values

        colors = {0: 'blue', 1: 'green', 2:'red',3:'purple',4:'yellow',5:'brown',6:'black', 7:'magenta'}
        markers = {0: 'o', 1: '*'}  # Circle for label 0, star for label 1

        # Plot the PCA results with clusters using Matplotlib
        plt.figure(figsize=(10, 8))
        for cluster in range(n_classes):
            subset = df_pca[df_pca['cluster'] == cluster]
            for label in range(n_classes):
                label_subset = subset[subset['label'] == label]
                plt.scatter(label_subset['PC1'], label_subset['PC2'], c=colors[cluster],
                            marker='o', label=f'Cluster {cluster}, RainTomorrow - {str(bool(label))}', alpha=0.6)

        # Plot the cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='+', s=100, color='red', label='Centroids')

        # Customize the plot
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('KMeans Clustering with PCA')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


        colors = {1: 'blue', 0: 'green'}
        # Create a new figure for the real labels plot
        plt.figure(figsize=(10, 8))
        # Scatter plot for each real label
        for label in [0, 1]:
            label_subset = df_pca[df_pca['label'] == label]
            plt.scatter(label_subset['PC1'], label_subset['PC2'], c=colors[label], marker=markers[label],
                        label=f'RainTomorrow - {str(bool(label))}', alpha=0.6)

        # Customize the plot
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Real Labels with PCA')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show both plots
        plt.show()

    if DO_Kmeans_performanceAnalysis:

        original_labels = Y.values
        predicted_labels = df['cluster']


        predicted_labels_fixed_labels = predicted_labels.replace({0:1,1:0})
        # Adjust predicted labels to match original labels
        conf_matrix = confusion_matrix(original_labels, predicted_labels_fixed_labels)

        f1 = f1_score(original_labels, predicted_labels_fixed_labels)
        accuracy = accuracy_score(original_labels, predicted_labels_fixed_labels)
        precision = precision_score(original_labels, predicted_labels_fixed_labels)
        recall = recall_score(original_labels, predicted_labels_fixed_labels)

        print(f"F1 Score: {f1}")
        print(f"Accuracy Score: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall Score: {recall}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})
        plt.title('Confusion Matrix Heatmap', fontsize=24)
        plt.xlabel('Predicted', fontsize=24)
        plt.ylabel('Actual', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()

 if DO_Kmeans_performanceAnalysis_manyClusters:

        original_labels = Y.values
        predicted_labels = df['cluster']


        predicted_labels_fixed_labels = predicted_labels.replace({0:1,1:0})
        # Adjust predicted labels to match original labels
        conf_matrix = confusion_matrix(original_labels, predicted_labels_fixed_labels)

        f1 = f1_score(original_labels, predicted_labels_fixed_labels)
        accuracy = accuracy_score(original_labels, predicted_labels_fixed_labels)
        precision = precision_score(original_labels, predicted_labels_fixed_labels)
        recall = recall_score(original_labels, predicted_labels_fixed_labels)

        print(f"F1 Score: {f1}")
        print(f"Accuracy Score: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall Score: {recall}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})
        plt.title('Confusion Matrix Heatmap', fontsize=24)
        plt.xlabel('Predicted', fontsize=24)
        plt.ylabel('Actual', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()

if DO_Kmeans3d:
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)

    # Perform PCA for visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['cluster'] = df['cluster']
    df_pca['label'] = Y.values

    colors = {0: 'blue', 1: 'green'}
    markers = {0: 'o', 1: '*'}  # Circle for label 0, star for label 1

    # Create a 3D scatter plot for the clusters
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    for cluster in [0, 1]:
        subset = df_pca[df_pca['cluster'] == cluster]
        for label in [0, 1]:
            label_subset = subset[subset['label'] == label]
            ax1.scatter(label_subset['PC1'], label_subset['PC2'], label_subset['PC3'], c=colors[cluster],
                        marker=markers[label], label=f'Cluster {cluster}, Label {label}', alpha=0.6)

    # Plot the cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2], marker='+', s=100, color='red', label='Centroids')

    # Customize the plot
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_zlabel('Principal Component 3')
    ax1.set_title('KMeans Clustering with PCA')
    ax1.legend()
    ax1.grid(True)

    colors = {1: 'green', 0: 'blue'}
    # Create a 3D scatter plot for the real labels
    ax2 = fig.add_subplot(122, projection='3d')
    for label in [0, 1]:
        label_subset = df_pca[df_pca['label'] == label]
        ax2.scatter(label_subset['PC1'], label_subset['PC2'], label_subset['PC3'], c=colors[label], marker=markers[label],
                    label=f'Label {label}', alpha=0.6)

    # Customize the plot
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_zlabel('Principal Component 3')
    ax2.set_title('Real Labels with PCA')
    ax2.legend()
    ax2.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()
