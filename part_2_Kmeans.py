import pandas as pd
import numpy as np
from part_2_nadav import handle_manual,handle_NaNs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score, confusion_matrix
import seaborn as sns
import itertools


if __name__ == '__main__':
    PrePrepareData = True
    PrepareData = True
    DO_PCA = False
    DO_Kmeans = True
    DO_Kmeans_performanceAnalysis = False
    DO_Kmeans_performanceAnalysis_manyClusters = False
    showWinner = True
    winner_transform_fromcode = True
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
        n_classes = 8
        kmeans = KMeans(n_clusters=n_classes, max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        df['cluster'] = kmeans.predict(X)

        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['cluster'] = df['cluster']
        df_pca['label'] = Y.values

        colors = {0: 'blue', 1: 'green', 2:'red',3:'purple',4:'yellow',5:'brown',6:'black', 7:'magenta',8:'lightblue',9:'ocean'}
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

        # Define the range of original labels (e.g., 0 to 3)
        original_labels_range = range(n_classes)  # Labels are 0, 1, 2, 3

        # Generate all possible mappings using itertools.product
        all_mappings = list(itertools.product([0, 1], repeat=len(original_labels_range)))
        all_label_mappings = [{original: new for original, new in zip(original_labels_range, mapping)} for mapping in
                              all_mappings]
        max_accuracy = 0
        winner_transform = []
        for i, mapping in enumerate(all_label_mappings):
            # Apply the mapping using the replace method
            transformed_series = predicted_labels.replace(mapping)
            print(f"Mapping {i}: {mapping}, Transformed Series: {transformed_series.tolist()}")

            conf_matrix = confusion_matrix(original_labels, transformed_series)

            f1 = f1_score(original_labels, transformed_series)
            accuracy = accuracy_score(original_labels, transformed_series)
            precision = precision_score(original_labels, transformed_series)
            recall = recall_score(original_labels, transformed_series)
            if accuracy>max_accuracy:
                max_accuracy = accuracy
                winner_transform = mapping

        # Winner
        transformed_series = predicted_labels.replace(winner_transform)
        conf_matrix = confusion_matrix(original_labels, transformed_series)
        f1 = f1_score(original_labels, transformed_series)
        accuracy = accuracy_score(original_labels, transformed_series)
        precision = precision_score(original_labels, transformed_series)
        recall = recall_score(original_labels, transformed_series)
        print(winner_transform)
        print(f"winner F1 Score: {f1}")
        print(f"winner Accuracy Score: {accuracy}")
        print(f"winner precision: {precision}")
        print(f"winner recall Score: {recall}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})
        plt.title('Confusion Matrix Heatmap', fontsize=24)
        plt.xlabel('Predicted', fontsize=24)
        plt.ylabel('Actual', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()

if showWinner:
    if winner_transform_fromcode:
        winner_transform = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 1}

    df['cluster'] = kmeans.predict(X)

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['cluster'] = df['cluster']
    df_pca['cluster'] = df_pca['cluster'].map(winner_transform)

    df_pca['label'] = Y.values

    colors = {1: 'blue', 0: 'green', 2: 'red', 3: 'purple', 4: 'yellow', 5: 'brown', 6: 'black', 7: 'magenta',
              8: 'lightblue', 9: 'ocean'}
    markers = {0: 'o', 1: '*'}  # Circle for label 0, star for label 1

    # Plot the PCA results with clusters using Matplotlib
    plt.figure(figsize=(10, 8))
    for cluster in range(2):
        print(cluster)
        label_subset = df_pca[df_pca['label'] == cluster]
        plt.scatter(label_subset['PC1'], label_subset['PC2'], c=colors[cluster],
                        marker='o', label='RainTomorrow True prediction' if cluster ==1 else 'RainTomorrow False prediction', alpha=0.6)
    # Customize the plot
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering with PCA')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()