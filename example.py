if DO_Kmeans:
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