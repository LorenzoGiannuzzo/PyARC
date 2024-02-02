def kmeans_clustering(df, n_clusters=3, random_state=42):
    # Creazione di un dataframe temporaneo per le features di clustering
    features = df.pivot_table(values='M_consumption', index=['User', 'Year', 'Month'], columns='Hour').reset_index()

    # Esecuzione del k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    features['Cluster'] = kmeans.fit_predict(features.iloc[:, 3:])

    # Uniamo i risultati al dataframe originale
    result_df = pd.merge(df, features[['User', 'Year', 'Month', 'Cluster']], on=['User', 'Year', 'Month'], how='left')

    return result_df