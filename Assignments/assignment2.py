import math

def euclidean_distance(x, u):
    return math.sqrt((x[0] - u[0]) ** 2 + (x[1] - u[1]) ** 2)

def update_clusters(dataset, cluster_centres):
    clusters = [[] for _ in range(len(cluster_centres))]
    for data_point in dataset:
        closest_cluster_centre = min(cluster_centres, key=lambda c: euclidean_distance(data_point, c))
        cluster_index = cluster_centres.index(closest_cluster_centre)
        clusters[cluster_index].append(data_point)
    return clusters

def update_centres(clusters):
    new_cluster_centres = []
    for cluster in clusters:
        if cluster:
            x_sum, y_sum = zip(*cluster)
            new_cluster_centre = (sum(x_sum) / len(cluster), sum(y_sum) / len(cluster))
            new_cluster_centres.append(new_cluster_centre)
    return new_cluster_centres

def k_means(dataset, initial_cluster_centres):
    cluster_centres = initial_cluster_centres
    while True:
        clusters = update_clusters(dataset, cluster_centres)
        new_cluster_centres = update_centres(clusters)
        if new_cluster_centres == cluster_centres:
            return new_cluster_centres
        cluster_centres = new_cluster_centres

def sum_of_squares_error(dataset, cluster_centres):
    error = sum(euclidean_distance(data_point, min(cluster_centres, key=lambda c: euclidean_distance(data_point, c))) ** 2 for data_point in dataset)
    return round(error, 4)

def main():
    dataset = [(0.22, 0.33), (0.45, 0.76), (0.73, 0.39), (0.25, 0.35), (0.51, 0.69),
               (0.69, 0.42), (0.41, 0.49), (0.15, 0.29), (0.81, 0.32), (0.50, 0.88),
               (0.23, 0.31), (0.77, 0.30), (0.56, 0.75), (0.11, 0.38), (0.81, 0.33),
               (0.59, 0.77), (0.10, 0.89), (0.55, 0.09), (0.75, 0.35), (0.44, 0.55)]

    initial_cluster_centres = [(float(input()), float(input())) for _ in range(3)]

    final_cluster_centres = k_means(dataset, initial_cluster_centres)
    error = sum_of_squares_error(dataset, final_cluster_centres)
    print(error)

if __name__ == '__main__':
    main()
