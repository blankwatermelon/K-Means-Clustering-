import numpy as np

class KMeans:
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.centroids = None
        self.assignments = None

    @staticmethod
    # eucledian distance
    def euclidean_distance(point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2))

    # randomly choose k data points as initial centroids
    def initialize_random(self):
        indices = np.random.choice(len(self.dataset), self.k, replace=False)
        self.centroids = self.dataset[indices]

    # chooeses farthest data points as centroids
    def initialize_farthest_first(self):
        indices = np.random.choice(len(self.dataset), 1)
        self.centroids = self.dataset[indices]
        
        for _ in range(1, self.k):
            distances = np.array([min(self.euclidean_distance(point, centroid) for centroid in self.centroids) for point in self.dataset])
            next_centroid_index = np.argmax(distances)
            self.centroids = np.vstack([self.centroids, self.dataset[next_centroid_index]])

    # uses the kmeans++ method 
    def initialize_kmeanspp(self):
    # Step 1: Choose the first centroid randomly from the dataset
        first_centroid_index = np.random.choice(len(self.dataset))
        self.centroids = np.array([self.dataset[first_centroid_index]])

        # Step 2: Initialize an array to store the minimum distance to any centroid for each point
        distances = np.full(len(self.dataset), np.inf)

        # Step 3: Iteratively select the remaining k-1 centroids
        for _ in range(1, self.k):
            # Update the minimum distance to any centroid for each point
            for i, point in enumerate(self.dataset):
                dist_to_new_centroid = self.euclidean_distance(point, self.centroids[-1])
                distances[i] = min(dist_to_new_centroid, distances[i])

            # Choose the next centroid with a probability proportional to the square of the distance
            probabilities = distances ** 2
            probabilities /= probabilities.sum()  # Normalize to sum to 1

            next_centroid_index = np.random.choice(len(self.dataset), p=probabilities)
            self.centroids = np.vstack([self.centroids, self.dataset[next_centroid_index]])


    # assigment of data points to the closest centroid
    def assign_clusters(self):
        num_points = self.dataset.shape[0]  # Number of data points
        num_centroids = self.centroids.shape[0]  # Number of centroids
        distances = np.zeros((num_points, num_centroids))  # Initialize an array to hold the distances

        # Calculate the distance from each point to each centroid
        for point in range(num_points):
            for centroid in range(num_centroids):
                distances[point, centroid] = self.euclidean_distance(self.dataset[point], self.centroids[centroid])

        # Assign each point to the nearest centroid
        self.assignments = np.argmin(distances, axis=1)

    # update centroids using the average position of the points 
    def update_centroids(self):
        new_centroids = np.zeros((self.k, self.dataset.shape[1]))  # Create an array to hold the new centroids

        for i in range(self.k):
            # Calculate the mean of the points assigned to centroid i
            assigned_points = self.dataset[self.assignments == i]
            if len(assigned_points) > 0:  # Avoid division by zero
                new_centroids[i] = assigned_points.mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]  # Retain the old centroid if no points are assigned

        return new_centroids

    # takes the method then runs kmeans to convergence
    def cluster(self, method='random'):
        if method == 'random':
            self.initialize_random()
        elif method == 'farthest':
            self.initialize_farthest_first()
        elif method == 'kmeans++':
            self.initialize_kmeanspp()
        else:
            raise ValueError("Invalid initialization method specified.")
        
        iterations = []  # To store each iteration's centroids and assignments
        while True:
            old_centroids = self.centroids.copy()
            self.assign_clusters()
            self.centroids = self.update_centroids()
            iterations.append((self.assignments.copy(), self.centroids.copy()))

            # Check for convergence
            if np.all(np.sum((old_centroids - self.centroids) ** 2, axis=1) < 1e-8):
                break

        return iterations

