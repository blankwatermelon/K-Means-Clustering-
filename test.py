import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans  # Assuming your KMeans class is in kmeans.py

# Test function to generate a random dataset, run KMeans, and display a plot
def test_kmeans():
    # Generate a random dataset
    dataset = np.random.rand(100, 2) * 20 - 10  # Random points within [-10, 10] range
    
    k = 4  # Number of centroids

    # Initialize KMeans
    kmeans = KMeans(dataset, k)

    # Run the clustering algorithm (you can choose initialization method)
    iterations = kmeans.cluster(method='kmeans++')  # You can also try 'farthest' or 'kmeans++'

    # Final centroids and assignments after convergence
    final_assignments = iterations[-1][0]  # Cluster assignments
    final_centroids = iterations[-1][1]  # Centroids

    print(f"Number of iterations until convergence: {len(iterations)}")

    # Plot the dataset and centroids
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset[:, 0], dataset[:, 1], c=final_assignments, cmap='viridis', label='Data Points')
    plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

    # Set axis limits and labels
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title('KMeans Clustering')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

# Run the test
if __name__ == '__main__':
    test_kmeans()
