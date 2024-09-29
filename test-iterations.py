import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans  # Assuming your KMeans class is in kmeans.py

# Test function to generate a random dataset, run KMeans, and visualize the iterations
def test_kmeans_iterations():
    # Generate a random dataset
    dataset = np.random.rand(100, 2) * 20 - 10  # Random points within [-10, 10] range
    k = 4  # Number of centroids

    # Initialize KMeans
    kmeans = KMeans(dataset, k)

    # Run the clustering algorithm (you can choose initialization method)
    iterations = kmeans.cluster(method='kmeans++')  # You can also try 'farthest' or 'kmeans++'

    # Log and plot the results for each iteration
    for i, (assignments, centroids) in enumerate(iterations):
        print(f"Iteration {i+1}:")
        print(f"Centroids:\n{centroids}\n")
        print(f"Assignments:\n{assignments}\n")
        
        # Plot the current state of the clusters and centroids
        plt.figure(figsize=(8, 6))
        plt.scatter(dataset[:, 0], dataset[:, 1], c=assignments, cmap='viridis', label='Data Points')
        plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label=f'Centroids (Iteration {i+1})')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.title(f'KMeans Clustering - Iteration {i+1}')
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

    # Final iteration results
    print(f"Final Centroids:\n{iterations[-1][1]}")
    print(f"Final Assignments:\n{iterations[-1][0]}")

# Run the test
if __name__ == '__main__':
    test_kmeans_iterations()
