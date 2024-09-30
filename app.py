from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from kmeans import KMeans  # Ensure KMeans class is properly implemented

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the main HTML page

@app.route('/kmeans', methods=['POST'])
def kmeans():
    # Extract the number of centroids and the method from the request
    k = int(request.form['centroids'])
    method = request.form['method']

    # Validate k (make sure it's a positive integer)
    if k <= 0:
        return jsonify({'error': 'Number of centroids must be a positive integer.'}), 400

    # Generate a random dataset
    dataset = np.random.rand(100, 2) * 20 - 10  # Random dataset within range [-10, 10]

    # Create KMeans instance anew each time
    kmeans = KMeans(dataset, k)

    # Run the clustering algorithm
    iterations = kmeans.cluster(method)  # Run the KMeans algorithm

    # Final centroids and assignments after convergence
    final_assignments = iterations[-1][0]  # Cluster assignments
    final_centroids = iterations[-1][1]  # Centroids

    # Create a plot
    img = io.BytesIO()
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
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Return the image data as JSON response
    return jsonify({'image': f"data:image/png;base64,{plot_url}"})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)  # Run the Flask app
