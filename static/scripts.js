document.addEventListener('DOMContentLoaded', () => {
    const methodSelect = document.getElementById('method-select'); 
    const centroidsInput = document.getElementById('centroids-input'); 
    const runButton = document.getElementById('runButton'); 
    const resetButton = document.getElementById('reset-button'); 
    const generateButton = document.getElementById('generate-button'); 
    const plotContainer = document.querySelector('.plot-container'); 
    const plotMessage = document.getElementById('plot-message'); // Reference to plot-message element

    runButton.addEventListener('click', () => {
        const method = methodSelect.value; 
        const k = parseInt(centroidsInput.value); 

        // Create the data object
        const data = {
            centroids: k,
            method: method
        };

        // Send a POST request to the Flask server
        fetch('/kmeans', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json(); 
        })
        .then(result => {
            if (result.image) {
                // Only hide the message if it exists
                if (plotMessage) {
                    plotMessage.style.display = 'none'; // Hide the message
                }
                plotContainer.innerHTML = `<img src="${result.image}" alt="KMeans Clustering Plot" style="max-width: 100%; height: auto;">`;
            } else if (result.error) {
                alert(result.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing your request. Please try again.');
        });
    });

    resetButton.addEventListener('click', () => {
        centroidsInput.value = '';
        plotContainer.innerHTML = '';  
        // Only show the message if it exists
        if (plotMessage) {
            plotMessage.style.display = 'block'; // Show the message again
        }
    });

    generateButton.addEventListener('click', () => {
        alert('Dataset generation is not implemented yet.');
    });
});
