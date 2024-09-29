# Makefile for setting up and running Flask K-means app

# Command to install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Command to run the web application on localhost:3000
run:
	@echo "Running the Flask app on http://localhost:3000"
	FLASK_APP=app.py flask run --host=0.0.0.0 --port=3000
