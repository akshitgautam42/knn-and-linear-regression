#!/bin/bash

# Create main directories
mkdir -p assignments/{1..5}/{figures,data/interim} data/{external,interim} models/{knn,linear-regression,performance-measures}

# Create assignment files
touch assignments/{1..5}/a{1..5}.py

# Create README files
touch README.md assignments/{1..5}/README.md models/README.md

# Create model files
touch models/knn/knn.py models/linear-regression/linear-regression.py

# Create placeholder for external data
mkdir -p data/external/spotify-2
touch data/external/{linreg.csv,regularisation.csv,spotify.csv,spotify-2/{test.csv,train.csv,validate.csv}}

# Create interim data directories
mkdir -p data/interim/{1..5}

# Set up git (optional)
git init
echo "data/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore

echo "Project structure created successfully!"