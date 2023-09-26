#!/bin/bash
# This script builds Sphinx documentation and pushes it to the gh-pages branch

# Exit on error
set -e

# Load personal access token
# Load personal access token and remove any carriage return characters
GIT_PERSONAL_ACCESS_TOKEN=$(grep -o 'GIT_PERSONAL_ACCESS_TOKEN=.*' .env | cut -f 2- -d '=' | tr -d '\r')

# Stash the .env file to ignore it while switching branches
git stash push .env

# Switch to gh-pages branch
git checkout gh-pages

# Copy the generated HTML files to the root directory
cp -r docs/build/html/* .

# Add and commit the changes
git add .
git commit -m "Update documentation"

# Use the personal access token to push changes
git push https://${GIT_PERSONAL_ACCESS_TOKEN}@github.com/vector-raw-materials/vector-geology.git gh-pages

# Switch back to the original branch
git checkout -

# Pop the stash to get back the .env file
git stash pop

echo "Documentation updated and pushed to gh-pages."