#!/bin/bash
# This script builds Sphinx documentation and pushes it to the gh-pages branch

# Exit on error
set -e

# Load personal access token
# Load personal access token and remove any carriage return characters
#GIT_PERSONAL_ACCESS_TOKEN=$(grep -o 'GIT_PERSONAL_ACCESS_TOKEN=.*' .env | cut -f 2- -d '=' | tr -d '\r')
source .env

# Switch to gh-pages branch
git checkout gh-pages

# Copy the generated HTML files to the root directory
cp -r docs/build/html/* .

# Add and commit the changes
git add .

# Unstage the .env file
git reset .env

git commit -m "Update documentation"

# Use the personal access token to push changes
# * NOTE(Miguel): Pushing directly is becoming a bit of a pain so for now this script will just prepare the branch so that we can just push with our normal git config

#git push https://${GIT_PERSONAL_ACCESS_TOKEN}@github.com/vector-raw-materials/vector-geology.git gh-pages

# Switch back to the original branch
#git checkout -
#
#echo "Documentation updated and pushed to gh-pages."
