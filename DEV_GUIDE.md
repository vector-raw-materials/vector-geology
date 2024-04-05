### To push the docs in a fancy way

- `git worktree add ../VisualBayesicDocs gh-pages` This will create a new folder called `VisualBayesicDocs` in the parent directory of the current repo. This folder will contain the `gh-pages` branch of the repo. This is where the docs will be pushed to.
- `cp -r -force ./docs/build/html/* ../VisualBayesicDocs/` This will copy the contents of the `docs/build/html` folder to the `VisualBayesicDocs` folder.
- `cd ../VisualBayesicDocs` This will change the current directory to the `VisualBayesicDocs` folder.
- `git add .` This will add all the files in the current directory to the staging area.
- `git commit -m "Update docs"` This will commit the changes to the `gh-pages` branch.
- `git push origin gh-pages` This will push the changes to the `gh-pages` branch of the repo.
- `cd ../VisualBayesic` This will change the current directory back to the `VisualBayesic` folder.