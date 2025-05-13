#!/bin/bash

module load git

# Ensure you're on the main branch
git checkout main

# Fetch latest changes from upstream
git fetch upstream

# Merge upstream/main into your main branch
git rebase upstream/main

# Push updated main branch to your fork (origin)
git push origin main

echo "✅ Fork is now up to date with upstream."
