#!/bin/bash

# --- This script converts your local submodule setup into a single monorepo ---

MAIN_REPO=/home3/rudrapra001/FYP-Research

cd "$MAIN_REPO" || exit 1

echo ">>> Backing up current .git and .gitmodules..."
# Create a backup of your submodule setup just in case you want to go back
mv .git .git_submodule_backup
mv .gitmodules .gitmodules_submodule_backup
echo ">>> Backup complete."
echo ""

echo ">>> De-initializing all submodules..."
# This command removes the submodule entries from the index
git submodule deinit --all -f
echo ""

echo ">>> Removing nested .git directories from sub-projects..."
# Find and delete all .git folders within the subdirectories
find . -mindepth 2 -name ".git" -type d -exec rm -rf {} +
echo ">>> Cleanup of sub-project repos complete."
echo ""

echo ">>> Re-initializing the main repository for a monorepo structure..."
# We use the backup to restore the main repo's history
mv .git_submodule_backup .git
# We no longer need the submodule config file
rm -f .gitmodules_submodule_backup

echo ">>> Staging all project files for the new monorepo commit..."
# Add all the sub-project files (which are now just regular files) to the main repo
git add .

echo ">>> Committing the unified project..."
git commit -m "Feat: Convert project to monorepo structure"

echo ""
echo "✅ Conversion to monorepo complete."
echo "You can now push this single repository to GitHub."
echo "Run 'git push' to update your remote."