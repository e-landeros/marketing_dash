#!/bin/bash
# Configuration
SPACE_REPO="https://huggingface.co/spaces/elanderos/marketing-analytics-dashboard"
DEPLOY_DIR="deploy_space"
echo "🚀 Starting Deployment to Hugging Face Space..."
# 1. Clean previous deployment directory if exists
if [ -d "$DEPLOY_DIR" ]; then
    echo "Cleaning up previous deployment directory..."
    rm -rf "$DEPLOY_DIR"
fi
# 2. Clone the repository
echo "📥 Cloning the Space repository..."
git clone "$SPACE_REPO" "$DEPLOY_DIR"
if [ ! -d "$DEPLOY_DIR" ]; then
    echo "❌ Failed to clone repository. Check your internet connection or URL."
    exit 1
fi
# 3. Copy Application Files
echo "📂 Copying files..."
cp app.py "$DEPLOY_DIR/"
cp requirements.txt "$DEPLOY_DIR/"
cp Dockerfile "$DEPLOY_DIR/"
# Create directory structure for data if needed
mkdir -p "$DEPLOY_DIR/data/raw"
cp data/raw/synthetic_leads.csv "$DEPLOY_DIR/data/raw/"
# 4. Commit and Push
echo "⬆️ Pushing to Hugging Face..."
cd "$DEPLOY_DIR" || exit
# Check if there are changes
if [ -z "$(git status --porcelain)" ]; then
    echo "⚠️ No changes to deploy."
else
    # Initialize LFS
    echo "📦 Configuring Git LFS for large files..."
    git lfs install
    git lfs track "data/raw/synthetic_leads.csv"
    git add .gitattributes
    git add .
    git commit -m "Deploy update via script $(date)"
    git push
    echo "✅ Successfully deployed!"
fi
# 5. Cleanup
cd ..
echo "🧹 Cleaning up..."
rm -rf "$DEPLOY_DIR"
echo "🎉 Done! Check your space at: $SPACE_REPO"
