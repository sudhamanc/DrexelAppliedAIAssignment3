# GitHub Repository Setup Instructions

## Step 1: Create the Repository on GitHub

1. Go to https://github.com and log in with your account (sudhaman@gmail.com)

2. Click the "+" icon in the top-right corner and select "New repository"

3. Fill in the repository details:
   - **Repository name**: `DrexelAppliedAI-Assignment3`
   - **Description**: "Credit Card Fraud Detection using XGBoost and Neural Networks - Drexel Applied AI Fall 2025"
   - **Visibility**: Choose "Public" or "Private" (your preference)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)

4. Click "Create repository"

## Step 2: Copy the Repository URL

After creating the repository, GitHub will show you a page with setup instructions.
You'll see a URL like one of these:

- HTTPS: `https://github.com/YOUR_USERNAME/DrexelAppliedAI-Assignment3.git`
- SSH: `git@github.com:YOUR_USERNAME/DrexelAppliedAI-Assignment3.git`

Copy the HTTPS URL (easier for first-time setup).

## Step 3: Run These Commands

Open your terminal and run the following commands:

```bash
cd "/Users/sudhamanc/My Drive/VisionDocuments/Drexel/Courses/Fall2025/AppliedAI/Assignments/Assignment3"

# Add the remote repository (replace <REPO_URL> with the URL you copied)
git remote add origin <REPO_URL>

# Push your code to GitHub
git push -u origin main
```

### Example:
If your GitHub username is "sudhamanc", the command would be:
```bash
git remote add origin https://github.com/sudhamanc/DrexelAppliedAI-Assignment3.git
git push -u origin main
```

## Step 4: Authenticate (if prompted)

If this is your first time pushing to GitHub from this computer, you may be prompted to authenticate:

- **Option 1 (Recommended)**: Use a Personal Access Token
  1. Go to https://github.com/settings/tokens
  2. Click "Generate new token (classic)"
  3. Give it a name (e.g., "MacBook Air")
  4. Select scopes: at minimum check "repo"
  5. Click "Generate token"
  6. Copy the token (you won't see it again!)
  7. When prompted for password, paste the token

- **Option 2**: Use SSH keys (more permanent setup)
  - Follow: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## Verification

After pushing, visit your repository URL in a browser:
`https://github.com/YOUR_USERNAME/DrexelAppliedAI-Assignment3`

You should see:
- ✓ README.md displayed on the main page
- ✓ All your Python files
- ✓ fraud_results/ folder with visualizations
- ✓ data/ folder with the dataset
- ✓ requirements.txt
- ✓ rundemo.py

## Repository Contents

Your repository now contains:
```
DrexelAppliedAI-Assignment3/
├── .gitignore
├── README.md
├── requirements.txt
├── rundemo.py
├── fraud_binary_xgboost.py
├── fraud_multiclass_mlp.py
├── data/
│   └── creditcardFraudTransactions.csv
└── fraud_results/
    ├── fraud_binary_confusion_matrix.png
    ├── fraud_binary_metrics.png
    ├── fraud_binary_roc_curve.png
    ├── multiclass_confusion_matrix.png
    ├── multiclass_metrics.png
    └── multiclass_per_class.png
```

## Troubleshooting

**Problem**: "remote origin already exists"
**Solution**: 
```bash
git remote remove origin
git remote add origin <REPO_URL>
```

**Problem**: "failed to push"
**Solution**: Make sure you copied the correct repository URL and have proper authentication

**Problem**: "Permission denied"
**Solution**: Check your GitHub credentials or use a Personal Access Token

---

After completing these steps, your code will be live on GitHub and you can share the repository URL with others!
