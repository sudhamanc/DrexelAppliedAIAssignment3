# Dataset Directory

This directory ships with a compressed copy of the Credit Card Fraud Detection dataset.

## Download Instructions

The archive `creditcardFraudTransactions.csv.zip` is included in the repository (~66 MB). When you run `rundemo.py`, it automatically extracts `creditcardFraudTransactions.csv` beside the archive if the CSV is not already present.

If you prefer to download it manually from Kaggle instead:
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Rename it to `creditcardFraudTransactions.csv`
4. Place it in this `data/` directory (you can remove the bundled ZIP if you like)

## Dataset Information

- **Original File**: creditcard.csv
- **Renamed to**: creditcardFraudTransactions.csv
- **Size**: ~143 MB
- **Rows**: 284,807 transactions
- **Columns**: 31 features (Time, V1-V28, Amount, Class)

After extraction, your directory structure should look like:
```
data/
├── README.md (this file)
├── creditcardFraudTransactions.csv.zip
└── creditcardFraudTransactions.csv
```

Then you can run the fraud detection models!
