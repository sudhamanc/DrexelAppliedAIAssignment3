# Dataset Directory

This directory ships with the dataset required by the fraud detection demos.

## Included Files

- `creditcardFraudTransactions.csv.zip` (~66 MB) — compressed dataset bundled with the repo
- `creditcardFraudTransactions.csv` (~143 MB) — extracted CSV (created automatically when the demo runs)

## Usage

- `rundemo.py` and the individual training scripts will extract the ZIP if the CSV is missing.
- If you clean the folder to save space, restore either the ZIP or the CSV before executing the models.

## Dataset Information

- **Size**: ~143 MB (CSV)
- **Rows**: 284,807 transactions
- **Columns**: 31 features (Time, V1–V28, Amount, Class)

Expected layout after extraction:
```
data/
├── README.md (this file)
├── creditcardFraudTransactions.csv.zip
└── creditcardFraudTransactions.csv
```

Then you can run the fraud detection models!
