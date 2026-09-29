import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "training_data_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

df = pd.read_csv(SUMMARY_PATH)

df["has_division"] = (df["divisions"] > 0).astype(int)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["has_division"]
)

train_df = train_df.drop(columns=["has_division"])
val_df = val_df.drop(columns=["has_division"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_DIR / "train_samples.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val_samples.csv", index=False)

print("=" * 60)
print("TRAIN / VALIDATION SPLIT")
print("=" * 60)
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

print()
print("Training divisions:")
print(f"With divisions: {(train_df['divisions'] > 0).sum()}")
print(f"Without divisions: {(train_df['divisions'] == 0).sum()}")

print()
print("Validation divisions:")
print(f"With divisions: {(val_df['divisions'] > 0).sum()}")
print(f"Without divisions: {(val_df['divisions'] == 0).sum()}")

print()
print(f"Saved: {OUTPUT_DIR / 'train_samples.csv'}")
print(f"Saved: {OUTPUT_DIR / 'val_samples.csv'}")