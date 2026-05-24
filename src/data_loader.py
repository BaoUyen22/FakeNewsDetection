from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
MULTISPACE_PATTERN = re.compile(r"\s+")


class DataLoader:
    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        random_state: int = 42,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.random_state = random_state

    def load_raw_data(
        self,
        fake_filename: str = "Fake.csv",
        true_filename: str = "True.csv",
    ) -> pd.DataFrame:
        fake_path = self.raw_dir / fake_filename
        true_path = self.raw_dir / true_filename

        if not fake_path.exists():
            raise FileNotFoundError(f"Missing file: {fake_path}")
        if not true_path.exists():
            raise FileNotFoundError(f"Missing file: {true_path}")

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        required_cols = {"title", "text"}
        for name, df in (("Fake.csv", fake_df), ("True.csv", true_df)):
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

        fake_df = fake_df.copy()
        true_df = true_df.copy()
        fake_df["label"] = 1
        true_df["label"] = 0

        return pd.concat([fake_df, true_df], ignore_index=True)

    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text):
            return ""

        value = unicodedata.normalize("NFKC", str(text))
        value = value.lower()
        value = URL_PATTERN.sub(" ", value)
        value = HTML_PATTERN.sub(" ", value)
        value = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in value)
        value = MULTISPACE_PATTERN.sub(" ", value).strip()
        return value

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        if "title" not in data.columns:
            data["title"] = ""
        if "text" not in data.columns:
            raise ValueError("Column 'text' is required.")

        data["title"] = data["title"].fillna("").astype(str)
        data["text"] = data["text"].fillna("").astype(str)
        data["raw_text"] = (data["title"] + " " + data["text"]).str.strip()

        data["cleaned_text"] = data["raw_text"].map(self.clean_text)
        data = data[data["cleaned_text"].str.len() > 0].copy()
        data = data.drop_duplicates(subset=["cleaned_text", "label"]).reset_index(drop=True)

        ordered_cols = ["cleaned_text", "label", "title", "text"]
        optional_cols = [col for col in ("subject", "date") if col in data.columns]
        return data[ordered_cols + optional_cols]

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=df["label"],
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame | None = None,
    ) -> dict[str, Path]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.processed_dir / "train.csv"
        test_path = self.processed_dir / "test.csv"
        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

        output_paths = {"train": train_path, "test": test_path}

        if full_df is not None:
            full_path = self.processed_dir / "all_cleaned.csv"
            full_df.to_csv(full_path, index=False, encoding="utf-8-sig")
            output_paths["all"] = full_path

        stats = {
            "total_rows": int((len(train_df) + len(test_df))),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "label_distribution_train": train_df["label"].value_counts().sort_index().to_dict(),
            "label_distribution_test": test_df["label"].value_counts().sort_index().to_dict(),
        }
        stats_path = self.processed_dir / "split_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        output_paths["stats"] = stats_path

        return output_paths

    def run_pipeline(self, test_size: float = 0.2) -> dict[str, Path]:
        raw_df = self.load_raw_data()
        clean_df = self.preprocess_data(raw_df)
        train_df, test_df = self.split_data(clean_df, test_size=test_size)
        return self.save_processed_data(train_df, test_df, full_df=clean_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split fake news dataset into train/test.")
    parser.add_argument("--raw-dir", default="data/raw", help="Folder containing Fake.csv and True.csv")
    parser.add_argument("--processed-dir", default="data/processed", help="Output folder for processed CSV files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    loader = DataLoader(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        random_state=args.random_state,
    )
    outputs = loader.run_pipeline(test_size=args.test_size)

    print("Processed data files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
