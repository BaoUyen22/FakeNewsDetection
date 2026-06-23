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

    def analyze_subject_distribution(
        self,
        fake_filename: str = "Fake.csv",
        true_filename: str = "True.csv",
    ) -> dict:
        """
        Thống kê phân phối cột 'subject' từ Fake.csv và True.csv
        """
        fake_path = self.raw_dir / fake_filename
        true_path = self.raw_dir / true_filename

        if not fake_path.exists() or not true_path.exists():
            return {"error": "Missing raw data files"}

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        result = {}
        
        if "subject" in fake_df.columns:
            fake_subjects = fake_df["subject"].value_counts().to_dict()
            result["Fake.csv"] = {
                "total": len(fake_df),
                "subject_counts": fake_subjects,
            }
        else:
            result["Fake.csv"] = {"error": "Column 'subject' not found"}

        if "subject" in true_df.columns:
            true_subjects = true_df["subject"].value_counts().to_dict()
            result["True.csv"] = {
                "total": len(true_df),
                "subject_counts": true_subjects,
            }
        else:
            result["True.csv"] = {"error": "Column 'subject' not found"}

        return result

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
        
        # Xóa cột subject và date để tránh data leakage
        fake_df = fake_df.drop(columns=["subject", "date"], errors="ignore")
        true_df = true_df.drop(columns=["subject", "date"], errors="ignore")
        
        # Bỏ Reuters tag trong text (thường có trong True.csv)
        fake_df["text"] = fake_df["text"].str.replace(r"^.*?\(Reuters\)\s*[-–]\s*", "", regex=True)
        true_df["text"] = true_df["text"].str.replace(r"^.*?\(Reuters\)\s*[-–]\s*", "", regex=True)
        
        fake_df["label"] = 1
        true_df["label"] = 0
        
        fake_df = fake_df[["title", "text", "label"]]
        true_df = true_df[["title", "text", "label"]]

        return pd.concat([fake_df, true_df], ignore_index=True)

    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text):
            return ""

        value = unicodedata.normalize("NFKC", str(text))
        value = URL_PATTERN.sub(" ", value)
        value = HTML_PATTERN.sub(" ", value)
        value = re.sub(r"@[\w_]+", " ", value)
        value = re.sub(r"#[\w_]+", " ", value)
        value = re.sub(r"(.)\1{2,}", r"\1\1", value)
        value = MULTISPACE_PATTERN.sub(" ", value).strip()
        return value

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng cleaning cho từng split riêng biệt (sau khi đã split)
        Clean riêng title và text để có thể train riêng hoặc kết hợp
        """
        data = df.copy()

        if "title" not in data.columns:
            data["title"] = ""
        if "text" not in data.columns:
            raise ValueError("Column 'text' is required.")

        # Fill NaN và convert to string
        data["title"] = data["title"].fillna("").astype(str)
        data["text"] = data["text"].fillna("").astype(str)
        
        # Clean riêng từng cột
        data["title"] = data["title"].map(self.clean_text)
        data["text"] = data["text"].map(self.clean_text)
        
        # Fill NaN sau khi clean (trong trường hợp clean_text trả về NaN)
        data["title"] = data["title"].fillna("")
        data["text"] = data["text"].fillna("")
        
        # Xóa các dòng có CẢ title VÀ text đều rỗng
        data = data[
            (data["title"].str.strip().str.len() > 0) | 
            (data["text"].str.strip().str.len() > 0)
        ].copy()
        
        # Nếu text rỗng nhưng title có → copy title sang text
        mask_empty_text = data["text"].str.strip().str.len() == 0
        data.loc[mask_empty_text, "text"] = data.loc[mask_empty_text, "title"]
        
        # Xóa trùng lặp dựa trên cả title + text + label
        data = data.drop_duplicates(subset=["title", "text", "label"]).reset_index(drop=True)

        # Giữ 3 cột: title (cleaned), text (cleaned), label
        return data[["title", "text", "label"]]

    def split_data(
        self, df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data TRƯỚC KHI clean (chỉ split raw data)
        train (80%), val (10%), test (10%)
        """
        # Tính test_size cho lần split đầu tiên
        test_val_size = val_size + test_size  # 0.2
        
        # Split: train (80%) vs temp (20%)
        train_df, temp_df = train_test_split(
            df,
            test_size=test_val_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=df["label"],
        )
        
        # Split temp thành val (10%) và test (10%)
        # val_size / (val_size + test_size) = 0.5 để chia đôi temp
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=self.random_state,
            shuffle=True,
            stratify=temp_df["label"],
        )
        
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, Path]:
        """
        Lưu train/val/test đã clean (KHÔNG lưu all_cleaned.csv để tránh leak)
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.processed_dir / "train.csv"
        val_path = self.processed_dir / "val.csv"
        test_path = self.processed_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

        output_paths = {"train": train_path, "val": val_path, "test": test_path}

        stats = {
            "total_rows": int((len(train_df) + len(val_df) + len(test_df))),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "label_distribution_train": train_df["label"].value_counts().sort_index().to_dict(),
            "label_distribution_val": val_df["label"].value_counts().sort_index().to_dict(),
            "label_distribution_test": test_df["label"].value_counts().sort_index().to_dict(),
        }
        stats_path = self.processed_dir / "split_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        output_paths["stats"] = stats_path

        return output_paths

    def run_pipeline(self, val_size: float = 0.1, test_size: float = 0.1) -> dict[str, Path]:
        """
        Pipeline đúng: Load → Split → Clean từng tập riêng → Save
        """
        # Bước 1: Load raw data 
        raw_df = self.load_raw_data()
        
        # Bước 2: Split trước 
        train_raw, val_raw, test_raw = self.split_data(raw_df, val_size=val_size, test_size=test_size)
        
        # Bước 3: Clean 
        train_clean = self.preprocess_data(train_raw)
        val_clean = self.preprocess_data(val_raw)
        test_clean = self.preprocess_data(test_raw)
        
        # Bước 4: Save 
        return self.save_processed_data(train_clean, val_clean, test_clean)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split fake news dataset into train/val/test.")
    parser.add_argument("--raw-dir", default="data/raw", help="Folder containing Fake.csv and True.csv")
    parser.add_argument("--processed-dir", default="data/processed", help="Output folder for processed CSV files")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--analyze-subject", action="store_true", help="Show subject distribution before processing")
    args = parser.parse_args()

    loader = DataLoader(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        random_state=args.random_state,
    )
    #python src/data_loader.py --analyze-subject
    # Thống kê subject nếu được yêu cầu
    if args.analyze_subject:
        print("\n=== Subject Distribution Analysis ===")
        stats = loader.analyze_subject_distribution()
        for filename, data in stats.items():
            print(f"\n{filename}:")
            if "error" in data:
                print(f"  {data['error']}")
            else:
                print(f"  Total rows: {data['total']}")
                print(f"  Subject counts:")
                for subject, count in sorted(data["subject_counts"].items(), key=lambda x: x[1], reverse=True):
                    print(f"    {subject}: {count}")
        print("\n" + "="*40 + "\n")
    
    outputs = loader.run_pipeline(val_size=args.val_size, test_size=args.test_size)

    print("Processed data files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
