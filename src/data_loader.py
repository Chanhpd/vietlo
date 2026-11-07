"""
Data Loader & Preprocessing Module for Vietlott Data.
Handles automatic downloading, caching, anomaly cleaning, and feature enrichment.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOW_MAPPING = {
    0: "Thứ Hai",
    1: "Thứ Ba",
    2: "Thứ Tư",
    3: "Thứ Năm",
    4: "Thứ Sáu",
    5: "Thứ Bảy",
    6: "Chủ Nhật"
}


class VietlottDataLoader:
    def __init__(
        self,
        lottery_code: str = "power655",
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        url: Optional[str] = None
    ):
        self.lottery_code = lottery_code.lower()
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Default URLs
        default_urls = {
            "power655": "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power655.jsonl",
            "mega645": "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl",
        }
        self.url = url or default_urls.get(self.lottery_code)
        if not self.url:
            raise ValueError(f"Unsupported lottery code: {self.lottery_code}. Must be 'power655' or 'mega645'")

        raw_filename = "power655.jsonl" if self.lottery_code == "power655" else "power645.jsonl"
        self.raw_file = self.raw_dir / raw_filename
        self.processed_file = self.processed_dir / f"{self.lottery_code}_clean.csv"

    def fetch_data(self, force_download: bool = False) -> Path:
        """Download latest data from GitHub repository with caching."""
        if not self.raw_file.exists() or force_download:
            logger.info(f"Downloading {self.lottery_code} data from {self.url}...")
            try:
                response = requests.get(self.url, timeout=30)
                response.raise_for_status()
                with open(self.raw_file, "w", encoding="utf-8") as f:
                    f.write(response.text)
                logger.info(f"Successfully downloaded to {self.raw_file}")
            except Exception as e:
                if self.raw_file.exists():
                    logger.warning(f"Download failed ({e}), falling back to cached file: {self.raw_file}")
                else:
                    raise RuntimeError(f"Failed to fetch data and no cached file found: {e}")
        else:
            logger.info(f"Using cached raw file: {self.raw_file}")
        return self.raw_file

    def load_clean_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load raw data, clean anomalies, compute temporal and structural features.
        Returns cleaned DataFrame sorted chronologically.
        """
        raw_path = self.fetch_data(force_download=force_refresh)
        records = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(data)
                except json.JSONDecodeError as err:
                    logger.warning(f"Skipping malformed JSON line {line_no}: {err}")

        if not records:
            raise ValueError(f"No records found in {raw_path}")

        raw_df = pd.DataFrame(records)
        logger.info(f"Loaded {len(raw_df)} raw records for {self.lottery_code}")

        # Standardize date and sort
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df = raw_df.sort_values("date").drop_duplicates(subset=["id"]).reset_index(drop=True)

        # Parse results: separate main numbers from bonus number
        main_numbers_list = []
        bonus_numbers_list = []

        is_power655 = (self.lottery_code == "power655")

        for idx, row in raw_df.iterrows():
            res = row.get("result", [])
            if not isinstance(res, list) or len(res) < 6:
                logger.warning(f"Row {idx} (ID: {row.get('id')}) has invalid result: {res}. Skipping.")
                main_numbers_list.append(None)
                bonus_numbers_list.append(None)
                continue

            # First 6 numbers are always the main combination
            main_nums = sorted([int(x) for x in res[:6]])
            main_numbers_list.append(main_nums)

            if is_power655:
                # 7th number is the bonus / special number (if available)
                if len(res) >= 7:
                    bonus_numbers_list.append(int(res[6]))
                else:
                    logger.warning(f"Power 6/55 Draw ID {row.get('id')} lacks 7th bonus number. Setting to None.")
                    bonus_numbers_list.append(None)
            else:
                bonus_numbers_list.append(None)

        raw_df["main_numbers"] = main_numbers_list
        raw_df["bonus_number"] = bonus_numbers_list

        # Drop invalid rows
        df = raw_df.dropna(subset=["main_numbers"]).copy()

        # Unpack individual numbers num_1 to num_6
        for i in range(6):
            df[f"num_{i+1}"] = df["main_numbers"].apply(lambda nums: nums[i])

        # Date & Temporal Features
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.day_of_week.map(DOW_MAPPING)
        df["day_of_week_num"] = df["date"].dt.day_of_week

        # Draw intervals (days since last draw)
        df["days_since_last_draw"] = df["date"].diff().dt.days.fillna(0).astype(int)

        # Structural & Statistical lottery features
        df["sum_numbers"] = df["main_numbers"].apply(sum)
        df["odd_count"] = df["main_numbers"].apply(lambda nums: sum(1 for n in nums if n % 2 != 0))
        df["even_count"] = 6 - df["odd_count"]
        df["range_span"] = df["num_6"] - df["num_1"]
        df["mean_number"] = df["main_numbers"].apply(lambda nums: sum(nums) / 6.0)

        # Consecutive numbers count in draw (e.g. [14, 15] has 1 consecutive pair)
        df["consecutive_pairs_count"] = df["main_numbers"].apply(
            lambda nums: sum(1 for i in range(len(nums) - 1) if nums[i+1] == nums[i] + 1)
        )

        # Save processed data only when refreshing or if file does not exist yet
        # (prevents dirtying git working tree on read-only cloud deployments like Streamlit Cloud)
        if force_refresh or not self.processed_file.exists():
            export_df = df.copy()
            export_df["main_numbers_str"] = export_df["main_numbers"].apply(lambda nums: ",".join(map(str, nums)))
            export_df.to_csv(self.processed_file, index=False)
            logger.info(f"Saved processed dataset ({len(df)} draws) to {self.processed_file}")

        return df


if __name__ == "__main__":
    for code in ["power655", "mega645"]:
        loader = VietlottDataLoader(lottery_code=code)
        df = loader.load_clean_data(force_refresh=False)
        print(f"Loaded {code}: shape {df.shape}")
        print(df[["date", "id", "num_1", "num_2", "num_3", "num_4", "num_5", "num_6", "bonus_number", "day_of_week", "sum_numbers"]].tail(3))
