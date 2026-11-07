"""
Live Daily Evaluation & Model Drift Tracker.
Automates daily prediction logging, out-of-sample ground-truth evaluation,
and drift monitoring to verify whether the model is holding up over time.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_loader import VietlottDataLoader
from src.strategies import (
    ColdOverdueStrategy,
    GameTheoryAntiCrowdStrategy,
    HotFrequencyStrategy,
    HybridStrategy,
    MarkovTransitionStrategy,
    MLRankingStrategy,
    RandomBaselineStrategy,
    RecursiveAdaptiveStrategy,
    XGBoostRankingStrategy,
)
from src.wheeling import CombinatorialWheeler


class DailyModelTracker:
    def __init__(self, lottery_code: str = "power655", tracker_dir: str = "data"):
        self.lottery_code = lottery_code.lower()
        self.tracker_dir = Path(tracker_dir)
        self.tracker_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_file = self.tracker_dir / f"{self.lottery_code}_daily_tracker.csv"
        self.pending_file = self.tracker_dir / f"{self.lottery_code}_pending_predictions.json"

    def record_pending_prediction(self, target_draw_id: str, predictions: Dict[str, List[List[int]]]) -> None:
        """Log predictions made before the draw takes place."""
        clean_predictions = {}
        for s_name, tickets in predictions.items():
            clean_predictions[s_name] = [[int(x) for x in t] for t in tickets]

        payload = {
            "target_draw_id": target_draw_id,
            "created_at": datetime.now().isoformat(),
            "predictions": clean_predictions
        }
        with open(self.pending_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=int)

    def evaluate_latest_draw(self) -> Dict[str, Any]:
        """
        Fetch latest draw from data source.
        If a pending prediction exists for that draw, compare against ground truth,
        record the results to tracker CSV, and generate predictions for the next draw.
        """
        loader = VietlottDataLoader(self.lottery_code)
        df = loader.load_clean_data(force_refresh=True)

        latest_draw = df.iloc[-1]
        latest_id = str(latest_draw["id"])
        latest_date = str(latest_draw["date"].date())
        actual_main = set(latest_draw["main_numbers"])
        actual_bonus = latest_draw["bonus_number"]

        evaluation_records = []

        if self.pending_file.exists():
            try:
                with open(self.pending_file, "r", encoding="utf-8") as f:
                    pending_data = json.load(f)
            except Exception:
                pending_data = {}

            pending_id = str(pending_data.get("target_draw_id", ""))
            predictions = pending_data.get("predictions", {})

            # Compare if pending prediction matches latest draw
            if pending_id == latest_id:
                for strat_name, tickets in predictions.items():
                    for ticket_idx, ticket in enumerate(tickets, 1):
                        matches = sorted(list(set(ticket).intersection(actual_main)))
                        hit_count = len(matches)

                        # Calculate prize
                        prize = 0
                        if self.lottery_code == "power655":
                            if hit_count == 6:
                                prize = 30000000000
                            elif hit_count == 5 and (actual_bonus and actual_bonus in ticket):
                                prize = 3000000000
                            elif hit_count == 5:
                                prize = 40000000
                            elif hit_count == 4:
                                prize = 500000
                            elif hit_count == 3:
                                prize = 50000
                        else:  # mega645
                            if hit_count == 6:
                                prize = 12000000000
                            elif hit_count == 5:
                                prize = 10000000
                            elif hit_count == 4:
                                prize = 300000
                            elif hit_count == 3:
                                prize = 30000

                        evaluation_records.append({
                            "draw_id": latest_id,
                            "draw_date": latest_date,
                            "strategy": strat_name,
                            "ticket_index": ticket_idx,
                            "ticket": "-".join(f"{x:02d}" for x in ticket),
                            "actual_draw": "-".join(f"{x:02d}" for x in sorted(list(actual_main))),
                            "hit_count": hit_count,
                            "matching_numbers": ",".join(map(str, matches)),
                            "prize_vnd": prize,
                            "cost_vnd": 10000,
                            "profit_vnd": prize - 10000,
                            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                # Append to tracker CSV
                eval_df = pd.DataFrame(evaluation_records)
                if self.tracker_file.exists():
                    existing_df = pd.read_csv(self.tracker_file)
                    eval_df = pd.concat([existing_df, eval_df]).drop_duplicates(
                        subset=["draw_id", "strategy", "ticket_index"]
                    ).reset_index(drop=True)
                eval_df.to_csv(self.tracker_file, index=False)

        # Generate fresh predictions for the upcoming draw
        next_id = f"{int(latest_id) + 1:05d}"
        total_numbers = 55 if self.lottery_code == "power655" else 45

        rec_strat = RecursiveAdaptiveStrategy(total_numbers, 6, alpha=0.04)
        strategies = {
            "Recursive Adaptive (Hedge+Wheel)": rec_strat,
            "Game Theory (Anti-Crowd)": GameTheoryAntiCrowdStrategy(total_numbers, 6),
            "Hybrid Balanced": HybridStrategy(total_numbers, 6, 50),
            "XGBoost Ranker": XGBoostRankingStrategy(total_numbers, 6, 200),
            "Markov Transition": MarkovTransitionStrategy(total_numbers, 6),
            "Cold Overdue": ColdOverdueStrategy(total_numbers, 6),
            "Random Baseline": RandomBaselineStrategy(total_numbers, 6)
        }

        fresh_preds = {}
        for s_name, strat in strategies.items():
            raw_tickets = strat.generate_tickets(df, num_tickets=4)
            fresh_preds[s_name] = [[int(x) for x in t] for t in raw_tickets]

        # Combine Recursive Adaptive Top-10 Pool with Probability-Weighted Wheeling System (3-if-4)
        wheeler = CombinatorialWheeler(pick_count=6)
        top_pool, pool_weights = rec_strat.get_top_candidates(df, top_k=10)
        wheel_res = wheeler.generate_abbreviated_wheel(
            top_pool,
            target_guarantee=3,
            condition_drawn=4,
            max_tickets=4,
            number_weights=pool_weights
        )
        fresh_preds["Wheeling 3-if-4 (Recursive Top-10)"] = [[int(x) for x in t] for t in wheel_res["tickets"]]

        self.record_pending_prediction(next_id, fresh_preds)

        return {
            "latest_draw_id": latest_id,
            "latest_draw_date": latest_date,
            "actual_numbers": sorted(list(actual_main)),
            "new_prediction_for_draw": next_id,
            "evaluation_records_count": len(evaluation_records),
            "fresh_predictions": fresh_preds
        }

    def get_performance_summary(self) -> pd.DataFrame:
        """Compute live drift and cumulative accuracy metrics from tracker CSV."""
        if not self.tracker_file.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.tracker_file)
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby("strategy").agg(
            total_tickets=("ticket_index", "count"),
            avg_hits=("hit_count", "mean"),
            win_count=("prize_vnd", lambda s: (s > 0).sum()),
            total_spend=("cost_vnd", "sum"),
            total_prize=("prize_vnd", "sum"),
            net_profit=("profit_vnd", "sum")
        ).reset_index()

        summary["avg_hits"] = summary["avg_hits"].round(3)
        summary["win_rate_pct"] = (summary["win_count"] / summary["total_tickets"] * 100).round(2)
        summary["roi_pct"] = (summary["net_profit"] / summary["total_spend"] * 100).round(2)

        return summary
