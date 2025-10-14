"""
congress.py
-----------

Author  : Michael Dickens
Created : 2025-10-11

Create a histogram of Congressional votes on bills. Originally written by
Claude, with some modifications by me.

Data downloaded from: https://voteview.com/articles/data_help_rollcalls

"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_vote_histogram(csv_path, window_pct=5, start_date=None, end_date=None, bins=50):
    """
    Create a histogram of vote percentages from Voteview data.

    Parameters:
    -----------
    csv_path : str
        Path to the HSall_rollcalls.csv file
    start_date : str, optional
        Filter votes after this date (format: 'YYYY-MM-DD')
    end_date : str, optional
        Filter votes before this date (format: 'YYYY-MM-DD')
    bins : int
        Number of bins for histogram (default: 50)
    """

    if float(int(window_pct)) == window_pct:
        # If possible, convert to int so it looks better when printing
        window_pct = int(window_pct)

    print("Loading data...")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    print("Filtering for passage votes...")
    passage_df = df[
        df["vote_question"].str.contains("On Passage", case=False, na=False)
    ].copy()

    if start_date:
        start_date = pd.to_datetime(start_date)
        passage_df = passage_df[passage_df["date"] >= start_date]
        print(f"Filtered to votes on or after {start_date}")
    else:
        start_date = datetime.datetime(1789, 1, 1)

    if end_date:
        end_date = pd.to_datetime(end_date)
        passage_df = passage_df[passage_df["date"] <= end_date]
        print(f"Filtered to votes on or before {end_date}")
    else:
        end_date = datetime.now()

    # Calculate vote percentage (yea / (yea + nay))
    passage_df["vote_pct"] = (
        passage_df["yea_count"] / (passage_df["yea_count"] + passage_df["nay_count"])
    ) * 100

    # Remove any NaN values
    passage_df = passage_df.dropna(subset=["vote_pct"])

    print(f"\nAnalyzing {len(passage_df)} passage votes")
    print(f"Date range: {passage_df['date'].min()} to {passage_df['date'].max()}")

    # Calculate some statistics
    within_window = len(
        passage_df[(passage_df["vote_pct"] >= 50 - window_pct) & (passage_df["vote_pct"] <= 50 + window_pct)]
    )

    print(
        f"\nVotes within {window_pct} percentage points of 50%: {within_window} ({within_window/len(passage_df)*100:.1f}%)"
    )

    plt.figure(figsize=(12, 6))
    plt.title(f"US Senate/House vote percentages, {start_date.year} to {end_date.year}")
    plt.hist(passage_df["vote_pct"], bins=bins, edgecolor="black", alpha=0.7)

    # Add a vertical line at 50%
    plt.axvline(x=50, color="red", linestyle="--", linewidth=2, label="50% threshold")

    # Shade the "close vote" region
    plt.axvspan(50 - window_pct, 50 + window_pct, alpha=0.2, color="yellow", label=f"Within {window_pct}% of threshold")

    # cut off plt to just show 40–70
    # plt.xlim(40, 70)

    plt.xlabel("Percentage Voting Yea", fontsize=12)
    plt.ylabel("Number of Votes", fontsize=12)

    # Create title with date range
    if start_date or end_date:
        date_str = f" ({start_date or 'beginning'} to {end_date or 'present'})"
    else:
        date_str = " (all time)"
        plt.title(
            f"Distribution of Vote Percentages on Bill Passage{date_str}",
            fontsize=14,
            fontweight="bold",
        )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt, passage_df


if __name__ == "__main__":
    csv_file = "data/Congress_votes.csv"

    plt, data = plot_vote_histogram(
        csv_file, window_pct=1, start_date="2000-01-01"
    )

    plt.savefig("vote_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()
