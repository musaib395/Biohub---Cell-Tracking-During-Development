import numpy as np
import pandas as pd

CSV_PATH = "data/processed/lineage_tracks.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 65)
print("       CELL TRACKING PERFORMANCE EVALUATION (CTC METRICS)")
print("=" * 65)


# 1. Detection Evaluation (DET Score Metrics)
def evaluate_detection_metrics(dataframe):
    total_frames = dataframe["Timepoint"].nunique()
    total_detections = len(dataframe)

    # Detections per timepoint
    tp_counts = dataframe.groupby("Timepoint").size()

    metrics = {
        "Total Timepoints": total_frames,
        "Total Detected Cells": total_detections,
        "Mean Cells / Frame": round(tp_counts.mean(), 2),
        "Min Cells / Frame": tp_counts.min(),
        "Max Cells / Frame": tp_counts.max(),
    }
    return metrics


# 2. Tracking & Lineage Continuity (TRA Score Metrics)
def evaluate_tracking_metrics(dataframe):
    tracks = dataframe["TrackID"].unique()
    num_tracks = len(tracks)

    division_events = dataframe[dataframe["ParentID"] != -1]
    num_divisions = len(division_events) // 2  # 2 daughters per event

    # Track length statistics
    track_lengths = dataframe.groupby("TrackID").size()

    metrics = {
        "Total Unique Track IDs": num_tracks,
        "Division Events Detected": num_divisions,
        "Mean Track Duration (frames)": round(track_lengths.mean(), 2),
        "Max Track Duration (frames)": track_lengths.max(),
        "Min Track Duration (frames)": track_lengths.min(),
    }
    return metrics


det_stats = evaluate_detection_metrics(df)
tra_stats = evaluate_tracking_metrics(df)

print("\n[Detection (DET) Metrics]")
for k, v in det_stats.items():
    print(f"  ├── {k:<30}: {v}")

print("\n[Tracking & Lineage (TRA) Metrics]")
for k, v in tra_stats.items():
    print(f"  ├── {k:<30}: {v}")

print("\n" + "=" * 65)
print("EVALUATION COMPLETE!")
print("=" * 65)