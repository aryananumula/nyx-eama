from pathlib import Path

import pandas as pd


def load_data():
    output_dir = Path("output")

    if not output_dir.exists():
        print("Output directory not found.")
        return None, None, None

    points = pd.read_csv(output_dir / 'points.csv')
    analogs = pd.read_csv(output_dir / 'analogs.csv')
    metadata = pd.read_csv(output_dir / 'metadata.csv')

    print(f"Loaded {len(points):,} point measurements")
    print(f"Loaded {len(analogs):,} analog measurements")
    print(f"Loaded {len(metadata)} files metadata")

    return points, analogs, metadata

def explore_data(points, analogs, metadata):
    # Show trials and conditions
    print(f"Trials: {sorted(metadata['trial'].unique())}")
    print(f"Conditions: {sorted(metadata['condition'].unique())}")

    # Show some marker types
    racket_markers = points[points['point_label'].str.contains('racket', case=False, na=False)]['point_label'].unique()
    print(f"Racket markers: {list(racket_markers[:5])}")

    # Show sensor channels
    force_channels = analogs[analogs['channel_label'].str.startswith('F')]['channel_label'].unique()
    print(f"Force channels: {list(force_channels)}")

def main():
    points, analogs, metadata = load_data()

    if points is None:
        return

    explore_data(points, analogs, metadata)

    # Testing getting player tp1 forehand racket data
    # racket_data = points[
    #     (points['trial'] == 'tp1') &
    #     (points['condition'] == 'fh') &
    #     (points['point_label'].str.contains('racket', case=False, na=False))
    # ]
    # print(f"Player tp1 forehand racket data: {len(racket_data):,} measurements")

    # Testing getting force data for a session
    # force_data = analogs[
    #     (analogs['trial'] == 'tp1') &
    #     (analogs['session'] == 's1') &
    #     (analogs['channel_label'].str.startswith('F'))
    # ]
    # print(f"Force data for tp1 session s1: {len(force_data):,} measurements")

    # Testing getting count shots by condition
    # shots_by_condition = metadata['condition'].value_counts()
    # print(f"\nShots by condition:")
    # for condition, count in shots_by_condition.items():
    #     print(f"  {condition}: {count} sessions")

    return points, analogs, metadata

if __name__ == "__main__":
    points, analogs, metadata = main()
    if points is not None:
        print(f"\nTotal points data: {len(points):,} rows")
    if analogs is not None:
        print(f"Total analogs data: {len(analogs):,} rows")
    if metadata is not None:
        print(f"Total metadata: {len(metadata)} rows")
