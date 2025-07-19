from pathlib import Path

import pandas as pd


def load_data():
    output_dir = Path("thetis_output")

    if not output_dir.exists():
        print("Output directory not found.")
        return None

    # Load any CSV files found in thetis_output
    csv_files = list(output_dir.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in thetis_output directory.")
        return None

    data = {}
    for csv_file in csv_files:
        key = csv_file.stem  # filename without extension
        try:
            data[key] = pd.read_csv(csv_file)
            print(f"Loaded {key}: {len(data[key]):,} rows")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    return data


def explore_data(data):
    if not data:
        print("No data to explore.")
        return


    for name, df in data.items():
        print(f"\n{name.upper()} Dataset:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")


def main():
    data = load_data()

    if data is None:
        return

    explore_data(data)

    return data


if __name__ == "__main__":
    data = main()
