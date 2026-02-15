"""ETL script to merge YouTube spam ham CSV files."""

import os

import pandas as pd

from spam_ham_detector.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def merge_youtube_csvs(raw_dir: str) -> pd.DataFrame:
    """
    Load all YouTube spam ham CSV files and merge into single DataFrame.

    Parameters
    ----------
    raw_dir : str
        Directory containing the youtube-spam-collection folder

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns from the CSV files
    """
    youtube_dir = os.path.join(raw_dir, 'youtube-spam-collection')

    # Find all CSV files
    csv_files = [f for f in os.listdir(youtube_dir) if f.endswith('.csv')]
    print(f'Found {len(csv_files)} CSV files')

    all_data = []
    for filename in sorted(csv_files):
        filepath = os.path.join(youtube_dir, filename)
        print(f'- Reading {filename}')
        df = pd.read_csv(filepath, encoding='utf-8')
        all_data.append(df)

    # Merge all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f'Total rows: {len(merged_df)}')
    print('-' * 40)

    return merged_df


def main() -> None:
    """
    Merge all YouTube spam CSV files and save to interim.

    Loads all CSV files from data/raw/youtube-spam-collection/
    and saves merged file to data/interim/comments.csv.
    """
    df = merge_youtube_csvs(RAW_DATA_DIR)

    # Save to interim
    output_path = INTERIM_DATA_DIR / 'comments.csv'
    print(f'Saving to {output_path}...')
    df.to_csv(output_path, index=False)
    print('-' * 40)

    print('Done!')


if __name__ == '__main__':
    main()
