"""
Fraud Detection Project: Data Loader
Loads the Kaggle "Credit Card Fraud Detection" CSV into the PostgreSQL
`transactions` table created by schema.sql.

Usage:
    python load_data.py /path/to/creditcard.csv

Before running:
    1. Make sure PostgreSQL is running.
    2. Create the database:  createdb fraud_detection   (or via psql: CREATE DATABASE fraud_detection;)
    3. Run schema.sql against it:  psql -d fraud_detection -f schema.sql
    4. Update DB_CONFIG below with your actual credentials.
"""

import sys
import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG


def get_engine():
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


def load_csv_to_db(csv_path: str):
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Rename columns to match the lowercase schema (Kaggle CSV uses Time, Amount, V1..V28, Class)
    rename_map = {"Time": "tx_time", "Amount": "amount", "Class": "class"}
    rename_map.update({f"V{i}": f"v{i}" for i in range(1, 29)})
    df = df.rename(columns=rename_map)

    print(f"Loaded {len(df):,} rows. Class balance:")
    print(df["class"].value_counts(normalize=True))

    engine = get_engine()

    print("Writing to PostgreSQL table 'transactions' ...")
    df.to_sql(
        "transactions",
        engine,
        if_exists="append",  # schema.sql already created the table structure
        index=False,
        chunksize=10_000,
        method="multi",
    )
    print("Done. Data loaded successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_data.py /path/to/creditcard.csv")
        sys.exit(1)

    load_csv_to_db(sys.argv[1])