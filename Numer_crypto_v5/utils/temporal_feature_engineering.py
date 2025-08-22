
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pyarrow.parquet as pq
import cudf

class AggregatedYiedlFeatureEngineering:
    def __init__(self, data_dir: str = "/media/knight2/EDB/numer_crypto_temp/data"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def _load_data_in_chunks(self, file_path):
        self.logger.info(f"Loading data in chunks from {file_path}")
        parquet_file = pq.ParquetFile(file_path)
        self.logger.info(f"File has {parquet_file.num_row_groups} row groups.")
        for i in range(parquet_file.num_row_groups):
            self.logger.info(f"Reading row group {i}")
            table = parquet_file.read_row_group(i)
            # Convert Arrow Table to pandas first, then to cudf
            df = cudf.from_pandas(table.to_pandas())
            # Ensure feature columns are int8
            for col in df.columns:
                if df[col].dtype == 'float64' or df[col].dtype == 'float32':
                    df[col] = df[col].astype('float32')
                if 'feature' in col and df[col].dtype != 'int8':
                    df[col] = df[col].astype('int8')
            yield df

    def run(self, data_df):
        self.logger.info("Generating aggregated yield features...")
        data_df = cudf.from_pandas(data_df)

        yield_historical_path = "/media/knight2/EDB/numer_crypto_temp/data/raw/yield/yield_historical.parquet"
        yield_latest_path = "/media/knight2/EDB/numer_crypto_temp/data/raw/yield/yield_latest.parquet"
        
        historical_chunks = self._load_data_in_chunks(yield_historical_path)
        latest_chunks = self._load_data_in_chunks(yield_latest_path)

        all_chunks = list(historical_chunks) + list(latest_chunks)

        self.logger.info("Combining historical and latest yield data...")
        yield_df = cudf.concat(all_chunks, ignore_index=True)
        yield_df = yield_df.drop_duplicates(subset=['date', 'symbol'], keep='last')

        data_df['date'] = cudf.to_datetime(data_df['date'])
        yield_df['date'] = cudf.to_datetime(yield_df['date'])

        data_df = data_df.merge(yield_df, on=["date", "symbol"], how="left")
        data_df = data_df.sort_values(by=["symbol", "date"])

        pvm_cols = [col for col in data_df.columns if col.startswith('pvm')]
        onchain_cols = [col for col in data_df.columns if col.startswith('onchain')]

        data_df["pvm_mean"] = data_df[pvm_cols].mean(axis=1)
        data_df["onchain_mean"] = data_df[onchain_cols].mean(axis=1)

        data_df["log_close"] = np.log(data_df["close"])
        data_df["log_volume"] = np.log(data_df["volume"])
        data_df["high_low_spread"] = data_df["high"] - data_df["low"]
        data_df["open_close_spread"] = data_df["open"] - data_df["close"]
        data_df["log_high"] = np.log(data_df["high"])
        data_df["log_low"] = np.log(data_df["low"])

        data_df["log_close_lag1"] = data_df.groupby("symbol")["log_close"].shift(1)
        data_df["log_volume_lag1"] = data_df.groupby("symbol")["log_volume"].shift(1)
        data_df["high_low_spread_lag1"] = data_df.groupby("symbol")["high_low_spread"].shift(1)
        data_df["open_close_spread_lag1"] = data_df.groupby("symbol")["open_close_spread"].shift(1)
        data_df["log_high_lag1"] = data_df.groupby("symbol")["log_high"].shift(1)
        data_df["log_low_lag1"] = data_df.groupby("symbol")["log_low"].shift(1)

        data_df["pvm_mean_lag1"] = data_df.groupby("symbol")["pvm_mean"].shift(1)
        data_df["onchain_mean_lag1"] = data_df.groupby("symbol")["onchain_mean"].shift(1)

        features_df = data_df[[
            "date",
            "symbol",
            "target",
            "log_close_lag1",
            "log_volume_lag1",
            "high_low_spread_lag1",
            "open_close_spread_lag1",
            "log_high_lag1",
            "log_low_lag1",
            "pvm_mean_lag1",
            "onchain_mean_lag1",
        ]]

        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method="ffill")
        features_df = features_df.fillna(method="bfill")

        self.logger.info("Aggregated yield features generated.")

        return features_df.to_pandas()
