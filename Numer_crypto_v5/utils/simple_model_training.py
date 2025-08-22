
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class SimpleModelTraining:
    def __init__(self, logger):
        self.logger = logger

    def run(self, data_df):
        self.logger.info("Training simple model...")

        # Drop rows with missing values
        data_df = data_df.dropna()

        # Select features and target
        features = [
            "log_close_lag1",
            "log_volume_lag1",
            "high_low_spread_lag1",
            "open_close_spread_lag1",
            "log_high_lag1",
            "log_low_lag1",
            "pvm_mean_lag1",
            "onchain_mean_lag1",
        ]
        target = "target"

        X = data_df[features]
        y = data_df[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate the RMSE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        self.logger.info(f"Simple model trained. RMSE: {rmse}")

        return model, rmse
