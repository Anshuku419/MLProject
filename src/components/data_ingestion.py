import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_features, categorical_features):
        try:
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline (only if you have categorical features)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent"))
                ]
            )

            # Column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_features),
                    ("cat", cat_pipeline, categorical_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation...")

            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Assume target column name
            target_column = "target"
            numerical_features = [col for col in train_df.columns if col != target_column]
            categorical_features = []  # add categorical column names if available

            input_features_train = train_df.drop(columns=[target_column])
            target_feature_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column])
            target_feature_test = test_df[target_column]

            # Preprocessor
            preprocessor = self.get_data_transformer_object(
                numerical_features, categorical_features
            )

            input_features_train_arr = preprocessor.fit_transform(input_features_train)
            input_features_test_arr = preprocessor.transform(input_features_test)

            # Save preprocessor
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            with open(self.config.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info("Data transformation completed successfully.")

            return (
                input_features_train_arr,
                target_feature_train,
                input_features_test_arr,
                target_feature_test,
                self.config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
