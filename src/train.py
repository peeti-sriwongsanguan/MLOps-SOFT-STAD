import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data, prepare_data_for_model
from model import WalmartSalesModel, Config

def main():
    mlflow.set_tracking_uri("mlflow/mlruns")
    mlflow.set_experiment("walmart-sales-forecast")

    with mlflow.start_run():
        # Load and preprocess data
        df = load_data("data/walmart_cleaned.csv")
        preprocessed_df, feature_scaler, target_scaler = preprocess_data(df)
        X, y = prepare_data_for_model(preprocessed_df)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train = torch.FloatTensor(X_train.values)
        y_train = torch.FloatTensor(y_train.values).unsqueeze(-1)  # Add feature dimension
        X_test = torch.FloatTensor(X_test.values)
        y_test = torch.FloatTensor(y_test.values).unsqueeze(-1)  # Add feature dimension

        # Print shapes for debugging
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

        # Initialize and train the model
        configs = Config()
        configs.n_features = X_train.shape[1]  # Set the number of features
        model = WalmartSalesModel(configs)
        model.set_scaler(target_scaler)  # Set the scaler for denormalization
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("seq_len", configs.seq_len)
        mlflow.log_param("pred_len", configs.pred_len)
        mlflow.log_param("d_model", configs.d_model)
        mlflow.log_param("d_core", configs.d_core)
        mlflow.log_param("e_layers", configs.e_layers)
        mlflow.log_param("n_features", configs.n_features)

        # Evaluate the model
        evaluation_metrics = model.evaluate(X_test, y_test)
        print("Model evaluation metrics:")
        print(evaluation_metrics)

        # Log metrics
        for metric_name, metric_value in evaluation_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Prepare input example and signature for MLflow
        input_example = X_train[:configs.seq_len].unsqueeze(0).numpy()  # Shape: (1, seq_len, n_features)
        output_example = model.model(torch.FloatTensor(input_example).to(model.device)).cpu().detach().numpy()
        signature = infer_signature(input_example, output_example)

        # Save the model with input example and signature
        mlflow.pytorch.log_model(model.model, "soft_model", input_example=input_example, signature=signature)

if __name__ == "__main__":
    main()
