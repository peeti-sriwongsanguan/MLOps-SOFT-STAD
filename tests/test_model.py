import pytest
import torch
import numpy as np
from src.model import WalmartSalesModel, Config


@pytest.fixture
def sample_data():
    X = torch.rand(100, 52, 10)  # 100 samples, 52 weeks, 10 features
    y = torch.rand(100, 13)  # 100 samples, 13 weeks prediction
    return X, y


def test_model_initialization():
    config = Config()
    model = WalmartSalesModel(config)
    assert isinstance(model.model, torch.nn.Module)


def test_model_forward_pass(sample_data):
    X, _ = sample_data
    config = Config()
    model = WalmartSalesModel(config)

    # Prepare input data
    x_enc = X
    x_mark_enc = torch.zeros_like(X)  # Assuming no special encoding for time steps
    x_dec = torch.zeros(X.shape[0], config.pred_len, X.shape[2])
    x_mark_dec = torch.zeros_like(x_dec)

    # Forward pass
    with torch.no_grad():
        output = model.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (X.shape[0], config.pred_len, X.shape[2])


def test_model_fit(sample_data):
    X, y = sample_data
    config = Config()
    model = WalmartSalesModel(config)

    # Implement a simple fit method in WalmartSalesModel for this test
    model.fit(X, y)

    # Add assertions to check if the model parameters have been updated
    # This will depend on how you implement the fit method


def test_model_predict(sample_data):
    X, _ = sample_data
    config = Config()
    model = WalmartSalesModel(config)

    # Implement a simple predict method in WalmartSalesModel for this test
    predictions = model.predict(X)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (X.shape[0], config.pred_len, X.shape[2])


def test_model_evaluate(sample_data):
    X, y = sample_data
    config = Config()
    model = WalmartSalesModel(config)

    # Implement a simple evaluate method in WalmartSalesModel for this test
    metrics = model.evaluate(X, y)

    assert isinstance(metrics, dict)
    assert 'MSE' in metrics
    assert 'RMSE' in metrics