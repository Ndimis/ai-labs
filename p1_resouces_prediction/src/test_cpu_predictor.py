import pytest
import numpy as np
from cpu_predictor import model  # Import the trained model from your script

def test_model_prediction_shape():
    """
    Ensure the model returns a single prediction value for a single input.
    """
    test_input = np.array([[500]])
    prediction = model.predict(test_input)
    assert prediction.shape == (1, 1), "Output shape should be (1, 1)"

def test_positive_correlation():
    """
    In our logic, more connections should generally mean higher CPU load.
    We test if the slope (coefficient) is positive.
    """
    assert model.coef_[0][0] > 0, "Model slope should be positive for resource prediction"

def test_prediction_logic():
    """
    Basic sanity check: 10,000 connections should predict a higher load 
    than 10 connections.
    """
    low_load = model.predict(np.array([[10]]))
    high_load = model.predict(np.array([[10000]]))
    assert high_load > low_load, "Higher connections must result in higher predicted load"