from src.train_synthetic import make_data, train

def test_training_converges():
    X, y = make_data(n=300)
    w, b = train(X, y)
    assert abs(w - 3.0) < 0.5
    assert abs(b - 2.0) < 0.5