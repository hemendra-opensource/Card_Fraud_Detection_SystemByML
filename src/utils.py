import os
import logging
import joblib

def setup_logging(name="project"):
    """
    Sets up a basic console and file logging configuration.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger

def ensure_dir(path):
    """
    Ensures that a directory exists, creating it if necessary.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def save_model(model, filepath):
    """
    Saves a model object using joblib.
    """
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(model, filepath)
    print(f"Model saved successfully to {filepath}")

def load_model(filepath):
    """
    Loads a model object using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model file found at {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded successfully from {filepath}")
    return model
