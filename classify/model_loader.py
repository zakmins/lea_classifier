import os
from keras.models import load_model # type: ignore

# Load the model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../lea_classifier_XC.keras')
model = load_model(MODEL_PATH)
