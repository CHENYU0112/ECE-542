import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'path_to_your_saved_model.h5'  # Update this path
model = load_model(r"C:\Users\Intel\Desktop\ECE542\proj_F\542_final_proj\model_traing_predict\trained_model_simpleMLP.keras")

# Load the test dataset
X_test = pd.read_pickle(r"C:\Users\Intel\Desktop\ECE542\proj_F\542_final_proj\data preprocessing\X_test.pk1")
y_test = pd.read_pickle(r"C:\Users\Intel\Desktop\ECE542\proj_F\542_final_proj\data preprocessing\y_test.pk1")

# Evaluate the model on the test data
model.evaluate(X_test, y_test, verbose=1)


# Predictions
predictions = model.predict(X_test)

# Displaying some predictions
print("Some predictions:", predictions[:5])
