import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
# import seaborn as sns
# import matplotlib.pyplot as plt

# read data in
X_train = pd.read_pickle(r"C:\Users\Intel\Desktop\ECE542\proj_F\542_final_proj\data preprocessing\X_train.pk1")
y_train = pd.read_pickle(r"C:\Users\Intel\Desktop\ECE542\proj_F\542_final_proj\data preprocessing\y_train.pk1")

# check format is correct
print("shape of training data",X_train.shape)
print("shape of test data",y_train.shape)

#model def
model = Sequential([
    Dense(13, activation='relu', input_shape=(13,)),  # Input layer: '10' for the number of features in the dataset
    Dense(8, activation='relu'),                      # Hidden layer with 8 neurons
    Dense(1, activation='linear')                     # Output layer: '1' because we’re predicting a single value
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#traing the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose = 2)

#save model for next prediction
model.save('trained_model_simpleMLP.keras')


