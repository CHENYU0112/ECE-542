import pandas as pd
from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt

# read df in
df = pd.read_pickle(r"C:\Users\Intel\Desktop\ECE542\proj_F\542_final_proj\data exploration\raw_df.pk1")
# check format is correct
print(df.head())

#drop STRN as it's not related to result according to correlation analysis
df = df.drop(['STRN'],axis=1)

#shuffle the data
df_shuffled = df.sample(frac=1).reset_index(drop=True)

#split train test data
#training
X = df_shuffled.iloc[:,:-1]
#testing
y = df_shuffled.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#save to pickle for next task
X_train.to_pickle("X_train.pk1")
y_train.to_pickle("y_train.pk1")
X_test.to_pickle("X_test.pk1")
y_test.to_pickle("y_test.pk1")

