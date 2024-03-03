import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# read data in
df=pd.read_csv(r"C:\Users\Intel\Desktop\ECE542\proj_F\sgemm\sgemm_product.csv", encoding='utf-8')
# check format is correct
print(df.head())

#add a new column - mean of execution time as result
df['avg_run'] = df[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']].mean(axis=1)
df = df.drop(['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'],axis=1)
#normalize inputs
# columns_to_normalize = ['MWG','NWG','KWG','MDIMC','NDIMC','MDIMA','NDIMB','KWI','VWM','VWN']
columns_to_normalize = df.columns[:10]
df_to_normalize = df[columns_to_normalize]
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_to_normalize), columns=columns_to_normalize) 
df[columns_to_normalize] = df[columns_to_normalize].astype(float)
df.update(df_normalized)
# min_val = df['MWG'].min()
# max_val = df['MWG'].max()
# print("MWG min = ",min_val,",max = ",max_val)
# df["MWG"] = (df["MWG"]-min_val)/(max_val-min_val)
print("after normalize")
print(df.iloc[136130:136135])

#Correlation Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,fmt=".2f")
# plt.show()

#save to pickle for next task
df.to_pickle('raw_df.pk1')


