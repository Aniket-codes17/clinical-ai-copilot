import pandas as pd
df=pd.read_csv("data.csv")
# print(df.head())
# print(df.shape)
# print(df.columns)

# print(df.describe())
# print(df.isnull().sum())
import matplotlib.pyplot as plt
df["age"].hist()
plt.savefig("age_chart.png")
plt.show()