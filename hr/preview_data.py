import pandas as pd
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/HR-Employee-Attrition-data.csv")
display(df.head(5))
print("Number of rows: ", df.shape[0])

df_train, df_test = train_test_split(df, test_size=0.25)

print("Number training rows: ", df_train.shape[0])
print("Number test rows: ", df_test.shape[0])