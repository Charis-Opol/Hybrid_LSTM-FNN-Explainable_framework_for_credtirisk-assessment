import pandas as pd
path = r'C:\Users\chari\Documents\New project\credit-risk-framework\data\raw\Uganda_Mobile_Money_Logs_3000.xlsx'
df = pd.read_excel(path)
print(df.columns.tolist())
print(df.head(5).to_string())
