import pandas as pd
import os
p='PEO(200M)1wt_carraue_Fitting.xlsx'
df=pd.read_excel(p)
print('shape', df.shape)
print('columns', df.columns.tolist())
for i, col in enumerate(df.columns):
    s=pd.to_numeric(df[col], errors='coerce')
    print(i, repr(col), 'numeric_count', s.notna().sum(), 'first5', s.dropna().head().tolist())
print('\nraw head')
print(df.head(10).to_string())
