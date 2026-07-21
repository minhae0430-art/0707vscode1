import pandas as pd
import os

path = r'c:\Users\밍혜\Desktop\VScode\260721_anode_visc._HB\260720_anode_HB_Fitting.xlsx'
print('exists', os.path.exists(path))
xl = pd.ExcelFile(path)
print('sheets', xl.sheet_names)
for s in xl.sheet_names:
    df = pd.read_excel(path, sheet_name=s, header=None)
    print('\n==== SHEET', s, '====')
    print(df.head(20).to_string(index=False, header=False))
    print('shape', df.shape)
    print('rows 0-5 values:')
    for i in range(min(5, df.shape[0])):
        print(i, df.iloc[i].tolist())
