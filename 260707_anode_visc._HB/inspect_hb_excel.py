import os
import pandas as pd

path = r'c:\Users\밍혜\Desktop\VScode\260707_anode_visc._HB\260707_anode_visc.HB.xlsx'
print('exists', os.path.exists(path))
xl = pd.ExcelFile(path)
print('sheets', xl.sheet_names)
for s in xl.sheet_names:
    df = pd.read_excel(path, sheet_name=s)
    print('\nSHEET', s, 'shape', df.shape)
    print(df.head(20).to_string())
    print('columns', list(df.columns))
