from os import PRIO_PGRP
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(1, 4), columns=list('ABCD'))
print(df)
posLs = ['A','B','C','D']
for index, row in df.iterrows():
    ls = [i for i in posLs if row[i] > 0 ]
print(ls)
 