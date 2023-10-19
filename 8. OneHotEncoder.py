# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:54:54 2023

@author: ds448
"""


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
df=sns.load_dataset('tips')
print(df)
encoder=OneHotEncoder()
#Fit encoder to datafram and transform the categorical variable
encoded=encoder.fit_transform(df[['sex']])
encoded_df=pd.DataFrame(encoded.toarray(),columns=encoder.get_feature_names_out())
pd.concat([df,encoded_df],axis=1)
#For our created dataset
#encoded=encoder.fit_transform(df[['color']])
#import pandas as pd
#encoded_df=pd.DataFrame(encoded.toarray(),columns=encoder.get_feature_names_out())
#pd.concat([df,encoded_df],axis=1)
