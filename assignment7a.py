# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:00:27 2021

@author: hamid
"""

#Subtask 2.1: Collecting Data
import numpy as np
import pandas as pd

df = pd.read_csv("usnews.csv")
rows = [(r'^ ',r''), (r' {2,}',r' '), (r' - ',r' '), (r'%',r'pct')]
for r in rows:
    df.columns = df.columns.str.replace(r[0], r[1])

df = df.applymap(lambda x: np.NaN if x=='*' else x)

df['College Name'] = df['College Name'].str.replace(r' {2,}',r' ')

d = {1: "Public", 2: "Private"}
df['Public private cat']= df['Public/private'].map(lambda x: d.get(x,x))

print(df.head(6))

#Subtask 2.3: Interquartile Ranges
def interquartile(i_input, column_q1, column_q3, new_column):    
    Third_q = i_input[column_q3].factorize()[0]
    First_q = i_input[column_q1].factorize()[0]
    result = Third_q - First_q
    i_input[new_column] = result 
    return i_input

IQR_Math_SAT = interquartile(df,'First quartile Math SAT','Third quartile Math SAT','IQR-Math SAT')

IQR_Verbal_SAT = interquartile(df,'First quartile Verbal SAT','Third quartile Verbal SAT','IQR-Verbal SAT')

df.head(5)

#Subtask 2.4: Saving for statistical analyais
df.fillna("", inplace=True)
df.to_csv('usnews_processed.csv', index=False)




#Subtask 3.1: Using control flow only


import pandas as pd
import numpy as np

cps=pd.read_csv("cps.csv")


cps.shape[0]

def unionized_wages_cf(cps): 
    unionized=0
    totalwage=0
    for xx in range(len(cps)):
        row=cps.iloc[xx,:]
        if row['union']=='Union':
            unionized+=1
            totalwage=totalwage+row['wage']
    average_wage = totalwage/unionized
    return(average_wage)

hourly_wage_cf = unionized_wages_cf(cps)
hourly_wage_cf

def unionized_wages_ap(row):
    if row['union'] == "Union": 
        return(row['wage'])
    else:
        return(np.nan)

#Subtask 3.2: Using apply
def union_wages_apply(cps):    
    cps['unionWage']=cps.apply(unionized_wages_ap,axis=1)
    hourly_wage_ap=np.mean(cps['unionWage'])
    return hourly_wage_ap
union_wages_apply(cps)
#Subtask 3.3: pandas only
def unionized_wages_pandas(cps):
    return cps[cps.union == 'Union'].wage.mean()
hourly_wage_pd = cps[cps.union == 'Union'].wage.mean()
print('Hourly mean wages for Union Worker: ', hourly_wage_pd)

#Subtask 3.4: Comparing time to run
# %timeit unionized_wages_cf(cps)

# %timeit union_wages_apply(cps)

# %timeit unionized_wages_pandas(cps)



#unionized_wages_fastest = %timeit unionized_wages_pandas



'''unionized_wages_pandas is the fastest'''


