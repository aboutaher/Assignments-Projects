# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 06:16:38 2021

@author: hamid
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import os
def write_plot_pdf(plot,filename, autoincrement=False):
    file_name = is_pdf(filename)
    if autoincrement:
        dir_list = [path for path in os.listdir() if path.startswith(filename[:-4]+'_')]
        
        if len(dir_list)> 0 \
        and dir_list[-1][:-4].endswith(('0','1','2','3','4','5','6','7','8','9')):
            last_file_number = int(dir_list[-1][dir_list[-1].rfind('_')+1:-4])
            filename = is_pdf(filename[:-4]+f'{last_file_name+1}')
        else:
            filename = is_pdf(filename[:-4]+'_0')
    plot.savefig(filename)


                
df = pd.read_csv('Cars2005.csv')
df.head()

plt.figure(figsize=(10, 6))
plt.scatter(x='Liter',y='Price', data=df, cmap='Spectral', s=100)
plt.xlabel('Liter')
plt.ylabel('Price')
plt.ylim(0,)
plt.title('Relationship between Liter and Price')
plt.legend('Liter_Per_Price ')
plt.savefig('fig1')
plt.savefig('price_liter.pdf')



fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(x = "Liter", y = "Price", data =df, hue = "Leather")
plt.title('Liter_Price_Leather')
ax.set_xlabel('Liter')
ax.set_ylabel('Price')
plt.show()
plt.savefig('fig2')
plt.savefig('price_liter_leather.pdf')



plt.figure(figsize=(10, 6))
sns.histplot(df,x=df['Price'], hue='Leather',multiple='stack')
plt.savefig('fig3')
plt.savefig('price_leather_stacked.pdf')


plt.figure(figsize=(10, 6))
sns.histplot(df,x=df['Price'], hue='Leather',multiple='dodge',bins=10)
plt.savefig('fig4')
plt.savefig('price_leather_unstacked.pdf')


plt.figure(figsize=(8, 6))
agg_df = df.groupby(['Type', 'Leather'])['Price'].sum().unstack().fillna(0)
agg_df.plot(kind='bar', stacked=False).set(title='price_leather_by_Type_unstacked')
plt.savefig('fig5')
plt.savefig('type_leather.pdf')


import seaborn as sns
import pandas as pd
import numpy as np
df = pd.read_csv('Cars2005.csv')
def f(row):
    if row['Make']=='Cadillac':
        val = 'Luxury'
    elif row['Make']=='Buick':
        val = 'Standard'
    else:
        val = 'Economy'
    return val

df['Tier'] = df.apply(f, axis=1)
df

tier_prices = df.groupby('Tier').Price.median()
tier_prices.head()
tier_prices.describe()

import numpy as np
sns.barplot(x = 'Tier', y = 'Price', hue = 'Leather',
           data = df, estimator= np.median)
plt.savefig('fig6')
plt.savefig('tier_prices_barplot.pdf')


plt.figure(figsize=(10, 6))
sns.set_style("darkgrid")
box_plot = sns.boxplot(x="Tier",y="Price",data=df)
medians = df.groupby(['Tier'])['Price'].median()
vertical_offset = df['Price'].median() * 0.05 # offset from median for display

for xtick in box_plot.get_xticks():
    box_plot.text(xtick,medians[xtick] + vertical_offset,medians[xtick], 
            horizontalalignment='center',size='small',color='w',weight='semibold')

plt.savefig('fig7')
plt.savefig('tier_prices_boxplot.pdf')






#write_plot_pdf(fig1,'price_liter.pdf', True)
# write_plot_pdf(fig2,'price_liter_leather.pdf',False)
# write_plot_pdf(fig3,'price_leather_stacked.pdf')
# write_plot_pdf(fig4,'price_leather_unstacked.pdf',False)
# write_plot_pdf(fig5, 'type_leather.pdf')
# write_plot_pdf(fig6,'tier_prices_barplot.pdf',False)
# write_plot_pdf(fig7,'tier_prices_boxplot.pdf')



# write_plot_pfd(fig1,'price_liter.pdf')
# write_plot_pfd(fig2,'price_liter.pdf')
# write_plot_pfd(fig3,'price_liter.pdf')
# write_plot_pfd(fig4,'price_liter.pdf')
# write_plot_pfd(fig5,'price_liter.pfd', True)
# write_plot_pfd(fig2,'price_liter.pdf',True)
# write_plot_pfd(fig2,'price_liter.pdf',True)






