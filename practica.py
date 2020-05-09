# -*- coding: utf-8 -*-
"""
Created on Sat May 2 5:33:42 2020

@author: m3t4l
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from textwrap import wrap

df = pd.read_csv('survey_results_public.csv')
df['YearsCode'].replace('Less than 1 year', '0.5', inplace=True)
df['YearsCode'].replace('More than 50 years', '51', inplace=True)
df['YearsCode'] =  df['YearsCode'].astype('float64') # Convierte la columna a flotantes
#df.head()

def minimo(col):
    return "{:.2f}".format(col.min())

def maximo(col):
    return "{:.2f}".format(col.max())

def q1(col):
    return "{:.2f}".format(col.quantile(0.25))

def q2(col):
    return "{:.2f}".format(col.quantile(0.50))

def q3(col):
    return "{:.2f}".format(col.quantile(0.75))

def mediana(col):
    return "{:.2f}".format(col.mean())

def desv_std(col):
    return "{:.2f}".format(col.std())

def sub_str(str_, sub_str_):
    '''
        str_ es un string
        No contiene valores nulos
    '''
    sub_str_ = sub_str_.replace(r'++', r'\+\+')
    reg = re.compile('(;|^)' + sub_str_ + '(;|$)')
#     print(reg)
    return bool(reg.findall(str_))
#     return sub_str_ in str_

def unicos(col):
    l = list(col.unique())
    l = ';'.join(l).split(';')
    return list(set(l))


def filter_not_nulls(df, *cols):
    f = df[cols[0]].notnull()
    for col in cols[1:]:
        f = f & df[col].notnull()
    return df[f]

def cinco_numeros(col):
    min_ = minimo(col)
    max_ = maximo(col)
    q1_   = q1(col)
    q2_   = q2(col)
    q3_   = q3(col)
    return min_, max_, q1_,q2_,q3_

def cajas(df, col_a, col_b, nrows=1, ncols=None, filename=None, truncate=20):
    new_df = filter_not_nulls(df, col_b, col_a) # Filtra para las columnas gender y salary
    #print(new_df)
    uniques_ = unicos(new_df[col_b]) # uniques almacenará los valores diferentes de género
    uniques_
    s = '|Minimo = {0}, Maximo = {1}, Q1 = {2}, Q2 = {3}, Q3 = {4}, Std = {5}, Mean = {6}|'
#     plt.figure().subtitle('Boxplots')
    if not ncols:
        ncols = len(uniques_)
    for i, uni in enumerate(uniques_):
        # Filtrar por cada género
        filter_df = new_df[col_b].apply(sub_str, args=(uni,))
        # Dataframe filtrado
        g_df = new_df[filter_df]
        # Asignamos dataframe filtrado a cada género
        print(len(g_df))
        print(uni, s.format(*cinco_numeros(g_df[col_a]), 
                                    desv_std(g_df[col_a]),mediana(g_df[col_a])))
        plt.subplot(nrows, ncols, i+1)
        #plt.boxplot(g_df[col_a], sym='')
        plt.boxplot(g_df[col_a])
        plt.ylabel('Amount')
        plt.xlabel(uni[:truncate])
        #plt.xlabel(uni[:truncate])
        plt.title(f'Boxplot for {uni[:truncate]}')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False) 
        plt.tight_layout()
    if filename:
        plt.savefig(filename)
    #plt.show()
    plt.clf()
    
    
def median_mean_str_dev(df,col_a,col_b,nrows=1,ncols=None, truncate=10):
    new_df = filter_not_nulls(df, col_b, col_a) # Filtra para las columnas gender y salary
    uniques_ = unicos(new_df[col_b]) # uniques almacenará los valores diferentes de género
    s = '{0}|Q2 = {1}, Std = {2}, Mean = {3}|'
    for i in uniques_:
        filter_df = new_df[col_b] == i  # 
        #filter_df = new_df[col_b].apply(sub_str, args=(i,))
        f_df = new_df[filter_df]
        print(len(f_df))                  # Dataframe filtrado para cada país
        print(s.format(i, q2(f_df[col_a]), desv_std(f_df[col_a]),mediana(f_df[col_a])))
        

def bar_freq_graph(df,col_a):
    new_df = filter_not_nulls(df,col_a)
    uniques_ = unicos(new_df[col_a])
    freqs = {}
    for i in uniques_:
        freq = sum(new_df[col_a].apply(sub_str,args=(i,)))
        freqs[i] = freq
    height = np.arange(len(freqs))
    plt.figure(figsize=(12,10))
    plt.bar(height=list(freqs.values()), x = height)
    plt.xticks(height,freqs.keys(),rotation=90)
    
def bar_freq_graph2(df,col_a):
    new_df   = filter_not_nulls(df, col_a) # new_df is indexable
    devtypes = unicos(new_df[col_a])
    freqs = {} # keys = devtypes, values = Frec
    for devtype in devtypes:
        freq = sum(new_df[col_a].apply(sub_str, args=(devtype,)))
        freqs[devtype] = freq
        freqs = sorted(freqs.items(), key=lambda item : item[1], reverse=True)
        freqs = {k:v for k, v in freqs}
        x = np.arange(len(freqs))
    plt.figure(figsize=(12,10), facecolor='w')
    plt.bar(height=freqs.values(), x=x)
    plt.xticks(x, freqs.keys(), rotation=90)
    
    
def hist(df, col_a, col_b, nrows=1, ncols=None,
        xlabel=None, ylabel=None, filename=None, 
        nbins=10, fontsize=12):
    new_df = filter_not_nulls(df, col_b, col_a)
    uniques_ = unicos(new_df[col_b])
    if not ncols:
        ncols = len(uniques_)
    if ylabel is None:
        ylabel = "Amount"
    if not xlabel:
        xlabel = col_a
    for i, unique in enumerate(uniques_):
        #print(unique)
        #print(i)
        filter_df = new_df[col_b].apply(sub_str, args=(unique,))
        f_df = new_df[filter_df]
        #print(f_df)
        plt.subplot(nrows,ncols,i+1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title('\n'.join(wrap(unique,30)), fontsize=fontsize)
        plt.hist(f_df[col_a], bins=nbins)
        plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
    
def corr(df,col_a,col_b):
    new_df = filter_not_nulls(df, col_a, col_b)
    x = new_df[col_a].to_numpy()
    y = new_df[col_b].to_numpy()
    corr = np.corrcoef(x=x, y=y)
    print(corr)
    plt.figure(figsize=(9,9), facecolor='w')
    plt.scatter(x=x, y=y)
    plt.title(col_a + 'and ' + col_b +' correlation')
    plt.xlabel(col_a)
    plt.ylabel(col_b)
    
    
    
# P1 - Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per gender.
#plt.figure(figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
#cajas(df,'ConvertedComp','Gender',ncols=3,nrows=1,filename='P1.png',truncate=10)
    
# P2 - Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per ethnicity
#plt.figure(figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
#cajas(df,'ConvertedComp','Ethnicity',ncols=3,nrows=3,filename='P2.png',truncate=10)
    
#P3 - Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per developer type.
#plt.figure(figsize=(35, 25), dpi=80, facecolor='w', edgecolor='k')
#cajas(df,'ConvertedComp','DevType',ncols=8,nrows=4,filename='P3.png',truncate=20)

#P4 - Compute the median, mean and standard deviation of the annual salary per country.
#median_mean_str_dev(df,"ConvertedComp","Country",nrows=12,ncols=5,truncate=10)

#P5 - Obtain a bar plot with the frequencies of responses for each developer type.
#bar_freq_graph(df,"DevType")

#P6 Plot histograms with 10 bins for the years of experience with coding per gender
'''
plt.figure(figsize=(4,6), facecolor='w')
hist(df, 'YearsCode', 'Gender', xlabel='Experience', ylabel='',
    nrows=3, ncols=1, nbins=10,filename="P6.png")
'''
    
#P7 - Plot histograms with 10 bins for the average number of working hours per week, per developer type.
'''
new_df = filter_not_nulls(df, 'WorkWeekHrs', 'DevType') # Limpiamos los nulos
filter_df = ((new_df['WorkWeekHrs'] < 70) & 
             (new_df['WorkWeekHrs'] > 10))              # Eliminamos pocas y muchas horas
new_df = new_df[filter_df]        

plt.figure(figsize=(8,16), facecolor='w')
hist(new_df, 'WorkWeekHrs', 'DevType', 
     xlabel='Worked hours per week', ylabel='',
     nrows=8, ncols=3, nbins=10)
'''
#P8 - Plot histograms with 10 bins for the age per gender
'''
new_df = filter_not_nulls(df, 'Age', 'Gender') # Limpiamos los nulos
filter_df = ((new_df['Age'] < 90) & 
             (new_df['Age'] > 5))           # Eliminamos pocas y muchas horas
new_df = new_df[filter_df]                              # Sin valores nulos
                                                        # Sin valores "no posibles"
# figsize = (width, height)
plt.figure(figsize=(8,4), facecolor='w')
hist(new_df, 'Age', 'Gender', 
     xlabel='Age', ylabel='',
     nrows=1, ncols=3, nbins=50)
'''

#P9 - Compute the median, mean and standard deviation of the age per programming language.
#median_mean_str_dev(df,"Age","LanguageWorkedWith",nrows=12,ncols=5,truncate=10)    

#P10 - Compute the correlation between years of experience and annual salary.
corr(df,'ConvertedComp','YearsCode')

#P11 - Compute the correlation between the age and the annual salary.
#corr(df,'ConvertedComp','Age')

#P12 - Compute the correlation between educational level and annual salary. In this case, 
# replace the string of the educational level by an ordinal index (e.g. Primary/elementary
# school = 1, Secondary school = 2, and so on).

#P13
#bar_freq_graph2(df,'LanguageWorkedWith')