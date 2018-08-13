# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 11:16:50 2018

@author: Ankita
"""
#Traditional functional programming
age = [1,2,3,4]
i = 0
for e in age:
    age[i] = e +10
    i = i + 1
print (age)

#Convert above traditional functional programming to better in python
age = [1,2,3,4]
for i,e in enumerate(age):
    age[i] = e + 10
print(age)

#Convert above traditional functional programming to MOST EFFICIENT in python by using
#map is more effective, scalable and parallel process mechanism
#Use map object instead of for loop
age = [1,2,3,4]
weight = [5,6,7,8]
def incr(e,f):
    return e + f + 10
total = list(map(incr,age,weight))
print(total)


#Let us write even shorter code Using lambda
#Lambda is anonymous function/in-line funtion
#Note that lambda is used only when you don't want to re-use the function and just at one place.
age = [1,2,3,4]
i = 0
age = list(map(lambda e: e + 10,age))
print (age)    

#applying map
def incr(a):
    return a+10
import pandas as pd
df = pd.DataFrame({'c1':[10,20,30]})
df['c1'].map(incr)
#OR
import pandas as pd
df = pd.DataFrame({'c1':[10,20,30]})
df['c1'].map(lambda a: a+10)  #With out dummy function




    