# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:02:20 2018

@author: Ankita
"""
#Understand random.randint() function

import numpy as np
import pandas as pd

#Beautiful np.random.randint() function

#We are saying that from 0 to 1(Coin) and random tosses of 1000 times
coin_tosses = np.random.randint(0,2,1000)
pd.crosstab(index=coin_tosses, columns="Count")

#We are saying that from 0 to 6(Dice) and random tosses of 1000 times
dice_throws = np.random.randint(1,7,1000) 
pd.crosstab(index=dice_throws, columns="Count")