# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:52:02 2018

@author: Ankita
"""

#Strings
name = "abcde"
namesq = 'xyz'
print(type(name))
print(type(namesq))

#access string content
print(name[0])
print(name[2:5])

#modify string content
name[0] = 'A' #Check this

name + 'xyz'
name = name + 'xyz'
print(name)
name = name.upper()
print(name)
name = "mr"
name = name.capitalize()
print(name)
name = "abcde"
name = name.upper()
print(name)
name = name.replace('AB','pq')
print(name)

isinstance(name, str) #True
isinstance(name, int) #False
