# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
a = 10
print(type(a))
a = 13.6
print(type(a))
a= 'abc'
print(type(a))
a = False
print(type(a))
"""
list
"""
list1 = [1,2,3]
print(type(list1))
list2 = ['abc',2,13.6]
print(type(list2))

print(list1[2:])
list1[2]= 100
print(list1)
list1.append(10)
print(list1)
list1.extend(list2)
print(list1)
list1.extend(10)
print(list1)
list1.pop(1)
print(list1)

"""
