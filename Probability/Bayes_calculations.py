# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:00:53 2018

@author: Ankita
"""
#Baye's rule
def bayes(prior, likelihood, evidence):
        return (prior * likelihood) / evidence
    
#1st Round, 
#Prior belief is 1.48 out of 1000 people have cancer and test raliability is assuming that only 93% accurate
print(bayes(0.00148, 0.93, (0.00148*0.93 + 0.998*0.01)))

#2nd Round
#Posterior Probabilty of 1st round is 0.12 and this becomes the prior probability for 2nd round
#Prior belief is 0.12 out of 1000 people have cancer provided 1st round of test is +ve and test raliability is assuming that only 93% accurate
print(bayes(0.12, 0.93, (0.12*0.93 + 0.88*0.01)))


######In General########################
#prior belief is 2% and cancer test raliability is assuming that only 90% accurate
print(bayes(0.2, 0.9, (0.2*0.9 + 0.98*0.1)))

#prior belief:2% and test reliability:99%
print(bayes(0.02, 0.99, (0.02*0.99 + 0.98*0.01)))

###applying bayes rule for cancer diagnosis
#prior belief:2% and test reliability:100%
print(bayes(0.02, 1, (0.02*1 + 0.98*0)))

#prior belief:50% and test reliability:90%
print(bayes(0.5, 0.9, (0.5*0.9 + 0.5*0.1)))

#prior belief:50% and test reliability:95%
print(bayes(0.5, 0.95, (0.5*0.95 + 0.5*0.05)))

###applying bayes rule for detecting fair coin or not
#prior belief on being fair:90% and coin toss gives head
print(bayes(0.9, 0.5, (0.9*0.5 + 0.1*1)))

#prior belief on being fair:81% and coin toss gives head
print(bayes(0.81, 0.5, (0.81*0.5 + 0.19*1)))

#prior belief on being fair:68% and coin toss gives head
print(bayes(0.68, 0.5, (0.68*0.5 + 0.32*1)))

#prior belief on being fair:51% and coin toss gives tail
print(bayes(0.51, 0.5, (0.51*0.5 + 0.49*0)))