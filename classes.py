# -*- coding: utf-8 -*-
"""
Classes and auxiliary functions for a model ecossystem simulation.

Alexandre Barbosa
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime

""" Functions """

def in_hull(points, hull): # test if points in 'points' are in 'hull'
    # https://stackoverflow.com/questions/16750618/
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
     
    return hull.find_simplex(points) >= 0 

def ternary_plot(ecossystem):
        if ecossystem.p == 3:
            trianglex = np.zeros(ecossystem.m)
            triangley = np.zeros(ecossystem.m)
            points = np.zeros((ecossystem.m,2))
            for i in range(ecossystem.m):
                a = ecossystem.community[i].metabolic_strategy[0]
                b = ecossystem.community[i].metabolic_strategy[1]
                c = ecossystem.community[i].metabolic_strategy[2]
                trianglex[i] = 0.5*(2*b+c)/(a+b+c)
                triangley[i] = math.sqrt(3.)/2*(c/(a+b+c))
                points[i] = [trianglex[i], triangley[i]]
                plt.scatter([trianglex[i]], [triangley[i]])
                
            plt.plot([0,0.5,1,0], [0,math.sqrt(3)/2.,0,0], color='k', markersize='4')        

            plt.scatter([0.5*(2*ecossystem.nutrient_supply_rates[1]+ecossystem.nutrient_supply_rates[2])/(ecossystem.nutrient_supply_rates.sum())], 
            [math.sqrt(3.)/2*(ecossystem.nutrient_supply_rates[2]/(ecossystem.nutrient_supply_rates.sum()))], marker='D', color='k')
            
            from scipy.spatial import ConvexHull
            
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k--')
            
            plt.axis('equal') # avoid distortions
            plt.axis('off')
           
            plt.savefig("Ternary_Plot_m{}_p{}_{}.jpg".format(ecossystem.m, ecossystem.p, datetime.now().strftime('%Y-%m-%d %H-%M-%S')))
            plt.show()
            
def sigma_sum(i, ecossystem): # auxiliary function to calculate growth 
    sigma_sum = 0.
    for sigma in range(ecossystem.m):
        sigma_sum += ecossystem.community[sigma].population * ecossystem.community[sigma].metabolic_strategy[i]
    if sigma_sum != 0:
        return sigma_sum
    else:
        return 1 # since a_sigma, i = 0, everything is fine and the term doesn't contribute

def growth(j, dt, loss_rate, c, ecossystem): # dn_sigma/dt = growth (= constant * n_sigma)
    growth = 0
    for i in range(0, ecossystem.p):
        growth += (ecossystem.community[j].metabolic_strategy[i] * ((ecossystem.nutrient_supply_rates[i]  - c[i] * loss_rate[i])/ sigma_sum(i, ecossystem)) - ecossystem.community[j].delta/ecossystem.p)
    growth = growth * ecossystem.community[j].population 
    if ecossystem.community[j].population > 1:
        ecossystem.community[j].population += dt*growth
    else: #extinction
        ecossystem.community[j].population = 0
    return growth

def supply_rates(p, S): # for multiple simulations
    
    while(1):
        nutrient_supply_rates =  np.random.dirichlet(np.ones(p)) # uniform distribution with this constraint
        nutrient_supply_rates *= S
        
        if all(i >= 1/(5*p) for i in nutrient_supply_rates):
                    break  
    #print(nutrient_supply_rates)        
    return nutrient_supply_rates

def switch_tradeoff(r, p, m, E):
    
    metabolic_strategy = np.zeros((m, p))
    if r <= math.factorial(p)/(math.factorial(r)*math.factorial(p-r)): 
            for j in range(m):
                non_zero_alphas = np.random.choice(p, size=r, replace=False)
                for i in non_zero_alphas:
                    metabolic_strategy[j, i] = E
                    
            metabolic_strategy = np.unique(metabolic_strategy, axis=0)
        
            while (len(metabolic_strategy) < m):
                for j in range(m - len(metabolic_strategy)):
                    non_zero_alphas = np.random.choice(p, size=r, replace=False)
                    new_row = np.zeros(p)
                    for i in non_zero_alphas:
                        new_row[i] = E
                    metabolic_strategy = np.vstack([metabolic_strategy, new_row])
                    
                metabolic_strategy = np.unique(metabolic_strategy, axis=0)
    
    return metabolic_strategy
        

""" A Species is defined by its metabolic strategy, which obeys set constraints. """

class Species: #indexed by sigma
    def __init__(self, metabolic_strategy, population, delta):
        self.metabolic_strategy = metabolic_strategy # alpha_sigma,i (array of size p)
        self.population = population
        self.delta = delta
        
    def set_population(self, population):
        self.population = population
    
    def set_tradeoffs(self, p, metabolic_tradeoff, E, *args, **kwargs): # energetic constraint
        exponent = kwargs.get('exponent', None) # p-norm tradeoff
        r = kwargs.get('r', p) # switch tradeoff
        
        if metabolic_tradeoff == 'linear': # fixed energy budget
            self.metabolic_strategy = E * np.random.dirichlet(np.ones(p))
            
        elif metabolic_tradeoff == 'quadratic': # fixed quadratic energy budget            
            #https://mathworld.wolfram.com/HyperspherePointPicking.html
            self.metabolic_strategy =  np.abs(E * np.random.normal(size=p))
            self.metabolic_strategy /= np.linalg.norm(self.metabolic_strategy)
            
        elif metabolic_tradeoff == 'p-norm':  # p-norm sum of a_i
            #https://mathworld.wolfram.com/HyperspherePointPicking.html 
            from scipy.stats import gennorm
            """while(1):
                self.metabolic_strategy = gennorm.rvs(exponent, size=p)
                self.metabolic_strategy /= np.linalg.norm(self.metabolic_strategy, ord=exponent)
                self.metabolic_strategy *= E
                
                # reject if any is non-zero
                if all(i >= 0 for i in self.metabolic_strategy):
                    break    """
                    
            self.metabolic_strategy = np.abs(gennorm.rvs(exponent, size=p))
            self.metabolic_strategy /= np.abs(np.linalg.norm(self.metabolic_strategy, ord=exponent))
            self.metabolic_strategy *= E
                    
                    
                
        elif metabolic_tradeoff == 'bounded-p-norm':  # bounded p-norm sum of a_i
            # https://stats.stackexchange.com/questions/352668/
            from scipy.stats import gennorm
            while (1):
                self.metabolic_strategy = gennorm.rvs(exponent, size=p)
                s = np.zeros(p)
                # generate p random signs
                for i in range(p):
                    if np.random.random() < 0.5:
                        s[i] = 1
                    else:
                        s[i] = -1
                self.metabolic_strategy = s*self.metabolic_strategy
                w = np.power(np.random.random(size=p), 1/p)
                self.metabolic_strategy /= np.linalg.norm(self.metabolic_strategy, ord=exponent)
                self.metabolic_strategy = E*(w * self.metabolic_strategy)
                
                if all(i >= 0 for i in self.metabolic_strategy):
                    break   
                
        elif metabolic_tradeoff == 'switch-linear' and r <= p: # pick r non-zero alphas, the rest should sum to E
            non_zero_alphas = np.random.choice(p, size=r, replace=False)
            non_zero_array =  E*np.random.dirichlet(np.ones(r))
            j = 0
            for i in non_zero_alphas:
                self.metabolic_strategy[i] = E * non_zero_array[j]
                j += 1
                
        elif metabolic_tradeoff == 'switch-boolean' and r <= p:
            non_zero_alphas = np.random.choice(p, size=r, replace=False)
            #print(non_zero_alphas)
            for i in non_zero_alphas:
                self.metabolic_strategy[i] = E #(or E/m, but I don't think it matters)
            
        elif metabolic_tradeoff == 'switch-p-norm' and r <=p:
            non_zero_alphas = np.random.choice(p, size=r, replace=False)
            non_zero_array = np.zeros(r)
            from scipy.stats import gennorm
            while(1):
                non_zero_array = gennorm.rvs(exponent, size=p)
                non_zero_array /= np.linalg.norm(non_zero_array, ord=exponent)
                non_zero_array *= E
                
                # reject if any is non-zero
                if all(i >= 0 for i in non_zero_array):
                    break   
            j = 0
            for i in non_zero_alphas:
                self.metabolic_strategy[i] = E * non_zero_array[j]
                j += 1
                
                
        # An additional possible tradeoff would be an 'approximate' linear constraint drawn from a 'Boltzmann' distribution at temperature T
            
        else:
            print('Invalid Constraint')
            
        with open('output.txt', 'a') as f:
            print(self.metabolic_strategy , file=f)  # Python 3.x"""
                
        #print(self.metabolic_strategy)
        
     
""" An Ecossystem is defined by its nutrient supply rates and by the community (set of species) it hosts. """
        
class Ecossystem:
    def __init__(self, nutrient_supply_rates, community):
        self.nutrient_supply_rates = nutrient_supply_rates #s_i (array of size p)
        self.community = community
        self.p = len(nutrient_supply_rates) # number of nutrients
        self.m = len(community) # number of species
        self.S = sum(nutrient_supply_rates)
        
    def set_supply_rates(self, nutrient_supply_rates): # set fixed supply rates
        self.nutrient_supply_rates = nutrient_supply_rates
        self.p = len(nutrient_supply_rates)
        self.S = sum(nutrient_supply_rates)
        
        with open('output.txt', 'a') as f:
            print('Nutrient Supply Rates: ', self.nutrient_supply_rates , file=f)  
        
    def set_random_supply_rates(self, p, S):
        self.nutrient_supply_rates =  np.random.dirichlet(np.ones(p)) #uniform distribution with this constraint
        self.nutrient_supply_rates /= self.nutrient_supply_rates.sum()
        self.nutrient_supply_rates *= S
        self.p = p
        self.S = sum(self.nutrient_supply_rates)
        
        with open('output.txt', 'a') as f:
            print('Nutrient Supply Rates: ', self.nutrient_supply_rates , file=f)  
        
    def set_community(self, m, tradeoff, E, death_rates, *args, **kwargs):
        
        with open('output.txt', 'a') as f:
            print('Metabolic Strategies: \n ', file=f)
        #S = np.sum(self.nutrient_supply_rates)
        
        for j in range(m):
            self.community.append( Species(np.zeros(self.p), self.S/(m*np.mean(death_rates)), death_rates[j]) )
            self.community[j].set_tradeoffs(self.p, tradeoff, E, *args, **kwargs)    
        self.m = m
        # switch tradeoff
        
       # r = kwargs.get('r', p) # switch tradeoff
        
    def set_community_switch(self, m, tradeoffs, death_rates):
        
        with open('output.txt', 'a') as f:
            print('Metabolic Strategies: \n ', file=f)
        #S = np.sum(self.nutrient_supply_rates)
        
        for j in range(m):
            self.community.append( Species(tradeoffs[j], self.S/(m*np.mean(death_rates)), death_rates[j]) )
            with open('output.txt', 'a') as f:
                print(tradeoffs[j], file=f)
        self.m = m
        
        
    def fluctuations(self, mean, std, scale, npoints): # white gaussian noise
        return scale * np.random.normal(mean, std, size=npoints*self.m)
                        
    def coexistence_condition(self): # derived in Wingreen et al., Phys. Rev. Lett. 118.028103
        
        trianglex = np.zeros(self.m)
        triangley = np.zeros(self.m)
        points = np.zeros((self.m,2))
        for i in range(self.m):
            a = self.community[i].metabolic_strategy[0]
            b = self.community[i].metabolic_strategy[1]
            c = self.community[i].metabolic_strategy[2]
            trianglex[i] = 0.5*(2*b+c)/(a+b+c)
            triangley[i] = math.sqrt(3.)/2*(c/(a+b+c))
            points[i] = [trianglex[i], triangley[i]]
            
            diamond = [ 0.5*(2*self.nutrient_supply_rates[1]+self.nutrient_supply_rates[2])/(self.nutrient_supply_rates.sum()), 
            math.sqrt(3.)/2*(self.nutrient_supply_rates[2]/(self.nutrient_supply_rates.sum()))]
            
        with open('output.txt', 'a') as f:
            print('Convex Hull Coexistence Condition:', in_hull(diamond, points) , file=f)  # Python 3.x"""
            
        return in_hull(diamond, points)