# -*- coding: utf-8 -*-
"""
A resource-competition ecological model, where species face different metabolic trade-offs, based on

Metabolic Trade-Offs Promote Diversity in a Model Ecosystem, PRL 118, 028103 (2017)
A. Posfai, T. Taillefumier, N. S. Wingreen

Alexandre Barbosa
"""

import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import mstats
import math

""" Simulation Fixed Parameters """

m = 30 # number of species
p = 20 # number of resources
nutrient_supply_rates = supply_rates(p, 1.) # in a loop, this functions sets the same nutrient supply rates for different runs

""" Run the Simulation """

def run_simulation(m, p, tradeoff, *args, **kwargs):
    
    """ Initialization """
    
    # Write parameters to output file
    current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    
    exponent = kwargs.get('exponent', None)
    r = kwargs.get('r', None)
    death_rates = kwargs.get('delta', np.ones(m)*0.01) #0.01 * np.random.uniform(low=0.5, high=1.0, size=m)
    loss_rate = kwargs.get('loss_rate', 0.1*np.average(death_rates)*np.ones(p))
    noise_std = kwargs.get('noise_std', 1.)
    noise_scale = kwargs.get('noise_scale', 0.1)
    supply_fluctuations_scale = kwargs.get('supply_fluctuations_scale', 0.01)
    supply_fluctuations_frequency = kwargs.get('supply_fluctuations_frequency', np.average(death_rates))
    supply_fluctuations_phi = kwargs.get('supply_fluctuations_phi', 0.5)
    tf = kwargs.get('tf', 20/np.average(death_rates)) # simulation time
    dt = kwargs.get('dt', 0.5) # time step for solving ODE
    
    with open('output.txt', 'a') as f:
        print(' \n Simulation Parameters: m = {}, p = {}, {} tradeoff, k = {}, r = {} \n Start: {}'.format(m, p, tradeoff, exponent, r, current_time), file=f)
    
    # Create the ecossystem (class object)
    ecossystem = Ecossystem([], [])
    ecossystem.set_random_supply_rates(p, 1) # S = 100     or
    #ecossystem.set_supply_rates(nutrient_supply_rates) 
    ecossystem.set_community(m, tradeoff, 1., death_rates, exponent=exponent, r=r) # E = 1
    #ecossystem.set_community_switch(m, tradeoffs, death_rates)
    
    delta = np.average(death_rates)
    
    """ Simulation Time Parameters """

    npoints = int(tf/dt)
    t = np.linspace(0, tf, npoints)
    
    # Adding White Gaussian Noise
    
    noise = ecossystem.fluctuations(0, noise_std, noise_scale, npoints)
    
    if not tradeoff == 'switch-boolean' or tradeoff == 'switch-p-norm': # otherwise, the convexity hull condition returns an error
        ternary_plot(ecossystem) # false for p != 3
        ecossystem.coexistence_condition()

    y = np.zeros((npoints, m)) # array to store points
    
    # Handle the case where there are unused nutrients
    
    S = 0
    
    for i in range(p):
        if sigma_sum(i, ecossystem) != 1:
            S += ecossystem.nutrient_supply_rates[i]
            
    for j in range(m):
        ecossystem.community[j].set_population(S/delta)
    
    # Set Initial Conditions
    for i in range(m):
        y[0, i] = ecossystem.community[i].population
        
    
    # Add a sinusoidal perturbation to the nutrient supply rates
    a = supply_fluctuations_scale
    w = supply_fluctuations_frequency
    phi = supply_fluctuations_phi

    
    """ Run the Simulation """
    
    exctinction_threshold = 0.001 # = 1 individual, so the total population is S/(exctinction_treshold*delta)
        
    for k in range(1, npoints):
        c = np.zeros(p)
        for i in range(p):
            ecossystem.nutrient_supply_rates[i] += (a*math.cos(w*t[k]+phi)-a*math.cos(w*t[k-1]+phi))
            c[i] = ecossystem.nutrient_supply_rates[i]/sigma_sum(i, ecossystem)
        for j in range(m): # = growth * n_sigma + noise
            y[k, j] = y[k-1, j] + growth(j, dt, loss_rate, c, ecossystem) * ecossystem.community[j].population * dt + noise[k*(j+1)] * dt**(0.5)
            ecossystem.community[j].population = y[k, j]
            if y[k, j] <= exctinction_threshold: # check for exctinctions
                for n in range(k, npoints):
                    y[n, j] = 0
            
    """ Plots """

    coexisting_species = 0
    
    final_population = np.zeros(m)
    final_population_avg = np.zeros(m)
    
    for sigma in range(m):
        final_population[sigma] = y[npoints-1, sigma] 
        final_population_avg[sigma] = np.mean(y[npoints-101:npoints-1, sigma])

    for i in range (1, m+1):
        plt.plot(t*delta, y[:, i-1]/(S/delta), label='Species %s' % i)
        if y[npoints-1, i-1] > (np.sum(final_population)*5/(m*delta*tf)):
            coexisting_species += 1      
       
    plt.xlabel('t (1/δ)')
    plt.ylabel('Population (S/δ)')
    plt.axis([0, tf*delta, 0, 1.15])

    if m <= 10: # disable legend for large number of species
        plt.legend() 
    
    plt.savefig("Population_m{}_p{}_{}.jpg".format(ecossystem.m, ecossystem.p, current_time))
    plt.show()
    
        
    np.sort(final_population)
    
    diversity_index_0 = 1./mstats.hmean(final_population_avg/np.sum(final_population_avg)) # inverse of harmonic mean
    diversity_index_1 = 1./mstats.gmean(final_population_avg/np.sum(final_population_avg)) # inverse of geometric mean
    diversity_index_2 = 1./np.mean(final_population_avg/np.sum(final_population_avg))  # inverse of arithmetic mean
    berger_parker_index = np.amax(final_population_avg)/np.sum(final_population_avg) # abundance of the most abundant species
    diversity_index_infty = 1./berger_parker_index

    # Print results to file

    with open('richness.txt', 'a') as f:
        print(coexisting_species, file=f)
        
    with open('r.txt', 'a') as f:
        print('{}'.format(r), file=f)

    with open('diversity_index_0.txt', 'a') as f:
       print('{} \n'.format(diversity_index_0), end =" ", file=f) 
    
    with open('diversity_index_1.txt', 'a') as f:
        print('{} \n'.format(diversity_index_1), end =" ", file=f)
    
    with open('diversity_index_2.txt', 'a') as f:
        print('{} \n'.format(diversity_index_2), end =" ", file=f)
        
    with open('berger_parker_index.txt', 'a') as f:
        print('{} \n'.format(berger_parker_index), end =" ", file=f)
        
    with open('diversity_index_infty.txt', 'a') as f:
        print('{} \n'.format(diversity_index_infty), end =" ", file=f)
            
    with open('population.txt', 'a') as f:
        for i in range(m-1):
            print('{},'.format(final_population_avg[i]/np.sum(final_population_avg)), end =" ", file=f)
        print('{} \n'.format(final_population_avg[m-1]/np.sum(final_population_avg)), end =" ", file=f)
        
# Example: Run the Simulation in a loop using the 'k-norm' trade-off, with different exponents
        
k_lst = [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5]
    
for k in k_lst:
    for i in range(30):
        run_simulation(m, p, 'p-norm', exponent=k, delta=np.ones(m)*0.01, tf = 5/0.01, loss_rates=(np.zeros(p)), noise_scale = 0.,
                       supply_fluctuations_scale=0.)