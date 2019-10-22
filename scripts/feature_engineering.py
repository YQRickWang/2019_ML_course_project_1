# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:39:22 2019

@author: Rick Wang
"""

import numpy as np

undefined = -999



def compute_particle_component (particle_t, particle_eta, particle_phi):
    '''
    Description: This function takes the transverse momentum and its azimuth
angle and outputs the x component of momentum vector.
    
    Arguments:
        
    particle_t - the transverse momentum of praticles

    particle_eta - the pseudorapidity of the transverse momentum of a particle
    
    particle_phi - the azimuth angle of the transverse momentum of a particle    
    
    Output:
    
    1. x component of the momentum vector
    2. y component of the momentum vector
    3. z component of the momentum vector
    
    '''
    
    #get the length of particle_t
    length = len(particle_t)
    
    #create an array to store the x, y, z component of each particle
    particle_x = np.zeros(length)
    particle_y = np.zeros(length)
    particle_z = np.zeros(length)
    
    #get the valid index of particle (if the value is not -999 then it's valid)
    valid_index = (particle_t != undefined)
    
    #compute the x, y, z component for valid index
    particle_x[valid_index] = (particle_t[valid_index]
                               * np.cos(particle_phi[valid_index]))
    particle_y[valid_index] = (particle_t[valid_index]
                               * np.sin(particle_phi[valid_index]))    
    particle_z[valid_index] = (particle_t[valid_index]
                               * np.sinh(particle_eta[valid_index]))    
    #set undefined for invalid ones
    particle_x[~valid_index] = undefined
    particle_y[~valid_index] = undefined
    particle_z[~valid_index] = undefined
    
    #return the results
    return particle_x, particle_y, particle_z

def particle_energy(particle_t,particle_eta,particle_phi):
    '''
    Description: This function takes the transverse momentum, its azimuth
angle and pseudorapidity then outputs the energy of the particle.
        
    Arguments:
        
    particle_t - the transverse momentum of praticles

    particle_eta - the pseudorapidity of the transverse momentum of a particle

    particile_phi - the azimuth angle of the transverse momentum of a particle    
    
    Output:
    
    1. the energy of the particle
    
    '''
    
    #get the length of particle_t
    length = len(particle_t)
    
    #create an array to store the energy of the particle
    energy = np.zeros(length)
    
    #get the valid index of particle (if the value is not -999)
    valid_index = (particle_t!=-999)
    
    #compute each component of particle
    px, py, pz = compute_particle_component(particle_t, 
                                             particle_eta, particle_phi)
    #compute energy for valid index
    energy[valid_index] = np.sqrt(px[valid_index]**2
                                  + py[valid_index]**2
                                  + pz[valid_index]**2)
    
    #set undefined for invalid ones
    energy[~valid_index] = undefined
    
    #return the results
    return energy

def cross_product (particle_t_1, particle_eta_1, particle_phi_1
                   particle_t_2, particle_eta_2, particle_phi_2):
    '''
    Description: This function takes two different particles' transverse 
momentum vector then outputs the cross product of two vectors.
        
    Arguments:
        
    particle_t_1 - the transverse momentum of praticle 1

    particle_eta_1 - the pseudorapidity of the transverse momentum of 
particle 1

    particile_phi_1 - the azimuth angle of the transverse momentum of 
particle 1

    particle_t_2 - the transverse momentum of praticle 2

    particle_eta_2 - the pseudorapidity of the transverse momentum of 
particle 2

    particile_phi_2 - the azimuth angle of the transverse momentum of 
particle 2   
    
    Output:
    
    1. the cross product of two vectors
    
    '''    
    
    #get the length of particle_t_1 and particle_t_2
    length = len(particle_t_1)
    
    #create arrays to store each component of cross product of two vectors
    cp_x = np.zeros(length)
    cp_y = np.zeros(length)
    cp_z = np.zeros(length)
    
    #compute each component of two vectors
    px_1, py_1, pz_1 = compute_particle_component(particle_t_1, 
                                                  particle_eta_1, 
                                                  particle_phi_1)
    px_2, py_2, pz_2 = compute_particle_component(particle_t_2,
                                                  particle_eta_2,
                                                  paticle_phi_2)
    