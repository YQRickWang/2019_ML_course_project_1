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

def cross_product (particle_t_1, particle_eta_1, particle_phi_1,
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
    
    #compute the cross vector
    for t in range(length):
        if(px_1[t]!=undefined and px_2[t]!=undefined):
            temp = np.cross(np.array([px_1[t],py_1[t],pz_1[t]]),
                            np.array([px_2[t],py_2[t],pz_2[t]])))
            cp_x[t] = temp[0]
            cp_y[t] = temp[1]
            cp_z[t] = temp[2]
        else:
            cp_x[t] = undefined
            cp_y[t] = undefined
            cp_z[t] = undefined
        
    #return the results
    return cp_x,cp_y,cp_z

def dot_product(particle_t_1, particle_eta_1, particle_phi_1,
                particle_t_2, particle_eta_2, particle_phi_2):
    '''
    Description: This function takes two different particles' transverse 
momentum vector then outputs the dot product of two vectors.
        
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
    
    1. the dot product of two vectors
    
    '''

    #get the length of the particle 1 and 2
    length = len(particle_t_1)
    
    #create an array to store dot product
    dp = np.zeros(length)
    
    #compute each component of two vectors
    px_1, py_1, pz_1 = compute_particle_component(particle_t_1, 
                                                  particle_eta_1, 
                                                  particle_phi_1)
    px_2, py_2, pz_2 = compute_particle_component(particle_t_2,
                                                  particle_eta_2,
                                                  paticle_phi_2)    
    #compute to dot product
    for t in range(length):
        if(px_1[t]!=undefined and px_2[t]!=undefined):
            dp[t] = np.inner(np.array([px_1[t],py_1[t],pz_1[t]]),
                             np.array([px_2[t],py_2[t],pz_2[t]]))
        else:
            dp[t] = undefined
    
    #return the result
    return dp

def cosine_similarity(particle_t_1, particle_eta_1, particle_phi_1,
                      particle_t_2, particle_eta_2, particle_phi_2):
    '''
    Description: This function takes two different particles' transverse 
momentum vector then outputs the cosine similarity of two vectors.
        
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
    
    1. the cosine similarity of two vectors
    
    '''
    #get the length of particle 1 and 2
    length = len(particle_t_1)
    
    #create an array to store cosine similarity
    cs = np.zeros(length)
        
    #find the valid index of two vectors
    valid_index = (particle_t_1!=undefined)&(particle_t_2!=undefined)
    
    #compute dot product
    dp = dot_product(particle_t_1, particle_eta_1, particle_phi_1,
                     particle_t_2, particle_eta_2, particle_phi_2)
    
    #compute cosine similarity
    cs[valid_index] = dp/(particle_t_1*particle_t_2)
    cs[~valid_index] = undefined
    
    return cs

def determinant_vector(particle_t_1, particle_eta_1, particle_phi_1,
                       particle_t_2, particle_eta_2, particle_phi_2,
                       particle_t_3, particle_eta_3, particle_phi_3):
    '''
    Description: This function takes three different particles' transverse 
momentum vector then outputs the determinant vector of two vectors.
        
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

    particle_t_3 - the transverse momentum of praticle 3

    particle_eta_3 - the pseudorapidity of the transverse momentum of 
particle 3

    particile_phi_3 - the azimuth angle of the transverse momentum of 
particle 3
    
    Output:
    
    1. the determinant vector of three vectors
    
    '''
    
    #get the length of particle 1 and 2
    length = len(particle_t_1)
    
    #create an array to store determinant vector
    dv = np.zeros(length)
    
    #compute each component of two vectors
    px_1, py_1, pz_1 = compute_particle_component(particle_t_1, 
                                                  particle_eta_1, 
                                                  particle_phi_1)
    px_2, py_2, pz_2 = compute_particle_component(particle_t_2,
                                                  particle_eta_2,
                                                  paticle_phi_2)
    px_3, py_3, pz_3 = compute_particle_component(particle_t_3,
                                                  particle_eta_3,
                                                  paticle_phi_3) 
    
    #compute the determinant vector
    for t in range(length):
        if((px_1[t]!=undefined) 
            and (px_2[t]!=undefined) 
            and (px_3[t]!=undefined)):
            
            temp = np.array([[px_1[t],py_1[t],pz_1[t]],
                             [px_2[t],py_2[t],pz_2[t]],
                             [px_3[t],py_3[t],pz_3[t]]])
            dv[t] = np.linalg.det(temp)
        else:
            dv[t] = undefined
    
    #return result
    return dv

def particle_component_sum(particle_t, particle_eta, particle_phi):
    '''
    Description: This function takes a particle's transverse 
momentum vector then outputs the sum of the component
        
    Arguments:
        
    particle_t - the transverse momentum of praticle

    particle_eta - the pseudorapidity of the transverse momentum of 
particle

    particile_phi - the azimuth angle of the transverse momentum of 
particle
    
    Output:
    
    1. the particle component sum
    
    '''   
    
    #get the length of particle
    length = len(particle_t)
    
    #create an array to store the information
    p_sum = np.zeros(length)
    
    #compute each component of the particle
    px, py, pz = compute_particle_component(particle_t, 
                                                  particle_eta, 
                                                  particle_phi)
    #find the valid index
    valid_index = (px!=undefined)
    
    #compute the sum
    p_sum[valid_index] = px[valid_index]
                         + py[valid_index]
                         + pz[valid_index]
                         
    p_sum[~valid_index] = undefined
    
    return p_sum

def compute_transverse_mass(particle_t_1, particle_eta_1,
                            particle_t_2, particle_eta_2):
    '''
    Description: This function takes two particles' transverse 
momentum vector then outputs the transverse mass
        
    Arguments:
        
    particle_t_1 - the transverse momentum of praticle 1

    particile_phi_1 - the azimuth angle of the transverse momentum of 
particle 1

    particle_t_2 - the transverse momentum of praticle 2

    particile_phi_2 - the azimuth angle of the transverse momentum of 
particle 2
    
    Output:
    
    1. the transverse mass of two particles
    
    '''

    #compute the length of particle 1 and particle 2
    length = len(particle_t_1)
    
    #create an array to store transverse mass
    mass = np.zeros(length)
    
    #find valid index
    valid_index = (particle_t_1!=undefined)&(particle_t_2!=undefined)
    
    #compute some temporary variabels
    px_1 = particle_t_1[valid_index]*np.cos(particle_phi_1[valid_index]
    py_1 = particle_t_1[valid_index]*np.sin(particle_phi_1[valid_index]
    px_2 = particle_t_2[valid_index]*np.cos(particle_phi_2[valid_index]
    py_2 = particle_t_2[valid_index]*np.sin(particle_phi_2[valid_index]
    
    #compute the mass
    mass[valid_index] = np.sqrt((particle_t_1[valid_index]
                                 +particle_t_2[valid_index])**2
                                 -(px_1+px_2)**2
                                 -(py_1+py_2)**2)
    #check the nan value and replace them
    length_mass = len(mass[valid_index])
    median = np.median(mass[np.isnan(mass)==0])
    for t in range(length_mass):
        if(np.isnan(mass[valid_index][t])==1):
            mass[valid_index][t] = median

    mass[~valid_index] = undefined
    
    return mass

def compute_invariant_mass(particle_t_1, particle_eta_1, particle_phi_1,
                           particle_t_2, particle_eta_2, particle_phi_2):
    '''
    Description: This function takes two particles' transverse 
momentum vector then outputs the invariant mass
        
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
    
    1. the transverse mass of two particles
    
    '''    
    
    
    