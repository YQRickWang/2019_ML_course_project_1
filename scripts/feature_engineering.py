# -*- coding: utf-8 -*-
"""some functions to be used to derive features."""

import numpy as np

#Compute Transverse Mass (need to be modifed)
def transverse_mass(a_t,a_phi,b_t,b_phi):
    
    
    mass = np.sqrt((a_t+b_t)**2-(a_t*np.cos(a_phi)+b_t*np.cos(b_phi))**2-(a_t*np.sin(a_phi)+b_t*np.sin(b_phi))**2)
    
    
    return mass

#Compute Invariant mass
def invariant_mass(a_t,a_eta,a_phi,b_t,b_eta,b_phi):
    a_z = a_t*np.sinh(a_eta)
    b_z = b_t*np.sinh(b_eta)
    
    a_xyz = np.sqrt(a_t**2+(a_z)**2)
    b_xyz = np.sqrt(b_t**2+(b_z)**2)
    ab_x = a_t*np.cos(a_phi)+b_t*np.cos(b_phi)
    ab_y =a_t*np.sin(a_phi)+b_t*np.sin(b_phi)
    ab_z =a_z+b_z
    
    
    mass = np.sqrt((a_xyz+b_xyz)**2-(ab_x)**2-(ab_y)**2-(ab_z)**2)
    
    return mass

#Compute Modulus of Vector Sum
def modulus_vector(a_t,a_phi,b_t,b_phi,c_met,c_phi):
    
    x = a_t*np.cos(a_phi)+b_t*np.cos(b_phi)+c_met*np.cos(c_phi)
    y = a_t*np.sin(a_phi)+b_t*np.sin(b_phi)+c_met*np.sin(c_phi)
    
    
    p_t = np.sqrt(x**2+y**2)
    
    modulus = p_t*1.0
    return modulus


#Compute Pseudorapidity Separation
def pseudorapidity_separation(jet_num,leading_eta,subleading_eta):
    
    
    N = len(leading_eta)
    sep = np.zeros(N)
    for t in range(N):
        if(jet_num[t]<=1):
            sep[t] = -999
        else:
            sep[t] = np.abs(leading_eta[t] - subleading_eta[t])
    return sep
