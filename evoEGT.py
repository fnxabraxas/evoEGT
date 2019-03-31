#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:11:51 2019

@author: abraxas
"""

import numpy as np


def calcH(N,Z):   
# Input: N group size, Z population size
# Output: H[k,K] hypergeometric function (k individuals in group, K individuals in population)
    import numpy as np
    from scipy.stats import hypergeom  
    H=np.zeros((N+1,Z+1))
    for K in range(0,Z+1):         
        for k in range(0,N+1):
            H[k,K]=hypergeom.pmf(k,Z,K,N)
    return H


def calcWpop(Wgroup,HZ,info=True):
# Input: Wgroup[i,j,k,ip] coeficients that will be multiplied by payparam to obtain the payoff that i-type obtains agains j-type when there is k of i-type in the group (k=0..N)
# Input: HZ if it is a 2-dim array HZ=H (matrix from hypergeometric function H[k,K], k=0..N-1, K=0..Z-1), if it is a scalar HZ=Z and H is calculated    
# Input: info shows some information
# Output: Wpop[i,j,K,ip] coeficients that will be multiplied by payparam to obtain the payoff that i-type obtains agains j-type when there is K of i-type in the population
    import sys
    shape=list(Wgroup.shape)
    N=shape[2]-1
    if info: print('N=',N)
    if np.isscalar(HZ):
        Z=HZ
        if info: print('Z=',Z,' ==>  Calculating H...')
        H=calcH(N-1,Z-1)  # H[k,K] probability of taken k individuals in a group if there is K individuals of that type in the population
    elif len(HZ.shape)==2:
        H=HZ
        if((H.shape[0]-1)!=(N-1)):
            sys.exit('Error: H should have dimensions N x Z')
        else:
            Z=H.shape[1]
            if info: print('Read H ==>  Z=',Z)
    else:
        sys.exit('Error: HZ should be either Z (scalar) or H (dim=2)')
    shape[2]=Z+1
    Wpop=np.zeros(shape)
    Wpop[:,:,0,:]=-888      # flag for the impossible case of no focal type in the population
    for K in range(1,Z+1):
        for k in range(np.max([1,N-Z+K]),np.min([K,N])+1):  # k number of i-type individuals in the group
            Wpop[:,:,K,:]=Wpop[:,:,K,:]+ H[k-1,K-1]*Wgroup[:,:,k,:]  # k-1, K-1 because the focal player (cooperator) is excluded   
    return Wpop


def calcFIXM(W,payparam,beta,check=True):
# Input: W[i,j,K,ip] coeficients that will be multiplied by payparam to obtain the payoff that i-type obtains agains j-type when there is K of i-type in the population (K=0..Z)
# Input: payparam[ip] payoff parameters of the model, selection strength for Fermi, check check that inputs are coherent and show information about it
# Output: fixM probability transition matrix (calculated with fixation probabilities and Fermi rule), from row to columns
# CAREFUL: this function is only valid assuming that there are two strategies in the population (since it uses fixation probabilities)
    import sys
    payparam=np.array(payparam)
    nstr=W.shape[0]
    fixM=np.zeros((nstr,nstr))  
    Z=W.shape[2]-1
    if check:
        print('Z= ',Z)
        print('Number of strategies: ', nstr)
        if W.shape[3]==len(payparam): 
            print('Number of payoff parameters: ', len(payparam))
        else:
            sys.exit('Error: len(W.shape-3)!=len(payparam)')
    D=(W-np.flip(W,2).transpose((1,0,2,3)))[:,:,1:Z,:]  # W_i-W_j for K of i;  flip (take W for j strategy that has N-K in the population); K from 1 to Z-1
    cumD=np.cumsum(D,axis=2)
    fixM=1./(1.+np.sum(np.exp(np.dot(cumD,beta*payparam)),axis=2))
    fixM=fixM/nstr # it can be nstr-1, but not much difference    
    np.fill_diagonal(fixM, 0) # necessary?
    np.fill_diagonal(fixM,1.-np.sum(fixM,axis=1))   
    return fixM    


def calcSD(tranM):
# Input: tranM probability transition matrix (transitions from row to column)    
# Output: SD stationary distribution
    from discreteMarkovChain import markovChain
    mc=markovChain(tranM)
    mc.computePi('eigen') # We can use 'linear', 'power', 'krylov' or 'eigen'
    SD=(mc.pi).reshape(-1,1)
    return SD


def Wgroup2SD(Wgroup,HZ,payparam,beta,infocheck=True):
# Input: Wgroup[i,j,k,ip] coeficients that will be multiplied by payparam to obtain the payoff that i-type obtains agains j-type when there is k of i-type in the group (k=0..N)
# Input: HZ if it is a 2-dim array HZ=H (matrix from hypergeometric function H[k,K], k=0..N-1, K=0..Z-1), if it is a scalar HZ=Z and H is calculated    
# Input: payparam[ip] payoff parameters of the model, selection strength for Fermi
# Input: infocheck check that inputs are coherent and show information about the parameters
# Output: fixM probability transition matrix (calculated with fixation probabilities and Fermi rule), from row to columns 
    Wpop=calcWpop(Wgroup,HZ,info=infocheck) 
    fixM=calcFIXM(Wpop,payparam,beta,check=infocheck)
    SD=calcSD(fixM)
    return SD, fixM