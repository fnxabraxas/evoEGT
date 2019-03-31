#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:40:35 2019

@author: abraxas
"""

import evoEGT as evo

import numpy as np


def calcH(N,Z):   
# Input: N group size, Z population size
# Output: H[k,K] hypergeometric function (k individuals in group, K individuals in population)
    import numpy as np
    from scipy.stats import hypergeom  
    H=np.zeros((N+1,Z+1))
    H[0,0]=1  # H(0,:)=0, H(0,0)=1  Attention!
    for K in range(1,Z+1):         
        for k in range(0,N+1):
            H[k,K]=hypergeom.pmf(k,Z,K,N)
    return H

def aeps(pact,eps):
# Input: pact probability to perform the desired action (without error), eps probability of comitting an error 
# Output: action actually performed 
    return pact*(1.-2.*eps)+eps


def transfW2Wgen(Wori):
# transform WCD (Wori) into W for calcWpop (Wgen) 
    N1=Wori.shape[1]
    Wgen=np.zeros((2,2,N1,2))-777
    Wgen[1,0,:,:]=Wori[1,:,:]
    Wgen[0,1,:,:]=np.flip(Wori[0,:,:],axis=0)
    return Wgen



def calcWCD(N,eps,pF=0.5):
# Input: N group size, eps error when trying to perform an action, r multiplicative constant for the PGG (assuming c=1), pF probability of following leader
# Output: WCD[i,k,ip] payoffs (i=0 defector, i=1 cooperator; k number of cooperators in the group; ip coef associated to the parameter payoffs r (ip=0) and c (ip=1))
    WCD=np.zeros((2,N+1,2))
    eps1=1.-eps
    for k in range(0,N+1): # k number of cooperators
        benefit=( (k/N)*(        # leader is a cooperator
                        eps1       # leader cooperates
                        +(1.-pF)*((k-1)*eps1+(N-k)*eps)   # individuals do not follow and cooperate
                        +pF*(N-1)*(eps1**2+eps**2) # individuals follow the leader (she cooperates or defects) and cooperate
                        ) 
                +(1.-k/N)*(        # leader is a defector
                        eps             # leader cooperates
                        +(1.-pF)*((k)*eps1+(N-k-1)*eps)   # individuals do not follow and cooperates
                        +pF*(N-1)*+2.*eps1*eps  # individuals follow the leader (she defects or cooperates) and cooperate
                        ) )
        benefit=benefit/N       
        for i in [0,1]:    # i=1 cooperator, i=0 defector
            cost=( (1./N)*aeps(i,eps)     # focal is leader and cooperates
                + (1.-1./N)* (            # focal is not leader
                             (1.-pF)*aeps(i,eps)  # focal does not follow the leader and cooperates
                             +pF*( aeps((k-i)/(N-1),eps)*eps1 + aeps((N-k-1+i)/(N-1),eps)*eps )   # focal follows the leader (she cooperates or defects) and cooperates
                             ) )
            WCD[i,k,0]=benefit
            WCD[i,k,1]=cost
        WCD[1,0,:]=-999
        WCD[0,N,:]=-999
    return WCD 


def coop_pF_r(pFv,rv,N,HZ,beta,eps):
# Input: pFv, rv (vectors with values of pF and r), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)
    npF=len(pFv)
    nr=len(rv)
    M=np.zeros((npF,nr))
    for ipF in range(0,npF):
        pF=pFv[ipF]
        WCD=calcWCD(N,eps,pF=pF)
        Wgen=transfW2Wgen(WCD) # transforming to evoEGT format
        for ir in range(0,nr):
            r=rv[ir]
            print(ipF,ir,pF,r)
            SD,fixM = evo.Wgroup2SD(Wgen,H,[r,-1.],beta,infocheck=False)
            M[ir,ipF] = SD[1]
    return M

def plotCOOPheat(M,pFv,rv,label):
# Input: M (matrix from "coop_pF_r" function), pFv, rv (vectors with values of pF and r), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(6,6))
    h=ax.imshow(M,origin='lower', interpolation='none',cmap='hot',aspect='auto')
    nticksY=10
    nticksX=5
    ax.set_xticks(np.linspace(0, M.shape[1]-1, nticksX));
    ax.set_yticks(np.linspace(0, M.shape[0]-1, nticksY));
    ax.set_xticklabels(np.linspace(pFv[0],pFv[-1],nticksX));
    ax.set_yticklabels(np.linspace(rv[0],rv[-1],nticksY));
    ax.set_xlabel(r'$p_F$', fontsize=14)
    ax.set_ylabel(r'$r$', fontsize=14)
    cb=f.colorbar(h, fraction=0.1,format='%.2f')
    cb.set_label(label=r'$f_C$')
    f.savefig(label+'.eps',bbox_inches='tight',dpi=300)
    f.clf()     
    return


if __name__ == "__main__":

    import time

    t0=time.time()

#### One try ########################################    
    eps=0. #0.01
    Z=100
    N=10
    r=5
    beta=1.
    H=calcH(N-1,Z-1)
    payparam=np.array([r,-1.]) # assuming c=-1
    WCD=calcWCD(N,eps,pF=0.5)
    print('WCD')
    print(WCD[...,0])
    print(WCD[...,1])
    print(np.dot(WCD,payparam))
    #Wtotavg=calcWtotavg(WCD,N,Z)
    Wg=transfW2Wgen(WCD)
    SD,fixM=evo.Wgroup2SD(Wg,H,payparam,beta,infocheck=True)
    print('SD')
    print(SD)
    print('time spent: ',time.time()-t0)
#####################################################

####### Plot heatmap #########################################
#    eps=0. #0.01
#    Z=100
#    N=5
#    beta=1.
#    pFv=np.linspace(0.,1.,num=50)
#    rv=np.linspace(1.,2.*N,num=50)
#    
#    labfilenpy='coop_pF_r'
#    #----------------------------------------------------------------
#    M=coop_pF_r(pFv,rv,N,Z,beta,eps)  # calculate matrix for heatmap
#    np.save(labfilenpy,M)             # save matrix for heatmap
#    #----------------------------------------------------------------
#    
#    M=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
#    label='heatCD'
#    plotCOOPheat(M,pFv,rv,label)      # plot heatmap
#####################################################
    
    