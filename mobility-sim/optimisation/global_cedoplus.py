#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Copyright INRIA. <lerela.inria -at- lio -dot- re>

#This software is governed by the CeCILL license under French law and
#abiding by the rules of distribution of free software. You can  use, 
#modify and/ or redistribute the software under the terms of the CeCILL
#license as circulated by CEA, CNRS and INRIA at the following URL
#"http://www.cecill.info". 

#As a counterpart to the access to the source code and rights to copy,
#modify and redistribute granted by the license, users are provided only
#with a limited warranty  and the software's author,  the holder of the
#economic rights,  and the successive licensors  have only  limited
#liability. 

#In this respect, the user's attention is drawn to the risks associated
#with loading,  using,  modifying and/or developing or reproducing the
#software by the user in light of its specific status of free software,
#that may mean  that it is complicated to manipulate,  and  that  also
#therefore means  that it is reserved for developers  and  experienced
#professionals having in-depth computer knowledge. Users are therefore
#encouraged to load and test the software's suitability as regards their
#requirements in conditions enabling the security of their systems and/or 
#data to be ensured and,  more generally, to use and operate it in the 
#same conditions as regards security. 

#The fact that you are presently reading this means that you have had
#knowledge of the CeCILL license and that you accept its terms.

from ampl import generate_temp_dat, generate_nl_file, execute_ampl, read_ampl

import numpy as np
import logging

logger = logging.getLogger(__name__)

def fmax_ampl(L, B, p, q, solver, real=False):
    """ Solves the global cedo plus problem with AMPL. Returns an np.array of the affectation."""
    
    params = {
        'L': L, # nb of nodes
        'B': B, # buffer size,
        'p': p, # delivery probabilities
        'n': len(q), # nb of contents
        'q': q, # query rates
        }

    tmp_dat = generate_temp_dat(params)
    
    output = execute_ampl(
        "global_cedoplus%s.mod" % (real and "_real" or ""), 
        solver,
        tmp_dat.name, 
       )
       
    o = output[0].decode('utf-8')
    x = read_ampl(o.split('\n'))
    
    tmp_dat.close()
    
    return np.array(x)
    
    
# (Non-working) deprecated SciPy routines

def global_cedoplus_func(x, q, r1, r2, laws, sign = 1.0):
    """ 
        CEDO global objective function.
        Sum of the DR = Sum of p_i q_i, with p_i     = 1 - exp(-lambda TTL n_i) 
                                                    = 1 - exp(-lambda TTL)**n_i
                                                    = 1 - law**n_i
        
        x: variable to optimize,
        q: the request rates,
        r: reference to range(0, size) (so that we don't recreate it at each time),
        law: P[Y>TTL] (optimization since we won't have to compute P[Y<=TTL]=1-P[Y>TTL]),
        law_param: lambda*TTL because the derivative will need it and the arguments must be the same,
        sign: a way to maximize (since scipy only minimizes).
    """
    t = 0
    p = 1
    for i in r1: # contents
        for k in r2: # nodes
            p *= 1 - x[i*len(r2) + k] * laws[i][k] 
        t += q[i]*(1-p)
    return sign*t
    
def global_cedoplus_func_deriv(x, q, r1, r2, laws, sign = 1.0):
    """ 
        Derivative of the CEDO global objective function.
        
        x: variable to optimize,
        q: the request rates,
        r: reference to range(0, size) (so that we don't recreate it at each time),
        law: P[Y>TTL] (optimization since we won't have to compute P[Y<=TTL]=1-P[Y>TTL]),
        law_param: lambda*TTL because it comes into the derivative,
        sign: a way to maximize (since scipy only minimizes).
    """
    t = 1
    p = 1
    l = []

    for i in r1: # contents
        for k in r2: # nodes
            p = q[i] * laws[i][k]
            for k2 in r2:
                if k2!=k:
                    p *= 1 - x[i*len(r2) + k2] * laws[i][k2] 
            l.append(p)
    return sign*np.array(l)  #sign*q[i]*(1-p)
                                
def fmax(q, L, B, laws, method = 'COBYLA'):
    """ 
        Maximizes the CEDO objective function. 
    
        q: is a (C, 1) np.array of the request rates.
        L: is the number of nodes.
        B: is the buffer capacity of each node.
        law_param: is the parameter lambda*TTL of the exponential law.
    
    """
      
    C = q.shape[0]
    LB = L*B # caching the value so we don't recompute it each time
    r1 = range(0, C) # to be reused several times
    r2 = range(0, L)
    r3 = range(0, L*C)

    constraints = []
    
    negative_x_array = -1 * np.ones(C)
              
          
    for i in r1:
        # max number of replicates lower than nodes count constraint
        constraints.append({'type': 'ineq',
          'fun' : lambda x:  x[i*L : (i+1)*L].sum() - L, # gammas matching all nodes for content i
          'jac' : lambda x: np.array([(x in range(i*L, (i+1)*L)) and -1 or 0 for x in r3])
          })
          
          # jacobian: -1 for involved gammas, ie gammas matching all the nodes of this content, 0 for the others
          # remember also that L=len(r2) and C=len(r1)

    
    for k in r2:
        contents_index = [i*L + k for i in r1]
        print(contents_index)
        constraints.append({'type':'ineq', 'fun':lambda x:  x[contents_index].sum() - B})
        constraints.append({'type': 'ineq',
          'fun' : lambda x:  B - x[contents_index].sum(), # gammas matching all contents for node k
          'jac' : lambda x: np.array([(x in contents_index) and -1 or 0 for x in r3])})

    # so basically, some of those constraints are just not respected. no error message, no warning, but the solver does not respect them.
    # note that SLSQP solver does not handle well this kind of constraints (cf. doc), so we use COBYLA

    #for j in r3:
    #    constraints.append({'type': 'ineq', 'fun': lambda x:- x[j]})
        
    #for j in r3:
    #    constraints.append({'type': 'ineq', 'fun': lambda x: x[j] -1})
        

    logger.info("Starting global cedoplus minimization with %s contents, %s nodes, buffer size %s, %s constraints." % (C, L, B, len(constraints)))
    
    #res = fmin_cobyla(global_cedo_func,
        #np.ones(C), 
        #constraints,
        #consargs=(),
        #args = (q, r, law, law_param, -1.0),
        #rhoend=1e-7,
        #catol=0.01
        #)
        
        
    #res = minimize(global_cedo_func, 
        #np.ones(C), 
        #args = (q, r, law, law_param, -1.0),
        #jac = global_cedo_func_deriv,
        #constraints = constraints,
        #bounds = [(1, L) for c in r],
        #method = method,
        #options={'disp': True}
        #)
        
        
    res = minimize(global_cedoplus_func, 
        np.zeros(C*L), 
        args = (q, r1, r2, laws, -1.0),
        #jac = global_cedo_plus_func_deriv,
        constraints = constraints,
        #bounds = [(0, 1) for c in range(0, L*C)],
        method = method,
        options={'disp': True,'maxiter':10000,'catol':.001,'tol':0.001,'maxfev':10000}
        )
        
    return res
