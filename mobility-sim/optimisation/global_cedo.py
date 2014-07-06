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

from ampl import generate_temp_dat, execute_ampl, read_ampl

import numpy as np
import logging

logger = logging.getLogger(__name__)

def fmax_ampl(L, B, ndp, q, solver):
    """ Solves the global cedo problem with AMPL. Returns an np.array of the affectation."""
    
    params = {
        'L': L, # nb of nodes
        'B': B,  # buffer size
        'ndp': ndp, # exp(-lambda TTL)
        'n': len(q), # nb of contents
        'q': q, # query rates
        }

    tmp_dat = generate_temp_dat(params)
        
    output = execute_ampl(
        "global_cedo.mod", 
        solver,
        tmp_dat.name, 
       )
       
    x = read_ampl(output[0].decode('utf-8').split('\n'))
    
    #print(call_couenne(nl))
    
    tmp_dat.close()
    return np.array(x)

# SciPy routines, deprecated

def global_cedo_func(x, q, r, law, law_param, sign = 1.0):
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
    for i in r:
        t += q[i]*(1-law**x[i])
    return sign*t
    
def global_cedo_func_deriv(x, q, r, law, law_param, sign = 1.0):
    """ 
        Derivative of the CEDO global objective function.
        
        x: variable to optimize,
        q: the request rates,
        r: reference to range(0, size) (so that we don't recreate it at each time),
        law: P[Y>TTL] (optimization since we won't have to compute P[Y<=TTL]=1-P[Y>TTL]),
        law_param: lambda*TTL because it comes into the derivative,
        sign: a way to maximize (since scipy only minimizes).
    """
    l = []
    for i in r:
        l.append(q[i] * law_param * law**x[i])
    return sign*np.array(l)
                                
def fmax(q, L, B, law_param, method = 'SLSQP'):
    """ 
        Maximizes the CEDO objective function. 
    
        q: is a (C, 1) np.array of the request rates.
        L: is the number of nodes.
        B: is the buffer capacity of each node.
        law_param: is the parameter lambda*TTL of the exponential law.
            We need it (and not its exponential) because we'll have to 
            pass it to the objective function derivative.
    
    """
    
    #assert(law_param > 0)
    
    C = q.shape[0]
    LB = L*B # caching the value so we don't recompute it each time
    r = range(0, C) # to be reused several times

    law = np.exp(- law_param)
    
    constraints = []
    
    negative_x_array = -1 * np.ones(C)
    
    ## capacity constraint
    #constraints.append(lambda x: LB - x.sum())
    
    
    #for i in r:
        #jac = np.zeros(C)
        #jac[i] = 1
        
        ## max number of replicates constraint
        #constraints.append(lambda x: L - x[i])
         
         ## min number of replicates contraint
        #constraints.append(lambda x: x[i] - 1)
          
          
    
    # capacity constraint
    constraints.append({'type': 'ineq',
      'fun' : lambda x: LB - x.sum() ,
      'jac' : lambda x: negative_x_array})
    
    
    logger.info("Starting global cedo minimization with %s contents, %s nodes, buffer size %s, %s constraints." % (C, L, B, len(constraints)))

    res = minimize(global_cedo_func, 
        np.ones(C), 
        args = (q, r, law, law_param, -1.0),
        jac = global_cedo_func_deriv,
        constraints = constraints,
        bounds = [(1, L) for c in r],
        method = method,
        options={'disp': True}
        )

    # x is the solution
    return np.round(res.x) 

