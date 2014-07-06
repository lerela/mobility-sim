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


from models import Trace
from utils import save_results, content_affectation_random

import argparse
import itertools as it
import logging
import numpy as np
import random
import stats
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

random.seed()

# For pickle, just in case
sys.setrecursionlimit(50000)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="Trace")
    parser.add_argument('-d', '--distance', type=int, required = False, default=50)
    parser.add_argument('-t','--ttl', type=float,
        help = "Time to live (in seconds). Embedded in the parameter of the law but needed for the simulation.")
    parser.add_argument('-b','--buffer_size', type=int,
        help = "Buffer capacity of one node.", default = 10)
    parser.add_argument('-c','--contents', type=int, 
        help = "Number of contents", default = 70)
    parser.add_argument('-n','--nodes', type=int, 
        help = "Maximum number of nodes to process", required = False)
    parser.add_argument('-r','--cedoruns', type=int, 
        help = "Number of cedo runs to average", default = 5)
    parser.add_argument('-s','--solver', type=str, 
        help = "AMPL solver to use (must be in PATH)", 
        default = 'couenne',
        required = False)
    parser.add_argument('-a', action="store_true", help="Run the distributed algorithm.")
    args = parser.parse_args()

    trace = Trace(args.filename, args.nodes, args.distance)
    trace = trace.setup()
    trace.save()
    
    trace.build_contents(args.contents)  
    #trace.display_stats()
    
    logger.info("Cedo+")
    trace.configure_nodes("cedoplus", args.ttl, args.buffer_size)
    a_cedoplus_global = trace.affectation("cedoplus", args.buffer_size, args.solver)
    r_cedoplus_global = trace.run_simulation(args.ttl, args.a, save = True)
    
    # We want to average a few CEDO runs for an accurate comparison    
    r_cedos = []
    for i in range(0, args.cedoruns):
        logger.info("Regular cedo %s" % i)
        param = trace.configure_nodes("cedo", args.ttl, args.buffer_size)
        a_cedo = trace.affectation("cedo", args.buffer_size, args.solver, param)
        r_cedos.append(np.array(trace.run_simulation(args.ttl, args.a, replay = True)))
    
    r_cedo = np.zeros(r_cedos[0].shape)
    for r in r_cedos:
        r_cedo += r
    r_cedo = r_cedo / args.cedoruns
    
    save_results(args.filename, 
        B=args.buffer_size, 
        C=args.contents,
        L=args.nodes,
        TTL=args.ttl, 
        d=args.distance,
        a=args.a and "dis" or "glo",
        affectation_cedo = a_cedo,
        affectation_cedoplus_global = a_cedoplus_global,
        result_cedo = r_cedo,
        result_cedoplus_global = r_cedoplus_global,
        )
