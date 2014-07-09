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

from stats import mean

import itertools as it


# Those tools cannot be used out of the box. They are simply useful as a basis for displaying methods.




def build_array_imt(l):
    """ Builds a list of list for plotting purposes. "Faster" nodes are close to the origin (faster meaning lower mean IMT). We can then display any metrics corresponding to a pair of nodes."""
    
    list_slow = sorted(l, key= lambda v: v.mean_imt())
    list_fast = sorted(l, key= lambda v: v.mean_imt(), reverse = True)
    result = []
    for slow_node in list_slow:
        line = []
        for fast_node in list_fast:
            if fast_node == slow_node:
                i = 0
            t = tuple(sorted([slow_node.pk, fast_node.pk]))
            try:
                # Number of IMT
                i = len(slow_node._im[t][1]) # 1 is the list of the IMT, 0 the timestamp of the last imt
                # Mean imt
                i = mean(slow_node._im[t][1])
            except KeyError:
                i = 0
            line.append(i)
        # Number of IMT: normalization to get a frequency
        sl = sum(line)
        result.append([i/sl for i in line])
    return results

def print_imt_pop_contents(list_nodes):
    """ One line per node, first item is node's mean IMT, then print the list of contents it owns sorted by popularity (most popular first). """
    
    # numbering the contents because we don't care about their real request rates, just their relative order
    for i, c in enumerate(lc): c.i = i
    
    for n in list_nodes:
        print(n.mean_imt(), end=" ")
        lcv = sorted(list(n._has_set), key=lambda c: c.q, reverse = True)
        for c in lcv:
            print("%s" % str(c.i+1), end=" "),
        print("")

def concatenate_results(d, f):
    """ Concatenates CEDO and CEDO+ results and write them to a file."""
    
    for a, b in it.zip_longest(d['result_cedo'], d['result_cedoplus_global']):
        f.write(bytes(" ".join([str(i) for i in (a.tolist() + list(b))]) + "\n", "utf-8"))
