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

import heapq
import numpy as np

from subprocess import Popen, PIPE

import itertools as it
import logging
import pickle
import random

logger = logging.getLogger(__name__)

class FixedSortedList(object):
    """ A sorted list of fixed size (using heapq internally)."""
    
    def __init__(self, max_length, h = []):        
        self.max_length = max_length
        
        # taking the :max_length highest values of :h.
        self._heap = sorted(h, key=lambda k: k[0], reverse=True)[:max_length]
        heapq.heapify(self._heap)
        self._set = set([])
        
        if self._heap:
            self._current_len = len(self._heap)
        else:
            self._current_len = 0

        self._pos = -1 # iterator position
    
    def add(self, item):
        """ Adds item to the heap. If the heap is full, returns the item that couldn't make it. In general, this item is NOT the one that was provided to the method. """ 
    
        # Python's heap queues are sorted by increasing order so we just have to fill the queue 
        # And when we reach its maximum size, we can pushpop to keep the highest value and expell the lowest
                
        if self._current_len >= self.max_length:
            c = heapq.heappushpop(self._heap, item)
            self._set.add(item[1])
            self._set.remove(c[1])
        else:
            heapq.heappush(self._heap, item)
            c = (None, None)
            self._set.add(item[1])
            self._current_len += 1
        return c

    
    def __contains__(self, item):
        """ Check for item in the second element of the tuples."""
        return item in self._set
        
        for i in self._heap:
            if i[1] == item: return True
        return False

    def sort(self, reverse = False):
        """ Will break the heap if reverse is True! """
        self._heap.sort(reverse = reverse)

    def pop_last(self):
        self.sort() # heapq does not keep the heap sorted!
        r = self._heap.pop() #python pop: last element
        self._current_len -= 1
        self._set.remove(r[1])
        return r
 
    def pop(self):
        r = heapq.heappop(self._heap) # heapq pop: first element
        self._current_len -= 1 # we only decrease length if we've successfully poped the heap
        self._set.remove(r[1])
        return r

    def replace(self, e):
        return heapq.heapreplace(self._heap, e)
    
    def __len__(self):
        return self._current_len
 
    def __iter__(self):
        self._pos = -1
        return self
 
    def __next__(self):
        self._pos += 1
        if self._pos == self._current_len:
            raise StopIteration
        return self._heap[self._pos] # we don't catch indexerror because if there is, we're in an inconsistent state and we
            # prefer to be made aware (-:

def content_affectation_random(nodes, contents, counts):
    """ Randomly affects the contents to the nodes, respecting the counts. """
    
    # copying the list to avoid modifying it
    nodes = [n for n in nodes]
    
    # coherence check
    assert(len(counts) == len(contents))

    ch = [(content, count) for content, count in it.zip_longest(contents, counts)]
    
    # we need to fill the nodes with the most popular contents first to avoid starving... but even that might not work (careful)
    ch = sorted(ch, key=lambda i: i[1], reverse = True)

    logger.info("Starting random affectation of content. %d nodes and %d contents, %d replicates." % (len(nodes), len(contents), sum(counts)))
    
    t = 0
    for content, count in ch:
        
        v = True # just to get in the loop
        s = set([])
        
        while count > 0:
            n = set(nodes)
            if n.issubset(s): # all the remaining nodes have been seen by this content
                logger.warning("Ooch, {} cop(y|ies) of content {} could not be assigned.".format(count, content))
                break

            try:
                v = random.choice(nodes)
            except IndexError: # should only happen once or two (because we've rounded up the affectation so there might
                    # be some side effects)
                logger.warning("Oops, couldn't affect as much content as needed... %s remained." % count)
                break
            else:
                s.add(v)
            
            if content not in v._initial._set:
                if not v.full(initial = True):
                    v.add(content, initial = True)
                    t += 1
                    count -= 1
                else:
                    logger.debug("%s is full (%d contents), dropping it from the candidate nodes list." % (v, len(v._has)))
                    nodes.remove(v) # no need to keep a full node in the list
    
    logger.debug("%s replicates set." % sum([len(c.nodes) for c in contents]))
    return contents

def content_affectation(nodes, contents, affectations):
    """ Affect the contents to the nodes in a deterministic manner.
    
    Args:
        affectations: a np.array with a line per content and a column per node."""
    
    L = len(nodes)
    C = len(contents)
    
    # coherence check
    #assert(len(affectations) == L*C)
    
    #logger.info("Starting deterministic affectation of content. %d nodes and %d contents, %d replicates." % (L*C, len(affectations)))
    
    for i, content in enumerate(contents):
        for k, node in enumerate(nodes):
            if affectations[i, k] > 0.5: # AMPL might return non exact values
                node.add(content, initial = True)

    return contents
    
def content_affectation_real(nodes, contents, affectations):
    """ Do not really affect the contents but use real number in the affectation to compute a mean of the IMTs of the affected (weigthed) nodes."""
    
    L = len(nodes)
    C = len(contents)
    
    # coherence check
    #assert(len(affectations) == L*C)
    
    content_affectation = []
    
    for i, content in enumerate(contents):
        m = 0
        r = 0
        for k, node in enumerate(nodes):
            mi = node.mean_imt()
            if not mi:
                #logger.info("Mean imt of node %s was None. Affectation was %s." % (node.pk, affectations[i, k]))
                pass
            else:
                m += affectations[i, k] * node.mean_imt()
            r += affectations[i, k]
                
        content_affectation.append((content.q, r, m/r))
                            
    return content_affectation

def content_affectation_value(contents):
    """A metric for the affectation.
    
    Returns:
        A list with a tuple per content that holds its popularity, the number of nodes that have cached it, and the mean, min, and max IMTs of the nodes that host the contents.
    """
    
    content_affectation = []
    
    for i, content in enumerate(contents):
        m = 0
        l = []

        for v in content.nodes:
            mi = v.mean_imt()
            
            # not really correct: if mean_imt is None it means that the node has never encountered anybody so its mean should be infinite, but then it would crush the rest...
            if mi:
                l.append(mi)

        r = len(content.nodes)
        
        # Content is not cached, weird... but useless to compute its stats
        if not r:
            continue
        
        m = sum(l)/r
        
        content_affectation.append((content.q, r, m, l and min(l), l and max(l)))
        
    return content_affectation

def display_affectation(nodes, contents):
    """ Displays the affectation as returned by AMPL. """
    
    for n in sorted(nodes, key=lambda i: i.pk):
        l = []
        for c in contents:
            if n.has(c):
                l.append("1")
            else:
                l.append("0")
        print (" ".join(l))


def save_results(filename, **kwargs):
    """ Pickle the results in a file whose name is derivated from the provided filename and the different kwargs that are either str, int or float."""
    
    # sorting so that all filenames have the same pattern
    items = sorted(list(kwargs.items()), key=lambda i: i[0])
    
    # Generating the filename
    fn = [filename]
    for name, arg in items:
        if type(arg) in [str, int, float]:
            fn.append("%s%s" % (name, arg))
    
    with open("_".join(fn) + ".pickle", "wb") as f:
        
        if "contents" in kwargs:
            for c in kwargs['contents']:
                c.nodes = set([]) # avoid stupid recursion
                
        pickle.dump(kwargs, f)


# Deprecated
def call_couenne(nl):
    logger.info("Calling couenne with file %s" % nl)
    ampl = Popen(['couenne', nl], stdin=PIPE, stdout=PIPE)
    result = ampl.communicate()
    return result
