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

from optimisation import global_cedo, global_cedoplus
from stats import mean
from statsmodels.distributions.empirical_distribution import ECDF
from utils import content_affectation, content_affectation_random, content_affectation_real, content_affectation_value, display_affectation, FixedSortedList

import logging
import numpy as np
import os
import pickle
import random
import sys

logger = logging.getLogger(__name__)

MARGIN = 15 # interval below which we consider that two vehicles have not actually been separated. FIXME: must be a parameter

class Trace(object):
    """Represent a trace file.
    The file must be filled with lines respecting this format:
        timestamp node_id x_coordinate y_coordinate
    Additional fields can be included but will be discarded.
    
    Args:
        filename: filename of the trace to fetch.
        nodes_number: number of nodes to fetch.
        distance: distance below which nodes are considered "in contact".
    """
        
    def __init__(self, filename, nodes_number = 150, distance = 50):
        
        self.filename = filename
        self.nodes_number = nodes_number
        self.distance = distance
        
        # Caching it to avoid eternal recomputation
        self.square_distance = distance*distance
        
        # Dictionary of node_id: node_object
        self.nodes = {}
        
        # List of content objects
        self.contents = []
        # Content rates in the same order than contents
        self.contents_rates = None
        
        # Dictionary of all the inter-meetings.
        # Format: (node1_id, node2_id): (last_meeting_timestamp, [IMT1, ...])
        # The key tuple is sorted by node id to avoid duplicates
        self._all_im = {}
        
        # List of epoch objects
        self._epochs = []
        
        # Dictionary of pairwise meeting probabilities
        # Format: (node1_id, node2_id): meeting probability
        self._probs = {}
    
    @property 
    def pickled_filename(self):
        """ Filename of the (pickled) compiled trace associated to these trace parameters. """
        
        return "{filename}_trace_d{distance}_L{nb_nodes}.pickle".format(
            filename = self.filename,
            distance = self.distance,
            nb_nodes = self.nodes_number)
    
    def setup(self):
        """ Loads the compiled trace if it exists, build it otherwise.
        
        Returns:
            A Trace object, either self or the loaded trace.
        """
        
        if os.path.exists(self.pickled_filename):
            logger.info("This trace exists on disk. Loading it.")
            try:
                return self.load()
            except Exception as e:
                logger.warning("Could not load the trace (%s). Resuming." % e)
        
        self.build_nodes(self.nodes_number)
        self.process()
        
        return self
    
    def build_contents(self, n):
        """Build contents and assign them random request rates according to a Zipf distribution. Normalize the request rates.
            
        Args:
            n: number of contents to build.
        """
                
        logger.info("Building %d random contents." % n)
        
        qs = np.random.zipf(2.,n)

        self.contents_rates = qs.astype(float) / qs.sum() # normalization
        
        logger.debug("Random popularity rates are: %s" % self.contents_rates)

        # Creating the contents, we want them to be in the same order than self.contents_rates    
        self.contents = [Content(i, q) for i, q in enumerate(qs)]

        logger.info("%d contents have been built." % len(qs))
    
    def build_nodes(self, limit):
        """Instanciate the list of nodes.
        
        Args:
            limit: maximum number of nodes to fetch. If None, all the trace is browsed (and it can take quite a long time).
        """
        
        logger.info("Building %s nodes from provided trace." % limit)
        
        with open(self.filename, "rb") as f:

            # Enumeration to limit the number of nodes
            for i, raw_line in enumerate(f):
                
                if i == limit:
                    break
                
                l = raw_line.strip().split(maxsplit = 2)
                
                node_id = l[1].decode('utf-8')
                
                try:
                    self.nodes[node_id]
                except KeyError: #not yet registered
                    node = Node(node_id, self._all_im)
                    self.nodes[node_id] = node
        logger.debug("%d nodes have been fetched." % i)
        
    def configure_nodes(self, method, TTL, buffer_size, reset = False):
        """Reset the nodes, set their type and compute their delivery probabilities.
        
        Args:
            method: cedo or cedoplus. Defines the node type and the way to compute the delivery probabilities.
            TTL: time-to-live.
            buffer_size: size of the clean and fresh buffer of the nodes.
            reset: also resetting the intermeeting information (so we need to re-run process afterwards).
        """
        
        if method.startswith("cedoplus"):
            node_type = "cedoplus"
        elif method.startswith("cedo"):
            node_type = "cedo"
        else:
            raise Exception("Node type associated to method {} is unknown.".format(method))
        
        # Resetting the contents affectation
        for c in self.contents:
            c.nodes = set([])
        
        # Resetting each node
        for n in self.nodes.values():
            n.reset(full = reset, clean = True, buffer_size = buffer_size)
            n.set_node_type(node_type)
        
        # Computing the probabilities
        self.proba_pairs(method, TTL)
        
        f = getattr(self, "parameter_%s" % method)
        return f(TTL)
    
    def process(self):
        """ Compile the trace by reading the trace file line by line. Find the nodes in reach and associate them by filling self._all_im and their respective ._im dictionaries. """
        
        logger.info("Starting to read the trace.")
        
        with open(self.filename, "rb") as f:
        
            current_time = None
            
             # list because we'll have to sort it
            nodes = list(self.nodes.values())

            # number of epochs (displaying purposes)
            e = 0 
                       
            # initialization of current_time (we don't want to process the intermeetings before the first epoch has been unrolled)
            first_line = next(f).strip().split()
            current_time = Epoch(int(first_line[0]))
            f.seek(0) # be kind rewind
            
            for raw_line in f:
                l = raw_line.strip().split()
                
                # clean the data types. 
                # 0 is epoch, 1 identifier, 2 x-coordinate, 3 y-coordinate. 
                line = [int(l[0]), l[1].decode('utf-8'), float(l[2]), float(l[3])]
                
                #change of epoch. we'll miss the last one though.
                if line[0] != current_time.timestamp: 
                    
                    current_time = Epoch(line[0])
                    self._epochs.append(current_time)
                    
                    # Arbitrary display. TODO: make it customizable
                    e += 1
                    if e % 1000 == 0:
                        logger.debug("Processing time %d.", current_time.timestamp)

                    #sorting by x to fasten processing (when (x1-x2)^2 > intercontact_distance^2, we can exit)
                    nodes.sort(key=lambda v: v.x)

                    # computing the intermeeting times for each node in this epoch
                    for i, node1 in enumerate(nodes):
                        
                        # not using node1, node2 in it.combinations(tmp_vehicules, 2):
                        # because it would yield all the combinations, and we try to be smart by breaking when two nodes start to be too far away

                        # iterations on the nodes pairs. no need to consider previous nodes because they have already considered the current one.

                        for j, node2 in enumerate(nodes[i+1:]): 
                            # 1-D euclidian distance. saving it to reuse it in the 2-D distance.
                            diff_x = (node1.x-node2.x)**2

                            if diff_x > self.square_distance: #then we're too far, since the list is sorted along the x axis we can break
                                break

                            #euclidian distance
                            elif (diff_x + (node1.y-node2.y)**2) < self.square_distance: # this is a match: the two nodes are meeting
                                
                                #logger.debug("%s and %s are meeting at time %d." % (node1, node2, current_time))
                                
                                node1.meet(node2, current_time.timestamp)
                                node2.meet(node1, current_time.timestamp)
                                current_time.append(node1, node2)

                # after we have switched epoch (or not), we can update the vehicules
                node_id = line[1]
                
                # We'll only consider the vehicules we've defined as such
                node = self.nodes.get(node_id)
                if not node:
                    continue
                
                node.update_pos(line[2], line[3])
    
    def run_simulation(self, TTL, distributed, request_rate = 50, save = False, replay = False):
        """ Run the simulation. 
        
        Args:
            TTL: time-to-live of the requests.
            distributed: a boolean specifying if we use the distributed algorithm or not (ie. if we the nodes buffer should be updated).
            request_rate: number of epochs between requests. since this is a random process, request_rate is just a mean.
            save: record the requests to replay the simulation later.
            replay: replay a saved simulation.
        
        Returns:
            List of statistics compiled during the simulation.

        """
        
        # Starting timestamp
        t = self._epochs[0].timestamp
        
        # A rate is easier to visualize but a frequency easier to use
        request_frequency = 1/request_rate
        
        # List to save the metrics we will regularly compute
        stats = []
                
        if save == replay == True:
            raise Exception("Impossible to save and replay a simulation at the same time.")
            
        if replay:
            logger.info("Replaying the simulation.")
        else:
            logger.info("Starting the simulation with request rate %s." % request_rate)
        
        # Utility list to pick a random content, caching it here to be accessed by all the nodes
        contents_rates_cumsum = self.contents_rates.cumsum()
        
        # Looping over the epochs
        for i, e in enumerate(self._epochs):
            
            # Saving stats every 100 epochs: arbitrary (FIXME)
            if i % 100 == 0:
                stats.append(self.stats())
            
            # Just some output: arbitrary (FIXME)
            if i % 25000 == 0:
                logger.debug("Processing time %d.", e.timestamp)
                #self.display_stats()
            
            for n in self.nodes.values():
                # Sending to the node the signal @new_epoch with the number of elapsed time units.                
                # On first iteration this difference will be zero, that's not a problem since new_epoch will just do nothing
                n.new_epoch(e.timestamp - t)
                
                # Generate or replay requests
                if replay:
                    if e in n.requests:
                        n.request_content(n.requests[e], TTL)
                        
                else:
                    # We want to avoid making requests all the time but we also want to avoid determinism, hence the use of request_frequency
                    r = random.random()
                    
                    if r <= request_frequency:
                        requested_content = n.make_request(
                            contents_rates_cumsum, 
                            self.contents,
                            TTL)
                            
                        if save:
                            n.requests[e] = requested_content
            
            # Now we can exchange contents between meeting nodes
            for n1, n2 in e.meetings:
                n1.transfer(n2, distributed)
                n2.transfer(n1, distributed)
            
            t = e.timestamp
               
        logger.info("Simulation ended.")
        self.display_stats()
        
        return stats
    
    def proba_pairs(self, method, TTL):
        """ Compute the inter-meeting probability between each pair of nodes. Since this is likely to be heavy, we'll cache the result on disk. 
        
        Args:
            method: cedo or cedoplus, only used to compute the cached probabilities filename. The nodes already have this attribute set and will use it during their computation.
            TTL: time-to-live of the requests.
        """
        
        # FIXME: this does not include the TTL
        path = self.pickled_filename + "_proba%s" % method
        if os.path.exists(path):
            logger.debug("Loading probabilities %s." % method)
            
            with open(path, "rb") as f:
                self._probs = pickle.load(f)
                
                # Reloading the probabilities in nodes memory
                for nodes, p in self._probs.items():
                    n1, n2 = nodes
                    self.nodes[n1]._probs[n2] = p
                    self.nodes[n2]._probs[n1] = p
                return
                
        logger.info("Building pair probabilities.")

        l = list(self.nodes.values())
        p = {}
        
        for i, n1 in enumerate(l):
            for j, n2 in enumerate(l[i:]):
        
                t = tuple(sorted([n1.pk, n2.pk]))
                    
                prob = n1.im_probability(n2, TTL)
                # We could assert that n1.im_probability(n2) == n2.im_probability(n1)
                                
                # @im_probability has saved this probability in the nodes memory, we just have to remember it in the trace
                p[t] = prob
                
        with open(path, "wb") as f:
            logger.debug("Saving probabilities %s." % method)
            pickle.dump(p, f)
    
        self._probs = p
    
    def parameter_cedo(self, TTL):
        logger.debug("Estimating CEDO parameter.")
        l = self._probs.values()
        m = mean(l)
        logger.debug("Overall CEDO delivery probability is %s." % m)
        for n in self.nodes.values():
            n.non_delivery_probabilities = 1-m
        return 1-m
                        
    def parameter_cedoplus(self, TTL):
        logger.debug("Estimating CEDO+ parameter.")
        for n in sorted(list(self.nodes.values()), key=lambda i: i.pk):
            p = n.set_probability_delivery_cedoplus(TTL)
            logger.debug("Probability delivery for node %s is %s." % (n, p))

    # Deprecated
    def parameter_cedo_mean(self, TTL):
        """ Computes the CEDO distribution parameter with the mean of all the IMTs. """
        raise Exception() #deprecated
        logger.debug("Estimating CEDO mean parameter.")
        
        imts = []
        
        for timestamp, l in self._all_im.values():
            imts.extend(l)
        
        m = imts and mean(imts) or sys.maxsize
        return self.apply_ndp(TTL, m)
    
    # Deprecated
    def parameter_cedo_meanmean(self, TTL):
        """ Computes the CEDO distribution parameter with the mean of the mean IMT per each pair of meeting nodes. """
        raise Exception() #deprecated
        
        logger.debug("Estimating CEDO mean of means parameter.")
        
        imts = []
        
        for timestamp, l in self._all_im.values():
            if l:
                imts.append(mean(l))
        
        m = imts and mean(imts) or sys.maxsize
        return self.apply_ndp(TTL, m)
    
    # Deprecated
    def apply_ndp(self, TTL, m):
        raise Exception() #deprecated
        
        # The delivery probability of one node is 1-exp(-\lambda TTL) and \lambda = 1/m.
        ndp = np.exp(- TTL / m)
        
        logger.debug("CEDO's NON delivery probability is %s." % ndp)

        for n in self.nodes.values():
            n.non_delivery_probabilities = ndp
        return ndp
    
    # Deprecated
    def parameter_cedoplus_precise(self, TTL):
        raise Exception() #deprecated
        for n in self.nodes.values():
            n.set_precise_probability_delivery(TTL, len(self.nodes))
        
    # Deprecated    
    def parameter_cedoplus_global(self, TTL):
        raise Exception() #deprecated
        for n in self.nodes.values():
            n.set_global_probability_delivery(TTL)
        
    def affectation(self, method, buffer_size, solver, param = None):
        """Launch the CEDO(+) problem computation and the following contents affectation.
        
        Args:
            method: cedo or cedoplus.
            buffer_size: buffer_size of the nodes (used in the optimization problem).
            solver: solver to use in AMPL.
            param: optional parameter exp(-\lambda TTL) to use in CEDO.
        """
        
        # Needed to get a deterministic order
        nodes = sorted(list(self.nodes.values()), key = lambda n: n.pk)
        
        # Resetting the contents. Should have been done in @configure_nodes anyway.
        for c in self.contents:
            c.nodes = set([])

        if method == "cedo":
            if not param:
                raise Exception("The exponential law parameter (Exp[-\lambda TTL]) must be defined for a CEDO affectation.")
                
            logger.info("Starting cedo optimisation.")
            
            # Computing the affectation. Saving it for later use.
            self.a = global_cedo.fmax_ampl(
                    len(self.nodes), 
                    buffer_size,
                    param,
                    self.contents_rates,
                    solver)
            
            # Randomly affect the nodes respecting the counts
            content_affectation_random(nodes,
                self.contents,
                self.a)
                            
            #display_affectation(nodes, self.contents)
            return content_affectation_value(self.contents)

        elif method == "cedoplus":
            logger.info("Starting cedo+ optimisation.")
            
            self.a = global_cedoplus.fmax_ampl(
                    len(self.nodes), 
                    buffer_size,
                    [1-v.non_delivery_probabilities for v in nodes],
                    self.contents_rates,
                    solver)
                    
            content_affectation(nodes, self.contents, self.a)
            
            #display_affectation(nodes, self.contents)
            return content_affectation_value(self.contents)
            
        elif method == "cedoplusreal":
            logger.info("Starting real cedo+ optimisation.")
            a = global_cedoplus.fmax_ampl(
                    len(self.nodes), 
                    buffer_size,
                    [1-v.non_delivery_probabilities for v in nodes],
                    self.contents_rates,
                    solver,
                    real = True)
            return content_affectation_real(nodes, self.contents, a)

        else:
            raise Exception("Unknown method %s." % method)
            
    def renact_affectation(self, method):
        """Use the saved assignation to reaffect contents to nodes. In the case of CEDO, this allows to randomly reaffect contents to nodes without having to solve again the optimization problem.
        """
        
        if not getattr(self, "a", np.array([])).any():
            raise Exception("Can't renact if no assignation has already been computed.")
            
        nodes = sorted(list(self.nodes.values()), key = lambda n: n.pk)
        
        if method == "cedo":
            content_affectation_random(nodes,
                self.contents,
                self.a)
        elif method == "cedoplus":
            content_affectation(nodes, self.contents, self.a)
        else:
            raise Exception("Can't renact method %s." % method)
            
        #display_affectation(nodes, self.contents)
        return   
            
    def stats(self):
        """ Compute some metrics that help us to compare the algorithms.
        
        Returns:
            Tuple of metrics:
                - number of requests.
                - number of satisfied requests.
                - number of pending requests.
                - number of unsatisfied requests.
                - mean of the delivery time.
                - number of distincts owned contents (should remained fixed).
        """
        
        DR, US, E, R, TR = (0,0,0,0,0)
        i = 0
        s = set([])
        
        for n in self.nodes.values():
            DR += n._fulfilled    
            US += n._unsatisfied
            E += n._pending
            R += n._requests_counter
            if n._fulfilled:
                i += 1
                TR += n._deliverytime / n._fulfilled
            s.update(n._has._set)
        
        if i:
            TR = TR/i
        
        return (R, DR, E, US, TR, len(s))
    
    def display_stats(self):
        """ Display and return some metrics to provide output.
        
        Returns:
            Dictionary of metrics.
            - number of pairs of nodes that have met.
            - number of intermeeting times that have been computed.
            - number of content replicates over all the nodes.
            - number of owned contents, should be the same that the previous.
            - number of distinct owned contents.
            - number of contents that are expected by the nodes.
        """
        
        s = set([])
        [s.update(n._has._set) for n in self.nodes.values()]
        
        ts = 0
        for n in self.nodes.values():
            ts += len(n._has) + len(n._initial)
        
        values = {
            'number of meeting pairs': len(self._all_im),
            'number of intermeeting times': sum([len(l[1]) for l in self._all_im.values()]),
            'number of content replicates': sum([len(c.nodes) for c in self.contents]),
            'number of owned contents': ts, 
            'number of distinct owned contents': len(s),
            'number of expected contents': sum([len(n._expects) for n in self.nodes.values()])
            }
        
        logger.info("==== Statistics ====")
        l = sorted(list(values.items()))
        for i in l:
            logger.info("%s: %s" % i)
            
        logger.info("{} requests ({} fulfilled, {} already existing and {} unsatisfied.".format(*self.stats()))
            
        return values
        
    def save(self):
        logger.info("Saving trace.")
        
        for c in self.contents:
            c.nodes = set([])
                
        with open(self.pickled_filename, "wb") as f:
            pickle.dump(self, f)
            
    def load(self):
        logger.info("Loading file %s." % self.pickled_filename)
        with open(self.pickled_filename, "rb") as f:
            t = pickle.load(f)
        return t

class Epoch(object):
    """Represent an epoch with the meetings that happened during it."""
    
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.meetings = []
    
    def append(self, n1, n2):
        t = tuple(sorted([n1, n2], key = lambda v: v.pk))
        self.meetings.append(t)
        
class Content(object):
    def __init__(self, pk, q, delivery_probabilites = {}):
        self.pk = pk
        self.q = q # query rate
        self.nodes = set([])
    
    def __str__(self):
        return "Content %s" % self.pk
    
    def __lt__(self, other):
        return self.q < other.q
    
    def request(self):
        """ Generate − or don't − a query of this content. """
        r = random.random()
        if self.q <= r:
            return True
        return False
    
    def non_delivery_probability(self):
        """
        Probability of not being delivered: P[Y>TTL] = Prod P[Y_i>TTL]. 
        For the general case, we let P[Y_i>TTL] depends of each node and each content.
        """
        non_delivery_prob = 1
        for n in self.nodes:
            non_delivery_prob *= n.non_delivery_probability_of_content(self)
        return non_delivery_prob
    
    def delivery_probability(self):
        return 1 - self.non_delivery_probability()
        
class Query(object):
    def __init__(self, content, TTL):
        self.content = content
        self.TTL = TTL
        self.initial_TTL = TTL # probably useless 
    
    def decrease_TTL(self, step = 1):
        """ Decreases the TTL and returns True if the query has expired. """
        self.TTL -= step
        if self.TTL == 0:
            return True
        return False
        
    def expired(self):
        return self.TTL < 0
    
    def lasted(self):
        return self.initial_TTL - self.TTL

class Node(object):
    def __init__(self, pk, all_im, buffer_size=0, non_delivery_probabilities=None, node_type=None):
        
        from utils import FixedSortedList # avoiding circular imports
        
        self.pk = pk # can be an int or a string or any unique identifier
        self._all_im = all_im # reference to a global im dictionary
        self._im = {} # dictionary of intermeetings.
            # Format: (node1_id, node2_id): (last_meeting_timestamp, [IMT1, …])
        self._probs = {}

        self.set_new_buffer(buffer_size)
        
        # can be a list or an int
        self.non_delivery_probabilities = non_delivery_probabilities
        
        self._expects = {}
        
        self.x = None
        self.y = None
        
        self._fulfilled = 0
        self._deliverytime = 0
        self._delivered = 0
        self._pending = 0
        self._unsatisfied = 0
        self._requests_counter = 0

        self.node_type = node_type

        self.requests = {} # save the random requests for later use
        
    def reset(self, clean = True, full = True, buffer_size = None):    
        """
            Resets the node.
            If :full, also reset the affectation.
            If :keep_im, do not destroy the intermeeting times.
        """
            
        from utils import FixedSortedList # avoiding circular imports

        self._expects = {}
        self.x = None
        self.y = None
        
        self._fulfilled = 0
        self._deliverytime = 0
        self._delivered = 0
        self._unsatisfied = 0
        self._pending = 0
        self._requests_counter = 0

        if full:
            self._im = {}
            self.requests = {}
        if clean:
            self.set_new_buffer(buffer_size or self.buffer_size)
            
    def __str__(self):
        return "Node %s" % self.pk
        
    def set_node_type(self, node_type):
        self.node_type = node_type
        self.compute_probability = getattr(self, "compute_probability_%s" % self.node_type)

    def set_new_buffer(self, buffer_size):
        """ Sets the buffer_size and resets the buffer. """
        
        self.buffer_size = buffer_size
        self._has = FixedSortedList(buffer_size)
        self._initial = FixedSortedList(buffer_size)
    
    def update_pos(self, x, y):
        """ Updates the node position. """
        self.x = x
        self.y = y
    
    def full(self, initial = False):
        if initial:
            return len(self._initial) >= self.buffer_size
        return len(self._has) >= self.buffer_size
    
    def deliver(self):
        """ Increases the number of delivered contents. """
        self._delivered += 1
    
    def fulfill(self, q):
        """ Increases the number of fulfilled requests. """

        try:
            self._expects[q.content].remove(q)
        except KeyError:  # can happen if the node autofulfills its request
            pass 
        else:
            if len(self._expects[q.content]) == 0:
                del self._expects[q.content]

        self._deliverytime += q.lasted()
        self._fulfilled += 1
        
    def unsatisfied(self, q):
        """ Increases the number of unsatisfied requests. """
        #assert(content in self._expects)
        
        self._expects[q.content].remove(q)
        if len(self._expects[q.content]) == 0:
            del self._expects[q.content]
        #raise Exception("unsatisfied:%s" % content.pk)

        self._unsatisfied += 1
    
    def expects(self, content):
        """ Returns True if the content belongs to the interests table. """
        return content in self._expects
        
    def has(self, content):
        """ Returns True if the content is in the buffer. """
        return content in self._initial or content in self._has
        
    def add(self, content, initial = False):
        """ Adds a content to the internal buffer if there's room.
        
        Args:
            content: Content object to add.
            initial: a boolean that chooses if we need to add the content to the internal buffer (meaning that we won't remove it during the content exchange process).
        
        Returns:
            A Content object that we couldn't cache (either :content or an other content from the buffer).
        """

        t = (self.compute_utility(content), content)
        
        if initial:
            # We want to make sure we are not gonna pop a content from this list
            assert(len(self._initial) < self.buffer_size)
            
            self._initial.add(t)
            
            # The content must know this node has cached it
            content.nodes.add(self)
            
            return

        u, expelled_content = self._has.add(t)

        # if expelled_content is the content we tried to add to the buffer, it could not make it, we can drop it (ie do nothing)
        
        if expelled_content and expelled_content != content:
            # we must tell expelled_content that we do not cache it anymore
            expelled_content.nodes.remove(self)
        
        # content has successfully been added to the heap: either because an other content has been kicked out or because nothing has been returned
        if expelled_content != content: 
            content.nodes.add(self)
        
        return expelled_content
    
    def new_epoch(self, step = 1):
        """Go through a new epoch: decreases the TTL of all the expected contents.
        
        Args:
            step: amount of units to be substracted from the TTL.
        """
        
        # list() is needed because we're gonna modify the dictionary self._expects and we can't do this if we're iterating of its values, so we have to work on a copy
        for lq in list(self._expects.values()):
            for q in lq:
                # we could not get the content on time so we drop the query
                if q.decrease_TTL(step): 
                    self.unsatisfied(q)
    
    def meet(self, node, timestamp):
        """ Computes IMT. Only considers IMT longer than MARGIN seconds (or units).
            
        Args:
            node: Node object that we're meeting.
            timestamp: timestamp of the meeting.
        """
        
        # we want a unique identifier for this pair of vehicules.
        # sorting to avoid duplicates
        t = tuple(sorted([node.pk, self.pk]))
        
        if not t in self._im: # first meeting, not much to do
            
            # so the other node might have already met us, let's check that 
            # before recreating an im list
            l = self._all_im.get(t)
            
            if not l: # the nodes have not yet met so we need to create a im list
                l = [timestamp, []]
                self._all_im[t] = l
                
            # double reference to the im list so that it's stored only once (in all_imt).
            # this way, when fetching for an existing meeting, we won't have to look through the entire all_im list (which is gonna be huge)
            self._im[t] = l 

        
        else: # we've already met, yey!
            
            last_timestamp, list_imt = self._im[t]
        
            # computing IMT
            imt = timestamp - last_timestamp
            
            if imt != 0: # easy check that will prevent an useless memory access if this
                # intermeeting has been processed by the other node
                if imt > MARGIN:
                    list_imt.append(imt)
                    
                self._im[t][0] = timestamp # updating the last timestamp of the meeting, will also update 
                    # all_im because we're modifying a list.
                    # update even if this is not considered as a new meeting because it does mean the two nodes
                    # are still in contact (and we're only interested in *inter*meeting times)

    def transfer(self, node, distributed):
        """Exchange content between self and the provided node.
        
        Args:
            node: a Node object to exchange content with.
            distributed: a boolean specifying if we use the distributed algorithm or not.
        """
        
        # checking our "interest table".
        # list() is needed because we're gonna modify the dictionary self._expects and we can't do this if we're iterating over it
        
        # lq is a list because we can have multiple queries toward a same content
        for c, lq in list(self._expects.items()):
            # Checking if the provided node has a content we expect
            if node.has(c):
                # Now we can satisfy all the queries for this content
                # list() is needed so that we can modify self._expects[c]
                for q in list(lq):
                    self.receive(q, distributed)
                
    def receive(self, q, distributed):
        """Method called when a content is received. We fulfill our request and we decide if we want to include it in our buffer if we're using the distributed algorithm.
        
        Args:
            q: Query object.
            distributed: a boolean specifying if we use the distributed algorithm or not (ie. if we should update our buffer).
        """
        
        #assert(self.expects(content)) # consistency

        # we increase our fullfilled counter
        self.fulfill(q) 
        
        # if we use the distributed algorithm, we want to try adding this content to our buffer
        
        if distributed:
            if not self.has(q.content):
                c = self.add(q.content)

                #logger.debug("Tried to add content %s to node %s. The content that was discarded was %s." % (content, self, c))
    
    def make_request(self, contents_rates_cumsum, contents, TTL):
        """ Pick a content to request and creates the corresponding query with @request_content.
        
        Args:
            contents: list of available Content objects.
            contents_rates_cumsum: list of cumulated sums of the content query rates for random choice.
            TTL: time-to-live of the request to create.
        
        Returns:
            The Content object that has been picked.
        """

        content = None
                
        # contents_rates_cumsum is a sorted list (because cumulated sums are increasing). we find the indice where we should insert the random number :r if we wanted to keep the list sorted. perfect way to pick a random point according to its probability to appear.
        r = random.random()
        i = contents_rates_cumsum.searchsorted(r) 
            
        content = contents[i] # works because qr_cumsum[-1] = 1 and r < 1
        
        self.request_content(content, TTL)

        return content    
    
    def request_content(self, content, TTL):     
        """Creates a query and add it to the node.
        
        Args:
            content: Content object to request.
            TTL: time-to-live of the request to create.
        """
        
        # This is a new request :)
        self._requests_counter += 1
        
        if self.has(content): # well, we got it, nothing to do
            # actually we want to remember this brillant success
            self._fulfilled += 1
        
        # We could ignore the requests of an already expected content. But it would mess all the statistics.
        #elif content in self._expects:
            # we are already expecting it, so we just reset the TTL
            #self._pending += 1
            #q = self._expects[content]
            #q.TTL = TTL
        
        # We can create the query
        else:
            q = Query(content, TTL) 
            l = self._expects.get(content, [])
            l.append(q)
            self._expects[content] = l
    
    def mean_imt(self):
        """ Mean of all the IMTs with other nodes. """
        
        list_imt = []

        for k, l in self._im.values():
            list_imt.extend(l)
        
        if not list_imt:
            return None
        
        return mean(list_imt)
    
    def mean_imt_pairs(self):
        """ Mean of the pairwise mean of the IMTs. """
        
        list_imt = []
        for k, l in self._im.values():
            list_imt.append(np.array(l).mean())
        
        return mean(array_imt)
    
    
    def non_delivery_probability_of_content(self, content):
        """Probability that *this* node will not deliver *this* content. This is NOT the global delivery probability of the content."""
        
        try:
            return self.non_delivery_probabilities[content]
        except (TypeError, IndexError): # not a list :)
            return self.non_delivery_probabilities
            
    def compute_utility(self, content):
        """ This method returns the (omniscient) utility of the provided content for this particular node.
        
        Interesting flow: it will usually need to fetch the existing probability of delivery for this content (content.(non_)delivery_probability), which in turn will call the (non_)delivery_probability_of_content of each node that has the content in its buffer.
        
        Args:
            content: the Content object we want to compute the utility of.
        """
        
        return getattr(self, "compute_utility_%s" % self.node_type)
        
    def compute_utility_cedo(self, content):
        """ In the CEDO case, the utility actually does not depend of the node: 
        U_i = q_i \lambda TTL Exp[-\lambda TTL n_i]
            
        We discard \lambda TTL because it has no impact in the utility ranking (same term for all contents). Exp[-\lambda TTL n_i] is actually the non delivery probability of the content, so let's ask it directly.
        """
        return content.q * content.non_delivery_probability()
    
    def compute_utility_cedoplus(self, content):
        """ In the CEDO+ case, the utility does also depend of the own node delivery probability of the content.
        """
        return content.q * content.non_delivery_probability() * (1 - self.non_delivery_probability_of_content(content))
    
    def im_probability(self, node, TTL):
        """Launch the computation of the meeting probability with the provided node and sets the value in the nodes _probs dictionary."""
        
        t = tuple(sorted([self.pk, node.pk]))

        timestamp, list_imt = self._im.get(t, (None, None))
        
        # if those nodes never met, their IM probability is zero
        if not list_imt:
            p = 0
        else:
            p = self.compute_probability(list_imt, TTL)
            
        self._probs[node.pk] = p
        node._probs[self.pk] = p
        
        return p
    
    def compute_probability_cedo(self, l, TTL):
        """Compute the probability of meeting a node before TTL units of time using the list of the IMTs with this node.
        
        Args:
            l: list of the IMTs between self and the node.
            TTL: time-to-live to use in the probability computation.
            
        Returns:
            1 - Exp[- TTL / mean(l)] because the parameter \lambda of the exponential law of this node's IMTs can be estimated by the inverse of the mean of the IMTs.
        """
        
        m = mean(l)
        dp = 1 - np.exp(- TTL / m)
        
        return dp
        
    def compute_probability_cedoplus(self, l, TTL):
        """Compute the probability of meeting a node before TTL units of time using the list of the IMTs with this node.
        
        Args:
            l: list of the IMTs between self and the node.
            TTL: time-to-live to use in the probability computation.
            
        Returns:
            TTL/m - \int_0^{TTL} F(l)/m as shown in our paper.
        """
        
        # To integrate numerically, we need to manually sample the space.
        ls = np.linspace(0,TTL,TTL*1000)
        
        # Estimated cumulative distribution function from the provided list of IMTs
        ecdf = ECDF(l)
        
        # Integrating it from 0 to TTL
        i = np.trapz(ecdf(ls), ls)
        
        # Computing the mean of these IMTs
        m = mean(l)
        
        return (TTL-i)/m
        
    def set_probability_delivery_cedoplus(self, TTL):
        l = list(self._probs.values())
        m = mean(l)
        self.non_delivery_probabilities = 1 - m
        return m
    
    def set_probability_delivery(self, parameter):
        raise Exception() #deprecated
        self.non_delivery_probabilities = 1 - parameter

    def set_precise_probability_delivery(self, TTL, L):
        """
            Sets the custom probability delivery of this node (used in CEDO+).
            Needs the TTL and the total number of nodes L.
            Computes F(x) = P(Y <= x) = 1/L \sum P(Y_i <= x) and computes this last term with the expression \frac{TTL}{m} - \frac{1}{m} \int_0^{TTL} F_i(x) \, \mathrm{d}x.
        """
        raise Exception() #deprecated

        list_imt = []
        ls = np.linspace(0,TTL,TTL*1000)
        
        # we set a list of probabilities (one for each content) but as of now we just set them to the same values since we can't distinguish probability delivery by contents
        #total = np.zeros(ls.shape)
        total = 0
        i = 0
        
        for timestamp, list_imt_pair in self._im.values():
            list_imt_pair = np.array(sorted(list_imt_pair))
            
            if not list_imt_pair.any():
                # might happen if the nodes only met once
                continue
            list_imt.append(list_imt_pair.mean())        
            i += 1

            
        # computing the cumulative distribution function
        ecdf = ECDF(list_imt)
        # integrating it from 0 to TTL
        i = np.trapz(ecdf(ls), ls)
        # computing the mean of these IMTs
        m = mean(list_imt)
        # adding the contribution of this pair to the overall probability
        total = TTL/m - i/m
        
        #if not i:
        #    sef.non_delivery_probabilities = 1
        #    return 1
        # 61 is weighted version, 60 the raw version
        delivery_probability =  total 
        self.non_delivery_probabilities = 1 - delivery_probability
        return self.non_delivery_probabilities
        
    def set_global_probability_delivery(self, TTL):
        """
            Sets the custom probability delivery of this node (used in CEDO+).
            Needs the TTL and the total number of nodes L.
            Computes F(x) = \frac{TTL}{m} - \frac{1}{m} \int_0^{TTL} F(x) \, \mathrm{d}x with all the IMTs.
        """
        raise Exception() #deprecated
        list_imt = []
        ls = np.linspace(0,TTL,TTL*1000)
        
        for timestamp, list_imt_pair in self._im.values():
            list_imt.extend(list_imt_pair)
        
        if not list_imt:
            self.non_delivery_probabilities = 1
            return self.non_delivery_probabilities
            
        list_imt = np.array(sorted(list_imt))
        ecdf = ECDF(list_imt)
        i = np.trapz(ecdf(ls), ls)
        m = list_imt.mean()
        self.non_delivery_probabilities = 1 - (TTL/m - i/m)
        return self.non_delivery_probabilities
