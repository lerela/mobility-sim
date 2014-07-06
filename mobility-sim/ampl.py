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

from subprocess import Popen, PIPE

import numpy as np
import tempfile

import logging

logger = logging.getLogger(__name__)

AMPL_PATH = '/home/Outils/ampl/ampl'
MODELS_INDIR = "/home/Outils/mobility-sim/mobility-sim/models/"
MODELS_OUTDIR = "/home/Outils/mobility-sim/mobility-sim/models/compiled/"

# Part of this file has been borrowed somewhere... but where..? Sorry to the author.

def execute_ampl(model, solver, *files, **kwargs):
    """ Execute the AMPL model with the provided solver and data.
    
    Args:
        model: model name. the model will be fetched in MODELS_INDIR.
        solver: solver to use. since the problems are integer, the solver needs to know how to deal with them. we tested KNITRO and Couenne.
        files: data files to use.
    """
    
    logger.debug("Executing the ampl model %s with solver %s." % (model, solver))

    input = 'option solver %s;\n' % solver
    
    # If KNITRO, we want to use Intel MKL to speed it up
    if solver == 'knitro':
        input += 'option knitro_options "blasoption=1";\n' # ms_enable=1 ms_maxsolves=15 ms_savetol=1.0e-3";\n' 
    
    input += 'model ' + MODELS_INDIR + model + ';\n'
    
    for f in files:
        if f.endswith('.dat'):
            input += 'data ' + f + ';\n';
      
    if 'code' in kwargs:
        input += kwargs['code']
        
    # Compiling it. Not really needed anymore
    #nl = MODELS_OUTDIR + model
    #input += 'write g' + nl + ';\n';
    
    input += 'solve;'
    
    input += '_display x;'
    
    ampl = Popen([AMPL_PATH], stdin=PIPE, stdout=PIPE, shell = True)
    return ampl.communicate(bytes(input, "utf-8"))
    
def read_ampl(raw_output):
    header = None
    var = None
    data = []
    i = 0
    output = iter(raw_output)
    
    for l in output:
        
        if not l.startswith('_display'):
            continue
        header = l
        break
    
    if not header:
        return None
        
    command, nkeycols, ndatacols, nrows = header.split(' ')
    name = next(output)
    
    nkeycols = int(nkeycols)
    ndatacols = int(ndatacols)
    nrows = int(nrows)
    
    for l in output:
        if nrows <= 0:
            break
        data.append(l)
        nrows -= 1
        
    if nkeycols == 0:
        return float(data.rstrip("\n")) # TODO: convert to float optionally
      
    if nkeycols == 0:  
        result = []
    else:
        result = {}
        
        
    for line in data:
        values = line.split(",")
        if nkeycols == 1:
            key = int(values[0]) - 1
        else:
            key = tuple([int(i) -1 for i in values[:nkeycols]])
        if ndatacols == 1:
            value = float(values[nkeycols])
        else:
            value = None
        if nkeycols > 0:
            result[key] = value
        else:
            result.append(value)
    if ndatacols == 0:
        return result.keys()
    
    d = [0] * nkeycols

    for coordinates in result.keys():
        if isinstance(coordinates, int):
            if (coordinates + 1) > d[0]:
                d[0] = coordinates + 1
            continue
        
        for i, c in enumerate(coordinates):
            if (c+1) > d[i]:
                d[i] = c + 1
    arr = np.empty(d)
    for coordinates, value in result.items():
        arr[coordinates] = value
        
    return arr

def generate_dat_param(name, raw_value):
    """ Creates an ampl definition of the param :name with the value
    :raw_value, that can be a list or a value. """
    
    # We want to avoid scientific notation, hence the {:.10f}
    if isinstance(raw_value, (list, np.ndarray)):
        value = " ".join(["{0} {1:.10f}".format(i+1, q) for i, q in enumerate(raw_value)])
    else:
        value = str(raw_value)
        
    return "param {name} := {value};\n".format(name = name, value = value)

def generate_temp_dat(params):
    """ Generates a .dat file in a temporary file (physical file so it
    can be fed to ampl). Contains all the numerical values matching the 
    problem. """
    
    logger.info("Generating a temporary file with AMPL data.")
    f = tempfile.NamedTemporaryFile(suffix='.dat')

    for name, value in params.items():
        f.write(bytes(generate_dat_param(name, value), "utf-8"))
        
    f.seek(0) # back to the beginning before returning it
    return f

# Deprecated
def generate_nl_file(stub, model, *files, **kwargs):
    code = ''
    
    for key, value in kwargs.items():
        if key == 'code':
            code = value
        else:
            raise Exception('Invalid parameter: ' + key)
            
    input = 'model ' + MODELS_INDIR + model + ';\n'
    
    for f in files:
        f = f.replace('$', stub)
        if f.endswith('.dat'):
            input += 'data ' + f + ';\n';
      
    if 'code' in kwargs:
        input += kwargs['code']
        
    nl = MODELS_OUTDIR + stub
    
    input += 'write g' + nl + ';\n';
    ampl = Popen([AMPL_PATH], stdin=PIPE)
    ampl.communicate(bytes(input, "utf-8"))
    
    return nl
