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
 

"""
Process the cabspotting trace by gathering all the files in one and converting their coordinates.
"""
 
import os, sys, pickle
import mpl_toolkits.basemap.pyproj as pyproj
import scipy.interpolate
from numpy import interp
import threading, time

t_min, t_max = (sys.maxsize,0)
wgs84=pyproj.Proj("+init=EPSG:4326")
epsg3493=pyproj.Proj("+init=EPSG:3493")
d = {}
td = []

PATH = "/home/cabspottingdata/cabspottingdata/"

threadLock = threading.Lock()

def write_to_file(f,s):
	threadLock.acquire()
	f.write(bytes(" ".join(s), "UTF-8"))
	f.flush()
	threadLock.release()

c=0
for fn in os.listdir(PATH):
	if fn.startswith("new_"):
		c += 1
		with open(PATH + fn, "rb") as f:
			# 3 lists for each file
			t = ([], [], [])
			for raw_line in f:
				l = raw_line.split()
				# clean the data types
				#lon, lat, time
				line = [float(l[0]), float(l[1]), int(l[3])]
				if line[2] < t_min:
					t_min = line[2]
				if line[2] > t_max:
					t_max = line[2]
				x, y = pyproj.transform(wgs84, epsg3493, line[1], line[0])
				t[0].append(x)
				t[1].append(y)
				t[2].append(line[2])
			# saving the trace of this particular cab in the dic
			d[fn[4:]] = t


# empty list of lists, each list containing a "line" (v_id x y)
td = []

times = list(range(t_min, t_max))
times0 = list(range(0, t_max-t_min,10))
	
with open(PATH + "processed", "wb") as f:
	
	# working, but too slow
	"""for t in times:
		if t%100 == 0:
			print(t)

		results = []
		for i, l in d.items():
			results.append((i, interp(t, l[2], l[0]), interp(t, l[2], l[1])))

		for r in results:
			f.write(bytes("%d %s %f %f" % (t, r[0], r[1], r[2]), "UTF-8"))"""

	# working, but consuming too much memory
	"""for i, l in d.items():
		print(i)
		#times are decreasing so we need to reverse them
		l[2].reverse() #y
		l[1].reverse() #x
		l[0].reverse() #t

		# we interpolate x and y with all the times we want
		td.append((i, interp(times, l[2], l[0]).tolist(), interp(times, l[2], l[1]).tolist()))
		d[i]=None"""


	for i, l in d.items():
		# i is  cab id and l the tuple of the three lists
		
		#times are decreasing so we need to reverse the lists
		l[2].reverse() #y
		l[1].reverse() #x
		l[0].reverse() #t

		# we interpolate x and y with all the times we want
		interp_x = interp(times, l[2], l[0]).tolist()
		interp_y = interp(times, l[2], l[1]).tolist()

		s = []
		[s.extend([str(t+t_min), i, str(interp_x[t]), str(interp_y[t]), "\n"]) for t in times0]

		t = threading.Thread(target=write_to_file, args=(f,s))
		t.start()
	
	time.sleep(5) # nasty way to make sure everything has been written down
