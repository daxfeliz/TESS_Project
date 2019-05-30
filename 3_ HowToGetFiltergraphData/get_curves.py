import os
from os import listdir
from os.path import isfile, isdir
import numpy
import time
import sys

filename=str(sys.argv[-1]) 
#^^^ INPUT FILE downloaded from filtergraph (ASCII, space delimited)
#    Example: Sector_01.txt

#read in the list
f = open(filename, 'r') 
lines = f.readlines()
f.close()
nstars = len(lines)

#make an output directory if it doesn't exist
if (os.path.isdir('curves') == 0):
	os.system('mkdir curves')
if (os.path.isdir('curves/clean/') == 0):
	os.system('mkdir curves/clean/')
if (os.path.isdir('curves/raw/') == 0):
	os.system('mkdir curves/raw/')
	
print ('Getting light curves...')

for ii in range(0,nstars-1):
	pts = lines[ii].split(' ')
	nme = pts[0]+'_'+pts[1]+'_'+pts[2]+'_'+pts[3]+'.lc'

	if (os.path.isfile('curves/clean/'+nme) == 0):
		os.system('wget http://astro.phy.vanderbilt.edu/~oelkerrj/tess_ffi/'+pts[1]+'/clean/'+nme+' -O curves/clean/'+nme)
	if (os.path.isfile('curves/raw/'+nme) == 0):
		os.system('wget http://astro.phy.vanderbilt.edu/~oelkerrj/tess_ffi/'+pts[1]+'/raw/'+nme+' -O curves/raw/'+nme) 

	if (ii % 100) == 0:
		print ('Working on the next 100 curves at '+time.strftime("%c"))

print ('Finished getting light curves at '+time.strftime("%c"))
