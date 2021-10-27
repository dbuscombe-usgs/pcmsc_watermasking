
# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os, json, exifread, sys, getopt
import numpy as np
from tkinter import filedialog
from tkinter import *
from skimage.io import imsave, imread
from glob import glob 
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

#====================================================
def main(f, i, settings):
	#print(f)
	#file = 'sample'+os.sep+'CAM432_20170125222240_40.npz'
	# load file
	dat = np.load(f, allow_pickle=True)

	#create a dictionary of variables
	data = dict()
	for k in dat.keys():
		try:
			data[k] = dat[k]
		except:
			pass
	del dat

	#print(data['meta'].item())

	if settings['make_mask']>0:
		out_direc = os.path.normpath(settings['SAMPLE_DIREC']+os.sep+ 'masks_'+str(settings['make_mask']))
		outfile = out_direc + os.sep+ data['meta'].item()['image_filename'].replace('.jpg','_mask.jpg')
		exec("tmp=data['mask"+str(settings['make_mask'])+"']") 
		imsave(outfile,255*tmp.astype('uint8'),quality=100)
	
	elif settings['make_prob_map']==1:
		out_direc = os.path.normpath(settings['SAMPLE_DIREC']+os.sep+ 'prob_stack')
		outfile = out_direc + os.sep+ data['meta'].item()['image_filename'].replace('.jpg','_probstack.jpg')
		imsave(outfile,np.dstack((data['prob_land'],data['conf'],data['var'])).astype('uint8'),quality=100)
		#yellow = high prob land , high confidence, low variability
		#green = low prob of land, high confidence, low variability
		#purple = high prob land, low confidence, high variability
		#blue = low prob land, low confidence, high variability
		#red = high probability of land, low confidence, low variability
		
	if settings['make_all_masks']==1:
		for k in np.arange(1,7):
			out_direc = os.path.normpath(settings['SAMPLE_DIREC']+os.sep+ 'masks_'+str(k))
			outfile = out_direc + os.sep+ data['meta'].item()['image_filename'].replace('.jpg','_mask.jpg')
			try:
				del tmp
			except:
				pass
			#exec("tmp=data['mask"+str(k)+"']") 
			var = 'mask'+str(k)
			tmp = data[var]
			imsave(outfile,255*tmp.astype('uint8'),quality=100)	
			del tmp

	if settings['make_overlays']==1:
		for k in np.arange(1,7):
			out_direc = os.path.normpath(settings['SAMPLE_DIREC']+os.sep+ 'overlays_'+str(k))
			outfile = out_direc + os.sep+ data['meta'].item()['image_filename'].replace('.jpg','_overlay.png')
			try:
				del tmp
			except:
				pass
			#exec("tmp=data['mask"+str(k)+"']") 
			var = 'mask'+str(k)
			tmp = data[var]
			im=imread(i, as_gray=False)
			plt.imshow(im); plt.axis('off')
			plt.imshow(255*tmp.astype('uint8'),cmap='bwr', alpha=0.25); plt.axis('off')
			plt.savefig(outfile, dpi=100,bbox_inches='tight')
			plt.close('all')
			del tmp
			
	del data

#====================================================
###==================================================================
#===============================================================
if __name__ == '__main__':

	argv = sys.argv[1:]
	try:
		opts, args = getopt.getopt(argv,"h:m:c:p:a:o:")
	except getopt.GetoptError:
		print('python parse_watermask.py -m mask mode -c make probability stack -p parallel processing -a store all masks -o make overlay images\n')
		print('Defaults:\n')
		print('-m  = 0 (0=dont print a mask file, otherwise number indicates degree of liberalism; 1=most conservative, 6=most liberalk)\n')
		print('-c  = 1 (print probability stack, alt: 0=dont)\n')						
		print('-p  = 1 (use parallel proc, alt: 0=dont use parallel proc)\n')		
		print('-a  = 1 (print all 6 masks in npz file, alt: 0=dont)\n')			
		print('-o  = 1 (print all 6 overlays, alt: 0=dont)\n')			
		
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('Example usage 1 [use all defaults]: python parse_watermask.py')
		
			sys.exit()

		elif opt in ("-m"):
			make_mask = arg
			make_mask = int(make_mask)
			
		elif opt in ("-c"):
			make_prob_map = arg
			make_prob_map = int(make_prob_map)

		elif opt in ("-p"):
			use_parallel = arg
			use_parallel = int(use_parallel)

		elif opt in ("-a"):
			make_all_masks = arg
			make_all_masks = int(make_all_masks)

		elif opt in ("-o"):
			make_overlays = arg
			make_overlays = int(make_overlays)
			
	if 'make_mask' not in locals():
		make_mask = 0

	if 'make_prob_map' not in locals():
		make_prob_map = 1
		
	if 'use_parallel' not in locals():
		use_parallel = 1

	if 'make_all_masks' not in locals():
		make_all_masks = 1

	if 'make_overlays' not in locals():
		make_overlays = 1
		
	if make_mask >6:
		make_mask = 6

	if make_all_masks >1:
		make_all_masks = 1

	if make_prob_map >1:
		make_prob_map = 1
		
	if make_all_masks ==1:		
		make_mask = 0
		
	#====================================================
	print('........................................')
	print('ooooo...................................')
	print('xxxxoo..................................')
	print('xxxxxxo..............GENERATE...........')
	print('xxxxxxxo................METASHAPE.......')
	print('xxxxxxxxxxo.................INPUTS......')
	print('xxxxxxxxxxxxxo..........................')
	print('xxxxxxxxxxxxxxxxxxxxo...................')
	print('xxxxxxxxxxxxxxxxxxxxxxxxxo..............')
	print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxo...........')
	print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxo..........')
	print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxo......')
	print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxo.....')
	print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxo...')
	print('xxxxxxxxWESTCOASTxxxxxxxxxxxxxxxxxxxxo..')
	print('xxxxxxxxxxx WATERMASKER xxxxxxxxxxxxxxo.')
	print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxo')
	print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

	root = Tk()
	root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of npz files")
	SAMPLE_DIREC = os.path.normpath(root.filename+os.sep+'agisoft')
	print('                                      ')
	print("Working on npz in %s" %  (root.filename))
	root.withdraw()

	sample_filenames = sorted(glob(root.filename+os.sep+'*.npz'))
	print("%i npz files" % (len(sample_filenames)))
	
	if make_overlays==1:
		root = Tk()
		root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of jpg files")
		print('                                      ')
		print("Working on npz in %s" %  (root.filename))
		root.withdraw()	

		image_filenames = sorted(glob(root.filename+os.sep+'*.jpg'))
		print("%i image files" % (len(image_filenames)))
	else:
		image_filenames = [None for k in sample_filenames]
	
	try:
		os.mkdir(SAMPLE_DIREC)
	except:
		pass

	settings = dict()
	settings['make_mask'] = make_mask
	settings['make_prob_map'] = make_prob_map
	settings['make_all_masks'] = make_all_masks
	settings['make_overlays'] = make_overlays
	settings['SAMPLE_DIREC'] = SAMPLE_DIREC

	try:
		if settings['make_mask']>0:
			os.mkdir(SAMPLE_DIREC+os.sep+'masks_'+str(settings['make_mask']))
	except:
		pass

	try:
		if settings['make_prob_map']>0:
			os.mkdir(SAMPLE_DIREC+os.sep+'prob_stack')
	except:
		pass
		
	if settings['make_all_masks']==1:
		for k in np.arange(1,7):
			try:
				os.mkdir(SAMPLE_DIREC+os.sep+'masks_'+str(k))
			except:
				pass
		
	if settings['make_overlays']==1:
		for k in np.arange(1,7):
			try:
				os.mkdir(SAMPLE_DIREC+os.sep+'overlays_'+str(k))
			except:
				pass
				
	
	try:
		if use_parallel==1:
			w = Parallel(n_jobs=-2, verbose=0, timeout=1000, max_nbytes=None)(delayed(main)(f, i, settings) for f,i in zip(sample_filenames, image_filenames))
		else:
			for counter,(f,i) in enumerate(zip(sample_filenames, image_filenames)):
				main(f, i, settings)
				print('%i out of %i done'%(counter,len(sample_filenames)))		
	except:
		print("Something went wrong with parallel ... reverting to serial - WARNING: significantly slower")
		for counter,(f,i) in enumerate(zip(sample_filenames, image_filenames)):
			try:
				main(f, i, settings)
				print('%i out of %i done'%(counter,len(sample_filenames)))
			except:
				pass
	
	
	#=====