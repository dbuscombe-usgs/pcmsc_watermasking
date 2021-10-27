import PIL.ExifTags
import PIL.Image
import tensorflow as tf #numerical operations on gpu
import os
from tkinter import filedialog
from tkinter import *
from joblib import Parallel, delayed
from tqdm import tqdm

#-----------------------------------
def get_exif(f):
	img = PIL.Image.open(f)
	exif_data = img.getexif()
	exif = {
		PIL.ExifTags.TAGS[k]: str(v)
		for k, v in img._getexif().items()
		if k in PIL.ExifTags.TAGS } #if v !=0
	img.close()
	return exif

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of images to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()

#sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))

sample_filenames = sorted(glob(sample_direc+os.sep+'*.jpg'))


# print(exif)
f = sample_filenames[0]

exif = get_exif(f)

print(exif)

f = sample_filenames[100]

exif = get_exif(f)

print(exif)

#great!

#w = Parallel(n_jobs=-1, verbose=0, max_nbytes=None)(delayed(get_exif)(f) for f in tqdm(sample_filenames))

for counter,f in enumerate(sample_filenames):
	print(f)
	get_exif(f)

# Traceback (most recent call last):
  # File "test_exif.py", line 48, in <module>
    # get_exif(f)
  # File "test_exif.py", line 16, in get_exif
    # for k, v in img._getexif().items()
# AttributeError: 'NoneType' object has no attribute 'items'



# w = Parallel(n_jobs=-1, verbose=100, max_nbytes=None)(delayed(get_exif)(f) for f in tqdm(sample_filenames))
		
# for counter,f in enumerate(sample_filenames):
	# get_exif(f)


# import PIL.ExifTags
# import PIL.Image

# img = PIL.Image.open(f)
# exif_data = img.getexif()
# exif = {
	# PIL.ExifTags.TAGS[k]: str(v)
	# for k, v in exif_data.items()
	# if k in PIL.ExifTags.TAGS } #if v !=0
	
# print(exif)

# # profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -99999.0, 'width': 19284, 'height': 4620, 'count': 1, 'crs': CRS.from_epsg(6347), 'transform': Affine(1.0, 0.0, 433044.0,
       # # 0.0, -1.0, 3899656.0), 'blockxsize': 256, 'blockysize': 256, 'tiled': True, 'compress': 'lzw', 'interleave': 'band'}

# # profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -99999.0, 'width': 19284, 'height': 4620, 'count': 1, 'crs': CRS.from_epsg(6347), 'transform': Affine(1.0, 0.0, 433044.0,
       # # 0.0, -1.0, 3899656.0), 'blockxsize': 256, 'blockysize': 256, 'tiled': True, 'compress': 'lzw', 'interleave': 'band'}

        # profile.update(driver='GTiff', dtype=out_stack.dtype, count=1, compress='deflate', photometric='minisblack', nodata=0, tiled=True, interleave='band',
						# blockxsize=256, blockysize=256)
	   
        # with rasterio.Env():
            # with rasterio.open(outfile, 'w', **profile) as dst:
                # dst.write(out_stack[:,:,0])
 
# profile = {'driver':'GTiff', 'height':out_stack.shape[1], 'width':out_stack.shape[2], 'count':np.ndim(out_stack), 'dtype':out_stack.dtype,  'nodata':0,'crs':'None', 'transform':"Affine(1.0, 0.0, 0.0,0.0, 1.0, 0.0)"}
					
					
# with rasterio.Env():					
   # with rasterio.open('test.tif', 'w', **profile) as dst:
      # dst.write(out_stack[0], 1)
      # dst.write(out_stack[1], 2)
      # dst.write(out_stack[2], 3)
		
# # import tifffile  # http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html


# # metadata = dict(model='george', shape=out_stack.shape, dtype=out_stack.dtype.str)
# # print(out_stack.shape, out_stack.dtype, metadata['model'])

# # metadata = json.dumps(metadata)
# # options = dict(tile=(256, 256), photometric='rgb')#, compression='jpeg')
# # tifffile.imsave('model.tif', out_stack, description=metadata, **options )
# out_stack = np.dstack((est_label,conf,var))

		# imsave(outfile.replace('.tif','.png'),(100*out_stack).astype('uint8'),compression=9)

		# with rasterio.open(f) as src:
		   # profile = src.profile
		   # img = src.read()

		# #profile.update(driver='GTiff', dtype=out_stack.dtype, count=np.ndim(out_stack), compress='JPEG', photometric='minisblack', nodata=0, tiled=True, interleave='band', blockxsize=256, blockysize=256)

		# profile.update(driver='PNG', compression=9)
		# with rasterio.open(outfile, 'w', **profile) as dst:
		   # dst.write(out_stack.T)

# from tifffile import TiffWriter
# data = np.arange(1024*1024*3, dtype='float32').reshape((1024, 1024, 3))


# with TiffWriter('temp.tif', bigtiff=True) as tif:
	# options = dict(tile=(256, 256), photometric='minisblack',compression='deflate')
	# tif.save(data, **options)
	# # # save pyramid levels to the two subifds
	# # # in production use resampling to generate sub-resolutions
	# # tif.save(data[::2, ::2], subfiletype=1, **options)
	# # tif.save(data[::4, ::4], subfiletype=1, **options)


# # with tifffile.TiffFile('microscope.tif') as tif:
    # # data = tif.asarray()
    # # metadata = tif[0].image_description
# # metadata = json.loads(metadata.decode('utf-8'))
# # print(data.shape, data.dtype, metadata['microscope'])			