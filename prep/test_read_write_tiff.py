
import numpy as np
import PIL.Image 
import PIL.ExifTags
import tifftools
import piexif
import tifffile

def get_exif(f):
	img = PIL.Image.open(f)
	exif_data = img.getexif()
	exif = {
		PIL.ExifTags.TAGS[k]: str(v)
		for k, v in img._getexif().items()
		if k in PIL.ExifTags.TAGS}
	return exif,img
	

#fname = r'CAM432_20170125221820_30.jpg'	
fname = r'CAM001_20210326152247_80.tif'


# info = tifftools.read_tiff(fname)
# # info['ifds'][0]['tags'][tifftools.Tag.ImageDescription.value] = {
    # # 'data': 'A dog digging.',
    # # 'datatype': tifftools.Datatype.ASCII
# # }


# nx = info['ifds'][0]['tags'][256]['data'][0]
# ny = info['ifds'][0]['tags'][257]['data'][0]
	
# #make up some model output data, 5 bands pf 

# prob_land = conf = var = mask1 = mask2 = np.random.randint(low=0, high=1, size=(nx,ny)).astype(np.uint8)

# out_stack = np.dstack((prob_land*100,conf*100,var*100))#, mask1*100, mask2*100))

# #tmp = info['ifds'][0]['tags'][34675]['data']

# out_stack = np.dstack((prob_land*100,conf*100,var*100))
# pil_img = PIL.Image.fromarray(out_stack)


# exif_bytes = piexif.dump(exif_dict)
# pil_img.save('test_PIL_piexif.tif', "tiff", exif=exif_bytes, compress='deflate')
# #106MB! compression does nothing

# info['data'] = out_stack

# exififd = info['ifds'][0]['tags'][tifftools.Tag.EXIFIFD.value]['ifds'][0]
# exififd['tags'][tifftools.constants.EXIFTag.FNumber.value] = {
    # 'data': [54, 10],
    # 'datatype': tifftools.Datatype.RATIONAL
# }
# tifftools.write_tiff(info, 'image_exif_tagged.tif')	



exif,img = get_exif(fname)

nx, ny, nz = np.shape(img)
	
#make up some model output data, 5 bands pf 

prob_land = conf = var = mask1 = mask2 = np.random.randint(low=0, high=1, size=(nx,ny)).astype(np.uint8)

out_stack = np.dstack((prob_land*100,conf*100,var*100, mask1*100, mask2*100))

# option 1, PIL

img.save('test_PIL.tif', "tiff", exif=exif)
#106mb

img.save('test_PIL_deflate.tif', "tiff", exif=exif, compress='deflate')
#106mb - too large! compression does nothing

# option 2, tifffile 
tifffile.imwrite('test_tifffile_metadata.tif', out_stack, photometric='rgb',  metadata=exif, compress='deflate')
#324 kb 

data = tifffile.imread('test_tifffile_metadata.tif')
# works! but exif data is not exif data  ... it is tag data. but this is still the ONLY option for us


# option 3, piexif 
#pip install piexif
img = PIL.Image.open(fname)
exif_dict = piexif.load(img.info['exif'])

#altitude = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
#print(altitude)

out_stack = np.dstack((prob_land*100,conf*100,var*100))
pil_img = PIL.Image.fromarray(out_stack)
# cant be 5 bands

exif_bytes = piexif.dump(exif_dict)
pil_img.save('test_PIL_piexif.tif', "tiff", exif=exif_bytes, compress='deflate')
#106MB! compression does nothing


		
		#out_stack = np.dstack((est_label*100,conf*100,var*100)).astype(np.uint8)
		#out_stack = np.dstack((est_label*100,conf*100,var*100)).astype(np.uint8) #, land_conservative, land_liberal)).astype(np.uint8)
		# #tifffile.imwrite(outfile, out_stack, photometric='rgb', compression='deflate', metadata=exif)


# # option 4, piexif with input tif

# fname = r'CAM432_20170125221820_30.tif'	
# img = PIL.Image.open(fname)
# exif_dict = piexif.load(img.info['exif'])

# out_stack = np.dstack((prob_land*100,conf*100,var*100))
# pil_img = PIL.Image.fromarray(out_stack)
# # cant be 5 bands

# exif_bytes = piexif.dump(exif_dict)
# pil_img.save('test_PIL_piexif.tif', "tiff", exif=exif_bytes, compress='deflate')
# #106MB! compression does nothing



			#outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks'))
			
			#out_stack = np.dstack((est_label*100,conf*100,var*100)).astype(np.uint8)
						
			#tifffile.imwrite('test.tif', out_stack, photometric='rgb', compression='deflate', metadata={'model':'george'})#, metadata=exif)

			
# fname = r'F:\mask_bench_20170125\CAM432_20170125221828_30.jpg'		

# import piexif
# from PIL import Image

# img = Image.open(fname)
# exif_dict = piexif.load(img.info['exif'])

# altitude = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
# print(altitude)

# exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (140, 1)


# exif_bytes = piexif.dump(exif_dict)
# img.save('_%s' % 'test.tif', "tiff", exif=exif_bytes)

		
# # import tifftools
# info = tifftools.read_tiff('watermask_out_5band_test.tif')
# info['ifds'][0]['tags'][tifftools.Tag.ImageDescription.value] = {
    # 'data': 'A dog digging.',
    # 'datatype': tifftools.Datatype.ASCII
# }


# exififd = info['ifds'][0]['tags'][tifftools.Tag.EXIFIFD.value]['ifds'][0]
# exififd['tags'][tifftools.constants.EXIFTag.FNumber.value] = {
    # 'data': [54, 10],
    # 'datatype': tifftools.Datatype.RATIONAL
# }
# tifftools.write_tiff(info, 'photograph_tagged.tif')	

# info = tifftools.read_tiff(f) #'IMG_0036_1.tif')
# # Get a reference to the IFD with GPS tags.  If the GPS data is in a different
# # location, you might need to change this.
# gpsifd = info['ifds'][0]['tags'][tifftools.Tag.GPSIFD.value]['ifds'][0][0]
# # Set the altitude tag; this assumes it already exists and is stored as a rational
# gpsifd['tags'][tifftools.constants.GPSTag.GPSAltitude.value]['data'] = [140, 1]
# # Write the output; this copies all image data from the original file.
# tifftools.write_tiff(info, 'new_file.tif')

# exififd = info['ifds'][0]['tags'][tifftools.Tag.EXIFIFD.value]['ifds'][0]
# exififd['tags'][tifftools.constants.EXIFTag.FNumber.value] = {
    # 'data': [54, 10],
    # 'datatype': tifftools.Datatype.RATIONAL
# }
# tifftools.write_tiff(info, 'photograph_tagged.tif')		
	
# from PIL import Image
# from PIL.TiffTags import TAGS

# with Image.open('watermask_out_5band_test.tif') as img:
    # meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}		
		
		