# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2023, Marda Science LLC
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

import random, string, glob, os

# configfile = "F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2_windows.json"
# folder = "F:\watermasking_benchmark\CenCA_coastal_20170125\images"

# folder = "/mnt/d/for_watermasking/20230105_auto_landscape"
# out_folder = "/mnt/d/watermasking_results/20230105_auto_landscape"

# folder = "/mnt/d/for_watermasking/CenCA_coastal_20151209/images"
# out_folder = "/mnt/d/watermasking_results/CenCA_coastal_20151209"

folder = "/mnt/d/for_watermasking/CenCA_coastal_20160126/images"
out_folder = "/mnt/d/watermasking_results/CenCA_coastal_20160126"

configfile = "/mnt/f/dbuscombe_github/pcmsc_watermasking/config/watermask_benchmark_deploy_v2_unix.json"

##=============================================================

configfile = os.path.normpath(configfile)
folder = os.path.normpath(folder)

# write out sample filenames to temporary file
tmp_file = ''.join(random.choices(string.ascii_letters + string.digits, k=7))+'.sh'

sample_filenames = glob.glob(folder+os.sep+"*.jpg")

print(configfile)
print(folder)
print(tmp_file)
print("{} files to process".format(len(sample_filenames)))

with open(tmp_file, "w") as tf:
    for f in sample_filenames:
        tf.write('python watermask_single_image.py -f '+f+' -c '+configfile+' -o '+out_folder+'\n')

