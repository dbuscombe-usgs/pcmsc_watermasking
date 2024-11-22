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


# folder = "/mnt/d/for_watermasking/CenCA_coastal_20151209/images"
# out_folder = "/mnt/d/watermasking_results/CenCA_coastal_20151209"

# folder = "/mnt/d/for_watermasking/CenCA_coastal_20160126/images"
# out_folder = "/mnt/d/watermasking_results/CenCA_coastal_20160126"

# folder = "/mnt/d/for_watermasking/CenCA_coastal_20220609/images"
# out_folder = "/mnt/d/watermasking_results/CenCA_coastal_20220609"


########### July/Aug 2024

folder = "/mnt/d/for_watermasking/SoCA_coastal_20220302/images"
out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20220302"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20200918/images"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20200918"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20200506/images"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20200506"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20180913/images"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20180913"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20170301/images"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20170301"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20170301/images"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20170301"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20221002/images"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20221002"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20220928/jpg_adobe"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20220928"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20230308/jpg_adobe"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20230308"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20231012/jpg_adobe"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20231012"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20160928/images"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20160928"

# folder = "/mnt/d/for_watermasking/SoCA_coastal_20171227/images"
# out_folder = "/mnt/d/watermasking_results/SoCA_coastal_20171227"

# folder = "/mnt/d/for_watermasking/SoCA_Thomas-fire_20180123/jpg_adobe"
# out_folder = "/mnt/d/watermasking_results/SoCA_Thomas-fire_20180123"

# folder = "/mnt/d/for_watermasking/CenCA-SoCA_coastal_20180329/images"
# out_folder = "/mnt/d/watermasking_results/CenCA-SoCA_coastal_20180329"



##### test
# folder = "/mnt/d/for_watermasking/test_multiband/jpg_adobe"
# out_folder = "/mnt/d/watermasking_results/test_multiband"


# test_case = 'sand'
test_case = 'water'

model = 'segformer'
# model = 'resunet'

####=======================================================================
# if test_case=='water':
configfile = f"/mnt/f/dbuscombe_github/pcmsc_watermasking/config/{test_case}mask_benchmark_deploy_{model}_unix.json"

print(configfile)

# elif test_case=='sand':
#     ### sand!!!
#     configfile = f"/mnt/f/dbuscombe_github/pcmsc_watermasking/config/sandmask_benchmark_deploy_resunet_unix.json"

# else:
#     print("test case either water or sand")
#     import sys; sys.exit(2)


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
        if test_case=='water':
            tf.write('python watermask_single_image.py -f '+f+' -c '+configfile+' -o '+out_folder+'\n')
        elif test_case=='sand':
            tf.write('python sandmask_single_image.py -f '+f+' -c '+configfile+' -o '+out_folder+'\n')
