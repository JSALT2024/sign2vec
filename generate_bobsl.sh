python sign2vec/generate_dataset.py 

import os
os.system('python -m pip install pyyaml==5.1')
import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
os.system('git clone 'https://github.com/facebookresearch/detectron2')
dist = distutils.core.run_setup("./detectron2/setup.py")

# Install dependencies
dep = ' '.join(["\""+x+"\"" for x in dist.install_requires])
os.system(f"python -m pip install {dep}")
sys.path.insert(0, os.path.abspath('./detectron2'))