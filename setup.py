#! /usr/bin/env python
"""
Setup for sfforecaster
"""
import os
import sys
from setuptools import setup


def read(fname):
	"""Read a file"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
	""" Get the package version number """
	import sfforecaster
	return sfforecaster.__version__


PY_MAJOR_VERSION=sys.version_info.major
PY_MINOR_VERSION=sys.version_info.minor
print("PY VERSION: maj=%s, min=%s" % (PY_MAJOR_VERSION,PY_MINOR_VERSION))

reqs= []
reqs.append('numpy')
reqs.append('pillow') 
reqs.append('astropy')
reqs.append('scikit-image')
reqs.append('scikit-learn')
reqs.append('torch')
reqs.append('torchvision')
reqs.append('tqdm')
reqs.append('transformers')
reqs.append('accelerate')
reqs.append('evaluate')
reqs.append('matplotlib')
reqs.append('wandb')

##data_dir = 'data'

setup(
	name="sfforecaster",
	version=get_version(),
	author="Simone Riggi",
	author_email="simone.riggi@gmail.com",
	description="Solar flare forecaster application based on transformer models",
	license = "GPL3",
	url="https://github.com/inaf-oact-ai/solar-flare-forecaster",
	keywords = ['radio', 'source', 'classification', 'transformers'],
	long_description=read('README.md'),
	long_description_content_type='text/markdown',
	download_url="https://github.com/inaf-oact-ai/solar-flare-forecaster/archive/refs/tags/v1.0.0.tar.gz",
	packages=['sfforecaster'],
	install_requires=reqs,
	scripts=['scripts/run.py'],
	classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Astronomy',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3'
	]
)

