#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        
        # Call original build_ext command
        build_ext.run(self)

_integration = Extension('pumpprobe._integration',
                    sources = ['pumpprobe/_integration.cpp'],
                    extra_compile_args=['-O3'])
                    
_convolution = Extension('pumpprobe._convolution',
                    sources = ['pumpprobe/_convolution.cpp',
                               'pumpprobe/convolution.cpp'],
                    extra_compile_args=['-O3'])
                 
v = 1.0

setup(name='pumpprobe',
      version=v,
      description='Tools for the analysis of pump probe experiments on C. elegans',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['pumpprobe'],
      ext_modules = [_integration,_convolution],
      package_data={'pumpprobe': ['anatlas_neuron_positions.txt',
                                  'aconnectome.json',
                                  'aconnectome_ids.txt',
                                  'aconnectome_white_1986_L4.csv',
                                  'aconnectome_white_1986_A.csv',
                                  'aconnectome_white_1986_whole.csv',
                                  'aconnectome_witvliet_2020_7.csv',
                                  'aconnectome_witvliet_2020_8.csv',
                                  'aconnectome_ids_ganglia.json',
                                  'sensoryintermotor_ids.json',
                                  'esconnectome_monoamines_Bentley_2016.csv',
                                  'esconnectome_neuropeptides_Bentley_2016.csv',
                                  'GenesExpressing-unc-7-unc-9-inx-_-eat-5-thrs2.csv',
                                  'GenesExpressing-neuropeptides.csv',
                                  'GenesExpressing-neuropeptide-receptors.csv',
                                  'GenesExpressing-daf-2-thrs2.csv',
                                  'GenesExpressing-npr-4-thrs2.csv',
                                  'GenesExpressing-npr-11-thrs2.csv',
                                  'GenesExpressing-pdfr-1-thrs2.csv']}
     )
