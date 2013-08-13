from setuptools import setup, Extension
import numpy

# define the extension module
stickyext = Extension('stickydesign._stickyext', sources=['src/_stickyext.c'],
                          include_dirs=[numpy.get_include()])

# run the setup
setup(
    name='stickydesign',
    version='0.2',
    packages=['stickydesign'],
    ext_modules=[stickyext],
    package_dir = {'stickydesign': 'src'},
    package_data={'stickydesign': ['params/dnastackingbig.csv']}
)
