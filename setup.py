from setuptools import setup, Extension
import numpy

# define the extension module
stickyext = Extension('stickydesign._stickyext', sources=['src/_stickyext.c'],
                          include_dirs=[numpy.get_include()])

# run the setup
setup(
    name='stickydesign',
    version='0.2dev',

    install_requires = ['numpy'],

    packages=['stickydesign'],
    ext_modules=[stickyext],
    package_dir = {'stickydesign': 'src'},
    package_data={'stickydesign': ['params/dnastackingbig.csv']},

    author = "Constantine Glen Evans",
    author_email = "cevans@dna.caltech.edu",
    description = "StickyDesign DNA Tile Sticky End Package",
    url = 'http://dna.caltech.edu/stickydesign'
)
