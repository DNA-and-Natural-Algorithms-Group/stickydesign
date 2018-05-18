from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy # NOQA
        self.include_dirs.append(numpy.get_include())


# define the extension module
stickyext = Extension(
    'stickydesign._stickyext',
    sources=['stickydesign/_stickyext.c'])

#    include_dirs=[numpy.get_include()])

# run the setup
setup(
    name='stickydesign',
    version='0.7.0',
    setup_requires=['numpy'],
    packages=['stickydesign'],
    ext_modules=[stickyext],
    cmdclass={'build_ext':build_ext},
    package_data={'stickydesign': ['params/dnastackingbig.csv']},
    author="Constantine Glen Evans",
    author_email="cevans@evans.foundation",
    description="StickyDesign DNA Tile Sticky End Package",
    url='http://dna.caltech.edu/StickyDesign',
    zip_safe=True,
    test_suite='nose.collector')
