from setuptools import setup

# run the setup
setup(
    name='stickydesign2',
    version='0.1.0.dev1',
    install_requires=['numpy'],
    packages=['stickydesign2'],
    author="Constantine Glen Evans",
    author_email="cevans@evans.foundation",
    description="StickyDesign DNA Tile Sticky End Package",
    url='http://dna.caltech.edu/StickyDesign',
    zip_safe=True,
    test_suite='nose.collector')
