The StickyDesign Sticky End Sequence Design Package
===================================================

# Introduction

StickyDesign is a Python package developed by the Winfree lab at Caltech (http://dna.caltech.edu) for designing sticky end sequences for DNA tile systems, especially systems using DX and other tile structures where sticky ends in a single tile are separated by double-stranded regions (better support for single stranded tiles is still under development). 

# Requirements

StickyDesign is implemented as a Python library that can easily be used interactively through an interactive Python environment like iPython[1]. It uses portions of the general scientific computing ecosystem for Python, and also uses a small Python C extension to greatly accelerate energetics calculations. As such, it likely requires a minimum:

* Python >= 2.6
* Numpy
* Scipy
* A working C compiler setup
* Python development headers

The easiest way to ensure the installation of these requirements is to use a scientific Python distribution like Enthought or Anaconda; the Scipy website has a [list of these distributions](http://scipy.org/install.html).

# Installation

StickyDesign uses the standard Setuptools/Distutils system for Python packages, based around a setup.py. Thus, to install the system into your global Python distribution (whether global to the computer or just your account), just run the following (with sudo if necessary):

    ./setup.py install

To install to another location, or set options for the installation, consult the documentation for the install command:
    
	./setup.py install --help 

To only build the library in a `build/` subdirectory, use the build command:

    ./setup.py build
	
The built library (a directory named `stickydesign`) can be moved to wherever you run your Python interpreter and used without installation.	
	
Running setup.py with no options will provide a number of options and directions to further documentation.

# Use

The stickydesign package has inline documentation available for all of its functions, which provide details on the use of each function. 

Lists of sticky end sequences are held in `endarray` classes. These contain the sequences themselves, the adjacent bases on both sides (the base itself on the end side, the complement of the base on the other side), and a end *type*. The end type specifies what the edges of the ends look like, and currently has two possible values: 

* 'DT', for ends where the 5' end continues to a double-stranded (D) region and the 3' end is terminal (T), and
* 'TD', for ends where the 5' end is terminal (T), and the 3' end continues to a double-stranded region (D).

As examples, a usual DAO-E tile will have two DT ends and two TD ends, while a DAO-O tile will have all of the same type (eg, NAoMI-B has four TD ends). Other end types, to support things like single-stranded tiles and toeholds for branch migration, are still being considered.

To get end and complement sequences along with their adjacent nucleotide, use the `.ends` or `.comps` properties of the class.

For usual sequence design, the `easyends` function provides a simple but configurable interface to the sequence design code. At its simplest level, for example, trying to design ten sticky ends (each with a complement included) of type DT, and using the defaults for target interaction energy and maximum non-orthogonal binding, you might do the following (in iPython, with In showing user input and Out showing Python output):

	In [1]: import stickydesign

	In [2]: stickydesign.easyends('DT',5,number=10)
	WARNING:root:Calculated optimal interaction energy is 8.354.
	Out[2]: <endarray (10): type DT; ['accgtat', 'tcgaaga', 'gaaacgt',
					'actgtca', 'ctgtgac', 'catgacc', 'cgttcaa', 
					'cgtactg', 'cggtatg', 'cgaacaa']>
					
	In [3]: Out[2].ends
	Out[3]: <endarray (10): type DT; ['accgta', 'tcgaag', 'gaaacg', 
		'actgtc', 'ctgtga', 'catgac', 'cgttca', 'cgtact', 'cggtat', 'cgaaca']>

	In [4]: Out[2].comps
	Out[4]: <endarray (10): type DT; ['atacgg', 'tcttcg', 'acgttt', 
		'tgacag', 'gtcaca', 'ggtcat', 'ttgaac', 'cagtac', 'catacc', 'ttgttc']>

To get an array of interactions between an set of sticky ends, use the 
`energy_array_uniform` function. This array can then be plotted with `matplotlib`:

	In[5]: energyarray = stickydesign.energy_array_uniform( Out[2], 
		stickydesign.energetics_santalucia() )
		
	In[6]: import matplotlib.pyplot as pyplot
	
	In [7]: pyplot.rc('image',interpolation='nearest')

	In [8]: pyplot.imshow(energyarray)
	Out[8]: <matplotlib.image.AxesImage at 0x10c23c450>

	In [9]: colorbar()
	Out[9]: <matplotlib.colorbar.Colorbar instance at 0x10c1d83b0>

# About

StickyDesign and related software is a project of the Winfree lab at Caltech[2]. It is currently maintained by Constantine Evans (cge@dna.caltech.edu).

[1](http://ipython.org)
[2](http://dna.caltech.edu)
