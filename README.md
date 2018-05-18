The StickyDesign Sticky End Sequence Design Package
===================================================

# Introduction

StickyDesign is a Python package developed by the Winfree lab at
Caltech (http://dna.caltech.edu) for designing sticky end sequences
for DNA tile systems, especially systems using DX and other tile
structures where sticky ends in a single tile are separated by
double-stranded regions (better support for single stranded tiles is
still under development).

# Requirements

StickyDesign is implemented as a Python library that can easily be
used interactively through an interactive Python environment like
iPython[1]. It uses portions of the general scientific computing
ecosystem for Python, and also uses a small Python C extension to
greatly accelerate energetics calculations. As such, it likely
requires a minimum:

* Python >= 2.6 (Python 3.6 or higher recommended)
* Numpy
* Scipy
* A working C compiler setup
* Python development headers

The easiest way to ensure the installation of these requirements is to
use a scientific Python distribution like Enthought or Anaconda; the
Scipy website has a [list of these
distributions](http://scipy.org/install.html).

# Installation

The easiest way to install stickydesign is via Pip.

To install via PyPI (stable releases, may be outdated):

	pip install stickydesign
	
To install from github's main branch (stable, more up to date):

    pip install git+https://github.com/DNA-and-Natural-Algorithms-Group/stickydesign.git

To install from github's main development branch (unstable):

    pip install git+https://github.com/DNA-and-Natural-Algorithms-Group/stickydesign.git@dev

Alternatively, normal python installation methods (easy_install, setup.py) may
be used. 

Installation requires a working C compiler so that energy model
speedups can be compiled.  If compilation does not work for you,
please let us know.


# Use

The stickydesign package has inline documentation available for all of its functions, which provide details on the use of each function. 

Lists of sticky end sequences are held in `endarray` classes. These contain the sequences themselves, the adjacent bases on both sides (the base itself on the end side, the complement of the base on the other side), and a end *type*. The end type specifies what the edges of the ends look like, and currently has three possible values: 

* 'DT', for ends where the 5' end continues to a double-stranded (D) region and the 3' end is terminal (T), and
* 'TD', for ends where the 5' end is terminal (T), and the 3' end continues to a double-stranded region (D). (uses EnergeticsDAOE energy model)
* 'S', for 'ends' that are really just sequences, where nothing is assumed about the adjacent regions. (uses EnergeticsBasic energy model)

As examples, a usual DAO-E tile will have two DT ends and two TD ends, while a DAO-O tile will have all of the same type (eg, NAoMI-B has four TD ends). Other end types, to support things like single-stranded tiles and toeholds for branch migration, are still being considered.

To get end and complement sequences along with their adjacent nucleotide, use the `.ends` or `.comps` properties of the class.  (For 'S', these will just return the sequence and complement.)

For usual sequence design, the `easyends` function provides a simple but configurable interface to the sequence design code. At its simplest level, for example, trying to design ten sticky ends (each with a complement included) of type DT, and using the defaults for target interaction energy and maximum non-orthogonal binding, you might do the following (in iPython, with In showing user input and Out showing Python output):

	In [1]: import stickydesign

	In [2]: stickydesign.easyends('DT',5,number=10, energetics=stickydesign.EnergeticsDAOE())
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
		stickydesign.EnergeticsDAOE() )
		
	In[6]: import matplotlib.pyplot as pyplot
	
	In [7]: pyplot.rc('image',interpolation='nearest')

	In [8]: pyplot.imshow(energyarray)
	Out[8]: <matplotlib.image.AxesImage at 0x10c23c450>

	In [9]: colorbar()
	Out[9]: <matplotlib.colorbar.Colorbar instance at 0x10c1d83b0>

For 'S' type ends, use EnergeticsBasic in place of EnergeticsDAOE; this is now the default on easyends.

To change parameters for the energetics classes, see the documentation for those classes: for example, temperature (in Celsius) can be specified using, eg, `EnergeticsBasic(temperature=33)`.

# About

StickyDesign and related software is a project of the Winfree lab at Caltech[2]. It is currently maintained by Constantine Evans (cge@dna.caltech.edu).

[1](http://ipython.org)
[2](http://dna.caltech.edu)

