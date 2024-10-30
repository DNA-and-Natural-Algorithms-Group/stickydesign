Changelog
==========

0.9.2
-----

* Update versions to support Numpy 2.
* Numerous linting and typing fixes.

0.9.1
-----

* Update stickydesign-accel rust dependencies, pyo3 organization.

0.9.0a1
-------

* Deprecated the old C extension.  This greatly simplifies the build system, but for now, disables EnergeticsBasicOld.
* Moved to a setup.cfg and pyproject.toml build system with setuptools.
* Created a new package, stickydesign-accel, that contains Rust extensions to accelerate the newer EnergeticsBasic.  When installed, this gives a significant (around 20x, I think) speedup.

0.8.3
-----

* Python and numpy updates.

0.8.1 - 0.8.2
-------------

* Minor fixes and updates.

0.8.0
-----

Major changes:

* Added stickydesign2 (used internally by alhambra)

0.7.0
-----

Major changes:

* Rearranged energetics.  There are now three energetics classes:
    * EnergeticsOld: this is the old energetics class (energetics_santalucia)
    * EnergeticsBasic: this is an energetics class, based on energetics_daoe, which assumes that we know nothing about anything adjacent to the sequences.  This likely makes the most sense for toeholds, or as a base for creating your own classes.  It uses the new 'S' 'end' type, which is really just a sequence.
    * EnergeticsDAOE: this is an improved version of energetics_daoe.  It makes the most sense for sticky ends for DAO-E tiles.
* Added the multimodel module, which contains an endchooser that tries to optimize for multiple energetics classes at once.
* Added documentation!

Fixes:

* Fixed various installation issues in both Python 2 and Python 3.
* Removed auto-import of multimodel, which requires Python 3.
* Fixed examples in the README.

0.4.3
-----

* Include ∆S parameters and thus temperature dependence for internal single-base
  mismatches, from Allawi 97 + 98 (4 papers in total) and Peyret 99. 
