Changelog
==========

0.7.2
-----

Updated release to support more recent versions of Numpy.

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
