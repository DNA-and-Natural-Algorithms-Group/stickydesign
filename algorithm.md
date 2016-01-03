# energetics_daoe - new

* Consider two ends with some offset. The "first" end of the two will be read 5
  to 3, the second 3 to 5 -> "left to right" in a diagram.
* For each paired pair of bases (2 nt on one end, 2 on the other), calculate
  three values:
    * ens: if a matching NN pair, then the dG for the pair, otherwise 0.
    * ltmm: if only the "righthand" bases match, then the dG for a terminal
      mismatch on the left-hand side (5' on the first end, 3' on the second),
      otherwise 0. Note that currently this is just the sum of two terminal
      dangle terms.
    * rtmm: if only the "lefthand" bases match, then the dG for a terminal
      mismatch on the right-hand side, otherwise 0.
    * intmm: in either of the previous two cases, the dG assuming the mismatch
      is internal, otherwise 0.
* Depending upon the offset:
    * For zero offset, if they are already matching, `ens[0]` and `ens[-1]` have
      the dG for adjacent coaxial stacks added to them (optionally with nick
      correction), and have the dG for the preexisting dangles in the unbound
      ends *subtracted*. This gives adjusted binding dGs for those bases. Note
      that this ignores an unusual case where the end of one of the ends has an
      internal mismatch. It's unclear how realistic our parameters would be in
      that case, or whether it would ever reasonably be favorable.
    * For "stretching" offset, `ens[0]` and `ens[-1]` have terminal dangle
      parameters added to them for the continuation of the ends.
    * For "contracting" offset, `ens[0]` and `ens[-1]` have the adjacent coaxial
      stacks added, and tail penalties.
* Now, reading left to right, by NN pairs, we start with an accumulator (acc) of 0,
  and a max binding strength (bindmax) of 0:
    * If the pair is matching, then `ens` is added to the accumulator.
    * Otherwise, if the pair can be a right terminal mismatch (rtmm != 0), then
      if bindmax < acc + rtmm, then bindmax is set to acc + rtmm, which
      corresponds to the end of binding with a dangle. Then intmm is added to
      the accumulator, corresponding to the start of an internal mismatch.
    * Otherwise, if the pair can be a left terminal mismatch (ltmm != 0), and
      we're not on the last pair, we check to see whether ltmm > acc+intmm and
      the next pair is matching, in which case we assume we have a left terminal
      mismatch, as it will be stronger, and we *set* (not add) acc = ltmm;
      otherwise, we add intmm to acc (we assume an internal mismatch).
      Note that this ignores the possibility of just the last base being paired,
      with the penultimate an internal mismatch. This is unlikely and Santa
      Lucia advises against using internal mismatch parameters in such cases.
    * If none of these are the case, add a "loop penalty" to acc, which
      approximates the dG penalty for a symmetric loop.
* At the end of this process, if acc > bindmax, set bindmax = acc.
* Do this for every offset, and find the maximum bindmax.
