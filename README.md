
This code is intended for the Mu2e Analysis Group currently consisting of:

* Caltech: L. Borrel, B. Echenard, D. Hitlin, H. Jafree, S. Middleton, F. Porter
* LBNL: R. Bonventre, D. Brown
* Berkeley: Y. Kolomensky, V. Singh
* CUNY-York: A. Edmonds
* Northwestern: S. Dittmer, C. Kampa, M. Schmitt
* plus others who may join


the py-fitter code contains likelihood fitting code developed using the python package "zfit".

# Code Review Policy:

Before submitting a PR the author should check that the code converges on all available samples and produces meaningful results.

When code is altered, a PR is necessary no matter how minor. There are some rules as to how long a PR can "hang" however before being merged. If the review is not active in the process than group leader can merge a PR within a week. In circumstances were the reviewer is active we expect a turn around as follows:

* minor change (a few lines, no obvious breaks expected) - 1 day turn around, unless there is vacation, then it would be 1 day + vacation return date (reviewer should communicate this)
* some chance of breaking change (e.g. more than a few files changed) - 3 day turnaround
* major change - this requires substantial checks by the review and ~ a week turnaround, if nothing comes up during review, is expected
