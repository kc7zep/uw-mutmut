Exception-related mutations

1. Mutate raise
This mutation nullifies a raise or raise Exception() by replacing it with pass.

To see an example:
In this branch:
python setup.py install # to make the current implementation active
cd examples/exception
mutmut run --paths-to-mutate=sample
You should see 15 mutations created of which 6 survive.
mutmut show 7 # should show the new mutation
