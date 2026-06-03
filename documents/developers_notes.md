# Useful commands

## Check unit test coverage

cd MMEXOFAST/source/
coverage run --omit="mmexofast/unit_tests/*,*/MulensModel/*,*/sfit_minimizer/*" -m pytest mmexofast/unit_tests -v -s --fast
coverage html
