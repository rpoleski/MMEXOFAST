# Useful commands

## Check unit test coverage

cd MMEXOFAST/source/
coverage run --omit="mmexofast/unit_tests/*,*/MulensModel/*,*/sfit_minimizer/*" -m pytest mmexofast/unit_tests -v -s
coverage html

(need to run without the --fast flag to get proper coverage information, BUT this is slow)