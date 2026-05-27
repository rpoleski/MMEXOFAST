from setuptools import setup, find_packages

setup(
    name="mmexofast",
    package_dir={"": "source"},
    packages=find_packages(where="source"),
)
