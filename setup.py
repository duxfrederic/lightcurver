from setuptools import setup, find_packages

setup(
    name="lightcurver",
    version="0.1.0",
    packages=find_packages(),
    package_data={"lightcurver": ["config/*.yaml"]},
)
