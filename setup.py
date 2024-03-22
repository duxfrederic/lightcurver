import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as test_command


class PyTest(test_command):
    def finalize_options(self):
        test_command.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        err_no = pytest.main(self.test_args)
        sys.exit(err_no)


with open('VERSION', 'r') as version_file:
    version = version_file.read().strip()

setup(
    name="lightcurver",
    version=version,
    author="Frédéric Dux",
    author_email="duxfrederic@gmail.com",
    description="A thorough structure for precise photometry and deconvolution of time series of wide field images.",
    long_description=open('README.md').read(),
    packages=find_packages(),
    package_data={"lightcurver": ["pipeline/*.yaml"]},
    cmdclass={'test': PyTest}
)
