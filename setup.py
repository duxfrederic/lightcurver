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


setup(
    name="lightcurver",
    version="1.0.0",
    author="Frédéric Dux",
    author_email="duxfrederic@gmail.com",
    description="A thorough structure for precise photometry and deconvolution of time series of wide field images.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={"lightcurver": ["pipeline/*.yaml"]},
    cmdclass={'test': PyTest}
)
