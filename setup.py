from setuptools import setup, find_packages

setup(
    name = 'sphimg',
    version = 0.1,

    packages = find_packages(),
    requires = ['numpy', 'scipy', 'healpy', 'h5py', 'cora'],
    package_data = {'sphimg.telescope' : ['gmrtpositions.dat'] },
    scripts = ['scripts/sph-makeproducts', 'scripts/sph-runpipeline'],

    # metadata for upload to PyPI
    author = "J. Richard Shaw, Shifan Zuo",
    author_email = "sfzuo@bao.ac.cn",
    description = "Transit telescope spherical analysis map-making",
    license = "GPL v3.0",
    url = "http://github.com/zuoshifan/sphimg"
)
