import setuptools

__version__ = "0.3.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparepo",
    version=__version__,
    description="Spatial loading for AREPO.",
    url="https://github.com/jborrow/sparepo",
    author="Josh Borrow",
    author_email="josh@joshborrow.com",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numba>0.50.0", "attrs", "h5py>3.0.0", "numpy"],
)
