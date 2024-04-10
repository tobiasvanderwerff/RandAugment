from setuptools import setup, find_packages

setup(
    name="randaugment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "albumentations",
        "numpy"
    ],
)