# -*- coding: utf-8 -*-

try:
    import setuptools
    from setuptools import setup, find_packages
except:
    print("Please install setuptools")

import os
long_description = "PyTorch sandbox of Adversarial Attacks"
if os.path.exists("README.md"):
    long_description = open("README.md").read()

setup(
    name="nnattacks",
    version="0.0.1",
    description="PyTorch sandbox of Adversarial Attacks",
    long_description=long_description,
    license="MIT",
    author="Masanari Kimura",
    author_email="mkimura@klis.tsukuba.ac.jp",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "scipy",
        "tqdm",
    ]
)