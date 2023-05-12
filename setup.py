#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="beessl",
    version="0.1.0",
    description="beeSSL: A benchmark for SSL in Beehive Monitoring tasks",
    author="Heitor Guimarães",
    url="https://github.com/Hguimaraes/beeSSL",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/Hguimaraes/beeSSL/issues",
        "Source Code": "https://github.com/Hguimaraes/beeSSL",
    },
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "pandas==2.0.1",
        "torch==2.0.0",
        "torchaudio==2.0.1",
        "speechbrain>=0.5.14",
        "transformers==4.28.1"
    ],
    python_requires=">=3.8"
)