#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="beessl",
    version="0.1.0",
    description="HUnEE-B: beeHive monitoring Universal pErformancE Benchmark",
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
        "pandas==2.1.0",
        "torch==2.3.0",
        "torchaudio==2.3.0",
        "torchvision==0.18.0",
        "speechbrain==0.5.14",
        "transformers==4.28.1",
        "scikit-learn==1.2.2",
        "einops==0.7.0",
        "timm==0.4.5",
        "easydict==1.13",
        "pytorch_lightning==2.2.5"
        "nnAudio==0.3.3",
        "librosa==0.10.2.post1"
    ],
    python_requires=">=3.10"
)