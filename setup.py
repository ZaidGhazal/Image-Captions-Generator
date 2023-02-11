"""This module contains the setup script for the project."""
from setuptools import setup

REQUIRED_PACKAGES = [
    required
    for required in open("requirements.txt", encoding="utf-8").read().splitlines()
]


setup(
    name="Image Captions Generator",
    author="Zaid Ghazal",
    author_email="zaid.ghazal20@gmail.com",
    description="A web application that generates captions for images",
    long_description=open("README.md").read(),
    python_requires=">=3.8",
    version="1.0.0",
    packages=["src"],
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
)
