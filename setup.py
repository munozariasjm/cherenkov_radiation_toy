from setuptools import setup, find_packages

setup(
    name="cherenkov_toy",
    version="0.0.1",
    packages=find_packages(),
    description="Toy theoretical model for Cherenkov radiation",
    author="munozariasjm",
    install_requires=["numpy", "scipy", "matplotlib"],
)