from setuptools import setup, find_packages

setup(
    name="mini-cuda-llm",
    version="0.1.0",
    description="Python wrapper for mini CUDA vector add",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "matplotlib"],
)
