from setuptools import setup, find_packages

setup(
    name="async-blt",
    version="0.1.0",
    description="asynchronous byte latent transformer implementation",
    author="laerdon",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "einops>=0.6.1",
        "aiofiles>=23.0.0",
        "tqdm>=4.65.0",
    ],
)
