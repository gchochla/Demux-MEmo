from setuptools import setup, find_packages

setup(
    name="emorec-demux-memo",
    version="1.0.0",
    author="Georgios Chochlakis, Gireesh Mahajan",
    author_email="chochlak@usc.edu, type591234@gmail.com",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "pandas",
        "torchvision",
        "scikit-learn",
        "emoji",
        "ekphrasis",
        "matplotlib",
        "pyyaml",
        "langcodes",
        "language_data",
        "sacremoses",
        "protobuf",
        "sentencepiece",
    ],
    extras_require={"dev": ["black", "pytest"]},
)
