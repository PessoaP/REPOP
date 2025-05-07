from setuptools import setup, find_packages

print( find_packages() )
setup(
    name="repop",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch",
                      "numpy",
                      "matplotlib",
                      "scikit-learn"],
    author="Pedro Pessoa",
    author_email="ppessoa@asu.edu",
    description="REPOP - Library for REconstructing bacterial POpulations from Plate counts.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PessoaP/REPOP",
    license="CC-BY-NC-4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
)