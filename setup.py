from setuptools import setup, find_packages

setup(
    name="culture_aware_autism_screening",
    version="1.0.0",
    description="A framework for detecting cultural bias in ASD screening tools",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "lightgbm>=3.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.8",
)
