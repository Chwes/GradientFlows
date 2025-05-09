from setuptools import setup, find_packages

setup(
    name='gradientflow',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'torch',
        'hydra-core',
        'omegaconf',
    ],
    entry_points={
        'console_scripts': [
            'run-gradientflow=main:main',
        ]
    }
)