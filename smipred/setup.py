from setuptools import setup, find_packages

setup(
    name='smi-pred',
    version='1.0.0',
    author='Ho Yeon Jang',
    author_email='wkdghdus23@gmail.com',
    description='A unified BERT training package for property predictions',
    # Read the long description from the README file
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wkdghdus23/SMILES-BERT-Predictor',
    # Automatically find all packages in the directory
    packages=find_packages(),
    # List of dependencies required to run the package
    install_requires=[
        'torch>=2.7.0',
        'transformers>=4.57.1',
        'tqdm>=4.67.0',
        'pandas>=2.2.2',
        'numpy>=2.1.0',
    ],
    python_requires='>=3.9.0',
    # Classifiers to categorize the package on PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Framework :: PyTorch',
        'Author :: Ho Yeon Jang, Sogang University'
    ],
    # Define console scripts to allow running the package from the command line
    entry_points={
        'console_scripts': [
            'smipred=smipred.main:main'
        ],
    },
)
