import os
from setuptools import setup, find_packages


def read(fname):
    content = None
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        content = f.read()
    return content


setup(
    name='DeepRL-Algos',
    version='0.0.1',
    description=(
        '...' 
    ),
    author='Ritchie',
    author_email='ritchie-huang@outlook.com',
    maintainer='Ritchie',
    license='MIT License',
    packages=find_packages(),
    long_discription=read('README.md'),
    platforms=["all"],
    url='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries'
    ],
    install_requires=[
        'pandas',
        'scipy',
        'termcolor',
        'torch',
        'tqdm',
        'tensorflow',
        'PyYAML',
        'numpy',
        'matplotlib',
        'tensorboard',
    ]
)
