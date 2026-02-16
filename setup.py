import setuptools

setuptools.setup(
    name="dataTools",
    version="1.0.0",
    author="SophieBauchinger",
    author_email="bauchinger@iau.uni-frankfurt.de",
    description="Useful tools for analysing atomospheric measurement data",
    url="https://github.com/SophieBauchinger/dataTools.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>3.6',
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'metpy'
    ]
)
