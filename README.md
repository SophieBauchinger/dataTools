# dataTools

dataTools is a Python library for dealing with atmospheric tracer measurements from ground-based stations and aircraft campaigns. 

## Table of Contents
- [About](#-about)
- [Installation](#-installation)
- [Usage](#-usage)

## About
Have a look at the wiki :) 

## Installation

To build the package, follow these steps:

```bash
# Open a terminal (Command Prompt or PowerShell for Windows, Terminal for macOS or Linux)

# Ensure Git is installed
# Visit https://git-scm.com to download and install console Git if not already installed

# Clone the repository
git clone https://github.com/SophieBauchinger/dataTools.git

3. ???
4. Profit
```

## Usage

```python
import dataTools
from dataTools.data.Caribic import Caribic

# create CARIBIC data object
caribic = Caribic()

# filter for latitudes > 30°N and return updated Caribic object
caribic_gt_30N = caribic.sel_latitudes(30, 90)

```
