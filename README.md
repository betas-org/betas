![logo](docs/logo_black.png)
#

[![Build Status](https://travis-ci.org/betas-org/betas.svg?branch=master)](https://travis-ci.org/betas-org/betas)
[![Coverage Status](https://coveralls.io/repos/github/betas-org/betas/badge.png?branch=master)](https://coveralls.io/github/betas-org/betas?branch=master)
![Language](https://img.shields.io/badge/language-python-blue.svg)
![License](https://img.shields.io/badge/license-MIT-black.svg)
![Version](https://img.shields.io/pypi/v/betas.svg)

This package allows users to visualize model performance, model fit, or model assumptions with one line of code by creating an instance of a plotting class and reusing that instance for various plotting methods.

## Team Members
Joel Stremmel

Yiming Liu

Cathy Jia

Mengying Bi

Arjun Singh

## Data

Data Set 1: The Spam dataset ([Source](https://web.stanford.edu/~hastie/ElemStatLearn/))

Data Set 2: The College dataset ([Source](http://www-bcf.usc.edu/~gareth/ISL/))

Data Set 3: Iris - (Example dataset in `seaborn`)

Data Set 4: The Auto dataset ([Source](http://www-bcf.usc.edu/~gareth/ISL/))

## Software
**Programming Languages**

[Python](https://www.python.org)

**Python Packages**

[numpy](http://www.numpy.org) >= 1.13.1

[pandas](https://pandas.pydata.org) >= 0.23.1

[matplotlib](https://matplotlib.org) >= 2.0.2

[seaborn](https://seaborn.pydata.org) >= 0.9.0

[scipy](http://scipy.github.io/devdocs/) == 1.2.0

[scikit-learn](https://scikit-learn.org) >= 0.20.2

[statsmodels](https://www.statsmodels.org) >= 0.9.0

[dash](https://dash.plot.ly) >= 0.43.0

[bokeh](https://bokeh.pydata.org) >= 1.0.4

## License Information

MIT License


## Structure
This package has the following structure:
```
betas/
  |- betas/
     |- __init__.py
     |- binary_score_diagnostics.py
     |- binary_score_plot.py
     |- clustering_evaluate.py
     |- download.js
     |- pca_evaluate.py
     |- README.md
     |- regression_analysis_plot.py
     |- regression_diagnostics.py
     |- setup.cfg
     |- test_analysis_plot.py
     |- test_binary_score_plot.py
     |- test_clustering_evaluate.py
     |- test_pca_evaluate.py
     |- tool.py
  |- data/
     |- college.csv
     |- spam.data.txt
     |- spam.traintest.txt
     |- spam_score_label.csv
  |- dist/
     |- betas-v1.0.tar.gz
  |- docs/
     |- Functional_Specification.pdf
     |- Project_Component_Specification.pdf
     |- Project_Summary.pdf
     |- Technology_Review.pdf
     |- logo_black.png
     |- logo_white.png
  |- example/
     |- demo_analysis_plot.ipynb
     |- demo_clustering_evaluate.ipynb
     |- demo_pca_evaluate.ipynb
  |- environment.yml
  |- LICENSE.txt
  |- README.md
  |- requirements.txt
  |- setup.cfg
  |- setup.py
```

## Installation
`pip install betas`
