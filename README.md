![logo](docs/logo_black.png)
###

[![Build Status](https://travis-ci.org/betas-org/betas.svg?branch=master)](https://travis-ci.org/betas-org/betas)
[![Coverage Status](https://coveralls.io/repos/github/betas-org/betas/badge.svg?branch=master)](https://coveralls.io/github/betas-org/betas?branch=master)
![Language](https://img.shields.io/badge/language-python-blue.svg)
![Version](https://img.shields.io/pypi/v/betas.svg?colorB=orange)
![License](https://img.shields.io/badge/license-MIT-9cf.svg)
![Code Size](https://img.shields.io/github/languages/code-size/betas-org/betas.svg?colorB=pink)
![Contributors](https://img.shields.io/github/contributors/betas-org/betas.svg?colorB=blueviolet)


## Background
Our project aim to create a simple and convenient visualization tool, Betas, for data scientists and data analysts to analyze model performance with visualizations in Python. Users are able to simply run one-line code to generate custom plots for analyzing linear regression model with assumptions diagnostics, computing model scores in binary classification, and presenting performance for principal component analysis (PCA) and clustering. This tool also helps users to fit machine learning models to datasets without a detailed understanding of how the models work. Betas package is pip installable and easy to use by following our example IPython notebooks, in which we are using the Spam dataset and College dataset as demonstration. In addition, we have two interactive web dashboard designed for model diagnostics in linear regression and binary classification.

## Team Members
Joel Stremmel

Yiming Liu

Cathy Jia

Mengying Bi

Arjun Singh

## Data

Data Set 1: The Spam data ([Source](https://web.stanford.edu/~hastie/ElemStatLearn/))

Data Set 2: The Breast Cancer data ([Source](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/datasets/data))

Data Set 3: The College data ([Source](http://www-bcf.usc.edu/~gareth/ISL/))

Data Set 4: The Auto data ([Source](http://www-bcf.usc.edu/~gareth/ISL/))

## Software

**Programming Languages**

[Python](https://www.python.org)

**Python Packages**

[numpy](http://www.numpy.org) >= 1.13.1

[pandas](https://pandas.pydata.org) >= 0.23.1

[matplotlib](https://matplotlib.org) >= 2.0.2

[seaborn](https://seaborn.pydata.org) >= 0.9.0

[scipy](http://scipy.github.io/devdocs/) <= 1.2.0

[scikit-learn](https://scikit-learn.org) >= 0.20.2

[statsmodels](https://www.statsmodels.org) >= 0.9.0

[dash](https://dash.plot.ly) >= 0.43.0

[bokeh](https://bokeh.pydata.org) >= 1.0.4

## Structure
This package has the following structure. See betas library [documentation](https://github.com/betas-org/betas/blob/master/betas/README.md) for details.

```
betas/
  |- betas/
     |- README.md
     |- __init__.py
     |- binary_score_diagnostics.py
     |- binary_score_plot.py
     |- clustering_evaluate.py
     |- download.js
     |- pca_evaluate.py
     |- regression_analysis_plot.py
     |- regression_diagnostics.py
     |- setup.cfg
     |- test_analysis_plot.py
     |- test_binary_score_plot.py
     |- test_clustering_evaluate.py
     |- test_pca_evaluate.py
  |- data/
     |- college.csv
     |- spam.data.txt
     |- spam.traintest.txt
     |- spam_score_label.csv
  |- dist/
     |- betas-v1.1.tar.gz
  |- docs/
     |- Final_Presentation.pdf
     |- Functional_Specification.pdf
     |- Component_Specification.pdf
     |- Project_Summary.pdf
     |- Technology_Review.pdf
     |- logo_black.png
     |- logo_white.png
  |- examples/
     |- demo_analysis_plot.ipynb
     |- demo_binary_score_plot.ipynb
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

## License Information

[MIT License](https://github.com/betas-org/betas/blob/master/LICENSE.txt)
