![logo](docs/logo_black.png)
#

[![Build Status](https://travis-ci.org/betas-org/betas.svg?branch=master)](https://travis-ci.org/betas-org/betas)
[![Coverage{:target="_blank"} Status](https://coveralls.io/repos/github/betas-org/betas/badge.svg?branch=master)](https://coveralls.io/github/betas-org/betas?branch=master){:target="_blank"}
![Language](https://img.shields.io/badge/language-python-blue.svg)
![License](https://img.shields.io/badge/license-MIT-black.svg)


This package allows users to visualize model performance, model fit, or model assumptions with one line of code by creating an instance of a plotting class and reusing that instance for various plotting methods.

## Team Members
Joel Stremmel

Yiming Liu

Cathy Jia

Monique Bi

Arjun Singh

## Data

Data Set 1:

Data Set 2:

## Software
**Programming Languages**

Python

**Python Packages**

[numpy](http://www.numpy.org){:target="_blank"} >= 1.13.1

[pandas](https://pandas.pydata.org){:target="_blank"} >= 0.23.1

[matplotlib](https://matplotlib.org){:target="_blank"} >= 2.0.2

[seaborn](https://seaborn.pydata.org){:target="_blank"} >= 0.9.0

[scikit-learn](https://scikit-learn.org){:target="_blank"} >= 0.20.2

[statsmodels](https://www.statsmodels.org){:target="_blank"} >= 0.9.0

[dash](https://dash.plot.ly){:target="_blank"} >= 0.43.0

[bokeh](https://bokeh.pydata.org){:target="_blank"} >= 1.0.4

## License Information

MIT License


## Structure
This package has the following structure:
```
betas/
  |- betas/
     |- __init__.py
     |- Binary Classification/
        |- app_bokeh.py
        |- app_dash.py
        |- binary_score_plot.py
        |- data.py
        |- download.js
        |- probability_plot.py
        |- sample_probability_plot.ipynb
        |- test_binary_score_plot.py
        |- test_probability_plot.py
        |- tool.py
     |- Linear Regression Analysis/
        |- README.md
        |- test_analysis_plot.ipynb
        |- model_diagnostics.py
        |- analysis_plot.py
        |- test_analysis_plt.ipynb
        |- test_analysis_plt.html
     |- pca/
        |- README.md
        |- pca_evaluate.py
        |- test_pca_evaluate.ipynb
        |- test_pca_evaluate.html
  |- docs/
     |- Functional_Specification.pdf
     |- Project_Component_Specification.pdf
     |- Technology_Review.pdf
     |- logo_black.png
     |- logo_white.png
  |- environment.yml
  |- LICENSE.txt
  |- README.md
  |- requirements.txt
  |- setup.cfg
  |- setup.py
```

## Installation
