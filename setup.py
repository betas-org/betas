from distutils.core import setup
setup(
  name = 'betas',
  packages = ['betas'],
  version = 'v0.2.3',      
  license = 'MIT',
  description = 'This package allows users to visualize model performance, model fit, or model assumptions with one line of code by creating an instance of a plotting class and reusing that instance for various plotting methods.',
  author = 'Joel Stremmel, Yiming Liu, Cathy Jia, Monique Bi, Arjun Singh',
  author_email = 'jstremme@uw.edu, liuy379@uw.edu, cathyjia@uw.edu, mybi@uw.edu, arjuns13@uw.edu',
  url = 'https://github.com/betas-org/betas',
  download_url = 'https://github.com/betas-org/betas/archive/v0.2.3.tar.gz',
  keywords = ['data science', 'machine learning', 'data visualization', 'visualization', 'model performance', 'model evaluation'],
  install_requires = [
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'scikit-learn',
          'statsmodels',
          'scipy',
          'bokeh',
          'dash'
      ],
  classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)