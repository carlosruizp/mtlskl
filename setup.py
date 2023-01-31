from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='mtlskl',
      version='0.1',
      description='Package for Multiple Task Learning using sklearn guidelines',
      long_description=readme(),
      url='http://github.com/storborg/funniest',
      author='Carlos Ruiz Pastor',
      author_email='carlosruizpastor@protonmail.com',
      license='MIT',
      packages=['mtlskl'],
      install_requires=[
          'scikit-learn', 'numpy', 'scipy'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      #scripts=['bin/funniest-joke'],
    )
