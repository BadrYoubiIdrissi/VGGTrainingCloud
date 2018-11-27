from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Training VGG model on audio',
      author='Badr YOUBI IDRISSI',
      author_email='badryoubiidrissi@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'pysoundfile'
      ],
      zip_safe=False)