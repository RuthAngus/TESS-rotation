from setuptools import setup

setup(name='tess_rotation',
      version='0.1rc0',
      description='Tools for measuring TESS rotation periods',
      url='http://github.com/RuthAngus/TESS-rotation',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['tess_rotation'],
      include_package_data=True,
      install_requires=['numpy', 'tqdm', 'astropy', 'matplotlib', 'eleanor'],
      zip_safe=False)
