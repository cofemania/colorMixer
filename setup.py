from setuptools import setup
from Cython.Build import cythonize

setup(
    name='deltae_cython',
    ext_modules=cythonize("deltae_cython.pyx", compiler_directives={'language_level': "3"}),
    zip_safe=False,
)
