
Code Description
================

...

Some technical information on qgs:

* qgs is optimized to run ensemble of initial conditions on multiple core, using `Numba`_ jit-compilation and
  `multiprocessing`_ workers. As such, qgs might not work on Windows (not tested).

* qgs has a `tangent linear model`_ optimized to run ensemble of initial conditions as well, with a broadcasted
  integration of the tangent model thanks to `Numpy`_.


Modules
-------


.. toctree::
   :maxdepth: 2

   technical/configuration
   technical/inner_products
   technical/tensors
   technical/functions
   technical/integrators
   technical/misc

.. _Numba: https://numba.pydata.org/
.. _Numpy: https://numpy.org/
.. _multiprocessing: https://docs.python.org/3.7/library/multiprocessing.html#module-multiprocessing
.. _tangent linear model: http://glossary.ametsoc.org/wiki/Tangent_linear_model
