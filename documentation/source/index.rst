.. qgs documentation master file, created by
   sphinx-quickstart on Mon Jan 27 20:14:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================================
Quasi-Geostrophic Spectral model (qgs)
======================================


qgs is quasi-geostrophic atmospheric model which is integrally written in Python.
The lower boundary condition can be either a surface with orography or a dynamical ocean.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   files/general_information
   files/model_description
   files/technical_description
   files/user_guide
   files/references
   files/examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

About
=====

Part of the code comes from the Python `MAOOAM`_ implementation by Maxime Tondeur and Jonathan Demaeyer.

**Please cite the code description article if you use (a part of) this software for a publication:**

* Demaeyer J., De Cruz, L. and Vannitsem, S. , (2020). qgs: A flexible Python framework of reduced-order multiscale climate models.
  *Journal of Open Source Software*, **5**\(56), 2597, `https://doi.org/10.21105/joss.02597 <https://doi.org/10.21105/joss.02597>`_.

Please consult the qgs `code repository <http://www.github.com/Climdyn/qgs>`_ for updates.

qgs is licensed under the `MIT`_ license:

.. code-block:: none

   The MIT License (MIT)

   Copyright (c) 2020-2021 qgs Developers and Contributors


   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.


.. _MIT: https://opensource.org/licenses/MIT
.. _MAOOAM: https://github.com/Climdyn/MAOOAM