.. _subsurface-manual:

########################
vector-geology Documentation
########################

:Release: |version|
:Date: |today|
:Source: `github.com/vector-raw-materials/vector-geology <https://github.com/vector-raw-materials/vector-geology>`_

----


.. include:: ../../README.rst
  :start-after: sphinx-inclusion-marker


Explore Our Comprehensive Guides
--------------------------------
.. toctree::
   :maxdepth: 2
   :caption: Galleries

   examples/index
   external/external_examples

.. include:: examples/index.rst

.. include:: external/external_examples.rst  


Requirements
------------
The requirements for the core functionality of the package are:

.. include:: ../../requirements.txt
   :literal:

Optional requirements
---------------------

There are many optional requirements, depending on the data format you want to
read/write. Currently, the ``requierements_opt.txt`` reads like:

.. include:: ../../requirements_opt.txt
   :literal:


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   manual
   changelog
   contributing
   maintenance

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Galleries

   examples/index
   external/external_examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   code-geological-formats
   code-interfaces
   code-reader
   code-structs
   code-utils
   code-viz
   code-writer
