Quickstart
==========

Installation
------------

#. Clone the package

   .. code:: shell

      git clone "https://github.com/FrancisH-C/TSload"

#. Run the installation from the package root folder

   .. code:: shell

      python -m pip install -e .

Basics
------

Dataset
~~~~~~~

The collection of all the data stored in a single path is called the
``dataset``.

The data
~~~~~~~~

The ``datatype`` refers type of the data which informs about the
structure of the data. A given ``datatype`` as the exact same
``features``. ``Datatype`` is a collection of multiple categories of
input from different ``ID``. The ``datatype`` can be stored in a
sequence of ``split``.

The Metadata
~~~~~~~~~~~~

Every dataset has ``metadata`` that gives information about every
``datatype``. Here is a simple example metadata information.

::

   datatype = "Stocks"
   split = ["date1", "date2"]
   ID = ["ABC", "XYZ"]
   features = ["side", "quantity", "price"]

Note that the data is separated on two files using the split.

Usage
-----

Use exactly one loader per datatype.

Initialization
~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   import pandas as pd
   from TSload import TSloader, DataFormat

.. code:: python

   path = "data/example_simple"
   datatype = "simulated"
   loader = TSloader(path, datatype)

Write data
~~~~~~~~~~

#. Add datatype

   Create a DataFrame

   .. code:: python

      d = {"ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
          "timestamp": list(map(str, range(0,10))),
           "feature0": list(range(10)), "feature1": list(range(10,20))}
      df = pd.DataFrame(data=d)

   .. code:: python

      loader.initialize_datatype(df=df)
      loader.df

#. Add ID

   Create a DataFrame

   .. code:: python

      ID = "added_ID"
      d_ID = {"timestamp": list(map(str, range(0,5))), "feature0": list(range(5)) ,"feature1": list(range(10,15))}
      df = pd.DataFrame(data=d_ID)

   .. code:: python

      loader.add_ID(df, ID=ID, collision="overwrite")
      print(loader.metadata) # in memory
      print(loader.df) # in memory

#. Add feature

   It is definitely easier to add the datatype correctly in the first
   place than to use ``add_feature``. Here, we add feature for
   ``name1``.

   Create a DataFrame

   .. code:: python

      ID = "added_ID"
      feature = "added_feature"
      d = {"timestamp": list(map(str, range(4))), feature: list(range(10, 14))}
      df = pd.DataFrame(data=d)

   .. code:: python

      loader.add_feature(df, ID=ID, feature=feature)
      print(loader.df)

   :RESULTS:

#. Write data on disk

   Don't forget to write the changes on disk.

   .. code:: python

      loader.write()

Load data
~~~~~~~~~

.. code:: python

   read_loader = TSloader(path, datatype, permission="read")

#. Data

   .. code:: python

      print(read_loader.df)

#. Metadata

   An important note about metadata, is that it is unordered. Thus, the
   order can change without notice.

   .. code:: python

      print(read_loader.metadata)
