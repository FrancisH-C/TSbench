.. raw:: org

   #+property: header-args :tangle yes

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

   path = "data/quickstart/"
   datatype = "Stock"
   loader = TSloader(path, datatype)

Add data
~~~~~~~~

Two different way to add data are presented, starting with the most
useful. For more ways to add data see examples in the notebooks. Here is
`an example <../notebooks/example_operations.ipynb>`__ of such a
notebook.

#. Add ID

   Create a DataFrame with features

   .. code:: python

      d_ID = {"feature0": list(range(10)), "feature1": list(range(10,20))}

      df_ID = pd.DataFrame(data=d_ID)
      print(df_ID)

   Add to a given ID

   .. code:: python

      ID = "added_ID"
      loader.add_ID(df_ID, ID=ID, collision="overwrite")
      print(loader.df) # in memory

#. Add feature

   It is definitely easier to add the datatype correctly in the first
   place than to use ``add_feature``. Here, we add feature for
   ``name1``.

   Create a DataFrame

   .. code:: python

      ID = "added_ID"
      feature = "added_feature"
      d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(10, 14))}
      df_feature = pd.DataFrame(data=d_feature)

   .. code:: python

      loader.add_feature(df_feature, ID=ID, feature=feature)
      print(loader.df)

   :RESULTS:

#. Add datatype

   #. Create a complete datatype

      .. code:: python

         d_dtype = {"ID": np.hstack((["name1" for _ in range(5)],
                            ["name2" for _ in range(5)])),
                 "dim" : ["0" for _ in range(10)],
                "timestamp": list(map(str, range(0,10))),
                "feature0": list(range(10)), "feature1": list(range(10,20))}

         df_dtype = pd.DataFrame(data=d_dtype)
         print(df_dtype)

      .. code:: python

         loader.initialize_datatype(df=df_dtype)
         print(loader.df)
         print(loader.metadata)

Read and write data
~~~~~~~~~~~~~~~~~~~

#. Write data on disk

   Don't forget to write the previous changes on disk.

   .. code:: python

      loader.write()

#. Read data

   .. code:: python

      read_loader = TSloader(path, datatype, permission="read")

   .. code:: python

      print(read_loader.df)

#. Read metadata

   An important note about metadata, is that it is unordered. Thus, the
   order can change without notice.

   .. code:: python

      print(read_loader.metadata)

   Remove the data for future runs

   .. code:: python

      loader.rm_dataset()
