.. raw:: org

   #+property: header-args :tangle yes

Quickstart
==========

Installation
------------

See :doc:`installation` for dependency extras, R support, and alternative
environment setups.

#. Clone the package

   .. code:: shell

      git clone "https://github.com/FrancisH-C/TSbench"

#. Create and activate a virtual environment

   .. code:: shell

      python -m venv .venv
      source .venv/bin/activate

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
   from TSbench import TSloader, DataFormat

.. code:: python

   path = "data/quickstart/"
   datatype = "Stock"
   loader = TSloader(path, datatype)

Add data
~~~~~~~~

Two different way to add data are presented, starting with the most
useful.

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

Experiments
-----------

An ``Experiment`` runs the generate → train → forecast → output pipeline from a
typed ``ExperimentConfig``. The config is assembled from small dataclasses:
``General`` (paths, parallelism), one ``Stage`` per generate/train/forecast
step, and an optional ``Output`` (metrics). They provide defaults, validation,
and editor autocomplete.

.. code:: python

   from numpy.random import Generator, PCG64

   from TSbench.experiment import Experiment, ExperimentConfig, General, Stage, Output
   from TSbench.metrics import mae, rmse
   from TSbench.TSmodels import ARMA

   model = ARMA(
       lag=1, rg=Generator(PCG64(1234)), dim_label=["d0"], feature_label=["feature"]
   )

   config = ExperimentConfig(
       general=General(data_path="data/exp/"),
       generate=Stage(models=[model], params={"N": 100}),
       # Train on the first 80 points and forecast the last 20, so the forecast
       # overlaps the generated values and can be scored.
       train=Stage(models=[model], input_loaders_params={"end_index": 80}),
       forecast=Stage(
           models=[model], params={"T": 20}, input_loaders_params={"end_index": 80}
       ),
       output=Output(metrics=[mae, rmse]),
   )

   exp = Experiment(config)
   exp.run()              # no flags = run every configured stage
   print(exp.results)     # {ID: {model: {metric_name: value}}}

``run()`` with no arguments auto-detects and runs every configured stage; pass
explicit flags (e.g. ``run(generate=True)``) to run a subset. The only path the
experiment reads or writes is the dataset directory named in ``General`` (see
:doc:`more` for how data is stored). For complete, runnable scripts, see
:doc:`examples`.
