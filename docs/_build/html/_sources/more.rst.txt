More information
================

Permission
----------

Permission can help avoid unwanted alteration of data, but limits
considerably the number of available methods. With an increasing level
of permission for the loader, the options are :

-  'read' : Read only the data on disk and change it on memory.
-  'write' : Add only new datatype and new ID. Any operation that would
   remove or change data or metadata will raise an error.
-  'overwrite' (default) : Do all operations and write on disk.

Performance
-----------

It is recommended to add timeseries with the method ``add_ID``.

Multiprocessing
~~~~~~~~~~~~~~~

Multiprocessing is possible. Parallel attribute must be set to ``True``
if the loader is intended to be used in parallel. The metadata output
will therefore be base on loader datatype and split. To generate the
``metadata.pqt``, the method ``merge_metadata`` must be used after all
the parallel processes are completed.

How the data is stored
----------------------

The data
~~~~~~~~

Data is indexed using ``(ID, timestamp)``.

======= ============== ============== = ==============
ID: str timestamp: str feature 1: any … feature n: any
======= ============== ============== = ==============
======= ============== ============== = ==============

Metadata
~~~~~~~~

Metadata is indexed using ``datatype``. List elements in metadata are
unique and unordered.

============= ================ ============== ===================
datatype: str split: list[str] IDs: list[str] features: list[str]
============= ================ ============== ===================
============= ================ ============== ===================

The splits
~~~~~~~~~~

First, the ``split`` is defined and store in metadata. The data is
written on disk accordingly.

Then, ``split_names`` is an array of names to tell which names to work
with from the original ``split``. ``split_indices`` can be used instead.
