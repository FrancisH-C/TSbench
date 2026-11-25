Examples
========

Each script under ``examples/experiments/`` is a **runnable experiment**: it
builds an ``ExperimentConfig`` and runs it (see :doc:`quickstart` for the config
shape, and :doc:`more` for how data is stored). Run them with ``python``:

.. code:: shell

   python examples/experiments/minimal.py     # generate a series, then forecast it
   python examples/experiments/benchmark.py   # score forecasters against a known DGP
   python examples/experiments/template.py    # annotated full template (every config knob)

``minimal.py`` — generate, then forecast
----------------------------------------------------

Generate a series with an ARMA model, then train and forecast it.

.. literalinclude:: ../../examples/experiments/minimal.py
   :language: python

``benchmark.py`` — score against a known DGP
----------------------------------------------------

Use an ARMA process as the data-generating "truth", hold out the tail of the
series, and score a Constant baseline and an ARMA forecaster against the actual
values with MAE / RMSE.

.. literalinclude:: ../../examples/experiments/benchmark.py
   :language: python

``template.py`` — every config knob
----------------------------------------------------

An annotated, copy-paste starting point that covers the full config surface,
including the advanced escape hatches: a custom ``initialize``, an ``Output``
transform, and the ``Process``-based pre/run-process stages.

.. literalinclude:: ../../examples/experiments/template.py
   :language: python
