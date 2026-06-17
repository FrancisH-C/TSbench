Installation
============

Install TSbench into a virtual environment, then add any optional dependency
groups you need. The optional R model support has its own setup further down.

Virtual environment setup
-------------------------

Install TSbench into a virtual environment so its dependencies stay isolated
from the system Python.

**Linux**

.. code-block:: shell

   python -m venv .venv
   source .venv/bin/activate

**Windows (conda)**

.. code-block:: shell

   conda create -n TSbench python=3.10
   conda activate TSbench

To expose the environment as a Jupyter kernel:

.. code-block:: shell

   python -m pip install ipykernel ipython
   python -m ipykernel install --name TSbench --user

R support (rGARCH via rpy2)
---------------------------

TSbench can run R models (``rGARCH``, wrapping R's ``rugarch`` / ``rmgarch``)
directly from Python through `rpy2 <https://rpy2.github.io/>`_. R support is
**optional** and **Linux only**.

The rest of this page is a complete, step-by-step guide for **Ubuntu**. Brief
notes for Arch Linux and other distributions follow at the end.

What you need
~~~~~~~~~~~~~

R support has three layers, installed in this order:

1. **A recent R interpreter.** The current ``rpy2`` requires **R ≥ 4.5.0**.
   Ubuntu's default repositories ship an older R, so you must install R from
   the CRAN apt repository (below).
2. **The CRAN R packages** that ``rGARCH`` wraps: ``rugarch``, ``rmgarch``,
   ``MTS``, and ``jsonlite``.
3. **The Python bridge**, ``rpy2``, installed via the ``[R]`` extra of TSbench.

.. note::

   Activate your virtual environment before installing the Python pieces, so
   ``rpy2`` lands in the project environment rather than the system Python.
   See `Virtual environment setup`_ above.

Ubuntu
~~~~~~

Step 1 -- Install a recent R from CRAN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the CRAN apt repository, which provides the latest R 4.x. These are the canonical CRAN
steps (see the upstream `CRAN apt instructions
<https://cloud.r-project.org/bin/linux/ubuntu/>`_ for the authoritative,
release-specific version):

.. code-block:: shell

   # Helper packages used to add the repository
   sudo apt update -qq
   sudo apt install --no-install-recommends software-properties-common dirmngr

   # Add the CRAN signing key
   wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
     | sudo tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc > /dev/null

Next, register the CRAN repository. Its URL ends with your Ubuntu release
**codename** followed by ``-cran40`` -- for example ``noble-cran40`` on 24.04 or
``jammy-cran40`` on 22.04. Rather than hard-coding the codename, detect it
automatically so the same commands work on any release.

Read the codename from ``/etc/os-release``, which is always present and needs no
extra package. ``$(. /etc/os-release && echo $VERSION_CODENAME)`` sources the
file in the substitution subshell and echoes its ``VERSION_CODENAME``
(e.g. ``noble``):

.. code-block:: shell

   echo "deb [signed-by=/etc/apt/trusted.gpg.d/cran_ubuntu_key.asc] https://cloud.r-project.org/bin/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME)-cran40/" \
     | sudo tee /etc/apt/sources.list.d/r-project.list > /dev/null

Verify you got R ≥ 4.5.0:

.. code-block:: shell

   R --version

Step 2 -- Install system build dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both ``rpy2`` and several CRAN packages compile from source, so you need a C /
Fortran toolchain, the Python headers, and a few numeric libraries:

.. code-block:: shell

   sudo apt install -y \
     build-essential gfortran python3-dev \
     r-base-dev libgmp-dev libmpfr-dev libnlopt-dev

* ``build-essential`` / ``gfortran`` -- C and Fortran compilers used to build R
  packages and the ``rpy2`` C extension.
* ``python3-dev`` -- Python headers required to build ``rpy2``.
* ``r-base-dev`` -- R headers and the ``R CMD`` toolchain.
* ``libgmp-dev`` / ``libmpfr-dev`` / ``libnlopt-dev`` -- numeric libraries
  pulled in by ``rugarch`` / ``rmgarch`` and their dependencies.

Step 3 -- Install the CRAN R packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   R -e 'install.packages(c("jsonlite","rugarch","rmgarch","MTS"), repos="https://cloud.r-project.org")'

This compiles the packages into your user R library and can take several
minutes. It must finish without error before continuing.

R library location
^^^^^^^^^^^^^^^^^^

Run ``R`` as your normal user, not with ``sudo``. As your user, R installs and
loads packages from a per-user library (the first entry of ``.libPaths()``).
This is the recommended setup and what Step 3 uses. Because ``rpy2`` starts R as the user running your Python process, keeping everything in this user library is what lets
``rpy2`` find the packages you installed. Check where R looks with:

.. code-block:: shell

   R -e '.libPaths()'

To make the user library explicit and stable across sessions: 

.. code-block:: shell

   R_LIBS_USER="$HOME/R/library"
   mkdir -p "$R_LIBS_USER"
   echo "R_LIBS_USER=$R_LIBS_USER" >> ~/.Renviron


Step 4 -- Install TSbench with R support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With your virtual environment active:

.. code-block:: shell

   python -m pip install .[R]        # or  .[all]  for everything
   python -m pip install -e .[R]     # editable install for development

Verifying the installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm the Python bridge can reach R and the wrapper imports:

.. code-block:: shell

   python -c "import rpy2.situation; rpy2.situation.report()"
   python -c "from TSbench.TSmodels import rGARCH; print('rGARCH OK')"

The first command prints where ``rpy2`` found R (its ``R_HOME``, R version, and
linked libraries). The second confirms TSbench's optional R import path is
active.

Finally, run the R-marked tests (skipped by default):

.. code-block:: shell

   python -m pytest --run-R

Troubleshooting
~~~~~~~~~~~~~~~

**``package 'X' was found, but >= Y is required``**
    Your existing CRAN packages are stale. Refresh your user library, then
    reinstall:

    .. code-block:: shell

       R -e 'update.packages(ask=FALSE, repos="https://cloud.r-project.org", lib.loc=.libPaths()[1])'

**``rpy2`` build fails / cannot find R headers**
    Make sure ``r-base-dev`` and ``python3-dev`` are installed (Step 2) and that
    ``R`` is on your ``PATH`` (``which R``). ``rpy2`` locates R by running
    ``R RHOME`` at build time.

**``R_HOME`` not found at runtime**
    If ``rpy2`` builds but fails to start R, set ``R_HOME`` explicitly:

    .. code-block:: shell

       export R_HOME="$(R RHOME)"

**``libR.so: cannot open shared object file``**
    R was built without a shared library, or the loader cannot find it. Reinstall
    ``r-base`` from the CRAN repo (Step 1), or add R's library directory to the
    loader path:

    .. code-block:: shell

       export LD_LIBRARY_PATH="$(R RHOME)/lib:$LD_LIBRARY_PATH"

**``rugarch`` fails to load with ``undefined symbol: VECTOR_PTR``**
    A ``pytest`` run aborts while loading the R namespace:

    .. code-block:: text

       rpy2.rinterface_lib.embedded.RRuntimeError: Error: package or namespace
       load failed for 'rugarch' ... undefined symbol: VECTOR_PTR

    Your ``Rcpp`` / ``rugarch`` binaries were compiled under an older R (e.g.
    4.3.3). R 4.5.0 removed the internal ``VECTOR_PTR`` macro, so the old
    binaries are incompatible. ``apt`` has no prebuilt packages for R 4.5+
    (hence ``Unable to locate package r-cran-rugarch``) -- R must recompile them
    from source. Start R as your normal user (see `R library location`_), force a
    rebuild against the new engine, then quit:

    .. code-block:: r

       update.packages(checkBuilt = TRUE, ask = FALSE)
       install.packages(c("Rcpp", "rugarch"))   # if the error persists
       quit(save = "no")

    If R prompts for a CRAN mirror, choose ``1`` (the global cloud mirror).
    Afterwards the ``pytest`` run completes without the dynamic-linking error.

Removing R support
~~~~~~~~~~~~~~~~~~

R support is fully optional. To remove just the Python bridge:

.. code-block:: shell

   python -m pip uninstall rpy2

The core Python models continue to work; ``rGARCH`` simply stops being
importable.

Other distributions
~~~~~~~~~~~~~~~~~~~

**Arch Linux.** Install the build toolchain, then pull the packages from the
AUR (``r-rugarch``, ``r-rmgarch``, ``r-mts``, ``r-jsonlite``) or use the manual
route below:

.. code-block:: shell

   sudo pacman -S gcc-fortran tcl tk
   R -e 'install.packages(c("jsonlite","rugarch","rmgarch","MTS"), repos="https://cloud.r-project.org")'
   python -m pip install .[R]

**Any distribution (manual).** Install a recent R (≥ 4.5.0) through your
package manager, then:

.. code-block:: shell

   R -e 'install.packages(c("jsonlite","rugarch","rmgarch","MTS"), repos="https://cloud.r-project.org")'
   python -m pip install .[R]
