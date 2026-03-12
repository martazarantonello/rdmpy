Getting Started
===============

Installation
------------

We recommend using a virtual environment to manage dependencies. Follow these steps to set up your environment:

**Step 1: Create a Virtual Environment**

.. code-block:: bash

   python -m venv rdmpy_env

**Step 2: Activate the Virtual Environment**

On Linux/macOS:

.. code-block:: bash

   source rdmpy_env/bin/activate

On Windows:

.. code-block:: bash

   rdmpy_env\Scripts\activate

**Step 3: Install rdmpy**

.. code-block:: bash

   pip install rdmpy

To install in editable mode (for development):

.. code-block:: bash

   pip install -e .

Verify Installation
--------------------

After installation, you can verify that rdmpy is installed correctly by running:

.. code-block:: python

   import rdmpy
   print("rdmpy version:", rdmpy.__version__)


