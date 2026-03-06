How to Contribute
=================

We welcome contributions to rdmpy! This project aims to build a comprehensive toolkit for analyzing UK rail incidents and delay propagation. Whether you're interested in data science, systems engineering, or railway operations, there are many ways you can help.

Types of Contributions
----------------------

We accept several types of contributions:

* **New datasets**: Additional railway data sources that complement our UK Rail Data Marketplace integration
* **Code improvements**: Bug fixes, performance optimizations, new analysis features
* **Documentation**: Improvements to existing documentation, tutorials, additional demos
* **Testing**: Unit tests, integration tests, data validation improvements
* **Analysis & Examples**: New Jupyter notebooks, use cases, visualizations, research insights
* **Data preprocessing enhancements**: Improvements to the preprocessor module for better data quality

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork locally**:

   .. code-block:: bash

      git clone https://github.com/your-username/rdmpy.git
      cd rdmpy

3. **Create a new branch** for your contribution:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

4. **Install the project in development mode**:

   .. code-block:: bash

      pip install -e .

5. **Make your changes** and test them locally

6. **Commit with a clear message**:

   .. code-block:: bash

      git commit -m "Add description of your changes"

7. **Push to your fork and submit a Pull Request**

Code Guidelines
---------------

* Follow PEP 8 style guidelines for Python code
* Add docstrings to new functions and classes
* Include unit tests for new features
* Update documentation when adding new functionality
* Keep commits focused and descriptive

Setting Up Your Environment
----------------------------

To set up a development environment:

.. code-block:: bash

   # Install dependencies
   pip install -r requirements.txt
   
   # For documentation development
   pip install -r docs/requirements.txt
   
   # For testing
   pip install pytest

Running Tests
-------------

.. code-block:: bash

   pytest tests/

Building Documentation Locally
-------------------------------

.. code-block:: bash

   cd docs
   make clean
   make html

Open ``docs/_build/html/index.html`` to view the built documentation.

Data Contribution Guidelines
-----------------------------

If you're contributing new data analysis or datasets:

* Ensure data sources are properly documented and cite the Rail Data Marketplace
* Include any preprocessing steps required
* Document any assumptions or limitations
* Provide example usage in a Jupyter notebook
* Ensure compliance with data licensing requirements

Contact & Questions
-------------------

For questions or suggestions, please:

* Open an issue on GitHub for bugs or feature requests
* Start a discussion for broader ideas or improvements
* Contact the project maintainers at ji-eun.byun@glasgow.ac.uk

Thank You
---------

Thank you for considering contributing to rdmpy! Your contributions help advance our understanding of railway system behavior and resilience.
