.. aggmap documentation master file, created by
   sphinx-quickstart on Tue Jul 26 16:11:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../images/logo.png
  :width: 600
  :align: center

Jigsaw-like aggmap: A Robust and Explainable Multi-Channel Omics Deep Learning Tool
===========================================================================

aggmap package is developed to enhance the learning of the unordered and unstructured omics data. aggmap includes theree major modules, they are:

- **AggMap**: an unsupervised novel feature aggregation tool, which is developed to Aggregate and Map omics features into multi-channel 2D spatial-correlated image-like feature maps (Fmaps) based on their intrinsic correlations.
- **AggMapNet**: a simple yet efficient CNN-based supervised learning model, which is developed for learning the output structured Fmaps of AggMap .
- **Explainers**: the model explaination modules (Shapley-explainer and Simply-explainer), which are developed to calculate the local and global feature importance, and based on the 2D-grid of AggMap, we can generate the explaination saliency-map based on the the feature importance score 


The details for the underlying mathematics can be found in
`our paper on NAR <https://academic.oup.com/nar/article/50/8/e45/6517966>`_:

Shen W X, Liu Y, Chen Y, et al. AggMapNet: enhanced and explainable low-sample omics deep learning with feature-aggregated multi-channel networks[J]. Nucleic Acids Research, 2022, 50(8): e45-e45.

You can find the software `on github <https://github.com/shenwanxiang/bidd-aggmap/>`_.

**Installation**

PyPI install, presuming you have all its requirements installed:

.. code:: bash
    # create an aggmap env
    conda create -n aggmap python=3.7
    conda activate aggmap
    pip install --upgrade pip
    pip install aggmap    




.. toctree::
   :maxdepth: 2
   :caption: Background on aggmap:

   how_aggmap_works
   performance

.. toctree::
   :maxdepth: 2
   :caption: Examples of aggmap usage

   MNIST/F-MNIST reconstruction
   Multi-omcs based Covid-19 detection
   Breast Cancer Detection 

.. toctree::
   :caption: API Reference:

   api
   
   

**Contribute**
----------

- Issue Tracker: github.com/shenwanxiang/bidd-aggmap/issues
- Source Code: github.com/shenwanxiang/bidd-aggmap/

**Support**
-------

If you are having issues, please let us know.
We have a mailing list located at: wanxiang.shen@u.nus.edu

**License**
-------

The project is licensed under the  GPL-3.0 license.



**Indices and tables**
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
