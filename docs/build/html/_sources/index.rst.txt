.. aggmap documentation master file, created by
   sphinx-quickstart on Tue Jul 26 16:11:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to aggmap's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


aggmap
========

aggmap package is developed to enhance the learning of the unordered and unstructured omics data. aggmap includes theree major modules or classes, they are:

- AggMap: an unsupervised novel feature aggregation tool, which is developed to Aggregate and Map omics features into multi-channel 2D spatial-correlated image-like feature maps (Fmaps) based on their intrinsic correlations.
- AggMapNet: a simple yet efficient CNN-based supervised learning model, which is developed for learning the structured AggMap output Fmaps.
- Explainers: the model explaination modules (Shapley-explainer and Simply-explainer), which are developed to calculate the local and global feature importance, and based on the 2D-grid of AggMap, we can generate the explaination saliency-map based on the the feature importance score 

Look how easy it is to use:

.. code:: python3
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from aggmap import AggMap, AggMapNet

    # Data loading
    data = load_breast_cancer()
    dfx = pd.DataFrame(data.data, columns=data.feature_names)
    dfy = pd.get_dummies(pd.Series(data.target))

    # AggMap object definition, fitting, and saving 
    mp = AggMap(dfx, metric = 'correlation')
    mp.fit(cluster_channels=5, emb_method = 'umap', verbose=0)
    mp.save('agg.mp')

    # AggMap visulizations: Hierarchical tree, embeddng scatter and grid
    mp.plot_tree()
    mp.plot_scatter(enabled_data_labels=True, radius=5)
    mp.plot_grid(enabled_data_labels=True)

    # Transoformation of 1d vectors to 3D Fmaps (-1, w, h, c) by AggMap
    X = mp.batch_transform(dfx.values, n_jobs=4, scale_method = 'minmax')
    y = dfy.values

    # AggMapNet training, validation, early stopping, and saving
    clf = AggMapNet.MultiClassEstimator(epochs=50, gpuid=0)
    clf.fit(X, y, X_valid=None, y_valid=None)
    clf.save_model('agg.model')

    # Model explaination by simply-explainer: global, local
    simp_explainer = AggMapNet.simply_explainer(clf, mp)
    global_simp_importance = simp_explainer.global_explain(clf.X_, clf.y_)
    local_simp_importance = simp_explainer.local_explain(clf.X_[[0]], clf.y_[[0]])

    # Model explaination by shapley-explainer: global, local
    shap_explainer = AggMapNet.shapley_explainer(clf, mp)
    global_shap_importance = shap_explainer.global_explain(clf.X_)
    local_shap_importance = shap_explainer.local_explain(clf.X_[[0]])



Installation
------------

Install $project by running:

    install project

Contribute
----------

- Issue Tracker: github.com/$project/$project/issues
- Source Code: github.com/$project/$project

Support
-------

If you are having issues, please let us know.
We have a mailing list located at: project@google-groups.com

License
-------

The project is licensed under the BSD license.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
