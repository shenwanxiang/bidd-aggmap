
[![Example](https://img.shields.io/badge/Example-MNIST-green)](https://github.com/shenwanxiang/bidd-aggmap/tree/master/paper/example)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6474351.svg)](https://doi.org/10.5281/zenodo.6474351)
[![Paper](https://img.shields.io/badge/Paper-Nuclear%20Acid%20Research-brightgreen)](https://doi.org/10.1093/nar/gkac010)


# Jigsaw-like AggMap

## A Robust and Explainable Omics Deep Learning Tool

----

### Installation

install aggmap by:
```bash
# create an aggmap env
conda create -n aggmap python=3.7
conda activate aggmap
pip install --upgrade pip
pip install aggmap
```

----

### Usage

```python
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
```

