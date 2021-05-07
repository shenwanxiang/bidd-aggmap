
<a href="url"><img src="./doc/logo.png" align="left" height="170" width="130" ></a>

# AggMap

## A Fully Automatic and Fast Flow to Generate 2D Multi-Channel Biometric Signatures for Convolutional Models


----
### How It Works?

- AggMap flowchart of feature mapping and agglomeration into ordered (spatially correlated) multi-channel feature maps (Fmaps)

![how-it-works](./doc/how-it-works.png)

----
### Proof-of-Concepts on MNIST Dataset

- It can reconstruct to the original image from completely randomly permuted (disrupted) MNIST data:
![reconstruction](./doc/reconstruction.png)

`Org1`: the original grayscale images (channel = 1), `OrgRP1`: the randomized images of Org1 (channel = 1), `RPAgg1, 5`: the reconstructed images of `OrgPR1` by AggMap feature restructuring (channel = 1, 5 respectively, each color represents features of one channel). `RPAgg5-tkb`: the original images with the pixels divided into 5 groups according to the 5-channels of `RPAgg5` and colored in the same way as `RPAgg5`.

----

### Example for Restructured Fmaps

- The example on WDBC dataset: click [here](https://github.com/shenwanxiang/bidd-aggmap/blob/master/paper/00_example_breast_cancer/03_BCD_feature_maps.ipynb) to find out more!

![Fmap](./doc/WDBC.png)


----

### Installation

install aggmap by:

```bash
git clone https://github.com/shenwanxiang/bidd-aggmap.git
cd bidd-aggmap
pip install -r requirements.txt --user

# add molmap to PYTHONPATH
echo export PYTHONPATH="\$PYTHONPATH:`pwd`" >> ~/.bashrc

# init bashrc
source ~/.bashrc
```


### Usage


```python
from aggmap import AggMap

#create AggMap object
mp = AggMap(dfx, metric = 'correlation')

#fit AggMap
mp.fit(cluster_channels = 5)

#transform
X = mp.transform(dfx.values)

#save AggMap object
mp.save('./test.mp')
```
