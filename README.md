# bidd-aggmap


## AggMap -  Enhancing Feature Representation Ability by Agglomeration and Multi-Channel Operations

## AggMap - Enhancing Feature Representation Ability by Unsurpervised Embedding  and Multi-Channel Operations


### 2D signature map


## Installation

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


## Usage


```python
from aggmap import AggMap

#create AggMap object
mp = AggMap(dfx, metric = 'correlation')

#fit AggMap
mp.fit(cluster_channels = 3)

#transform
X = mp.transform(dfx.values)

#save AggMap object
mp.save('./test.mp')
```
