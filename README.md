# bidd-rsmap


## AggMap -  Enhancing Feature Representation Ability by Agglomeration and Multi-Channel Operations


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
mp.fit(var_thr = 1e-4, cluster_channels = 3,split_channels = True)

#transform
X = mp.transform(dfx.values)

#save AggMap object
mp.save('./test.mp')
```
