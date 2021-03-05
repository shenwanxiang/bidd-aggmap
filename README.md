# AggMap

![alt text](./doc/logo.png =50x50)





## A Fully Automated and Fast Flow to Generate 2D Biometric Signatures for Convolutional Models


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
mp.fit(cluster_channels = 5)

#transform
X = mp.transform(dfx.values)

#save AggMap object
mp.save('./test.mp')
```
