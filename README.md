# bidd-rsmap


## RFMap - An Efficient Shallow Neural Network by Reconstructed Multi-Channel Feature Maps



## Installation

install molmap by:

```bash
git clone https://github.com/shenwanxiang/bidd-rfmap.git
cd bidd-rfmap
pip install -r requirements.txt --user

# add molmap to PYTHONPATH
echo export PYTHONPATH="\$PYTHONPATH:`pwd`" >> ~/.bashrc

# init bashrc
source ~/.bashrc
```


## Usage


```python
from rfmap import RFMAP

#create rfmap object
mp = RFMAP(dfx, metric = 'correlation')

#fit rfmap
mp.fit(var_thr = 1e-4, cluster_channels = 3,split_channels = True)

#transform
X = mp.transform(dfx.values)

#save mp object

mp.save('./test.mp')
```
