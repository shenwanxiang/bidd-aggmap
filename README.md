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
mp = RFMap()

mp.fit()

X = mp.transform(x)

```