

# About

[MTS ML CUP](https://ods.ai/competitions/mtsmlcup)


Final solution: all features (aggregates (mean, median, min, max, std) by user_id (prices, counts, categorical features) + mean encoded (w2v + node2vec) url vectors) + 10 cv DANets averaging

Hardware:

CPU: Intel Xeon E5-2650V2

RAM: 64 Gb DDR3

GPU: RTX 3060 (12 Gb)

# Installation

```
pip install -r requirements.txt
```

# If you want to try scrapping / parsing features
```
pip install transformers sentence-transformers
pip install selenium
```
\+ you need to install Chrome Driver for selenium

# If you want to try graph features
```
pip install "tensorflow<2.11"
```
1) Download stellargraph package (https://github.com/stellargraph/stellargraph)
2) Move into stellargraph dir
3) Comment #python_requires=">=3.6.0, <3.9.0"
4) pip install .
