# A Sea of Words: An In-Depth Analysis of Anchors for Text Data
Code for the paper [A Sea of Words: An In-Depth Analysis of Anchors for Text Data](https://arxiv.org/pdf/2205.13789.pdf).

## Requirements
Some non-standard packages need to be installed:
 - ```anchor``` (repo from the authors)
 ```
 pip install anchor-exp
 ```
 - ```spacy``` and a pretrained model:
 ```
 pip install spacy
 python -m spacy download en_core_web_sm
 ```

## Use

- ```exhaustive_anchors``` contains 
    -- ```empirical_anchor_text.py```: our exhaustive-empirical implementation of Anchors for text data
    -- ```similarity.py```: script to compute the Jaccard similarity between exhaustive-empirical Anchors and official implementation
- in ```monte-carlo```, the script ```linear_binomial_distribution.py``` validates *Approximating the precision of a linear classifier*
- run ```generate_figures.py``` to visualize some experiments of *Analysis on explainable classifiers*: figures will be saved in ```results```.
