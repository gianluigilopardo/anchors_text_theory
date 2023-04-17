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
- in ```monte-carlo```, the script ```precision_approximation.py``` validates *Approximating the precision of a linear classifier*, ```norm_tf_idf.py``` validates the statement for the normalized TF-IDF vectorization.  
- ```analysis``` contains experiments for linear models (```linear_model```),  if-then rules (```simple_rules```), and neural networks (```nn_gradient```).  Their results are available in ```results```.
- run ```generate_figures.py``` to visualize some experiments of *Analysis on explainable classifiers*: figures will be saved in ```results```.

## Citing this work
If you use this code please cite
```
@InProceedings{pmlr-v206-lopardo23a,
  title = 	 {A Sea of Words: An In-Depth Analysis of Anchors for Text Data},
  author =       {Lopardo, Gianluigi and Precioso, Frederic and Garreau, Damien},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {4848--4879},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/lopardo23a/lopardo23a.pdf},
  url = 	 {https://proceedings.mlr.press/v206/lopardo23a.html},
  abstract = 	 {Anchors (Ribeiro et al., 2018) is a post-hoc, rule-based interpretability method. For text data, it proposes to explain a decision by highlighting a small set of words (an anchor) such that the model to explain has similar outputs when they are present in a document. In this paper, we present the first theoretical analysis of Anchors, considering that the search for the best anchor is exhaustive. After formalizing the algorithm for text classification, we present explicit results on different classes of models when the vectorization step is TF-IDF, and words are replaced by a fixed out-of-dictionary token when removed. Our inquiry covers models such as elementary if-then rules and linear classifiers. We then leverage this analysis to gain insights on the behavior of Anchors for any differentiable classifiers. For neural networks, we empirically show that the words corresponding to the highest partial derivatives of the model with respect to the input, reweighted by the inverse document frequencies, are selected by Anchors.}
}
```
