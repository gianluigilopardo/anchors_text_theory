
# A Sea of Words: An In-Depth Analysis of Anchors for Text Data

Official code for the paper ["A Sea of Words: An In-Depth Analysis of Anchors for Text Data"](https://proceedings.mlr.press/v206/lopardo23a.html), AISTATS 2023.

## Dependencies

The required Python packages are listed in  `requirements.txt`.

Install the dependencies using the following commands:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage
- ```exhaustive_anchors``` contains 
    -- ```empirical_anchor_text.py```: our exhaustive-empirical implementation of Anchors for text data
    -- ```similarity.py```: script to compute the Jaccard similarity between exhaustive-empirical Anchors and official implementation
- in ```monte-carlo```, the script ```precision_approximation.py``` validates *Approximating the precision of a linear classifier*, ```norm_tf_idf.py``` validates the statement for the normalized TF-IDF vectorization.  
- ```analysis``` contains experiments for linear models (```linear_model```),  if-then rules (```simple_rules```), and neural networks (```nn_gradient```).  Their results are available in ```results```.
- run ```generate_figures.py``` to visualize some experiments of *Analysis on explainable classifiers*: figures will be saved in ```results```.

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@InProceedings{lopardo23anchors,
  title={A Sea of Words: An In-Depth Analysis of Anchors for Text Data},
  author={Lopardo, Gianluigi and Precioso, Frederic and Garreau, Damien},
  booktitle={Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = {4848--4879},
  year = {2023},
  volume = {206},
  series = {Proceedings of Machine Learning Research},
  pdf = {https://proceedings.mlr.press/v206/lopardo23a/lopardo23a.pdf},
  url = {https://proceedings.mlr.press/v206/lopardo23a.html},
}
```
