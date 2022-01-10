# Dictionary Learning with Uniform Sparse Representations for Anomaly Detection

Implementation of the Uniform DL Representation for AD algorithm described in
P. Irofti and C. Rusu and A. Pătrașcu, "Dictionary Learning with Uniform Sparse Representations for Anomaly Detection".

If you use our [work](https://cs.unibuc.ro/~pirofti/papers/IroftiRusuPatrascu21_AD-USR-DL.pdf) in your research, please cite as:
```
@article{IRP21,
  title={Dictionary Learning with Uniform Sparse Representations for Anomaly Detection}, 
  author = {Irofti, P. and Rusu, C. and Pătrașcu, A.},
  year={2021},
}
```

The algorithm is implemented in [ksvd_supp.py](ksvd_supp.py). Have a look at the experiments for full examples:
* [Real-data on ODDS databases](test_odds.py) 
* [Synthetic generated data](test_synthetic.py)

## Requirements

* the [dictlearn package](https://gitlab.com/unibuc/graphomaly/dictionary-learning)
```
pip install dictlearn
```

* the [ODDS database](http://odds.cs.stonybrook.edu/) for the real-data experiments
