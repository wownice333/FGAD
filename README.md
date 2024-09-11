# FGAD: Towards Effective Federated Graph Anomaly Detection via Self-boosted Knowledge Distillation

The code of paper Towards Effective Federated Graph Anomaly Detection via Self-boosted Knowledge Distillation.

### Dependencies

- python 3.8, pytorch, torch-geometric, torch-sparse, numpy, scikit-learn, pandas, dtaidistance

If you have installed above mentioned packages you can skip this step. Otherwise run:

    pip install -r requirements.txt

## Reproduce graph data results in the single-dataset setting

To generate results

    python FGAD_oneDS.py --data_group IMDB-BINARY --eval True

To train FGAD without loading saved weight files

    python FGAD_oneDS.py --data_group IMDB-BINARY --eval False

## Reproduce graph data results in the multi-dataset setting

To generate results

    python FGAD_multiDS.py --data_group molecules

The optional datasets in this code include mix, biochem, molecules and small.

If you've found FGAD useful for your research, please cite our paper as follows:

```
@inproceedings{cai2024towards,
  title={Towards Effective Federated Graph Anomaly Detection via Self-boosted Knowledge Distillation},
  author={Cai, Jinyu and Zhang, Yunhe and Lu, Zhoumin and Guo, Wenzhong and Ng, See-Kiong},
  booktitle={ACM Multimedia 2024},
  year={2024}
}
```



