# FOP-DMAC-MACPF

## Note

The implementation of the following methods can be found in this codebase:
- [**FOP**: Factorizing Optimal Joint Policy of Maximum-Entropy Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v139/zhang21m/zhang21m.pdf)
- [**DMAC**: Divergence-Regularized Multi-Agent Actor-Critic](https://arxiv.org/abs/2110.00304)
- [**MACPF**: More Centralized Training, Still Decentralized Execution: Multi-Agent Conditional Policy Factorization](https://arxiv.org/abs/2209.12681) 

## Installation

- 1. install SMAC following https://github.com/oxwhirl/smac
- 2. install required packages: pip install -r requirements.txt 

## How to run

```
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 src/main.py --config=fop/dmac/dfop --env-config=sc2 with env_args.map_name=2c_vs_64zg seed=1
```
Environment variable **CUBLAS_WORKSPACE_CONFIG** is recommended to enforce deterministic behavior of RNN.

<!---
## Results

Here, we provide results in three different SMAC scenarios using default hyperparameters.
![3s_vs_3z](./img/3s_vs_3z.png)![2c_vs_64zg](./img/2c_vs_64zg.png)![MMM2](./img/MMM2.png)
-->

## Citation

### Citation

If you are using the codes, you are welcomed cite our paper.

[Tianhao Zhang, Yueheng Li, Chen Wang, Guangming Xie and Zongqing Lu. *FOP: Factorizing Optimal Joint Policy of Maximum-Entropy Multi-Agent Reinforcement Learning*. ICML'21.](https://proceedings.mlr.press/v139/zhang21m.html)

    @inproceedings{zhang2021fop,
            title={Fop: Factorizing optimal joint policy of maximum-entropy multi-agent reinforcement learning},
            author={Zhang, Tianhao and Li, Yueheng and Wang, Chen and Xie, Guangming and Lu, Zongqing},
            booktitle={International Conference on Machine Learning},
            pages={12491--12500},
            year={2021},
            organization={PMLR}
    }

[Kefan Su and Zongqing Lu. *Divergence-Regularized Multi-Agent Actor-Critic*. ICML'22.](https://proceedings.mlr.press/v162/su22b.html)

    @inproceedings{su2022divergence,
            title={Divergence-regularized multi-agent actor-critic},
            author={Su, Kefan and Lu, Zongqing},
            booktitle={International Conference on Machine Learning},
            pages={20580--20603},
            year={2022},
            organization={PMLR}
    }

[Jiangxing Wang, Deheng Ye and Zongqing Lu. *More Centralized Training, Still Decentralized Execution: Multi-Agent Conditional Policy Factorization*.](https://arxiv.org/abs/2209.12681)

    @article{wang2022more,
        title={More Centralized Training, Still Decentralized Execution: Multi-Agent Conditional Policy Factorization},
        author={Wang, Jiangxing and Ye, Deheng and Lu, Zongqing},
        journal={arXiv preprint arXiv:2209.12681},
        year={2022}
    }
