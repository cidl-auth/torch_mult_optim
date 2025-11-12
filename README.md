# Multiplicative update rules for accelerating deep learning training and increasing robustness

This repository implements the method presented in paper ["Multiplicative update rules for accelerating deep learning training and increasing robustness"](https://www.sciencedirect.com/science/article/abs/pii/S0925231224001231)

| Base Optimizer | Multiplicative                                   | Hybrid                                             | Updates      |
|----------------|--------------------------------------------------|----------------------------------------------------|--------------|
| SGD            | [MSGD](./torch_mult_optim/mult/m_sgd.py)         | [MSGD](./torch_mult_optim/hybrid/h_sgd.py)         | M_ABS, H_ABS |
| Adam           | [MAdam](./torch_mult_optim/mult/m_adam.py)       | [HAdam](./torch_mult_optim/hybrid/h_adam.py)       | M_ABS, H_ABS |
| Adagrad        | [MAdagrad](./torch_mult_optim/mult/m_adagrad.py) | [HAdagrad](./torch_mult_optim/hybrid/h_adagrad.py) | M_ABS, H_ABS |
| RMSprop        | [MRMSprop](./torch_mult_optim/mult/m_rmsprop.py) | [HRMSprop](./torch_mult_optim/hybrid/h_rmsprop.py) | M_ABS, H_ABS |


## Citation 
```
@article{kirtas2024multiplicative,
  title={Multiplicative update rules for accelerating deep learning training and increasing robustness},
  author={Kirtas, Manos and Passalis, Nikolaos and Tefas, Anastasios},
  journal={Neurocomputing},
  volume={576},
  pages={127352},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgments 
This work has received funding from the research project ”Energy Efficient and Trustworthy Deep Learning - DeepLET” is implemented in the framework of H.F.R.I call “Basic research Financing (Horizontal support of all Sciences)” under the National Recovery and Resilience Plan “Greece 2.0” funded by the European Union – NextGenerationEU (H.F.R.I. Project Number: 016762). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.


![image](https://deeplet.csd.auth.gr/wp-content/uploads/2024/05/cropped-logo-final-small-2.png)
![image](https://deeplet.csd.auth.gr/wp-content/uploads/2025/03/Logo-DeepLET-EN-300x53.png)
![image](https://deeplet.csd.auth.gr/wp-content/uploads/2025/03/HFRI_LOGO_SMALL4.jpg)

