<br />
<div align="center">
    <h1 align="center">DepSign@RANLP2023</h1>
    <img src="https://images.emojiterra.com/twitter/v14.0/512px/1f3c6.png" alt="https://emojiterra.com/trophy" width="200">
  
  <br />

  <br />
  
  [Deep Learning Brasil group](https://www.linkedin.com/company/inteligencia-artificial-deep-learning-brasil) submission at [ABSAPT 2022](https://sites.google.com/inf.ufpel.edu.br/absapt2022/).
</div>


Submission files are available on [DeepLearningBrasil_task1.csv](DeepLearningBrasil_task1.csv) and [DeepLearningBrasil_task2.csv](DeepLearningBrasil_task2.csv). Here is the [presentation](presentation.pdf).
## Installation

```bash
pip install -r requirements.txt
```
All experiments were made on V100 GPU (32GB).


## How-to

1. Train ensemble

```bash
bash SOE/train_ensemble.sh
```

2. Predict ensemble

```bash
bash SOE/predict_ensemble.sh
```

##  Citation
```bibtex
@inproceedings{gomes2022deep,
  title={Deep learning brasil at absapt 2022: Portuguese transformer ensemble approaches},
  author={Gomes, JRS and Rodrigues, RC and Garcia, EAS and Junior, AFB and Silva, Diogo Fernandes Costa and Maia, Dyonnatan Ferreira},
  booktitle={Proceedings of the Iberian Languages Evaluation F{\'o}rum (IberLEF 2022), co-located with the 38th Conference of the Spanish Society for Natural Language Processing (SEPLN 2022), Online. CEUR. org},
  year={2022}
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/). See `LICENSE.txt` for more information.

## Acknowledgments

This work has been supported by the [AI Center of Excellence (Centro de Excelência em Inteligência Artificial – CEIA)](https://www.linkedin.com/company/inteligencia-artificial-deep-learning-brasil) of the Institute of Informatics at the Federal University of Goiás (INF-UFG).
