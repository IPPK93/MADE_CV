sports_classification_contest
==============================

Repo with author's solution for the [kaggle contest](https://www.kaggle.com/competitions/vk-made-sports-image-classification/overview) for MADE Deep CV course.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    |
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │                         predictions
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

To run this you will need to be in current directory. Otherwise we won't guarantee that everything will work as expected.
You will need unarchived data from contest in the `data/raw/` folder.

1. Create env (we used conda env, seems fine):
```
make create_environment
```

2. Activate it:
```
conda activate MADE_CV
```

3. Generate id<->label dictionaries:
```
python -m src --mode generate
```

4. Run train:
```
python -m src --mode train
```

5. After training process is finished - you may predict
```
python -m src --mode test --model-path models/efficientnet_v2_m_clf-87-65-model.pth
```

Models and submission will be saved in `models/` folder. During training losses are logged via Tensorboard, log directory is `reports/runs/`.

We definitely need to offer more customisation (data paths/hyperparameters/etc.) but currently it is like that.

--------

### IMPORTANT NOTE: Unfortunately, we did not have enough resources to run this program on our notebook. That's why what you see is actually taken from the notebook (in the `notebooks/` folder), which was run on Google Colab. We did run a lightened version of it on local machine, though. Everything should work fine. Contact us if something goes wrong.

--------


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
