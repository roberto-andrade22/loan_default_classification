# Predicting customer defaulting on a loan:<br><br>Home Credit Group classification algorithms analysis

Roberto Andrade Martínez

### Summary
<br>

Given Home Credit's default risk [data](https://www.kaggle.com/competitions/home-credit-default-risk/data), this project aims to build classification algorithms that will predict whether or not a client will default on a loan. Secondly, the project also aims to begin building simpler models such as logistic regression classifiers, working up to XGboost to try and compare not only their accuracy but also their efficiency regarding computing power required to implement, as often the more complicated models may not be a feasible option to implement for some companies given the amount of resources they require to operate.

As the objective of the project is only to predict, at no point will we touch on output interpretability nor try to point out the most prominent features that drive defaulting on a loan.

In order to accomplish this we will use Python and in particular [Scikit learn 1.1.3](https://scikit-learn.org/stable/index.html) for both data processing as well as model implementation. For Gradient Boosted Trees, we will use the [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) package. 

### Content
- [Data processing](https://github.com/roberto-andrade22/loan_default_classification/blob/main/data_preparation.ipynb)

- [Trying different models](https://github.com/roberto-andrade22/loan_default_classification/blob/main/ML_predictions.ipynb)

- [XGBoost & its performance metrics]()

- [Conclusions]()