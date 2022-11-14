### Implications of the model

The problem we are facing has some challenges: the data is highly imbalanced (around 92% of the observations are negative); we do not know the associated costs and benefits of a client defaulting or paying back, respectively; and the outcome itself is such a complex thing to predict (not a natural phenomenon were we may expect a parametric relationship to exist), that we may need to leverage data we don't have or extensively transform what we have to create features that may explain a client defaulting.

Because of these challenges, we had to conduct intensive feature engineering to reduce the bias in our predictions: create something that resembled the actual behavior more similarly. Because we did not know the costs and benefits we did not have a financial metric to optimize directly, and instead had to focus on one that balanced somewhat well the trade-offs of flagging more defaulting clients and mistakenly flagging non-defaulting clients. We did, however, keep an eye on the overall accuracy of the model. Given the imbalanced nature of the dataset, just by always predicting that a client would not default we would be right around 92% of the time. Therefore we wanted to create a model that would at least have that level of accuracy, but at the same time was able to capture the patterns to flag defaulting clients well.

When optimizing for the chosen metric, AUC, we found the following results for several models:


```python
import pandas as pd
metrics = pd.read_csv('performance_metrics.csv', index_col = 0)
metrics.sort_values(by = 'roc_auc', ascending= False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roc_auc</th>
      <th>accuracy</th>
      <th>runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LGBM</th>
      <td>0.790970</td>
      <td>0.920126</td>
      <td>1788.541508</td>
    </tr>
    <tr>
      <th>random_forest</th>
      <td>0.758290</td>
      <td>0.919271</td>
      <td>1005.142762</td>
    </tr>
    <tr>
      <th>SVC</th>
      <td>0.752688</td>
      <td>0.915304</td>
      <td>12.469002</td>
    </tr>
    <tr>
      <th>logistic_regression_full</th>
      <td>0.696794</td>
      <td>0.909928</td>
      <td>30.610333</td>
    </tr>
    <tr>
      <th>decision_tree</th>
      <td>0.617587</td>
      <td>0.897792</td>
      <td>78.623214</td>
    </tr>
    <tr>
      <th>logistic_regression_l1</th>
      <td>0.481147</td>
      <td>0.919271</td>
      <td>33.661335</td>
    </tr>
  </tbody>
</table>
</div>



Ignoring runtime for a second, it can be clearly seen that Lightweight Gradient Boosting Model (LGBM) results are the clear winners: The model captures effectively those clients who will default while not rejecting those who would not default if given the loan better than any other model.

This would be a massive improvement over the default of assuming every client will pay back: it will be as accurate as that one, but better in that this one will actually avoid a lot of the costs associated with a client defaulting. The other default assumption would be to assume that no client would ever pay back and not hand out any loans, which of course would result in the bank closing. Deploying this model would strike a great balance which would allow the bank to collect interest and principal on the loans they give while reducing the risk of giving loans that will not be paid back.

The model does have some limitations and setbacks that should be addressed when implementing on a larger scale.
- Ideally, some more feature engineering should be conducted to further feed information to the model as to how defaulting really works.
- Because of the large nature of the data, this model is likely not perfectly tuned: when getting access to the adequate computing power, the model can and should be modified to improve performance.
- Speaking about computing power, given that we did not have access to GPU resources, we did not try to implement more powerful models, such as XGBoost. That model would very likely perform better than LGBM, and should be tried if possible. Any improvement, even if marginal, would implicate massive savings in costs for the bank when scaled to thousands of transactions.
