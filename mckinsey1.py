# Import libraries necessary for this project
#loading the libraries required
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor

import sys
from sklearn.cross_validation import cross_val_score, cross_val_predict

from sklearn.metrics import r2_score


#loading data
train = pd.read_csv('train_aWnotuB.csv', parse_dates=[0])
test = pd.read_csv('test_BdBKkAj.csv', parse_dates=[0])
submit = pd.read_csv("sample_submission_EZmX9uE.csv")


# Saving id variables to create final submission
ids_test = test['ID'].copy()


#converting to datetime objects
train["DateTime"] = pd.to_datetime(train["DateTime"] )
test["DateTime"] = pd.to_datetime(test["DateTime"] )



column_train = train.iloc[:,0]
column_test = test.iloc[:,0]


new = pd.DataFrame({"year": column_train.dt.year,
              "month": column_train.dt.month,
              "day": column_train.dt.day,
              "hour": column_train.dt.hour,
              "dayofyear": column_train.dt.dayofyear,
              "week": column_train.dt.week,
              "dayofweek": column_train.dt.dayofweek,
              "quarter": column_train.dt.quarter,
             })
			 

			
			
new1 = pd.DataFrame({"year": column_test.dt.year,
              "month": column_test.dt.month,
              "day": column_test.dt.day,
              "hour": column_test.dt.hour,
              "dayofyear": column_test.dt.dayofyear,
              "week": column_test.dt.week,
              "dayofweek": column_test.dt.dayofweek,
              "quarter": column_test.dt.quarter,
             })
			 

			 
merged_test = pd.merge(left=test, left_index=True,
                  right=new1, right_index=True,
                  how='inner')
merged_test.drop(['DateTime','ID'], axis=1, inplace=True)


merged_train = pd.merge(left=train, left_index=True,
                  right=new, right_index=True,
                  how='inner')
merged_train.drop(['DateTime','ID'], axis=1, inplace=True)
			 
			


y_train = merged_train['Vehicles'].values

merged_train.drop(['Vehicles'], axis=1, inplace=True)
X_train = merged_train.values



#performance metric
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true,y_predict)
    
    # Return the score
    return score

#model function

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1,11)}

    #Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])


preds_dt = reg.predict(merged_test)

# Submission file
submit = pd.DataFrame({'ID': ids_test, 'Vehicles': preds_dt})
submit = submit[['ID', 'Vehicles']]

submit.ix[submit['Vehicles'] < 0, 'Vehicles'] = y_train.min()  # changing min prediction to min value in train
submit.to_csv("dt1.csv", index=False)	

#Model giving 6.92
