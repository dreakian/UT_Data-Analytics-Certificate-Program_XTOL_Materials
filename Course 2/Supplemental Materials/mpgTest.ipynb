{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#imports\n",
    "#numpy,pandas,scipy, math, matplotlib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy\n",
    "from math import sqrt\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "#estimators\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.svm import SVR\n",
    "from sklearn import linear_model\n",
    "\n",
    "#model metrics\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "#cross validation\n",
    "from sklearn.cross_validation import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Model Choice\n",
    "modelSVR = SVR()\n",
    "modelRF = RandomForestRegressor()\n",
    "modelLR = LinearRegression()\n",
    "#model = linear_model.LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>mpg</th>\n",
       "      <th>cylinders</th>\n",
       "      <th>displacement</th>\n",
       "      <th>horsepower</th>\n",
       "      <th>weight</th>\n",
       "      <th>acceleration</th>\n",
       "      <th>model year</th>\n",
       "      <th>origin</th>\n",
       "      <th>car name</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>18.0</td>\n",
       "      <td>8</td>\n",
       "      <td>307.0</td>\n",
       "      <td>130</td>\n",
       "      <td>3504</td>\n",
       "      <td>12.0</td>\n",
       "      <td>70</td>\n",
       "      <td>1</td>\n",
       "      <td>chevrolet chevelle malibu</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15.0</td>\n",
       "      <td>8</td>\n",
       "      <td>350.0</td>\n",
       "      <td>165</td>\n",
       "      <td>3693</td>\n",
       "      <td>11.5</td>\n",
       "      <td>70</td>\n",
       "      <td>1</td>\n",
       "      <td>buick skylark 320</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>18.0</td>\n",
       "      <td>8</td>\n",
       "      <td>318.0</td>\n",
       "      <td>150</td>\n",
       "      <td>3436</td>\n",
       "      <td>11.0</td>\n",
       "      <td>70</td>\n",
       "      <td>1</td>\n",
       "      <td>plymouth satellite</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>16.0</td>\n",
       "      <td>8</td>\n",
       "      <td>304.0</td>\n",
       "      <td>150</td>\n",
       "      <td>3433</td>\n",
       "      <td>12.0</td>\n",
       "      <td>70</td>\n",
       "      <td>1</td>\n",
       "      <td>amc rebel sst</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17.0</td>\n",
       "      <td>8</td>\n",
       "      <td>302.0</td>\n",
       "      <td>140</td>\n",
       "      <td>3449</td>\n",
       "      <td>10.5</td>\n",
       "      <td>70</td>\n",
       "      <td>1</td>\n",
       "      <td>ford torino</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    mpg  cylinders  displacement  horsepower  weight  acceleration  \\\n",
       "0  18.0          8         307.0         130    3504          12.0   \n",
       "1  15.0          8         350.0         165    3693          11.5   \n",
       "2  18.0          8         318.0         150    3436          11.0   \n",
       "3  16.0          8         304.0         150    3433          12.0   \n",
       "4  17.0          8         302.0         140    3449          10.5   \n",
       "\n",
       "   model year  origin                   car name  \n",
       "0          70       1  chevrolet chevelle malibu  \n",
       "1          70       1          buick skylark 320  \n",
       "2          70       1         plymouth satellite  \n",
       "3          70       1              amc rebel sst  \n",
       "4          70       1                ford torino  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#data\n",
    "rawData=pd.read_csv('auto-mpg.csv')\n",
    "rawData.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 400 entries, 0 to 399\n",
      "Data columns (total 9 columns):\n",
      "mpg             400 non-null float64\n",
      "cylinders       400 non-null int64\n",
      "displacement    400 non-null float64\n",
      "horsepower      400 non-null int64\n",
      "weight          400 non-null int64\n",
      "acceleration    400 non-null float64\n",
      "model year      400 non-null int64\n",
      "origin          400 non-null int64\n",
      "car name        400 non-null object\n",
      "dtypes: float64(3), int64(5), object(1)\n",
      "memory usage: 28.2+ KB\n"
     ]
    }
   ],
   "source": [
    "rawData.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Summary of feature sample\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>cylinders</th>\n",
       "      <th>displacement</th>\n",
       "      <th>horsepower</th>\n",
       "      <th>weight</th>\n",
       "      <th>acceleration</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8</td>\n",
       "      <td>307.0</td>\n",
       "      <td>130</td>\n",
       "      <td>3504</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>350.0</td>\n",
       "      <td>165</td>\n",
       "      <td>3693</td>\n",
       "      <td>11.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>318.0</td>\n",
       "      <td>150</td>\n",
       "      <td>3436</td>\n",
       "      <td>11.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8</td>\n",
       "      <td>304.0</td>\n",
       "      <td>150</td>\n",
       "      <td>3433</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>302.0</td>\n",
       "      <td>140</td>\n",
       "      <td>3449</td>\n",
       "      <td>10.5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   cylinders  displacement  horsepower  weight  acceleration\n",
       "0          8         307.0         130    3504          12.0\n",
       "1          8         350.0         165    3693          11.5\n",
       "2          8         318.0         150    3436          11.0\n",
       "3          8         304.0         150    3433          12.0\n",
       "4          8         302.0         140    3449          10.5"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#features\n",
    "features = rawData.iloc[:,1:6]\n",
    "print('Summary of feature sample')\n",
    "features.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#dependent variable\n",
    "depVar = rawData['mpg']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>cylinders</th>\n",
       "      <th>displacement</th>\n",
       "      <th>horsepower</th>\n",
       "      <th>weight</th>\n",
       "      <th>acceleration</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8</td>\n",
       "      <td>307.0</td>\n",
       "      <td>130</td>\n",
       "      <td>3504</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>350.0</td>\n",
       "      <td>165</td>\n",
       "      <td>3693</td>\n",
       "      <td>11.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>318.0</td>\n",
       "      <td>150</td>\n",
       "      <td>3436</td>\n",
       "      <td>11.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8</td>\n",
       "      <td>304.0</td>\n",
       "      <td>150</td>\n",
       "      <td>3433</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>302.0</td>\n",
       "      <td>140</td>\n",
       "      <td>3449</td>\n",
       "      <td>10.5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   cylinders  displacement  horsepower  weight  acceleration\n",
       "0          8         307.0         130    3504          12.0\n",
       "1          8         350.0         165    3693          11.5\n",
       "2          8         318.0         150    3436          11.0\n",
       "3          8         304.0         150    3433          12.0\n",
       "4          8         302.0         140    3449          10.5"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Training Set (Feature Space: X Training)\n",
    "X_train = (features[:-100])\n",
    "#feature_train_count = len(features_train.index)\n",
    "#print('The number of observations in the X training set is:',str(feature_train_count))\n",
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The number of observations in the Y training set are: 300\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0    18.0\n",
       "1    15.0\n",
       "2    18.0\n",
       "3    16.0\n",
       "4    17.0\n",
       "Name: mpg, dtype: float64"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Dependent Variable Training Set (y Training)\n",
    "y_train = depVar[:-100]\n",
    "#depVar_test=depVar[-100:]\n",
    "y_train_count = len(y_train.index)\n",
    "print('The number of observations in the Y training set are:',str(y_train_count))\n",
    "y_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The number of observations in the feature testing set is: 100\n",
      "     cylinders  displacement  horsepower  weight  acceleration\n",
      "300          4         105.0          70    2150          14.9\n",
      "301          4          85.0          65    2020          19.2\n",
      "302          4          91.0          69    2130          14.7\n",
      "303          4         151.0          90    2670          16.0\n",
      "304          6         173.0         115    2595          11.3\n"
     ]
    }
   ],
   "source": [
    "#Testing Set (X Testing)\n",
    "X_test = features[-100:]\n",
    "X_test_count = len(X_test.index)\n",
    "print('The number of observations in the feature testing set is:',str(X_test_count))\n",
    "print(X_test.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#Testing Set (Y Testing;ground truth)\n",
    "y_test = depVar[-100:]\n",
    "y_test_count = len(y_test.index)\n",
    "print('The number of observations in the Ground Truth are:',str(y_test_count))\n",
    "y_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((225, 5), (75, 5))"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)\n",
    "X_train.shape, X_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b> Random Forest Regression Model Fitting</b>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.80335568  0.78881571  0.83139016]\n"
     ]
    }
   ],
   "source": [
    "#Model Fitting\n",
    "modelRF.fit(X_train,y_train)\n",
    "print(cross_val_score(modelRF, X_train, y=y_train))\n",
    "#modelRF.score(X_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Support Vector Regression Model Fitting and Scoring</b>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-0.00058268 -0.15567904  0.01017114]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.22197051516765165"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "modelSVR.fit(X_train,y_train)\n",
    "print(cross_val_score(modelSVR, X_train, y_train)) \n",
    "modelSVR.score(X_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Linear Regression Model Fitting and Scoring</b>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.76546419  0.77432528  0.78207003]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.78704730373399734"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "modelLR.fit(X_train,y_train)\n",
    "print(cross_val_score(modelLR, X_train, y_train)) \n",
    "modelLR.score(X_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Trained Model Performance</b>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Model Performance\n",
    "rSquared = modelRF.score(X_train,y_train)\n",
    "mse = np.mean((modelRF.predict(X_test) - y_test) ** 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean squared error: 5.46: (Lower numbers are better)\n",
      "R Squared of training: 0.96: (Higher numbers are better, but be careful of overfitting)\n"
     ]
    }
   ],
   "source": [
    "print('Mean squared error: %.2f' % mse + ': (Lower numbers are better)')\n",
    "print('R Squared of training: %.2f' % rSquared + ': (Higher numbers are better, but be careful of overfitting)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Make Predictions with Trained Model</b>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R Squared: 0.881\n",
      "RMSE: 2.336\n"
     ]
    }
   ],
   "source": [
    "#Make Predictions\n",
    "predictions = modelRF.predict(X_test)\n",
    "predRsquared = r2_score(y_test,predictions)\n",
    "rmse = sqrt(mean_squared_error(y_test, predictions))\n",
    "print('R Squared: %.3f' % predRsquared)\n",
    "print('RMSE: %.3f' % rmse)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Post Resampling (Ground Truth vs Predictions (higher numbers are better)</b>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8a6a9332e8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plt.scatter(features_test, depVar_test,  color='black')\n",
    "plt.scatter(y_test, predictions, color=['blue','green'], alpha = 0.5)\n",
    "plt.xlabel('Ground Truth')\n",
    "plt.ylabel('Predictions')\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
