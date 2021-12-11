# https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8
# https://www.kaggle.com/amark720/iris-species-rare-visualization-tools/notebook
# http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx/notebooks/ml_cc_machine_learning_problems2.html
# https://cambridgecoding.wordpress.com/2016/05/16/expanding-your-machine-learning-toolkit-randomized-search-computational-budgets-and-new-algorithms-2/
# https://stats.stackexchange.com/questions/329390/logistic-regression-vs-svm
# https://towardsdatascience.com/understanding-random-forest-58381e0602d2
# https://towardsdatascience.com/support-vector-machine-simply-explained-fee28eba5496
# from My_Projects.Iris_Species.ML_Iris_Species import Iris_Species
# https://medium.com/analytics-vidhya/accuracy-vs-f1-score-6258237beca2
# https://moonbooks.org/Articles/How-to-increase-the-size-of-axes-labels-on-a-seaborn-heatmap-in-python-/
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import streamlit as st

from PIL import Image


#def Front_end_Iris_Species():
# Disable Warning about graphic plots
st.set_option('deprecation.showPyplotGlobalUse', False)
# Title :
st.markdown(
    '<h2 style="background-color:white; text-align:center; font-family:arial;color:Black">Iris Species</h2>',
    unsafe_allow_html=True)

# Presentation of the Iris Species :
st.subheader('About : ')
st.write('In this project, we will do some exploratory data analysis on the famous '
         'Iris dataset. The Iris Dataset contains four features (length and width of sepals and petals)'
         ' of 50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor).'
         )
st.write('Just for reference, here are pictures of the three flowers species:')
image = Image.open('./Images/flowers2.png')
st.image(image=image, width=660)

# Load Datasets :
st.subheader('Load and Analyse Data : ')
st.write('The dataset seems to be like : ')

data = load_iris()
data = pd.DataFrame(data=np.c_[data['data'], data['target']], columns=data['feature_names'] + ['target'])
data.loc[data['target'] == 0, 'target_name'] = 'setosa'
data.loc[data['target'] == 1, 'target_name'] = 'versicolor'
data.loc[data['target'] == 2, 'target_name'] = 'virginica'

data1 = data.copy()
Y = data1['target']
data.drop(columns=['target'], inplace=True)
data1.drop(columns=['target', 'target_name'], inplace=True)
X = data1

st.dataframe(data=data)
st.write('We have 4 Variables : sepal length, sepal width,  petal length, petal width and one target. The target '
         'contains 3 classes. The three classes are the following classes : Setosa, Versicolor and Virginica. '
         ' So, we have a problem classifiaction to resolve.')
st.write('We verify if features contain missing values.')
features_null = pd.DataFrame(data=np.c_[data.isnull().sum().index, data.isnull().sum().values],
                             columns=["Features", 'Count of missing values'])
st.table(data=features_null)
st.write('All features are without missing values.')
st.write("We want to learn more about the data. "
         "We can calculate basic statistics on each of the data frame's columns : ")
st.dataframe(data.describe())
st.write('Numbers can tell a lot, but sometimes it is better to '
         'compare the feature of several individual observations on a set of numerical variables '
         'using a other plot "s tool named Parallel plot.')
image_FPC = Image.open('./Images/Features_Parallel_Coord.png')
st.image(image=image_FPC)
st.write('Each vertical bar represents a feature (column or variable) like petal length (cm) for example.'
         ' Values are then plotted as series of lines connected across each axis.'
         ' We notice Parallel plot allow to detect interesting patterns.'
         ' We see that even for Parallel plot, we can easily classify setosa '
         'among the 2 others species (Versicolor and  Virginica)'
         ' according to petal width (cm) and petal lenght (cm) features.')
st.subheader("Machine Learning's Models :")
st.write("After loading and analysing our dataset, we have a better understanding of it"
         " and we'll use 3 algorithms to solve this classification problem and so predict our dataset.")
st.write("But before we start, let's start by simply presenting these 3 algorithms: ")
Image_Algo2 = Image.open('./Images/3Algorithms.png')
st.image(image=Image_Algo2)
st.write(
    "Logistic regression is a statistical model that uses the logistic function to allow us to predict the classes of our target variable."
    " The Random Forest consists of a set of individual decision trees of which each individual tree in the Random Forest. "
    "spits out a class prediction and the class with the most votes becomes our model's prediction."
    " Support Vector Machine creates a line or a hyperplane with a margin which separates the data into classes.")

# Build Model

st.write("After choosing and defining our algorithms, we have to train and evaluate them. "
         "The training part is not covered in this project so let's go directly to the evaluation "
         "part by using a confusion matrix that is a table used to describe the performance of a classification model "
         "on a set of test data for which the true values are known."
         )
cl = st.selectbox("Choose your algorithm's matrix confusion",
                  ('Random Forest Classifier', 'Logistic Regression', 'Support Vector Machine'))
if cl == 'Random Forest Classifier':
    st.image('./Images/conf_matrix_Random_Classifier.png')
elif cl == 'Logistic Regression':
    st.image('./Images/conf_matrix_Logistic_Regression.png')
elif cl == 'Support Vector Machine':
    st.image('./Images/conf_matrix_Support_Vector_Machine.png')

st.write("On the one hand, our algorithm is able to predict without error the setosa species of the 2 other species"
         " (versicolor, virginia). On the other hand, to predict whether the species is virginica or Versicoloris "
         "is more difficult because we find  False Positives and False Negatives for each of the 3 algorithms.")
st.write(
    "Even if the confusion matrix gives us a good overview on the performance of the different algorithms,"
    " it is advisable to add 2 metrics : f1_score and accuracy."
    " Accuracy is used when the True Positives and True negatives are more important while F1-score is used when "
    "the False Negatives and False Positives are crucial.")

conf_matrix = {}
metrics = []
models = {'Random_Classifier': RandomForestClassifier(n_estimators=10),
          'Logistic_Regression': LogisticRegression(solver='saga', max_iter=5000),
          'Support_Vector_Machine': SVC()}
Strat_kf = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
for names, model in models.items():
    Y_pred = cross_val_predict(estimator=model, X=X, y=Y, cv=Strat_kf, n_jobs=-1)
    pickle.dump(model, open(f'C:/Users/Abdel/Desktop/Iris_Project/Modeles_ML/iris_model_{names}.pkl', 'wb'))
    conf_matrix.update({names: pd.DataFrame(confusion_matrix(y_true=Y, y_pred=Y_pred))})
    acc_scor = accuracy_score(y_true=Y, y_pred=Y_pred)
    f1_scor = f1_score(y_true=Y, y_pred=Y_pred, average='weighted')
    metrics.append([acc_scor, f1_scor])

metrics_score = pd.DataFrame(data=metrics, columns=['Accuracy', 'F1_score'],
                             index=['Random_Classifier', 'Logistic_Regression', 'Support Vector Machine'])
c1, c2, c3 = st.columns((0.5, 3, 0.5))
c2.dataframe(data=metrics_score)

st.write("In using this 2 metrics, we find that the precision and the f1_score are almost equal but less than 1. "
         "This explains because when we see the confusion matrix of each of the algorithms, we noticed "
         "that the True Positives and the True Negatives are almost equal and ditto for the False Negatives"
         " and the False Positives.")
st.subheader('Conclusion : ')
st.write("In the load and Analysis data part, we've conclued that setosa was easily differentiable from the other 2"
         " species but it was therefore more difficult to distinguish if the the species is either a versicolor or "
         "a virginica because they presented very similar features.")
st.write(
    "In the Machine Learning's Models part, we've got the same conclusion because algorithms are able to easily "
    "predict setosa from the two others sepecies but it was  more difficult to predict if "
    "the species is either a versicolor or a virginica.")
st.write('We can conclude that the more the targets have similar observations in features, '
         'the more difficult it will be for our algorithms to predict our data.')

st.subheader('Application of IRIS flower identification :')
st.write("Now, we can create a application of IRIS flower identification with our previous work. "
         "The Application is very simple. it contains a simple button submit, a random observation, "
         "a already trained model and the comparaison between the true label and the model result that give us the predicted label.")
st.write('To make a application, we need to select a random observation '
         'and use our best model to predict the Iris Species.')

if st.button('Select Randomly a observation'):
    obser_rnd = np.random.randint(0, data.shape[0])
    data_obs=data.loc[obser_rnd]
    data_obs=pd.DataFrame(data_obs).transpose()
    data_obs0 = data_obs.copy(deep=True)
    data_obs0.drop(columns=['target_name'], inplace=True)
    st.dataframe(data=data_obs0)
    st.write("The osbervation , which is randomly selected, 'll be injected in the trained model just below.")
    st.image('./Images/Pic_Trained_Model.png')
    c5, c6, c7, c8, c9, c10 = st.columns((0.4, 1, 0.4, 0.4, 1, 0.4))
    c1, c2 = st.columns((1, 1))
    if data.loc[obser_rnd, 'target_name'] == 'versicolor':
        c1.image('./Images/Versicolor.jpg')
        c6.write('True Label : versicolor')
    elif data.loc[obser_rnd, 'target_name'] == 'setosa':
        c1.image('./Images/Setosa.jpg')
        c6.write('True Label : setosa')
    elif data.loc[obser_rnd, 'target_name'] == 'virginica':
        c1.image('./Images/Virginica.jpg')
        c6.write('True Label : virginica')

    filename = './Modeles_ML/iris_model_Support_Vector_Machine.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    data_obs.drop(columns=['target_name'], inplace=True)
    model = loaded_model.fit(X=X, y=Y)
    Y_pred = model.predict(data_obs)
    if Y_pred == 1:
        c9.write('Predicted Label : versicolor')
        c2.image('./Images/Versicolor.jpg')
    elif Y_pred == 0:
        c9.write('Predicted Label : setosa')
        c2.image('./Images/Setosa.jpg')
    elif Y_pred == 2:
        c9.write('Predicted Label : virginica')
        c2.image('./Images/Virginica.jpg')


    
