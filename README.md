# Reddit-flair-detection
This is a reddit flair detection repository developed using flask and python, it's live at https://flair-it-up.herokuapp.com/. 
### Structure

The directory contains web sub directories and a sub directory for hosting model and other scripts:

1. [app.py](https://github.com/abhishek-parashar/Reddit-flair-detection/blob/master/app.py)The file which contains all the main backend operations of the website and used to run the flask server locally.
   
2. [Procfile](https://github.com/abhishek-parashar/Reddit-flair-detection/blob/master/Procfile) for setting up heroku.
    
3. [model](https://github.com/divyanshuaggarwal/Reddit-Flair-Detector/blob/master/Flair_Detection.ipynb) contains the saved model.

4. [requirement.txt](https://github.com/abhishek-parashar/Reddit-flair-detection/blob/master/requirements.txt) contains all the dependencies.

5. [templates](https://github.com/abhishek-parashar/Reddit-flair-detection/tree/master/templates) contains the html file.

6. [static](https://github.com/abhishek-parashar/Reddit-flair-detection/tree/master/static) contains the css file.

7. [nltk.txt](https://github.com/abhishek-parashar/Reddit-flair-detection/blob/master/nltk.txt) contains the nltk dependency.

8. [Scripts](https://github.com/abhishek-parashar/Reddit-flair-detection/tree/master/scripts) the directory contains scripts for data extraction, model, expolatory data analysis and experiment log manager notebooks.
  
### Codebase

The entire code has been developed using Python programming language and is hosted on Heroku. The analysis and model is developed using nltk library and various machine learning models, The website is developed using Flask. 

### How to run the project:

  1. Open the `Terminal`.
  2. Clone the repository by entering `https://github.com/abhishek-parashar/Reddit-flair-detection`.
  3. Ensure that `Python3` and `pip` are installed on the system.
  4. Create a `virtualenv` by executing the following command: `virtualenv venv`.
  5. Activate the `venv` virtual environment by executing the follwing command: `source venv/bin/activate`.
  6. Enter the cloned repository directory and execute `pip install -r requirements.txt`.
  7. Now, execute the following command: `flask run` and it will point to the `localhost` server with the port `5000`.
  8. Enter the `IP Address: http://localhost:5000` on a web browser and use the application.
  
### Dependencies

The following dependencies can be found in [requirements.txt](https://github.com/abhishek-parashar/Reddit-flair-detection/blob/master/requirements.txt):

  1. [praw](https://praw.readthedocs.io/en/latest/)
  2. [scikit-learn](https://scikit-learn.org/)
  3. [nltk](https://www.nltk.org/)
  4. [Flask](https://palletsprojects.com/p/flask/)
  5. [Gensim](https://radimrehurek.com/gensim/)
  6. [pandas](https://pandas.pydata.org/)
  7. [numpy](http://www.numpy.org/)
  8. [scikit-learn](https://scikit-learn.org/stable/index.html)
  9. [gunicorn](https://gunicorn.org/)
  10. [XGBoost](https://xgboost.readthedocs.io/en/latest/)
  
### Approach

I went through a lot of litrature and Youtube videos for the following task, the resources can be seen in the resorces section. After going through these resources and tutorials. I collected data from reddit using Praw module. I used nltk to remove bad words and applied various machine learning models to it. Only top 10 comments were taken, I used TFID DICT VECTORIZER to convert to word embeddings. Finaly deployed it using Flask and Heroku. 
  
### Results

#### Title as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Linear SVM                 | 0.7418032786885246|
| Logistic Regression        | **0.75409836**    |
| Random Forest              | 0.7336065573770492|
| MLP                        | 0.5327868852459017|
| XGBoost                    | 0.7008196721311475|

#### Body as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Linear SVM                 | 0.3442622950819672|
| Logistic Regression        | 0.3237704918032787|
| Random Forest              | **0.3770491803**  |
| MLP                        | 0.2663934426229508|
|XGBoost                     | 0.3688524590163934|

#### URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Linear SVM                 | 0.2745901639344262|
| Logistic Regression        | **0.3073770491**  |
| Random Forest              | 0.2622950819672131|
| MLP                        | 0.2254098360655737|
| XGBoost                    | 0.2172131147540983|

#### Comments as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Linear SVM                 | 0.430327868852459 |
| Logistic Regression        | 0.4344262295081967|
| Random Forest              | **0.438524590163**|
| MLP                        | 0.3073770491803279|
| XGBoost                    | 0.4180327868852459|

#### Title + Comments + URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Linear SVM                 | 0.7090163934426229|
| Logistic Regression        | 0.7131147540983607|
| Random Forest              | 0.7745901639344263|
| MLP                        | 0.5532786885245902|
| XGBoost                    | **0.8278688**     |

### Inferences 
 There various iterferences as discussed in the [EDA notebook](https://github.com/abhishek-parashar/Reddit-flair-detection/blob/master/scripts/Exploratory_Data_Analysis.ipynb) from the results we can infer that combined features give the best result probably because of the larger word embeddings present. We can also infer that the title as a feature also provides better results this can be attributed to the fact that title mainly consists of the required words or embeddings that is, it is in line with the flairs. 


### References

#### 1. For data collection:
1. https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
2. https://api.mongodb.com/python/current/tutorial.html

#### 2. For Building machine learning model:
1. https://medium.com/themlblog/splitting-csv-into-train-and-test-data-1407a063dd74
2. https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
3. https://medium.com/@robert.salgado/multiclass-text-classification-from-start-to-finish-f616a8642538
4. https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
5. https://www.districtdatalabs.com/text-analytics-with-yellowbrick
6. Applied AI course- https://www.appliedaicourse.com/

#### 3.For Building the Website and Deploying it:
1.	https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b
2.	https://towardsdatascience.com/deploying-a-deep-learning-model-on-heroku-using-flask-and-python-769431335f66
3.	https://medium.com/analytics-vidhya/deploy-machinelearning-model-with-flask-and-heroku-2721823bb653
4.	https://www.youtube.com/watch?v=UbCWoMf80PY
5.	https://www.youtube.com/watch?v=mrExsjcvF4o
6.	https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7
