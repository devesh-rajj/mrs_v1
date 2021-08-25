import pandas as pd
from surprise import accuracy
from surprise.model_selection.validation import cross_validate
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import Dataset
from surprise.model_selection import train_test_split


data = Dataset.load_builtin('ml-100k');
trainset, testset = train_test_split(data, test_size=0.2, random_state=10)

from collections import defaultdict

def get_pred(predictions, n=10):
    top_pred = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_pred[uid].append((iid, est))      #Mapping list of (movieid, predicted rating) to each userid

    for uid, user_ratings in top_pred.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_pred[uid] = user_ratings[:n]    #Sorting and displaying the top 'n' predictions
    print(top_pred)
    return top_pred

class collab_filtering_based_recommender_model():
    def __init__(self, model, trainset, testset, data):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.data = data
        self.pred_test = None
        self.recommendations = None
        self.top_pred = None
        self.recommenddf = None

    def fit_and_predict(self):        
        #print('-------Fitting the train data-------')
        self.model.fit(self.trainset)       

        #print('-------Predicting the test data-------')
        self.pred_test = self.model.test(self.testset)        
        rmse = accuracy.rmse(self.pred_test)
        #print('-------RMSE for the predicted result is ' + str(rmse) + '-------')   
        self.top_pred = get_pred(self.pred_test)
        self.recommenddf = pd.DataFrame(columns=['userId', 'MovieId', 'Rating'])
        for item in self.top_pred:
            subdf = pd.DataFrame(self.top_pred[item], columns=['MovieId', 'Rating'])
            subdf['userId'] = item
            cols = subdf.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            subdf = subdf[cols]        
            self.recommenddf = pd.concat([self.recommenddf, subdf], axis = 0)        
        

    def recommend(self, user_id, n=5):
        print('-------Recommending top ' + str(n)+ ' movies for userid : ' + user_id + '-------')
        df = self.recommenddf[self.recommenddf['userId'] == user_id].head(n)
        return df
        

from surprise.model_selection import RandomizedSearchCV

def find_best_model(model, parameters,data):
    clf = RandomizedSearchCV(model, parameters, n_jobs=-1, measures=['rmse'])
    clf.fit(data)             
    print(clf.best_score)
    print(clf.best_params)
    print(clf.best_estimator)
    return clf

sim_options = {
    "name": ["pearson_baseline"],
    "min_support": [3, 4, 5],
    "user_based": [True],
}
params = { 'k': range(30,50,1), 'sim_options': sim_options}
clf = find_best_model(KNNWithMeans, params,data)

knnwithmeans = clf.best_estimator['rmse']
col_fil_knnwithmeans = collab_filtering_based_recommender_model(knnwithmeans, trainset, testset,data)

col_fil_knnwithmeans.fit_and_predict()

result_knn_user1 = col_fil_knnwithmeans.recommend(user_id='31', n=10)
print(result_knn_user1)
result_knn_user2 = col_fil_knnwithmeans.recommend(user_id='2', n=10)
print(result_knn_user2)
