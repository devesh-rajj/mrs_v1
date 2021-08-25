from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import render,get_object_or_404,redirect
from django.db.models import Q
from django.http import Http404
from .models import Movie,Myrating,Movielens
from django.contrib import messages
from .forms import UserForm
from django.db.models import Case, When
import numpy as np 
import pandas as pd
from collections import defaultdict
from surprise.model_selection import train_test_split
from surprise.model_selection import RandomizedSearchCV
from surprise import accuracy
from surprise.model_selection.validation import cross_validate
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import Reader
from surprise import dataset
#-------------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------------
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
        df = self.recommenddf[self.recommenddf['userId'] == user_id].head(n)
        return df
	


#-------------------------------------------------------------------------------------------------------------------------------------------------------

# for recommendation
def recommend(request):
	if not request.user.is_authenticated:
		return redirect("login")
	if not request.user.is_active:
		raise Http404
	def find_best_model(model, parameters,data):
		clf = RandomizedSearchCV(model, parameters, n_jobs=-1, measures=['rmse'])
		clf.fit(data)             
		print(clf.best_score)
		print(clf.best_params)
		print(clf.best_estimator)
		return clf
	df1=pd.DataFrame(list(Movielens.objects.all().values()))
	df2=pd.DataFrame(list(Myrating.objects.all().values()))
	df1.drop(['id'],axis=1)
	df2.drop(['id'],axis=1)
	frames=[df1,df2]
	data=pd.concat(frames)
	#data=pd.DataFrame(list(Myrating.objects.all().values()))
	#data.drop(['id'],axis=1)
	reader = Reader(line_format='user item rating', rating_scale=(1, 5))
	class MyDataset(dataset.DatasetAutoFolds):
		def __init__(self, df, reader):

			self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in
								zip(df['user_id'], df['movie_id'], df['rating'])]
			self.reader=reader
	data = MyDataset(data, reader)
	trainset, testset = train_test_split(data, test_size=0.2)
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
	current_user_id= int(request.user.id)
	result_knn_user1 = col_fil_knnwithmeans.recommend(user_id= current_user_id, n=10)
	print(result_knn_user1)
	movie_list=list(result_knn_user1['MovieId'])
	print(movie_list)
	result=[]
	for i in movie_list:
		r=Movie.objects.get(pk=i)
		result.append(r)
	return render(request,'web/recommend.html',{'movie_list':result})


# List view
def index(request):
	movies = Movie.objects.all()
	query  = request.GET.get('q')
	if query:
		movies = Movie.objects.filter(Q(title__icontains=query)).distinct()
		return render(request,'web/list.html',{'movies':movies})
	return render(request,'web/list.html',{'movies':movies})


# detail view
def detail(request,movie_id):
	if not request.user.is_authenticated:
		return redirect("login")
	if not request.user.is_active:
		raise Http404
	movies = get_object_or_404(Movie,id=movie_id)
	#for rating
	if request.method == "POST":
		rate = request.POST['rating']
		ratingObject = Myrating()
		ratingObject.user   = request.user
		ratingObject.movie  = movies
		ratingObject.rating = rate
		ratingObject.save()
		messages.success(request,"Your Rating is submited ")
		return redirect("index")
	return render(request,'web/detail.html',{'movies':movies})


# Register user
def signUp(request):
	form =UserForm(request.POST or None)
	if form.is_valid():
		user      = form.save(commit=False)
		username  =	form.cleaned_data['username']
		password  = form.cleaned_data['password']
		user.set_password(password)
		user.save()
		user = authenticate(username=username,password=password)
		if user is not None:
			if user.is_active:
				login(request,user)
				return redirect("index")
	context ={
		'form':form
	}
	return render(request,'web/signUp.html',context)				


# Login User
def Login(request):
	if request.method=="POST":
		username = request.POST['username']
		password = request.POST['password']
		user     = authenticate(username=username,password=password)
		if user is not None:
			if user.is_active:
				login(request,user)
				return redirect("index")
			else:
				return render(request,'web/login.html',{'error_message':'Your account disabled'})
		else:
			return render(request,'web/login.html',{'error_message': 'Invalid Login'})
	return render(request,'web/login.html')

#Logout user
def Logout(request):
	logout(request)
	return redirect("login")




