#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[2]:


movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('dataset/tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies.shape


# In[6]:


credits.shape


# In[7]:


#now we merge the dataset
movie= movies.merge(credits, on='title')


# In[8]:


movie.head()


# In[9]:


movie.shape


# In[10]:


movie.columns


# In[11]:


#now we select only important column because some irrelevant column removes that we didn't need
nmovie = movie[['movie_id','title','overview','genres','keywords','original_language','cast', 'crew']]


# In[12]:


nmovie.shape


# preprocessing/analyze the data

# In[13]:


#overview of new dataset
nmovie.isnull().sum()


# In[14]:


nmovie.dropna(inplace=True)


# In[15]:


nmovie.shape


# In[16]:


#check any duplicate movie or not
nmovie.duplicated().sum()


# In[17]:


nmovie.head(2)


# In[18]:


nmovie.iloc[0]['genres']


# In[19]:


#covert str to list through ast.literal_eval for genres
import ast

def convert_genres(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
        
    return l


# In[20]:


nmovie['genres']=nmovie['genres'].apply(convert_genres)


# In[21]:


nmovie.head(2)


# In[22]:


nmovie.iloc[0]['keywords']


# In[23]:


#covert str to list through ast.literal_eval for keywords
import ast

def convert_keywords(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
        
    return l


# In[24]:


nmovie['keywords']=nmovie['keywords'].apply(convert_keywords)


# In[25]:


nmovie.head(2)


# In[26]:


nmovie.iloc[0]['cast']


# In[27]:


#covert str to list through ast.literal_eval for cast
import ast

def convert_cast(text):
    l = []
    counter=0
    for i in ast.literal_eval(text):
        if counter<4:
            l.append(i['name'])
        counter+=1
            
        
    return l


# In[28]:


nmovie['cast']=nmovie['cast'].apply(convert_cast)
nmovie.head(2)


# In[29]:


nmovie.iloc[0]['crew']


# In[30]:


#covert str to list through ast.literal_eval for crew
import ast

def check_director(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job']=='Director':
            l.append(i['name'])
            break
            
        
    return l


# In[31]:


nmovie['crew']=nmovie['crew'].apply(check_director)
nmovie.head(2)


# In[32]:


nmovie.iloc[0]['overview']


# In[33]:


#covert into list
nmovie['overview']=nmovie['overview'].apply(lambda x:x.split())
nmovie.head()


# In[34]:


nmovie.iloc[0]['overview']


# In[35]:


nmovie.head(2)


# In[36]:


#convert str to list for language
nmovie['original_language'] = nmovie['original_language'].apply(lambda x: [x])


# In[37]:


nmovie.iloc[0]['original_language']


# In[38]:


language_counts = nmovie['original_language'].value_counts()
print(language_counts)


# In[39]:


#recome spaces on all columns
def remove_space(word):
    l=[]
    for i in word:
        l.append(i.replace(" ",""))
    return l


# In[40]:


nmovie['cast']=nmovie['cast'].apply(remove_space)
nmovie['crew']=nmovie['crew'].apply(remove_space)
nmovie['keywords']=nmovie['keywords'].apply(remove_space)
nmovie['genres']=nmovie['genres'].apply(remove_space)


# In[41]:


nmovie.head()


# In[42]:


#generate tags
nmovie['tags'] = nmovie['overview']+nmovie['genres']+nmovie['keywords']+nmovie['original_language']+nmovie['cast']+nmovie['crew']


# In[43]:


nmovie.head(2)


# In[44]:


nmovie.iloc[0]['tags']


# In[45]:


#remove some columns that we concatenate in tags
newmoviedf=nmovie[['movie_id','title','tags']]


# In[46]:


newmoviedf.head()


# In[47]:


#convert list to str in tags
newmoviedf['tags']=newmoviedf['tags'].apply(lambda x:" ".join(x))


# In[48]:


newmoviedf.head()


# In[49]:


newmoviedf.iloc[0]['tags']


# In[50]:


#here we see some characters in uppercase and lowercase so we convert all uppercase into lowercase
newmoviedf['tags']=newmoviedf['tags'].apply(lambda x:x.lower())


# In[51]:


newmoviedf.head()


# In[52]:


newmoviedf.iloc[0]['tags']


# In[53]:


newmoviedf.shape


# In[54]:


import nltk
from nltk.stem import PorterStemmer


# In[55]:


ps = PorterStemmer()


# In[56]:


def stems(text):
    l=[]
    for i in text.split():#convert string into list
        l.append(ps.stem(i))
    
    return " ".join(l) #again convert to string through .join


# In[57]:


newmoviedf['tags'] = newmoviedf['tags'].apply(stems)


# In[58]:


newmoviedf['tags'][0]


# now we convert all tags into vector bcz find closest vector through similar words
# 
# texts ------> vector(Bag of words)

# In[59]:


from sklearn.feature_extraction.text import CountVectorizer
#removing unwanted words like in is etc
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[60]:


vector = cv.fit_transform(newmoviedf['tags']).toarray()


# In[61]:


vector


# In[62]:


vector[0]


# In[63]:


vector.shape


# In[64]:


cv.get_feature_names()


# here we have 5000 dimensional co-ordinate and all movie are vector 
# now we calculate distance between two movies but 
# if we get max distance then similarity is low viceversa
# 
# so for this we use cosine distance

# In[65]:


#cosine semalirity using
from sklearn.metrics.pairwise import cosine_similarity


# In[66]:


similarity=cosine_similarity(vector)#we calculate all each vector to all vectords one by one
similarity


# In[67]:


similarity[2]


# In[68]:


similarity.shape


# In[69]:


def recommend(movie):
    # Filter the DataFrame to find the index of the movie
    movies_index = newmoviedf[newmoviedf['title'] == movie]
    
    if not movies_index.empty:  # Check if any movie is found
        movies_index = movies_index.index[0] #Fetching the index of that movie
        distances = similarity[movies_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        print("Movies you may like:")
        for i in movies_list:
            print(newmoviedf.iloc[i[0]]['title'])
    else:
        print("Sorry, the movie is not found in the database.")
        

recommand_movie = input('Movie, you want to watch: ')
print('')
recommend(recommand_movie)


# In[70]:


import pickle
pickle.dump(newmoviedf,open('artificats/movie_dict.pkl','wb'))
pickle.dump(newmoviedf,open('artificats/movie_list.pkl','wb'))
pickle.dump(similarity,open('artificats/similarity_list.pkl','wb'))


# In[ ]:




