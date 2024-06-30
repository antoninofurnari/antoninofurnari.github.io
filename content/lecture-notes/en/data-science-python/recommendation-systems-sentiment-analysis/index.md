---
title: "Recommendation Systems, Sentiment Analysis"
disable_share: true
toc: true
note:
    lang: eng
math: true
---

In this laboratory we will see how to:
 * Build a recommendation system based on collaborative filtering;
 * Use word embeddings using Spacy;
 * Perform sentiment analysis with VADER;

## 1. Recommendation Systems

### 1.1 Dataset
We will use a dataset movies rated by users. We can load the data directly from its URL as follows:


```python
import pandas as pd
data = pd.read_csv('http://antoninofurnari.it/downloads/movieratings.csv')
data.info()
data.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100003 entries, 0 to 100002
    Data columns (total 4 columns):
     #   Column      Non-Null Count   Dtype 
    ---  ------      --------------   ----- 
     0   user_id     100003 non-null  int64 
     1   item_id     100003 non-null  int64 
     2   rating      100003 non-null  int64 
     3   item_title  100003 non-null  object
    dtypes: int64(3), object(1)
    memory usage: 3.1+ MB





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
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>item_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>50</td>
      <td>5</td>
      <td>Star Wars (1977)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>290</td>
      <td>50</td>
      <td>5</td>
      <td>Star Wars (1977)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>79</td>
      <td>50</td>
      <td>4</td>
      <td>Star Wars (1977)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>50</td>
      <td>5</td>
      <td>Star Wars (1977)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>50</td>
      <td>5</td>
      <td>Star Wars (1977)</td>
    </tr>
  </tbody>
</table>
</div>



The dataset contains $100003$ observations. Each observation consists in a rating given by a user to a movie. Specifically, the variables have the following meaning:
 * `user_id` the id of the user rating an item;
 * `item_id` the id of the item (a movie);
 * `rating` a number between $1$ and $5$ indicanting the rating given by the user to the movie;
 * `item_title`: the title of the rated movie.

Each user can rate an item at most $1$ time, hence each line can be related to different users rating the same item or to different items rated by the same user. Let's count how many unique users and items we have in the dataset:


```python
print("Number of users: {}".format(data['user_id'].nunique()))
print("Number of items: {}".format(data['item_id'].nunique()))
```

    Number of users: 944
    Number of items: 1682

<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="qmark.jpg"></td>
<td>

**Question 1**

How many rows we would have in the dataset if each user had rated each item? If we constructed a utility matrix, would it be sparse of dense?


 </td>
</tr>
</table>


Before proceeding, let's split the dataset into a training and a test set:


```python
import numpy as np
import random
from sklearn.model_selection import train_test_split
np.random.seed(42)
random.seed(42)
train_data, test_data = train_test_split(data, test_size=0.25, )
```

Let's now build the utility matrix. Recall that it is a matrix showing how a user has rated a given item. We can create the utility matrix using a pivot table:


```python
U = train_data.pivot_table(index='user_id',columns='item_id',values='rating')
U.head()
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
      <th>item_id</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>1673</th>
      <th>1674</th>
      <th>1675</th>
      <th>1676</th>
      <th>1677</th>
      <th>1678</th>
      <th>1679</th>
      <th>1680</th>
      <th>1681</th>
      <th>1682</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1638 columns</p>
</div>



The matrix is sparse. It contains `NaN` values for user-movies pairs for which we do not have a rating.

### 1.2 Collaborative Filtering

We will now implement a user-user collaborative filter. Remember that a user-user collaborative filter works as follows:
 1. Consider a user $x_i$ and an item $s_j$ which not been rated by user $x_i$;
 2. Build a profile for each user by considering the rows of the utility matrix normalized by subtracting the mean;
 3. Find a set $N$ of similar users who have rated item $s_j$;
 4. Estimate the utility value $u(x_i, s_j)$ computing a weighted average of rating given by the similar users;

Let's first see an example for `N=3`, `x_i=0` and `s_j=1`:


```python
N=3
xi = 0
sj = 1
```

First, we need to compute user profiles for all users. We can do this by first replacing all missing values with zeros, then normalizing each row by subtracting its mean (computed before adding the zeros):


```python
profiles = (U-U.mean(1)).fillna(0)
profiles.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>1673</th>
      <th>1674</th>
      <th>1675</th>
      <th>1676</th>
      <th>1677</th>
      <th>1678</th>
      <th>1679</th>
      <th>1680</th>
      <th>1681</th>
      <th>1682</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.300518</td>
      <td>-0.891304</td>
      <td>1.285714</td>
      <td>-1.315789</td>
      <td>0.253968</td>
      <td>1.426829</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.722222</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.300518</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1642 columns</p>
</div>

<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="qmark.jpg"></td>
<td>

**Question 2**

Why do we need to subtract the mean to obtain user profiles? 

 </td>
</tr>
</table>



Now we should compute the cosine similarity between the row corresponding to $x_i$ and all the other rows:


```python
import numpy as np
cosine = lambda x,y: np.dot(x, y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))

profiles.apply(lambda x: cosine(profiles.loc[xi],x), axis=1).head()
```




    user_id
    0    1.000000
    1    0.004347
    2    0.087988
    3    0.000000
    4    0.116171
    dtype: float64



The $j^{th}$ row reports the similarity distance between the $j^{th}$ profile and the profile of user $x_i$. The largest similarity (1) is between $x_i$ and $x_i$. Let's remove this row:


```python
similarities = profiles.apply(lambda x: cosine(profiles.loc[xi],x), axis=1).drop(xi)
similarities.head()
```




    user_id
    1    0.004347
    2    0.087988
    3    0.000000
    4    0.116171
    5    0.000000
    dtype: float64



We should now select only the users who have rated item $s_j$:


```python
selected_users = U.loc[:,sj].dropna().index
selected_users
```




    Int64Index([  1,   2,   5,   6,  10,  13,  17,  20,  21,  25,
                ...
                923, 924, 927, 929, 930, 932, 933, 936, 938, 941],
               dtype='int64', name='user_id', length=355)


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="qmark.jpg"></td>
<td>

**Question 3**


Why do we need to select the users who have rated the considering item?


 </td>
</tr>
</table>


Let's now select the distances related to the selected users:


```python
similarities.loc[selected_users].head()
```




    user_id
    1     0.004347
    2     0.087988
    5     0.000000
    6     0.011841
    10   -0.108046
    dtype: float64



We can now sort the distances with `sort_values`:

The last rows represent the most similar items. Let's select the last two items:


```python
selected_similarities = similarities.loc[selected_users].sort_values()[-N:]
selected_similarities
```




    user_id
    395    0.207021
    679    0.214723
    97     0.345497
    dtype: float64



We can now store the user ids in an array:


```python
similar_users = selected_similarities.index
similar_users
```




    Int64Index([395, 679, 97], dtype='int64', name='user_id')



Let's now see how the most similar users have rated movie $s_j$:


```python
print(U.loc[similar_users[0],sj], U.loc[similar_users[1],sj], U.loc[similar_users[2],sj])
```

    5.0 3.0 4.0


We can now compute the average rating using a weighted average:


```python
predicted_rating = (U.loc[similar_users,sj]*selected_similarities).sum()/selected_similarities.sum()
predicted_rating
```




    3.9899620761918646



For convenience, let's now write an object to perform recommendations following the `scikit-learn` API:


```python
class CollaborativeFilter():
    def __init__(self, N):
        self.N = N
        self.cosine = lambda x,y: np.dot(x, y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))
        
    def fit(self, U):
        self.U = U
        self.profiles = (U-U.mean(1)).fillna(0)
        
    def predict(self, xi, sj):
        similarities = self.profiles.apply(lambda x: self.cosine(self.profiles.loc[xi],x), axis=1).drop(xi)
        selected_users = self.U.loc[:,sj].dropna().index
        selected_similarities = similarities.loc[selected_users].sort_values()[-self.N:]
        similar_users = selected_similarities.index
        predicted_rating = (self.U.loc[similar_users,sj]*selected_similarities).sum()/selected_similarities.sum()
        return predicted_rating
```

We can use the Collaborative Filter as follows:


```python
cf = CollaborativeFilter(N=3)
cf.fit(U)
print(cf.predict(xi,sj))
print(cf.predict(5,18))
```

    3.9899620761918646
    4.125945120766069


### 1.3 Performance Assessment
To assess the performance of our method, we can see how the filter works with the test data, for which we have the ground truth ratings:


```python
test_data.head()
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
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>item_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87184</th>
      <td>510</td>
      <td>326</td>
      <td>4</td>
      <td>G.I. Jane (1997)</td>
    </tr>
    <tr>
      <th>98871</th>
      <td>151</td>
      <td>1264</td>
      <td>4</td>
      <td>Nothing to Lose (1994)</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>1</td>
      <td>265</td>
      <td>4</td>
      <td>Hunt for Red October, The (1990)</td>
    </tr>
    <tr>
      <th>11529</th>
      <td>546</td>
      <td>219</td>
      <td>5</td>
      <td>Nightmare on Elm Street, A (1984)</td>
    </tr>
    <tr>
      <th>39495</th>
      <td>457</td>
      <td>183</td>
      <td>5</td>
      <td>Alien (1979)</td>
    </tr>
  </tbody>
</table>
</div>



Let's define a function to iterate over the rows of the test set and compute the related ratings:


```python
from tqdm import tqdm
def predict(cf, test_data):
    predicted_ratings = []
    for i, ann in tqdm(test_data.iterrows(), total=len(test_data)):
        #if the user was not in the original utility matrix, let's just append a nan
        try:
            rating = cf.predict(ann['user_id'], ann['item_id'])
        except:
            rating = np.nan
        predicted_ratings.append(rating)
    return np.array(predicted_ratings)
```

To speedup testing, let's select a few items randomly from the test set:


```python
_, test_data_tiny = train_test_split(test_data, test_size=0.02)
len(test_data_tiny)
```




    501




```python
cf3 = CollaborativeFilter(N=3)
cf3.fit(U)
predicted_ratings_3 = predict(cf3, test_data_tiny)
```

    100%|█████████████████████████████████████████| 501/501 [00:33<00:00, 14.87it/s]


We can see recommendation as a regression problem and evaluate the system using MAE:


```python
def mae(y_true, y_pred):
    return (y_true-y_pred).abs().mean()

mae(test_data_tiny['rating'], predicted_ratings_3)
```




    0.8396598738455379



Let's compare this with a collaborative filter with `N=5`:


```python
cf5 = CollaborativeFilter(N=5)
cf5.fit(U)
predicted_ratings_5 = predict(cf5, test_data_tiny)
mae(test_data_tiny['rating'], predicted_ratings_5)
```
    100%|█████████████████████████████████████████| 501/501 [00:34<00:00, 14.51it/s]
    0.788623628125405



Let's push N to some large value such as $50$:


```python
cf50 = CollaborativeFilter(N=50)
cf50.fit(U)
predicted_ratings_50 = predict(cf50, test_data_tiny)
mae(test_data_tiny['rating'], predicted_ratings_50)
```
    100%|█████████████████████████████████████████| 501/501 [00:32<00:00, 15.34it/s]
    0.8032535120960659


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="qmark.jpg"></td>
<td>

**Question 4**

Why do we obtain better results for larger Ns? What is the effect of choosing a very small N?

 </td>
</tr>
</table>




If you are interested in recommendation systems, you can check the `surprise` library: http://surpriselib.com/.

## 2. Word Embeddings

SpaCy provides different pre-trained word embeddings which can be used "off-the-shelf". The small model we have used does not include word embeddings, so we need to download a larger one. Specifically, SpaCy provides different models including different vocabulary sizes:

 * [**en_core_web_md**](https://spacy.io/models/en#en_core_web_md) (116MB) 20K embeddings
 * [**en_core_web_lg**](https://spacy.io/models/en#en_core_web_lg) (812MB) 685K embeddings
 * [**en_vectors_web_lg**](https://spacy.io/models/en#en_vectors_web_lg) (631MB) 1.1M embeddings
 
All Embeddings have $300$ dimensions. The embeddings have not been learned using the algoritms seen in this course (GloVe). Instead they have learned using an algorithm called "word2vec". The underlying principle behind this algorithm is however the same: words which are used in a similar way have similar embeddings, also called "vectors".

We will use `en_code_web_md`, but the larger models should also been considered when building applications which strongly rely on word vectors. We can install the model with the following command:

> `python -m spacy download en_core_web_md`  

After installing the model, we can load it as follows:


```python
import spacy
nlp = spacy.load('en_core_web_md')
doc = nlp("Is this the region, this the soil, the clime")
```

We can access the word vector with the `vector` property of each token:


```python
v = doc[0].vector
print(len(v)) #vector dimension
v[:10] #let's show only the first 10 components of the vector
```

    300
    array([ 1.6597  ,  4.7975  ,  0.49976 , -0.39231 , -3.1763  ,  2.5721  ,
            0.023483, -0.047588, -2.3754  ,  3.5058  ], dtype=float32)



### 2.1 Token Properties
Each token also has three additional properties:
 * `is_oov`: indicates whether the word is out of vocabulary;
 * `has_vector`: indicates whether the word has a word embedding;
 * `vector_norm`: is the L2 norm of the word vector.

Let's see an example:


```python
doc = nlp('Heilo how are you?')
for w in doc:
    print(w.text, "\t", w.is_oov, w.has_vector, w.vector_norm)
```

    Heilo 	 False True 37.983173
    how 	 False True 90.45195
    are 	 False True 89.23195
    you 	 False True 70.9396
    ? 	 False True 68.08072


The word "hello" has been mispelled as "heilo", hence it has been identified as "out of vocabulary". Since no word embedding is availble for this word, a vector containing all zeros is assigned (hence the vector norm is 0). Let's check the word vector:


```python
doc[0].vector[:10] #first 10 components
```




    array([-2.6324 , -0.39889,  1.0277 , -0.22824,  0.58977, -2.9811 ,
           -0.34429,  0.91616,  0.7227 , -2.3488 ], dtype=float32)



 We can obtain a word vector for a document by averaging the vectors of all the words:


```python
import numpy as np

v = np.mean([w.vector for w in doc],0)
v[:10]
```




    array([-1.77478  ,  2.635882 , -5.93964  , -1.59687  , -1.28356  ,
            1.2811   , -1.7377741,  2.723094 , -3.9060802,  1.7923   ],
          dtype=float32)



The same result can be obtained by simply calling `vector` on the document:


```python
doc.vector[:10]
```




    array([-1.77478  ,  2.635882 , -5.93964  , -1.59687  , -1.28356  ,
            1.2811   , -1.7377741,  2.723094 , -3.9060802,  1.7923   ],
          dtype=float32)


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="qmark.jpg"></td>
<td>

**Question 5**

Are there any shortcomings in averaging word vectors to obtain embeddings for sentences? What happens if the sentence is very long and related to different topics?


 </td>
</tr>
</table>

### 2.2 Similarity
Each document/vector has a `similarity` method which allows to compute the similarity between two documents/vectors based on their word embedding:


```python
doc = nlp("Is this the region, this the soil, the clime")
word = nlp("that")
for w in doc:
    print("{} vs {} \t {:02f}".format(word.text, w.text,w.similarity(word)))
```

    that vs Is 	 -0.127594
    that vs this 	 0.716428
    that vs the 	 0.549108
    that vs region 	 0.337794
    that vs , 	 0.405670
    that vs this 	 0.716428
    that vs the 	 0.549108
    that vs soil 	 0.281650
    that vs , 	 0.405670
    that vs the 	 0.549108
    that vs clime 	 0.440914


Similarly with documents:


```python
doc1 = nlp("Love is real, real is love")
doc2 = nlp("All you need is love")
doc3 = nlp("Principles of Accounting is designed to meet the scope of a two-semester accounting course.")
print("Similarity between document 1 and 2: {:02f}".format(doc1.similarity(doc2)))
print("Similarity between document 1 and 3: {:02f}".format(doc1.similarity(doc3)))
```

    Similarity between document 1 and 2: 0.735970
    Similarity between document 1 and 3: 0.503123


### 2.3 Word Arithmetic
Let's see an example of arithmentic between word embeddings. For instance, let's try to see what is the closest word to:

<pre>"brother" - "man" + "woman"</pre>


```python
from tqdm import tqdm
brother = nlp('brother')
man = nlp('man')
woman = nlp('woman')
sister = nlp('sister')

v = brother.vector - man.vector + woman.vector

cosine = lambda x,y: np.dot(x, y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))

words = []
similarities = []
for word in tqdm(nlp.vocab, total=len(nlp.vocab)):
    if word.has_vector:
        if word.is_lower:
            words.append(word.text)
            similarities.append(cosine(v, word.vector))
np.array(words)[np.argsort(similarities)[::-1]]
```

    100%|█████████████████████████████████████| 790/790 [00:00<00:00, 213059.42it/s]
    array(['sister', 'brother', 'she', 'and', 'who', 'woman', 'havin', 'b.',
           'accounting', 'where', '’cause', 'that', 'that’s', 'who’s',
           'lovin', 'lovin’', 'semester', 'when', 'was', 'the', 'co', 'would',
           'somethin', 'r.', 'c.', 'region', 'designed', 'had', 'love', 'd.',
           'to', 'they', 'two', 'were', 'he', 'those', 'could', 'might',
           '’bout', 'cause', 'there', "there's", 'clime', 'course', 'should',
           'nothin', 'these', 'meet', 'ought', "'s", 'a', 'not', 'and/or',
           'e.g.', 'have', 'did', 'has', 'o.o', 'real', 'you', 'may', 'this',
           'what', "what's", 'of', '-o', '’s', 'c', 'ä', 'ö', 'i.e.', 'all',
           'how', 'space', 'e.', 'u.', 'does', 'b', "somethin'", 'somethin’',
           'must', 'scope', 'm.', 'are', 'it', 'bout', 's.', 'why', 're',
           'soil', 'need', 'f.', 'n.', 'j.', 'l.', 'i.', 'o.', 'or', 'is',
           "that's", 'co.', 'v.', 'p', 'p.', 'h', 'h.', 'on', 'ü.', 'j', 'z.',
           'y.', 'x.', 'z', 'q', 'k', 'k.', 'q.', "n't", 'n’t', 'can', '‘s',
           'am', 'we', "'cause", 'g.', '-p', '-x', '\\t', 'e', 'let’s', 'y',
           'ü', 'r', 'p.m.', 'g', 'ol', 'd', 'a.m.', '’m', 'a.', 'n', 'f',
           'x', 'm', 's', 'man', 'i', 'do', 'w.', 'w', ':x', 'let', 'vs.',
           'sha', 'v', "who's", "let's", "'m", ':o)', 'pm', 'a.m', 'p.m',
           "'d", "o'clock", 'o’clock', 've', 'l', '’re', "he's", "she's",
           "it's", 'v.s.', 'e.g', 'i.e', "'re", '’d', 'c’m', "c'm", '’ve',
           "havin'", 'w/o', 'got', ':o', ':p', ':-p', 'ca', 'u', "'ve", 'gon',
           'll', '’ll', "'ll", 'na', 't.', 't', 'wo', "nuthin'", 'nothin’',
           "nothin'", '\\n', 'it’s', 'dare', 'vs', "doin'", "'bout", 'doin’',
           'doin', 'o', "'coz", 'nt', "ma'am", 'y’', "y'", 'goin', 'goin’',
           "goin'", 'ta', 'ma’am', 'cuz', "'cuz", 'cos', "'cos", 'coz', '’em',
           'em', 'nuff', "'nuff", 'ai', "'em", 'ol’', "ol'", "lovin'"],
          dtype='<U10')



As one could expect, the closest word is "sister".

<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="qmark.jpg"></td>
<td>

**Question 6**

Why are "4-year-old" and "daughter" also close to the computed vector?


 </td>
</tr>
</table>



## 3. Sentiment Analysis
We will now see how to perform sentiment anlysis on a text using VADER. We will use a module included in the NLTK library. First, we need to download the VADER lexicon as follows:


```python
import nltk
nltk.download('vader_lexicon')
```

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     /Users/...
    True



We will use the `SentimentIntensityAnalyzer` object:


```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
```

To obtain the `negative`, `neutral`, `positive` and `compound` scores of VADER, we will use the `polarity_scores` method:


```python
sid.polarity_scores("The movie was great.")
```




    {'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.6249}



This will return a dictionary containing the four scores. Let's try with another few examples:


```python
sid.polarity_scores("The movie was awful.")
```




    {'neg': 0.5, 'neu': 0.5, 'pos': 0.0, 'compound': -0.4588}




```python
sid.polarity_scores("The movie was more than GREAT!")
```




    {'neg': 0.0, 'neu': 0.481, 'pos': 0.519, 'compound': 0.7509}




```python
sid.polarity_scores("The movie was OK")
```




    {'neg': 0.0, 'neu': 0.506, 'pos': 0.494, 'compound': 0.4466}


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="qmark.jpg"></td>
<td>

**Question 7**

Compare the third example to the first one. Why does the third one have a larger compound value than the first one?


 </td>
</tr>
</table>


### 3.1 Sentiment Analysis and Movie Reviews
Let's now see how we can use VADER to analyze movie reviews. We will use the movie reviews dataset seen in the previous laboratries:


```python
import pandas as pd
reviews=pd.read_csv('http://antoninofurnari.it/downloads/reviews.csv')
reviews.info()
reviews.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5006 entries, 0 to 5005
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   author  5006 non-null   object 
     1   review  5006 non-null   object 
     2   rating  5006 non-null   float64
    dtypes: float64(1), object(2)
    memory usage: 117.5+ KB





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
      <th>author</th>
      <th>review</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dennis_Schwartz</td>
      <td>in my opinion , a movie reviewer's most import...</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dennis_Schwartz</td>
      <td>you can watch this movie , that is based on a ...</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dennis_Schwartz</td>
      <td>this is asking a lot to believe , and though i...</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dennis_Schwartz</td>
      <td>no heroes and no story are the main attributes...</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dennis_Schwartz</td>
      <td>this is not an art movie , yet i saw it an art...</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



Let's analyze the first review with VADER:


```python
sid.polarity_scores(reviews.loc[0]['review'])
```




    {'neg': 0.134, 'neu': 0.753, 'pos': 0.113, 'compound': -0.8923}



The "compound" score is negative, which is coherent with low rating assigned to the review. Let's see if this happens systematically. To do so, we need to compute the "compound" value for each review. We will first define a `vader_polarity` function to compute the polarity of the review using the compound:


```python
vader_polarity = lambda x: sid.polarity_scores(x)['compound']
vader_polarity(reviews.loc[0]['review'])
```




    -0.8923



Let's now compute the polarity score to each review (this might take a while):


```python
tqdm.pandas()
reviews['polarity']=reviews['review'].progress_apply(vader_polarity)
reviews.head()
```

    100%|██████████████████████████████████████| 5006/5006 [00:13<00:00, 378.55it/s]





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
      <th>author</th>
      <th>review</th>
      <th>rating</th>
      <th>polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dennis_Schwartz</td>
      <td>in my opinion , a movie reviewer's most import...</td>
      <td>0.1</td>
      <td>-0.8923</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dennis_Schwartz</td>
      <td>you can watch this movie , that is based on a ...</td>
      <td>0.2</td>
      <td>0.8927</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dennis_Schwartz</td>
      <td>this is asking a lot to believe , and though i...</td>
      <td>0.2</td>
      <td>0.9772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dennis_Schwartz</td>
      <td>no heroes and no story are the main attributes...</td>
      <td>0.2</td>
      <td>0.0316</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dennis_Schwartz</td>
      <td>this is not an art movie , yet i saw it an art...</td>
      <td>0.2</td>
      <td>0.9903</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2 Classifying reviews with no training
To see if the polarity tells us something about the rating, let's consider a binary classification task. We will consider a review as negative if the rating is smaller than $0.5$. Similarly, we will classify a review as positive if the polarity is positive:


```python
reviews['label']=reviews['rating']>=0.5
reviews['predicted_label']=reviews['polarity']>0
reviews.head()
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
      <th>author</th>
      <th>review</th>
      <th>rating</th>
      <th>polarity</th>
      <th>label</th>
      <th>predicted_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dennis_Schwartz</td>
      <td>in my opinion , a movie reviewer's most import...</td>
      <td>0.1</td>
      <td>-0.8923</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dennis_Schwartz</td>
      <td>you can watch this movie , that is based on a ...</td>
      <td>0.2</td>
      <td>0.8927</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dennis_Schwartz</td>
      <td>this is asking a lot to believe , and though i...</td>
      <td>0.2</td>
      <td>0.9772</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dennis_Schwartz</td>
      <td>no heroes and no story are the main attributes...</td>
      <td>0.2</td>
      <td>0.0316</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dennis_Schwartz</td>
      <td>this is not an art movie , yet i saw it an art...</td>
      <td>0.2</td>
      <td>0.9903</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Let's now assess the performance of our classifier:


```python
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy: {:0.2f}%".format(accuracy_score(reviews['label'], reviews['predicted_label'])*100))
confusion_matrix(reviews['label'], reviews['predicted_label'])
```
    Accuracy: 74.03%
    array([[ 434,  823],
           [ 477, 3272]])



The model is not perfect, but the result is not bad, considering that we have not trained at model at all!

<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="qmark.jpg"></td>
<td>

**Question 8**


How does this model compares with respect to the model base don bag of words seen in the previous laboratories?


 </td>
</tr>
</table>

### 3.3 VADER Scores as Features
We can do something slightly more complicated by considering all VADER scores as features and training a logistic regressor on top of them. We should now define a function that maps a review to a feature vector containing the VADER scores:


```python
vader_features = lambda x: np.array(list(sid.polarity_scores(x).values()))
vader_features(reviews.iloc[0]['review'])
```




    array([ 0.134 ,  0.753 ,  0.113 , -0.8923])



We will use a very small training set to see what we can done with very little training.


```python
from sklearn.model_selection import train_test_split
np.random.seed(123)
reviews_train, reviews_test = train_test_split(reviews, test_size=0.99) #let's use only 1% of the data for training
print(len(reviews_train))
```

    50


Let's compute a feature vector for each review of the training and test sets:


```python
x_train = np.vstack(list(reviews_train['review'].progress_apply(vader_features)))
x_test = np.vstack(list(reviews_test['review'].progress_apply(vader_features)))
```

    100%|██████████████████████████████████████████| 50/50 [00:00<00:00, 341.12it/s]
    100%|██████████████████████████████████████| 4956/4956 [00:12<00:00, 382.48it/s]


Let's obtain the related labels:


```python
y_train = reviews_train['label']
y_test = reviews_test['label']
```

Let's now train the logistic regressor. We will normalize features using a MinMaxScaler:


```python
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train, y_train)
reg.score(x_test, y_test)
```




    0.7487893462469734



Using only $50$ reviews for training we have obtained a small boost on performance. We can also try to fit a linear regressor to predict the ratings:


```python
from sklearn.linear_model import LinearRegression
y_train = reviews_train['rating']
y_test = reviews_test['rating']

reg = LinearRegression()
reg.fit(x_train, y_train)
y_test_pred = reg.predict(x_test)
mae(y_test, y_test_pred)
```




    0.14381416319971027



We have obtained a relatively small MAE error, considering the range of the reviews.

## References
[1] Recommendation systems library. http://surpriselib.com/

[2] Word vectors in Spacy. https://spacy.io/usage/vectors-similarity

[3] NLTK and VADER. https://www.nltk.org/_modules/nltk/sentiment/vader.html

# Exercises

<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="code.png"></td>
<td>

**Exercise 1**

Build an item-item Collaborative Filter and compare its performance with respect to the user-user Collaborative Filter built in this laboratory. Experiment with different values of `N`. Which of the two approaches achieves the best performance?


 </td>
</tr>
</table>


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="code.png"></td>
<td>

**Exercise 2**
Represent each review of the movie review dataset using the word embeddings. Use a small portion of the data to train a logistic regressor for review classification on the dataset. How does this model compares with respect to the one built using VADER words embeddings?

 </td>
</tr>
</table>

<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none;" src="code.png"></td>
<td>

**Exercise 3**

Consider the ham vs spam dataset seen in the previous laboratories. Represent each element using features extracted with VADER and word embeddings. Train two logistic regressors using the two set of features using a small portion of the data as training set. Compare the two models with a logistic regressor built starting from a bag of words representation. Use the same train/test split for training/evaluation. Which of the models performs best? Why?

 </td>
</tr>
</table>
