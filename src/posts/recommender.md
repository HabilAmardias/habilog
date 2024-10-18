---
title: Recommender System Overview
desc: Things I learned about recommender system
date: '2024-10-16'
tags:
    - Recommender System
    - Machine Learning
published: true
---


## Introduction

Recommender systems are tools (systems) designed to provide suggestions for items that a user might be interested in. Recommender system analyze user preference to recommend product, services or content to user. 
It’s commonly used in online service providers such as e-commerce, streaming services and social media to provide personalized recommendations. 
Recommender system can reduce user effort to finding items they preferred which can affect user satisfaction, indirectly add business value to online service providers. 
In general, before recommender system provice recommendation, users have to give some kind of review or feedback to a product or service and then retrieved by recommender system as input to give recommendation to users based on those feedback.

![Recommender System](recommender_system.png)
*Illustration of Recommender System (Ferreira, 2020)*

## Feedback

Recommender system need user feedbacks to give recommendation, there are 2 type of feedbacks used for recommender system;


- **Explicit Feedback**, is a rating explicitly given by users to express their satisfaction with an item, for example numbers of stars after buying a product, like or dislike after watching a video, etc. 
Note that this feedback provides detailed information, but it is hard to collect as most users typically don’t give an explicit rating for each item


- **Implicit Feedback**, is mainly concerned with user-item interactions, for example purchases history, browsing history, list of songs played by a user, user clicks, etc. 
The advantage of implicit feedback is often readily available. However, this feedback is less detailed compared to explicit feedback and noisier, for example a user may buy a product for someone else.


Once the system collected explicit/implicit feedback, the system can create interaction matrix $R\in \mathbb{R}^{m×n}$ with $m$ users and $n$ items, and the entry of $R$, $r_{ui}\in \mathbb{R}$, 
represent quantified feedback from user $i$ for item $j$.

![Interaction Matrix](interaction_matrix.webp)
*Source: [Recommender Systems — A Complete Guide to Machine Learning Models](https://tinyurl.com/3vbp8akr)*

## Approach

There are 3 main types of recommender system;

- **Content-based Filtering** rely on the description of an item and a user's profile to make suggestions. Each item, such as a movie or book, 
is characterized by a set of tags, while each user is represented by a profile based on their preferences and past behavior (e.g., previously watched movies). 
For instance, if a user has watched or is currently exploring a specific movie, the system will recommend other similar titles.
However note that since this approach bases its recommendations on the user's past behavior, it may recommend similar items over and over, limiting diversity in the recommendations (over-specialization).

In content-based filtering, the recommendation process follows four essential steps. It starts with extracting the features that describe each item in the system. 
For example, a popular method like TF-IDF (Term Frequency-Inverse Document Frequency) is often used to capture the most relevant characteristics of an item.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

vectorizer=TfidfVectorizer(stop_words='english')
tfidf_matrix=pd.DataFrame(vectorizer.fit_transform(movies['genres']).toarray(),index=movies['movieId'])
```

Once the item features are identified, the system then builds a user profile based on their previous interactions, such as items they've liked or engaged with. 
This profile serves as a representation of the user's preferences.

```python
def get_user_preference_vector(ratings:pd.DataFrame,
                               tfidf_matrix:pd.DataFrame,
                               user_id:int,
                               min_rating:int):
    reviewed_movies=ratings[ratings['userId']==user_id] #filter by user id
    reviewed_movies_refined=reviewed_movies[reviewed_movies['rating']>=min_rating] #filter by min rating
    preference=np.zeros((tfidf_matrix.shape[1],),dtype=np.float32)
    denominator=0
    for line in reviewed_movies_refined.itertuples():
        movie_index=line[2]
        movie_rating=line[3]
        movie_vector=tfidf_matrix.loc[movie_index].values
        preference+=movie_vector*movie_rating
        denominator+=movie_rating
    preference/=denominator
    #return user preference and reviewed movie index
    return preference,list(map(lambda x:x.item(),reviewed_movies['movieId'].values))
```

After constructing the user profile, the system moves on to measuring the similarity between the user’s preferences and the features of the items in the dataset. 
Common techniques like Cosine Similarity, Euclidean Distance, or the dot product are used to calculate this similarity score, helping the system determine how closely an item matches the user's tastes.

$$CosineSimilarity(x,y)=\frac{x.y}{\left \lVert x \right \rVert \left \lVert y \right \rVert} \text{, where } x,y \in \mathbb{R}^n,
$$

$$EuclideanDistance(x,y)=\sqrt{\sum_{i=1}^{n} (x_i-y_i)^2} \text{, where } x,y \in \mathbb{R}^n,
$$

Finally, the system recommends items that have the highest similarity to the user’s profile, ensuring personalized suggestions that align with their interests.

```python
def cb_get_recommendation(ratings:pd.DataFrame,
                          titles:pd.DataFrame,
                          tfidf_matrix:pd.DataFrame,
                          user_id:int,
                          min_rating:int,
                          n_recs:int,
                          return_title:bool=True):
    user_preference,reviewed_movies=get_user_preference_vector(ratings,tfidf_matrix,user_id,min_rating)
    recommendation=[]
    for line in tfidf_matrix.itertuples():
        movie_index=line[0]
        if movie_index in reviewed_movies:
            continue #skip if movie is reviewed by user
        movie_vector=np.array(line[1:])
        numerator=np.dot(movie_vector,user_preference)
        denominator=np.linalg.norm(movie_vector)*np.linalg.norm(user_preference)
        similarity=numerator/denominator #measure cosine similarity
        recommendation.append((movie_index,similarity.item()))
    recommendation=sorted(recommendation,key=lambda x: x[1],reverse=True) #sort by similarity, descending
    recommendation=recommendation[:n_recs]
    if return_title:
        recommendation=list(map(lambda x:titles.loc[x[0]].values.item(),recommendation))
    else:
        recommendation=list(map(lambda x:x[0],recommendation))
    return recommendation

print('Recommendation list for user 5 with Content-Based Filtering: ', 
      cb_get_recommendation(ratings,titles,tfidf_matrix,user_id=5,min_rating=3,n_recs=3))
```

- **Collaborative Filtering** takes a different approach to recommendations by focusing on user activity and feedback rather than relying on item features or individual user profiles. This method predicts how a user will rate or engage with a particular item by analyzing the interactions of all users, uncovering patterns and interdependencies between users and items based on shared behaviors.

However, collaborative filtering has its challenges. It struggles in "cold-start" situations, such as when new users or items are introduced to the platform with little or no interaction data. Additionally, issues like scalability and data sparsity can affect its performance, particularly when dealing with large datasets with few interactions.

One of the most popular techniques in this approach is Matrix Factorization, which helps to break down user-item interactions into latent factors, enabling more accurate predictions. Let $R \in \mathbb{R}^{m \times n}$ is interaction matrix with $m$ users and $n$ items, its entry, $r_{ui} \in R$, is quantified feedback from user $u$ for item $i$, matrix factorization goal is to factorize interaction matrix into user latent matrix $P \in \mathbb{R}^{m \times k}$ and item latent matrix $Q \in \mathbb{R}^{n \times k}$ with $k$ is latent factor, or we can say that we want to find $p_u$, $q_i$, $b_u$ and $b_i$ such that

$$argmin_{p_u,q_i,b_u,b_i}\sum_{(u,i) \in K}[(r_{ui}-\hat{r}_{ui})^2+\lambda(\left \lVert p_u \right \rVert _2^2 + \left \lVert q_i \right \rVert _2^2 + b_u^2 +b_i^2)],
$$

$$\hat{r}_{ui} = p_u.q_i + b_u +b_i,
$$

where $p_u \in \mathbb{R}^k$ is u-th user latent vector, $q_i \in \mathbb{R}^k$ is the i-th item latent vector, $b_u$ is bias term for user, $b_i$ is bias term for item, $\lambda$ is regularization term and $K = \{(u,i) \text{ | } r_{ui} \text{ is known}\}$.

To find $p_u$, $q_i$, $b_u$ and $b_i$, we can first initialize both $p_u$ and $q_i$ for each user and item with some random values, and then update $p_u$, $q_i$, $b_u$ and $b_i$ iteratively with gradient-based optimization algorithm optimization such as gradient-descent or Adam to minimize the objective function.

```python
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

def load_data(ratings:pd.DataFrame, num_users:int, num_items:int):
    users, items, scores = [], [], []
    inter = np.zeros((num_users,num_items))
    for line in ratings.itertuples():
        #movieId and userId is one-indexed, so need to subtract ids with 1
        user_index, item_index = int(line[1] - 1), int(line[2] - 1) 
        score = int(line[3])
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        inter[user_index,item_index] = score
    return users, items, scores, inter

u, i, scores, inter = load_data(ratings,num_users,num_items)
inter_tensor=torch.from_numpy(inter).float()
user_ids,item_ids=torch.nonzero(inter_tensor, as_tuple=True)
inters=inter_tensor[user_ids,item_ids]

class MF(nn.Module):
    def __init__(self, latent_factor, num_users, num_items):
        super(SVD, self).__init__()
        self.P=nn.Embedding(num_users,latent_factor)
        self.Q=nn.Embedding(num_items,latent_factor)
        self.bu=nn.Embedding(num_users,1)
        self.bi=nn.Embedding(num_items,1)
    def forward(self, user_id, item_id):
        P=self.P(user_id)
        Q=self.Q(item_id)
        b_u=self.bu(user_id)
        b_i=self.bi(item_id)
        logit=(P*Q).sum(dim=1) + torch.squeeze(b_u) + torch.squeeze(b_i)
        return 5*F.sigmoid(logit) #scale to bounded interval [0,5]

latent_factor=25
model=MF(latent_factor,num_users,num_items)

num_epochs=1000
criterion=nn.MSELoss()
lambda_reg=0.01
optimizer=torch.optim.Adam(model.parameters(), lr=0.1)
for epoch in range(num_epochs):
    model.train()
    preds:torch.Tensor = model(user_ids,item_ids)
    loss:torch.Tensor = criterion(preds,inters)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def get_rating(model:MF,user_id:int,item_id:int,titles:pd.DataFrame):
    u=torch.tensor([user_id-1],dtype=torch.int64)
    i=torch.tensor([item_id-1],dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        pred:torch.Tensor=model(u,i)
    return titles.loc[item_id].values.item(),pred.item()

get_rating(model,5,2,titles)
```

For implicit feedback (Hu et al., 2008), we need to introduce several new terms, let $T$ is a matrix which its entries, $t_{ui}$ represent how much user $u$ consume item $i$
(e.g. the percentage of movie $i$ that user $u$ has watched), then interaction matrix entries, $r_{ui}$ is defined as

$$r_{ui} = \begin{cases}
            0 & t_{ui} = 0 \\
            1 & t_{ui} > 0
            \end{cases}
$$

However notice that user $u$ does not consume item $i$ not necessarily means user $u$ does not like item $i$, it can be just user $u$ does not know item $i$ exist, this also applies the other way around, user $u$ consume item $i$ not necessarily means user $u$ like item $i$, so for implicit feedback we need to introduce additional term called confidence matrix $C$ with its entries defined as

$$c_{ui} = 1+\alpha t_{ui}
$$

where $\alpha \in \mathbb{R}$ is rate of confidence increase. By introducing these new terms, the objective function for implicit feedback also changes a bit

$$argmin_{p_u,q_i,b_u,b_i}\sum_{(u,i) \in K}[c_{ui}(r_{ui}-\hat{r}_{ui})^2+\lambda(\left \lVert p_u \right \rVert _2^2 + \left \lVert q_i \right \rVert _2^2 + b_u^2 +b_i^2)],
$$

There are other algorithms for this approach, for example AutoRec (Sedhain et al., 2015) that use AutoEncoder Neural Network architecture to reconstruct item/user partially observed vector, and there is also Neural Collaborative Filtering with Bayesian Personalized Ranking Loss (Pairwise Learning) (He et al., 2017).


- **Hybrid**, this approach combines advantages of both Content-Based and Collaborative Filtering methods and allows them to obtain the best results. There are several types of this approach, one of them is Cascade Hybrid. Cascade Hybrid filters the output of one recommender using another recommender. For example, content-based filtering could first generate a broad list of recommendations, 
and collaborative filtering could then rank or refine those recommendations.

```python
def cascade_get_recommendation(ratings:pd.DataFrame,
                               titles:pd.DataFrame,
                               tfidf_matrix:pd.DataFrame,
                               user_id:int,
                               min_rating:int,
                               initial_n_recs:int,
                               model:MF,
                               n_recs:int):
    #content-based filtering for recommendation candidate
    recommendation_list=cb_get_recommendation(ratings,
                                           titles,
                                           tfidf_matrix,
                                           user_id,
                                           min_rating,
                                           n_recs=initial_n_recs,
                                           return_title=False)
    #refine recommendation with collaborative filtering
    predicted_ratings=[]
    for ids in recommendation_list:
        title,pred=get_rating(model,user_id,ids,titles)
        predicted_ratings.append((title,pred))
    predicted_ratings=sorted(predicted_ratings,reverse=True,key=lambda x: x[1]) #sort by rating
    final_recommendation=predicted_ratings[:n_recs]
    return final_recommendation
```

Thank you for reading this article :)


## Reference
- Ferreira D, Silva S, Abelha A, Machado J. Recommendation System Using Autoencoders. Applied Sciences. 2020; 10(16):5510. https://doi.org/10.3390/app10165510

- Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. 2015. AutoRec: Autoencoders Meet Collaborative Filtering. In Proceedings of the 24th International Conference on World Wide Web (WWW '15 Companion). Association for Computing Machinery, New York, NY, USA, 111–112. https://doi.org/10.1145/2740908.2742726

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).

- Y. Hu, Y. Koren and C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," 2008 Eighth IEEE International Conference on Data Mining, Pisa, Italy, 2008, pp. 263-272, doi: 10.1109/ICDM.2008.22.

- [Recommender Systems — A Complete Guide to Machine Learning Models](https://tinyurl.com/3vbp8akr)