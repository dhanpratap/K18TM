
# Building a song recommender
-------------
Dataset used:
-------------
Million Songs Dataset
Source: http://labrosa.ee.columbia.edu/millionsong/
Paper: http://ismir2011.ismir.net/papers/OS6-1.pdf

The current notebook uses a subset of the above data containing 10,000 songs obtained from:
https://github.com/turi-code/tutorials/blob/master/notebooks/recsys_rank_10K_song.ipynb

```python
%matplotlib inline

import pandas
from sklearn.cross_validation import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation
```

# Load music data


```python
#Read userid-songid-listen_count triplets
#This step might take time to download data from external sources
triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
song_df_2 =  pandas.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left") 
```

# Explore data

Music data shows how many times a user listened to a song, as well as the details of the song.


```python
song_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>song_id</th>
      <th>listen_count</th>
      <th>title</th>
      <th>release</th>
      <th>artist_name</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOAKIMP12A8C130995</td>
      <td>1</td>
      <td>The Cove</td>
      <td>Thicker Than Water</td>
      <td>Jack Johnson</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOBBMDR12A8C13253B</td>
      <td>2</td>
      <td>Entre Dos Aguas</td>
      <td>Flamenco Para Niños</td>
      <td>Paco De Lucia</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOBXHDL12A81C204C0</td>
      <td>1</td>
      <td>Stronger</td>
      <td>Graduation</td>
      <td>Kanye West</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOBYHAJ12A6701BF1D</td>
      <td>1</td>
      <td>Constellations</td>
      <td>In Between Dreams</td>
      <td>Jack Johnson</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SODACBL12A8C13C273</td>
      <td>1</td>
      <td>Learn To Fly</td>
      <td>There Is Nothing Left To Lose</td>
      <td>Foo Fighters</td>
      <td>1999</td>
    </tr>
  </tbody>
</table>
</div>



## Length of the dataset


```python
len(song_df)
```




    2000000



## Create a subset of the dataset


```python
song_df = song_df.head(10000)

#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
```

## Showing the most popular songs in the dataset


```python
song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song</th>
      <th>listen_count</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3660</th>
      <td>Sehr kosmisch - Harmonia</td>
      <td>45</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>4678</th>
      <td>Undo - Björk</td>
      <td>32</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>5105</th>
      <td>You're The One - Dwight Yoakam</td>
      <td>32</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>Dog Days Are Over (Radio Edit) - Florence + Th...</td>
      <td>28</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>3655</th>
      <td>Secrets - OneRepublic</td>
      <td>28</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>4378</th>
      <td>The Scientist - Coldplay</td>
      <td>27</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>4712</th>
      <td>Use Somebody - Kings Of Leon</td>
      <td>27</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>3476</th>
      <td>Revelry - Kings Of Leon</td>
      <td>26</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>1387</th>
      <td>Fireflies - Charttraxx Karaoke</td>
      <td>24</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>1862</th>
      <td>Horn Concerto No. 4 in E flat K495: II. Romanc...</td>
      <td>23</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>1805</th>
      <td>Hey_ Soul Sister - Train</td>
      <td>22</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>5032</th>
      <td>Yellow - Coldplay</td>
      <td>22</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>808</th>
      <td>Clocks - Coldplay</td>
      <td>21</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>2620</th>
      <td>Lucky (Album Version) - Jason Mraz &amp; Colbie Ca...</td>
      <td>20</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>2299</th>
      <td>Just Dance - Lady GaGa / Colby O'Donis</td>
      <td>19</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>456</th>
      <td>Billionaire [feat. Bruno Mars]  (Explicit Albu...</td>
      <td>18</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>2689</th>
      <td>Marry Me - Train</td>
      <td>18</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>3064</th>
      <td>OMG - Usher featuring will.i.am</td>
      <td>18</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>4543</th>
      <td>Tive Sim - Cartola</td>
      <td>18</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>142</th>
      <td>Alejandro - Lady GaGa</td>
      <td>17</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>726</th>
      <td>Catch You Baby (Steve Pitron &amp; Max Sanna Radio...</td>
      <td>17</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>1410</th>
      <td>Float On - Modest Mouse</td>
      <td>17</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>3868</th>
      <td>Somebody To Love - Justin Bieber</td>
      <td>17</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>631</th>
      <td>Bulletproof - La Roux</td>
      <td>16</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>1143</th>
      <td>Drop The World - Lil Wayne / Eminem</td>
      <td>16</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>3038</th>
      <td>Nothin' On You [feat. Bruno Mars] (Album Versi...</td>
      <td>16</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>4465</th>
      <td>They Might Follow You - Tiny Vipers</td>
      <td>16</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>870</th>
      <td>Cosmic Love - Florence + The Machine</td>
      <td>15</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>899</th>
      <td>Creep (Explicit) - Radiohead</td>
      <td>15</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>Halo - Beyoncé</td>
      <td>15</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5094</th>
      <td>You Yourself are Too Serious - The Mercury Pro...</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5098</th>
      <td>You'll Never Know (My Love) (Bovellian 07 Mix)...</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5100</th>
      <td>You're A Wolf (Album) - Sea Wolf</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5102</th>
      <td>You're Gonna Miss Me When I'm Gone - Brooks &amp; ...</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5103</th>
      <td>You're Not Alone - ATB</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5104</th>
      <td>You're Not Alone - Olive</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5108</th>
      <td>You've Passed - Neutral Milk Hotel</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5109</th>
      <td>Young - Hollywood Undead</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5111</th>
      <td>Younger Than Springtime - William Tabbert</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5112</th>
      <td>Your Arms Feel Like home - 3 Doors Down</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5113</th>
      <td>Your Every Idol - Telefon Tel Aviv</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5114</th>
      <td>Your Ex-Lover Is Dead (Album Version) - Stars</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5115</th>
      <td>Your Guardian Angel - The Red Jumpsuit Apparatus</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5117</th>
      <td>Your House - Jimmy Eat World</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5118</th>
      <td>Your Love - The Outfield</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5121</th>
      <td>Your Mouth - Telefon Tel Aviv</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5123</th>
      <td>Your Song (Alternate Take 10) - Cilla Black</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5126</th>
      <td>Your Visits Are Getting Shorter - Bloc Party</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5127</th>
      <td>Your Woman - White Town</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5130</th>
      <td>Ze Rook Naar Rozen - Rob De Nijs</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5131</th>
      <td>Zebra - Beach House</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5132</th>
      <td>Zebra - Man Man</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5133</th>
      <td>Zero - The Pain Machinery</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5135</th>
      <td>Zopf: Pigtail - Penguin Café Orchestra</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5137</th>
      <td>aNYway - Armand Van Helden &amp; A-TRAK Present Du...</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5139</th>
      <td>high fives - Four Tet</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5140</th>
      <td>in white rooms - Booka Shade</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5143</th>
      <td>paranoid android - Christopher O'Riley</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5149</th>
      <td>¿Lo Ves? [Piano Y Voz] - Alejandro Sanz</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5150</th>
      <td>Época - Gotan Project</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
<p>5151 rows × 3 columns</p>
</div>



## Count number of unique users in the dataset


```python
users = song_df['user_id'].unique()
```


```python
len(users)
```




    365



## Quiz 1. Count the number of unique songs in the dataset


```python
###Fill in the code here
songs = song_df['song'].unique()
len(songs)
```




    5151



# Create a song recommender


```python
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
print(train_data.head(5))
```

                                           user_id             song_id  \
    7389  94d5bdc37683950e90c56c9b32721edb5d347600  SOXNZOW12AB017F756   
    9275  1012ecfd277b96487ed8357d02fa8326b13696a5  SOXHYVQ12AB0187949   
    2995  15415fa2745b344bce958967c346f2a89f792f63  SOOSZAZ12A6D4FADF8   
    5316  ffadf9297a99945c0513cd87939d91d8b602936b  SOWDJEJ12A8C1339FE   
    356   5a905f000fc1ff3df7ca807d57edb608863db05d  SOAMPRJ12A8AE45F38   
    
          listen_count                 title  \
    7389             2      Half Of My Heart   
    9275             1  The Beautiful People   
    2995             1     Sanctify Yourself   
    5316             4     Heart Cooks Brain   
    356             20                 Rorol   
    
                                                    release      artist_name  \
    7389                                     Battle Studies       John Mayer   
    9275             Antichrist Superstar (Ecopac Explicit)   Marilyn Manson   
    2995                             Glittering Prize 81/92     Simple Minds   
    5316  Everything Is Nice: The Matador Records 10th A...     Modest Mouse   
    356                               Identification Parade  Octopus Project   
    
          year                                   song  
    7389     0          Half Of My Heart - John Mayer  
    9275     0  The Beautiful People - Marilyn Manson  
    2995  1985       Sanctify Yourself - Simple Minds  
    5316  1997       Heart Cooks Brain - Modest Mouse  
    356   2002                Rorol - Octopus Project  


## Simple popularity-based recommender class (Can be used as a black box)


```python
#Recommenders.popularity_recommender_py
```

### Create an instance of popularity based recommender class


```python
pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')
```

### Use the popularity model to make some predictions


```python
user_id = users[5]
pm.recommend(user_id)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>song</th>
      <th>score</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3194</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Sehr kosmisch - Harmonia</td>
      <td>37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4083</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Undo - Björk</td>
      <td>27</td>
      <td>2</td>
    </tr>
    <tr>
      <th>931</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Dog Days Are Over (Radio Edit) - Florence + Th...</td>
      <td>24</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4443</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>You're The One - Dwight Yoakam</td>
      <td>24</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3034</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Revelry - Kings Of Leon</td>
      <td>21</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3189</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Secrets - OneRepublic</td>
      <td>21</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4112</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Use Somebody - Kings Of Leon</td>
      <td>21</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Fireflies - Charttraxx Karaoke</td>
      <td>20</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1577</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Hey_ Soul Sister - Train</td>
      <td>19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1626</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Horn Concerto No. 4 in E flat K495: II. Romanc...</td>
      <td>19</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### Quiz 2: Use the popularity based model to make predictions for the following user id (Note the difference in recommendations from the first user id).


```python
###Fill in the code here
user_id = users[8]
pm.recommend(user_id)

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>song</th>
      <th>score</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3194</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Sehr kosmisch - Harmonia</td>
      <td>37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4083</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Undo - Björk</td>
      <td>27</td>
      <td>2</td>
    </tr>
    <tr>
      <th>931</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Dog Days Are Over (Radio Edit) - Florence + Th...</td>
      <td>24</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4443</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>You're The One - Dwight Yoakam</td>
      <td>24</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3034</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Revelry - Kings Of Leon</td>
      <td>21</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3189</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Secrets - OneRepublic</td>
      <td>21</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4112</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Use Somebody - Kings Of Leon</td>
      <td>21</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Fireflies - Charttraxx Karaoke</td>
      <td>20</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1577</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Hey_ Soul Sister - Train</td>
      <td>19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1626</th>
      <td>9bb911319fbc04f01755814cb5edb21df3d1a336</td>
      <td>Horn Concerto No. 4 in E flat K495: II. Romanc...</td>
      <td>19</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



## Build a song recommender with personalization

We now create an item similarity based collaborative filtering model that allows us to make personalized recommendations to each user. 

## Class for an item similarity based personalized recommender system (Can be used as a black box)


```python
#Recommenders.item_similarity_recommender_py
```

### Create an instance of item similarity based recommender class


```python
is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')
```

### Use the personalized model to make some song recommendations


```python
#Print the songs for the user in training data
user_id = users[5]
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)
```

    ------------------------------------------------------------------------------------
    Training data songs for the user userid: 4bd88bfb25263a75bbdd467e74018f4ae570e5df:
    ------------------------------------------------------------------------------------
    Just Lose It - Eminem
    Without Me - Eminem
    16 Candles - The Crests
    Speechless - Lady GaGa
    Push It - Salt-N-Pepa
    Ghosts 'n' Stuff (Original Instrumental Mix) - Deadmau5
    Say My Name - Destiny's Child
    My Dad's Gone Crazy - Eminem / Hailie Jade
    The Real Slim Shady - Eminem
    Somebody To Love - Justin Bieber
    Forgive Me - Leona Lewis
    Missing You - John Waite
    Ya Nada Queda - Kudai
    ----------------------------------------------------------------------
    Recommendation process going on:
    ----------------------------------------------------------------------
    No. of unique songs for the user: 13
    no. of unique songs in the training set: 4483
    Non zero values in cooccurence_matrix :2097





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>song</th>
      <th>score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Superman - Eminem / Dina Rae</td>
      <td>0.088692</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Mockingbird - Eminem</td>
      <td>0.067663</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>I'm Back - Eminem</td>
      <td>0.065385</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>U Smile - Justin Bieber</td>
      <td>0.064525</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Here Without You - 3 Doors Down</td>
      <td>0.062293</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Hellbound - J-Black &amp; Masta Ace</td>
      <td>0.055769</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>The Seed (2.0) - The Roots / Cody Chestnutt</td>
      <td>0.052564</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>I'm The One Who Understands (Edit Version) - War</td>
      <td>0.052564</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Falling - Iration</td>
      <td>0.052564</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4bd88bfb25263a75bbdd467e74018f4ae570e5df</td>
      <td>Armed And Ready (2009 Digital Remaster) - The ...</td>
      <td>0.052564</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### Quiz 3. Use the personalized model to make recommendations for the following user id. (Note the difference in recommendations from the first user id.)


```python
user_id = users[7]
#Fill in the code here
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)

```

    ------------------------------------------------------------------------------------
    Training data songs for the user userid: 9d6f0ead607ac2a6c2460e4d14fb439a146b7dec:
    ------------------------------------------------------------------------------------
    Swallowed In The Sea - Coldplay
    Life In Technicolor ii - Coldplay
    Life In Technicolor - Coldplay
    The Scientist - Coldplay
    Trouble - Coldplay
    Strawberry Swing - Coldplay
    Lost! - Coldplay
    Clocks - Coldplay
    ----------------------------------------------------------------------
    Recommendation process going on:
    ----------------------------------------------------------------------
    No. of unique songs for the user: 8
    no. of unique songs in the training set: 4483
    Non zero values in cooccurence_matrix :3429





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>song</th>
      <th>score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>She Just Likes To Fight - Four Tet</td>
      <td>0.281579</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>Warning Sign - Coldplay</td>
      <td>0.281579</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>We Never Change - Coldplay</td>
      <td>0.281579</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>Puppetmad - Puppetmastaz</td>
      <td>0.281579</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>God Put A Smile Upon Your Face - Coldplay</td>
      <td>0.281579</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>Susie Q - Creedence Clearwater Revival</td>
      <td>0.281579</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>The Joker - Fatboy Slim</td>
      <td>0.281579</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>Korg Rhythm Afro - Holy Fuck</td>
      <td>0.281579</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>This Unfolds - Four Tet</td>
      <td>0.281579</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9d6f0ead607ac2a6c2460e4d14fb439a146b7dec</td>
      <td>high fives - Four Tet</td>
      <td>0.281579</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### We can also apply the model to find similar songs to any song in the dataset


```python
is_model.get_similar_items(['U Smile - Justin Bieber'])
```

    no. of unique songs in the training set: 4483
    Non zero values in cooccurence_matrix :271





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>song</th>
      <th>score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>Somebody To Love - Justin Bieber</td>
      <td>0.428571</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>Bad Company - Five Finger Death Punch</td>
      <td>0.375000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Love Me - Justin Bieber</td>
      <td>0.333333</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>One Time - Justin Bieber</td>
      <td>0.333333</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Here Without You - 3 Doors Down</td>
      <td>0.333333</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>Stuck In The Moment - Justin Bieber</td>
      <td>0.333333</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Teach Me How To Dougie - California Swag District</td>
      <td>0.333333</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>Paper Planes - M.I.A.</td>
      <td>0.333333</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>Already Gone - Kelly Clarkson</td>
      <td>0.333333</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>The Funeral (Album Version) - Band Of Horses</td>
      <td>0.300000</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### Quiz 4. Use the personalized recommender model to get similar songs for the following song.


```python
song = 'Yellow - Coldplay'
###Fill in the code here
is_model.get_similar_items([song])
```

    no. of unique songs in the training set: 4483
    Non zero values in cooccurence_matrix :969





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>song</th>
      <th>score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>Fix You - Coldplay</td>
      <td>0.375000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>Creep (Explicit) - Radiohead</td>
      <td>0.291667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Clocks - Coldplay</td>
      <td>0.280000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>Seven Nation Army - The White Stripes</td>
      <td>0.250000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Paper Planes - M.I.A.</td>
      <td>0.208333</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>Halo - Beyoncé</td>
      <td>0.200000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>The Funeral (Album Version) - Band Of Horses</td>
      <td>0.181818</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>In My Place - Coldplay</td>
      <td>0.181818</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>Kryptonite - 3 Doors Down</td>
      <td>0.166667</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>When You Were Young - The Killers</td>
      <td>0.166667</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>


