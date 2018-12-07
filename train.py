import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

pokemon_df = pd.read_csv('pokemon.csv')
combats_df = pd.read_csv('combats.csv')
test_df = pd.read_csv('tests.csv')
prediction_df = test_df.copy()
pokemon_df.head()

#cleaning
pokemon_df['Type 2'] = pokemon_df['Type 2'].fillna('None')
pokemon_df['Type 1'] = pokemon_df['Type 1'].replace('Fighting', 'Fight')
pokemon_df['Type 2'] = pokemon_df['Type 2'].replace('Fighting', 'Fight')
pokemon_df['Legendary'] = pokemon_df['Legendary'].map({False: 0, True:1})

#Number of legenary vs non-legendary
sns.countplot(x='Legendary', data=pokemon_df, order=pokemon_df['Legendary'].value_counts().index)
plt.xticks(rotation=90)
plt.show()

sns.countplot(x='Type 1', data=pokemon_df, order=pokemon_df['Type 1'].value_counts().index)
plt.xticks(rotation=90)
plt.show()

sns.countplot(x='Type 2', data=pokemon_df, order=pokemon_df['Type 2'].value_counts().index)
plt.xticks(rotation=90)
plt.show()

print(pokemon_df['Generation'].value_counts())

sns.countplot(x='Generation', data=pokemon_df, order=pokemon_df['Generation'].value_counts().index)
plt.show()

#calculate total stats, normally when attributes are higher, the pokemon are stronger 
pokemon_df['Total_stats'] = pokemon_df['HP'] + pokemon_df['Attack'] + pokemon_df['Defense'] + pokemon_df['Sp. Atk'] + pokemon_df['Sp. Def'] + pokemon_df['Speed']
print(pokemon_df.iloc[:, [1, -1]].head())

group_df = pokemon_df.drop(['#', 'Legendary'], axis=1)
pokemon_groups = group_df.groupby('Generation')
pokemon_groups_mean = pokemon_groups.mean()

sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['Total_stats'])
plt.show()

sns.distplot(pokemon_df['Total_stats'], hist=False, rug=True);

fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(15, 10))
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['Attack'], color='red', ax=axes[0][0])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['Defense'], color='blue', ax=axes[0][1])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['HP'], color='black', ax=axes[1][0])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['Speed'], color='green', ax=axes[1][1])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['Sp. Atk'], color='orange', ax=axes[2][0])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['Sp. Def'], color='purple', ax=axes[2][1])

plt.show()

name_dict = dict(zip(pokemon_df['#'], pokemon_df['Name']))
combats_name_df = combats_df[['First_pokemon', 'Second_pokemon', 'Winner']].replace(name_dict)
combats_name_df.head()
combats_df.head()

first_battle = combats_name_df['First_pokemon'].value_counts()
second_battle = combats_name_df['Second_pokemon'].value_counts()
win_counts = combats_name_df['Winner'].value_counts()
total_battle = first_battle + second_battle
win_percentage = win_counts / total_battle
win_percentage = win_percentage.sort_values()

win_percentage.tail()

type_df = pokemon_df.iloc[:, 0:4]
type_df = type_df.drop('Name', axis=1)
type_df.head()

stats_df = pokemon_df.drop(['Type 1', 'Type 2', 'Name', 'Generation', 'Total_stats'], axis=1)
stats_df.head()

type_dict = type_df.set_index('#').T.to_dict('list')
print(type_dict)

stats_dict = stats_df.set_index('#').T.to_dict('list')

combats_df.Winner[combats_df.Winner == combats_df.First_pokemon] = 0
combats_df.Winner[combats_df.Winner == combats_df.Second_pokemon] = 1

def replace_things(data):
    
    data['First_pokemon_stats'] = data.First_pokemon.map(stats_dict)
    data['Second_pokemon_stats'] = data.Second_pokemon.map(stats_dict)
    data['First_pokemon'] = data.First_pokemon.map(type_dict)
    data['Second_pokemon'] = data.Second_pokemon.map(type_dict)

    return data

train_df = replace_things(combats_df)
print(train_df.head(5))

def calculate_stats(data):
    
    stats_col = ['HP_diff', 'Attack_diff', 'Defense_diff', 'Sp.Atk_diff', 'Sp.Def_diff', 'Speed_diff', 'Legendary_diff']
    diff_list = []

    for row in data.itertuples():
        diff_list.append(np.array(row.First_pokemon_stats) - np.array(row.Second_pokemon_stats))

    stats_df = pd.DataFrame(diff_list, columns=stats_col)
    data = pd.concat([data, stats_df], axis=1)
    data.drop(['First_pokemon_stats', 'Second_pokemon_stats'], axis=1, inplace=True)

    return data

train_df = calculate_stats(train_df)
print(train_df.head(5))

#I used the predesigned groups online

def calculate_effectiveness(data):

    '''
        this function creates a new column of each pokemon's effectiveness against it's enemy.
        every effectiveness starts with 1, if an effective type is found on enemy's type, effectiveness * 2
        if not very effective is found on enemy's type, effectiveness / 2
        if not effective is found on enemy's type, effectiveness * 0
        
        This function creates 4 new columns
            1. P1_type1, pokemon 1 first type effectiveness against the enemy's type
            2. P1_type2, pokemon 1 second type effectiveness against the enemy's type
            3. P2_type1, pokemon 2 first type effectiveness against the enemy's type
            4. P2_type2, pokemon 2 second type effectiveness against the enemy's type
    '''
    
    very_effective_dict = {'Normal': [],
                           'Fight': ['Normal', 'Rock', 'Steel', 'Ice', 'Dark'],
                           'Flying': ['Fight', 'Bug', 'Grass'],
                           'Poison': ['Grass', 'Fairy'],
                           'Ground': ['Poison', 'Rock', 'Steel', 'Fire', 'Electric'],
                           'Rock': ['Flying', 'Bug', 'Fire', 'Ice'],
                           'Bug': ['Grass', 'Psychic', 'Dark'],
                           'Ghost': ['Ghost', 'Psychic'],
                           'Steel': ['Rock', 'Ice', 'Fairy'],
                           'Fire': ['Bug', 'Steel', 'Grass', 'Ice'],
                           'Water': ['Ground', 'Rock', 'Fire'],
                           'Grass': ['Ground', 'Rock', 'Water'],
                           'Electric': ['Flying', 'Water'],
                           'Psychic': ['Fight', 'Poison'],
                           'Ice': ['Flying', 'Ground', 'Grass', 'Dragon'],
                           'Dragon': ['Dragon'],
                           'Dark': ['Ghost', 'Psychic'],
                           'Fairy': ['Fight', 'Dragon', 'Dark'],
                           'None': []}

    not_very_effective_dict = {'Normal': ['Rock', 'Steel'],
                               'Fight': ['Flying', 'Poison', 'Bug', 'Psychic', 'Fairy'],
                               'Flying': ['Rock', 'Steel', 'Electric'],
                               'Poison': ['Poison', 'Rock', 'Ground', 'Ghost'],
                               'Ground': ['Bug', 'Grass'],
                               'Rock': ['Fight', 'Ground', 'Steel'],
                               'Bug': ['Fight', 'Flying', 'Poison', 'Ghost', 'Steel', 'Fire', 'Fairy'],
                               'Ghost': ['Dark'],
                               'Steel': ['Steel', 'Fire', 'Water', 'Electric'],
                               'Fire': ['Rock', 'Fire', 'Water', 'Dragon'],
                               'Water': ['Water', 'Grass', 'Dragon'],
                               'Grass': ['Flying', 'Poison', 'Bug', 'Steel', 'Fire', 'Grass', 'Dragon'],
                               'Electric': ['Grass', 'Electric', 'Dragon'],
                               'Psychic': ['Steel', 'Psychic'],
                               'Ice': ['Steel', 'Fire', 'Water', 'Psychic'],
                               'Dragon': ['Steel'],
                               'Dark': ['Fight', 'Dark', 'Fairy'],
                               'Fairy': ['Posion', 'Steel', 'Fire'],
                               'None': []}

    not_effective_dict = {'Normal': ['Ghost'],
                          'Fight': ['Ghost'],
                          'Flying': [],
                          'Poison': ['Steel'],
                          'Ground': ['Flying'],
                          'Rock': [],
                          'Bug': [],
                          'Ghost': ['Normal'],
                          'Steel': [],
                          'Fire': [],
                          'Water': [],
                          'Grass': [],
                          'Electric': ['Ground'],
                          'Psychic': ['Dark'],
                          'Ice': [],
                          'Dragon': ['Fairy'],
                          'Dark': [],
                          'Fairy': [],
                          'None': []}

    p1_type1_list = []
    p1_type2_list = []
    p2_type1_list = []
    p2_type2_list = []

    for row in data.itertuples():
        nested_type = [[1, 1], [1, 1]]

        #manipulating values if found on dictionary
        for i in range(0,2):
            for j in range(0,2):
                if row.Second_pokemon[j] in very_effective_dict.get(row.First_pokemon[i]):
                    nested_type[0][i] *= 2
                if row.Second_pokemon[j] in not_very_effective_dict.get(row.First_pokemon[i]):
                    nested_type[0][i] /= 2
                if row.Second_pokemon[j] in not_effective_dict.get(row.First_pokemon[i]):
                    nested_type[0][i] *= 0

                if row.First_pokemon[j] in very_effective_dict.get(row.Second_pokemon[i]):
                    nested_type[1][i] *= 2
                if row.First_pokemon[j] in not_very_effective_dict.get(row.Second_pokemon[i]):
                    nested_type[1][i] /= 2
                if row.First_pokemon[j] in not_effective_dict.get(row.Second_pokemon[i]):
                    nested_type[1][i] *= 0

        p1_type1_list.append(nested_type[0][0])
        p1_type2_list.append(nested_type[0][1])
        p2_type1_list.append(nested_type[1][0])
        p2_type2_list.append(nested_type[1][1])

    data = data.assign(P1_type1=p1_type1_list, P1_type2=p1_type2_list, P2_type1=p2_type1_list, P2_type2=p2_type2_list)
    data = data.drop(['First_pokemon', 'Second_pokemon'], axis=1)

    return data

train_df = calculate_effectiveness(train_df)
train_df.head()

y_train_full = train_df['Winner']
x_train_full = train_df.drop('Winner', axis=1)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(x_train_full, y_train_full, train_size=0.8, test_size=0.2, random_state=100)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
clf = KNeighborsClassifier()
clf.fit(train_X, train_y) 
knn_train_y = clf.predict(train_X)
knn_test_y = clf.predict(test_X)
knn_train_y_acc = metrics.accuracy_score(train_y, knn_train_y)
knn_test_y_acc = metrics.accuracy_score(test_y, knn_test_y)
print('knn')
print(knn_train_y_acc)
print (knn_test_y_acc)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_X, train_y)
lr_train_y = clf.predict(train_X)
lr_test_y = clf.predict(test_X)
lr_train_y_acc = metrics.accuracy_score(train_y, lr_train_y)
lr_test_y_acc = metrics.accuracy_score(test_y, lr_test_y)
print('LogisticRegression')
print(lr_train_y_acc)
print (lr_test_y_acc)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(train_X, train_y)
gnb_train_y = clf.predict(train_X)
gnb_test_y = clf.predict(test_X)
gnb_train_y_acc = metrics.accuracy_score(train_y, lr_train_y)
gnb_test_y_acc = metrics.accuracy_score(test_y, lr_test_y)
print('GaussianNB')
print(gnb_train_y_acc)
print (gnb_test_y_acc)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_X, train_y) 
rfc_train_y = clf.predict(train_X)
rfc_test_y = clf.predict(test_X)
rfc_train_y_acc = metrics.accuracy_score(train_y, rfc_train_y)
rfc_test_y_acc = metrics.accuracy_score(test_y, rfc_test_y)
print('RandomForestClassifier')
print(rfc_train_y_acc)
print (rfc_test_y_acc)

!pip install torch torchvision
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
#Define the model 
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

batch_size = 50
num_epochs = 200
learning_rate = 0.01
size_hidden= 100

#Calculate some other hyperparameters based on data.  
batch_no = len(train_X) // batch_size  #batches
cols=train_X.shape[1] #Number of columns in input matrix
n_output=1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print("Executing the model on :",device)
class Net(torch.nn.Module):
    def __init__(self, n_feature, size_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(cols, size_hidden)   # hidden layer
        self.predict = torch.nn.Linear(size_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
net = Net(cols, size_hidden, n_output)
#Adam is a specific flavor of gradient decent which is typically better
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
criterion = torch.nn.MSELoss(size_average=False)  # this is for regression mean squared loss

train_X=train_X.values
train_y=train_y.values
test_X=test_X.values
test_y=test_y.values

from sklearn.utils import shuffle
from torch.autograd import Variable
running_loss = 0.0
for epoch in range(num_epochs):
    #Shuffle just mixes up the dataset between epocs
    train_X, train_y = shuffle(train_X, train_y)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        inputs = Variable(torch.FloatTensor(train_X[start:end]))
        labels = Variable(torch.FloatTensor(train_y[start:end]))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print("outputs",outputs)
        #print("outputs",outputs,outputs.shape,"labels",labels, labels.shape)
        loss = criterion(outputs, torch.unsqueeze(labels,dim=1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print('Epoch {}'.format(epoch+1), "loss: ",running_loss)
    running_loss = 0.0

    from sklearn.metrics import r2_score

X = Variable(torch.FloatTensor(train_X)) 
result = net(X)
pred=result.data[:,0].numpy()
print(len(pred),len(train_y))
r2_score(pred,train_y)

def calculate_r2(x,y=[]):
    """
    This function will return the r2 if passed x and y or return predictions if just passed x. 
    """
    # Evaluate the model with the test set. 
    X = Variable(torch.FloatTensor(x))  
    result = net(X) #This outputs the value for regression
    result=result.data[:,0].numpy()
  
    if len(y) != 0:
        r2=r2_score(result, y)
        print("R-Squared", r2)
        #print('Accuracy {:.2f}'.format(num_right / len(y)), "for a total of ", len(y), "records")
        return pd.DataFrame(data= {'actual': y, 'predicted': result})
    else:
        print("returning predictions")
        return result

result1=calculate_r2(train_X,train_y)
result2=calculate_r2(test_X,test_y)        

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit( train_X, train_y )

print('R2 for Train)', lm.score( train_X, train_y ))
print('R2 for Test (cross validation)', lm.score(test_X, test_y))


