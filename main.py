import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#importando dados
#fonte: https://www.kaggle.com/c/titanic/data
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

#verificar as dimensões dos dataframes
train.shape
test.shape

train.head()

#verificando a coluna sexo
sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()

#verificando a coluna Pclass
class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar(color='r') # r para indicar a cor vermelha(red)
plt.show()

#verificando a distribuição de idades no treino
train["Age"].describe()

#criando um histograma para visualizar como foi o grau de sobrevivência de acordo com as idades
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()

#para facilitar o trabalhodo algoritmo, vamos criar ranges fixos de idades.
# e ao mesmo tempo vamos tratar os missing values
def process_age(df,cut_points,label_names):
df["Age"] = df["Age"].fillna(-0.5)
df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
return df
cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

#índice de sobrevivência entre as idades
pivot = train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar(color='g')
plt.show()

#Preparando os dados para o modelo de Machine Learning

#removendo a relação numerica presente na coluna P class
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass","Sex","Age_categories"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)

#Criando o modelo

#criando um objeto LogistcRegression
lr = LogisticRegression()

#treinando o modelo
columns = ['Pclass_2', 'Pclass_3', 'Sex_male']
lr.fit(train[columns], train['Survived'])

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
           'Age_categories_Missing','Age_categories_Infant',
           'Age_categories_Child', 'Age_categories_Teenager',
           'Age_categories_Young Adult', 'Age_categories_Adult',
           'Age_categories_Senior']

lr = LogisticRegression()
lr.fit(train[columns], train["Survived"])

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                   penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                   verbose=0, warm_start=False)
#avaliando o modelo
holdout = test
all_X = train[columns]
all_y = train['Survived']
train_X, test_X, train_y, test_y = train_test_split(
all_X, all_y, test_size=0.20,random_state=0)

#matriz de previsões.
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

#verificando a acuracia

accuracy = accuracy_score(test_y, predictions)
print(accuracy)

cross_val_score(estimator, X, y, cv = None)

#usando cross validation para um medida de erro mais precisa
lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
scores.sort()
accuracy = scores.mean()
print(scores)
print(accuracy)

#fazendo previsões usando novos dados
lr = LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])

#Construindo outro modelo

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

#retirando os dados irrelevantes
train.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)
test.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)

#Preparando os dados para o modelo de Machine Learning

#fazendo uso dos dummies de novo
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

new_data_train.isnull().sum().sort_values(ascending = False).head(10)

#tratando valores nulos encontrados
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace = True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace = True)

new_data_test.isnull().sum().sort_values(ascending = False).head(10)

new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace = True)

#separado as features para a criação do modelo
X = new_data_train.drop("Survived", axis = 1) #tirando apenas a coluna target
y = new_data_train["Survived"] # colocando somente a coluna target

tree = DecisionTreeClassifier(max_depth = 3, random_state = 0)
tree.fit(X,y)

#avaliando o modelo
tree.score(X,y)

#Enviando a previsão para o Kaggle
previsao = pd.DataFrame()
previsao["PassengerId"] = new_data_test["PassengerId"]
previsao["Survived"] = tree.predict(new_data_test)

previsao.to_csv('previsao.csv',index = False)