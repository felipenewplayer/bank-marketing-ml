import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Carregamento do dataset
df = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\IA\Projetos\bank-marketing-ml\RegressaoLogistica\data\bank-full.csv',sep=';',quotechar='"')

# Analisar a estrutura e tratar os dados
print(df.head(5))
print(df.info())
print(df.describe())

# Separar colunas numéricas e categóricas

cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
num = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Processamento de dados
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
    ])

# Separar X e y

X = df.drop('y', axis=1)
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
# Montar pipeline de pré-processamento

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=1000))])
# Treina e avalia o modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')


# Salvar o modelo treinado

with open('RegressaoLogistica/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)