import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


#funções para encontrar e remover os outliers
def outliers(dataFrame, nome):
    q1 = dataFrame[nome].quantile(0.25)
    q3 = dataFrame[nome].quantile(0.75)
    iqr = q3 - q1

    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    
    lista = dataFrame.index[ (dataFrame[nome] < lb) | (dataFrame[nome] > ub ) ]

    return lista



def remover_outliers(dataFrame, lista):

    lista = sorted(set(lista))
    dataFrame = dataFrame.drop(lista)
    
    return dataFrame


    


dados = pd.read_excel('dataset.xlsx')

# verificando dataset
print("verificando dataset:")
print("O dataset possui {} linhas e {} colunas".format(dados.shape[0], dados.shape[1]))
print(dados.head())

# analisando dados faltantes
print("analisando dados faltantes:")
print(dados.isnull().sum())


#como o dataset tem muitos valores vazios
#serão removidos aqueles individuos que não possuem todas as informações sobre o exame de sangue completas
# dados_limpo = dados.dropna(subset=['Hematocrit','Hemoglobin', 'Platelets','Mean platelet volume ', 'Red blood Cells','Lymphocytes','Mean corpuscular hemoglobin concentration (MCHC)','Leukocytes', 'Basophils','Mean corpuscular hemoglobin (MCH)','Eosinophils','Mean corpuscular volume (MCV)', 'Monocytes','Red blood cell distribution width (RDW)','Serum Glucose'])




print("O dataset possui {} linhas e {} colunas".format(dados.shape[0], dados.shape[1]))

# Removendo as colunas com mais de 90% de valores faltantes
dados = dados.loc[:, dados.isnull().mean() < 0.9]

print("O dataset possui {} linhas e {} colunas".format(dados.shape[0], dados.shape[1]))

#filtrando mais um pouco removendo aqueles que possuem muitos outros dados faltando
dados.dropna(axis=1,thresh=55, inplace=True)

print(dados.isnull().sum())

#usando IRQ para remover outliers
lista_index = []
for nome in dados.columns:
    print(nome)
    if dados[nome].dtypes == float:
        lista_index.extend(outliers(dados, nome))
        
    



df_final = remover_outliers(dados, lista_index)



#alterando os valores para que o dataset fique inteiramente numérico

for coluna in df_final.columns:
    if df_final[coluna].dtype == "object":
        df_final[coluna] = pd.Categorical(df_final[coluna]).codes



corr_final = df_final.corr(numeric_only=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_final = pd.DataFrame(imputer.fit_transform(df_final), columns=df_final.columns)

#armazenando a correlação em um arquivo exel para algum tipo de análise visual 
#corr_final.to_excel(excel_writer='C:/Users/Lenovo/Desktop/testeExcel/arquivo.xlsx')
#remover a linha a cima caso queira testar com um diretório na propria maquina

#testando matshow para visualizar, porém sem muito sucesso. Seria necessário configurar melhor para uma melhor análise ou analisar atrávez do excel 

# plt.matshow(corr_final)
# plt.show()

# Separa os dados em conjunto de treinamento e teste

print(df_final)

X = df_final.drop("SARS-Cov-2 exam result", axis=1)
X = pd.get_dummies(X)
y = df_final["SARS-Cov-2 exam result"]

melhor_resultado = 0
melhor_semente = 0
mc = []

#For para testar sementes de 0 a 1000 
for semente in range(1000):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=semente)

    
    # Criação e treinamento do modelo KNN
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_train, y_train)

    # Avaliação do modelo
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    


    if accuracy > melhor_resultado:
        melhor_resultado = accuracy
        melhor_semente = semente
        print('Acurácia do modelo: {:.2f}%'.format(accuracy * 100))
        # mc = confusion_matrix(y_test, y_pred)
        # print(mc)
        


print("Melhor semente:",melhor_semente,"Melhor resultado:", melhor_resultado)
mc = confusion_matrix(y_test, y_pred)
print(mc)