import pandas as pd
import matplotlib.pyplot as plt


#funções para encontrar e remover os outliers
def outliers(dataFrame, nome):
    q1 = dataFrame[nome].quantile(0.25)
    q3 = dataFrame[nome].quantile(0.75)
    iqr = q3 - q1

    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    #lista = ["pica"]
    lista = dataFrame.index[ (dataFrame[nome] < lb) | (dataFrame[nome] > ub ) ]

    return lista



def remover_outliers(dataFrame, lista):

    lista = sorted(set(lista))
    dataFrame = dataFrame.drop(lista)
    print("calma la espera")
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
dados_limpo = dados.dropna(subset=['Hematocrit','Hemoglobin', 'Platelets','Mean platelet volume ', 'Red blood Cells','Lymphocytes','Mean corpuscular hemoglobin concentration (MCHC)','Leukocytes', 'Basophils','Mean corpuscular hemoglobin (MCH)','Eosinophils','Mean corpuscular volume (MCV)', 'Monocytes','Red blood cell distribution width (RDW)','Serum Glucose'])




print("O dataset possui {} linhas e {} colunas".format(dados_limpo.shape[0], dados_limpo.shape[1]))

#filtrando mais um pouco removendo aqueles que possuem muitos outros dados faltando
dados_limpo.dropna(axis=1,thresh=55, inplace=True)

print(dados_limpo.isnull().sum())

#usando IRQ para remover outliers
 
print(dados_limpo.columns)
lista_index = []
for nome in dados_limpo.columns:
    print(nome)
    if dados_limpo[nome].dtypes == float:
        lista_index.extend(outliers(dados_limpo, nome))
        
    

print(lista_index)

df_final = remover_outliers(dados_limpo, lista_index)

print(df_final)


#alterando alguns valores para que o dataset fique inteiramente numérico
df_final = df_final.replace(['negative'], 0)
df_final = df_final.replace(['positive'], 1)
df_final = df_final.replace(['not_detected'],0)
df_final = df_final.replace(['detected'],1)



corr_final = df_final.corr(numeric_only=True)

#armazenando a correlação em um arquivo exel para algum tipo de análise visual 
#corr_final.to_excel(excel_writer='C:/Users/Lenovo/Desktop/testeExcel/arquivo.xlsx')
#remover a linha a cima caso queira testar com um diretório na propria maquina

#testando matshow para visualizar, porém sem muito sucesso. Seria necessário configurar melhor para uma melhor análise ou analisar atrávez do excel 
plt.matshow(corr_final)
plt.show()

