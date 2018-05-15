import pandas as pd
import PyQt4
import matplotlib
matplotlib.use('agg')

fields=["buying_price","popularity","maintainence_cost"]
df = pd.read_csv('cars.csv', skipinitialspace=True, usecols=fields)
print(type(df))


    
    
    


j=0
data={}
buy=[]
pop=[]
man=[]
for i in df.buying_price:
    buy.append(i)
for i in df.popularity:
    pop.append(i)
for i in df.maintainence_cost:
    man.append(i)

for i in range(len(pop)):
    try:
        data[(pop[i],buy[i],man[i])]+=1
    except:
        data[(pop[i],buy[i],man[i])]=1
print(data)

matplotlib.use('qt4agg')
import matplotlib.pyplot as plt

plt.hist( list(data.values()), normed=True, bins=6)
plt.show()