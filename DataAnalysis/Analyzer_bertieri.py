#%% import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from datetime import datetime
import matplotlib.lines as mlines

plt.close('all')
bertieri_df = pd.read_csv("Datasets/BERTIERI/Bertieri.csv",sep=";")
print(bertieri_df.columns)

#%% filtering

df = bertieri_df[["clu_name","clu_datastipula","clu_datainiziocontratto","clu_datadisdetta","clu_datascadenza","clu_canonemensile","clu_modellocontratto","st_contratto","tipo_stanza","id_postoletto","clu_tipologialetto","clu_balcone","tipo_letto",'clu_metriquadraticommerciali',"clu_ariacondizionata","clu_inquilinoid","statopostoletto","nome_postoletto"]]


#df = df.drop(df[df.st_contratto=="Attivo"].index)

print(df.shape[0])

df_to_print =df[df.duplicated(keep=False)]
print(df_to_print.head)
df = df.drop_duplicates()

print(df.shape[0])

#%%
print(df.shape[0])
df = df.dropna( subset=["clu_canonemensile","clu_datainiziocontratto","clu_datastipula"])
print(df.shape[0])




print(df.shape[0])

df["durata_inizio_scadenza"] = pd.to_datetime(df["clu_datascadenza"],dayfirst=True)-pd.to_datetime(df["clu_datainiziocontratto"],dayfirst=True)
df["durata_inizio_disdetta"] = pd.to_datetime(df["clu_datadisdetta"],dayfirst=True)-pd.to_datetime(df["clu_datainiziocontratto"],dayfirst=True)
df["durata_affitto"] =df[["durata_inizio_scadenza","durata_inizio_disdetta"]].min(axis=1)

df = df[df["durata_affitto"]>=pd.Timedelta(days=0, hours=0)]
print(df.shape[0])


print("numero totale di contratti:"+str(df.clu_name.count()))

df = df.drop(df[df.st_contratto=="Attivo"].index)
df["d_s"] = pd.to_datetime(df["clu_datastipula"],dayfirst=True)
df.sort_values(by=["d_s"],inplace=True)

#--------------------CUSTOM FILTER HERE--------------------------

df = df[df["durata_affitto"]>=pd.Timedelta(days=30, hours=0)]
df = df[df["clu_canonemensile"] > 150]
df = df[df["clu_tipologialetto"] == 100000000]
df = df[df["tipo_stanza"] ==  "Stanza singola"]
df = df[df.clu_modellocontratto=="CNT-IND"]
df = df[pd.to_datetime(df["clu_datainiziocontratto"])>pd.datetime(2014,1,1)]


canoni_distinct = df["clu_canonemensile"].unique().tolist()
canoni_distinct.sort()

canoni_presenti_5 = []
for c in canoni_distinct:

        d = df[df["clu_canonemensile"]==c]

        if d.shape[0]>5:
                canoni_presenti_5.append(c)
        
df = df[df['clu_canonemensile'].isin(canoni_presenti_5)]
                


#---------------------------------------------------------------






print("numero di contratti 'Fine Validità':"+str(df.clu_name.count()))
print("numero di contratti indeterminati:"+str(df[df.clu_modellocontratto=="CNT-IND"].clu_name.count()))
print("numero di contratti determinati:"+str(df[df.clu_modellocontratto=="CNT-DET"].clu_name.count()))

print("numero di contratti stanze singole:"+str(df[df.tipo_stanza ==  "Stanza singola"].clu_name.count()))
print("numero di contratti stanze doppie:"+str(df[df.tipo_stanza ==  "Stanza doppia"].clu_name.count()))

df.to_csv("BertieriSel.csv",index=False)

#%% CONFRONTO FREQUENZA CANONE / TIPO STANZA

df_prezzi =  df['clu_canonemensile'].value_counts().sort_index()


canoni = df_prezzi.index
frequenza = df_prezzi.values

#
frequenza_singole = []
frequenza_doppie = []
for p in canoni:
    df_p = df[df.clu_canonemensile == p]
    frequenza_singole.append(df_p[df_p.tipo_stanza ==  "Stanza singola"].clu_name.count())
    frequenza_doppie.append(df_p[df_p.tipo_stanza ==  "Stanza doppia" ].clu_name.count())





columns = canoni
data  = [frequenza_singole,frequenza_doppie]
colors = plt.cm.BuPu(np.linspace(0.35, 1.2, 3)) #TODO controllare
n_rows = len(data)

index = np.arange(len(columns)) 
bar_width = 0.4
# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)



plt.legend(["singola","doppia"])
plt.ylabel("numero di contratti fine validità")
plt.xlabel("canone mensile")
plt.yticks()

ind_ticks = index[0::6]
can_lab = canoni[0::6]



plt.xticks(ticks =ind_ticks, labels=can_lab)
plt.title("Frequenza Canone Mensile")
plt.savefig("Frequenza Canone Mensile", dpi=400)
plt.close()



#%% numeri di postoletto


numero_di_contratti = df['id_postoletto'].value_counts().values

c = Counter(numero_di_contratti)
labels = []
sizes = []

[(labels.append(i),sizes.append( c[i] / len(numero_di_contratti) * 100.0)) for i in c]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.0f%%', textprops={'fontsize':8},
        startangle=90)

ax1.set_title("occorrenze posto letto in contratti conclusi")
ax1.axis('equal')

plt.savefig("occorrenze", dpi=400)
plt.close()


#%%

for j in range(7):

        inizio = df['clu_datainiziocontratto'].tolist()[j*50:(j+1)*50]
        stipula = df['clu_datastipula'].tolist()[j*50:(j+1)*50]
        disdetta = df['clu_datadisdetta'].replace(np.nan, '01/01/2021', regex=True).tolist()[j*50:(j+1)*50]
        scadenza = df['clu_datascadenza'].tolist()[j*50:(j+1)*50]
        modello_contratto =df['clu_modellocontratto'].tolist()[j*50:(j+1)*50]


        inizio = [datetime.strptime(d, "%d/%m/%Y") for d in inizio]
        scadenza = [datetime.strptime(d, "%d/%m/%Y") for d in scadenza]
        disdetta = [datetime.strptime(d, "%d/%m/%Y") for d in disdetta]
        stipula = [datetime.strptime(d, "%d/%m/%Y") for d in stipula]

        




        y = np.arange(len(inizio))




        plt.scatter(inizio,y, s=5, c='b', marker="^")
        plt.scatter(scadenza,y, s=5,c ='r', marker="v")
        plt.scatter(stipula,y, s=1, color='y',marker='.') 

        blue_star = mlines.Line2D([], [], color='b',  marker='^', linestyle='None',
                          markersize=5, label='dt inizio')

        red_square = mlines.Line2D([], [], color='r', marker='v', linestyle='None',
                          markersize=5, label='dt scadenza')

        purple_triangle = mlines.Line2D([], [], color='y', marker='.', linestyle='None',
                          markersize=5, label='dt stipula')

        disdetta_x = mlines.Line2D([], [], color='k', marker='x', linestyle='None',
                          markersize=5, label='dt disdetta')

        lb  = mlines.Line2D([], [], color='b', marker='_', linestyle='None',
                          markersize=5, label='cnt ind')
        lr  = mlines.Line2D([], [], color='r', marker='_', linestyle='None',
                          markersize=5, label='cnt det')
    

        plt.legend(handles=[blue_star, red_square, purple_triangle, disdetta_x, lb, lr])


        for i in range(len(inizio)):
                if disdetta[i] != datetime(2021, 1, 1, 0, 0): 
                        plt.scatter(disdetta[i],y[i],s =5, marker="x", color='k')

        for i in range(len(inizio)):        
                x_a = [stipula[i],scadenza[i]]
                y_a = [y[i],y[i]]
                if(modello_contratto[i] == "CNT-IND"):
                        plt.plot(x_a,y_a,color='b',linewidth=0.22)
                else:
                        plt.plot(x_a,y_a,color='m' ,linewidth=0.22)

        plt.title("Timeline contratto "+ str(j))
        #plt.gcf().set_size_inches((20, 30)) 
        plt.gcf().autofmt_xdate()
        plt.xlabel("Data")
        plt.yticks(ticks=y,labels=df['clu_name'].tolist()[j*50:(j+1)*50],fontsize=4)
        plt.savefig(str(j), dpi=400)
        plt.close()

        #plt.show()

'''
# si considera la più stretta fra le due

inizio = df['clu_datainiziocontratto'].tolist()
stipula = df['clu_datastipula'].tolist()
disdetta = df['clu_datadisdetta'].replace(np.nan, '01/01/2021', regex=True).tolist()
scadenza = df['clu_datascadenza'].tolist()
modello_contratto =df['clu_modellocontratto'].tolist()
canoni = df["clu_canonemensile"]


inizio = [datetime.strptime(d, "%d/%m/%Y") for d in inizio]
scadenza = [datetime.strptime(d, "%d/%m/%Y") for d in scadenza]
disdetta = [datetime.strptime(d, "%d/%m/%Y") for d in disdetta]
stipula = [datetime.strptime(d, "%d/%m/%Y") for d in stipula]

durata = []
for i in range(len(inizio)):
        if (disdetta[i]>scadenza[i]):
                durata.append(scadenza[i]-inizio[i])
        else:
                durata.append(disdetta[i]-inizio[i])
durata_days = []
for d in durata:
        durata_days.append(d.days)

print(durata_days)
print("min duration in days"+str(np.min(durata_days)))
print("max duration in days:"+str(np.max(durata_days)))
print("average duration in days: "+str( np.average(durata_days)))
print("std: "+str(np.std(durata_days)))

print("avg")

'''


#%% CONFRONTO CANONI E DURATA  MEDIA
'''
canoni_distinct = df["clu_canonemensile"].unique().tolist()
canoni_distinct = np.sort(canoni_distinct)
print(canoni_distinct)

average_duration = []
number_of_element = []
std_list = []

for c in canoni_distinct:
        df_partial = df[df["clu_canonemensile"]==c]

        inizio = df['clu_datainiziocontratto'].tolist()
        scadenza = df['clu_datascadenza'].tolist()
        disdetta = df['clu_datadisdetta'].replace(np.nan, '01/01/2021', regex=True).tolist()    

        inizio = [datetime.strptime(d, "%d/%m/%Y") for d in inizio]
        scadenza = [datetime.strptime(d, "%d/%m/%Y") for d in scadenza]
        disdetta = [datetime.strptime(d, "%d/%m/%Y") for d in disdetta]


        durata = []
        for i in range(len(inizio)):
                if (disdetta[i]>scadenza[i]):
                        durata.append(scadenza[i]-inizio[i])
                else:
                        durata.append(disdetta[i]-inizio[i])
        durata_days = []
        for d in durata:
                durata_days.append(d.days)
        
        average_duration.append(np.average(durata_days))
        number_of_element.append(len(durata_days))
        std_list.append(np.std(durata_days))
        

print(np.sum(number_of_element))

'''
#%% DURATA MEDIA E VARIANZA
duration_days = []


for d in df["durata_affitto"].tolist():
        duration_days.append(d.days)

print("avg duration: "+str(np.average(duration_days)))
print("std: "+str(np.std(duration_days)))
print("min: "+str(np.min(duration_days)))
print("max: "+str(np.max(duration_days)))
#%%

canoni_distinct = df["clu_canonemensile"].unique().tolist()
canoni_distinct.sort()

print("canoni:")
print(canoni_distinct)
print("numero di canoni:"+str(len(canoni_distinct)))

#%%

avg_durata = []
std_durata = []
num_durata = []

for c in canoni_distinct:
        duration_list = df[df["clu_canonemensile"]==c].durata_affitto.to_list()
        duration_days = []
        for d in duration_list:
                duration_days.append(d.days)
        
        avg_durata.append(np.average(duration_days))
        std_durata.append(np.std(duration_days))
        num_durata.append(len(duration_days))
        
        



confidence_interval = (2*np.array(std_durata))/np.sqrt(num_durata)
yerr = np.linspace(0.05, 0.2, len(canoni_distinct))

plt.errorbar(canoni_distinct, avg_durata , yerr=confidence_interval, label='both limits (default)',elinewidth=0.9,capsize=2,errorevery=1)#check



plt.xlabel("canone")
plt.ylabel("durata media contratto (in giorni)")

plt.savefig("canoni_durata",dpi=400)
plt.close()

#%% scatter tipologia letto
y = np.arange(len(df))
posti_letto = df["clu_tipologialetto"].tolist()

clu_canone = df["clu_canonemensile"].tolist()
print(df["tipo_letto"].unique().tolist())

count_letto_singolo = 0
count_letto_piazza_e_mezzo = 0

for i in range(len(posti_letto)):
        if posti_letto[i] == 1:
                #1
                plt.scatter(y[i],clu_canone[i],s =5, c="r")
                count_letto_singolo +=1
        else:
                #1 1+1/""
                plt.scatter(y[i],clu_canone[i],s =5  ,c="g")
                count_letto_piazza_e_mezzo +=1


lg  = mlines.Line2D([], [], color='g', marker='.', linestyle='None',
                          markersize=5, label='"piazza 1 +1/2"')
lr  = mlines.Line2D([], [], color='r', marker='.', linestyle='None',
                          markersize=5, label='piazza singola')
plt.legend(handles=[lb, lr])
plt.title("relazione tipo_posto_letto / canone")
plt.xlabel("id_contratto")
plt.ylabel("canone_mensile")



lg  = mlines.Line2D([], [], color='g', marker='.', linestyle='None',
                          markersize=5, label='piazza 1 +1/2')
lr  = mlines.Line2D([], [], color='r', marker='.', linestyle='None',
                          markersize=5, label='piazza singola')

plt.legend(handles=[lg, lr])

#plt.show()
plt.savefig("relazione tipo_posto_letto canone",dpi=400)
print(" numero contratti letto singolo: "+str(count_letto_singolo))
print(" numero contratto piazza 1+1/2:" + str(count_letto_piazza_e_mezzo) )

plt.close()


#%% scatter balcone
y = np.arange(len(df))
posti_letto = df["clu_balcone"].tolist()
clu_canone = df["clu_canonemensile"].tolist()

count_letto_singolo = 0
count_letto_piazza_e_mezzo = 0

for i in range(len(posti_letto)):
        if posti_letto[i] == 1:
                #1
                plt.scatter(y[i],clu_canone[i],s =5, c="y")
                count_letto_singolo +=1
        else:
                #1 1+1/""
                plt.scatter(y[i],clu_canone[i],s =5  ,c="m")
                count_letto_piazza_e_mezzo +=1


lg  = mlines.Line2D([], [], color='y', marker='.', linestyle='None',
                          markersize=5, label='balcone')
lr  = mlines.Line2D([], [], color='m', marker='.', linestyle='None',
                          markersize=5, label='no balcone')

plt.legend(handles=[lg, lr])
plt.title("relazione balcone / canone")
plt.xlabel("id_contratto")
plt.ylabel("canone_mensile")


plt.show()
plt.savefig("relazione balcone canone",dpi=400)


#%% scatter metriquadratiq
y = np.arange(len(df))
metri_quadri = df["clu_metriquadraticommerciali"].tolist()
clu_canone = df["clu_canonemensile"].tolist()


metri_quadri_distinct = df['clu_metriquadraticommerciali'].unique().tolist()
metri_quadri_distinct.sort()
print(metri_quadri_distinct)


color_dicts =	{
  "13": "b",
  "14": "g",
  "15": "r",
  "16": "c",
  "17": "m",
  "18": "y",
  "19": "tab:orange",
  "20": "tab:brown",
  "21": "tab:pink",
  "22": "tab:gray"

}



for i in range(len(posti_letto)):
        plt.scatter(y[i],clu_canone[i],s =5, c=color_dicts[str(metri_quadri[i])])

l = []

for i in range(len(metri_quadri_distinct)):
        x = color_dicts[str(metri_quadri_distinct[i])]
        l.append(mlines.Line2D([], [], color=x, marker='.', linestyle='None',
                          markersize=5, label=str(metri_quadri_distinct[i])))



plt.legend(handles=l)
plt.title("relazione metriquadri / canone")
plt.xlabel("id_contratto")
plt.ylabel("canone_mensile")


#plt.show()
plt.savefig("relazione metriquadrati canone",dpi=400)

plt.close()
#%% scatter aria
y = np.arange(len(df))
posti_letto = df["clu_ariacondizionata"].tolist()
clu_canone = df["clu_canonemensile"].tolist()

count_letto_singolo = 0
count_letto_piazza_e_mezzo = 0

for i in range(len(posti_letto)):
        if posti_letto[i] == 1:
                #1
                plt.scatter(y[i],clu_canone[i],s =5, c="y")
                count_letto_singolo +=1
        else:
                #1 1+1/""
                plt.scatter(y[i],clu_canone[i],s =5  ,c="m")
                count_letto_piazza_e_mezzo +=1


lg  = mlines.Line2D([], [], color='y', marker='.', linestyle='None',
                          markersize=5, label='con Ariac condizionata')
lr  = mlines.Line2D([], [], color='m', marker='.', linestyle='None',
                          markersize=5, label='no AC')

plt.legend(handles=[lg, lr])
plt.title("relazione Aria Condizionata / canone")
plt.xlabel("id_contratto")
plt.ylabel("canone_mensile")


#plt.show()
plt.savefig("relazione aria condizionata canone",dpi=400)
plt.close()
#%%
#clu_inquilinoid
print(len(df["clu_canonemensile"].tolist()))
print(len(df["clu_inquilinoid"].unique().tolist()))

#%%

inquilino_list = df["clu_inquilinoid"].tolist()
inquilino_distinct =df["clu_inquilinoid"].unique().tolist()
y = np.arange(len(inquilino_distinct))

y_tick_list = []
counter = 0

for m in range(len(y)):
        inq_dist = inquilino_distinct[m]
        df_inq = df[df["clu_inquilinoid"]==inq_dist]
        if (df_inq["clu_inquilinoid"].count()>1):
                counter +=1

y_to_plot = 0
for m in range(len(y)):
        inq_dist = inquilino_distinct[m]
        df_inq = df[df["clu_inquilinoid"]==inq_dist]
        if (df_inq["clu_inquilinoid"].count()>1):
                y_tick_list.append(str(df_inq["clu_inquilinoid"].tolist()[0]))
                
                inizio = df_inq['clu_datainiziocontratto'].tolist()
                stipula = df_inq['clu_datastipula'].tolist()
                disdetta = df_inq['clu_datadisdetta'].replace(np.nan, '01/01/2021', regex=True).tolist()
                scadenza = df_inq['clu_datascadenza'].tolist()
                modello_contratto =df_inq['clu_modellocontratto'].tolist()
                        
                inizio = [datetime.strptime(d, "%d/%m/%Y") for d in inizio]
                scadenza = [datetime.strptime(d, "%d/%m/%Y") for d in scadenza]
                disdetta = [datetime.strptime(d, "%d/%m/%Y") for d in disdetta]
                stipula = [datetime.strptime(d, "%d/%m/%Y") for d in stipula]

                assert(len(inizio) == len(scadenza) and len(disdetta) == len(stipula) and len(stipula) ==  len(inizio))

                for a in range(len(inizio)):
                        plt.scatter(inizio[a],y_to_plot, s=5, c='b', marker="^")
                        plt.scatter(scadenza[a],y_to_plot, s=5,c ='r', marker="v")
                        plt.scatter(stipula[a],y_to_plot, s=1, color='y',marker='.')

                        if disdetta[a] != datetime(2021, 1, 1, 0, 0): 
                                plt.scatter(disdetta[a],y_to_plot,s =5, marker="x", color='k')
                        
                        x_a = [stipula[a],scadenza[a]]
                        y_a = [y_to_plot,y_to_plot]
                        if(modello_contratto[a] == "CNT-IND"):
                                plt.plot(x_a,y_a,color='b',linewidth=0.22)
                        else:
                                plt.plot(x_a,y_a,color='m' ,linewidth=0.22)

                y_to_plot += 1




        


blue_star = mlines.Line2D([], [], color='b',  marker='^', linestyle='None',
                          markersize=5, label='dt inizio')

red_square = mlines.Line2D([], [], color='r', marker='v', linestyle='None',
                          markersize=5, label='dt scadenza')

purple_triangle = mlines.Line2D([], [], color='y', marker='.', linestyle='None',
                          markersize=5, label='dt stipula')

disdetta_x = mlines.Line2D([], [], color='k', marker='x', linestyle='None',
                         markersize=5, label='dt disdetta')

lb  = mlines.Line2D([], [], color='b', marker='_', linestyle='None',
                          markersize=5, label='cnt ind')
lr  = mlines.Line2D([], [], color='r', marker='_', linestyle='None',
                          markersize=5, label='cnt det')   

plt.legend(handles=[blue_star, red_square, purple_triangle, disdetta_x, lb, lr])
plt.title("Timeline contratto per inquilini  ")
#plt.gcf().set_size_inches((20, 30)) 
plt.gcf().autofmt_xdate()
plt.xlabel("Data")
plt.yticks(ticks = np.arange(counter), labels = y_tick_list)

plt.savefig("timelinexinquilino", dpi=400)
        

plt.show()
plt.close() 
  

#%%
df = df.sort_values(by=['durata_affitto'],ascending=False)
df["durata_affitto_days"] =df["durata_affitto"].astype('timedelta64[D]')
x_tick = np.arange(df.size)
#%%

from sklearn import preprocessing
import matplotlib.cm as cm
import matplotlib.colors as colors


#canone_max = df["clu_canonemensile"].max()
#color = [str(item/canone_max) for item in df["clu_canonemensile"]]

canoni = np.array(df["clu_canonemensile"].to_list())


norm = colors.Normalize(canoni.min(), canoni.max())
colors = cm.Reds(norm(canoni))



plt.bar(df['clu_name'],df["durata_affitto_days"], color= colors)
plt.xticks(x_tick=np.arange(len(canoni)),labels=np.arange(len(canoni)))

plt.xlabel("id_affitto")

'''
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
'''

plt.ylabel("durata_contratto (days)")
plt.title("durata contratti")
plt.savefig("durata affitti", dpi=400)
plt.show()

#%% costruzione lista

from Experiment.RentScenario.Contratto import Contratto
contratti = []
for d in df.itertuples():       
        c = Contratto(d.clu_name,d.clu_canonemensile,d.durata_affitto,d.clu_datainiziocontratto,d.clu_datadisdetta,d.clu_datascadenza)
        contratti.append(c)

print(len(contratti))

import pickle
pickle.dump( contratti, open( "lista_contratti_saved.p", "wb" ) )

contratti = pickle.load( open( "lista_contratti_saved.p", "rb" ) )#%%


print(len(contratti))
print(contratti[0].data_inizio)
#%% sfitto
