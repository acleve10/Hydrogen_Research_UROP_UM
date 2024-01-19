from __future__ import division # convert int or long division arguments to floating point values before division
import numpy as np, scipy, pandas as pd
from scipy import optimize
from pyomo.environ import *
from pyomo.opt import SolverFactory
import itertools
import csv
from pyomo.core import Var
from pyomo.core import Param
from operator import itemgetter
from datetime import datetime
import os
import matplotlib.pyplot as plt

HorizonHours = 8759 ##planning horizon (e.g., 24, 48, 72 hours etc.)

data_name = 'test'

df_storageHigh = pd.read_csv('StorageFleetHigh.csv',header=0)
df_storageLow = pd.read_csv('StorageFleetLow.csv',header=0)




df_MarketDataLow = pd.read_csv('StdScen20_LowRECost_hourly_TX_2050.csv', skiprows=2)
df_MarketDataHigh = pd.read_csv('StdScen20_HighRECost_hourly_TX_2050.csv', skiprows=2)
df_MarketDataMid = pd.read_csv('StdScen20_MidCase_hourly_TX_2050.csv', skiprows=2)


def wholeOpt(df_storage, df_MarketData):
    with open(''+str(data_name)+'.dat', 'w') as f:
# storage set
        f.write('set Storage :=\n')
        for storage in range(0,len(df_storage)):
            name = str(df_storage.loc[storage,'Name'])
            name = name.replace(' ','_')
            f.write(name + ' ')
        f.write(';\n\n') 


        f.write('param HorizonHours := %d;' % HorizonHours)
        f.write('\n\n')
####### create parameter matrix for Storage
        f.write('param:' + '\t')
        for c in df_storage.columns:
            if c == 'Capacity' :
                f.write('maxstate' + '\t')
            if c == 'VOM':
                f.write('VOM' + '\t')
            if c == 'MinState' :
                f.write('minstate' + '\t')
            if c == 'eff':
                f.write('eff' + '\t')
            if c == 'engcapi':
                f.write('engcapi' + '\t')
            if c == 'engcapo':
                f.write('engcapo' + '\t')
        f.write(':=\n\n')
        for i in range(0,len(df_storage)):    
            for c in df_storage.columns:
                if c == 'Name':
                    unit_name = str(df_storage.loc[i,'Name'])
                    unit_name = unit_name.replace(' ','_')
                    f.write(unit_name + '\t')  
                if c == 'Capacity' or c == 'VOM' or c == 'MinState' or c == 'eff' or c == 'engcapi':
                    f.write(str((df_storage.loc[i,c])) + '\t')               
            f.write('\n')
        f.write(';\n\n')     

####### create parameter matrix 
        f.write('param:' + '\t')
        for c in df_MarketData.columns:
            if c == 'co2_rate_avg_gen' : 
                f.write('mef' + '\t')
            if c == 'energy_cost_enduse' :
                f.write('mcp' + '\t')
        f.write(':=\n\n')
        for i in range(0,HorizonHours + 1):    
            for c in df_MarketData.columns:
                if c == 'timestamp':
                    timestamp = str(i)
                    f.write(timestamp + '\t')  
                if c == 'co2_rate_avg_gen' or c == 'energy_cost_enduse' : 
                    f.write(str((df_MarketData.loc[i,c])) + '\t')               
            f.write('\n')
        f.write(';\n\n')     

    model = AbstractModel()


#Opperating information

    model.HorizonHours = Param(within=PositiveIntegers)
    model.HH_periods = RangeSet(0,model.HorizonHours)
    model.hh_periods = RangeSet(1,model.HorizonHours)

    model.Storage = Set( )

#MEF
    model.mef = Param(model.HH_periods)
#MCP
    model.mcp = Param(model.HH_periods)
#Max state of charge 
    model.maxstate = Param(model.Storage)
#Min state of charge 
    model.minstate = Param(model.Storage)
#Round Trip Efficency 
    model.eff = Param(model.Storage)
#VOM for storage
    model.VOM = Param(model.Storage)
#max in rate of energy
    model.engcapi = Param(model.Storage)
    model.engcapo = Param(model.Storage)

#Variables
    model.soc = Var(model.Storage,model.HH_periods, within=NonNegativeReals,initialize=0)

    model.inflow = Var(model.Storage,model.HH_periods, within=NonNegativeReals,initialize=0)

    model.output = Var(model.Storage,model.HH_periods, within=NonNegativeReals,initialize=0)

    model.CO2EMRate = Var(model.Storage,model.HH_periods, initialize=0)



    def Profit(model):
        revenue = sum(model.output[j,i] * model.mcp[i-1]for i in model.hh_periods for j in model.Storage )
        costsop = sum(model.engcapi[j] * model.VOM[j] for j in model.Storage)
        costElec = sum(model.inflow[j,i] * model.mcp[i-1]for i in model.hh_periods for j in model.Storage )
        profit = revenue - costElec -costsop
        return profit
    model.Profit = Objective(rule=Profit, sense=maximize)


######============Storage constraints==================########
    def Maxin(model,j,i):
        return model.inflow[j,i]  <= model.engcapi[j] 
    model.Maxin= Constraint(model.Storage,model.hh_periods,rule=Maxin)

    def Maxout(model,j,i):
        return model.output[j,i] <= model.engcapi[j] 
    model.Maxout= Constraint(model.Storage,model.hh_periods,rule=Maxout)

    def SoC(model,j,i):
        if i == 1:
            return model.soc[j,i] == (model.soc[j,i-1] - model.output[j,i] + (model.inflow[j,i] * model.eff[j])) * 0   
        else : 
            return model.soc[j,i] == (model.soc[j,i-1] - model.output[j,i] + (model.inflow[j,i] * model.eff[j]))  
    model.SoC = Constraint(model.Storage,model.hh_periods,rule=SoC)

#Constraints for Max & Min state of charge for discharging storage
    def MaxS(model,j,i):
        return model.soc[j,i] <=  model.maxstate[j] 
    model.MaxState = Constraint(model.Storage,model.hh_periods,rule=MaxS)

    def MinS(model,j,i):
        return model.soc[j,i] >= model.minstate[j]
    model.MinState = Constraint(model.Storage,model.hh_periods,rule=MinS)

    def CO(model,j,i):
        return model.CO2EMRate[j,i] == (model.inflow[j,i] * model.mef[i-1]) - (model.output[j,i] * model.mef[i-1])
    model.CO= Constraint(model.Storage,model.hh_periods,rule=CO)


    instance = model.create_instance('test.dat')
    opt = SolverFactory('gurobi')
    print(opt.solve(instance))

    soc = []
    inflow = []
    output=[]
    CO2 = []
    for day in range(1,2):
##The following section is for storing and sorting results                                            
        for v in instance.component_objects(Var, active=True):
            varobject = getattr(instance, str(v))
            a=str(v)
       
            if a=='output':
                for index in varobject:
                    if int(index[1]>0 and index[1]<HorizonHours + 1):
                        output.append((index[0], day,index[1]+((day-1)*HorizonHours),varobject[index].value))
        for v in instance.component_objects(Var, active=True):
            varobject = getattr(instance, str(v))
            a=str(v)
       
            if a=='inflow':
                for index in varobject:
                    if int(index[1]>0 and index[1]<HorizonHours + 1):
                        inflow.append((index[0], day,index[1]+((day-1)*HorizonHours),varobject[index].value))
        for v in instance.component_objects(Var, active=True):
            varobject = getattr(instance, str(v))
            a=str(v)
       
            if a=='soc':
                for index in varobject:
                    if int(index[1]>0 and index[1]<HorizonHours + 1):
                        soc.append((index[0], day,index[1]+((day-1)*HorizonHours),varobject[index].value))
        for v in instance.component_objects(Var, active=True):
            varobject = getattr(instance, str(v))
            a=str(v)
       
            if a=='CO2EMRate':
                for index in varobject:
                    if int(index[1]>0 and index[1]<HorizonHours + 1):
                        CO2.append((index[0], day,index[1]+((day-1)*HorizonHours),varobject[index].value))
    dis_pd=pd.DataFrame(output,columns=('Storage','Day','Hour','mwh_discharge'))
    inflow_pd=pd.DataFrame(inflow,columns=('Storage','Day','Hour','mwh_charged'))
    soc_pd=pd.DataFrame(soc,columns=('Storage','Day','Hour','SOC-MW'))
    CO2_pd=pd.DataFrame(CO2,columns=('Storage','Day','Hour','kg_CO2'))
    dis_pd.to_csv('discharge.csv') 
    inflow_pd.to_csv('inflow.csv') 
    soc_pd.to_csv('soc.csv')
    CO2_pd.to_csv('NetChangeInMEF.csv')
    CO2_pd=CO2_pd.sort_values(by=["Storage","Hour"])
    CO2_pd=CO2_pd.reset_index(drop=True)
    tot_CO2_pd=CO2_pd.groupby(['Hour'],as_index=False).sum()
    total = tot_CO2_pd['kg_CO2'].sum()
    print(total)
    concatenated_dataframes = pd.concat([dis_pd, inflow_pd, soc_pd, CO2_pd], axis=1)
    concatenated_dataframes.to_csv('LowCost_case_low.csv')
    return CO2_pd
###Plotting 

CO2_pdlow_L= wholeOpt(df_storageLow, df_MarketDataLow)
CO2_pdmid_L= wholeOpt(df_storageLow, df_MarketDataMid)
CO2_pdhigh_L= wholeOpt(df_storageLow, df_MarketDataHigh)

CO2_pdlow_H = wholeOpt(df_storageHigh, df_MarketDataLow)
CO2_pdmid_H= wholeOpt(df_storageHigh, df_MarketDataMid)
CO2_pdhigh_H= wholeOpt(df_storageHigh, df_MarketDataHigh)





'''

X = CO2_pdlow_H.loc[:,'Hour']
Y = CO2_pdlow_H.loc[:,'SOC-MW']
X2 = CO2_pdmid_H.loc[:,'Hour']
Y2 = CO2_pdmid_H.loc[:,'SOC-MW']
X3 = CO2_pdhigh_H.loc[:,'Hour']
Y3 = CO2_pdhigh_H.loc[:,'SOC-MW']
X4 = CO2_pdlow_L.loc[:,'Hour']
Y4 = CO2_pdlow_L.loc[:,'SOC-MW']
X5 = CO2_pdmid_L.loc[:,'Hour']
Y5 = CO2_pdmid_L.loc[:,'SOC-MW']
X6 = CO2_pdhigh_L.loc[:,'Hour']
Y6 = CO2_pdhigh_L.loc[:,'SOC-MW']

'''
X = CO2_pdlow_H.loc[:,'Hour']
Y = CO2_pdlow_H.loc[:,'kg_CO2'].div(1000)
X2 = CO2_pdmid_H.loc[:,'Hour']
Y2 = CO2_pdmid_H.loc[:,'kg_CO2'].div(1000)
X3 = CO2_pdhigh_H.loc[:,'Hour']
Y3 = CO2_pdhigh_H.loc[:,'kg_CO2'].div(1000)
X4 = CO2_pdlow_L.loc[:,'Hour']
Y4 = CO2_pdlow_L.loc[:,'kg_CO2'].div(1000)
X5 = CO2_pdmid_L.loc[:,'Hour']
Y5 = CO2_pdmid_L.loc[:,'kg_CO2'].div(1000)
X6 = CO2_pdhigh_L.loc[:,'Hour']
Y6 = CO2_pdhigh_L.loc[:,'kg_CO2'].div(1000)

YC = df_MarketDataLow.loc[1::,'energy_cost_enduse']
YC2 = df_MarketDataMid.loc[1::,'energy_cost_enduse']
YC3 = df_MarketDataHigh.loc[1::,'energy_cost_enduse']


plt.xlim(1, HorizonHours)
plt.plot(X,Y, label = 'LowCost_H')
plt.plot(X2,Y2, label = 'MidCost_H')
plt.plot(X3,Y3, label = 'HighCost_H')
plt.plot(X4,Y4, label = 'LowCost_L')
plt.plot(X5,Y5, label = 'MidCost_L')
plt.plot(X6,Y6, label = 'HighCost_L')

plt.xlabel('Time (Hour)')
plt.ylabel('Change in Emissions (Kg of CO2)')
plt.legend()
plt.show()

plt.xlim(1, HorizonHours)
plt.plot(X,YC, label = 'LowCost')
plt.plot(X2,YC2, label = 'MidCost')
plt.plot(X3,YC3, label = 'HighCost')
plt.legend()
plt.xlabel('Time (Hour)')
plt.ylabel('$/MWh')
plt.show()