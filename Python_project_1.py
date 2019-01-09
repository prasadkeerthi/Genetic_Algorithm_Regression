
# coding: utf-8

# In[747]:


import pandas as pd
import numpy as np
import math
import copy
import random
df=pd.read_csv('project_1.csv')


# In[748]:


df.shape


# In[749]:


cols = [np.arange(5,16)]
cols = np.delete(cols, 8)

df=df.drop(df.columns[cols],axis=1)



g=df.ix[:,[4]]
g=np.asarray(g)
g=g.ravel()
df=df.drop(df.columns[4],axis=1)
df['Abdomen circumference']=g

# In[750]:



# In[751]:

#normalizing all the values

max_a=max(df['Weight lbs'])
min_a=min(df['Weight lbs'])
df['Weight lbs']=(df['Weight lbs']-min_a)/(max_a-min_a)

max_a=max(df['Height inch'])
min_a=min(df['Height inch'])
df['Height inch']=(df['Height inch']-min_a)/(max_a-min_a)

max_a=max(df['Neck circumference'])
min_a=min(df['Neck circumference'])
df['Neck circumference']=(df['Neck circumference']-min_a)/(max_a-min_a)

max_a=max(df['Chest circumference'])
min_a=min(df['Chest circumference'])
df['Chest circumference']=(df['Chest circumference']-min_a)/(max_a-min_a)

max_a=max(df['Abdomen circumference'])
min_a=min(df['Abdomen circumference'])
df['Abdomen circumference']=(df['Abdomen circumference']-min_a)/(max_a-min_a)

max_a=max(df['bodyfat'])
min_a=min(df['bodyfat'])
df['bodyfat']=(df['bodyfat']-min_a)/(max_a-min_a)







# In[752]:

#splitting train and test data


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, df['bodyfat'], test_size=0.2,shuffle=False)
X_train.head()
y_test=X_test.bodyfat
X_test=X_test.drop(['bodyfat'], axis=1)
X_test=X_test.values.tolist()


# In[753]:




y_train=np.asarray(y_train)
y_train=y_train.ravel()


# In[754]:


y_test=np.asarray(y_test)


# In[755]:


Y_train=X_train.bodyfat
X_train=X_train.drop(['bodyfat'], axis=1)


# In[756]:


X_train.head()


# In[757]:
#generating weights between -1 and 1 with population of 500

A=[]
min_wt_val=-1
max_wt_val=1
for i in np.arange(0,500):
 k = np.random.uniform(low=min_wt_val, high=max_wt_val, size=(10,5) )
 A.append(k)


# In[758]:


original_weight_matrix=A
keep_original=copy.deepcopy(A) 


# In[759]:


X_train_list=X_train.values.tolist()
X_train_list=np.asarray(X_train_list)

y_train_list=np.asarray(Y_train)
a=min(y_train_list)
b=(max(y_train_list)-min(y_train_list))
y_train_list=(y_train_list-a)/b





# In[760]:


def calculate_yhat(wt,X_set):
    y_hat=[]
    for s in np.arange(0,len(wt)):
     k=0
     p=0
     for i in np.arange(0,len(X_set)):
      k=0  
      p=np.dot(X_set[i],wt[s].T)
      for j in p:

       k=k+1/(1+math.exp(-j))
     y_hat.append(k)
    return(y_hat)


# In[761]:


def calculate_fitness(est_y,weight_matrix):
    fvalue=[]
    k=0
    for j in np.arange(0,len(weight_matrix)):
     k=0   
     for i in np.arange(0,len(X_train_list)):
        k=k+(est_y[j]-y_train_list[i])**2
     fvali=(1-(k/len(X_train_list)))*100
     fvalue.append(fvali)
    return(fvalue)
 

def cross_over(weight_matrix,index) :
    for i in np.arange(0,len(weight_matrix)):

     for k in np.arange(0,len(weight_matrix[i])):
        weight_matrix[i][k]=(weight_matrix[i][k]+1)/2

     weight_matrix[i]=np.array(weight_matrix[i])*1000
     weight_matrix[i]=weight_matrix[i].astype(int)
     
     weight_matrix[i]=weight_matrix[i].ravel()
     binary_x=[]
     for j in weight_matrix[i]:
      binary_x.append(bin(j)[2:].zfill(10)) 
     weight_matrix[i]=binary_x
     weight_matrix[i]=np.reshape(weight_matrix[i], (10, 5))  
    
    

    
    
    children=[]
    
    for i in np.arange(0,len(weight_matrix)):
      x=weight_matrix[index].ravel()
      parent_1 = "".join(map(str, x))
      x=weight_matrix[i].ravel()
      parent_2 = "".join(map(str, x))
      C_Point = np.random.random_integers(2,499)
      chromosome=parent_1[0:C_Point]+parent_2[C_Point:]
      for mut in np.arange(0,0.05*len(chromosome)):
        rand=random.randint(1,499)
        if(chromosome[rand]=='0'):
            chromosome=chromosome[0:rand]+'1'+chromosome[rand+1:]
        if(chromosome[rand]=='1'):
            chromosome=chromosome[0:rand]+'0'+chromosome[rand+1:]
            
        
      children.append(chromosome)
        
    new_children=[]
    for k in np.arange(0,len(weight_matrix)):
        dec_child=[]
        for ns in np.arange(0,len(weight_matrix),10):
            dec_child.append(children[k][ns:ns+10])
        x=np.reshape(dec_child, (10, 5))   
        new_children.append(x)
        
      
    for i in np.arange(0,len(weight_matrix)):
     for j in np.arange(0,10):
        for k in np.arange(0,5):
            new_children[i][j][k]=int(new_children[i][j][k], 2)
    new_children = np.array(new_children, dtype=np.int32)
    
    new_children=new_children/1000
    new_children=(2*new_children)-1
    
    return(new_children)



def create_new_population(ko,nc,fv,nfv):
    joint_list=(fv+nfv)
    list.sort(joint_list,reverse=True)
    median_value=joint_list[499]
    
    newpopulation=[]
    
    for i in np.arange(0,min(len(fv),len(nfv),len(ko),len(nc))):
       # print(i,len(fv),len(nfv),len(newpopulation))
        if fv[i]>median_value:
            newpopulation.append(ko[i])
        if nfv[i]>median_value:
            newpopulation.append(nc[i])

   
    fv=nfv
    return(newpopulation)
    


# In[762]:
#first crosover

y_hat=calculate_yhat(original_weight_matrix,X_train_list)



fval=calculate_fitness(y_hat,original_weight_matrix)

max_index=np.argmax(fval)
temp=copy.copy(original_weight_matrix)
new_children=cross_over(original_weight_matrix,max_index)
original_weight_matrix=temp



new_y_hat=calculate_yhat(new_children,X_train_list)
new_fval=calculate_fitness(new_y_hat,new_children)

keep_original=copy.copy(original_weight_matrix)
new_pop=create_new_population(keep_original,new_children,fval,new_fval)
new_y_hat=calculate_yhat(new_pop,X_train_list)
new_fval=calculate_fitness(new_y_hat,new_pop)

max_index=np.argmax(new_fval)
fit_max=[]
fit_max.append(max(new_fval))



   



# In[763]:
#iterating till there is no significant improvement in the fitness values


for i in np.arange(0,30):
    print(i)
    print(max(fval),max(new_fval))
    if max(new_fval)>max(fval):
        max_index=np.argmax(new_fval)
        keep_original=copy.copy(new_pop)
        max_index=np.argmax(new_fval)
        new_pop=cross_over(new_pop,max_index)
        # new_pop=copy.copy(temp)
        fval=copy.copy(new_fval)
        new_pop=create_new_population(keep_original,new_pop,fval,new_fval)
        new_y_hat=calculate_yhat(new_pop,X_train_list)
        new_fval=calculate_fitness(new_y_hat,new_pop)
        fit_max.append(max(new_fval))

    else:
         
         max_index=np.argmax(fval)
         #temp=copy.copy(new_pop)
         new_pop=cross_over(keep_original,max_index)
         #keep_original=copy.copy(temp)
         #fval=copy.copy(new_fval)
         new_pop=create_new_population(keep_original,new_pop,fval,new_fval)
         new_y_hat=calculate_yhat(new_pop,X_train_list)
         new_fval=calculate_fitness(new_y_hat,new_pop)
         fit_max.append(max(fval))
         
        
   
    
     
        


    
     
   
 


# In[764]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

x = np.arange(0,len(fit_max))
y = fit_max


fig = plt.figure()
plt.plot(x, y,'o',color='black');
ax=fig.suptitle('Iteration vs Fitness Value', fontsize=20)


plt.show()

    


# In[765]:


p=y_test
a=min(p)
b=(max(p)-min(p))
p=np.array(p, dtype=int)
p=(p-a)/b


# In[789]:
#choosing best weight


max_index=np.argmax(new_fval)
op_wt=new_pop[max_index]
print("The weight with maximum fitness value is")
print(op_wt)


# In[790]:

#calculating prediction with best weights

expected_y=[]
for i in np.arange(0,len(X_test)):
 k=0  
 p=np.dot(X_test[i],op_wt.T)
 for j in p:

     k=k+1/(1+math.exp(-j))
 expected_y.append(k)



x1=[]
x2=[]
for i in np.arange(0,len(X_test)):
    x1.append(X_test[i][0])
    x2.append(X_test[i][1])
    



# In[791]:
#3d plot depicting actual values and prediction

y_test=np.asarray(y_test)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
fig.set_size_inches(10, 8)
ax = Axes3D(fig)
ax.scatter(x1, x2, expected_y,label='Calculated bodyfat',marker='o')
ax.scatter(x1, x2, y_test,label='Actual bodyfat',marker='o')
ax.set_zlabel('Body fat')
ax=fig.suptitle('Weight vs Height vs Bodyfat Normalised values', fontsize=20)
plt.legend(loc=4)
plt.xlabel('Weight_lbs')
plt.ylabel('Height_inch')
plt.show()



# In[792]:

#calculation of mean squared error
MSE=0
for i in np.arange(0,len(y_test)):
    MSE=MSE+(expected_y[i]-y_test[i])**2
MSE=MSE/len(y_test)


# In[793]:
print("mean squared error of the prediction is",MSE)





