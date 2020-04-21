import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from minisom import MiniSom
from pylab import plot,colorbar,bone,pcolor,show
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def get_data(name):
    data=pd.read_csv(name)
    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    return (X,y,data)
def scale(inp,label=None):
    if(label==False):
        sc=MinMaxScaler()
        inp=sc.fit_transform(inp)
    else:
        sc=StandardScaler()
        inp=sc.fit_transform(inp)
    return (inp,sc)

def SOM(inp):
    som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.6)
    som.random_weights_init(inp)
    som.train_random(inp,num_iteration=100)
    return som
def Plot(s,inp,dep):
    marker=["o","s"]
    color=["r","g"]
    bone()
    pcolor(s.distance_map())
    colorbar()
    for i,x in enumerate(inp):
        w=s.winner(x)
        plot(w[0]+0.5,w[1]+0.5,marker[dep[i]],markeredgecolor=color[dep[i]]
                            ,markerfacecolor="None",markeredgewidth=2,markersize=10)
    show()

def main():
    X,y,_=get_data("Credit_Card_Applications.csv")
    X,sc=scale(X)
    s=SOM(X)
    Plot(s,X,y)
    return (s,X,y,sc)

def Neural(inp,label):
    model=Sequential()
    model.add(Dense(4,activation="relu",input_dim=14))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    model.fit(inp,label,epochs=4,batch_size=1)
    return model
    


    
    
    
if __name__=="__main__":
    Som,X,Y,sc=main()
    mappings=Som.win_map(X)
    
    frauds=np.concatenate((mappings[(1,7)],mappings[(2,7)],mappings[(2,1)]),axis=0)
    frauds=sc.inverse_transform(frauds)
    is_fraud=np.zeros(690)
    _,_,dataset=get_data("Credit_Card_Applications.csv")
    I=dataset.iloc[:,1:-1].values
    I,_=scale(I,label=True)
    for i in range(690):
        if dataset.iloc[i,0] in frauds:
            is_fraud[i]=1
    
    model=Neural(I,is_fraud)
    plt.plot([1,2,3,4],model.history.history["loss"],c="red")
    plt.plot([1,2,3,4],model.history.history["accuracy"],"b-")
    plt.show()
    pred=model.predict(I)
    test=np.concatenate((dataset.iloc[:,0:1],pred),axis=1)
    final_prediction=test[test[:,1].argsort()]
    
    
    
    
    
    
        
    
    
    
    
    
    
        

    
    
    
    




    

    
