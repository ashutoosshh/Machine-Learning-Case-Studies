import pandas as pd

import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


#functin name: loadpreservemodel
# Descreption : split X<Y ,traning f=data testing datda
#parameter : none
#Return None
#Date: 14/03/26
#author :Ashutosh Gunjal
#------------------------------------------------------


def loadpreservedmodel(filename):
    loaded_model =joblib.load(filename)
    print("model Succesfully loaded")


    return loaded_model
#functin name: preservemodel
# Descreption : split X<Y ,traning f=data testing datda
#parameter : none
#Return None
#Date: 14/03/26
#author :Ashutosh Gunjal
#------------------------------------------------------


def preservemodel(model,filename):

    joblib.dump(model,filename)

    print("model preserved Sucesfully with name :",filename)
#functin name: train mode
# Descreption : split X<Y ,traning f=data testing datda
#parameter : none
#Return None
#Date: 14/03/26
#author :Ashutosh Gunjal
#------------------------------------------------------
def traintitanicmodel(df):

    X=df.drop("Survived",axis=1)

    Y=df["Survived"]


    print("features :")

    print(X.head())

    print("Lables")
    print(Y.head())

    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    print("X_train sahpe:",X_train.shape)

    print("X_test sahpe:",X_test.shape)
    print("Y_train sahpe:",Y_train.shape)
    print("Y_test sahpe:",Y_test.shape)

    model=LogisticRegression(max_iter=1000)

    model.fit(X_train,Y_train)

    print("Model Trained Succesfully")

    print(model.intercept_)

    print("coefficient")

    for feature,coefficient in zip (X.columns,model.coef_[0]):
        print(feature , ":" ,coefficient)

    preservemodel(model,"marvelloustitanic.pkl")

    loaded_model=loadpreservedmodel("marvelloustitanic.pkl")


    Y_pred=loaded_model.predict(X_test)
    accuracy=accuracy_score(Y_pred,Y_test)

    print("accuracy",accuracy*100)



#functin name: Display info
# Descreption : It displays the formated Title
#               
#parameter : title(str)
#Return None
#Date: 14/03/26
#author :Ashutosh Gunjal
#------------------------------------------------------
def displayinfo(title):
    print("\n"+"="*70)
    print(title)
    print("="*70)
#------------------------------------------------------

#functin name: show data
#descreption: it shows basic information about dataset
#parameter : dataset-> pandsas datdaframe object
# message--> heading text to display
# #Return None
#Date: 14/03/26
#author :Ashutosh Gunjal
#------------------------------------------------------



def showdata(df,message):
    displayinfo(message)

    print("First Five Rows of Dataset")

    print(df.head())

    print("\n sahpe od=f Dtatset")

    print(df.shape)

    print("\n columns names")
    print(df.columns.tolist())

    print("\n missing value in each column")

    print(df.isnull().sum())


    
#functin name: main
# Descreption : it does pre processing
               # it removes unnessary columns\
               #it handles misssinng value
               #it converts text data to numeric format
               #it does encoding CATEGORICAL colunms

#parameter : df--> pandas data frame
#Return :df--> clean  pandas dataframe 
#Date: 14/03/26
#author :Ashutosh Gunjal
def cleanTitanocdata(df):
    displayinfo("Step 2: Original data")

    print(df.head())
    #Remove unnecessary Columns
    drop_columns=["Passengerid","zero"]
    exsisting_columns=[col for col in drop_columns if col in df.columns]
    print("\n Columns to be drop")

    print(exsisting_columns)

    #drop the unwanted column 
    df=df.drop(columns=exsisting_columns)
    displayinfo("Step 2: Data After Colunm Removal")

    print(df.head())
#handel age column
    if "Age" in df.columns:
        print("Age Column before filling missing values")

        print(df["Age"].head(10))

        #coerce==> invalid valide get converted into nan
        df["Age"]=pd.to_numeric(df["Age"],errors="coerce")

        age_median=df["Age"].median()

        #Replace misssing values with median

        df["Age"]=df["Age"].fillna(age_median)

        print("Age column after preprocessing:")

        print(df["Age"].head())

        #handel fair column

    if "Fare" in df.columns:
        print("\n Fair Column before pre processing ")
        print(df["Fare"].head(10))
        df["Fare"]=pd.to_numeric(df["Fare"],errors="coerce")


        fare_median=df["Fare"].median()

        df["Fare"]=df["Fare"].fillna(fare_median)

        print("median of fair is:",fare_median)

        print("embark value before preprocessing")
        print(df["Embarked"].head(10))
#conver the dadtda intio String
        df["Embarked"]=df["Embarked"].astype(str).str.strip()

        #Remove misssing ValueError
        df["Embarked"]=df["Embarked"].replace(['nan','None',''],np.nan)

        embarked_mode=df["Embarked"].mode()[0]

        print("\n modee of embarked column:",embarked_mode)

        df["Embarked"]=df["Embarked"].fillna(embarked_mode)

    if "Sex" in df.columns:
        print("\n Fair Column before pre processing ")
        print(df["Sex"].head(10))
        df["Sex"]=pd.to_numeric(df["Sex"],errors="coerce")

        print("sex column after pre processing")

        print(df["Sex"].head(10))

    displayinfo("data after pre processig ")

    print(df.head())

    print("\n missng value after pre processing")

    print(df.isnull().sum())


    #Encode Embarked column
    df=pd.get_dummies(df,columns=["Embarked"],drop_first=True)
    print("\n Data After encoding ")

    print(df.head())
    print("Shape of Dataset",df.shape)
    #convert blooean column into integer 
    for col in df.columns:
        if df[col].dtype== bool:
            df[col]=df[col].astype(int)

    print("\n Data After encoding ")

    print(df.head())

    


    return df



#------------------------------------------------------

#functin name: MArvellousLogistic
# Descreption :This is main pipeline controller
#               it loads the dataset ,shows the raw data
#               it pre process the dataset and train the model

#parameter : Datapath of dataset file
#Return None
#Date: 14/03/26
#author :Ashutosh Gunjal
#------------------------------------------------------

def MarvellousLogistic(datapath):

    displayinfo("Step 1: Loading The dataset")

    df=pd.read_csv(datapath)
    showdata(df,"Initial Dataset ")
    df=cleanTitanocdata(df)

    traintitanicmodel(df)


#functin name: main
# Descreption :Starting point of parameter
#parameter : none
#Return None
#Date: 14/03/26
#author :Ashutosh Gunjal

def main():

    MarvellousLogistic("MarvellousTitanicDataset.csv")

    



if __name__=="__main__":

    main()