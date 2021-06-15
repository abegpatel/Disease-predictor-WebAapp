from flask import Flask, render_template,request,url_for,redirect,session,logging,Blueprint,flash
import sqlite3 as sql
#import os
#from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn import preprocessing
#from sklearn import utils

app = Flask(__name__)
app.secret_key = "super secret key"

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/services',methods=['GET','POST'])
def services():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('services.html')
@app.route('/product',methods=['GET','POST'])
def product():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('product.html')
@app.route('/index',methods=['GET','POST'])
def index():
    return render_template('index.html')
@app.route('/gallery',methods=['GET','POST'])
def gallery():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('gallery.html')
@app.route('/contact',methods=['GET','POST'])
def contact():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('contact.html')
@app.route('/login',methods=['GET','POST'])
def login():
    return render_template('login.html')
@app.route('/signup',methods=['GET','POST'])
def signup():
    return render_template('signup.html')
@app.route('/diabetes',methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')
@app.route('/heartattack',methods=['GET','POST'])
def heartattack():
    return render_template('heartattack_prediction.html')

@app.route('/kidneyd',methods=['GET','POST'])
def kidneyd():
    return render_template('kidneyd.html')

"""def insertUser(username,password,email,address,pin):
    con = sql.connect("C:/Users/Abeg/Desktop/dipredtion/user.db")
    cur = con.cursor()
    cur.execute("INSERT INTO users (username,password,email,address,pin) VALUES (?,?)", (username,password,email,address,pin))
    con.commit()
    con.close()
def retrieveUsers():
	con = sql.connect("C:/Users/Abeg/Desktop/dipredtion/user.db")
	cur = con.cursor()
	cur.execute("SELECT username, password FROM users")
	users = cur.fetchall()
	con.close()
	return users
"""
@app.route('/adderc',methods=['GET','POST'])
def adderc():
    if request.method == 'POST':
        try:
            username= request.form['username']
            password= request.form['password']
            email= request.form['email']
            address= request.form['address']
            pin= request.form['pin']
            con = sql.connect("C:/Users/Abeg/Desktop/dipredtion/user.db")
            cur = con.cursor()
            cur.execute("INSERT INTO users (username,password,email,address,pin) VALUES (?,?,?,?,?)", (username,password,email,address,pin))
            con.commit()
            con.close()
        except:
            con = sql.connect("C:/Users/Abeg/Desktop/dipredtion/user.db")
            cur = con.cursor()
            cur.rollback()
            msg = "error in insert operation"
      
        finally:
            return render_template('index.html')
            con.close() 
@app.route('/loggedin',methods=['GET','POST'])
def loggedin():
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        con = sql.connect("C:/Users/Abeg/Desktop/dipredtion/user.db")
        cur = con.cursor()
        c = cur.execute("SELECT username from users where username = (?)", [username])
        userexists = c.fetchone()
        if userexists:
            c = cur.execute("SELECT password from users where password = (?)", [password])
            passwcorrect = c.fetchone()
            if passwcorrect:
                #session['logged_in']=True
                #login_user(user)
                flash("logged in")
                return redirect(url_for('index'))
            else:
                return 'incorrecg pw'
        else:
            #return 'fail'
            return render_template('login.html')
"""if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        return render_template('index.html')
    else:
  	    return render_template('login.html')"""
    
@app.route('/dia_pred',methods=['GET','POST'])
def dia_pred():
    df=pd.read_csv("diabetes.csv")
    X=df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
    y=df[['Outcome']]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=50)
    outcome_corr=df.corr().iloc[:,-1].values.tolist()[:-1]
    outcome_corr.sort(reverse=True)
    meancrr=sum(outcome_corr)/len(outcome_corr)
    allowedcol=[]
    for i in outcome_corr:
        if i>meancrr:
            allowedcol.append(df.corr().iloc[:,-1].values.tolist()[:-1].index(i))
    lm=DecisionTreeClassifier()
    lm.fit(X_train,y_train)
    predictions=lm.predict(X_test)
    pd.crosstab(np.array(y_test.values.flatten().tolist()),predictions,rownames=['True'],colnames=['predicted'],margins=True)
    if request.method == 'POST':
        [a,b,c,d,e,f,g,h]=request.form['Pregnancies'],request.form['Glucose'],request.form['BloodPressure'],request.form['SkinThickness'],request.form['Insulin'],request.form['BMI'],request.form['DiabetesPedigreeFunction'],request.form['Age']
        pred=lm.predict([[a,b,c,d,e,f,g,h]])
        if pred[0]==1:
            res='Yes'
            x="YES!you have Diabetes!please consult with the Doctor"
        else:
            res='No'
            x='No!you dont have Diabetes'
        """con = sql.connect("C:/Users/Abeg/Desktop/database/user.db")
        cur = con.cursor()
        cur.execute("INSERT INTO diabetes (Pregnencies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,result) VALUES (?,?,?,?,?,?,?,?,?)", (request.form['Pregnancies'],request.form['Glucose'],request.form['BloodPressure'],request.form['SkinThickness'],request.form['Insulin'],request.form['BMI'],request.form['DiabetesPedigreeFunction'],request.form['Age'],res))
        con.commit()
        con.close()"""
        return render_template('diabetes.html',value=x)
    
    
    
    
@app.route('/Kidneyd_prediction',methods=['GET','POST'])
def Kidneyd_prediction():
    df=pd.read_csv("kidney_disease.csv")
    df["rbc"].fillna("normal", inplace=True)
    df["pc"].fillna("normal", inplace=True)
    df["pcc"].fillna("normal", inplace=True)
    df["ba"].fillna("notpresent", inplace=True)
    df["htn"].fillna("no", inplace=True)
    df["dm"].fillna("no", inplace=True)
    df["cad"].fillna("no", inplace=True)
    df["appet"].fillna("good", inplace=True)
    df["pe"].fillna("no", inplace=True)
    df["classification"].fillna("ckd", inplace=True)
    df["rbc"]=df["rbc"].astype("category").cat.codes
    df["pc"]=df["pc"].astype("category").cat.codes
    df["pcc"]=df["pcc"].astype("category").cat.codes
    df["ba"]=df["ba"].astype("category").cat.codes
    df["htn"]=df["htn"].astype("category").cat.codes
    df["dm"]=df["dm"].astype("category").cat.codes
    df["cad"]=df["cad"].astype("category").cat.codes
    df["appet"]=df["appet"].astype("category").cat.codes
    df["pe"]=df["pe"].astype("category").cat.codes
    df["ane"]=df["ane"].astype("category").cat.codes
    df["classification"]=df["classification"].astype("category").cat.codes
    df["pcv"]=pd.to_numeric(df['pcv'],errors='coerce')
    df["rc"]=pd.to_numeric(df['pcv'],errors='coerce')
    df["wc"]=pd.to_numeric(df['wc'],errors='coerce')
    df1=df.fillna(df.mean())
    outcome_corr=df.corr().iloc[:,-1].values.tolist()[:-1]
    outcome_corr.sort(reverse=True)
    meancrr=sum(outcome_corr)/len(outcome_corr)
    allowedcol=[]
    for i in outcome_corr:
        if i>meancrr:
            allowedcol.append(df.corr().iloc[:,-1].values.tolist()[:-1].index(i))
    sorted(allowedcol)
    fin=[]
    col=df1.columns.tolist()
    for i in col:
        if col.index(i) in allowedcol:
            fin.append(col[col.index(i)])
    df2=df1[fin]
    X=df2[["sg","rbc","pc","sod","hemo","pcv"]]
    y=df1[["classification"]]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=50)
    lm=RandomForestClassifier()
    lm.fit(X_train,y_train)
    predictions=lm.predict(X_test)
    np.array(y_test.values.flatten().tolist())
    if request.method == 'POST':
        [a,b,c,d,e,f]=request.form['sg'],request.form['rbc'],request.form['pc'],request.form['sod'],request.form['hemo'],request.form['pcv']
        pred=lm.predict([[a,b,c,d,e,f]])
        if pred[0]==0:
            res='Yes'
            x="YES!you have Kidney Disease!please consult with the Doctor"
        else:
            res='No'
            x='No!you dont have Kidney Disease'
        """rbc=request.form['rbc']
        pc=request.form['pc']
        if rbc==0:
            var1='normal'
        else:
            var1='abnormal'
        if pc==0:
            var2='normal'
        else:
            var2='abnormal'
        con = sql.connect("C:/Users/Abeg/Desktop/database/user.db")
        cur = con.cursor()
        cur.execute("INSERT INTO kidney (sg,rbc,pc,sod,hemo,pcv,result) VALUES (?,?,?,?,?,?,?)", (request.form['sg'],var1,var2,request.form['sod'],request.form['hemo'],request.form['pcv'],res))
        con.commit()
        con.close()"""
        return render_template('kidneyd.html',value=x)
        
   
        



@app.route('/heartattack_prediction',methods=['GET','POST'])
def heartattack_prediction():
    df=pd.read_csv("framingham.csv")
    df1=df.fillna(df.mean())
    fin=[]
    col=df1.columns.tolist()
    for i in col:
        fin.append(col[col.index(i)])
    df2=df1[fin]
    df3=df2.fillna(df2.mean())
    X=df3[["age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]]
    y=df3[["TenYearCHD"]]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=50)
    lm=DecisionTreeClassifier()
    lm.fit(X_train,y_train)
    predictions=lm.predict(X_test)
    pd.crosstab(np.array(y_test.values.flatten().tolist()),predictions,rownames=['True'],colnames=['predicted'],margins=True)
    if request.method == 'POST':
        [a,b,c,d,e,f,g,h,i,j,k,l,m,n]=request.form['age'],request.form['education'],request.form['currentSmoker'],request.form['cigsPerDay'],request.form['BPMeds'],request.form['prevalentStroke'],request.form['prevalentHyp'],request.form['diabetes'],request.form['totChol'],request.form['sysBP'],request.form['diaBP'],request.form['BMI'],request.form['heartRate'],request.form['glucose']
        pred=lm.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n]])
        if pred[0]==1:
            res='Yes'
            x="YES!you have a chance of heart-attack!please consult  with the Doctor"
        else:
            res='No'
            x='No!you dont have a chance of Heart Attack'
        """edu=request.form['education']
        cs=request.form['currentSmoker']
        bp=request.form['BPMeds']
        ps=request.form['prevalentStroke']
        ph=request.form['prevalentHyp']
        db=request.form['diabetes']
        #education
        if(edu==1):
            var1='Below College'
        elif(edu==2):
            var1='College'
        elif(edu==3):
            var1='Bachlor'
        else:
            var1='Master'
        #current smoker
        if(cs==1):
            var2='Yes'
        else:
            var2='No'
        #BPMEDS
        if(bp==1):
            var3='Yes'
        else:
            var3='No'
        #prevalentstroke
        if(ps==1):
            var4='Yes'
        else:
            var4='No'
        #prevalenthyp
        if(ph==1):
            var5='Yes'
        else:
            var5='No'
        #diabetes
        if(db==1):
            var6='Yes'
        else:
            var6='No'
        con = sql.connect("C:/Users/Abeg/Desktop/database/user.db")
        cur = con.cursor()
        cur.execute("INSERT INTO heartattack (Age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose,result) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (request.form['age'],var1,var2,request.form['cigsPerDay'],var3,var4,var5,var6,request.form['totChol'],request.form['sysBP'],request.form['diaBP'],request.form['BMI'],request.form['heartRate'],request.form['glucose'],res))
        con.commit()
        con.close()"""
        return render_template('heartattack_prediction.html',value=x)
#61	3.0	1	30.0	0.00000	0	1	0	225.0	150.0	95.0	28.58	65.0	103.000000	1
#63	1.0	0	0.0	0.00000	0	0	0	205.0	138.0	71.0	33.11	60.0	85.000000	1
if __name__ == "__main__":
    #app.secret_key = os.urandom(12)
    app.run(debug=True,host='127.0.0.1', port=5000)


