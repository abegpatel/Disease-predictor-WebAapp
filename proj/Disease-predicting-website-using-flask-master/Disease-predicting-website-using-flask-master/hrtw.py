from flask import Flask, render_template,request,url_for,redirect,session,logging,Blueprint,flash
import sqlite3 as sql
import os
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    if request.method == 'POST':
        return redirect(url_for('index'))
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
@app.route('/adderc',methods=['GET','POST'])
def adderc():
    if request.method == 'POST':
        try:
            username= request.form['username']
            password= request.form['password']
            email= request.form['email']
            add= request.form['address']
            pin= request.form['pin']
         
            with sql.connect("C:/Users/ASUS/Desktop/userdtb.db") as con:
                cur = con.cursor()
            
                cur.execute("INSERT INTO signupuser(username,password,email,address,pin)VALUES (?,?,?,?,?)",(username,password,email,add,pin))
            
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"
      
        finally:
            return render_template('index.html')
            con.close() 
@app.route('/loggedin',methods=['GET','POST'])
def loggedin():
    if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if not (username and password):
                flash("Username or Password cannot be empty.")
                return redirect(url_for('login'))
            else:
                if user.password == password:
                    return redirect(url_for('index'))
                else:
                    return render_template('login.html')
@app.route('/dia_pred',methods=['GET','POST'])
def dia_pred():
    df=pd.read_csv("C:/Users/ASUS/Downloads/diabetes.csv")
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
            x="you have diabetes!!please consult with your doctor"
        else:
            x="congratulation!!you have no diabetes"
        return render_template('diabetes.html',value=x)
@app.route('/heartattack_prediction',methods=['GET','POST'])
def heartattack_prediction():
    df=pd.read_csv("C:/Users/ASUS/Downloads/heart-disease-prediction-using-logistic-regression/framingham.csv")
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
        [a,b,c,d,e,f,g,h,i,j,k,l,m,n]=request.form['age'],request.form['education'],request.form['currentSmoker'],request.form['cigsPerDay'],request.form['BPMeds'],request.form['prevalentStroke'],request.form['prevalentHyp'],request.form['totChol'],request.form['sysBP'],request.form['diaBP'],request.form['BMI'],request.form['heartRate'],request.form['glucose']
        pred=lm.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n]])
        if pred[0]==1:
            x="you have diabetes!!please consult with your doctor"
        else:
            x="congratulation!!you have no diabetes"
        return render_template('diabetes.html',value=x)
#61	3.0	1	30.0	0.00000	0	1	0	225.0	150.0	95.0	28.58	65.0	103.000000	1
#63	1.0	0	0.0	0.00000	0	0	0	205.0	138.0	71.0	33.11	60.0	85.000000	1
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='127.0.0.1', port=5000)


