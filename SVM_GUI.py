import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import sched, time
from threading import Thread
import array as arr 
from PIL import Image, ImageFont, ImageDraw, ImageTk
from sklearn import datasets
from sklearn.datasets import make_blobs

root = Tk() 
root.title("SVM Göğüs Kanseri Risk Tahmin Uygulaması") 
root.maxsize(900, 600) 
root.config(bg="#008000") 

schedule = sched.scheduler(time.time, time.sleep)
PCA_df = pd.DataFrame()
data = pd.read_csv('data/data.csv', index_col=False)

left_frame = Frame(root, width=200, height=500, bg='grey')
left_frame.grid(row=0, column=0, padx=10, pady=5)
right_frame = Frame(root, width=650, height=400, bg='grey')
right_frame.grid(row=0, column=1, padx=10, pady=5)
pca = PCA(n_components=10)

def DrawImageBox(image):
    for widget in right_frame.winfo_children():
        widget.destroy()
    Label(right_frame, image=image).grid(row=0,column=0, padx=5, pady=5)

def float64_to_str(var):
    if type(var) is list:
        return str(var)[1:-1] 
    if type(var) is np.ndarray:
        try:
            return str(list(var[0]))[1:-1]
        except TypeError:
            return str(list(var))[1:-1] 
    return str(var) 

def SystemNotTrained():
        img = Image.new('RGBA' ,(650,400), 'white')
        textString = "SVM modeli eğite tıklayınız"
        font= ImageFont.truetype("arial.ttf",15)
        w,h= font.getsize(textString)
        draw = ImageDraw.Draw(img)
        draw.text(((650-w)/2,(400-h)/2), textString,font=font, fill='black')
        image = ImageTk.PhotoImage(img)
        schedule.enter(5, 1, DrawImageBox(image), ())
        schedule.run()

img = Image.new('RGBA' ,(650,400), 'white')
textString = "Hoş geldiniz...\nGöğüs kanseri ile ilgili tanı koymak için \nSVM modeli eğit butonuna tıklayınız"
font= ImageFont.truetype("arial.ttf",15)
w,h= font.getsize(textString)
draw = ImageDraw.Draw(img)
draw.text(((650-w)/2,(400-h)/2), textString,font=font, fill='black')
image = ImageTk.PhotoImage(img)
DrawImageBox(image)


Label(left_frame, text="Eren OKUR 21908613\nMEMYL 502 Ödev Çalışması").grid(row=0, column=0, padx=5, pady=5)
studentRawimage = Image.open("data/erenokur.jpg")
studentRawimage = studentRawimage.resize((150, 150), Image.ANTIALIAS)
studentImage = ImageTk.PhotoImage(studentRawimage)
Label(left_frame, image=studentImage).grid(row=1, column=0, padx=5, pady=5)



tool_bar = Frame(left_frame, width=180, height=185)
tool_bar.grid(row=2, column=0, padx=5, pady=5)


def TrainSVM():
    data = pd.read_csv('data/data.csv', index_col=False)
    data.drop('Unnamed: 0',axis=1, inplace=True)
    array = data.values
    givenDataValues = array[:,1:31]
    givenResultValues = array[:,0]
    le = LabelEncoder()
    givenResultValues = le.fit_transform(givenResultValues)
    X_train, X_test, y_train, y_test = train_test_split( givenDataValues, givenResultValues, test_size=0.25, random_state=7)
    scaler =StandardScaler()
    scalerTransformedData = scaler.fit_transform(givenDataValues)
    pca = PCA(n_components=10)
    fit = pca.fit(scalerTransformedData)
    X_pca = pca.transform(scalerTransformedData)
    TrainSVM.pcaExplainedVarianceRatio = pca.explained_variance_ratio_
    PCA_df['PCA_1'] = X_pca[:,0]
    PCA_df['PCA_2'] = X_pca[:,1]
    TrainSVM.clf = SVC(probability=True)
    TrainSVM.clf.fit(X_train, y_train)
    classifier_score = TrainSVM.clf.score(X_test, y_test)
    n_folds = 3
    cv_error = np.average(cross_val_score(SVC(), scalerTransformedData, givenResultValues, cv=n_folds))
    clf2 = make_pipeline(SelectKBest(f_regression, k=3),SVC(probability=True))
    scores = cross_val_score(clf2, scalerTransformedData, givenResultValues, cv=3)
    avg = (100*np.mean(scores), 100*np.std(scores)/np.sqrt(scores.shape[0]))
    TrainSVM.TrainedValues = X_train
    TrainSVM.TrainedResults = y_train
    y_pred = TrainSVM.clf.fit(X_train, y_train).predict(X_test)
    TrainSVM.ResultClassificationValues = metrics.confusion_matrix(y_test, y_pred)
    TrainSVM.testResultClassificationReport = classification_report(y_test, y_pred )
    img = Image.new('RGBA' ,(650,400), 'white')
    textString = "eğitim raporu \neğitim sonu ortalama score " +  float64_to_str( cv_error) + "\nortalama sonuç: "+ float64_to_str(avg[0])+ "\nhata payı: +- "+ float64_to_str(avg[1])
    font= ImageFont.truetype("arial.ttf",15)
    draw = ImageDraw.Draw(img)
    draw.text((100,150), textString,font=font, fill='black')
    image = ImageTk.PhotoImage(img)
    schedule.enter(5, 1, DrawImageBox(image), ())
    schedule.run()

def PrepareTestResults():
    if not PCA_df.empty:
        img = Image.new('RGBA' ,(650,400), 'white')
        textString = "eğitim raporu \n" + TrainSVM.testResultClassificationReport
        font= ImageFont.truetype("arial.ttf",15)
        draw = ImageDraw.Draw(img)
        draw.text((100,150), textString,font=font, fill='black')
        image = ImageTk.PhotoImage(img)
        schedule.enter(5, 1, DrawImageBox(image), ())
        schedule.run()
    else:
        SystemNotTrained()

def PrepareTestReport():
    if not PCA_df.empty:
        for widget in right_frame.winfo_children():
            widget.destroy()
        figure3 = plt.Figure(figsize=(5,4), dpi=100)
        ax3 = figure3.add_subplot(111)
        ax3.matshow(TrainSVM.ResultClassificationValues, cmap=plt.cm.Reds, alpha=0.3)
        scatter3 = FigureCanvasTkAgg(figure3, right_frame) 
        scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        for i in range(TrainSVM.ResultClassificationValues.shape[0]):
         for j in range(TrainSVM.ResultClassificationValues.shape[1]):
             ax3.text(x=j, y=i,
                    s=TrainSVM.ResultClassificationValues[i, j], 
                    va='center', ha='center')
        ax3.set_xlabel('gerçek değerler')
        ax3.set_ylabel('tahmin edilmiş değerler')
        ax3.set_title('Sayısal test raporu')
    else:
        SystemNotTrained()

def PreparePCMReport():
    if not PCA_df.empty:
        for widget in right_frame.winfo_children():
            widget.destroy()
        figure3 = plt.Figure(figsize=(5,4), dpi=100)
        ax3 = figure3.add_subplot(111)
        ax3.plot(PCA_df['PCA_1'][data.diagnosis == 'M'],PCA_df['PCA_2'][data.diagnosis == 'M'],'o', alpha = 0.7, color = 'r')
        ax3.plot(PCA_df['PCA_1'][data.diagnosis == 'B'],PCA_df['PCA_2'][data.diagnosis == 'B'],'o', alpha = 0.7, color = 'b')
        scatter3 = FigureCanvasTkAgg(figure3, right_frame) 
        scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        ax3.legend(['risk yüksek' , 'risk düşük']) 
        ax3.set_xlabel('düşük riskli PCA')
        ax3.set_ylabel('yüksek riskli PCA')
        ax3.set_title('Temel Birleşen Analizi (PCA)')
    else:
        SystemNotTrained()


def PrepareEigenReport():
    if not PCA_df.empty:
        for widget in right_frame.winfo_children():
            widget.destroy()
        figure3 = plt.Figure(figsize=(5,4), dpi=100)
        ax3 = figure3.add_subplot(111)
        ax3.plot(TrainSVM.pcaExplainedVarianceRatio)
        scatter3 = FigureCanvasTkAgg(figure3, right_frame) 
        scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        ax3.legend(['PCA ile elde dilmiş özvektör'], loc='best', borderpad=0.3,shadow=False,markerscale=0.4)
        ax3.set_xlabel('Özvektör')
        ax3.set_ylabel('Ana birleşenler')
        ax3.set_title('Öz Vektör Dağılımı')
    else:
        SystemNotTrained()

def PredictGivenValue():
    if not PCA_df.empty:
        file = fd.askopenfile()
        if file: 
           Testdata = pd.read_csv(file.name, index_col=False)
           Testdata.drop('Unnamed: 0',axis=1, inplace=True)
           array = Testdata.values
           givenDataValues = array[:,1:31]
           asked_pred = TrainSVM.clf.fit(TrainSVM.TrainedValues, TrainSVM.TrainedResults).predict(givenDataValues)
           x = len(asked_pred)
           my_string = "Test sonuçları:\n"
           if x <= 5:
               img = Image.new('RGBA' ,(650,400), 'white')
               font= ImageFont.truetype("arial.ttf",15)
               draw = ImageDraw.Draw(img)
               counter = 1
               for predictedValues in asked_pred:   
                   if predictedValues == 1:
                        textString = my_string + str(counter) + " numaralı test yüksek riskli  \n"
                   else:
                        textString = my_string + str(counter) + " numaralı test düşük riskli \n"      
                   counter = counter + 1 
               draw.text((100,150), textString,font=font, fill='black')
               image = ImageTk.PhotoImage(img)
               schedule.enter(5, 1, DrawImageBox(image), ())
               schedule.run()
           else:
                for widget in right_frame.winfo_children():
                    widget.destroy()
                figure3 = plt.Figure(figsize=(5,4), dpi=100)
                ax3 = figure3.add_subplot(111)
                unique, counts = np.unique(asked_pred, return_counts=True)
                test_dict = dict(zip(unique, counts))
                ax3.bar(test_dict.keys(), test_dict.values(), width=0.5, color='g')
                scatter3 = FigureCanvasTkAgg(figure3, right_frame) 
                scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
                ax3.set_xlabel('sayı')
                ax3.set_ylabel('sonuç')
                ax3.set_title('çoklu test sonuçları (1 yüksek, 0 düşük risk)')
    else:
        SystemNotTrained()

def Train():
    TrainSVM()

def Test():
    PrepareTestResults()

def TestReport():
    PrepareTestReport()

def PCMDistShow():
    PreparePCMReport()

def PCMEigenDistShow():
    PrepareEigenReport()

def Predict():
    PredictGivenValue()

Button(tool_bar, text="SVM modeli eğit",command=Train,bg='brown',fg='white').grid(row=1, column=0, padx=5, pady=5)
Button(tool_bar, text="SVM modeli eğitim testi başlat",command=Test,bg='brown',fg='white').grid(row=2, column=0, padx=5, pady=5)
Button(tool_bar, text="SVM modeli eğitim testi raporu hazırla",command=TestReport,bg='brown',fg='white').grid(row=3, column=0, padx=5, pady=5)
Button(tool_bar, text="SVM modeli PCM dağılımı göster",command=PCMDistShow,bg='brown',fg='white').grid(row=4, column=0, padx=5, pady=5)
Button(tool_bar, text="SVM modeli PCM öz vektör dağılımı",command=PCMEigenDistShow,bg='brown',fg='white').grid(row=5, column=0, padx=5, pady=5)
Button(tool_bar, text="test datası ile sonuç bul",command=Predict,bg='brown',fg='white').grid(row=6, column=0, padx=5, pady=5)

root.mainloop()
