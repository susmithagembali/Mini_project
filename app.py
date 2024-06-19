import streamlit as st
#data handling
import pandas as pd
import numpy as np
import seaborn as sns

#data visualization
import matplotlib.pyplot as plt

#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler

#feature selection
from sklearn.feature_selection import mutual_info_classif

#classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
#svm
from sklearn.svm import SVC

st.title("Analyses and Classification of Cancer Patients based on DNA Sequences")
html_template = """
    <div style="background-color:cyan;padding:10px">
    <h2 style="color:black;text-align:center;">Explore Different Classifiers</h2>
    </div>
"""
st.markdown(html_template,unsafe_allow_html=True)
classifier_name=st.sidebar.selectbox("select classifier",("random forest","svm","knn","heirarchial clustering"))

uploadbtn = st.button("Upload")

if "upload_btn" not in st.session_state:
    st.session_state.upload_btn = False

if uploadbtn or st.session_state.upload_btn:
    st.session_state.upload_btn = True

    f=st.file_uploader("Upload csv",type=["csv"])
    print(f)
    if "uploadbtn" is not None:
        with st.spinner("Please Wait ..."):
            dataframe=pd.read_csv(f)
            X=dataframe.iloc[:,2:-1]
            y=dataframe.iloc[:,-1]
            
            label_encoder=LabelEncoder()
            label_encoder.fit(y)
            y_encoded=label_encoder.transform(y)
            labels=label_encoder.classes_
            classes=np.unique(y_encoded)
            
            X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,random_state=42)
            
            min_max_scaler=MinMaxScaler()
            
            X_train_norm=min_max_scaler.fit_transform(X_train)
            
            X_test_norm=min_max_scaler.fit_transform(X_test)
            
            MI=mutual_info_classif(X_train_norm,y_train)
            
            n_features=300
            
            selected_scores_indices=np.argsort(MI)[::-1][0:n_features]
            
            X_train_selected=X_train_norm[:,selected_scores_indices]
            
            X_test_selected=X_test_norm[:,selected_scores_indices]
            
            RF=OneVsRestClassifier(RandomForestClassifier(max_features=0.2))
            
            RF.fit(X_train_selected,y_train)
            
            y_pred =RF.predict(X_test_selected)
            
            pred_prob = RF.predict_proba(X_test_selected)
            
            accuracy=np.round(balanced_accuracy_score(y_test,y_pred),4)

            precision=np.round(precision_score(y_test,y_pred,average = 'weighted'),4)

            recall=np.round(recall_score(y_test,y_pred,average = 'weighted'),4)

            f1score=np.round(f1_score(y_test,y_pred,average = 'weighted'),4)

            st.text("accuracy")
            st.text(accuracy)
            st.text("precision")
            st.text(precision)
            st.text("recall")
            st.text(recall)
            st.text("f1score")
            st.text(f1score)
            cm=metrics.confusion_matrix(y_test,y_pred)
            cm_df=pd.DataFrame(cm,index=labels,columns=labels)
            sns.heatmap(cm_df,annot=True,cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
   
         

                    
                
                

            



            