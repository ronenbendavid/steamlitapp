import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import category_encoders as ce
import os
import base64
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pdfrw import PdfReader
from pdfrw.findobjs import page_per_xobj
import eli5
from eli5.sklearn import PermutationImportance

def main():
    st.title("RIP AI for Auto settings ")    
    st.markdown("Select RIP Optimization based on file characterastics.")
    #a = ['info.Creator', 'info.Producer', str(1), 'product', 'PDF',1]
    if 'history_key' not in st.session_state:
        st.session_state['history_key'] = []
    if 'csv_key' not in st.session_state:
        st.session_state['csv_key'] = 'ripai_history.csv'
    if 'columns_key' not in st.session_state:
        st.session_state['columns_key'] = ['creator', 'producer', 'pages', 'product', 'type','label','predict']
    st.sidebar.title("RIP AI")
    st.sidebar.markdown("Welcome to RIP AI selection!")
    
    @st.cache(persist = True)
    def load_data_new(data):
        # label_names = ['0-Not Optimize','1-Optimize']
        #data = pd.read_csv("./steamlitapp/labeldataset2.csv")
        train_cols = data.columns[1:-1]
        label = data.columns[-1]
        X = data[train_cols]
        y = data[label]
        print("data is loaded! ")
        return X,y
    @st.cache(persist = True)
    def split_new(X,y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        encoded_model = ce.leave_one_out.LeaveOneOutEncoder().fit(x_train,y_train)
        X_train = encoded_model.transform(x_train,y_train)
        X_test = encoded_model.transform(x_test,y_test)
        return  X_train, X_test, y_train, y_test
    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()

        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        
        return data

    @st.cache(persist = True)
    def split(df):
        y = df.type
        x = df.drop(columns = ['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

        return x_train, x_test, y_train, y_test

    @st.cache(suppress_st_warning=True)
    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()


    def inferenceOneJob(X,y,info,num_pages,product,model):
            # model.fit(x_train, y_train)
            cx_test = np.array([info.Creator, info.Producer, str(num_pages), product, 'PDF'])
            pd_cx_test = pd.DataFrame(cx_test.reshape((1,5)),columns = ['creator', 'producer', 'pages', 'product', 'type'])
            Y_test = pd.DataFrame(np.array([1]),columns = ['label'])

            encoded_model = ce.leave_one_out.LeaveOneOutEncoder().fit(X,y)
            ex_test = encoded_model.transform(pd_cx_test)
            # ex_test = StandardScaler().transform(ex_test)
            #st.write('printing encodex X')
            #st.write(ex_test)
            y_predict = model.predict(ex_test)
            st.write(f'Optimization Results for file: {pdffilename.name} Type {pdffilename.type} Size {pdffilename.size} is: {y_predict}')
            label = st.session_state['label']
            y_predict = y_predict[0]
            line = [info.Creator, info.Producer, str(num_pages), product, 'PDF',label,y_predict]
            st.write(pd.DataFrame(np.array(line).reshape((1,7)),columns = st.session_state['columns_key']))
            # st.write(f"Adding line {line} to {st.session_state['history_key']}")
            st.session_state['history_key'].append(line)
            if (label == y_predict):
                return True;
            else:
                return False


    def extarct_pdf_info(pdffilename):
        pdf = PdfReader(pdffilename)
        info = pdf.Info
    #     print(info)
        num_pages = len(pdf.pages)
        #print(num_pages)
        #print(info.Creator)
        return info,num_pages


    def importance(x_test, y_test):
        perm = PermutationImportance(model).fit(x_test, y_test, groups=['creator', 'producer', 'pages', 'product', 'type'])
        perm.fit(x_test, y_test)
        weights = eli5.show_weights(perm)
        result = pd.read_html(weights.data)[0]
        st.write(result)
    csvfile = st.file_uploader("Upload csv file",type=['csv'])
    if csvfile:
        df = pd.read_csv(csvfile)
        st.write(f'name:{csvfile.name}')
        print(df.head(10));
    else:
        dummy = np.array(['creator', 'producer', 1, 'product', 'pdf',0])
        df = pd.DataFrame(dummy.reshape((1,6)),columns = ['creator', 'producer', 'pages', 'product', 'type','label'])
        df = pd.read_csv("./labeldataset2.csv")
        print(df.head(10));

    check = st.sidebar.checkbox("Check if you know file need optimization", False)
    if check:
        st.session_state['label'] = 1
    else:
        st.session_state['label'] = 0
    X,y = load_data_new(df)
    x_train, x_test, y_train, y_test = split_new(X,y)
    # df = load_data()
    # x_train, x_test, y_train, y_test = split(df)
    class_names = ['Not optimized', 'Optimize']
    st.sidebar.subheader("Choose My Classifier")
    pdffilename = st.file_uploader("Upload PDF file",type=['pdf'])
    product = st.sidebar.radio("Product", ("Commercial", "L&P"), key = 'product')
    if pdffilename:
        info,num_pages = extarct_pdf_info(pdffilename)
        st.write(f'name:{pdffilename.name}, creator: {info.Creator} ,producer:{info.Producer},pages: {num_pages}')

        filetype = pdffilename.type
        st.write(f'Type of file is {pdffilename.type}')        
    # classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine(SVM)", "LogisticRegression", "Random Forest","XGBoost","CatBoost"))
    classifier = st.sidebar.selectbox("Classifier", ("CatBoost","Random Forest"))
    if classifier == "Support Vector Machine(SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key = 'auto')
    
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        model = SVC(C = C, kernel = kernel, gamma = gamma)
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)
        if st.sidebar.button("Importance", key = 'importance'):
            model.fit(x_train, y_train)
            st.write("Importance by Support Vector Machine(SVM) Classifier")
            importance(x_test, y_test)
        if st.sidebar.button("Predict", key = 'predict'):
            model.fit(x_train, y_train)
            inferenceOneJob(X,y,info,num_pages,product,model)
    if classifier == "LogisticRegression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter') 
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        model = LogisticRegression(C = C, max_iter = max_iter)
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)          
        if st.sidebar.button("Importance", key = 'importance'):
            model.fit(x_train, y_train)
            st.write("Importance by LogisticRegression Classifier")
            importance(x_test, y_test)
        if st.sidebar.button("Predict", key = 'predict'):
            model.fit(x_train, y_train)
            inferenceOneJob(X,y,info,num_pages,product,model)
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")        
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Random Forest Results")
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)
        if st.sidebar.button("Importance", key = 'importance'):
            model.fit(x_train, y_train)
            st.write("Importance by Random Forest Classifier")
            importance(x_test, y_test)
        if st.sidebar.button("Predict", key = 'predict'):
            model.fit(x_train, y_train)
            inferenceOneJob(X,y,info,num_pages,product,model)
    if classifier == "XGBoost":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in XGBoost", 100, 5000, step = 10, key = 'n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        model =  XGBClassifier(random_state=1,bootstrap=False, class_weight= 'balanced', criterion= 'gini', \
                                max_depth= max_depth, max_features= 'auto', min_samples_leaf= 10, min_samples_split= 40,\
                                n_estimators= n_estimators)
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("XGBoost Results")       
            try:
                model.fit(x_train, y_train)
                
            except AttributeError as e:
                 st.write(f"caught an exception {e}: ")
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            #plot_metrics(metrics)
        if st.sidebar.button("Importance", key = 'importance'):
            model.fit(x_train, y_train)
            st.write("Importance by XGBoost Classifier")
            importance(x_test, y_test)
        if st.sidebar.button("Predict", key = 'predict'):
            model.fit(x_train, y_train)
            predict_succesfuly = inferenceOneJob(X,y,info,num_pages,product,model)
            if predict_succesfuly:
                st.write(":smile:" * 3)
            else:
                st.write(":cake:" * 3)
            df = pd.DataFrame(st.session_state['history_key'],columns=st.session_state['columns_key'])
            #if isinstance(df, pd.DataFrame):
            #    objecttodwl = df.to_csv(index=False)
            #else:
            #    objecttodwl = json.dumps(df)
            objecttodwl = df.to_csv(index=False)
            try:
                # some strings <-> bytes conversions necessary here
                b64 = base64.b64encode(objecttodwl.encode()).decode()

            except AttributeError as e:
                b64 = base64.b64encode(objecttodwl).decode()
            dl_link = f"""
                        <html>
                        <head>
                        <title>Start Auto Download file</title>
                        <script src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
                        <script>
                        $('<a href="data:text/csv;base64,{b64}" download="{st.session_state['csv_key']}">')[0].click()
                        </script>
                        </head>
                        </html>
                        """
            st.download_button(
                label="Download data as CSV",
                data=b64,
                file_name= st.session_state['csv_key'],
                mime='text/csv',
            )
    if classifier == "CatBoost":
        st.sidebar.subheader("Model Hyperparameters")
        # learning_rate = st.sidebar.number_input("learning_rate", 100, 5000, step = 10, key = 'n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter')
        learning_rate = st.sidebar.slider("learning_rate", 0.1, 1.0, key = 'learning_rate')
        # bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        model =  CatBoostClassifier(iterations=max_iter,learning_rate=learning_rate,depth=max_depth)
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("CatBoost Results")
            # model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)
        if st.sidebar.button("Predict", key = 'predict'):
            model.fit(x_train, y_train)
            predict_succesfuly = inferenceOneJob(X,y,info,num_pages,product,model)
            if predict_succesfuly:
                st.write(":smile:" * 3)
            else:
                st.write(":cake:" * 3)
            df = pd.DataFrame(st.session_state['history_key'],columns=st.session_state['columns_key'])
            #if isinstance(df, pd.DataFrame):
            #    objecttodwl = df.to_csv(index=False)
            #else:
            #    objecttodwl = json.dumps(df)
            objecttodwl = df.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=objecttodwl,
                file_name= st.session_state['csv_key'],
                mime='text/csv',
            )
              
        if st.sidebar.button("Importance", key = 'importance'):
            model.fit(x_train, y_train)
            st.write("Importance by CatBoost Classifier")
            importance(x_test, y_test)


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("RIP Data Set (Classification)")
        st.write(df)
    if st.sidebar.button("Save records", key='save'):
        st.write(f"Writing prediction history, total of {len(st.session_state['history_key'])} items")
        # st.write(st.session_state['history_key'])
        dfh = pd.DataFrame(st.session_state['history_key'],columns=st.session_state['columns_key'])
        st.write(f"Saving data table. During this session we collected {dfh.shape[0]} elements.")
        # historycsv = st.file_uploader("Upload History csv file", type=['csv'])
        # if historycsv:
        #     dfho = pd.read_csv(historycsv)
        # else:
        #     dfho = pd.read_csv("./ripai_history.csv")

        historycsv = dfh.to_csv().encode('utf-8')
        csvname = st.session_state['csv_key']
        st.download_button(
            label="Download data as CSV",
            data=historycsv,
            file_name= csvname,
            mime='text/csv',
        )
if __name__ == '__main__':
    main()
