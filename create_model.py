import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_model_from_scratch():
    st.header('Build Model')
    uploaded_file = st.file_uploader("Choose input file to train model:")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).reset_index(drop=True)
        Y = df['pIC50']
        X = df.drop('pIC50', axis=1)

        #Split Data into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        #Build Regression Model
        np.random.seed(100)

        st.subheader("Train Model and Show Metric Results:")
        model = RandomForestRegressor(n_estimators=200)
        model.fit(X_train, Y_train)
        #r2 = model.score(X_test, Y_test)
        #st.write("The R2 result is: " + str(r2))
        Y_pred = model.predict(X_test)
        metrics = {'R2': [model.score(X_test, Y_test)],
                    'MAE': [mean_absolute_error(Y_test, Y_pred)],
                    'MSE': [mean_squared_error(Y_test, Y_pred)]}
        st.dataframe(pd.DataFrame(metrics))

        # Download model button
        model_final = RandomForestRegressor(n_estimators=200)
        model_final.fit(X, Y)

        st.subheader('Click button to download the Random Forest Regressor model:')
        st.download_button("Download Model", data=pickle.dumps(model_final), file_name='model_random_forest.pkl', key='download-model') #It saves it in Downloads

        st.subheader('Scatter Plot of Real vs Predicted pIC50 Values')
        fig, ax = plt.subplots()
        sns.set(color_codes=True)
        sns.set_style("white")
        ax = sns.regplot(x=Y_test, y=Y_pred)
        ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
        ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
        #ax.figure.set_size_inches(5, 5)
        st.pyplot(fig)

def upload_pretrained_model():
    st.header("Get predictions using a pre-trained model")
    st.subheader("Upload trained model")
    uploaded_model = st.file_uploader("Upload Trained Model:")
    if uploaded_model is not None:
        loaded_model = None
        try:
            loaded_model = pickle.load(uploaded_model)
        except:
            load_model_error = st.error("Incorrect file, try again!")
        
        if loaded_model is not None:
            
            #Import file input file, what you want to have predictions:
            uploaded_input_file = st.file_uploader("Upload Model to be Predicted:")
            if uploaded_input_file is not None:
                df = pd.read_csv(uploaded_input_file)
                #df = df.head(5)
                #Drop NaN values
                df = df.dropna().reset_index(drop=True)
                st.dataframe(data=df)

                #Get fingerprints of the input file
                smiles = df['canonical_smiles'].to_list()
                fps = []
                for s in smiles:
                    mol = Chem.MolFromSmiles(str(s))
                    fp = Chem.RDKFingerprint(mol)
                    arr = np.zeros((0,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fps.append(arr)
                    
                X = pd.DataFrame(fps)
                X = X[list(map(int, loaded_model.feature_names_in_))]
                st.dataframe(data=X)

                Y_pred = loaded_model.predict(X)
                df['pred_pIC50'] = Y_pred
                st.dataframe(data=df)

                # Download file button
                csv_5 = df.to_csv(index=False).encode('utf-8')
                name_5 = uploaded_input_file.name.replace('.csv', '') + '_predictions.csv'
                st.write(name_5)
                st.download_button("Download CSV", csv_5, name_5, key='download-csv_5') #It saves it in Downloads