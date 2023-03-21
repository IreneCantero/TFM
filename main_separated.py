# Import libraries
import pandas as pd
import streamlit as st
from intro import intro
from get_data_preprocessing import obtain_and_preprocess_data
from decriptor_dataset import get_fingerprint_descriptors
from create_model import create_model_from_scratch, upload_pretrained_model

#Returns for every canonical_smile the final merged score
def bio_score(df: pd.DataFrame, model_type: str):
    final_scores = pd.DataFrame()
    canonical_smiles = []
    merged_scores = []
    fs = {}
    for cs in df['canonical_smiles'].unique():
        df_aux = df[df['canonical_smiles'] == cs]
        for row in df_aux.iterrows():
            row = row[1]
            nc = row['Number of Cancers']
            pb = row['bioactivity_prediction']
            if model_type == 'Classification':
                min_pb = int(list(pb)[1])
                max_pb = int(list(pb)[4])
                pb = np.mean([min_pb, max_pb])
            score = 0.8*pb + 0.2*nc
            if cs not in fs.keys():
                fs[cs] = score
            else:
                fs[cs] += score

        canonical_smiles.append(cs)
        merged_scores.append(fs[cs])
    final_scores['canonical_smiles'] = canonical_smiles
    final_scores['merged_score'] = merged_scores
    return final_scores

def mc_accuracy(y_true:list, y_pred:list):
    correct_class = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_class.append(1)
        else:
            correct_class.append(0)
    accuracy = np.sum(correct_class)/len(correct_class)
    return accuracy

    
if __name__ == "__main__":

    st.set_page_config(
            page_title="ML for Drug Discovery",
            page_icon="https://upload.wikimedia.org/wikipedia/commons/4/41/Caffeine_Molecule.png",
            #layout="wide",
            #initial_sidebar_state="expanded",
        )

    #current_path = os.getcwd()
    pd.set_option('display.max_columns', None)

    sidebar = st.sidebar
    with sidebar:
        st.header("ML for Drug Discovery")
        part_selectbox = st.selectbox("Select Project Part:", ("Part 1", "Part 2"))

    if part_selectbox == 'Part 1':

        input_selectbox = st.sidebar.selectbox(
            "Step Selection", ["Obtain and Preprocess Data", "Create Descriptor Dataset", "Create Model"]
        )

        #if input_selectbox == "Intro":
        #    intro()

        if input_selectbox == "Obtain and Preprocess Data":
            obtain_and_preprocess_data()
        
        elif input_selectbox == "Create Descriptor Dataset":
            get_fingerprint_descriptors()

        elif input_selectbox == "Create Model":
            #st.header('Create Model')
            option = st.selectbox("Do you have a model to upload?", ['No', 'Yes'])
            if option == 'No':
                create_model_from_scratch()
                    
            elif option == 'Yes':
                upload_pretrained_model()
    
    elif part_selectbox == 'Part 2':
        #input_selectbox_2 = st.sidebar.selectbox(
        #    "Step Selection", ["Intro", "Select Proteins"]
        #)

        #if input_selectbox_2 == "Intro":
        #    intro()

        #elif input_selectbox_2 == "Select Proteins":
        from chembl_webresource_client.new_client import new_client
        from rdkit import Chem
        from rdkit.Chem import DataStructs
        import random
        import numpy as np
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.preprocessing import LabelEncoder
        import seaborn as sns
        import matplotlib.pyplot as plt
        from get_data_preprocessing import get_pIC50

        #Hacer que se puedan escoger los cancers!!

        #Poner explanation en el streamlit sobre los acronimos!!!!!
        cancers_dict = {'BRC': 'breast cancer', 'OC': 'ovarian cancer', 'LC': 'lung cancer', 'LK': 'leukemia'}

        df_cancer_target = pd.DataFrame() #THIS OK
        for cancer_type in cancers_dict:
            df_aux = new_client.target.search(cancers_dict[cancer_type]).filter(target_type = 'SINGLE PROTEIN', organism = 'Homo sapiens')
            df_aux = pd.DataFrame.from_dict(df_aux)
            df_aux['Cancer Type'] = cancer_type
            df_cancer_target = pd.concat([df_cancer_target, df_aux], ignore_index=True, axis=0)
            
        st.dataframe(df_cancer_target) #TAKE OUT!!!!!!!
        
        cols = ['Cancer Type', 'pref_name', 'target_chembl_id']
        df_cancer_info = df_cancer_target[cols]
        #Se tendría que hacer también por UNIPROT IDº
        df_cancer_info['Cancer Types'] = df_cancer_info.groupby(['pref_name', 'target_chembl_id'], as_index=False)['Cancer Type'].transform(lambda x: ','.join(x))
        df_cancer_info = df_cancer_info.groupby(['pref_name', 'target_chembl_id', 'Cancer Types'], as_index=False).aggregate({'Cancer Type': "count"})
        df_cancer_info = df_cancer_info.rename(columns={'Cancer Type': 'Number of Cancers'})
        #st.dataframe(df_cancer_info)     

        # Select proteins
        proteins = st.multiselect('Select the proteins:', np.sort(list(df_cancer_target['pref_name'].unique())))
        
        if proteins:
            df_cancer_activity = pd.DataFrame()
            for protein in proteins:
                chembl_id = df_cancer_target[df_cancer_target['pref_name'] == protein]['target_chembl_id'].iloc[0]
                df_aux = new_client.activity.filter(target_chembl_id=chembl_id, standard_type="IC50")
                df_aux = pd.DataFrame.from_dict(df_aux)
                df_cancer_activity = pd.concat([df_cancer_activity, df_aux], ignore_index=True, axis=0)

            #Hacer bioactivity_class y eliminar intermedios?
            #Pasar a pIC50??
            cols = ['target_pref_name','assay_description','molecule_chembl_id','canonical_smiles','standard_value']
            df_cancer_activity = df_cancer_activity[cols]
            df_cancer_activity = df_cancer_activity.dropna(subset=['canonical_smiles', 'standard_value']).reset_index(drop=True)
            df_cancer_activity = df_cancer_activity.rename(columns={"standard_value": "IC50"})
            st.dataframe(df_cancer_activity)

            # Download file button
            csv_1 = df_cancer_activity.to_csv(index=False).encode('utf-8')
            name_1 = 'selected_target_activities_part_2.csv'
            st.download_button("Download CSV", csv_1, name_1, key='download-csv_1') #It saves it in Downloads

            #Obtain the fingerprints from the canonical smiles
            smiles = df_cancer_activity['canonical_smiles'].to_list()
            fps = []
            for s in smiles:
                mol = Chem.MolFromSmiles(str(s))
                fp = Chem.RDKFingerprint(mol)
                arr = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
                
            df_fp = pd.DataFrame(fps)
            # Remove non-explanatory columns
            selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
            descriptors_df = pd.concat([df_cancer_activity, pd.DataFrame(selection.fit_transform(df_fp))], axis=1)
            descriptors_df['IC50'] = descriptors_df['IC50'].astype('float')
            st.write("Display of the resulting dataframe after computing the fingerprints from the canonical smiles and removing non-explanatory columns:")
            st.dataframe(data=descriptors_df)

            # Download file button
            csv_2 = descriptors_df.to_csv(index=False).encode('utf-8')
            name_2 = 'selected_target_activities_figerprint_descriptors_part_2.csv'
            st.download_button("Download CSV", csv_2, name_2, key='download-csv_2') #It saves it in Downloads

            st.subheader("Train Model and Show Metrics Results:")

            #ax = sns.displot(descriptors_df, x = 'IC50', hue="target_pref_name", multiple="stack")
            #st.pyplot(ax)

            model_type = st.selectbox('Select model type:', ('Multi-Regression', 'Separated Linear Regressions', 'Classification'))

            if model_type == 'Multi-Regression':
                
                val_to_predict = st.radio('Which metric do you wanna use?', ('IC50', 'pIC50'))

                if val_to_predict == 'IC50':
                    y = descriptors_df['IC50']
                else:
                    y = get_pIC50(descriptors_df['IC50'].to_list())
                x = descriptors_df.copy()
                le = LabelEncoder()
                le.fit(x['target_pref_name'].unique())
                x['target_pref_name_labeled'] = le.transform(x['target_pref_name'])
                #x['target_pref_name_labeled'] = x['target_pref_name_labeled'].astype(int)
                x.columns = x.columns.astype(str)
                x = x.drop(columns=['target_pref_name', 'assay_description', 'molecule_chembl_id', 'canonical_smiles', 'IC50'])
                
                #Split Data into train and test
                X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

                #Build Multi-Regression Model
                np.random.seed(100)
                model = RandomForestRegressor(n_estimators=200)
                model.fit(X_train, Y_train)
                #r2 = model.score(X_test, Y_test)
                #st.write("The R2 result is: " + str(r2))
                Y_pred = model.predict(X_test)        
                metrics = {'R2': [round(model.score(X_test, Y_test),3)],
                            'MAE': [round(mean_absolute_error(Y_test, Y_pred),3)],
                            'MSE': [round(mean_squared_error(Y_test, Y_pred),3)]}
                st.dataframe(pd.DataFrame(metrics))

                # Plot result
                st.subheader('Scatter Plot of Real vs Predicted ' + val_to_predict + ' Values')
                fig, ax = plt.subplots()
                sns.set(color_codes=True)
                sns.set_style("white")
                ax = sns.regplot(x=Y_test, y=Y_pred)
                ax.set_xlabel('Experimental ' + val_to_predict, fontsize=10, fontweight='bold')
                ax.set_ylabel('Predicted ' + val_to_predict, fontsize=10, fontweight='bold')
                ax.figure.set_size_inches(5, 4)
                st.pyplot(fig)

            elif model_type == 'Separated Linear Regressions':

                val_to_predict = st.radio('Which metric do you wanna use?', ('IC50', 'pIC50'))

                lr_models = {}
                lr_preds = {}
                metrics_list = []
                for protein in descriptors_df['target_pref_name'].unique():
                    df_aux = descriptors_df[descriptors_df['target_pref_name']==protein].reset_index(drop=True)

                    if val_to_predict == 'IC50':
                        y = df_aux['IC50']
                    else:
                        y = get_pIC50(df_aux['IC50'].to_list())

                    x = df_aux.copy()
                    x = x.drop(columns=['target_pref_name', 'assay_description', 'molecule_chembl_id', 'canonical_smiles', 'IC50'])
                    
                    #Split Data into train and test
                    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

                    #Build Multi-Regression Model
                    np.random.seed(100)
                    model = RandomForestRegressor(n_estimators=200)
                    model.fit(X_train, Y_train)
                    #r2 = model.score(X_test, Y_test)
                    #st.write("The R2 result for " + protein + " is: " + str(r2))
                    Y_pred = model.predict(X_test)
                    metrics = {'Target': protein,
                            'R2': [round(model.score(X_test, Y_test),3)],
                            'MAE': [round(mean_absolute_error(Y_test, Y_pred),3)],
                            'MSE': [round(mean_squared_error(Y_test, Y_pred),3)]}
                    metrics_list.append(metrics)

                    lr_models[protein] = model
                    lr_preds[protein] = {'Y_test': Y_test, 'Y_pred': Y_pred}
                
                st.dataframe(pd.DataFrame.from_dict(metrics_list))

                # Plot result
                st.subheader('Scatter Plot of Real vs Predicted ' + val_to_predict + ' Values')
                fig, ax = plt.subplots()
                for idx, x in enumerate(lr_preds.keys()):
                    sns.regplot(x = lr_preds[x]['Y_test'], y = lr_preds[x]['Y_pred'], label=x)
                ax.set_xlabel('Experimental ' + val_to_predict, fontsize=10, fontweight='bold')
                ax.set_ylabel('Predicted ' + val_to_predict, fontsize=10, fontweight='bold')
                ax.legend(prop={'size': 6})
                ax.figure.set_size_inches(5, 4)
                st.pyplot(fig)

            elif model_type == 'Classification':
                
                from sklearn.metrics import confusion_matrix
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import precision_score, recall_score, f1_score

                st.write("For this Random Forest Classification model we will be using the metric pIC50.")
                y_aux = get_pIC50(descriptors_df['IC50'].to_list())
                y = []
                for val in y_aux:
                    s = '[' + str(int(val)) + ', ' + str(int(val)+1) + ')'
                    y.append(s)

                x = descriptors_df.copy()
                le = LabelEncoder()
                le.fit(x['target_pref_name'].unique())
                x['target_pref_name_labeled'] = le.transform(x['target_pref_name'])
                #x['target_pref_name_labeled'] = x['target_pref_name_labeled'].astype(int)
                x.columns = x.columns.astype(str)
                x = x.drop(columns=['target_pref_name', 'assay_description', 'molecule_chembl_id', 'canonical_smiles', 'IC50'])

                #Split Data into train and test
                X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

                #Build Multi-Classification Model
                num_classes = np.unique(y, return_counts=False)
                classifier = RandomForestClassifier(n_estimators = len(num_classes), criterion = 'entropy', random_state = 42)
                model = classifier.fit(X_train, Y_train)
                
                Y_pred = classifier.predict(X_test)

                metrics = {'Accuracy': [mc_accuracy(Y_test, Y_pred)],
                            'Precision': [precision_score(Y_test, Y_pred, average='macro')],
                            'Recall': [recall_score(Y_test, Y_pred, average='macro')],
                            'F1': [f1_score(Y_test, Y_pred, average='macro')]}
                st.dataframe(pd.DataFrame(metrics))

                cm = confusion_matrix(Y_test, Y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True)
                ax.set_title('Confussion Matrix')
                ax.set_xticklabels(list(np.unique(Y_test, return_counts=False)), rotation=45)
                ax.set_yticklabels(list(np.unique(Y_test, return_counts=False)), rotation=45)
                ax.figure.set_size_inches(4, 3)
                plt.rcParams.update({'font.size': 5})
                ax.tick_params(axis='both', which='minor', labelsize=5)
                ax.xaxis.set_tick_params(labelsize=5)
                ax.yaxis.set_tick_params(labelsize=5)
                plt.xlabel("Predicted Class")
                plt.ylabel("True Class")                    
                st.pyplot(fig)


            st.subheader("Get Predictions")
            option_input = st.selectbox('Select an option', ('Input File', 'Single SMILE', 'Random SMILES'))

            df = None
            if option_input == 'Input File':
                
                st.write("Upload a file containing at least a column with the canonical smiles with the name \'canonical_smiles\'.")
                uploaded_file = st.file_uploader("Choose input file to predict:")

                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).drop_duplicates().reset_index(drop=True)
                    #df = df.drop_duplicates(subset='canonical_smiles')
                    st.dataframe(df)

            elif option_input == 'Single SMILE':
                single_smile = st.text_input('Write the SMILE to get prediction:', '') #'N[C@H](C)C(=O)O'
                is_valid = Chem.MolFromSmiles(single_smile)
                while single_smile == '' or is_valid == None:
                    if is_valid == None:
                        st.error("Not a valid SMILE, try with another one!")
                    st.stop()

                df = pd.DataFrame({'canonical_smiles': [single_smile]})
                st.dataframe(df)

            elif option_input == 'Random SMILES':
                num_rnd_smiles = st.number_input('How many random SMILES do you want to predict?:', max_value=100, step=10)
                canonical_smiles = list(descriptors_df['canonical_smiles'].unique())

                if num_rnd_smiles == 0:
                    st.stop()
                smis = []
                while len(smis) < num_rnd_smiles:
                    rnd_cs = random.choice(canonical_smiles)
                    mol = Chem.MolFromSmiles(rnd_cs)
                    smi = Chem.MolToSmiles(mol,doRandom=True,canonical=False)
                    if smi not in smis:
                        smis.append(smi)
                df = pd.DataFrame({'canonical_smiles': smis})
                st.dataframe(df)

                # Download file button
                csv_rnd_cs = df.to_csv(index=False).encode('utf-8')
                name_rnd_cs = 'random_smiles.csv'
                st.download_button("Download CSV", csv_rnd_cs, name_rnd_cs, key='download-csv_rnd_cs') #It saves it in Downloads

            if df is None:
                st.stop()

            #Obtain the fingerprints from the canonical smiles
            smiles = df['canonical_smiles'].to_list()
            fps = []
            for s in smiles:
                mol = Chem.MolFromSmiles(str(s))
                fp = Chem.RDKFingerprint(mol)
                arr = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
                
            df_fp = pd.DataFrame(fps)
            # Remove non-explanatory columns
            df = pd.concat([df, pd.DataFrame(selection.transform(df_fp))], axis=1)
            
            input_fp = pd.DataFrame()
            for protein in proteins:
                df_aux = df.copy()
                df_aux['target_pref_name'] = protein
                if model_type in ['Multi-Regression', 'Classification']:
                    df_aux['target_pref_name_labeled'] = le.transform([protein])[0]    
                input_fp = pd.concat([input_fp, df_aux], ignore_index=True, axis=0)
            
            st.dataframe(input_fp)

            # Download file button
            csv_3 = input_fp.to_csv(index=False).encode('utf-8')
            name_3 = 'input_fingerprints_for'
            for protein in proteins:
                name_3 = name_3 + '_' + protein
            name_3 = name_3 + '.csv'
            st.download_button("Download CSV", csv_3, name_3, key='download-csv_3') #It saves it in Downloads

            #Obtain predictions:
            if model_type in ['Multi-Regression', 'Classification']:
                if 'molecule_chembl_id' in input_fp.columns:
                    x = input_fp.drop(columns=['molecule_chembl_id', 'canonical_smiles', 'target_pref_name'])
                else:
                    x = input_fp.drop(columns=['canonical_smiles', 'target_pref_name'])
                
                x.columns = x.columns.astype(str)
                input_fp['bioactivity_prediction'] = model.predict(x)
                input_fp = (input_fp.merge(df_cancer_info[['pref_name', 'Number of Cancers', 'Cancer Types']], left_on='target_pref_name', right_on='pref_name', how='left')).drop(columns=['pref_name'])
                input_fp = input_fp[['target_pref_name', 'canonical_smiles', 'Cancer Types', 'Number of Cancers', 'bioactivity_prediction']]
                bs = bio_score(input_fp, model_type)
                df_score = bs.merge(input_fp, on='canonical_smiles')
                if model_type == 'Multi-Regression':
                    df_score['bioactivity_prediction'] = df_score['bioactivity_prediction'].round(5)
                df_score['bioactivity_prediction'] = df_score['bioactivity_prediction'].astype(str)
                df_score['Number of Cancers'] = df_score['Number of Cancers'].astype(str)
                df_score['bioactivity_predictions'] = df_score.groupby(['canonical_smiles'], as_index=False)['bioactivity_prediction'].transform(lambda x: ','.join(x))
                df_score['Number of Cancers'] = df_score.groupby(['canonical_smiles'], as_index=False)['Number of Cancers'].transform(lambda x: ','.join(x))
                df_score = df_score.drop(columns=['target_pref_name', 'bioactivity_prediction', 'Cancer Types']).drop_duplicates().sort_values(by=['merged_score'], ascending=False).reset_index(drop=True)
                st.write("These are the top recommendations:")
                st.dataframe(df_score.head(10))

                # Download file button
                csv_4 = df_score.to_csv(index=False).encode('utf-8')
                name_4 = 'predictions_file.csv'
                st.download_button("Download CSV", csv_4, name_4, key='download-csv_4') #It saves it in Downloads

            elif model_type == 'Separated Linear Regressions':
                #Get predictions
                df_preds = pd.DataFrame()
                for protein in proteins:
                    df_aux = input_fp[input_fp['target_pref_name'] == protein]
                    x = df_aux.copy()
                    if 'molecule_chembl_id' in x.columns:
                        x = x.drop(columns=['molecule_chembl_id', 'canonical_smiles', 'target_pref_name'])
                    else:
                        x = x.drop(columns=['canonical_smiles', 'target_pref_name'])
                    #x = x.drop(columns=['molecule_chembl_id', 'canonical_smiles', 'target_pref_name'])
                    model = lr_models[protein]
                    df_aux['bioactivity_prediction'] = model.predict(x)
                    df_preds = pd.concat([df_preds, df_aux], ignore_index=True, axis=0)
                #st.dataframe(df_preds)

                cols = ['canonical_smiles', 'target_pref_name', 'bioactivity_prediction']
                df_preds = (df_preds[cols].merge(df_cancer_info[['pref_name', 'Number of Cancers', 'Cancer Types']], left_on='target_pref_name', right_on='pref_name', how='left')).drop(columns=['pref_name'])
                df_preds = df_preds[['target_pref_name', 'canonical_smiles', 'Cancer Types', 'Number of Cancers', 'bioactivity_prediction']]
                df_score = bio_score(df_preds,model_type)
                df_score = df_score.merge(df_preds, on='canonical_smiles')
                df_score['bioactivity_prediction'] = df_score['bioactivity_prediction'].round(5)
                df_score['bioactivity_prediction'] = df_score['bioactivity_prediction'].astype(str)
                df_score['Number of Cancers'] = df_score['Number of Cancers'].astype(str)
                df_score['bioactivity_predictions'] = df_score.groupby(['canonical_smiles'], as_index=False)['bioactivity_prediction'].transform(lambda x: ','.join(x))
                df_score['Number of Cancers'] = df_score.groupby(['canonical_smiles'], as_index=False)['Number of Cancers'].transform(lambda x: ','.join(x))
                df_score = df_score.drop(columns=['target_pref_name', 'bioactivity_prediction', 'Cancer Types']).drop_duplicates().sort_values(by=['merged_score'], ascending=False).reset_index(drop=True)

                st.write("These are the top recommendations:")
                st.dataframe(df_score.head(10))

                # Download file button
                csv_5 = df_score.to_csv(index=False).encode('utf-8')
                name_5 = 'prediction_input_file.csv'
                st.download_button("Download CSV", csv_5, name_5, key='download-csv_4') #It saves it in Downloads

                







