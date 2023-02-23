# Import libraries
import pandas as pd
import streamlit as st
from intro import intro
from get_data_preprocessing import obtain_and_preprocess_data
from decriptor_dataset import get_fingerprint_descriptors
from create_model import create_model_from_scratch, upload_pretrained_model

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
            "Step Selection", ["Intro", "Obtain and Preprocess Data", "Create Descriptor Dataset", "Create Model"]
        )

        if input_selectbox == "Intro":
            intro()

        elif input_selectbox == "Obtain and Preprocess Data":
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
        input_selectbox_2 = st.sidebar.selectbox(
            "Step Selection", ["Intro", "Select Proteins"]
        )

        if input_selectbox_2 == "Intro":
            intro()

        elif input_selectbox_2 == "Select Proteins":
            from chembl_webresource_client.new_client import new_client
            from rdkit import Chem
            from rdkit.Chem import DataStructs
            import numpy as np
            from sklearn.feature_selection import VarianceThreshold
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            import seaborn as sns
            import matplotlib.pyplot as plt

            #Hacer que se puedan escoger los cancers!!

            #Poner explanation en el streamlit sobre los acronimos!!!!!
            cancers_dict = {'BRC': 'breast cancer', 'OC': 'ovarian cancer', 'LC': 'lung cancer', 'LK': 'leukemia'}

            df_cancer_target = pd.DataFrame()
            for cancer_type in cancers_dict:
                df_aux = new_client.target.search(cancers_dict[cancer_type]).filter(target_type = 'SINGLE PROTEIN', organism = 'Homo sapiens')
                df_aux = pd.DataFrame.from_dict(df_aux)
                df_aux['Cancer Type'] = cancer_type
                df_cancer_target = pd.concat([df_cancer_target, df_aux], ignore_index=True, axis=0)
                
            st.dataframe(df_cancer_target) #TAKE OUT!!!!!!!

            cols = ['Cancer Type', 'pref_name', 'target_chembl_id']
            df_cancer_info = df_cancer_target[cols]
            #Se tendría que hacer también por UNIPROT ID
            df_cancer_info = df_cancer_info.groupby(['pref_name', 'target_chembl_id']).count()
            #st.dataframe(df_cancer_info)

            # Select proteins
            proteins = st.multiselect('Select the proteins:', list(df_cancer_target['pref_name']))
            
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
                st.write("Display of the resulting dataframe after computing the fingerprints from the canonical smiles and removing non-explanatory columns:")
                st.dataframe(data=descriptors_df)

                # Download file button
                csv_2 = descriptors_df.to_csv(index=False).encode('utf-8')
                name_2 = 'selected_target_activities_figerprint_descriptors_part_2.csv'
                st.download_button("Download CSV", csv_2, name_2, key='download-csv_2') #It saves it in Downloads

                # Train model
                descriptors_df['IC50'] = descriptors_df['IC50'].astype('float')
                y = descriptors_df['IC50']
                x = descriptors_df.copy()
                le = LabelEncoder()
                le.fit(x['target_pref_name'].unique())
                x['target_pref_name_labeled'] = le.transform(x['target_pref_name'])
                #x['target_pref_name_labeled'] = x['target_pref_name_labeled'].astype(int)
                x.columns = x.columns.astype(str)
                x = x.drop(columns=['target_pref_name', 'assay_description', 'molecule_chembl_id', 'canonical_smiles', 'IC50'])

                #Split Data into train and test
                X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                
                #Build Regression Model
                np.random.seed(100)

                st.subheader("Train Model and Show R2 Result:")
                model = RandomForestRegressor(n_estimators=200)
                model.fit(X_train, Y_train)
                r2 = model.score(X_test, Y_test)
                st.write("The R2 result is: " + str(r2))
                Y_pred = model.predict(X_test)

                st.subheader('Scatter Plot of Real vs Predicted pIC50 Values')
                fig, ax = plt.subplots()
                sns.set(color_codes=True)
                sns.set_style("white")
                ax = sns.regplot(x=Y_test, y=Y_pred)
                ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
                ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
                #ax.figure.set_size_inches(5, 5)
                st.pyplot(fig)

                





