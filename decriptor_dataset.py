import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import DataStructs
from sklearn.feature_selection import VarianceThreshold

def get_fingerprint_descriptors():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.dropna(subset=['canonical_smiles', 'pIC50']).reset_index(drop=True)
        st.dataframe(data=df)
        
        smiles = df['canonical_smiles'].to_list()
        fps = []
        for s in smiles:
            mol = Chem.MolFromSmiles(str(s))
            fp = Chem.RDKFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            
        df_fp = pd.DataFrame(fps)
        descriptors_df = pd.concat([df_fp, df['pIC50']], axis=1)
        st.write("Display of the resulting dataframe after computing the fingerprints from the canonical smiles:")
        st.dataframe(data=descriptors_df)

        # Download file button
        csv_3 = descriptors_df.to_csv(index=False).encode('utf-8')
        name_3 = 'bioactivity_data_pIC50_all_pubchem_descriptors_rdkit.csv'
        st.download_button("Download CSV", csv_3, name_3, key='download-csv_3') #It saves it in Downloads

        #Remove non-explanatory columns
        selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
        X = pd.concat([pd.DataFrame(selection.fit_transform(df_fp)), df['pIC50']], axis=1)
        st.write("Dataframe after removing non-explanatory columns:")
        st.dataframe(data=X)
    
        # Download file button
        csv_4 = X.to_csv(index=False).encode('utf-8')
        name_4 = 'bioactivity_data_pIC50_reduced_pubchem_descriptors_rdkit.csv'
        st.download_button("Download CSV", csv_4, name_4, key='download-csv_4') #It saves it in Downloads