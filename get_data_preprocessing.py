import numpy as np
import pandas as pd
import streamlit as st
from chembl_webresource_client.new_client import new_client


def ask_get_target():
    target = st.text_input("Which target are you searching for?")
    if not target:
        st.stop()
    if len(target) > 2: #Gives error of less than two characters
        res = new_client.target.search(target)
    else:
        res = []

    while len(res) == 0:
        target_aux = target
        st.error("Target name incorrect, try again!")
        if target_aux == target:
            st.stop()
        if len(target) > 2:
            res = new_client.target.search(target)
        else:
            st.error("Target name incorrect, try again!")

    st.write("The selected target is ", target, ".")
    return pd.DataFrame.from_dict(res), target

def ask_get_target_id(df: pd.DataFrame):
    min_index = 0
    max_index = len(df) - 1
    text_input_title = "Select target from index value [" + str(min_index) + ', ' + str(max_index)   + ']:'
    target_index = st.text_input(text_input_title)
    if not target_index:
        st.stop()
    target_index = int(target_index)

    while target_index < min_index or target_index > max_index:
        target_index_aux = target_index
        st.error("Incorrect index, try again!")
        if target_index_aux == target_index:
            st.stop()
    
    target_id = df['target_chembl_id'][target_index]
    df_activity = new_client.activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    st.write("The selected target is ", target_id, ".")
    return  pd.DataFrame.from_dict(df_activity), target_id

def get_df_preprocessed(df: pd.DataFrame):
    df = df[df.standard_value.notna()] #Filter out rows with NaN values
    bioactivity_class = []
    for i in df.standard_value:
        if float(i) >= 10000:
            bioactivity_class.append("inactive")
        elif float(i) <= 1000:
            bioactivity_class.append("active")
        else:
            bioactivity_class.append("intermediate")
    df['bioactivity_class'] = bioactivity_class
    columns_wanted = ['assay_description', 'molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'standard_value']
    return df[columns_wanted]

#The original def is obtained from data_professor's videos
def get_pIC50(ic50: list):
    pIC50 = []
    stnd_val_norm = []

    for i in ic50:
        i = float(i)
        if i > 100000000:
            i = 100000000
        stnd_val_norm.append(i)

    for i in stnd_val_norm:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    return pIC50

def obtain_and_preprocess_data():
    #Part 1
    st.title('ML for Drug Discovery')
    st.header('Get Target Name')
    df, target = ask_get_target()
    st.dataframe(data=df)

    #Part 2 #I asume that all have at least one 'SINGLE PROTEIN'
    st.header('Get Activity DF from Target ID')
    df_sp = df[df['target_type'] == 'SINGLE PROTEIN'].reset_index(drop=True) #Df of only single proteins
    st.dataframe(data=df_sp)
    df_activity, target_id = ask_get_target_id(df_sp)
    df_activity_final = get_df_preprocessed(df_activity)
    st.dataframe(data=df_activity_final)

    ## Download file button
    csv = df_activity_final.to_csv(index=False).encode('utf-8')
    name = 'bioactivity_data_' + target + '.csv'
    st.download_button("Download CSV", csv, name, key='download-csv') #It saves it in Downloads

    #Part 3 #Obtain pIC50 from IC50 (standard_value) and remove the intermediate bio class
    st.header('Obtain pIC50 from IC50 (standard_value)')
    st.write('This step is done to have uniformly data:')
    df_activity_final['pIC50'] = get_pIC50(df_activity_final['standard_value'])
    st.dataframe(data=df_activity_final)

    st.write('Now remove the \'intemediate\' rows from the bioactivity class:')
    df_activity_final = df_activity_final[df_activity_final['bioactivity_class'] != 'intermediate'].reset_index(drop=True)
    st.dataframe(data=df_activity_final)

    ## Download file button
    csv_2 = df_activity_final.to_csv(index=False).encode('utf-8')
    name = 'bioactivity_data_' + target + '_pIC50.csv'
    st.download_button("Download CSV", csv_2, name, key='download-csv_2') #It saves it in Downloads