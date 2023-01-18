import joblib
import numpy
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

#Import preprocess
from preprocessor import cleaner
from preprocessor import feature_encoder
# from preprocessor import pipeline_constructor
# from preprocessor import feature_engineering


img1 = open('image4.jpg')
# img1 = img1.resize((150, 150))
# st.image(img1, use_column_width=False)

# ## --- SELECTIONS DES DONNEES --- ## 

# # For gender
gen_display = ('Female', 'Male')
gen_options = list(range(len(gen_display)))
gen = st.selectbox("Genre", gen_options, format_func=lambda x: gen_display[x])

# # For Marital Status
mar_display = ('No', 'Yes')
mar_options = list(range(len(mar_display)))
mar = st.selectbox("Mariée", mar_options,
                   format_func=lambda x: mar_display[x])

# # No of dependets
dep_display = ('0', '1', '2', '3+')
dep_options = list(range(len(dep_display)))
dep = st.selectbox("Nombre d'enfant", dep_options,
                   format_func=lambda x: dep_display[x])

# # For edu
edu_display = ('Not Graduate', 'Graduate')
edu_options = list(range(len(edu_display)))
edu = st.selectbox("Education", edu_options,
                   format_func=lambda x: edu_display[x])

# # For emp status
emp_display = ('Yes', 'No')
emp_options = list(range(len(emp_display)))
emp = st.selectbox("Travailleur independant", emp_options,
                   format_func=lambda x: emp_display[x])

# # For Property status
prop_display = ('Rural', 'Semi-Urban', 'Urban')
prop_options = list(range(len(prop_display)))
prop = st.selectbox("Zone d'habitation", prop_options,
                    format_func=lambda x: prop_display[x])

# # Applicant Monthly Income
mon_income = float(st.number_input("Revenus demandeur", value=0))

# Credit history
credit_display = ('Yes', 'No')
credit_hst_options = list(range(len(credit_display)))
credit_hst = st.selectbox("Autre crédit en cours", credit_hst_options,
                          format_func=lambda x: credit_display[x])

# Co-Applicant Monthly Income
co_mon_income = float(st.number_input("Revenues co-demandeur", value=0))
# Loan AMount
loan_amt = float(st.number_input("Montant du credit", value=0))
# loan duration
dur = float(st.number_input("Durée du credit", value=0))

# ## ----------------------------------------------------- ## 

# if st.button("Submit"):
# #     ## --- TRAITEMENT DES DONNEES --- ##
#     model = joblib.load('./models/clf_model.joblib')
#     COLUMNS_NAMES = ['Gender', 'Married', 'Dependents', 'Education',
#     'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#     'Loan_Amount_Term', 'Credit_History', 'Property_Area'
#     ]
#     # dep = 3 if dep == "3+" else int(dep)
#     df = pd.DataFrame(data=[[gen, mar, dep, edu, emp, mon_income, 
#                             co_mon_income, loan_amt, dur, credit_display, prop]], columns=COLUMNS_NAMES)

#     # Process raw data
#     data = cleaner(df)
#     data_prcd = feature_encoder(data)
    
#     st.table(data_prcd)

#     # Unit testing on first feature engineering model improvement exp run #
#     # preds, probas = feature_engineering(X, y, lg_pipe)
#     # ------------------------------ ##
    
# #     ## --- PREDICTION --- ##
#     pred = model.predict(data_prcd)
#     proba = model.predict_proba(data_prcd)
    
#     if not pred == 0:
#         st.text(f"Le demandeur est eligble au credit avec un indicateur de confiance de {proba}")
#     else:
#         st.text(f"Le demandeur n'est pas eligble au credit avec un indicateur de confiance de {proba}")


if st.button("Submit"):
    #     ## --- TRAITEMENT DES DONNEES --- ##
    model = joblib.load('./models/clf_model.joblib')
    COLUMNS_NAMES = ['Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]
    # dep = 3 if dep == "3+" else int(dep)
    df = pd.DataFrame(data=[[gen, mar, dep, edu, emp, mon_income, 
                            co_mon_income, loan_amt, dur, credit_display, prop]], columns=COLUMNS_NAMES)

    st.write(df)

    
    # Process raw data
    data = cleaner(df)
    st.write(type(data))
    data_prcd = feature_encoder(data)
    
    st.write(type(data_prcd))
    
#     st.table(data_prcd)

#     # Unit testing on first feature engineering model improvement exp run #
#     # preds, probas = feature_engineering(X, y, lg_pipe)
#     # ------------------------------ ##
    
# #     ## --- PREDICTION --- ##
#     pred = model.predict(data_prcd)
#     proba = model.predict_proba(data_prcd)
    
#     if not pred == 0:
#         st.text(f"Le demandeur est eligble au credit avec un indicateur de confiance de {proba}")
#     else:
#         st.text(f"Le demandeur n'est pas eligble au credit avec un indicateur de confiance de {proba}")

