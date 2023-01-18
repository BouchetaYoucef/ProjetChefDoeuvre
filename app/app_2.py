import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
 
    # Pre-processing user input    
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1
 
    if Credit_History == "Unclear Debts":
        Credit_History = 0
    else:
        Credit_History = 1  
 
    LoanAmount = LoanAmount / 1000
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female"))
    Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    ApplicantIncome = st.number_input("Applicants monthly income") 
    LoanAmount = st.number_input("Total loan amount")
    Credit_History = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.success('Your loan is {}'.format(result))
        print(LoanAmount)
     
if __name__=='__main__': 
    main()

























# import streamlit as st
# #from PIL import Image
# import pickle
# from sklearn.preprocessing import LabelEncoder

# import yaml
# from pathlib import Path

# ##### Functions #####

# # def première_fonction(df):
# def features_encoding(df_clean):
#    lBE = LabelEncoder()
#    categ = ["Loan_Status","Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
#    df_clean[categ] = df_clean[categ].apply(lBE.fit_transform)
#    return df_clean    

# # def deuxième_fonction(df):
# def target_encoding(df_clean_encoded):
#    le = LabelEncoder()
#    df_clean_encoded['Loan_Status'] = le.fit_transform(df_clean_encoded['Loan_Status'])
#    return df_clean_encoded

# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=yaml.SafeLoader)

# valid_login = config['credentials']['username']
# valid_pwd = config['credentials']['password']

# def check_password():
#     """Returns `True` if the user had the correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if st.session_state["password"] == valid_pwd and st.session_state["login"] == valid_login:
#             st.session_state["password_correct"] = True 
#             del st.session_state["password"]  # don't store password
#             del st.session_state["login"]
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         # First run, show input for password.
#         st.text_input(
#             "Login", on_change=password_entered, key="login"
#         )
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         return False
#     elif not st.session_state["password_correct"]:
#         # Password not correct, show input + error.
#         st.text_input(
#             "Login", on_change=password_entered, key="login"
#         )
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         st.error("😕 Username/password is incorrect")
#         return False
#     else:
#         # Password correct.
#         return True

# st.session_state["datai"] = {}




# if check_password():
    
#     model = pickle.load(open('model.pkl', 'rb'))


    
#     st.title("Prédiction de prêt bancaire")

#     ## Account No
#     account_no = st.text_input('Numéro du compte')

#     ## Full Name
#     fn = st.text_input('Nom / Prénom')

#     ## For gender
#     gen_display = ('Femme','Homme')
#     gen_options = list(range(len(gen_display)))
#     gen = st.selectbox("Genre",gen_options, format_func=lambda x: gen_display[x])

#     ## For Marital Status
#     mar_display = ('Non','Oui')
#     mar_options = list(range(len(mar_display)))
#     mar = st.selectbox("Etat civil", mar_options, format_func=lambda x: mar_display[x])

#     ## No of dependets
#     dep_display = ('Non','Un','Deux','Plus de deux')
#     dep_options = list(range(len(dep_display)))
#     dep = st.selectbox("Dépendents",  dep_options, format_func=lambda x: dep_display[x])

#     ## For edu
#     edu_display = ('Non diplômé','Diplômé')
#     edu_options = list(range(len(edu_display)))
#     edu = st.selectbox("Education",edu_options, format_func=lambda x: edu_display[x])

#     ## For emp status
#     emp_display = ('Job','Business')
#     emp_options = list(range(len(emp_display)))
#     emp = st.selectbox("Employment Status",emp_options, format_func=lambda x: emp_display[x])

#     ## For Property status
#     prop_display = ('Rural','Semi-Urban','Urban')
#     prop_options = list(range(len(prop_display)))
#     prop = st.selectbox("Endroit de résidence",prop_options, format_func=lambda x: prop_display[x])

#     ## For Credit Score
#     cred_display = ('0','1')
#     cred_options = list(range(len(cred_display)))
#     cred = st.selectbox("Historique de prêt",cred_options, format_func=lambda x: cred_display[x])

#     ## Applicant Monthly Income
#     mon_income = st.number_input("Salaire mensuelle du demandeur",value=0)

#     ## Co-Applicant Monthly Income
#     co_mon_income = st.number_input("Salaire mensuelle du codemandeur)",value=0)

#     ## Loan AMount
#     loan_amt = st.number_input("Prix du prêt",value=0)

#     ## loan duration
#     dur_display = ['2 Mois','6 Mois','8 Mois','1 Année','16 Mois']
#     dur_options = range(len(dur_display))
#     dur = st.selectbox("Durée du prêt",dur_options, format_func=lambda x: dur_display[x])

#     if st.button("Prédiction du prêt"):
#         duration = 0
#         if dur == 0:
#             duration = 60
#         if dur == 1:
#             duration = 180
#         if dur == 2:
#             duration = 240
#         if dur == 3:
#             duration = 360
#         if dur == 4:
#             duration = 480
#         features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
#         print(features)
        
#         ##### TODO #####
#         # exécuter les fonctions de preprocessing
#         # def première_fonction(features):
#         def features_encoding(df_clean):
#            lBE = LabelEncoder()
#            categ = ["Loan_Status","Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
#            df_clean[categ] = df_clean[categ].apply(lBE.fit_transform)
#            return df_clean  
       
#         # def deuxième_fonction(features):
#         def target_encoding(df_clean_encoded):
#            le = LabelEncoder()
#            df_clean_encoded['Loan_Status'] = le.fit_transform(df_clean_encoded['Loan_Status'])
#            return df_clean_encoded
        
        
        
#         prediction = model.predict(features)
#         lc = [str(i) for i in prediction]
#         ans = int("".join(lc))
#         if ans == 0:
#             st.error(
#                 "Bonjour: " + fn +" || "
#                 "Numéro de compte: "+account_no +' || '
#                 'Suite à nos calcul vous ne pouvez pas prétendre à un prêt bancaire.'
#             )
#         else:
#             st.success(
#                 "Bonjour: " + fn +" || "
#                 "Numéro de compte "+account_no +' || '
#                 'Félécitations! Vous pouvez prétendre à un prêt bancaire !'
#             )

