import streamlit as st
import pandas as pd
import numpy as np
import ast
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from patsy import dmatrix

st.set_page_config(layout="wide")
st.logo("logo.png")

def campaign_level_model():
    name = st.sidebar.selectbox(
        "Country-Language",
        ("English - United States", "English - United Kingdom", "English - Canada")
    )
    channels = st.sidebar.selectbox(
        "Channels",
        ("Digital", "Digital & Social", "Digital & Linear TV", "Digital, Linear TV & Social")
    )
    IP_enabled = st.sidebar.selectbox(
        "IP Matching Enabled",
        ("True", "False")
    )
    industry_name = st.sidebar.selectbox(
        "Industry",
        ("Academia and Education", "Alcohol","Consumer Packaged Goods")
    )
    campaign_length = st.sidebar.number_input("Campaign Length (Days)", value=60, placeholder="Type a number...")
    expected_impressions = st.sidebar.number_input("Expected Impressions", value=1000000, placeholder="Type a number...")
    
    df = pd.read_csv('international_feasibility_data_all.csv')
    
    current_total_expr = """total_completes ~ log_impressions_actual + log_campaign_length + IP_enabled_api + industry_name + channels + name"""

    y_all, X_all = dmatrices(current_total_expr, df, return_type='dataframe')
    
    df_train = df.copy()
    
    #Using the statsmodels GLM class, train the Poisson regression model on the training data set
    poisson_training_results = sm.GLM(y_all, X_all, family=sm.families.Poisson()).fit()
    
    #Add the λ vector as a new column called 'BB_LAMBDA' to the Data Frame of the training data set
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    
    #add a derived column called 'AUX_OLS_DEP' to the pandas Data Frame. This new column will store the values of the dependent variable of the OLS regression
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['total_completes'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
    
    #use patsy to form the model specification for the OLSR
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
    
    #Configure and fit the OLSR model
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    
    #train the NB2 model on the training data set
    current_training_results = sm.GLM(y_all, X_all,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()

    expected_impressions = int(expected_impressions)
    campaign_length = int(campaign_length)
    IP_enabled = ast.literal_eval(IP_enabled.strip().title())
    
    data = dict(expected_impressions=expected_impressions, campaign_length=campaign_length, IP_enabled_api=IP_enabled, industry_name=industry_name, name=name, channels=channels, )
    new_campaign = pd.DataFrame(data, index=[0])
    
    new_campaign['log_campaign_length'] = np.log(new_campaign['campaign_length'])
    new_campaign['log_impressions_actual'] = np.log(new_campaign['expected_impressions'])
    
    design_info = X_all.design_info
    new_X = dmatrix(design_info, new_campaign, return_type="dataframe")
    nb2_predictions = current_training_results.get_prediction(new_X)
    final = round(nb2_predictions.summary_frame(alpha=0.1))
    final_completes = final.rename(columns={'mean':'Predicted Total Completes','mean_ci_lower':'90% CI Lower','mean_ci_upper':'90% CI Upper'})[['Predicted Total Completes','90% CI Lower','90% CI Upper']]

    current_imp_expr = """impressions_per_exposed_complete ~ log_impressions_actual + log_campaign_length + IP_enabled_api + industry_name + channels + name"""

    y_all, X_all = dmatrices(current_imp_expr, df, return_type='dataframe')

    df_train = df.copy()

    # Using the statsmodels GLM class, train the Poisson regression model on the training data set
    poisson_training_results = sm.GLM(y_all, X_all, family=sm.families.Poisson()).fit()

    # Add the λ vector as a new column called 'BB_LAMBDA' to the Data Frame of the training data set
    df_train['BB_LAMBDA'] = poisson_training_results.mu

    # add a derived column called 'AUX_OLS_DEP' to the pandas Data Frame. This new column will store the values of the dependent variable of the OLS regression
    df_train['AUX_OLS_DEP'] = df_train.apply(
        lambda x: ((x['impressions_per_exposed_complete'] - x['BB_LAMBDA']) ** 2 - x['BB_LAMBDA']) / x['BB_LAMBDA'],
        axis=1)

    # use patsy to form the model specification for the OLSR
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""

    # Configure and fit the OLSR model
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()

    # train the NB2 model on the training data set
    current_training_results_impressions = sm.GLM(y_all, X_all, family=sm.families.NegativeBinomial(
        alpha=aux_olsr_results.params[0])).fit()

    data = dict(expected_impressions=expected_impressions, campaign_length=campaign_length, IP_enabled_api=IP_enabled,
                industry_name=industry_name, name=name, channels=channels, )
    new_campaign = pd.DataFrame(data, index=[0])

    new_campaign['log_campaign_length'] = np.log(new_campaign['campaign_length'])
    new_campaign['log_impressions_actual'] = np.log(new_campaign['expected_impressions'])

    design_info = X_all.design_info
    new_X = dmatrix(design_info, new_campaign, return_type="dataframe")
    nb2_predictions = current_training_results_impressions.get_prediction(new_X)
    final = round(nb2_predictions.summary_frame(alpha=0.1))
    final_impressions = final.rename(columns={'mean': 'Predicted Impressions per Exposed Complete', 'mean_ci_lower': '90% CI Lower',
                                  'mean_ci_upper': '90% CI Upper'})[
        ['Predicted Impressions per Exposed Complete', '90% CI Lower', '90% CI Upper']]
    
    
    st.title("IM Feasibility Calculator: by Campaign")
    st.write("This dashboards displays the predictions of the campaign-level IM feasibility model")

    st.write("#### Predicted Impressions per Exposed Complete")
    st.dataframe(final_impressions, hide_index=True)

    st.write("#### Predicted Total Completes")
    st.dataframe(final_completes, hide_index=True)
    
campaign_level_model()