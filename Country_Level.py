import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from patsy import dmatrix

st.set_page_config(layout="wide")
st.logo("logo.png")

def country_level_model():
    name = st.sidebar.selectbox(
        "Country-Language",
        ('Arabic - Saudi Arabia','Arabic - United Arab Emirates','Chinese Simplified - China','Chinese Traditional - Hong Kong',
         'Chinese Traditional - Taiwan','Czech - Czech Republic','Dutch - Belgium','Dutch - Netherlands','English - Australia','English - Canada','English - India',
         'English - Indonesia','English - Ireland','English - Kenya','English - Malaysia','English - New Zealand','English - Pakistan','English - Philippines',
         'English - Singapore','English - South Africa','English - United Arab Emirates','English - United Kingdom','English - United States','Finnish - Finland',
         'French - Belgium','French - Canada','French - France','German - Austria','German - Germany','Greek - Greece','Hindi - India','Hungarian - Hungary',
         'Indonesian - Indonesia','Italian - Italy','Japanese - Japan','Korean - Korea','Malay - Malaysia','Polish - Poland','Portuguese - Brazil',
         'Portuguese - Portugal','Romanian - Romania','Russian - Russia','Spanish - Argentina','Spanish - Chile','Spanish - Colombia','Spanish - Ecuador','Spanish - Mexico',
         'Spanish - Peru','Spanish - Spain','Spanish - United States','Swahili - Kenya','Swedish - Sweden','Tagalog - Philippines','Thai - Thailand','Turkish - Turkey',
         'Vietnamese - Vietnam','Slovakia - Slovenia','Croatian - Croatia','German - Switzerland','Slovak - Slovakia','Spanish - Costa Rica',
         'English - Hong Kong','Serbian - Serbia','French - Switzerland','Spanish - Dominican Republic','Bulgarian - Bulgaria','Estonian - Estonia',
         'Latvian - Latvia','Lithuanian - Lithuania')
    )
    df = pd.read_csv('international_feasibility_data_all.csv')
    df['country_name'] = df['country_name'].replace({'United Arab Emirates (UAE)': 'United Arab Emirates'})
    df_country_level = df.groupby('name')[
        ['impressions_per_exposed_complete', 'total_completes', 'log_unique_respondents',
         'log_completed_sessions']].mean().reset_index()
    df_country_level[['language_name', 'country_name']] = df_country_level['name'].str.split(' - ', n=1, expand=True)

    country_mapping = pd.read_csv("UN_geoscheme_subregions.csv", encoding='latin1')
    continent = country_mapping[['country_name', 'Continental_region']]
    df_country_level = pd.merge(df_country_level, continent, how='left', on='country_name')

    internet_pop = pd.read_csv('Internet_pop_by_country.csv', encoding='latin1')
    df_country_level = pd.merge(df_country_level, internet_pop, how='left', on='country_name')
    df_country_level = df_country_level.replace(',', '', regex=True)
    df_country_level['log_internet_users'] = np.log(df_country_level['Internet users'].astype(int))

    country_total_expr = """total_completes ~ Continental_region + log_unique_respondents + log_completed_sessions + log_internet_users"""

    y_all, X_all = dmatrices(country_total_expr, df_country_level, return_type='dataframe')

    df_train = df_country_level.copy()

    # Using the statsmodels GLM class, train the Poisson regression model on the training data set
    poisson_training_results = sm.GLM(y_all, X_all, family=sm.families.Poisson()).fit()

    # Add the λ vector as a new column called 'BB_LAMBDA' to the Data Frame of the training data set
    df_train['BB_LAMBDA'] = poisson_training_results.mu

    # add a derived column called 'AUX_OLS_DEP' to the pandas Data Frame. This new column will store the values of the dependent variable of the OLS regression
    df_train['AUX_OLS_DEP'] = df_train.apply(
        lambda x: ((x['total_completes'] - x['BB_LAMBDA']) ** 2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)

    # use patsy to form the model specification for the OLSR
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""

    # Configure and fit the OLSR model
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()

    # train the NB2 model on the training data set
    country_training_results = sm.GLM(y_all, X_all,
                                      family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()

    data = dict(name=name)
    new_campaign = pd.DataFrame(data, index=[0])

    country_mapping = pd.read_csv("UN_geoscheme_subregions.csv",encoding='latin1')
    countries = country_mapping[['country_name', 'Continental_region']]
    new_campaign[['language_name', 'country_name']] = new_campaign['name'].str.split(' - ', n=1, expand=True)
    new_campaign = pd.merge(new_campaign, countries, how='left', on='country_name')

    sid_rsid_by_country = pd.read_csv('sid_rsid_by_country.csv')
    mp_country_languages = pd.read_csv('MP_country_languages.csv')
    df_unk = sid_rsid_by_country.merge(mp_country_languages, how='left', left_on='lk_countrylanguageid', right_on='id')[
        ['lk_countrylanguageid', 'name', 'unique_respondents', 'completed_sessions']]
    df_unk['name'] = df_unk['name'].str.strip()
    new_campaign = pd.merge(new_campaign, df_unk, how='left', on='name')

    new_campaign['log_unique_respondents'] = np.log(new_campaign['unique_respondents'])
    new_campaign['log_completed_sessions'] = np.log(new_campaign['completed_sessions'])

    internet_pop = pd.read_csv('Internet_pop_by_country.csv', encoding='latin1')[['country_name', 'Internet users']]
    internet_pop = internet_pop.replace(',', '', regex=True)
    new_campaign = pd.merge(new_campaign, internet_pop, how='left', on='country_name')
    new_campaign['log_internet_users'] = np.log(new_campaign['Internet users'].astype(int))

    design_info = X_all.design_info
    new_X = dmatrix(design_info, new_campaign, return_type="dataframe")
    nb2_predictions = country_training_results.get_prediction(new_X)
    final = round(nb2_predictions.summary_frame(alpha=0.1))
    final_completes = final.rename(columns={'mean': 'Predicted Average Total Completes', 'mean_ci_lower': '90% CI Lower',
                                  'mean_ci_upper': '90% CI Upper'})[
        ['Predicted Average Total Completes', '90% CI Lower', '90% CI Upper']]

    df_unk_table = df_unk[['name', 'unique_respondents', 'completed_sessions']]
    df_unk_table = df_unk_table.rename(
        columns={'name': 'Country-Language', 'unique_respondents': 'MP Unique Respondents',
                 'completed_sessions': 'MP Completed Sessions'})
    df_unk_table_country = df_unk_table[
        df_unk_table['Country-Language'] == name]

    expr = """impressions_per_exposed_complete ~ Continental_region + log_unique_respondents + log_completed_sessions + log_internet_users"""

    y_all, X_all = dmatrices(expr, df_country_level, return_type='dataframe')

    df_train = df_country_level.copy()

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
    country_training_results_impressions = sm.GLM(y_all, X_all, family=sm.families.NegativeBinomial(
        alpha=aux_olsr_results.params[0])).fit()

    data = dict(name=name)
    new_campaign = pd.DataFrame(data, index=[0])

    country_mapping = pd.read_csv("UN_geoscheme_subregions.csv",encoding='latin1')
    countries = country_mapping[['country_name', 'Continental_region']]
    new_campaign[['language_name', 'country_name']] = new_campaign['name'].str.split(' - ', n=1, expand=True)
    new_campaign = pd.merge(new_campaign, countries, how='left', on='country_name')

    sid_rsid_by_country = pd.read_csv('sid_rsid_by_country.csv')
    mp_country_languages = pd.read_csv('MP_country_languages.csv')
    df_unk = sid_rsid_by_country.merge(mp_country_languages, how='left', left_on='lk_countrylanguageid', right_on='id')[
        ['lk_countrylanguageid', 'name', 'unique_respondents', 'completed_sessions']]
    df_unk['name'] = df_unk['name'].str.strip()
    new_campaign = pd.merge(new_campaign, df_unk, how='left', on='name')

    new_campaign['log_unique_respondents'] = np.log(new_campaign['unique_respondents'])
    new_campaign['log_completed_sessions'] = np.log(new_campaign['completed_sessions'])

    internet_pop = pd.read_csv('Internet_pop_by_country.csv', encoding='latin1')[['country_name', 'Internet users']]
    internet_pop = internet_pop.replace(',', '', regex=True)
    new_campaign = pd.merge(new_campaign, internet_pop, how='left', on='country_name')
    new_campaign['log_internet_users'] = np.log(new_campaign['Internet users'].astype(int))

    design_info = X_all.design_info
    new_X = dmatrix(design_info, new_campaign, return_type="dataframe")
    nb2_predictions = country_training_results_impressions.get_prediction(new_X)
    final = round(nb2_predictions.summary_frame(alpha=0.1))
    final_impressions = final.rename(
        columns={'mean': 'Predicted Average Impressions per Exposed Complete', 'mean_ci_lower': '90% CI Lower',
                 'mean_ci_upper': '90% CI Upper'})[
        ['Predicted Average Impressions per Exposed Complete', '90% CI Lower', '90% CI Upper']]


    st.title("IM Feasibility Calculator: by Country")
    st.write("This dashboards displays the predictions of the country-level IM feasibility model. 
    The model uses continential region, internet census (population of the country that uses the internet), 
    count of unique marketplace respondents, and count of completed marketplace sessions to predict the average impressions per exposed complete or average total complates for each country. 
    The model is trained on data from currently-supported countries, and then predictions are calculated for new countries based on the model features.")

    st.write("#### Predicted Average Impressions per Exposed Complete")
    st.dataframe(final_impressions, hide_index=True)

    st.write("#### Predicted Average Total Completes")
    st.dataframe(final_completes, hide_index=True)

    st.write("#### Marketplace Census by Country (past 6 months)")
    st.dataframe(df_unk_table_country, hide_index=True)

country_level_model()
