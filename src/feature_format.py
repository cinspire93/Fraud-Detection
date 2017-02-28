from __future__ import division
import numpy as np
import pandas as pd

features_list = []

'''
We use features_list to compile features that we actually want
to use to train our models

Subsequently, everyone of our feature engineering function also appends
the engineered feature to our features_list
'''
def total_tickets(data):
    '''
    Input: Ticket data
    Output: DataFrame with number of tickets aggregated from list of dictionaries
            within the ticket column
    '''
    for row in data:
        quantity_total = 0
        for ticket_type in row['ticket_types']:
            quantity_total += ticket_type['quantity_total']
        row['num_tickets'] = quantity_total
    features_list.append('num_tickets')

class FeatureEngineering(object):

    def __init__(self, dataframe):
        self.df = dataframe

    def run_feat_engineer(self):
        self.email_domains()
        self.payout_type()
        self.sale_duration()
        self.fill_nans(['delivery_method'])
        self.user_type()
        self.upper()
        self.currency_dums()
        self.payee_checking()
        self.payout_ratio()
        self.cooldown()
        self.prev_payouts()
        self.social_media(['org_facebook', 'org_twitter'])

    def email_domains(self):
        '''
        Input: DataFrame
        Output: Add extra column that gives 1 for non(gmail, yahoo, hotmail) domains
                and 0 otherwise
        '''
        email_domains = ['gmail', 'yahoo', 'hotmail']
        features_list.extend(email_domains)
        for domain in email_domains:
            self.df[domain] = (self.df['email_domain'].str.startswith(domain)).astype(int)
        self.df['other'] = np.logical_not(self.df[email_domains].sum(axis=1).astype(bool)).astype(int)
        features_list.append('other')

    def payout_type(self):
        '''
        Input: DataFrame
        Output: Dummifies payout_type column, then append resulting columns to the
                original DataFrame
        '''
        dummies = pd.get_dummies(self.df['payout_type'], prefix='payout')
        self.df[dummies.columns] = dummies
        features_list.extend(dummies.columns.tolist())

    def sale_duration(self):
        '''
        Input: DataFrame
        Output: Split sale_duration2 into two categories
        '''
        self.df.loc[:,'sale_duration2'].fillna(0, inplace=True)
        self.df.loc[self.df['sale_duration2']<0, 'sale_duration2'] = 0
        features_list.append('sale_duration2')

    def fill_nans(self, columns):
        '''
        Input: DataFrame
        Output: Filling all NaN values with 0
                To be used with delivery_method
        '''
        df.loc[:,columns] = df[columns].fillna(0)
        features_list.extend(columns)

    def user_type(self):
        '''
        Input: DataFrame
        Output: Dummifies user_type column, then append resulting columns to the
                original DataFrame
        '''
        user_type_list = range(1,6)
        user_type_cols = []
        for i in user_type_list:
            name = 'user_type_'+str(i)
            self.df[name] = 0
            user_type_cols.append(name)
        self.df.loc[~self.df['user_type'].isin(user_type_list), 'user_type'] = 3
        dummies = pd.get_dummies(self.df['user_type'], prefix='user_type')
        self.df[dummies.columns] = dummies
        features_list.extend(user_type_cols)

    def upper(self):
        '''
        Input: DataFrame
        Output: Adds column that gives 1 when username is all CAPS, 0 otherwise
        '''
        self.df['is_upper'] = (self.df['name'].str.isupper()).astype(int)
        features_list.append('is_upper')

    def currency_dums(self):
        '''
        Input: DataFrame
        Output: Adds column that gives 1 when currency is GBP or MXN, 0 otherwise
        '''
        self.df['dummy_currency'] = (self.df['currency'].isin(['GBP', 'MXN'])).astype(int)
        features_list.append('dummy_currency')

    def payee_checking(self):
        '''
        Input: DataFrame
        Output: Adds column that gives checks whether payee has a name, gives 1
                if no name exists, and 0 otherwise
        '''
        self.df['payee_check'] = (self.df.payee_name=='').astype(int)
        self.df['payee_check'].fillna(0, inplace=True)
        features_list.append('payee_check')

    def payout_ratio(self):
        '''
        Input: DataFrame
        Output: Adds column that gives the payout percentage. Calculated by
                dividing number of payouts with number of orders+0.01, the 0.01
                is there to avoid dividing by 0
        '''
        self.df['payout_pct'] = self.df['num_payouts']/(self.df['num_order'] + 0.01)
        self.df.loc[self.df['payout_pct'] > 0,'payout_pct'] = 1
        features_list.append('payout_pct')

    def social_media(self, columns):
        '''
        Input: DataFrame
        Output: Checks for social media presence of a seller, returns 1 if yes and
                0 otherwise
        '''
        self.df.loc[:,columns].fillna(0, inplace=True)
        self.df[columns] = self.df[columns].astype(bool).astype(int)
        features_list.extend(columns)

    def cooldown(self):
        '''
        Input: DataFrame
        Output: Adds column that contains the time elapsed from time of account
                creation to first event created
        '''
        self.df['cd_period'] = self.df['event_created'] - self.df['user_created']
        self.df['cd'] = (self.df['cd_period'] < 60).astype(int)
        features_list.append('cd')

    def prev_payouts(self):
        '''
        Input: DataFrame
        Output: Adds column that gives 1 if user has no previous payouts, and 0
                otherwise
        '''
        self.df['prev_num_payouts'] = self.df['previous_payouts'].map(len)
        self.df['prev_pay_no'] = self.df['prev_num_payouts']==0
        features_list.append('prev_pay_no')


def feature_engineering(data):
    # Total_tickets calculation deals with a peculiar column type, therefore
    # we opt to engineer it outside of our DataFrame
    total_tickets(data)

    df = pd.DataFrame(data)

    fe = FeatureEngineering(df)
    fe.run_feat_engineer()

    # other features
    features_list.extend(['fb_published'])

    ## extract features we want
    X = df[features_list].values
    return X, np.array(features_list)
