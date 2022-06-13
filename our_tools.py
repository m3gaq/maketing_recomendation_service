"""#Модуль 
В данном модуле логический выделены инструменты,
которые используются в сервисе. 

Для работы модуля необходимо наличие следующих библиотек 
`pandas`, `numpy`. 
Для работы ряда фукнций модуля необходимо наличие следующих библиотек
`sklearn`,`plotly`.

Следущие функции рекомендованы к использованию

    * preprocess         - выполняет предобработку (очистка, трансформация, ...) 
                           датасета с месячными транкзакциями пользователей

    * generate_ds        - генериррует датасет аудитории каналов задданого размера 
                           (генерация в согласии с исслеедованиями целевой аудитории)

    * match_user_product - возвращает продуктовый профиль пользователей канала
                           (профиль составляется по схожести с аудиторией банка) 

    * plt_historic_data  - ...

    * plt_historic_data_returns  - ...

    * plt_historic_data_gender  - ...

"""

import pandas as pd
import numpy as np

df = pd.read_csv('https://hse.kamran.uz/share/hack2022_df.csv')
data_rfm = pd.read_csv('https://hse.kamran.uz/share/data_rfm.csv')

products = [
       'card_type_name_American Express Optimum',
       'card_type_name_American Express Premier',
       'card_type_name_Eurocard/MasterCard Gold',
       'card_type_name_Eurocard/MasterCard Mass',
       'card_type_name_Eurocard/MasterCard Platinum',
       'card_type_name_Eurocard/MasterCard Virt',
       'card_type_name_Eurocard/MasterCard World',
       'card_type_name_MasterCard Black Edition',
       'card_type_name_MasterCard Electronic',
       'card_type_name_MasterCard World Elite', 
       'card_type_name_MIR Supreme',
       'card_type_name_MIR Privilege Plus',
       'card_type_name_Дебет карта ПС МИР "Бюджетная"',
       'card_type_name_МИР Debit', 'card_type_name_МИР Копилка',
       'card_type_name_МИР СКБ', 'card_type_name_МИР СКБ ЗП',
       'card_type_name_VISA Classic', 'card_type_name_VISA Classic Light',
       'card_type_name_VISA Gold', 'card_type_name_VISA Infinite',
       'card_type_name_VISA Platinum', 'card_type_name_Visa Classic Rewards',
       'card_type_name_Visa Platinum Rewards', 'card_type_name_Visa Rewards',
       'card_type_name_Visa Signature', 
       'card_type_name_Priority Pass',
       ]
product_type = ['American Express']*2+['MasterCard']*8+['MIR']*7+['visa']*9+['Other']*1
product_type = {p:t for p,t in zip(products,product_type)}
user = ['gender','age','nonresident_flag']

def preprocess(df, ohe_cols=['card_type_name', 'city']):
    '''
    Возвращает список количества визитов из сырого js кода из Spymetrics.

    Parameters
    ----------
    df : DataFrame
        Сырой js код из Spymetrics.

    ohe_cols : list
    
    Returns
    -------
    (DataFrame, DataFrame)
        DataFrame c визитами на сайт.
        DataFrame c датами визитами на сайт.
    '''
    df.drop_duplicates(inplace = True)

    del df['term']
    del df['card_id']

    if len(ohe_cols)>1:
        del df['client_id']

    # преобразование небинарных признаков
    one_hot_df = pd.get_dummies(df, 
                                columns=ohe_cols, 
                                drop_first=False)
    
    from datetime import datetime, date
    today = date.today()
    one_hot_df['Year'] = pd.to_datetime(one_hot_df['birth_date'], format='%Y')
    one_hot_df['year'] = pd. DatetimeIndex(one_hot_df['Year']).year
    one_hot_df['age'] = today.year - one_hot_df['year']
    del one_hot_df['Year']
    del one_hot_df['year']
    del one_hot_df['birth_date']

    one_hot_df['life_account'] = one_hot_df['fact_close_date'] - one_hot_df['start_date']
    one_hot_df.loc[one_hot_df["gender"] == "М","gender"] = 1
    one_hot_df.loc[one_hot_df["gender"] == "Ж","gender"] = 0
    one_hot_df.loc[one_hot_df["nonresident_flag"] == "R","nonresident_flag"] = 0
    one_hot_df.loc[one_hot_df["nonresident_flag"] == "N","nonresident_flag"] = 1

    one_hot_df.loc[one_hot_df['card_type'] == "dc","card_type"] = 1
    one_hot_df.loc[one_hot_df['card_type'] == "cc","card_type"] = 0


    one_hot_df.loc[one_hot_df['product_category_name'] == "Кредитная карта","product_category_name"] = 1
    one_hot_df.loc[one_hot_df['product_category_name'] == "Договор на текущий счет для дебетовой карты",'product_category_name'] = 0

    one_hot_df[['start_date', 'fact_close_date']] = np.where(one_hot_df[['start_date', 'fact_close_date']].isnull(), 0, 1)
    one_hot_df['year'] = pd. DatetimeIndex(one_hot_df['create_date']).year
    del one_hot_df['create_date']
    one_hot_df.fillna(0, inplace=True)
    return one_hot_df

def try_different_clusters(K, data):
    from sklearn.cluster import KMeans
    cluster_values = list(range(1, K+1))
    inertias=[]
    clust_models=[]
    
    for c in cluster_values:
        model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)
        clust_models.append(model)
    
    return inertias,clust_models

def fit_clusters(one_hot_df):
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(init='k-means++',  max_iter=400, random_state=42)
    kmeans_model.fit(one_hot_df)

    outputs, clust_models = try_different_clusters(7, one_hot_df)
    distances = pd.DataFrame({"clusters": list(range(1, 8)),"sum of squared distances": outputs})

    import plotly.graph_objects as go
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

    elbow_fig.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),                  
                    xaxis_title="Количество кластеров",
                    yaxis_title="Сумма расстояний",
                    title_text="Оптимальное количество кластеров")
    
    return elbow_fig, clust_models, distances

def generate_ds(size=1000,db_size=0.3):
    def make_social_data(num_people, for_age, p_gender, p_res, p_act):
        gender = np.random.choice(binary, num_people, p=[p_gender, 1 - p_gender])
        nonresident_flag = np.random.choice(binary, num_people, p=[p_res, 1 - p_res])
        active = np.random.choice(binary, num_people, p=[p_act, 1 - p_act])
        age = np.random.choice(for_age, num_people)
        
        data_social_m = pd.DataFrame(columns=["gender", "age", "nonresident_flag", "active"])
        data_social_m["gender"], data_social_m["age"], data_social_m["nonresident_flag"], data_social_m["active"] = gender, age, nonresident_flag, active
        
        return data_social_m

    binary = np.arange(2)
    for_age_ = np.arange(65) + 20

    data_credit = make_social_data(int(size*db_size), for_age_, 0.44, 0.9, 0.7)
    data_deb = make_social_data(int(size*(1-db_size)), for_age_, 0.45, 0.9, 0.7)

    data_social_media = pd.concat([data_credit, data_deb], ignore_index=True)
    data_channel = np.random.randint(0,10,int(size*db_size)+int(size*(1-db_size)))
    data_social_media['channel_id'] = data_channel
    return data_social_media

def match_user_product(data_social_media):
    from sklearn.neighbors import KNeighborsRegressor
    channel_id = data_social_media['channel_id']
    data_social_media = data_social_media.drop(columns='channel_id')
   
    one_hot_df = preprocess(df.drop(columns='city').copy(), ohe_cols=['card_type_name'])
    one_hot_df = one_hot_df.groupby(['client_id','gender','age','nonresident_flag']).mean().reset_index().drop(columns=['client_id'])
   
    knrs = [pd.Series(KNeighborsRegressor().fit(one_hot_df[user],one_hot_df[product]).predict(data_social_media[user])) for product in products]

    ddd = pd.concat(knrs,axis=1)
    ddd.columns = products
    one_hot_df_ = pd.concat([data_social_media,ddd],axis=1)
    one_hot_df_['channel_id'] = channel_id

    return one_hot_df_

def plt_historic_data(data_plot: pd.DataFrame):
    """ 
    Визуализирует распределения интересов к продуктам у пользователей, перешедших по ссылке (купивших и не купивших)

    Parameters
    ----------
    data_plot: DataFrame
        Таблица проведенных рекламных кампаний по каналам.
    
    Returns
    -------
    fig: PlotlyFigure
        Столбчатый график распределения заказанного продукта по каналам
    """
    import plotly.express as px
    fig = px.bar(data_plot.groupby(by='source').mean(), 
               y = list(data_plot.groupby(by='source').mean().index), 
               x=data_plot.columns[-5:])
    fig.update_layout(
    title="Распределение заказанного продукта по каналам",
    xaxis_title="Интерес к продукту (процент покупок от общего количества перешедших)",
    yaxis_title="Каналы")
    return fig

def plt_historic_data_returns(data_plot: pd.DataFrame):
    """ 
    Визуализирует доходы со всех каналов 

    Parameters
    ----------
    data_plot: DataFrame
        Таблица проведенных рекламных кампаний по каналам.
    
    Returns
    -------
    fig: PlotlyFigure
        Столбчатый график доходов по каналам продвижения. 
    """
    import plotly.express as px
    fig = px.bar(data_plot.groupby(by='source').sum(), 
               x = list(data_plot.groupby(by='source').sum().index), 
               y = ['contract_sum'])
    fig.update_layout(
    title="Итоговая сумма контрактов по каналам",
    xaxis_title="source",
    yaxis_title="contract_sum_all")
    return fig

def plt_historic_data_gender(data_plot: pd.DataFrame):
    """ 
    Визуализирует распределение пола заинтересованных пользователей по каналам

    Parameters
    ----------
    data_plot: DataFrame
        Таблица проведенных рекламных кампаний по каналам.
    
    Returns
    -------
    fig: PlotlyFigure
        Столбчатый график распределения пола заинтересованных.
    """
    import plotly.express as px
    fig = px.bar(data_plot.groupby(by='source').mean(), 
               y = list(data_plot.groupby(by='source').mean().index), 
               x=['gender'])
    fig.update_layout(
    title="Распределение М/Ж по каналам",
    xaxis_title="Пол",
    yaxis_title="Каналы")
    return fig

def rfm_segments_compaign(data_marketing_compaign, data_rfm=data_rfm):
    """ 
    Визуализирует результаты сегментации на основе RFM модели 
    по истечению тестового срока клиентов, купивших продукт во время рекламной кампании

    Parameters
    ----------
    data_plot: DataFrame
        Таблица проведенных рекламных кампаний по каналам.
    
    Returns
    -------
    fig: PlotlyFigure
        Столбчатый график распредления сегментов пользователей по RFM, 
        пришедших из рекламных кампаний.
    """

    def rfm_query(adid):
      ans = data_rfm[
                     data_rfm['client_id']
                     .isin(data_marketing_compaign[
                                         ((data_marketing_compaign.is_deal == 1) & (data_marketing_compaign.adid == adid))
                                         ]
                           ['client_id']
                                        .unique())
                     ].rfm_score_name.value_counts()
      return ans
      
    data1 = rfm_query('8730nd')
    data2 = rfm_query('873kdb')
    data3 = rfm_query('n27cl3')

    fig = make_subplots(rows=1, cols=3, 
                        specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]], 
                                              subplot_titles=['id - 8730nd', 'id - 873kdb', 'id - n27cl3'])
    fig.add_trace(
        go.Pie(values = list(data1), labels=list(data1.index)),row=1, col=1)
    fig.add_trace(
        go.Pie(values = list(data2), labels=list(data2.index)),row=1, col=2)
    fig.add_trace(
        go.Pie(values = list(data3), labels=list(data3.index)),row=1, col=3)
    fig.update_layout(height=600, width=800, title_text="Сегмент пользователей по RFM, пришедших из рекламных кампаний")
    return fig