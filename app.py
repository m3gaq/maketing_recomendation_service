# embed streamlit docs in a streamlit app
import streamlit as st    
import pandas as pd 
import numpy as np         
import plotly.express as px
import pickle as pkl
import re
from pytrends.request import TrendReq

from our_parser import web_parse
import our_tools

def main():

    plt_topbar = pkl.load(open('plt_topbar.pkl','rb'))
    plt_topscat = pkl.load(open('plt_topscat.pkl','rb'))
    plt_product = pkl.load(open('plt_product.pkl','rb'))


    st.sidebar.title('Web Marketing Intelligence')
    st.title('Web Marketing Intelligence')
    instruments = ['Анализ схожести клиентской базы с пользователями каналов',
                   'Анализ проведенных рекламных кампаний в каналах',
                   'Мэтчинг продуктов банка с каналами продвижения',
                   'Тренды в веб пространтве',
                   'Статистика посещений сайтов']

    selected_instruments = st.sidebar.multiselect('Выберете инструмент',instruments,instruments[:2])
    @st.cache
    def get_related(df):
        if df.size > 0:
            related = ""
            for q in df.head(3)['query'].unique():
                related += q + ', '
            return(re.sub(r',\s$', '', related))
        else:
            return("No related queries found")

    @st.cache
    def get_top(country):
        if len(country.split()) == 1:
            ds = pytrend.trending_searches(pn=country.lower())
            ds.columns = ['trends']
            return ds.head(10).trends.unique()
        else:
            joined_name = ""
            for w in country.split():
                joined_name += w + "_"
            joined_name = re.sub(r'_$', '', joined_name)

            ds = pytrend.trending_searches(pn=joined_name.lower())
            ds.columns = ['trends']
            return ds.head(10).trends.unique()

    @st.cache
    def get_interest_over_time(pytrend):
        return pytrend.interest_over_time()

    st.write('Web Marketing Intelligence сервис ПАО «Банк Уралсиб» от команды MegaQuant.')

    if 'Анализ схожести клиентской базы с пользователями каналов' in selected_instruments:
        st.write('## Анализ схожести клиентской базы с пользователями каналов')
        st.write(plt_topbar)
        st.write(plt_topscat)

    if 'Мэтчинг продуктов банка с каналами продвижения' in selected_instruments:
        st.write('## Мэтчинг продуктов банка с каналами продвижения')
        file = st.file_uploader('Дайте csv файл с описанием пользователей каналов',type=['csv'])
        if file is not None: 
            data_social_media = pd.read_csv(file)
            one_hot_df = our_tools.match_user_product(data_social_media)
            plt_user_product = px.bar(one_hot_df.groupby('channel_id').mean(),our_tools.products)
            st.write(plt_user_product)
        else:
            st.info(
                f"""
                    👆 Попробуйте загрузить [channel_users.csv](https://hse.kamran.uz/share/channel_users.csv)
                    """
            )
            st.write(plt_product)

    if 'Анализ проведенных рекламных кампаний в каналах' in selected_instruments:
        st.write('## Анализ проведенных рекламных кампаний в каналах')
        file = st.file_uploader('Дайте csv по рекламным компаниеям в каналах',type=['csv'])
        if file is not None:
            df_ = pd.read_csv(file)
            st.write(our_tools.plt_historic_data(df_))
            st.write(our_tools.plt_historic_data_returns(df_))
            st.write(our_tools.plt_historic_data_gender(df_))
            st.write(our_tools.rfm_query(df_))
        else:
            st.info(
                f"""
                    👆 Попробуйте загрузить [channel_products.csv](https://hse.kamran.uz/share/channel_products.csv)
                    """
            )


    if 'Тренды в веб пространтве' in selected_instruments:
        st.write('## Тренды в веб пространтве')
        pytrend = TrendReq()
        country = 'russia'
        col, col1, col2, col3, col4 = st.columns(5)
        with col:
            topic  = st.text_input('тема анализа', 'Уралсиб')
        with col1:
            topic1 = st.text_input('тема анализа №1', 'Тинькофф')
        with col2:
            topic2 = st.text_input('тема анализа №2', 'Росбанк')
        with col3:
            topic3 = st.text_input('тема анализа №3', 'Банк Кузнецкий')
        with col4:
            topic4 = st.text_input('тема анализа №4', 'МКБ')

        kw_list = [topic,topic1,topic2,topic3,topic4]
        pytrend.build_payload(kw_list=kw_list, geo='RU')

        interest_over_time_df = get_interest_over_time(pytrend)

        st.write(px.line(interest_over_time_df.drop(columns='isPartial')))

        st.write(f'Сейчас набирает интерес:')

        interest_by_region_df = pytrend.interest_by_region()
        tops_one = get_top(country)

        for t in tops_one:
            with st.expander(f'{np.where(tops_one == t)[0] + 1} {t}'):

                if st.checkbox('Показать связанные запросы', key=f'{t}one'):
                    pytrend.build_payload(kw_list=[t])
                    related_queries = pytrend.related_queries()
                    rising = pd.DataFrame(data=related_queries.get(t).get('rising'))
                    st.write(get_related(rising))

    if 'Статистика посещений сайтов' in selected_instruments:
        st.write('## Статистика посещений сайтов')    
        col, col1, col2 = st.columns(3)
        with col:
            web_site  = st.text_input('сайт для анализа', 'www.uralsib.ru')

        st.write(web_parse(web_site))



if __name__ == '__main__':
    main()
