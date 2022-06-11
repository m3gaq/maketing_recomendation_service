# embed streamlit docs in a streamlit app
import streamlit as st    
import pandas as pd 
import numpy as np         
import plotly.express as px
import pickle as pkl
import re
from pytrends.request import TrendReq
def main():

    plt_topbar = pkl.load(open('plt_topbar.pkl','rb'))
    plt_topscat = pkl.load(open('plt_topscat.pkl','rb'))

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


    st.title('Web Marketing Intelligence')
    st.write('Web Marketing Intelligence сервис ПАО «Банк Уралсиб» от команды MegaQuant.')

    st.write('## Анализ схожести клиентской базы с пользователями каналов')
    st.write(plt_topbar)
    st.write(plt_topscat)
    
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




if __name__ == '__main__':
    main()
