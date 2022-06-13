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
    st.write('Привет! Это команда MegaQuant. Специально для тебя мы разработали рекомендательный сервис для оценки эффективности новых цифровых каналов продвижения продуктов банка. Обрати внимание на меню слева, чтобы выбрать инструменты для принятия взвешенных решений :)')
    instruments = ['Анализ схожести клиентской базы с пользователями каналов',
                   'Анализ проведенных рекламных кампаний в каналах',
                   'Мэтчинг продуктов банка с каналами продвижения',
                   'Оценка костов будущей маркетинговой кампании',
                   'Тренды в веб пространстве',
                   'Статистика посещений сайтов']

    selected_instruments = st.sidebar.multiselect('Выберете инструмент',instruments,instruments[:2]*0)
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

    if 'Анализ схожести клиентской базы с пользователями каналов' in selected_instruments:
        st.write('## Анализ схожести клиентской базы с пользователями каналов')
        st.write('В этой части мы сделали для тебя сегментацию клиентов. Для каждого кластера есть своя система оценки эффективности каналов. Например, для кластера 1 лучший канал - это авито. На этом графике ты можешь оценить эффективность канала не только по посещаемости сайта клиентами, но и по схожести ЦА Уралсиба с клиентами канала.')
        st.write(plt_topbar)
        st.write(plt_topscat)

    if 'Мэтчинг продуктов банка с каналами продвижения' in selected_instruments:
        st.write('## Мэтчинг продуктов банка с каналами продвижения')
        file = st.file_uploader('Загрузите csv файл с описанием пользователей каналов',type=['csv'])
        if file is not None: 
            st.write('На графике показана совместимость продуктов банка с каналами продвижения. Он показывает, какие продукты привлекательны в конкретном канале. Здесь не используются  исторические данные по маркетинговым кампаниям.')
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
        file = st.file_uploader('Загрузите csv по рекламным компаниеям в каналах',type=['csv'])
        if file is not None:
            st.write('Здесь приведен анализ предыдущих рекламных кампаний и RFM сегментация (см https://petyaeva.ru/moscityhack2022/documentation). Это поможет оценить интересы новой аудитории, похожей на ту, о которой мы уже что-то знаем. К тому же, можно оценить из какого канала какой сегмент пользователей по RFM, как правило, приходит, для понимания лояльности к нашему банку и платежеспособности.')
            df_ = pd.read_csv(file)
            st.write(our_tools.plt_historic_data(df_))
            st.write(our_tools.plt_historic_data_returns(df_))
            st.write('Ниже представлена статистика привлеченных пользователей по рекламным кампаниям и их оценка по RFM сегментации. Так, например, кампания с id 8730nd привлекла 20% лучших клиентов (champions) и скорее всего похожие рекламные кампании будут также успешны.')
            st.write(our_tools.plt_rfm_segments_compaign(df_))
        else:
            st.info(
                f"""
                    👆 Попробуйте загрузить [channel_products.csv](https://hse.kamran.uz/share/channel_products.csv)
                    """
            )

    if 'Оценка костов будущей маркетинговой кампании' in selected_instruments:
        st.write('## Оценка костов будущей маркетинговой кампании')
        st.write('''Перед запуском кампании мы хотим понять, какой оффер нам нужно предлагать клиентам и сколько дней предлагать, чтоб активировать этот оффер, на какой день предоагать оффер ([предложения взяты из uralsib.ru](https://www.uralsib.ru/)).''')
        st.write('''
* цена реализации оффера на одного человека (`cost`)

* список карт, к которым относится оффер (`offer`)

Итак, рассматриваем такие офферы:

1. **Оффер №1:** бесплатный пакет Premium обслуживания в первые два месяца. Стоимость одного месяца обслуживания -- 1000 рублей. Предлагается обладателям карт 'MIR Supreme' и 'MIR Privilege Plus', которые не используют эти карты

* `cost` = 2000 рублей 

* `offer` = ['MIR Supreme', 'MIR Privilege Plus']

2. **Оффер №2:** бесплатное обслуживание карты для обладателей карт 'Дебет карта ПС МИР "Бюджетная"', 'МИР Копилка', которые не используют эти карты. Стоимость одного месяца обслуживания -- 99 рублей.

* `cost` = 99 рублей 

* `offer` = ['Дебет карта ПС МИР "Бюджетная"', 'МИР Копилка']

3. **Оффер №3:** бесплатная подписка на 3 месяца на SMS-уведомления для обладателей карт 'МИР СКБ', 'Дебет карта ПС МИР "Бюджетная"', 'МИР Копилка', 'MIR Supreme', 'MIR Privilege Plus', которые не используют эти карты. Стоимость одного месяца подписки -- 59 рублей.

* `cost` = 59 рублей 

* `offer` = ['МИР СКБ', 'Дебет карта ПС МИР "Бюджетная"', 'МИР Копилка', 'MIR Supreme', 'MIR Privilege Plus']
''')
        st.write('Введите парамерты будущей маркетинговой кампании')
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            offer=st.selectbox('offer',[1,2,3],1)
        with col2:
            expected_growth=st.number_input('expected_growth in %',1,1000,10,1)/1000
        with col3:
            on_day=st.slider('on_day',0,100,0,1)
        with col4:
            zero_day=st.slider('zero_day',on_day,100,5,1)
        uti_fig, c = our_tools.utilization(offer, expected_growth, on_day, zero_day)
        st.write(f"Кост на утилизацию одного клиента {round(c, 0)} рублей.")
        st.write(uti_fig)
    if 'Тренды в веб пространстве' in selected_instruments:
        st.write('## Тренды в веб пространстве')
        st.write('Если ты хочешь исследовать внешний контекст, то тв можешь вбить ключевые слова интересующей тебя тематики. График тебе выведет количество запросов по этим словам. Пример: политика. График покажет, сколько людей искали в поисковике слово «Политика». На основе этого ты можешь понимать тенденции во внешней среде.')
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
        st.write('Если хочешь проверить посещаемость сайтов перед рекламной кампанией, то вбей ссылку в поле ниже. Пример: www.vk.ru. Ты увидишь посещаемость сайта за последние полгода. Чем это тебе поможет? Обратите внимание на аномальную посещаемость - низкую или высокую. Может быть, есть шанс устроить успешную рекламную кампанию именно сейчас за счет отсутствия информационного шума, а может быть, наоборот, стоит немного подождать.')
        col, col1, col2 = st.columns(3)
        with col:
            web_site  = st.text_input('сайт для анализа', 'www.uralsib.ru')

        st.write(web_parse(web_site))



if __name__ == '__main__':
    main()
