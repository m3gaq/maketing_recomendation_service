"""#Парсер Данных с Агрегатора Статистики Spymetrics

Данный скрипт парсит Spymetrics и получает данные о 
посещениях заданного веб-сайта.

Для работы скрипта необходимо наличие следующих библиотек 
`pandas`, `plotly`, `BeautifulSoup`, `requests_html`.

Скрипт может быть импортирован как модуль.

    * visits_data - возвращает список количества визитов из сырого js кода из Spymetrics. 
                    Вспомогательная фукнция.

    * web_parse   - принимает название сайта.
                    Возвращает количество посещений на заданный сайт 
"""

import re
import ast
import pandas as pd

import plotly.express as px
import datetime
from requests_html import HTMLSession
from bs4 import BeautifulSoup


def visits_data(js_script_str):
    """ 
    Возвращает список количества визитов из сырого js кода из Spymetrics.

    Parameters
    ----------
    js_script_str : str
        Сырой js код из Spymetrics.
    
    Returns
    -------
    (DataFrame, DataFrame)
        DataFrame c визитами на сайт.
        DataFrame c датами визитами на сайт.
    """

    js_script_str = re.sub("Highcharts.Map", "Highcharts.Chart", js_script_str)
    
    # Find the indices of new Highcharts.Chart
    lst_1 = [match.start() for match in re.finditer("new Highcharts.Chart", js_script_str)][1:]
    lst_2 = [match.end() for match in re.finditer("new Highcharts.Chart", js_script_str)]
    
    # Pairs of indices of consecutive new Highcharts.Chart to parse everything that's inbetween
    lst_tuples = list(zip(lst_2, lst_1))
    lst_lists = [list(elem) for elem in lst_tuples]
    
    # Adjust the indices to get rid of rubbish
    for t in lst_lists:
        t[0] = t[0] + 1
        t[1] = t[1] - 30
        
    # Extract the contents between the new Highcharts.Chart
    d1_str = js_script_str[lst_lists[0][0]:lst_lists[0][1]]
    d1_str = re.sub('false', "False", d1_str)
    d1_str = re.sub('null', '""', d1_str)
    re.sub("\\\\",'"', d1_str) ### careful, assignment may be needed!!!
    
    # Convert to dict
    d1 = ast.literal_eval(d1_str)
    
    return d1['chart']['title'], d1['series'][1]['data']

def web_parse(website):
    """
    Возвращает график посещений на website.

    Parameters
    ----------
    website : str
        Название веб-сайта в формате 'название.домен', например 'uralsib.ru'.

    Returns
    -------
    PlotlyFigure
        График-Линия посещений на website. 
        По оси x дата посещений, по оси y количество посещений.
    """

    root_url = "https://spymetrics.ru/ru/website/" 
    full_url = root_url + website

    # create an HTML Session object
    session = HTMLSession()

    # Use the object above to connect to needed webpage
    html= session.get(full_url).text

    soup = BeautifulSoup(html, "html.parser")
    list_js_scripts = soup.find_all("script", type="text/javascript")
    js_script_str = None
    for js_script in list_js_scripts:
        if "jQuery('#webcompareform')" in str(js_script):
            js_script_str = js_script.contents[0]
            visits = visits_data(js_script_str)

    today = datetime.date.today()
    first = today.replace(day=1)
    prev = [first - i * datetime.timedelta(days=25) for i in range(1,7)]
    monthes = []
    for date in prev:
        monthes.append(date.strftime("%b%Y"))
    monthes = monthes[::-1]

    fig = px.line(y=visits[1], x=monthes, title=f'Визиты сайта {website}', labels={'x': 'Месяц', 'y': 'Количество посещений'}, color_discrete_sequence=["#4C4C9D"])
    return fig