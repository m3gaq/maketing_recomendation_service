![Alt Text](https://media.giphy.com/media/4a5b4AH9TG7zEgsEEe/giphy.gif)
# Web Marketing Intelligence
**Рекомендательный сервис для оценки эффективности новых цифровых каналов продвижения продуктов банка**

MVP сервиса доступен тут: https://petyaeva.ru/moscityhack2022

Документация: https://petyaeva.ru/moscityhack2022/documentation

Сервис разработан в рамках хакатона [Moscow City Hack 2022](https://moscityhack2022.innoagency.ru/)

<!-- ### Функционал MVP:
1. Анализ схожести клиентской базы с пользователями каналов
2. Мэтчинг продуктов банка с каналами продвижения
3. Тренды в веб пространтве
4. Статистика посещений сайтов -->

## Описание сервиса
**Рекомендательный веб-сервис** (далее – Система), который поможет сотрудникам банка [Уралсиб](https://www.uralsib.ru/) (далее – банка) оценивать эффективность цифровых каналов продвижения банковских продуктов на основе данных о целевой аудитории банка и активности конкурентов.

Рекомендательный веб-сервис предоставляет аналитику и рекомендацию по эффективности использования новых и текущих каналов продвижения.

## Основные задачи, решаемые рекомендательным веб-сервисом:

**Live-мониторинг метрик эффективности каналов продвижения (и уже используемых банком, и новых) и рекомендация основных каналов продвижения для каждого из пользовательских сегментов банка за счет внешних данных**
* <ins>ноутбук с решением:</ins> [clusterization_segmentation](https://github.com/m3gaq/maketing_recomendation_service/blob/main/notebooks/clusterization_segmentation.ipynb) (алгоритм кластеризации и выдача рекомендаций каждому кластеру), [parser_data_for_effectivity](https://github.com/m3gaq/maketing_recomendation_service/blob/main/notebooks/parser_data_for_effectivity.ipynb) (live-парсинг данных о каналах и подсчет их эффективности)
* <ins>название раздела на веб-сервисе:</ins> Анализ схожести клиентской базы с пользователями каналов

**Оценка эффективности цифровых каналов продвижения (и уже используемых банком, и новых) и рекомендации по управлению рекламной кампанией для снижения расходов на рекламу за счет исторических данных** 
* <ins>ноутбук с решением:</ins> [rfm_and_marketing_campaigns](https://github.com/m3gaq/maketing_recomendation_service/blob/main/notebooks/rfm_and_marketing_campaigns.ipynb) (расчет и визуализация RFM-сегментации клиентов на основе транзакции и визуализация результатов маркетинговых кампаний)
* <ins>название раздела на веб-сервисе:</ins> Анализ проведенных рекламных кампаний в каналах

**Выявление активной аудитории банка в цифровых каналах и рекомендации по рекламируемым продуктам данной аудитории**
* <ins>ноутбук с решением:</ins> [the_best_product_sell](https://github.com/m3gaq/maketing_recomendation_service/blob/main/notebooks/the_best_product_sell.ipynb) (применение классификации для анализа наиболее привлекательного продукта в конкретном канале)
* <ins>название раздела на веб-сервисе:</ins> Мэтчинг продуктов банка с каналами продвижения

**Аналитика веб-окружения и рекомендации по управлению рекламной кампанией для снижения расходов на рекламу**
* <ins>ноутбуки с решением:</ins> [utilization_expectations](https://github.com/m3gaq/maketing_recomendation_service/blob/main/notebooks/utilization_expectations.ipynb) (оценка костов будущей маркетинговой кампании), [rfm_and_marketing_campaigns](https://github.com/m3gaq/maketing_recomendation_service/blob/main/notebooks/rfm_and_marketing_campaigns.ipynb) (оценка доходности рекламных кампаний на исторических данных), [the_best_product_sell](https://github.com/m3gaq/maketing_recomendation_service/blob/main/notebooks/the_best_product_sell.ipynb) (статистика посещений введенного сайта)
* <ins>название разделов на веб-сервисе:</ins> Оценка костов будущей маркетинговой кампании, Анализ проведенных рекламных кампаний в каналах, Тренды в веб пространстве, Статистика посещений сайтов


## Алгоритм работы
Работу Системы можно разделить на следующие **пять основных этапов**:
1. Извлечение данных,
2. Обработка данных,
3. Обработка запроса пользователя веб-сервиса на выдачу рекомендаций,
4. Формирование рекомендаций,
5. Выгрузка результатов на сайт Системы.
Все взаимодействие с пользователем Системы осуществляется посредством веб-сервиса. 

С сопроводительной документацией сервиса можно ознакомиться по [ссылке](https://drive.google.com/file/d/1LPEr6ie0CHD_j8XujqNoUEIvGF7-DBJ_/view?usp=sharing)

Система способна расширяться и дополняться реальными данными банка в будущем. На рисунке ниже представлена архитектура TO BE разработанного рекомендательного веб-сервиса. 

![alt text](https://github.com/m3gaq/maketing_recomendation_service/blob/main/screenshot/MVP_architect.png)


## Сценарий использования

Предполагаемый **сценарий использования** Системы выглядит следующим образом:

Маркетолог (Пользователь) заходит на веб-сервис, загружает данные, по которым он хочет получить аналитику и рекомендации, и выбирает интересующие его инструменты анализа среди следующих предложенных:
* Анализ схожести клиентской базы с пользователями каналов,
* Анализ проведенных рекламных кампаний в каналах,
* Мэтчинг продуктов банка с каналами продвижения,
* Тренды в веб-пространстве,
* Статистика посещений сайтов.

Система получает запрос Пользователя и обращается к построенным моделям машинного обучения. Модели при необходимости подгружают данные из Интернет-источников и выдают результат (аналитику/рекомендацию) в виде графиков, которые далее отображаются на сайте. Графики являются интерактивными и выдают дополнительную информацию при наведении на них. Маркетолог может использовать эти графики для всестороннего рассмотрения запроса и получения полной аналитики.




<!-- ### Интерфейс:
![alt text](https://github.com/m3gaq/maketing_recomendation_service/blob/main/screenshot/MVP_screenshot.png) -->


<img src="https://media.giphy.com/media/11JTxkrmq4bGE0/giphy.gif" width="450" height="400" />

