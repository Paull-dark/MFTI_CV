<h1><center>DESCRIPTION</center></h1>


<p style='text-align: center;'><span style="color: #0098f3; font-family: Segoe UI; font-size: 2.5em; font-weight: 300;">Итоговый проект курса от МФТИ</span></p>
<h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:blue; border:0; color:white' role="tab" aria-controls="home"><center>[Детектирование Объектов (сценарий 2)]</center></h2>


PROJECT DONE BY:

Paul Krivchenia 

[GIT](https://github.com/Paull-dark)

[Stepik_id](https://stepik.org/users/304346049)

[KAGGLE](https://www.kaggle.com/pauldark)


PROJECT TARGETS:
    
    Выбор датасета
    Предобработка данных
    Выбор фрэймворка
    Обучение нейросети
    Измерение метрик
    Поиск путей применения нашей модели

СОСТАВ ПРОЕКТА

    Основные части, на которые стоит обращать внимание(Проверяющему):
    
    - EDA.ipynb Ноутбук с постановкой задачи, анализом данных, визуализацией
    - Train.ipynb Ноутбук с тренировкой модели и выводом метрик
    - Predictions.ipynb ноутбук с предсказаниями и выводами по проекту.

    Вспомогательные файлы (НЕ ОБЯЗАТЕЛЬНЫ К ПРОСМОТРУ ПРОВЕРЯЮЩИМ!!!).
    В вспомогательные файлы были вынесенны некоторые функции, классы, которые могут быть
    применены не тоько в этом проекте но и в других.
    Напимер функции по печати плотности распределения данных, 
    конвертации типа файла dicom в jpg и другие.
    
    - config.py Файл с основными настройками модели и путями к даным в файловой системе.
    - plottings.py Файл для вывода графиков для анализа данных
    - med_vis.py файл для визуализации изображений dicom
    - dicom_to_jpg.py - файл для конвертации dicom to jpg
    - COVID19_dict.py Файл для создания датасета в формате COCO

КРАТКИЕ РЕЗУЛЬТАТЫ:

    - Описание возможного применения находится в ведении ноутбука EDA.ipynb
    - Датасет выбран. Ссылка на датасет в EDA.ipynb а разделе Введение.
    - В качестве фрэймворка выбран Detectron2. Почему именно он, описано в ноутбуке Train.ipynb
    - Модель обучена в ноутбуке Train.ipynb
    - Метрики измеренны в конце ноутбука Train.ipynb
    - Предсказания для тестовой выборки сделаны в ноутбуке Predictions.ipynb
    - Выводы по проекту приведены в Predictions.ipynb

Проект является (я надеюсь) воспроизводимым.

Для воспроизведения результатов ноутбука необходимо:

    - скачать датасет по ссылке https://www.kaggle.com/c/siim-covid19-detection/data
    - скачать config.py, plottings.py, med_vis.py, dicom_to_jpg.py COVID19_dict.py
    - Запустить EDA.ipynb (генерация исправленного csv с аннотациями)
    - Запустить Train.ipynb