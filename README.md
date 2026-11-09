# Тест 3D-фигуры для VK Mini App (Looksy)

Минимальная страница для проверки 3D-манекена с мерками на телефоне внутри VK Mini App.

## Что на странице

- Реалистичный гладкий женский манекен (без одежды и текстур лица)
- Вращение пальцем на 360° и pinch-zoom (OrbitControls)
- Ползунки: рост, обхват груди, обхват талии, обхват бёдер (см)
- Размер РФ и ростовка
- «Кадров в секунду»
- VK Bridge (`VKWebAppInit`)

## Модель и лицензия

Файл `body.glb` собран из **MakeHuman / MPFB2** базового меша и measure-таргетов
(`measure-bust-circ-*`, `measure-waist-circ-*`, `measure-hips-circ-*`, height).

Исходные данные MakeHuman **явно выпущены как CC0** (сентябрь 2020; Data Collection AB,
Joel Palmius, Jonas Hauquier). SMPL **не используется** (у SMPL отдельная
research/non-commercial лицензия, для публичного хостинга не подходит).

Код three.js — MIT.

## GitHub Pages

https://yafandina.github.io/looksy-3d-test/
