# KYC OCR test report — Seedream

- Изображений: **90**
- Успешно обработано: **90**
- Ошибки pipeline: **0**
- Полностью совпавших (все поля): **0** (0.0%)
- Средний **CER** по картинке: **0.050**
- Средний **CER** по полю (макро, среднее CER mean по 10 полям): **0.050**
- Средняя точность поля (exact): **74.6%**
- Среднее время на картинку: **3.17 сек**

## По полям

| Поле | N | Exact | Acc | CER mean | CER p90 | CER max | WER mean | Edit dist total |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| lastname | 90 | 82 | 91.1% | 0.031 | 0.000 | 1.000 | 0.031 | 23 |
| firstname | 90 | 85 | 94.4% | 0.013 | 0.000 | 0.333 | 0.013 | 5 |
| middlename | 90 | 87 | 96.7% | 0.014 | 0.000 | 0.667 | 0.014 | 11 |
| sex | 90 | 89 | 98.9% | 0.011 | 0.000 | 1.000 | 0.011 | 1 |
| birth_date | 90 | 84 | 93.3% | 0.028 | 0.000 | 1.000 | 0.028 | 20 |
| birth_place | 90 | 3 | 3.3% | 0.122 | 0.143 | 1.000 | 0.122 | 115 |
| date_of_issue | 90 | 84 | 93.3% | 0.037 | 0.000 | 1.000 | 0.037 | 27 |
| department_code | 90 | 83 | 92.2% | 0.041 | 0.000 | 1.000 | 0.041 | 22 |
| passport_issued_full | 90 | 7 | 7.8% | 0.113 | 0.273 | 1.000 | 0.113 | 582 |
| series_and_number | 90 | 67 | 74.4% | 0.093 | 0.200 | 1.000 | 0.093 | 84 |
| **среднее по полям** | — | — | — | **0.050** | — | — | — | — |

## Топ-20 худших картинок (по среднему CER)

| # | image | row | CER mean | Exact fields |
| --- | --- | --: | --: | --: |
| 1 | passport_0079_seedream.jpeg | 79 | 0.879 | 0/10 |
| 2 | passport_0068_seedream.png | 68 | 0.398 | 5/10 |
| 3 | passport_0072_seedream.png | 72 | 0.308 | 6/10 |
| 4 | passport_0070_seedream.png | 70 | 0.285 | 4/10 |
| 5 | passport_0041_seedream.png | 41 | 0.166 | 5/10 |
| 6 | passport_0078_seedream.png | 78 | 0.127 | 5/10 |
| 7 | passport_0027_seedream.png | 27 | 0.127 | 7/10 |
| 8 | passport_0025_seedream.png | 25 | 0.095 | 7/10 |
| 9 | passport_0051_seedream.png | 51 | 0.089 | 7/10 |
| 10 | passport_0004_seedream.png | 4 | 0.077 | 8/10 |
| 11 | passport_0019_seedream.jpeg | 19 | 0.068 | 7/10 |
| 12 | passport_0099_seedream.png | 99 | 0.062 | 5/10 |
| 13 | passport_0010_seedream.png | 10 | 0.062 | 5/10 |
| 14 | passport_0024_seedream.png | 24 | 0.061 | 8/10 |
| 15 | passport_0002_seedream.png | 2 | 0.060 | 7/10 |
| 16 | passport_0090_seedream.png | 90 | 0.059 | 7/10 |
| 17 | passport_0074_seedream.png | 74 | 0.055 | 8/10 |
| 18 | passport_0048_seedream.png | 48 | 0.053 | 7/10 |
| 19 | passport_0081_seedream.png | 81 | 0.050 | 6/10 |
| 20 | passport_0037_seedream.png | 37 | 0.049 | 7/10 |

## Распределение CER по картинкам

| Bucket | Count |
| --- | --: |
| 0 | 0 |
| ≤0.05 | 71 |
| ≤0.10 | 12 |
| ≤0.20 | 3 |
| ≤0.30 | 1 |
| ≤0.50 | 2 |
| ≤1.0 | 1 |
| >1.0 | 0 |

## Примеры расхождений (топ-10 по полю)

### lastname
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0070_seedream.png | `МОРОЗОВ` | `` | 1.000 |
| passport_0079_seedream.jpeg | `ВАЛИУЛЛИН` | `С.АКИИМЧ` | 0.889 |
| passport_0062_seedream.png | `ПОДШИВАЛОВ` | `ТОДШИВАДВ` | 0.300 |
| passport_0048_seedream.png | `КРЮКОВ` | `КРОКОВ` | 0.167 |
| passport_0090_seedream.png | `ВОЛОЩУК` | `ВОЛОШУК` | 0.143 |
| passport_0025_seedream.png | `КЛЮЧНИКОВ` | `КЛОЧНИКОВ` | 0.111 |
| passport_0073_seedream.png | `МЕЩЕРЯКОВА` | `МЕЦЕРЯКОВА` | 0.100 |
| passport_0006_seedream.png | `ОКРОПИРИДЗЕ` | `ОКРОЛИРИДЗЕ` | 0.091 |

### firstname
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0079_seedream.jpeg | `ФОМ` | `АОМ` | 0.333 |
| passport_0041_seedream.png | `ЮЛИЯ` | `ОЛИЯ` | 0.250 |
| passport_0087_seedream.png | `ЛЮДА` | `ЛОДА` | 0.250 |
| passport_0070_seedream.png | `ТАГИР` | `ТАТИР` | 0.200 |
| passport_0010_seedream.png | `ЮЛИАНА` | `ОЛИАНА` | 0.167 |

### middlename
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0079_seedream.jpeg | `ГРАНТОВИЧ` | `-ЬКМТОБМИЧ` | 0.667 |
| passport_0070_seedream.png | `АБДУЛОВИЧ` | `АБОДОВНЧ` | 0.444 |
| passport_0081_seedream.png | `ЮРЬЕВНА` | `ОРЬЕВНА` | 0.143 |

### sex
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0079_seedream.jpeg | `МУЖ.` | `Ж` | 1.000 |

### birth_date
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0041_seedream.png | `19.11.1981` | `` | 1.000 |
| passport_0079_seedream.jpeg | `28.06.1984` | `` | 1.000 |
| passport_0010_seedream.png | `13.02.1993` | `13.02.2993` | 0.125 |
| passport_0020_seedream.png | `26.08.1959` | `126.08.1959` | 0.125 |
| passport_0027_seedream.png | `17.11.1994` | `17.11.199` | 0.125 |
| passport_0055_seedream.png | `21.06.1982` | `121.06.1982` | 0.125 |

### birth_place
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0079_seedream.jpeg | `Г. СОЧИ` | `` | 1.000 |
| passport_0024_seedream.png | `Г. САНКТ-ПЕТЕРБУРГ` | `Г КЧЕЕЛПРЕУРГ` | 0.556 |
| passport_0070_seedream.png | `Г. МОСКВА` | `МОСКВА` | 0.333 |
| passport_0086_seedream.png | `Г. УФА` | `Г УЗА` | 0.333 |
| passport_0048_seedream.png | `Г. ТВЕРЬ` | `ТГ ТВЕРЬ` | 0.250 |
| passport_0076_seedream.png | `ПОС. СЕРГИЕВ ПОСАД МОСКОВСКОЙ ОБЛ.` | `ПОС. СЕРГИЕВ ПОСАД МОСКОВСН` | 0.235 |
| passport_0011_seedream.png | `Г. УФА` | `Г УФА` | 0.167 |
| passport_0015_seedream.png | `Г. УФА` | `Г УФА` | 0.167 |
| passport_0002_seedream.png | `Г. СОЧИ` | `Г СОЧИ` | 0.143 |
| passport_0007_seedream.png | `Г. ОМСК` | `Г ОМСК` | 0.143 |

### date_of_issue
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0068_seedream.png | `17.03.2006` | `` | 1.000 |
| passport_0072_seedream.png | `28.03.2011` | `` | 1.000 |
| passport_0079_seedream.jpeg | `10.12.2006` | `` | 1.000 |
| passport_0051_seedream.png | `31.03.2019` | `B1.03.2019` | 0.125 |
| passport_0078_seedream.png | `22.12.2021` | `12.12.2021` | 0.125 |
| passport_0099_seedream.png | `31.03.2024` | `01.03.2024` | 0.125 |

### department_code
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0068_seedream.png | `162-847` | `` | 1.000 |
| passport_0072_seedream.png | `642-108` | `` | 1.000 |
| passport_0079_seedream.jpeg | `614-393` | `` | 1.000 |
| passport_0019_seedream.jpeg | `591-195` | `59-195` | 0.167 |
| passport_0041_seedream.png | `251-410` | `251-420` | 0.167 |
| passport_0078_seedream.png | `420-458` | `420-45` | 0.167 |
| passport_0099_seedream.png | `612-793` | `012-793` | 0.167 |

### passport_issued_full
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0068_seedream.png | `ТП УФМС РОССИИ ПО РЕСПУБЛИКЕ ТАТАРСТАН В Г. ЗЕЛЕНОДОЛЬСК` | `` | 1.000 |
| passport_0072_seedream.png | `МЕЖРАЙОННЫЙ ОТДЕЛ УФМС ПО САРАТОВСКОЙ ОБЛ. В ОКТЯБРЬСКОМ Р-НЕ` | `` | 1.000 |
| passport_0079_seedream.jpeg | `МИГРАЦИОННЫЙ ПУНКТ ПО РОСТОВСКОЙ ОБЛ. В Г. ШАХТЫ` | `` | 1.000 |
| passport_0051_seedream.png | `ТП УФМС РОССИИ ПО Г. МОСКВЕ ПО РАЙОНУ КУНЦЕВО` | `РОССИИ У КУНЦЕВО` | 0.644 |
| passport_0004_seedream.png | `ОВМ МВД РОССИИ ПО ИРКУТСКОЙ ОБЛ. В Г. БРАТСК` | `ПО ОВТ УПТСН ОБЛ БРАЕСК` | 0.636 |
| passport_0017_seedream.png | `МЕЖРАЙОННЫЙ ОТДЕЛ УФМС ПО КЕМЕРОВСКОЙ ОБЛАСТИ - КУЗБАСС В Г. ПРОКОПЬЕВСК` | `МЕКРАЙОННЫЙ ОТДЕЛ УФМС В Г ПРОКОПЬЕВСК` | 0.486 |
| passport_0019_seedream.jpeg | `ОУФМС РОССИИ ПО ПЕРМСКОМУ КРАЮ И КОМИ-ПЕРМЯЦКОМУ ОКРУГУ В ДЗЕРЖИНСКОМ Р-НЕ` | `ОУФМС РОССИИ О ПЕРМСКОМУ АЕЗМ. КРАО РОССИИ И СЗМ-ЛСИПВЗЮ В ДЛЕРЕИНСКОМ Р-НЕ А` | 0.446 |
| passport_0074_seedream.png | `ОТДЕЛ ПО ВОПРОСАМ МИГРАЦИИ ПО ТЮМЕНСКОЙ ОБЛ. В ЛЕНИНСКОМ АДМ. ОКРУГЕ` | `СД ТО КОНРОСАМ МИГРАЦИИ ПО ТОМЕНСКОЙ ОБЛ КО` | 0.441 |
| passport_0090_seedream.png | `МИГРАЦИОННЫЙ ПУНКТ ПО РЕСП. БАШКОРТОСТАН В СОВЕТСКОМ Р-НЕ` | `МНТРАШОШЫН ЛУНТ ТО РЯЕС ТМАШОРТОСПА В ООНЕТСКОМ Р-НН` | 0.368 |
| passport_0070_seedream.png | `ОТДЕЛ УФМС РОССИИ ПО МОСКОВСКОЙ ОБЛ. В Г. ОРЕХОВО-ЗУЕВО` | `ОТДЕЛ УФМС РОССИЙ ПО МОСКОВСОЙ ОЕЛ ВТ ОГОВЛЕО` | 0.273 |

### series_and_number
| image | gt | pred | CER |
| --- | --- | --- | --: |
| passport_0027_seedream.png | `27 43 221035` | `` | 1.000 |
| passport_0068_seedream.png | `28 19 162535` | `1` | 0.900 |
| passport_0079_seedream.jpeg | `39 62 145767` | `49` | 0.900 |
| passport_0025_seedream.png | `19 11 580547` | `19` | 0.800 |
| passport_0078_seedream.png | `78 71 461888` | `11 19 1999` | 0.800 |
| passport_0070_seedream.png | `96 83 715643` | `63 75` | 0.600 |
| passport_0002_seedream.png | `85 45 948749` | `18 54 519487` | 0.400 |
| passport_0037_seedream.png | `99 47 681343` | `19 94 716813` | 0.400 |
| passport_0092_seedream.jpeg | `43 78 229592` | `13 22 9592` | 0.300 |
| passport_0010_seedream.png | `80 11 813328` | `18 01 181332` | 0.200 |
