## DATASET
* The main dataset is Finnish Broadcasting Company's (YLE) corpuses covering the years 2013-2023.
* The corpus is filtered with existing tags. From the metadata tag in the beginning of each article, starting with 'text\ '... the main department "Politiikka", ie. politics is fetched. From these articles the leads are fetched to extract the topics. The leads are accessed also from sentence-tag, where the paragraph-type is lead. Leads were chosen to create a smaller, but sufficient subset of the whole corpus to allow us to cover such a long timeframe (10years). Leads were chosen over headlines, as headlines were repeated at times in subheadlines, which had the same tags. 
* Dataset of YLE corpuses are in vrt-files which consists a lot of information about language itself and thus this sort of information is filtered out. On the other hand, the files have lemmas of each words, which shortcuts stemming from preprocess phrase.

* Nordic Tweet Stream: Laitinen, Mikko, Jonas Lundberg, Magnus Levin & Rafael Martins. 2018. The Nordic Tweet Stream: A Dynamic Real-Time Monitor Corpus of Big and Rich Language Data, Proc. of Digital Humanities in the Nordic Countries 3rd Conference, Helsinki, Finland, March 7-9, 2018, CEUR-WS.org, online CEUR-WS.org/Vol-2084/short10.pdf


## PREPROCESS
*
*
*



## ANALYSIS

# TF/IDF results:

Loaded 10 files, 3831 documents
label
neutral     3303
negative     432
positive      96


# TF-IDF top 50 (all documents):
  kirjoittaa           0.0206
  toimittaja           0.0203
  **suomi**            0.0194  *(Finland)* keyword 1
  yle                  0.0178
  **hallitus**         0.0178 *(Government)* keyword 2
  arvioida             0.0117
  myös                 0.0116
  sanoa                0.0111
  vuosi                0.0110
  voida                0.0105
  **presidentti**      0.0104 *(President)* keyword 3
  uusi                 0.0103
  **venäjä**           0.0101 *(Russia)* keyword 4
  tutkija              0.0100
  **puolue**             0.0100 *(Party)* keyword 5
  **ulkopolitiikka**     0.0099 *(Foreign affairs)* keyword 6
  **talouspolitiikka**   0.0097 *(Economic policy)* keyword 7
  tehdä                0.0097
  pitää                0.0096
  **turvallisuuspolitiikka**  0.0094 *(security policy)* keyword 8
  **yhdysvalta**       0.0094 *(United States (of America))* keyword 9
  saada                0.0092
  maa                  0.0084
  pekka                0.0084
  **perussuomalainen**     0.0082 *(The Finns Party (refers to a politician))*  keyword 10         
  **pääministeri**         0.0081 *(Prime minister)*           
  **keskusta**             0.0081 *(Center party)*
  mieli                0.0081
  muu                  0.0079
  haluta               0.0078
  aika                 0.0074
  tulla                0.0073
  puheenjohtaja        0.0072
  **maahanmuuttopolitiikka** 0.0070 *(Immigration policy)*
  suuri                0.0067
  **talous**               0.0066 *(Economics)*
  kertoa               0.0066
  erikoinentoimittaja  0.0063
  suomalainen          0.0063
  **eurooppa**             0.0062 *(Europe)*
  professori           0.0059
  **eduskunta**            0.0059 *(Parliament)*
  kansainvälinen       0.0057
  **ilmastopolitiikka**    0.0056 *(Climate policy)*
  **kansanedustaja**       0.0054 *(Member of parliament)*
  muutos               0.0054
  niinistö             0.0053
  hannu                0.0052
  **kokoomus**             0.0052 *(National Coalition Party)*
  asiantuntija         0.0052


# Additional information from the dataset: 


Top 50 most frequent words (all documents):
  politiikka             2025
  kirjoittaa              482
  toimittaja              453
  suomi                   392
  yle                     382
  hallitus                344
  arvioida                220
  myös                    187
  sanoa                   186
  voida                   182
  vuosi                   179
  presidentti             178
  venäjä                  172
  uusi                    171
  tutkija                 165
  puolue                  161
  pitää                   155
  tehdä                   154
  ulkopolitiikka          150
  turvallisuuspolitiikka    149
  saada                   147
  yhdysvalta              145
  talouspolitiikka        143
  pekka                   136
  maa                     135
  pääministeri            129
  perussuomalainen        129
  keskusta                128
  muu                     126
  puheenjohtaja           120
  aika                    116
  mieli                   116
  haluta                  115
  tulla                   110
  suuri                   104
  kertoa                   97
  talous                   95
  erikoinentoimittaja      91
  professori               90
  eurooppa                 88
  maahanmuuttopolitiikka     86
  suomalainen              86
  eduskunta                83
  kansainvälinen           83
  kansanedustaja           81
  niinistö                 78
  kokoomus                 77
  hannu                    75
  kinnunen                 74
  tikkala                  71


TF-IDF top 50 (positive):
  hyvä                 0.0463
  hyvin                0.0286
  **ulkopolitiikka**       0.0228
  voida                0.0183
  aika                 0.0181
  **hallitus**             0.0176
  kehua                0.0166
  uusi                 0.0164
  suomalainen          0.0161
  **suomi**                0.0158
  antaa                0.0156
  tehdä                0.0155
  **presidentti**          0.0151
  saada                0.0148
  **saksa**                0.0141
  vuosi                0.0138
  **talous**               0.0137
  oppitunti            0.0135
  maa                  0.0132
  päästä               0.0130
  kirjoittaa           0.0129
  tyytyväinen          0.0123
  sanoa                0.0121
  myös                 0.0116
  nainen               0.0113
  **poliitikko**           0.0112
  huippu               0.0112
  linja                0.0111
  kiitellä             0.0110
  **eurooppa**             0.0107
  sopia                0.0107
  muu                  0.0106
  yksi                 0.0105
  hieno                0.0104
  tukea                0.0104
  tylsä                0.0104
  muassa               0.0100
  petteri              0.0099
  työ                  0.0098
  **pääministerikisa**     0.0098
  asiantuntija         0.0097
  **putin**                0.0096
  forssa               0.0096
  asia                 0.0096
  ilmiö                0.0096
  pitää                0.0095
  tuore                0.0095
  tutkija              0.0093
  draamasarja          0.0093
  tulos                0.0093

TF-IDF top 50 (neutral):
  toimittaja           0.0219
  kirjoittaa           0.0215
  **suomi**                0.0197
  yle                  0.0197
  **hallitus**             0.0163
  arvioida             0.0129
  myös                 0.0119
  vuosi                0.0114
  uusi                 0.0109
  tutkija              0.0108
  **presidentti**          0.0107
  **venäjä**               0.0107
  **turvallisuuspolitiikka** 0.0107
  sanoa                0.0105
  voida                0.0105
  **puolue**               0.0102
  **talouspolitiikka**     0.0100
  **yhdysvalta**           0.0100
  tehdä                0.0098
  **ulkopolitiikka**       0.0096
  pitää                0.0095
  haluta               0.0090
  pekka                0.0089
  saada                0.0085
  **pääministeri**         0.0084
  muu                  0.0084
  **keskusta**             0.0083
  maa                  0.0077
  **perussuomalainen**     0.0077
  tulla                0.0075
  aika                 0.0073
  suuri                0.0071
  **talous**               0.0068
  **maahanmuuttopolitiikka** 0.0068
  erikoinentoimittaja  0.0068
  kertoa               0.0067
  puheenjohtaja        0.0067
  **eduskunta**            0.0064
  **eurooppa**             0.0063
  kansainvälinen       0.0063
  muutos               0.0061
  mieli                0.0061
  **ilmastopolitiikka**    0.0059
  robert               0.0058
  hannu                0.0058
  sundman              0.0057
  vaikuttaa            0.0057
  tikkala              0.0057
  asiantuntija         0.0056
  osa                  0.0056

TF-IDF top 50 (negative):
  **hallitus**             0.0290
  mieli                0.0229
  henkilöstöpolitiikka 0.0193
  arvostella           0.0180
  **suomi**                0.0175
  kirjoittaa           0.0152
  sanoa                0.0148
  saada                0.0134
  **perussuomalainen**     0.0125
  maa                  0.0123
  puheenjohtaja        0.0121
  toimittaja           0.0117
  suomalainen          0.0109
  syyttää              0.0107
  pitää                0.0104
  vastaan              0.0100
  **maahanmuuttopolitiikka** 0.0099
  professori           0.0097
  **ulkopolitiikka**       0.0096
  myös                 0.0095
  puolue               0.0091
  voida                0.0087
  **talouspolitiikka**     0.0083
  **keskusta**             0.0082
  **trumpin**              0.0082
  osoittaa             0.0078
  protestoida          0.0075
  kritisoida           0.0074
  vuosi                0.0074
  tehdä                0.0072
  **pääministeri**         0.0072
  mielenilmaus         0.0071
  epäonnistua          0.0071
  vastalause           0.0071
  **pakolainenpolitiikka** 0.0068
  helsinki             0.0068
  **venäjä**               0.0068
  huonosti             0.0068
  liian                0.0068
  toimia               0.0068
  **kansalainen**          0.0068
  timo                 0.0066
  **presidentti**          0.0065
  yle                  0.0065
  **kansanedustaja**       0.0064
  yrittäjä             0.0064
  pettyä               0.0064
  vastustaa            0.0064
  **leikkauspolitiikka**   0.0061
  **kunnallinenpolitiikka** 0.0061

Leads: 96 positive, 3303 neutral, 432 negative

Top 50 most frequent words. summary:
  word                  total    pos    neu    neg
  **politiikka** *(politics)*           2025     50   1773    202
  kirjoittaa              482      7    433     42
  toimittaja              453      3    421     29
  **suomi**  *(Finland)*             392      8    347     37
  yle                     382      2    363     17
  **hallitus**                344      9    272     63
  arvioida                220      3    208      9
  myös                    187      5    167     15
  sanoa                   186      6    152     28
  voida                   182      8    159     15
  vuosi                   179      6    160     13
  **presidentti** *(president)*    178      6    160     12
  **venäjä** *(Russia)*      172      1    160     11
  uusi                    171      8    156      7
  tutkija                 165      4    153      8
  **puolue**     *(party)*             161      3    141     17
  pitää                   155      4    134     17
  tehdä                   154      8    133     13
  **ulkopolitiikka**  *(foreign affairs)*        150      9    128     13
  **turvallisuuspolitiikka**  *(foreign and security policy)*  149      1    146      2
  saada                   147      6    120     21
  **yhdysvalta**   *(United States (of America))*           145      1    134     10
  **talouspolitiikka**  *(Economic policy)*      143      3    127     13
  pekka                   136      3    125      8
  maa                     135      6    108     21
  **pääministeri**            129      1    115     13
  **perussuomalainen**        129      2    106     21
  **keskusta**                128      1    112     15
  muu                     126      5    114      7
  puheenjohtaja           120      2     96     22
  aika                    116      7     98     11
  mieli                   116      4     78     34
  haluta                  115      0    114      1
  tulla                   110      2     98     10
  suuri                   104      2     95      7
  kertoa                   97      2     86      9
  **talous**                   95      5     85      5
  erikoinentoimittaja      91      2     84      5
  professori               90      1     74     15
  **eurooppa**                 88      4     77      7
  **maahanmuuttopolitiikka**     86      0     72     14
  suomalainen              86      5     66     15
  eduskunta                83      1     77      5
  **kansainvälinen**           83      1     78      4
  **kansanedustaja**           81      1     70     10
  niinistö                 78      2     70      6
  **kokoomus**                 77      2     67      8
  hannu                    75      0     71      4
  kinnunen                 74      1     70      3
  tikkala                  71      0     69      2



