## IDEA:
* count 越多越准确
* useful 越多越可信

## problem:
* test 中 review如何处理。
    * load_data中data2的user_reviews, user_rid的处理。
    * rid 取0是很奇怪的事情。
* tokenize
    * clean str 去掉数字会丢失信息。
* 是不是可以最后再pad
* xavier_init
* 设频率下限，过滤cold start。后面研究cold start可能要去掉。 min_count

* word2vec miss : miss: 0.4034938790099187

* encoding in data_pro.

## TODO:
* 看完代码
* 整合两个模型
* 运行NCF
* 跑yelp

## statics:
* music:
    * 64706/5541/3568 r/u/i
* yelp_academic:
    * 3874548/252332/134295 r/u/i (filtered)
    * 5996996 reviews.
* yelp13:
    * 预处理：
        45981 11537 (u, b)
        36473 4503 (u, b) res
        total_review_count: 229907
        res_review_count: 158430
    *  length
        u_len (user_review max num): 8 470
        i_len (item_review max num): 39 679
        u2_len (user_review max length): 274 1023
        i2_len (item_review max length): 275 1023

* yelp13res:
    * 158430/36473/4503 r/u/i
    * length:
        u_len (user_review max num): 7 258
        i_len (item_review max num): 72 651
        u2_len (user_review max length): 275 982
        i2_len (item_review max length): 277 982
    * vocab
        Vocabulary_user: 57190
        vocabulary_item: 66356

* yelp18:
    * 2742802/100685/88274


* yelp13res_filtered:
    * 10: 79304/2992/2555 r/u/i
    * 5: 107929/6863/3516
    * 2: 140159/18202/4499/140159 r/u/i


## 10/1:
* music:
    * NCF: EPOCH7: loss_valid 25.6588, rmse_valid 0.895825, mae_valid 0.665677
    * without-w2v:
        * NARRE: EPOCH6: loss_valid 12.6256, rmse_valid 0.8882, mae_valid 0.662897
        * DeepCoNN: EPOCH13: loss_valid 39.2713, rmse_valid 0.887416, mae_valid 0.661833
    * with-google embeddings:
        * NARRE: EPOCH13: loss_valid 24.8737, rmse_valid 0.881521, mae_valid 0.649733
        * DeepCoNN: EPOCH5： loss_valid 38.7615, rmse_valid 0.881569, mae_valid 0.648935

## 10/4
* yelp13:
    * NCF: EPOCH1: loss_valid 40.0915, rmse_valid 1.11948, mae_valid 0.890194
    * DeepCoNN: 4: loss_valid 21.3763, rmse_valid 1.15599, mae_valid 0.899932
    * DeepCONN++: 2: loss_valid 20.5105, rmse_valid 1.13232, mae_valid 0.888792
    * Narre: 5: loss_valid 20.5739, rmse_valid 1.13374, mae_valid 0.887212


* yelp13res:
    * NCF: EPOCH1: loss_valid 39.4458, rmse_valid 1.11073, mae_valid 0.891486
        * new: best rmse: 1.11054817414419 best mae: 0.8821973856415086
    * DeepCoNN: 2: loss_valid 21.6146, rmse_valid 1.16244, mae_valid 0.896651
    * DeepCoNNpp: 3: loss_valid 20.5306, rmse_valid 1.1329, mae_valid 0.875925
    * NARRE: 3: loss_valid 19.9772, rmse_valid 1.11719, mae_valid 0.87437



## 10/7:
* problem for ABSA:
    * 词表问题
    * clean_str tokenize 问题
    * multi-label问题
    * 空的问题。


## 1/10:
* aspect embedding 和 polarity embedding

* 简单计数+[拼接+线性relu融合]:
   * softmax归一化:
       * 1.1112, 0.8766
   * 频率归一：
       * 1.1174, 0.8840
* 按照aspect做softmax次数平均:

* 按照aspect做打分平均（-1，0，1）:
   * 拼接+线性relu融合:
       * best rmse: 1.1140314668 best mae: 0.879302608352
       * best rmse: 1.11489018911 best mae: 0.876370998947
       * best rmse: 1.1139460043846106 best mae: 0.8792366478612144
       * best rmse: 1.11091958688426 best mae: 0.876896386600117 [0, 1, 2]
       * best rmse: 1.1097404586886435 best mae: 0.8749051000752444 [0, 1, 2]
       * best rmse: 1.10854328582556 best mae: 0.8765231094591802 [1,2,3]
       * best rmse: 1.1101711581424445 best mae: 0.8762869149802016 [1,3,5]

   * 拼接+直接內积:
       * best rmse: 1.13779112882 best mae: 0.91998384701
       * best rmse: 1.13590597067 best mae: 0.917011440143 [0, 1, 2]
   * 拼接+拼接+MLP打分。

* 按照aspect做打分softmax平均（-1，0，1）:
    * best rmse: 1.1110142297953043 best mae: 0.8758883039465097 [0, 1, 2]
    * best rmse: 1.1092573675588477 best mae: 0.876110379348114 [1, 2, 3]
    * best rmse: 1.1087419637905396 best mae: 0.8721142188923185 [1, 3, 5]
    * best rmse: 1.1102226802677362 best mae: 0.8763017051224304 [1, 3, 5]
    * best rmse: 1.108175249046467 best mae: 0.8726302962678479 [1, 3, 5]
    * best rmse: 1.1102173096191816 best mae: 0.875620398660324 [1, 3, 5]
    * best rmse: 1.1097655497238983 best mae: 0.877002331939361 [1, 10, 100]

* 去掉CF:
    * 直接multiply:
        * best rmse: 1.09899970753 best mae: 0.86494510636
            * 去掉feature_bias:
                * best rmse: 1.1926502112 best mae: 0.942961617269
    * 接线性relu再multiply：
        * best rmse: 1.10039485323 best mae: 0.858107059948

!!!! * 去掉score, 只用feature_bias和global_bias:
        * best rmse: 1.0995682287 best mae: 0.871245739899

## 1/11
* 平均归一化不如softmax归一化
* [-1, 0, 1]不如[0, 1, 2]
* 通过对item_feature的调整，还是能够感觉到就是说这个东西还是在起作用的。
* 在None上面的处理可能也要慎重。
* 输入特征里面有0的话，可能对于神经网络会很特殊。
* 去掉feature_bias会很差。


## NCF:
* NARRE version:
    * best rmse: 1.10944693625 best mae: 0.882613162538
    * best rmse: 1.110414289  best mae: 0.883364365008

* 加双层非线性relu
    * best rmse: 1.11537556214  best mae: 0.891271160207
    * best rmse: 1.1173104984  best mae: 0.895717488318




* remove score
    * best rmse: 1.099777 58573  best mae: 0.871245739899
* remove feature_bias:
    * best rmse: 1.11246510159  best mae: 0.883299679488
    * best rmse: 1.10978311121  best mae: 0.885333538157

* remove dropout

* Adam

## APNCF:
* [feas + id]:
    * best rmse: 1.10968738106  best mae: 0.873444458251
* [id]
    * best rmse: 1.10753023809  best mae: 0.880256619762
    * direct multiply:
        * best rmse: 1.14554724806  best mae: 0.928239217

* [feas]
    * best rmse: 1.10718670151  best mae: 0.854195805855
    * best rmse: 1.10699080926  best mae: 0.855522247635
    * best rmse: 1.09921067285  best mae: 0.859842446456 [add dropout]
    * best rmse: 1.0999415369  best mae: 0.858709210467
    * best rmse: 1.10726737995  best mae: 0.856189407823 [without dropout]
* remove score:
    * best rmse: 1.0995682287  best mae: 0.871245739899
* remove Feature_bias:
    * [feas + id]:
        * best rmse: 1.11759726115 best mae: 0.881247679498
    * [id]
        * best rmse: 1.10824262223 best mae: 0.882378042147
        * best rmse: 1.10878757185  best mae: 0.881642073474
        * best rmse: 1.10743600815  best mae: 0.878885178842
        * best rmse: 1.10685644629  best mae: 0.880267234675

        * direct multiply:
            * best rmse: 1.16669878244  best mae: 0.954073443878

    * [feas]
        * best rmse: 1.1585531606  best mae: 0.899659761476
        * best rmse: 1.15914908176  best mae: 0.903173049069
        * best rmse: 1.15899465949  best mae: 0.900930254566

        * [new features]:
            * best rmse: 1.12803245314  best mae: 0.880210252401

        * direct multiply:
            * best rmse: 1.19192300187  best mae: 0.939746168789


## filtered10:
* [feas + id]:
    * best rmse: 0.996601155277  best mae: 0.782088150202
    * best rmse: 0.997322123692  best mae: 0.783135513039

* [id]
    * best rmse: 0.995675924029  best mae: 0.782775755322

* [feas]
    * best rmse: 0.993922556916  best mae: 0.782205565791
    * best rmse: 0.993902214855  best mae: 0.781259525859
* remove score:
    * best rmse: 0.994151321568  best mae: 0.784701468214
    * best rmse: 0.994151321568  best mae: 0.784701468214
* remove Feature_bias:
    * [feas + id]:
        * best rmse: 1.00146487092  best mae: 0.788324757482
        * best rmse: 1.00113105773  best mae: 0.781491911471
    * [id]
        * best rmse: 0.998826766755  best mae: 0.788155504509
        * direct multiply:
    * [feas]
        * best rmse: 1.06365534882  best mae: 0.838408136578

## filtered5:
* [feas + id]:
    * best rmse: 1.02578013222  best mae: 0.803840329138
    * best rmse: 1.0242355515  best mae: 0.803117889962
* [id]
    * best rmse: 1.02723830011  best mae: 0.802482369897
    * best rmse: 1.02669955228  best mae: 0.801835974606
* [feas]
    * best rmse: 1.01833516679  best mae: 0.794588049912
    * best rmse: 1.01776857569  best mae: 0.793654791653
* remove score:
    * best rmse: 1.01732122567  best mae: 0.79757274781
    * best rmse: 1.01732122567  best mae: 0.79757274781
* remove feature_bias:
    * [feas + id]
        * best rmse: 1.02641060612  best mae: 0.804433101931
        *
    * [id]
        * best rmse: 1.02695667831  best mae: 0.802761986988
    * [feas]
        * best rmse: 1.0842315362  best mae: 0.84635804047
            * best rmse: 1.07794901133  best mae: 0.833688305691 [proba no softmax in item_]
            * best rmse: 1.07795185127  best mae: 0.829911314446

    * [new feas]
        * best rmse: 1.05801568646  best mae: 0.825465411437

* score + fea_score

* proba feature.
* deep FM

## new feature:
*

##
* softmax
* feature* latent factors
* aspect sum (AALFM)
* deepFM
* 离散化。

* multi-hot embedding

* 对None对0的处理。
##
1. DeepFM
    * 实数特征
    * multi-hot 特征
    * 特征交叉。新特征。
2. ALFM
3. other design.


## yelp18
* NCF:
    * 100:
    * best rmse: 1.1273620059531775  best mae: 0.8714838332018631
    * best rmse: 1.1260847540094172  best mae: 0.8702920930794485
    * best rmse: 1.1270420061534028  best mae: 0.8676884008450082

    * 200:
        * best rmse: 1.1259788139742388  best mae: 0.8699050126104256
        * best rmse: 1.126045354640302  best mae: 0.8748896098595867
    * 128:
        * best rmse: 1.1257292566787256  best mae: 0.8719358031145074
        * best rmse: 1.125591219015093  best mae: 0.8738677131255201
    * 64:
        * best rmse: 1.1288295748040704  best mae: 0.8732019192139435
        * best rmse: 1.1286954127535636  best mae: 0.8735327576937264
    * 300:
        * best rmse: 1.1279614674043659  best mae: 0.8693170286448823
        * 
* without score:
    * 128:
        * best rmse: 1.1271231816956655  best mae: 0.8758579541699122
    * 160:
        * best rmse: 1.125397101502884  best mae: 0.8755585958419778
    