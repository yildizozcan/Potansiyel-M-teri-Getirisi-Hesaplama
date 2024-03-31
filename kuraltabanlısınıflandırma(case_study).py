########################################################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
########################################################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

### Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("/content/persona.csv")

def check_df(dataset_name,dataframe, head=5):
  print("\033[31m" + "***************************** << " + str(dataset_name.upper()) + " DATASET" " >> *****************************" + "\033[0m")
  print("\033[33m" + "**************" + " SHAPE "+ "**************" + "\033[0m")
  print(dataframe.shape)
  print("\033[33m" + "**************" + " TYPES "+ "**************" + "\033[0m")
  print(dataframe.dtypes)
  print("\033[33m" + "**************" + " HEAD "+ "**************" + "\033[0m")
  print(dataframe.head(head))
  print("\033[33m" + "**************" + " TAIL "+ "**************" + "\033[0m")
  print(dataframe.tail(head))
  print("\033[33m" + "**************" + " NA "+ "**************" + "\033[0m")
  print(dataframe.isnull().sum())
  print("\033[33m" + "**************" + " QUANTILES "+ "**************" + "\033[0m")
  print(dataframe.describe([0, 0.50, 0.95, 0.99, 1]).T)

check_df("Persona", df)

### Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
def unique_counter(dataframe, col, plot=False):
  print("\033[35m" + "Unique değer listesi: " + "\033[30m" + str(dataframe[col].unique()))
  print("\033[35m" + "Unique değer sayısı: " + "\033[30m" + str(dataframe[col].nunique()))
  print(pd.DataFrame({"Count": df[col].value_counts()}))

  if plot:
      plt.figure(figsize=(4, 2)) # Grafik boyutlarını bu şekilde ayarlayabilirsiniz.
      sns.countplot(x=col,data=dataframe)
      plt.show(block=True)
unique_counter(df, "SOURCE", plot = True)

### Soru 3:Kaç unique PRICE vardır?

unique_counter(df, "PRICE", plot = True)

### Soru 4:Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
unique_counter(df, "PRICE", plot = True)

### Soru 5:Hangi ülkeden kaçar tane satış olmuş?

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")

### Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY")["PRICE"].sum()

### Soru 7: SOURCE türlerine göre satış sayıları nedir?

unique_counter(df, "SOURCE", plot = True)

### Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY")["PRICE"].mean()

### Soru 9:SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE").agg({"PRICE": "mean"})

### Soru 10: COUNTRY-SOURCE kırılımındaPRICE ortalamaları nedir?

df.pivot_table(values="PRICE",index=["SOURCE","COUNTRY"],aggfunc="mean")

#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################

df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"})

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
print(agg_df)

#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()

agg_df.reset_index(inplace=True)

#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"],bins=[0,18,23,30,40,agg_df["AGE"].max()],labels=['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())])

agg_df.head()

#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

new_df = agg_df.groupby('customers_level_based')['PRICE'].mean().reset_index()
new_df.head()

#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,

new_df["SEGMENT"]= pd.qcut(new_df["PRICE"], 4, labels=["D", "C", "B", "A"]) #küçükten büyüğe !!!
new_df.head(30)

new_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum"]}).sort_values("SEGMENT", ascending=False).reset_index()

#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_df.loc[(new_df["customers_level_based"] == "TUR_ANDROID_FEMALE_31_40"),:]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_df.loc[(new_df["customers_level_based"] == "FRA_IOS_FEMALE_31_40"),:]

new_df.head(10)