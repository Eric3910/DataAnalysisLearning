# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:38:44 2018

@author: lenovo

使用TF-IDF矩阵对章节进行聚类

余弦相似：是指通过测量两个向量的夹角的余弦值来度量它们之间的相似性。
当两个文本向量夹角余弦等于1时，这两个文本完全重复；
当夹角的余弦值接近于1时，两个文本相似；夹角的余弦越小，两个文本越不相关。

k-means聚类：对于给定的样本集A，按照样本之间的距离大小，
将样本集A划分为K个簇A_1,A_2,⋯,A_K。
让这些簇内的点尽量紧密的连在一起，而让簇间的距离尽量的大。
K-Means算法是无监督的聚类算法。
目的是使得每个点都属于离它最近的均值（此即聚类中心）对应的簇A_i中。
这里的聚类分析使用的是nltk库。

下面的程序将使用k-means聚类算法对数据进行聚类分析，然后得到每一章所属类别，
并用直方图展示每一类有多少个章节。

  MDS降维、PCA降维、HC聚类


#【1】加载数据包及数据整理
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import jieba   #需要安装：pip install jieba
from pandas import read_csv
from scipy.cluster.hierarchy import dendrogram,ward
from scipy.spatial.distance import pdist,squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import nltk
from nltk.cluster.kmeans import KMeansClusterer

## 设置字体和 设置pandas显示方式
font=FontProperties(fname = "C:/Windows/Fonts/Hiragino Sans GB W3.otf",size=14)

pd.set_option("display.max_rows",8)
pd.options.mode.chained_assignment = None  # default='warn'

## 读取停用词和需要的词典
stopword = read_csv(r"C:\Users\yubg\OneDrive\2018book\syl-hlm\my_stop_words.txt",header=None,names = ["Stopwords"])
mydict = read_csv(r"C:\Users\yubg\OneDrive\2018book\syl-hlm\red_dictionary.txt",header=None, names=["Dictionary"])

RedDream = read_csv(r"C:\Users\yubg\OneDrive\2018book\syl-hlm\red_UTF82.txt",header=None,names = ["Reddream"])


#删除空白行和不需要的段，并重新设置索引
np.sum(pd.isnull(RedDream))  #查看数据是否有空白的行，如有则删除
indexjuan = RedDream.Reddream.str.contains("^第+.+卷") # 删除卷数据，使用正则表达式，包含相应关键字的索引
RedDream = RedDream[~indexjuan].reset_index(drop=True) ## 删除不需要的段，并重新设置索引


## 找出每一章节的头部索引和尾部索引
## 每一章节的标题
indexhui = RedDream.Reddream.str.match("^第+.+回")
chapnames = RedDream.Reddream[indexhui].reset_index(drop=True)

## 处理章节名，按照空格分割字符串
chapnamesplit = chapnames.str.split(" ").reset_index(drop=True)

## 建立保存数据的数据表
Red_df=pd.DataFrame(list(chapnamesplit),columns=["Chapter","Leftname","Rightname"])
## 添加新的变量
Red_df["Chapter2"] = np.arange(1,121)
Red_df["ChapName"] = Red_df.Leftname+","+Red_df.Rightname
## 每章的开始行（段）索引
Red_df["StartCid"] = indexhui[indexhui == True].index
## 每章的结束行数
Red_df["endCid"] = Red_df["StartCid"][1:len(Red_df["StartCid"])].reset_index(drop = True) - 1
Red_df["endCid"][[len(Red_df["endCid"])-1]] = RedDream.index[-1]
## 每章的段落长度
Red_df["Lengthchaps"] = Red_df.endCid - Red_df.StartCid
Red_df["Artical"] = "Artical"

## 每章节的内容
for ii in Red_df.index:
    ## 将内容使用""连接
    chapid = np.arange(Red_df.StartCid[ii]+1,int(Red_df.endCid[ii]))
    ## 每章节的内容替换掉空格
    Red_df["Artical"][ii] = "".join(list(RedDream.Reddream[chapid])).replace("\u3000","")
## 计算某章有多少字
Red_df["lenzi"] = Red_df.Artical.apply(len)


## 对红楼梦全文进行分词
## 数据表的行数
row,col = Red_df.shape
## 预定义列表
Red_df["cutword"] = "cutword"
for ii in np.arange(row):
    ## 分词
    cutwords = list(jieba.cut(Red_df.Artical[ii], cut_all=True))
    ## 去除长度为1的词
    cutwords = pd.Series(cutwords)[pd.Series(cutwords).apply(len)>1]
    ## 去停用此
    cutwords = cutwords[~cutwords.isin(stopword)]
    Red_df.cutword[ii] = cutwords.values


## 保存数据
Red_df.to_json(r"C:\Users\yubg\OneDrive\Red_dream_data.json")



'''
#【2】使用夹角余弦距离进行k均值聚类

## 准备工作，将分词后的结果整理成CountVectorizer（）可应用的形式
## 将所有分词后的结果使用空格连接为字符串，并组成列表，每一段为列表中的一个元素
'''
articals = []
for cutword in Red_df.cutword:
    articals.append(" ".join(cutword))
## 构建语料库，并计算文档－－词的TF－IDF矩阵
vectorizer = CountVectorizer()    
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(articals)

## tfidf 以稀疏矩阵的形式存储，将tfidf转化为数组的形式,文档－词矩阵
dtm = tfidf.toarray()

## 使用夹角余弦距离进行k均值聚类
kmeans = KMeansClusterer(num_means=3,       #聚类数目
                         distance=nltk.cluster.util.cosine_distance,  #夹角余弦距离
                         )
kmeans.cluster(dtm)

## 聚类得到的类别
labpre = [kmeans.classify(i) for i in dtm]
kmeanlab = Red_df[["ChapName","Chapter"]]
kmeanlab["cosd_pre"] = labpre
kmeanlab


## 查看每类有多少个分组
count = kmeanlab.groupby("cosd_pre").count()

## 将分类可视化
count.plot(kind="barh",figsize=(6,5))
for xx,yy,s in zip(count.index,count.ChapName,count.ChapName):
    plt.text(y =xx-0.1, x = yy+0.5,s=s)
plt.ylabel("cluster label")
plt.xlabel("number")
plt.show()    #显示如图8-16所示

'''
【3】MDS降维
多维标度（Multidimensional scaling，缩写MDS，又译“多维尺度”）
也称作“相似度结构分析”（Similarity structure analysis），属于多重变量分析的方法之一，
是社会学、数量心理学、市场营销等统计实证分析的常用方法。
MDS在降低数据维度的时候尽可能的保留样本之间的相对距离。
'''
## 聚类结果可视化
## 使用MDS对数据进行降维
#from sklearn.manifold import MDS
mds = MDS(n_components=2,random_state=123)
coord = mds.fit_transform(dtm)
print(coord.shape)
## 绘制降维后的结果
plt.figure(figsize=(8,8))
plt.scatter(coord[:,0],coord[:,1],c=kmeanlab.cosd_pre)
for ii in np.arange(120):
    plt.text(coord[ii,0]+0.02,coord[ii,1],s = Red_df.Chapter2[ii])
plt.xlabel("X")   
plt.ylabel("Y")  
plt.title("K-means MDS")  
plt.show()  #显示如图8-17所示
#针对在MDS下每个章节的相对分布情况，
#章节之间没有很明显的分界线（因为这是一本书，讲的是一个故事）,
#但并不是说我们根据章节聚类分析是没有意义的，因为每一个章节都是不一样的，
#而且相互之间的联系也是不同的。


"""
 【4】PCA降维
 #PCA降维
##是一种常见的数据降维方法，其目的是在“信息”损失较小的前提下，
将高维的数据转换到低维，从而减小计算量。
PCA通常用于高维数据集的探索与可视化，还可以用于数据压缩，数据预处理等。

"""
## 聚类结果可视化
## 使用PCA对数据进行降维
#from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(dtm)
print(pca.explained_variance_ratio_)
## 对数据降维
coord = pca.fit_transform(dtm)
print(coord.shape)
## 绘制降维后的结果
plt.figure(figsize=(8,8))
plt.scatter(coord[:,0],coord[:,1],c=kmeanlab.cosd_pre)
for ii in np.arange(120):
    plt.text(coord[ii,0]+0.02,coord[ii,1],s = Red_df.Chapter2[ii])
plt.xlabel("主成分1",FontProperties = font)   
plt.ylabel("主成分2",FontProperties = font)  
plt.title("K-means PCA")  
plt.show()   #显示如图8-18所示

"""
 【5】HC聚类
 #HC聚类(Hierarchical Clustering，层次聚类)是聚类算法的一种，
 通过计算不同类别数据点间的相似度来创建一棵有层次的嵌套聚类树。
 在聚类树中，不同类别的原始数据点是树的最低层，树的顶层是一个聚类的根节点。

"""
## 层次聚类
#from scipy.cluster.hierarchy import dendrogram,ward
#from scipy.spatial.distance import pdist,squareform
## 标签，每个章节的标题
labels = Red_df.Chapter.values
cosin_matrix = squareform(pdist(dtm,'cosine'))#计算每章的距离矩阵
ling = ward(cosin_matrix)  ## 根据距离聚类
## 聚类结果可视化
fig, ax = plt.subplots(figsize=(10, 15)) # 设置大小
ax = dendrogram(ling,orientation='right', labels=labels);
plt.yticks(FontProperties = font,size = 8)
plt.title("《红楼梦》各章节层次聚类",FontProperties = font)
plt.tight_layout() # 展示紧凑的绘图布局
plt.show()   #显示如图8-19所示





