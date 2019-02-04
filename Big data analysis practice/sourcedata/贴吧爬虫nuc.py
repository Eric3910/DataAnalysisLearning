#encoding:UTF-8  

import urllib.request
from bs4 import BeautifulSoup 
import time

print("***\n***\n***\n这是一个爬虫，正在爬取百度贴吧的一个内容，请耐心等候：。。。")
f = open('nuc_pachong.txt','a+',encoding='utf-8')     #打开文件,a+表示在文件末尾追加
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))#当前的时间
f.write("【时间："+end_time+"】\n【标题】中北大学贴吧"+'\n')


url="http://tieba.baidu.com/f?ie=utf-8&kw=%E4%B8%AD%E5%8C%97%E5%A4%A7%E5%AD%A6"  
html=urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, "lxml")
               #print(soup.prettify())#打印格式
all = soup.find_all("a",class_="j_th_tit ")#查找标签中的class_="zm-editable-content clearfix"，由于class在python中是保留变量名，所以class改写成class_;limit表示限制输出的数量
print(all)
ALL=str(all).split('</a>')
ALL.pop()    #如果删除最后一元素“]”后面会报错

i=0
for s in ALL:    
    q,w = s.split('title="')
    i+=1
    f.write('【标题'+str(i)+'】：'+ w +'\n') 
                    
f.close()
print("***\n***\n***\n恭喜你，已经完成任务，请你打开文件：nuc_pachong.txt查阅")


