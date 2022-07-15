# -*- coding: utf-8 -*-
import csv

with open("my_casme1.csv", "r") as f:
    reader = csv.reader(f)
    TP=0
    FP=0
    FN=0
    for row in reader:
        print(row)
        if(row[5]=="TP"):
            TP+=1
        elif(row[5]=="FP"):
            FP+=1
        else:
            FN+=1
    print(TP)
    print(FP)
    print(FN)
    print((2*TP)/(2*TP+FN+FP))


