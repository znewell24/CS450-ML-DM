# -*- coding: utf-8 -*-
"""
Assignment 03 Prove
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

def car_process(data):
    x = data
    
    data[data.isnull().any(axis=1)]
    x.buying = x.buying.fillna("unknown")
    x.maint = x.maint.fillna("unknown")
    x.doors = x.doors.fillna("unknown")
    x.persons = x.persons.fillna("unknown")
    x.lug_boot = x.lug_boot.fillna("unknown")
    x.safety = x.safety.fillna("unknown")
    x.target = x.target.fillna("unknown")
    
    buying_to_num = {"vhigh": 4, "high": 3, "med": 2, "low": 1, "unknown": 0}
    x.buying = x.buying.replace(buying_to_num, inplace=True)
    
    maint_to_num = {"vhigh": 4, "high": 3, "med": 2, "low": 1, "unknown": 0}
    x.maint = x.maint.replace(maint_to_num, inplace=True)
    
    doors_to_num = {"2": 2, "3": 3, "4": 4, "5more": 5, "unknown": 0}
    x.doors = x.doors.replace(doors_to_num, inplace=True)
    
    persons_to_num = {"2": 2, "4": 4, "more": 6, "unknown": 0}
    x.persons = x.persons.replace(persons_to_num, inplace=True)
    
    lug_boot_to_num = {"small": 1, "med": 2, "big": 3, "unknown": 0}
    x.lug_boot = x.lug_boot.replace(lug_boot_to_num, inplace=True)
    
    safety_to_num = {"low": 1, "med": 2, "high": 3, "unknown": 0}
    x.safety = x.safety.replace(safety_to_num, inplace=True)
    
    target_to_num = {"unacc": 1, "acc": 2, "good": 3, "vgood": 4, "unknown": 0}
    x.target = x.target.replace(target_to_num, inplace=True)
    
    y = np.array(x[["target"]].values)
    x = np.array(x[["buying", "maint", "doors", "persons", "lug_boot", "safety"]].values)
    
    
    return x, y

data = pd.read_csv("car.csv", header=None, skip_blank_lines=True, 
                   names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"])

dataX, dataY = car_process(data)

X_train,X_test,y_train,y_test=train_test_split(dataX,dataY,test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

def mpg_process(data):
    x = data.drop('car_name', axis=1)
    
    data[data.isnull().any(axis=1)]
    x.mpg = x.mpg.fillna("unknown")
    x.cylinders = x.cylinders.fillna("unknown")
    x.displacement = x.displacement.fillna("unknown")
    x.horsepower = x.horsepower.fillna("unknown")
    x.weight = x.weight.fillna("unknown")
    x.acceleration = x.acceleration.fillna("unknown")
    x.model_year = x.model_year.fillna("unknown")
    x.origin = x.origin.fillna("unknown")

    
    
    mpg_to_num = {"unknown": 0}
    x.mpg = x.mpg.replace(mpg_to_num, inplace=True)
    
    cylinders_to_num = {"unknown": 0}
    x.cylinders = x.cylinders.replace(cylinders_to_num, inplace=True)
    
    displacement_to_num = {"unknown": 0}
    x.displacement = x.displacement.replace(displacement_to_num, inplace=True)
    
    horsepower_to_num = {"unknown": 0}
    x.horsepower = x.horsepower.replace(horsepower_to_num, inplace=True)
    
    weight_to_num = {"unknown": 0}
    x.weight = x.weight.replace(weight_to_num, inplace=True)
    
    acceleration_to_num = {"unknown": 0}
    x.acceleration = x.acceleration.replace(acceleration_to_num, inplace=True)
    
    model_year_to_num = {"unknown": 0}
    x.model_year = x.model_year.replace(model_year_to_num, inplace=True)
    
    origin_to_num = {"unknown": 0}
    x.origin = x.origin.replace(origin_to_num, inplace=True)   
    
    y = np.array(x[["mpg"]].values)
    x = np.array(x.drop('mpg', axis=1))
    
    return x, y

data1 = pd.read_csv("auto-mpg.csv", header=None, delim_whitespace=True, skip_blank_lines=True, 
                   names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"])

dataX1, dataY1 = mpg_process(data1)

X_train1,X_test1,y_train1,y_test1=train_test_split(dataX1,dataY1,test_size=0.3)

regr = KNeighborsRegressor(n_neighbors=3)
regr.fit(X_train1, y_train1)
predictions = regr.predict(X_test1)

def student_process(data):
    x = data
    
    data[data.isnull().any(axis=1)]
    x.school = x.school.fillna("unknown")
    x.sex = x.sex.fillna("unknown")
    x.age = x.age.fillna("unknown")
    x.address = x.address.fillna("unknown")
    x.famsize = x.famsize.fillna("unknown")
    x.Pstatus = x.Pstatus.fillna("unknown")
    x.Medu = x.Medu.fillna("unknown")
    x.Fedu = x.Fedu.fillna("unknown")
    x.Fjob = x.Fjob.fillna("unknown")
    x.reason = x.reason.fillna("unknown")
    x.guardian = x.guardian.fillna("unknown")
    x.traveltime = x.traveltime.fillna("unknown")
    x.studytime = x.studytime.fillna("unknown")
    x.failures = x.failures.fillna("unknown")
    x.famsup = x.famsup.fillna("unknown")
    x.paid = x.paid.fillna("unknown")
    x.activities = x.activites.fillna("unknown")
    x.nursery = x.nursery.fillna("unknown")
    x.higher = x.higher.fillna("unknown")
    x.internet = x.internet.fillna("unknown")
    x.romantic = x.romantic.fillna("unknown")
    x.famrel = x.famrel.fillna("unknown")
    x.freetime = x.freetime.fillna("unknown")
    x.goout = x.goout.fillna("unknown")
    x.Dalc = x.Dalc.fillna("unknown")
    x.Walc = x.Walc.fillna("unknown")
    x.health = x.health.fillna("unknown")
    x.absences = x.absences.fillna("unknown")
    x.G1 = x.G1.fillna("unknown")
    x.G2 = x.G2.fillna("unknown")
    x.G3 = x.G3.fillna("unknown")    
    
    school_to_num = {"GP": 1, "MS": 2, "unknown": 0}
    x.school = x.school.replace(school_to_num, inplace=True)
    
    sex_to_num = {"M": 1, "F": 2, "unknown": 0}
    x.sex = x.sex.replace(sex_to_num, inplace=True)
    
    address_to_num = {"U": 1, "R": 2, "unknown": 0}
    x.address = x.address.replace(address_to_num, inplace=True)
    
    famsize_to_num = {"LE3": 1, "GT3": 2, "unknown": 0}
    x.famsize = x.famsize.replace(famsize_to_num, inplace=True)
    
    Pstatus_to_num = {"T": 1, "A": 2, "unknown": 0}
    x.Pstatus = x.Pstatus.replace(Pstatus_to_num, inplace=True)
    
    Mjob_to_num = {"teacher": 1, "health": 2, "services": 3, "at_home": 4, "other": 5, "unknown": 0}
    x.Mjob = x.Mjob.replace(Mjob_to_num, inplace=True)
    
    Fjob_to_num = {"teacher": 1, "health": 2, "services": 3, "at_home": 4, "other": 5, "unknown": 0}
    x.Fjob = x.Fjob.replace(Fjob_to_num, inplace=True)
    
    reason_to_num = {"home": 2, "reputation": 3, "course": 4, "other": 5, "unknown": 0}
    x.reason = x.reason.replace(reason_to_num, inplace=True)   
    
    guardian_to_num = {"mother": 1, "father": 2, "other": 3, "unknown": 0}
    x.guardian = x.guardian.replace(guardian_to_num, inplace=True)\
    
    famsup = {"yes": 1, "no": 2, "unknown": 0}
    x.famsup = x.famsup.replace(famsup, inplace=True)
    
    paid = {"yes": 1, "no": 2, "unknown": 0}
    x.paid = x.paid.replace(paid, inplace=True)
    
    activites = {"yes": 1, "no": 2, "unknown": 0}
    x.activites = x.activites.replace(activites, inplace=True)
    
    nursery = {"yes": 1, "no": 2, "unknown": 0}
    x.nursery = x.nursery.replace(nursery, inplace=True)
    
    higher = {"yes": 1, "no": 2, "unknown": 0}
    x.higher = x.higher.replace(higher, inplace=True)
    
    internet = {"yes": 1, "no": 2, "unknown": 0}
    x.internet = x.internet.replace(internet, inplace=True)
    
    romantic = {"yes": 1, "no": 2, "unknown": 0}
    x.romantic = x.romantic.replace(romantic, inplace=True)
    
    y = np.array(x[["G3"]].values)
    x = np.array(x.drop('G3', axis=1))
    
    return x, y

data2 = pd.read_csv("student-mat.csv", sep=';')

dataX2, dataY2 = mpg_process(data2)

X_train2,X_test2,y_train2,y_test2=train_test_split(dataX2,dataY2,test_size=0.3)

regr1 = KNeighborsRegressor(n_neighbors=3)
regr1.fit(X_train2, y_train2)
predictions = regr1.predict(X_test2)