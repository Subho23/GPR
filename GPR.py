# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:05:03 2017

@author: Subhadip Dey
"""

import numpy as np
import math
from scipy.optimize import minimize
import xlrd
from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
from random import randint
import matplotlib.pyplot as plt


 
global mat1
global Y
global row
X = []
Y = []
def mat_form(X):
    global mat1
    global row
    mat1 = np.zeros((row,row));
    for i in range(0,row):
        for j in range(0,row):
            x = X[i]
            y = X[j]
            z = x-y
            z_ = z.T
            mat1[i,j] = z*z_
    return(mat1)
    
def hyp_learn(v):
    global mat1
    global Y
    global row
    new_mat = np.zeros((row,row))
    k = Y.T
    for i in range(0,row):
        for j in range(0,row):
            new_mat[i,j] = (v[0]**2)*math.exp(-(mat1[i,j]/(2*(v[1]**2))))
    s = k*np.linalg.inv(new_mat)
    s = s*Y
    t = np.linalg.det(new_mat)
    t = math.log(t)
    out = t+s+(math.log(2*math.pi))
    out = out*0.5
    return(out)

#row = input('No of rows - > ')
#col = input('No of Xcols - > ')
#tot_Xdata = row*col
#X1 = raw_input('Enter X-data seperated by , - > ')
#t1 = X1.split(',')
root = Tkinter.Tk()
fname =  tkFileDialog.askopenfilename(filetypes=(("Input Excel File", "*.xlsx;*.xls"),("All files", "*.*") ))
root.destroy()
xl_workbook = xlrd.open_workbook(fname)
sheet_names = xl_workbook.sheet_names()
xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])
row = xl_sheet.nrows
col = xl_sheet.ncols-1
for i in range(0,row):
	for j in range(0,col):
		t = xl_sheet.row(i)
		X.append(float(t[j].value))
	Y.append(float(t[j+1].value))

X = np.array(X,dtype = np.float64)
X = np.resize(X,(row,col))
X = np.matrix(X)
print("X - > ")
print(X)


Y = np.array(Y,dtype = np.float64)
Y = np.reshape(Y,(row,1))
Y = np.matrix(Y)
print('Y - > ')
print(Y)
t = mat_form(X)
v0 = [20,20]
r = minimize(hyp_learn,v0,method = 'CG')

print('KernelInformation:')
print('         Squared Exponential Kernel - >')
print('         (nu^2)*exp(-(x1-x2)^2/(2*lambda^2))')
print('\n')
print('Kernel Parameters:')
global nu 
global sigma
nu  = r.x[0]
print('nu -> ')
print(nu)
sigma = r.x[1]
print('sigma -> ')
print(sigma)

#--- Prediction -------################################################
print('Prediction module started!')
root1 = Tkinter.Tk()
fname1 =  tkFileDialog.askopenfilename(filetypes=(("Test Excel file", "*.xlsx;*.xls"),("All files", "*.*") ))
root1.destroy()
xl_workbook1 = xlrd.open_workbook(fname1)
sheet_names1 = xl_workbook1.sheet_names()
xl_sheet1 = xl_workbook1.sheet_by_name(sheet_names[0])
row1 = xl_sheet1.nrows
col1 = xl_sheet1.ncols
x_ = []
for i in range(0,row1):
	for j in range(0,col1):
		t1 = xl_sheet1.row(i)
		x_.append(float(t1[j].value))
x_ = np.array(x_)
x_ = np.reshape(x_,(row1,col1))
x_ = np.matrix(x_)
print(x_)
Xnew = []
mean_lis = []
var_new_lis = []

for u in range(0,row1):
    global col1
    Xnew = []
    x_new = x_[u]
    row2 = x_new.shape[0]
    
    for j in range(0,row2):
        for i in range(0,col1):
    
    Xnew = np.array(Xnew, dtype = np.float64)
    Xnew = np.reshape(Xnew,(1,col))
    Xnew = np.matrix(Xnew)
    #---------- formation of XNew-------------------
    global new_mat2
    new_mat2 = []
    def new_k_form():
        global mat1
        global Y
        global row
        global nu
        global sigma
        new_mat2 = np.zeros((row,row))
        for i in range(0,row):
            for j in range(0,row):
                new_mat2[i,j] = (nu**2)*math.exp(-(mat1[i,j]/(2*(sigma**2))))
        #print(new_mat2)
        return(new_mat2)
    #--------- formation of inv(K) for prediction ---------
    def k_star(X,Xnew):
        global row
        mat4 = np.zeros((row,1))
        for i in range(0,row):
            w = X[i]
            u = Xnew
            q = w-u
            q_ = q.T
            mat4[i] = q*q_
        return(mat4)
    #-------------
    def new_kstar_form(mat4):
        global mat1
        global Y
        global row
        global nu
        global sigma
        new_mat4 = np.zeros((row,1))
        for i in range(0,row):
            for j in range(0,1):
                new_mat4[i,j] = (nu**2)*math.exp(-(mat4[i,j]/(2*(sigma**2))))
        return(new_mat4)
    #----------------------- formation of k*--------------------
    #mean = 0
    p1 = new_k_form()
    p1 = np.matrix(p1)
    p5 = k_star(X,Xnew)
    p2 = new_kstar_form(p5)
    p2 = np.matrix(p2)
    #print(np.linalg.inv(p1))
    #print(p2.T)
    #print(Y)
    k = p2.T
    #print(k)
    s = k*np.linalg.inv(p1)
    s = s*Y
    mean = s + 0
    #print('The predicted value is - >')
    #print(mean)
    mean_lis.append(float(mean))
    #print(k*np.linalg.inv(p1)*p2)
    var_new = float("{0:.2f}".format(float(nu**2)))- float("{0:.2f}".format(float(k*np.linalg.inv(p1)*p2)))
    #print('Predicted variance - > ')
    #print(var_new)
    var_new_lis.append(float(var_new))
    Xnew = []
mean_lis = np.array(mean_lis)
var_new_lis = np.array(var_new_lis)
print('The predicted value is - >')
print(mean_lis)
print('Predicted variance - > ')
print(var_new_lis)
num = []
for i in range(0,row1):
    num.append(i)
num = np.array(num)
actual = [3,9,6,5,11,6,9,5,7,10,11,3,5,9,8.9,6.7,11.4,10.5,11.4,4.8] #actual y value of test samples(the values are for the example excel)
actual = np.array(actual)

#print(num)
plt.subplot(2, 1, 1)
plt.plot(num,mean_lis, 'ko-')
plt.plot(num,actual, 'g^')
plt.title('Predicted value and variance')
plt.ylabel('Predicted value')

plt.subplot(2, 1, 2)
plt.plot(num,var_new_lis, 'r.-')
plt.xlabel('Number of predictive samples')
plt.ylabel('Variance')
#plt.plot(num,mean_lis)
#plt1.plot(num,variance_new_lis)
#plt.fill_between(num, mean_lis-var_new_lis, mean_lis+var_new_lis)
plt.show()
