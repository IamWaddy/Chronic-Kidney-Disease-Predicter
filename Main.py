from tkinter import *
from tkinter.messagebox import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import smtplib
from smtplib import *
import ML


from PIL import Image, ImageTk
 

def Forests(spg,alb,sug,bgrd,sec,pota,pcvm,wcc,rcc,dms):
    df=pd.read_csv("kidney_disease1.csv")
    feature=list(df.columns[0:10])
    y=df["class"]
    x=df[feature]
    clf = RandomForestClassifier(n_estimators=8,max_depth=2,random_state=42)
    clf = clf.fit(x, y)
    b=clf.predict([[spg,alb,sug,bgrd,sec,pota,pcvm,wcc,rcc,dms]])
    return (b)

def Svms(spg,alb,sug,bgrd,sec,pota,pcvm,wcc,rcc,dms):
    df=pd.read_csv("kidney_disease1.csv")
    feature=list(df.columns[0:10])
    y=df["class"]
    x=df[feature]
    clf = SVC(gamma=0.001,C=10)
    clf = clf.fit(x, y)
    c=clf.predict([[spg,alb,sug,bgrd,sec,pota,pcvm,wcc,rcc,dms]])
    return (c)

def Get_Data():
    
    spg=float(sg.get())
    alb=float(al.get())
    sug=float(su.get())
    bgrd=float(bgr.get())
    sec=float(sc.get())
    pota=float(pot.get())
    pcvm=int(pcv.get())
    wcc=int(wc.get())
    rcc=float(rc.get())
    dms=dm.get()
    if dms=='yes' or dms=='YES' or dms=='Yes':
        dms=1
    else:
        dms=0

    f=Forests(spg,alb,sug,bgrd,sec,pota,pcvm,wcc,rcc,dms)
    return (f) 

def Get_Data1():

    spg=float(sg.get())
    alb=float(al.get())
    sug=float(su.get())
    bgrd=float(bgr.get())
    sec=float(sc.get())
    pota=float(pot.get())
    pcvm=int(pcv.get())
    wcc=int(wc.get())
    rcc=float(rc.get())
    dms=dm.get()
    if dms=='yes' or dms=='YES' or dms=='Yes':
        dms=1
    else:
        dms=0

    s=Svms(spg,alb,sug,bgrd,sec,pota,pcvm,wcc,rcc,dms)
    return (s) 

def out13():
    eids=eid.get()
    b=Get_Data()
    if (b==1):
        content=("After Analysing your Report it is Seen that You may have Chronic Kidney Disease. It is advisable to visit Your Doctor.")
        showinfo('RESULT','Patient Have CKD')
        if askyesno('SEND EMAIL', 'Do You Want to SEND EMAIL to the Patient ?'):
            try:
                mail=SMTP("smtp.gmail.com",587)
                mail.starttls()
                mail.login("ckdpredicter@gmail.com","YouHaveCKD")
                mail.sendmail("ckdpredicter@gmail.com",eids,content)
                mail.close()
                showinfo("Email","Email has been sent")
            except:
                showinfo('Warning !','Internet Is Not Working')

                
    else:
        
        #print("Patient Do Not Have CKD")
        content=('After Analysing your Report it is Seen that You do not have Chronic Kidney Disease')
        showinfo('RESULT','Patient Do Not Have CKD')
        if askyesno('SEND EMAIL', 'Do You Want to SEND EMAIL to the Patient ?'):
            try:
                mail=SMTP("smtp.gmail.com",587)
                mail.starttls()
                mail.login("ckdpredicter@gmail.com","YouHaveCKD")
                mail.sendmail("ckdpredicter@gmail.com",eids,content)
                mail.close()
                showinfo("email","email has been sent")
            except:
                showinfo('Warning !','Internet Is Not Working')

                
def out12():
    eids=eid.get()
    c=Get_Data1()
    if (c==1):
        content=("After Analysing your Report it is Seen that You may have Chronic Kidney Disease. It is advisable to visit Your Doctor.")
        showinfo('RESULT','Patient Have CKD')
        if askyesno('SEND EMAIL', 'Do You Want to SEND EMAIL to the Patient ?'):
            try:                
                mail=SMTP("smtp.gmail.com",587)
                mail.starttls()
                mail.login("ckdpredicter@gmail.com","YouHaveCKD")
                mail.sendmail("ckdpredicter@gmail.com",eids,content)
                mail.close()
                showinfo("Email","Email has been sent")
            except:
                showinfo('Warning !','Internet Is Not Working')

                
    
    else:
        
        #print("Patient Do Not Have CKD")
        content=('After Analysing your Report it is Seen that You do not have Chronic Kidney Disease')
        showinfo('RESULT','Patient Do Not Have CKD')
        if askyesno('SEND EMAIL', 'Do You Want to SEND EMAIL to the Patient ?'):
            try:
                mail=SMTP("smtp.gmail.com",587)
                mail.starttls()
                mail.login("ckdpredicter@gmail.com","YouHaveCKD")
                mail.sendmail("ckdpredicter@gmail.com",eids,content)
                mail.close()
                showinfo('Email','Email has been sent')
            except:
                showinfo('Warning !','Internet Is Not Working')

                
'''def out13():
    tree,forest=get_data()
    if forest==1:
     out2.set('Hired')
    elif forest==0:
     out2.set('Not..Hired..')'''
    
def CRR():
    top1=Toplevel()
    top1.title("Correlation")
    '''top1.configure(background="#ffffcc")'''
    top1.configure(background="White")
    top1.geometry("671x563")
    top1.resizable(False, False)     
    load=Image.open('CRR.png')
    render=ImageTk.PhotoImage(load)
    panel = Label(top1,image=render)
    panel.image=render
    panel.grid(row=0,column=0,sticky=W,columnspan=10)
    print("THIS IS CORELATION MENU")

def IFs():
    print("THIS IS Important Features MENU")
    top1=Toplevel()
    top1.title("Important Feature")
    top1.configure(background="White")
    top1.geometry("759x254")
    top1.resizable(False, False)     
    load=Image.open('IF.png')
    render=ImageTk.PhotoImage(load)
    panel = Label(top1,image=render)
    panel.image=render
    panel.grid(row=0,column=0,sticky=W,columnspan=10)


def ROCC():

    print("THIS IS ROC MENU")
    top1=Toplevel()
    top1.title("Accuracy and ROC")
    top1.configure(background="white")
    top1.geometry("825x330")
    top1.resizable(False, False)     
    load=Image.open('ROC.png')
    render=ImageTk.PhotoImage(load)
    panel = Label(top1,image=render)
    panel.image=render
    panel.grid(row=0,column=0,sticky=W,columnspan=10)

def About():
    showinfo("ABOUT", "C.K.D Predicter is an application which predicts whether a patient may have CKD or not. The 9 Important Features are derived after finding the Correlation between 25 Features. The data was taken over a 2-month period in India with 25 features ( eg, red blood cell count, white blood cell count, etc). \n This Project is made under the guidance of Anil Vats Sir who has helped us put this project together. " )    
    print ("This is a simple example of a menu")
    


top=Tk()
top.title("C.K.D Predicter")
top.iconbitmap(r'favicon.ico')
top.configure(background="#ffffcc")
top.geometry("1000x800")
top.resizable(True, True)

menu=Menu(top)
top.config(menu=menu)
datamenu=Menu(menu)
menu.add_cascade(label="Data Analysis", menu=datamenu)
datamenu.add_command(label="Corelation", command=CRR)
datamenu.add_command(label="Important Features", command=IFs)
datamenu.add_command(label="ROC ", command=ROCC)

helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=About)

'''b1=Button(top,text="correlations", width=16,bg="#134e85",fg="Black",bd=4,command=out13)
b1.grid(row=0,column=0,pady=1)

b2=Button(top,text="Important feature", width=16,bg="#134e85",fg="Black",bd=4,command=out13)
b2.grid(row=0,column=1,pady=1)

b3=Button(top,text="ROC", width=16,bg="#134e85",fg="Black",bd=4,command=out13)
b3.grid(row=0,column=2,pady=1)'''

a1=Label(top,text="  C.K.D Predicter ",fg="Black",bg="#ffffcc",font="Arial 24 bold",bd=15)
a1.grid(row=0,column=1,pady=60,columnspan=10)

load=Image.open('Final_Logo.png')
render=ImageTk.PhotoImage(load)
panel = Label(image=render)
panel.place(x=0,y=0)
'''panel.grid(row=0,column=0,sticky=N+W,columnspan=10)'''

l=Label(top,text="NAME",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=6,column=0,sticky=W,pady=2)

l=Label(top,text="Email",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=6,column=3,sticky=W,pady=2)


l=Label(top,text="Specific Gravity",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=7,column=0,sticky=W,pady=2)

l=Label(top,text="Albumin",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=7,column=3,sticky=W,pady=2)

l=Label(top,text="Sugar",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=8,column=0,sticky=W,pady=2)

l=Label(top,text="Blood Glucose Random",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=8,column=3,sticky=W,pady=2)

l=Label(top,text="Serum Creatinine",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=9,column=0,sticky=W,pady=2)

l=Label(top,text="Potassium",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=9,column=3,sticky=W,pady=2)

l=Label(top,text="Packed Cell Volume",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l.grid(row=10,column=0,sticky=W,pady=2)

l1=Label(top,text="White Blood Cell Count",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l1.grid(row=10,column=3,sticky=W,pady=2)

l1=Label(top,text="Red Blood Cell Count",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l1.grid(row=11,column=0,sticky=W,pady=2) 

l1=Label(top,text="Diabetes Mellitus (Yes or No)",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l1.grid(row=11,column=3,sticky=W,pady=10) 

'''l1=Label(top,text="RESULT",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l1.grid(row=12,column=0,sticky=W,pady=2)

l1=Label(top,text="Analyze",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l1.grid(row=13,column=0,sticky=W,pady=1) 

l1=Label(top,text="SVM",font="Arial 12 ",bd=8,bg="#ffffcc",fg="Black")
l1.grid(row=14,column=0,sticky=W,pady=1)'''

name=StringVar()
e1=Entry(top,textvariable=name,width=35,bd=3,bg="powder blue")
e1.grid(row=6,column=1,pady=2)

eid=StringVar()
e1=Entry(top,textvariable=eid,width=35,bd=3,bg="powder blue")
e1.grid(row=6,column=4,pady=2)

sg=StringVar()
e1=Entry(top,textvariable=sg,width=35,bd=3,bg="powder blue")
e1.grid(row=7,column=1,pady=2)

al=StringVar()
e2=Entry(top,textvariable=al,width=35,bd=3,bg="powder blue")
e2.grid(row=7,column=4,pady=2)

su=StringVar()
e3=Entry(top,textvariable=su,width=35,bd=3,bg="powder blue")
e3.grid(row=8,column=1,pady=2)

bgr=StringVar()
e4=Entry(top,textvariable=bgr,width=35,bd=3,bg="powder blue")
e4.grid(row=8,column=4,pady=2)

sc=StringVar()
e5=Entry(top,textvariable=sc,width=35,bd=3,bg="powder blue")
e5.grid(row=9,column=1,pady=2)

pot=StringVar()
e6=Entry(top,textvariable=pot,width=35,bd=3,bg="powder blue")
e6.grid(row=9,column=4,pady=2)

pcv=StringVar()
e6=Entry(top,textvariable=pcv,width=35,bd=3,bg="powder blue")
e6.grid(row=10,column=1,pady=2)

wc=StringVar()
e6=Entry(top,textvariable=wc,width=35,bd=3,bg="powder blue")
e6.grid(row=10,column=4,pady=2)

rc=StringVar()
e6=Entry(top,textvariable=rc,width=35,bd=3,bg="powder blue")
e6.grid(row=11,column=1,pady=2)

dm=StringVar()
e6=Entry(top,textvariable=dm,width=35,bd=3,bg="powder blue")
e6.grid(row=11,column=4,pady=10)

b4=Button(top,text="Analyze with RF", width=16,bg="#134e85",fg="White",bd=4,command=out13)
b4.grid(row=20,column=1,columnspan=3,sticky=N,pady=5)

b5=Button(top,text="Analyze with SVM", width=16,bg="#134e85",fg="White",bd=4,command=out12)
b5.grid(row=21,column=1,columnspan=3,sticky=N,pady=2)

c1=Label(top,height=2,width=27,bg="#ffffcc")
c1.grid(row=6,column=13)

out2=StringVar()

c1=Label(top,textvariable=out2,height=2,width=27,fg="Black",bg="#ffffcc",font="Arial 14 bold")
c1.grid(row=13,column=1)

out2=StringVar()

c6=Label(top,textvariable=out2,height=2,width=27,fg="Black",bg="#ffffcc",font="Arial 14 bold")
c6.grid(row=13,column=1)


top.mainloop()
