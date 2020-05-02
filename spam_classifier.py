# ===========================Packages====================================================================================
from tkinter import *;
from tkinter.constants import *
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# ========================================Main Class=====================================================================

class MachineLearning:
    def __init__(self):
        self.data = None
        self.table = None
        self.selection_x = None
        self.selection_y = None
        self.X = None
        self.y = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.DF = None

        self.svm_model = None
        self.svm_predictions = None
        self.Accuracy = None

        self.le = LabelEncoder()

        self.window = Tk()
        self.color = 'grey95'
        self.window.geometry('900x620')
        self.window.resizable(False, False)
        self.window.configure(background=self.color)
        self.window.title('Machine Learning SVM')

        self.heading = Label(self.window, text="Machine Learning SVM", bg=self.color, pady=20,
                             font=("Helvetica", 35, "bold"))
        self.heading.place(width=620, height=100, bordermode=OUTSIDE, x=0, y=0)

        # ===============================File Selection and viewing======================================================
        self.frame = LabelFrame(self.window, text='File Selection', bg=self.color)
        self.frame.place(width=580, height=80, bordermode=OUTSIDE, x=20, y=100)

        self.name_label = Label(self.frame, text="File Name : ", bg=self.color, padx=10, pady=10,
                                font=("Helvetica", 15))
        self.name_label.place(width=120, height=30, bordermode=INSIDE, x=10, y=13)

        self.name = StringVar()
        self.name_entry = Entry(self.frame, exportselection=False, textvariable=self.name, font=("Helvetica", 12))
        self.name_entry.place(width=250, height=30, bordermode=INSIDE, x=130, y=13)

        self.name_select = Button(self.frame, text='Select', command=lambda: self.select())
        self.name_select.place(width=50, height=30, bordermode=INSIDE, x=395, y=13)

        self.df_show = Button(self.frame, text='Show', command=lambda: self.create_table(), state=DISABLED)
        self.df_show.place(width=50, height=30, bordermode=INSIDE, x=455, y=13)

        # =====================================GRAPH OF IRIS DATASET=====================================================
        self.graph = LabelFrame(self.window, text='Graph Plotting', bg=self.color)
        self.graph.place(width=700, height=80, bordermode=OUTSIDE, x=20, y=200)

        self.Amountpl = Button(self.graph, text=' Time vs Amount Fraud', command=lambda: self.AmountFraud_plot(),
                               state=DISABLED)
        self.Amountpl.place(width=250, height=30, bordermode=INSIDE, x=5, y=13)



        # ====================================Train Test Split===========================================================
        self.ttsplit = LabelFrame(self.window, text='Train Test Split', bg=self.color)
        self.ttsplit.place(width=700, height=80, bordermode=OUTSIDE, x=20, y=300)

        self.trainsplit = Button(self.ttsplit, text='split', command=lambda: self.train_test_split(), state=DISABLED)
        self.trainsplit.place(width=125, height=30, bordermode=INSIDE, x=5, y=13)

        # =====================================SVM GUI===================================================================
        self.svm = LabelFrame(self.window, text='Support Vector Machine Linear', bg=self.color)
        self.svm.place(width=700, height=80, bordermode=OUTSIDE, x=20, y=400)

        self.svm_pred = Button(self.svm, text='Predict', command=lambda: self.pred_svm(), state=DISABLED)
        self.svm_pred.place(width=125, height=30, bordermode=INSIDE, x=5, y=13)

        self.report = Button(self.svm, text='Report', command=lambda: self.svm_report(), state=DISABLED)
        self.report.place(width=125, height=30, bordermode=INSIDE, x=140, y=13)

        self.confusion_matrix = Button(self.svm, text=' Confusion_Matrix', command=lambda: self.cm_svm(),
                                       state=DISABLED)
        self.confusion_matrix.place(width=125, height=30, bordermode=INSIDE, x=280, y=13)

        self.plot_data = Button(self.svm, text='Plot', command=lambda: self.plot(), state=DISABLED)
        self.plot_data.place(width=125, height=30, bordermode=INSIDE, x=420, y=13)

        self.svm_error = Button(self.svm, text='Error', command=lambda: self.errors_svm(), state=DISABLED)
        self.svm_error.place(width=125, height=30, bordermode=INSIDE, x=560, y=13)

        self.window.mainloop()

    # ===================================Selecting an File===============================================================

    def select(self):
        try:
            self.data = pd.read_csv(self.name.get())
            self.Amountpl['state'] = NORMAL
            self.df_show['state'] = NORMAL
            self.trainsplit['state'] = NORMAL

        except FileNotFoundError:
            self.name.set("Invalid")

    def create_table(self):
        try:
            self.table.window.deiconify()
        except AttributeError:
            if self.data.shape[0] > 152:
                self.table = Table(self.data.head(152), self.window, self.name.get())
            else:
                self.table = Table(self.data, self.window, self.name.get())
        except TclError:
            if self.data.shape[0] > 152:
                self.table = Table(self.data.head(152), self.window, self.name.get())
            else:
                self.table = Table(self.data, self.window, self.name.get())

        self.trainsplit['state'] = NORMAL
        self.svm_pred['state'] = DISABLED
        self.report['state'] = DISABLED
        self.confusion_matrix['state'] = DISABLED
        self.plot_data['state'] = DISABLED
        self.svm_error['state'] = DISABLED

    # =============================Splitting Data  In Train And Test=====================================================
    def train_test_split(self):
        print("hello")
        self.DF = pd.read_csv(self.name.get())
        print(len(self.DF))
        x = self.DF.iloc[:, :-1]
        y = self.DF.iloc[:, -1]
        print("----------------------------------------------Dataset Values-------------------------------------------------")
        print(self.DF.head(5))
        print("Information about X dataframe:  ")
        print(x.info())
        print("Describing the data")
        self.DF[self.DF['Class'] == 1].describe()
        self.X = self.DF.drop(['Time', 'Class'], axis=1)
        self.y = self.le.fit_transform(self.DF['Class'])
        print(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20)
        print(self.X_train)
        print("---------------")
        print(self.X_test)
        print("---------------")

        print(self.y_train)
        print("---------------")

        print(self.y_test)   #classify the values
        self.svm_pred['state'] = NORMAL

        print("-----------------------------------------------")

        print(len(self.y_train))
        print("---------------")

        print(self.y_test)

        self.svm_pred['state'] = NORMAL

        print("-----------------------------------------------")

    # ==============================Predict Function For SVM=============================================================
    def pred_svm(self):
        self.svm_model = SVC(C=1.0, kernel='linear', class_weight='balanced')

        self.svm_model.fit(self.X_train, self.y_train)
        self.Accuracy = self.svm_model.score(self.X_train, self.y_train)
        print("Accuracy is=", self.Accuracy * 100)
        print("=============================================================")

        self.svm_predictions = self.svm_model.predict(self.X_test)
        print("predictions is=", self.svm_predictions)

        self.svm_error['state'] = NORMAL
        self.confusion_matrix['state'] = NORMAL
        self.report['state'] = NORMAL
        self.plot_data['state'] = NORMAL

    # ==============================AmountFraud Plot====================================================================
    def AmountFraud_plot(self):
        AmountFraud(self.window)  # Calling Sepal Class



    # =================================SVM Plotting=====================================================================
    def plot(self):  # plot of svm
        Scatter(self.window, self.y_test, self.svm_predictions)  # Calling Scatter Class

    # ================================SVM Report========================================================================
    def svm_report(self):  # report svm
        ClassificationReport(self.window, classification_report(self.le.inverse_transform(self.y_test),
                                                                self.le.inverse_transform(self.svm_predictions)),
                             'Support Vector Machine')
        # Calling ClassificationReport Class

    # ===============================SVM Error==========================================================================
    def errors_svm(self):  # error svm
        temp = [mean_absolute_error(self.y_test, self.svm_predictions), mean_squared_error(self.y_test,
                                                                                           self.svm_predictions),
                np.sqrt(mean_squared_error(self.y_test, self.svm_predictions))]
        Errors(self.window, temp, 'SVM')  # calling Error Class

    # ==========================SVM Confusion Matrix=====================================================================
    def cm_svm(self):
        ConfusionMatrix(self.window, confusion_matrix(self.le.inverse_transform(self.y_test),
                                                      self.le.inverse_transform(self.svm_predictions)),
                        'Support Vector Matrix',self.le.classes_)  # ConfusionMatrix Class

    # =====================================Table Class===================================================================


class Table:
    def __init__(self, data, master, name):
        self.master = master
        self.window = Toplevel(self.master)
        self.data = data
        self.name = name
        self.window.title(self.name)
        self.window.geometry('600x600')
        self.window.minsize(250, 250)

        self.frame = Frame(self.window)
        self.frame.pack(expand=True, fill=BOTH)

        self.canvas = Canvas(self.frame, background='white')

        self.h_scroll = Scrollbar(self.frame, orient=HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=BOTTOM, fill=X)
        self.v_scroll = Scrollbar(self.frame, orient=VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=RIGHT, fill=Y)

        self.canvas['xscrollcommand'] = self.h_scroll.set
        self.canvas['yscrollcommand'] = self.v_scroll.set
        self.canvas.pack(expand=True, fill=BOTH)

        self.label_frame = LabelFrame(self.canvas)
        self.canvas.create_window((0, 0), window=self.label_frame, anchor=N + W)

        self.shape = (data.shape[0], data.shape[1])

        Table.add_label(self, 0, 0, '#', font=('Helvetica', 15, 'bold'))
        for j in range(self.shape[1]):
            Table.add_label(self, 0, j + 1, self.data.columns[j], font=('Helvetica', 12, 'bold'))
        self.height = 20
        for i in range(self.shape[0]):
            Table.add_label(self, i + 1, 0, str(i + 1))
            ar = data.iloc[i].values
            for j in range(len(ar)):
                Table.add_label(self, i + 1, j + 1, ar[j])
        self.window.update()
        self.canvas.configure(scrollregion=self.label_frame.bbox(ALL))

    def add_label(self, i, j, text, font=('Helvetica', 10)):
        if j % 2 == 0:
            color = 'white'
        else:
            color = 'antique white'
        label = Label(self.label_frame, text=text, font=font, bg=color)
        label.grid(row=i, column=j, sticky=E + N + W + S)


# ============================================Confusion Matrix Class=====================================================
class ConfusionMatrix:
    def __init__(self, master, data, name, labels):
        self.data = data
        self.master = master
        self.name = name
        self.labels = sorted(labels)
        print(self.labels)

        self.total = np.sum(self.data)

        self.window = Toplevel(self.master)
        self.window.title(self.name + " Confusion Matrix")
        self.window.resizable(False, False)

        self.total_label = Label(self.window, text=f'Total = {self.total}', font=('Helvetica', 15, 'bold'),
                                 bg='antique white')

        self.total_label.grid(row=0, column=0, sticky=(N, S, E, W))

        for i in range(len(self.labels)):
            if i % 2 == 0:
                color = 'white'
            else:
                color = 'antique white'
            Label(self.window, text=f'Predicted\n{self.labels[i]}', font=('Helvetica', 15, 'bold'),
                  bg=color).grid(row=0, column=i + 1, sticky=(N, S, E, W))

        for i in range(len(self.labels)):
            if i % 2 == 0:
                color = 'white'
            else:
                color = 'antique white'
            Label(self.window, text=f'Actual\n{self.labels[i]}', font=('Helvetica', 15, 'bold'),
                  bg=color).grid(row=i + 1, column=0, sticky=(N, S, E, W))

            for j in range(len(self.labels)):
                color = ['grey90', 'grey80', 'grey70']
                Label(self.window, text=str(self.data[i][j]), font=('Helvetica', 15, 'bold'),
                      bg=color[(i + j) % 3]).grid(row=i + 1, column=j + 1, sticky=(N, S, E, W))


# ==================================================Error Class==========================================================
class Errors:
    def __init__(self, master, data, name):
        self.master = master
        self.data = data
        self.name = name

        self.window = Toplevel(self.master)
        self.window.title(self.name + " Errors")
        self.window.geometry('500x180')
        self.window.resizable(False, False)

        self.frame = Frame(self.window)
        self.frame.place(width=504, height=184, bordermode=OUTSIDE, x=0, y=0)

        self.text1 = Label(self.frame, text='Mean Absolute Error :', font=('Helvetica', 15, 'bold'), bg='antique white')
        self.text1.place(width=260, height=60, bordermode=INSIDE, x=0, y=0)
        self.text2 = Label(self.frame, text='Mean Squared Error :', font=('Helvetica', 15, 'bold'), bg='white')
        self.text2.place(width=260, height=60, bordermode=INSIDE, x=0, y=60)
        self.text3 = Label(self.frame, text='Root Mean Squared Error: ', font=('Helvetica', 15, 'bold'),
                           bg='antique white')
        self.text3.place(width=260, height=60, bordermode=INSIDE, x=0, y=120)

        self.value1 = Label(self.frame, text=str(data[0]), font=('Helvetica', 15, 'bold'), bg='antique white')
        self.value1.place(width=240, height=60, bordermode=INSIDE, x=260, y=0)
        self.value2 = Label(self.frame, text=str(data[1]), font=('Helvetica', 15, 'bold'), bg='white')
        self.value2.place(width=240, height=60, bordermode=INSIDE, x=260, y=60)
        self.value3 = Label(self.frame, text=str(data[2]), font=('Helvetica', 15, 'bold'), bg='antique white')
        self.value3.place(width=240, height=60, bordermode=INSIDE, x=260, y=120)


# ===================================Classification Report Class=========================================================
class ClassificationReport:
    def __init__(self, master, data, name):
        self.master = master
        self.data = data
        self.name = name

        self.window = Toplevel(self.master)
        self.window.title(self.name + " Classification Report")
        self.window.configure(background='white')
        self.window.resizable(False, False)
        y = 0

        Label(self.window, text='precision', font=('Helvetica', 15, 'bold'), anchor=E, bg='antique white').place(
            width=100, height=50, bordermode=INSIDE, x=150, y=y)
        Label(self.window, text='recall', font=('Helvetica', 15, 'bold'), anchor=E, bg='white').place(width=100,
                                                                                                      height=50,
                                                                                                      bordermode=INSIDE,
                                                                                                      x=250, y=0)
        Label(self.window, text='f1-score', font=('Helvetica', 15, 'bold'), anchor=E, bg='antique white').place(
            width=100, height=50, bordermode=INSIDE, x=350, y=y)
        Label(self.window, text='support', font=('Helvetica', 15, 'bold'), anchor=E, bg='white').place(width=100,
                                                                                                       height=50,
                                                                                                       bordermode=INSIDE,
                                                                                                       x=450, y=0)
        y = y + 50

        Label(self.window, bg='antique white').place(width=100, height=10, bordermode=INSIDE, x=150, y=y)
        Label(self.window, bg='antique white').place(width=100, height=10, bordermode=INSIDE, x=350, y=y)
        y = y + 10

        self.ar = self.data.split('\n\n')[1:]
        print(self.ar)
        self.part1 = self.ar[0].split('\n')

        for i in self.part1:
            temp = i.split()
            Label(self.window, text=temp[0], font=('Helvetica', 12, 'bold'), anchor=E, bg='white').place(width=150,
                                                                                                         height=30,
                                                                                                         bordermode=INSIDE,
                                                                                                         x=0, y=y)
            Label(self.window, text=temp[1], font=('Helvetica', 12), anchor=E, bg='antique white').place(width=100,
                                                                                                         height=30,
                                                                                                         bordermode=INSIDE,
                                                                                                         x=150, y=y)
            Label(self.window, text=temp[2], font=('Helvetica', 12), anchor=E, bg='white').place(width=100, height=30,
                                                                                                 bordermode=INSIDE,
                                                                                                 x=250, y=y)
            Label(self.window, text=temp[3], font=('Helvetica', 12), anchor=E, bg='antique white').place(width=100,
                                                                                                         height=30,
                                                                                                         bordermode=INSIDE,
                                                                                                         x=350, y=y)
            Label(self.window, text=temp[4], font=('Helvetica', 12), anchor=E, bg='white').place(width=100, height=30,
                                                                                                 bordermode=INSIDE,
                                                                                                 x=450, y=y)
            y = y + 30

        Label(self.window, bg='antique white').place(width=100, height=20, bordermode=INSIDE, x=150, y=y)
        Label(self.window, bg='antique white').place(width=100, height=20, bordermode=INSIDE, x=350, y=y)
        y = y + 20

        self.part2 = self.ar[1].split('\n')

        for i in self.part2:
            if i == '':
                continue
            temp = i.split()
            Label(self.window, text=temp.pop(), font=('Helvetica', 12), anchor=E, bg='white').place(width=100,
                                                                                                    height=30,
                                                                                                    bordermode=INSIDE,
                                                                                                    x=450, y=y)
            Label(self.window, text=temp.pop(), font=('Helvetica', 12), anchor=E, bg='antique white').place(width=100,
                                                                                                            height=30,
                                                                                                            bordermode=INSIDE,
                                                                                                            x=350, y=y)
            if len(temp) != 1:
                Label(self.window, text=temp.pop(), font=('Helvetica', 12), anchor=E, bg='white').place(width=100,
                                                                                                        height=30,
                                                                                                        bordermode=INSIDE,
                                                                                                        x=250, y=y)
            if len(temp) != 1:
                Label(self.window, text=temp.pop(), font=('Helvetica', 12), anchor=E, bg='antique white').place(
                    width=100, height=30, bordermode=INSIDE, x=150, y=y)
            else:
                Label(self.window, bg='antique white').place(width=100, height=30, bordermode=INSIDE, x=150, y=y)
            Label(self.window, text=" ".join(temp), font=('Helvetica', 12, 'bold'), anchor=E, bg='white').place(
                width=150, height=30, bordermode=INSIDE, x=0, y=y)
            y = y + 30

        self.window.geometry('550x' + str(y))


# ==============================================Scatter Class============================================================
class Scatter:
    def __init__(self, master, y_test, pred):
        self.master = master
        self.y_test = y_test
        print(self.y_test)
        self.pred = pred
        print(self.pred)
        self.window = Toplevel(self.master)
        self.window.title("Scatter Plot (y_test vs predictions)")
        self.window.configure(background='white')
        self.window.resizable(False, False)

        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.sub = self.figure.add_subplot(111, xlabel="Y_Predict", ylabel="Y_Test", title="Y_Predict & Y_Test")
        self.sub.scatter(self.y_test, self.pred, edgecolor='black')
        self.sub.plot()
        self.sub.legend()
        self.sub.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()


# ========================================AmountFraud Class====================================================================
class AmountFraud:
    def __init__(self, master):
        self.master = master
        self.plt = plt
        self.le1 = LabelEncoder()
        self.DF = pd.read_csv('Book2.csv')
        self.df_fraud = self.DF[self.DF['Class'] == 'Fraud']  # Recovery of fraud data
        print(self.df_fraud)
        self.x = self.df_fraud.get("Time")
        print(self.x)
        self.y = self.df_fraud.get("Amount")
        print(self.y)
        self.window = Toplevel(self.master)
        self.window.title("Scatter Plot Amount Fraud")
        self.window.configure(background='white')
        self.window.resizable(False, False)

        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.sub = self.figure.add_subplot(111, xlabel="Time", ylabel="Amount", title="Time vs Amount Class=0(Fraud)")
        self.sub.scatter(self.x, self.y)

        self.sub.plot()
        self.sub.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()





# ========================================Main Function==================================================================
if __name__ == '__main__':
    MachineLearning()

# =======================================================================================================================
