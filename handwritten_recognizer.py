# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:58:49 2016

@author: altescy
"""

import sys
from PyQt4 import QtGui, QtCore
import numpy as np
import QtCanvas
import dig_recognizer

class CanvasWidget(QtCanvas.Canvas):
    def __init__(self, parent=None):
       super(CanvasWidget, self).__init__(parent=parent)


    def mouseReleaseEvent(self, event):
        super(CanvasWidget, self).mouseReleaseEvent(event)
        self.parent.recognize_digit(self.canvas_data(size=8, inv=True),
                                    self.canvas_data(size=28, inv=True))



class DrawingWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(DrawingWidget, self).__init__(parent=parent)
        self.setup_ui()


    def setup_ui(self):
        self.canvas = CanvasWidget(parent=self)
        pal = self.canvas.palette()
        pal.setColor(QtGui.QPalette.Window, QtCore.Qt.white)
        self.canvas.setPalette(pal)
        self.canvas.setAutoFillBackground(True)
        self.canvas.setFixedSize(300, 300)

        self.clear_button = QtGui.QPushButton("clear", parent=self)
        self.clear_button.clicked.connect(self.canvas.clear_canvas)
        self.save_button = QtGui.QPushButton("save", parent=self)
        self.save_button.clicked.connect(self.canvas.save_canvas)
        self.save_button.setFixedWidth(200)
        
        self.results = {}
        self.label = ["kNN", "linear classifier", "logistic regression",
                      "SVM", "simpleNN", "CNN"]
        for label in self.label:
            self.results[label] = []
        
        self.table = QtGui.QTableWidget(parent=self)
        self.table.setFixedSize(QtCore.QSize(620,330))
        self.table.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.table.setRowCount(10)
        self.table.setColumnCount(6)
        for i, label in enumerate(self.label):
            self.table.setHorizontalHeaderItem(i,
                                               QtGui.QTableWidgetItem(label))
        for i in range(10):
            self.table.setVerticalHeaderItem(i, QtGui.QTableWidgetItem(str(i)))

        layout = QtGui.QGridLayout()
        layout.addWidget(self.canvas, 0, 0)
        layout.addWidget(self.table, 0, 1)
        layout.addWidget(self.clear_button, 1, 0)
        layout.addWidget(self.save_button, 1, 1)

        self.setLayout(layout)


    def recognize_digit(self, dig, dig28):
        self.results["kNN"] = dig_recognizer.kNN_method(dig)
        self.results["linear classifier"] = \
                          dig_recognizer.linear_clasiffer(dig)
        self.results["logistic regression"] = \
                          dig_recognizer.logistic_regression(dig.reshape(1,-1))
        self.results["SVM"] = dig_recognizer.svm(dig28)
        self.results["simpleNN"] = dig_recognizer.nn28x28(np.array([[dig28]],
                                                          dtype=np.float32))[0]
        self.results["CNN"]= dig_recognizer.cnn28x28(np.array([[dig28]],
                                      dtype=np.float32))[0]

        self.show_result()
        self.update()
    

    def show_result(self):
        for i, label in enumerate(self.label):
            if len(self.results[label]) > 1:
                maxval = np.argmax(self.results[label])        
                for j, val in enumerate(self.results[label]):
                    self.table.setItem(j, i, QtGui.QTableWidgetItem(str(val)))
                    if j == maxval:
                        self.table.setCurrentCell(j, i, 
                                              QtGui.QItemSelectionModel.Select)
                    else:
                        self.table.setCurrentCell(j ,i , 
                                            QtGui.QItemSelectionModel.Deselect)
            else:
                for j in range(10):
                    if self.results[label][0] == j:
                        self.table.setItem(j, i,
                                           QtGui.QTableWidgetItem(str(j)))
                        self.table.setCurrentCell(j, i, 
                                              QtGui.QItemSelectionModel.Select)
                    else:
                        self.table.setItem(j, i,
                                           QtGui.QTableWidgetItem(""))
                        self.table.setCurrentCell(j ,i , 
                                            QtGui.QItemSelectionModel.Deselect)

        self.table.update()



if __name__ == '__main__':
    app = 0
    app = QtGui.QApplication(sys.argv)

    panel = QtGui.QWidget()

    drawing_widget = DrawingWidget(parent=panel)

    window_size = QtCore.QSize(950, 380)

    panel_layout = QtGui.QVBoxLayout()
    panel_layout.addWidget(drawing_widget)
    panel.setFixedSize(window_size)

    main_window = QtGui.QMainWindow()
    main_window.setWindowTitle("Hand-written Digit Recognizer")
    main_window.setFixedSize(window_size)
    main_window.setCentralWidget(panel)

    main_window.show()
    app.exec_()



