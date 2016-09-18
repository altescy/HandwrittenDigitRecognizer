# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui
import numpy as np
from skimage import io

class Canvas(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)
        self.parent = parent
        self.scene = QtGui.QGraphicsScene(parent=self)
        self.pts = [[]]
        self.cur = [False, False]

        self.pen_color = QtCore.Qt.black
        self.pen_width = 40

        self.clear_flag = True
        self.first_flag = True
        self.image = QtGui.QPixmap(self.size())


    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self.first_flag == False:
            painter.drawPixmap(QtCore.QRectF(0,0,self.width(),self.height()),
                               self.image,
                               QtCore.QRectF(0,0,self.width(),self.height()))

        if self.cur[0] != False:
            pen = QtGui.QPen(QtGui.QBrush(self.pen_color), self.pen_width)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(self.poly(self.pts))

        if self.clear_flag == True:
            painter.fillRect(event.rect(), QtGui.QBrush(QtCore.Qt.white))
            self.clear_flag = False

        painter.end()


    def poly(self, pts):
        return QtGui.QPolygonF(map(lambda p: QtCore.QPointF(*p), pts))


    def mousePressEvent(self, event):
        self.cur = [event.x(), event.y()]

        if self.cur[0] < 0:
            self.cur[0] = 0
        elif self.cur[0] > self.width():
            self.cur[0] = self.width()

        if self.cur[1] < 0:
            self.cur[1] = 0
        elif self.cur[1] > self.height():
            self.cur[1] = self.height()

        self.pts = [self.cur]

        self.update()


    def mouseMoveEvent(self, event):
        if self.cur[0] == False:
            return

        self.cur = [event.x(), event.y()]

        if self.cur[0] < 0:
            self.cur[0] = 0
        elif self.cur[0] > self.width():
            self.cur[0] = self.width()

        if self.cur[1] < 0:
            self.cur[1] = 0
        elif self.cur[1] > self.height():
            self.cur[1] = self.height()

        self.pts.append(self.cur)

        self.update()


    def mouseReleaseEvent(self, event):
        self.image = self.grabCanvas()
        self.cur = [False, False]
        self.pts = [[]]
        self.first_flag = False


    def setPen(self, color, width):
        self.pen_color = color
        self.pen_width = width


    def clear_canvas(self):
        self.clear_flag = True
        self.update()
        self.image = self.grabCanvas()

    def grabCanvas(self):
        return self.image.grabWidget(self,QtCore.QRect(0,0,self.width(),
                                     self.height()))


    def save_canvas(self, filename="canv.png", size=8):
        pix = QtGui.QPixmap(self.size())
        pix = pix.grabWidget(self, QtCore.QRect(0, 0,
                                                self.width(), self.height()))
        pix = pix.scaled(size, size, QtCore.Qt.KeepAspectRatio,
                         QtCore.Qt.SmoothTransformation)
        pix.save(filename)


    def canvas_data(self, size=8, inv=False):
        self.save_canvas(filename="tmp.png", size=size)
        dig = np.array(io.imread('tmp.png', as_grey=True)).reshape(size**2)
        if inv:
            for i, d in enumerate(dig):
                if d < 0.5:
                    dig[i] = 1.0 - d
                elif d > 0.5:
                    dig[i] = d - 1.0

        return dig
