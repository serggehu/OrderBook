from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from multiprocessing import Process, Value, Array
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random

class Book():
    def __init__(self):
        self.data =[]
        self.last_time = 0
        self.last_L= None
        self.init_time = 0
        self.nB=[]
        self.ask_trades=np.zeros([0, 1,2])
        self.bid_trades=np.zeros([0, 1,2])
        self.last_ask_trade = 0.0
        self.last_bid_trade = 0.0
        self.ask=np.zeros([64,2])
        self.bid=np.zeros([64,2])
        self.last_x_ask = 0
        self.last_x_bid = 0
    def DataPrepare(self):
        data = pd.read_csv("C:/Users/Sergey/Documents/Python Scripts/DB/L2L1.txt", sep = ";")
        data = data[0:300]
        #print(data.head())
        data.columns = ["1","2","3","4","5","6","7","8","9"]
        #data = data[0:300]
        return data
    
    def NormaliseBook(self):
        maximum = np.amax(self.ask[:,0])
        minimum = np.amin(self.bid[:,0])
        dim0 = np.arange(minimum, maximum, 0.01)
        length = len(dim0) 
        nB = np.zeros([length, 3])
        
        self.bid = self.bid[::-1]
        nB[:,0] = dim0
        nB =np.append(self.ask[::-1], self.bid, axis = 0)
        return nB
    def UpdateBook(self,  b_arr, x_arr, y_arr, z_arr):
        
        for ind in self.data.index:
            
            cur_time =  int(str(self.data.loc[ind, "3"]) + str(self.data.loc[ind, "4"]))
           
            if(self.data.loc[ind,'1'] == "L2" and self.data.loc[ind,'5'] == 0 and self.init_time == cur_time):
                pos = self.data.loc[ind,'6']
                price = self.data.loc[ind,'8']
                volume = self.data.loc[ind,'9']
    
                if self.data.loc[ind,'2'] == 0:
                    self.ask[pos,0] = price
                    self.ask[pos,1] = volume
                elif self.data.loc[ind,'2'] == 1:
                    self.bid[pos,0] = price
                    self.bid[pos,1] = volume
                self.last_L = "L2"
              
          
            elif(self.data.loc[ind,'1'] == "L2" and self.data.loc[ind,'5'] == 1 and self.init_time != cur_time):
                
                if self.last_L =="L1":                                         #substract from OrderBook
                    pos = self.data.loc[ind,'6']
                    price = self.data.loc[ind,'8']
                    last_x_ask = self.last_x_ask
                    last_x_bid = self.last_x_bid
                    if self.data.loc[ind,'2'] == 0:
                        print("ask ", price, " last_x ", last_x_ask, " self.data.loc[ind,'9'] ", self.data.loc[ind,'9'])
                        # fill the l2 array
                        if(self.ask[pos, 0] != price): np.delete(self.ask, 0)         # nicht verstaandbar                        
                        volume = self.ask[pos,1] - self.data.loc[ind,'9']
                        if volume < 0: np.delete(self.ask, pos)
                        self.ask[pos,1] = volume
                        
                        # fill the trades array
                        if  last_x_ask == 0 or y_arr[last_x*2 - 2]!=price : 
                            print("INItiadet new ask")
                            if last_x_ask == 0: ind2 = 0
                            else: ind2 = last_x_ask *2-2
                            self.last_x = last_x_ask + 1
                            x_arr[ind2]= last_x_ask + 1
                            y_arr[ind2] = price
                            z_arr[ind2] = self.data.loc[ind,'9']
                            
                        else:
                            print("added to last ask")
                            ind2 = last_x_ask*2-2
                            x_arr[ind2]= last_x_ask 
                            y_arr[ind2] = price
                            z_arr[ind2] = + self.data.loc[ind,'9']
                            
                        print(" x ", x_arr[:])
                        print(" y ", y_arr[:])
                        print(" z ", z_arr[:])
                    
                    else:
                        print("bid ", price, " last_x ", last_x_bid, " self.data.loc[ind,'9'] ", self.data.loc[ind,'9'])
                        # fill the l2 array
                        if(self.bid[pos, 0] != price): np.delete(self.bid, 0)
                        volume = self.bid[pos,1] - self.data.loc[ind,'9']
                        if volume < 0: volume = 0
                        self.bid[pos,1] = volume
                       
                        # fill the trades array
                        if last_x_bid == 0 or y_arr[last_x_bid*2 - 1]!=price : 
                            print("initiated new bid")
                            self.last_x_bid = last_x_bid + 1
                            if last_x_bid == 0: ind2 = 1
                            else: ind2 = last_x_bid*2-1
                            x_arr[ind2]= last_x_bid + 1
                            y_arr[ind2] = price
                            z_arr[ind2] = +self.data.loc[ind,'9']
                            
                        else:
                            print("added to last bid")
                            ind2 = last_x_bid*2-1
                            x_arr[ind2]= last_x_bid 
                            y_arr[ind2] = price
                            z_arr[ind2] = + self.data.loc[ind,'9']
                        
                        print(" x ", x_arr[:])
                        print(" y ", y_arr[:])
                        print(" z ", z_arr[:])
                else:                      
                                            #update OrderBook
                    pos = self.data.loc[ind,'6']
                    price = self.data.loc[ind,'8']
                    
                    if self.data.loc[ind,'2'] == 0:                
                        self.ask[pos,1] = self.data.loc[ind,'9']
                    else:
                        self.bid[pos,1] = self.data.loc[ind,'9']
            
            
            if self.data.loc[ind,'1'] == "L2":
                self.last_L = "L2"
            elif self.data.loc[ind,'1'] == "L1":
                self.last_L = "L1"    
            
            if(self.init_time != cur_time):
               
                self.nB = self.NormaliseBook()
                          
                b = myBook.nB
                b = np.array(b[:, [0,1]]).flatten()
 
                for i in range(len(b_arr)):
                    b_arr[i]=b[i]
            
           
            
def MainProgram(arr, x_arr, y_arr, z_arr):

    while True:
        myBook.UpdateBook(arr, x_arr, y_arr, z_arr)
             
    
def runPQG(b_arr, x_arr, y_arr, z_arr):
    
    
    def update():
        
        time.sleep(1)
        start = time.time()
        global sp3
# =============================================================================
#         print("x_arr", x_arr[:])
#         print("y_arr", y_arr[:])
#         print("z_arr", z_arr[:])
# =============================================================================
        return 
        w.removeItem(sp3)
        # regular grid of starting positions
        pos1 = np.mgrid[0:10, 0:10, 0:1].reshape(3,10,10).transpose(1,2,0)
        # fixed widths, random heights
        size1 = np.empty((10,10,3))
        size1[...,0:2] = 0.4
        size1[...,2] = np.random.normal(size=(10,10))
        sp3 = gl.GLBarGraphItem(pos = pos1, size = size1)
        #sp3.setData(pos=pos3, color=color)
        w.addItem(sp3)
        end = time.time()
        
    
    
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
    w.setCameraPosition(distance = 20, azimuth = 0, elevation = 30)
    g = gl.GLGridItem()
    
    g.setSize(x=100,y=100,z=100)
    
    w.addItem(g)
    pos1 = np.mgrid[-10:-9,5:6,0:1].reshape(3,1,1).transpose(1,2,0)
    size1 = np.empty((1,1,3)) 
    size1[...,0:2] = 1
    size1[...,2] = 1
    sp3 = gl.GLBarGraphItem(pos = pos1, size = size1)
    w.addItem(sp3)
        
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(100)
        
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    

if __name__ == '__main__':
    myBook = Book()
    myBook.data=myBook.DataPrepare()
    myBook.init_time = int(str(myBook.data.loc[0, "3"]) + str(myBook.data.loc[0, "4"]))
    n = 128
    m = 2
    b_arr = Array('d', n*m)
    t = 10
    v = 1
    k = 2
    x_arr = Array("d", t*k*v, lock = False)
    y_arr = Array("d", t*k*v, lock = False)
    z_arr = Array("d", t*k*v, lock = False)
    
    p = Process(target=runPQG, args=(b_arr, x_arr, y_arr, z_arr))
    p.start()
    MainProgram(b_arr, x_arr, y_arr, z_arr)
    p.join()
    
 
