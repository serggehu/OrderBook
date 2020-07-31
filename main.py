from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtGui import QVector3D
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
import sys

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
        self.last_x_ind_ask = 0
        self.last_x_ind_bid = 0
        self.last_y_ind_ask = 0
        self.last_y_ind_bid = 0
        self.last_z_ind_ask = 0
        self.last_z_ind_bid = 0
        self.last_x_value_ask = 0
        self.last_x_value_bid = 0
        
    def DataPrepare(self):
        data = pd.read_csv("C:/Users/Sergey/Documents/Python Scripts/DB/test2.csv", sep = ";")
        #data = data[0:1000]
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
                    last_x_ask = self.last_x_ind_ask
                    last_x_bid = self.last_x_ind_bid
                    last_y_ask = self.last_y_ind_ask
                    last_y_bid = self.last_y_ind_bid
                    last_z_ask = self.last_z_ind_ask
                    last_z_bid = self.last_z_ind_bid
                    last_x_value_ask = self.last_x_value_ask 
                    last_x_value_bid = self.last_x_value_bid
                    if self.data.loc[ind,'2'] == 0:
                        
                        # fill the l2 array
                        if(self.ask[pos, 0] != price): np.delete(self.ask, 0)         # nicht verstaandbar                        
                        volume = self.ask[pos,1] - self.data.loc[ind,'9']
                        if volume < 0: np.delete(self.ask, pos)
                        self.ask[pos,1] = volume
                        
                        # fill the trades array
                        if  last_x_value_ask == 0 or y_arr[self.last_x_ind_ask]!=price or last_x_value_ask < last_x_value_bid: 
                            if last_x_value_ask == 0:
                                cur_x_value = self.last_x_value_ask +1
                                cur_x_ind = self.last_x_ind_ask
                                cur_y_ind = self.last_y_ind_ask
                                cur_z_ind  = self.last_z_ind_ask
                            else:
                                cur_x_value = self.last_x_value_ask +1
                                cur_x_ind = self.last_x_ind_ask + 2 
                                cur_y_ind = self.last_y_ind_ask + 2
                                cur_z_ind  = self.last_z_ind_ask + 2
                            x_arr[cur_x_ind]= cur_x_value
                            y_arr[cur_y_ind] = price
                            z_arr[cur_z_ind] = self.data.loc[ind,'9']
                            self.last_x_ind_ask = cur_x_ind
                            self.last_y_ind_ask = cur_x_ind
                            self.last_z_ind_ask = cur_x_ind
                            self.last_x_value_ask = cur_x_value
                        else:
                            z_ind  = self.last_z_ind_ask
                            z_arr[z_ind] = z_arr[z_ind] + self.data.loc[ind,'9']
                            
                    else:

                        # fill the l2 array
                        if(self.bid[pos, 0] != price): np.delete(self.bid, 0)
                        volume = self.bid[pos,1] - self.data.loc[ind,'9']
                        if volume < 0: volume = 0
                        self.bid[pos,1] = volume
                       
                        # fill the trades array
                        if last_x_value_bid == 0 or y_arr[self.last_x_ind_bid]!=price or last_x_value_ask > last_x_value_bid: 
                            if last_x_value_bid == 0:
                                cur_x_value = self.last_x_value_bid +1
                                cur_x_ind = self.last_x_ind_bid + 1
                                cur_y_ind = self.last_y_ind_bid + 1
                                cur_z_ind  = self.last_z_ind_bid + 1
                            else:
                                cur_x_value = self.last_x_value_bid +1
                                cur_x_ind = self.last_x_ind_bid + 2 
                                cur_y_ind = self.last_y_ind_bid + 2
                                cur_z_ind  = self.last_z_ind_bid + 2
                            x_arr[cur_x_ind]= cur_x_value
                            y_arr[cur_y_ind] = price
                            z_arr[cur_z_ind] = self.data.loc[ind,'9']
                            self.last_x_ind_bid = cur_x_ind
                            self.last_y_ind_bid = cur_x_ind
                            self.last_z_ind_bid = cur_x_ind
                            self.last_x_value_bid = cur_x_value
                        else:
                            z_ind  = self.last_z_ind_bid
                            z_arr[z_ind] = z_arr[z_ind] + self.data.loc[ind,'9']
             
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

    #while True:
    #    myBook.UpdateBook(arr, x_arr, y_arr, z_arr)
             
    myBook.UpdateBook(arr, x_arr, y_arr, z_arr)
#    print(y_arr[:])
    sys.exit()
    
def runPQG(b_arr, x_arr, y_arr, z_arr):
    
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def update():
                
        
        start = time.time()
        global sp3, sp4
        
        w.removeItem(sp3)
        w.removeItem(sp4)
        # regular grid of starting positions
        pos1 = np.mgrid[0:10, 0:10, 0:1].reshape(3,10,10).transpose(1,2,0)
        # fixed widths, random heights

        x_pos = np.array(x_arr[:])
        x_max = np.max(x_pos)
        x_pos_ask = x_pos[0::2]
        x_pos_bid = x_pos[1::2]
        
        x_pos_ask = x_pos_ask.reshape(len(x_pos_ask), 1, 1)
        x_pos_bid = x_pos_bid.reshape(len(x_pos_bid), 1, 1)
                        
        y_pos = np.array(y_arr[:])
#        y_pos = y_pos / np.max(y_pos)
        y_pos_ask = y_pos[0::2]
        y_pos_bid = y_pos[1::2]
        
        y_pos_ask = y_pos_ask*100
        y_pos_bid = y_pos_bid*100
        y_pos_ask = y_pos_ask.reshape(len(y_pos_ask), 1, 1)
        y_pos_bid = y_pos_bid.reshape(len(y_pos_bid), 1, 1)
        y_pos_ask_nonzero =y_pos_ask[np.nonzero(y_pos_ask[-10:])] 
        y_med = np.mean(y_pos_ask_nonzero) 
        print("y_med ", y_med)
#        print("np.nonzero(y_pos_ask) ", y_pos_ask[np.nonzero(y_pos_ask)])
#        sys.exit()
        z_size = np.array(z_arr[:])
        z_max = np.max(z_size)
#        z_size = softmax(z_size)
        z_size_ask = z_size[0::2]/10
        z_size_bid = z_size[1::2]/10
        z_size_ask = z_size_ask.reshape(len(z_size_ask), 1)
        z_size_bid = z_size_bid.reshape(len(z_size_bid), 1)
        
        
        
        arr_pos_ask = np.append(x_pos_ask, y_pos_ask, axis = 2)
        z_pos_ask = np.zeros(y_pos_ask.shape)
        arr_pos_ask = np.append(arr_pos_ask, z_pos_ask, axis = 2)
        
        arr_pos_bid = np.append(x_pos_bid, y_pos_bid, axis = 2)
        z_pos_bid = np.zeros(y_pos_bid.shape)
        arr_pos_bid = np.append(arr_pos_bid, z_pos_bid, axis = 2)
        
        arr_size_ask = np.empty(arr_pos_ask.shape)
        arr_size_ask[..., 0:2] = 0.5
        arr_size_ask[..., -1] = z_size_ask
        
        arr_size_bid = np.empty(arr_pos_bid.shape)
        arr_size_bid[..., 0:2] = 0.5
        arr_size_bid[..., -1] = z_size_bid
        
        sp3 = gl.GLBarGraphItem(pos = arr_pos_ask, size = arr_size_ask)        
        sp4 = gl.GLBarGraphItem(pos = arr_pos_bid, size = arr_size_bid)
        sp3.setColor((0., 0., 1., 1.))
        sp4.setColor((3., 5., 4., 2.))
        pos_c = QVector3D(x_max-10, y_med,  0)
        w.setCameraPosition(pos = pos_c, distance = 40, azimuth = 90, elevation = 60)
        w.addItem(sp3)
        w.addItem(sp4)
        end = time.time()
#        print(arr_pos_ask[:10])
        
    global sp3, sp4
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 7
    w.show()
    w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
    pos_c = QVector3D(0, 0,  0)
    g = gl.GLGridItem()
#    print("cameraPosition ", w.cameraPosition())
    
    g.setSize(x=100,y=100,z=100)
    
    w.addItem(g)
    pos1 = np.mgrid[-1:0,2:3,0:1].reshape(3,1,1).transpose(1,2,0)
    
    size1 = np.empty((1,1,3)) 
    size1[...,0:2] = 0.4
    size1[...,2] = 0.4
    
    pos2 = np.mgrid[1:2,1:2,2:3].reshape(3,1,1).transpose(1,2,0)
    size2 = np.empty((1,1,3)) 
    size2[...,0:2] = 0.4
    size2[...,2] = 0.4
    
    sp3 = gl.GLBarGraphItem(pos = pos1, size = size1)
    sp3.setColor((0., 0., 1., 1.))
    sp4 = gl.GLBarGraphItem(pos = pos2, size = size2)
    sp4.setColor((3., 5., 4., 2.))
    w.addItem(sp3)
    w.addItem(sp4)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
        
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
    t = 1000
    v = 1
    k = 2
    x_arr = Array("d", t*k*v, lock = False)
    y_arr = Array("d", t*k*v, lock = False)
    z_arr = Array("d", t*k*v, lock = False)
    
    p = Process(target=runPQG, args=(b_arr, x_arr, y_arr, z_arr))
    p.start()
    MainProgram(b_arr, x_arr, y_arr, z_arr)
    p.join()
    
 
