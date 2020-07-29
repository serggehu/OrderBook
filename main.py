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
                    last_x_ask = self.last_x_ind_ask
                    last_x_bid = self.last_x_ind_bid
                    last_y_ask = self.last_y_ind_ask
                    last_y_bid = self.last_y_ind_bid
                    last_z_ask = self.last_z_ind_ask
                    last_z_bid = self.last_z_ind_bid
                    last_x_value_ask = self.last_x_value_ask 
                    last_x_value_bid = self.last_x_value_bid
                    if self.data.loc[ind,'2'] == 0:
                   #     print("ask ", price, " last_x ", last_x_ask, " self.data.loc[ind,'9'] ", self.data.loc[ind,'9'])
                        
                        # fill the l2 array
                        if(self.ask[pos, 0] != price): np.delete(self.ask, 0)         # nicht verstaandbar                        
                        volume = self.ask[pos,1] - self.data.loc[ind,'9']
                        if volume < 0: np.delete(self.ask, pos)
                        self.ask[pos,1] = volume
                        
                        # fill the trades array
                        if  last_x_value_ask == 0 or y_arr[self.last_x_ind_ask]!=price or last_x_value_ask < last_x_value_bid: 
                   #         print("last_x_value_bid ",  last_x_value_bid, " y_arr[last_x_ind_bid] ",  y_arr[self.last_x_ind_bid], " price ",  price, " last_x_value_ask ",  last_x_value_ask, " last_x_value_bid ",  last_x_value_bid)
                   #         print("INItiadet new ask")
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
                   #         print("added to last ask")
                            z_ind  = self.last_z_ind_ask
                            z_arr[z_ind] = z_arr[z_ind] + self.data.loc[ind,'9']
                            
                   #     print(" x ", x_arr[:])
                   #     print(" y ", y_arr[:])
                   #     print(" z ", z_arr[:])
                    
                    else:
                  #      print("bid ", price, " last_x ", last_x_bid, " self.data.loc[ind,'9'] ", self.data.loc[ind,'9'])
                        # fill the l2 array
                        if(self.bid[pos, 0] != price): np.delete(self.bid, 0)
                        volume = self.bid[pos,1] - self.data.loc[ind,'9']
                        if volume < 0: volume = 0
                        self.bid[pos,1] = volume
                       
                        # fill the trades array
                        if last_x_value_bid == 0 or y_arr[self.last_x_ind_bid]!=price or last_x_value_ask > last_x_value_bid: 
                 #           print("initiated new bid")
                            #print("last_x_value_bid ",  last_x_value_bid, " y_arr[last_x_ind_bid] ",  y_arr[self.last_x_ind_bid], " price ",  price, " last_x_value_ask ",  last_x_value_ask, " last_x_value_bid ",  last_x_value_bid)
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
                #            print("added to last bid")
                            z_ind  = self.last_z_ind_bid
                            z_arr[z_ind] = z_arr[z_ind] + self.data.loc[ind,'9']
                        
# =============================================================================
#                         print(" x ", x_arr[:])
#                         print(" y ", y_arr[:])
#                         print(" z ", z_arr[:])
# =============================================================================
             
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
    sys.exit()
    
def runPQG(b_arr, x_arr, y_arr, z_arr):
    
    
    def update():
        
        
        start = time.time()
        global sp3
        
        w.removeItem(sp3)
        # regular grid of starting positions
        pos1 = np.mgrid[0:10, 0:10, 0:1].reshape(3,10,10).transpose(1,2,0)
        # fixed widths, random heights
        x_pos = x_arr[:]
        y_pos = y_arr[:]
        z = z_arr[:]
        x_pos = np.array(x_pos).reshape(3, 2, 1)
        y_pos = np.array(y_pos).reshape(3, 2, 1)
        z_size = np.array(z).reshape(3,2,1)/10
        
        arr_pos = np.append(x_pos, y_pos, axis = 2)
        z_pos = np.zeros(y_pos.shape)
        arr_size = np.empty(arr_pos.shape)
        
        arr_pos = np.append(arr_pos, z_pos, axis = 2)
        arr_size = np.append(arr_size, z_size, axis = 2)
        arr_size[..., 0:2] = 0.2
        
 
        sp3 = gl.GLBarGraphItem(pos = arr_pos, size = arr_size)
        #sp3.setData(pos=pos3, color=color)
        w.addItem(sp3)
        end = time.time()
        
    
    global sp3
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
    pos_c = QVector3D(0, 33,  0)
    w.setCameraPosition(pos = pos_c, distance = 10, azimuth = 0, elevation = 30)
    g = gl.GLGridItem()
    print("cameraPosition ", w.cameraPosition())
    
    g.setSize(x=100,y=100,z=100)
    
    w.addItem(g)
    pos1 = np.mgrid[-10:-9,5:6,0:1].reshape(3,1,1).transpose(1,2,0)
    size1 = np.empty((1,1,3)) 
    size1[...,0:2] = 1
    size1[...,2] = 1
    sp3 = gl.GLBarGraphItem(pos = pos1, size = size1)
    w.addItem(sp3)

#    print("pos1 ", pos1)
#    print("size1 ", size1)

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
    t = 3
    v = 1
    k = 2
    x_arr = Array("d", t*k*v, lock = False)
    y_arr = Array("d", t*k*v, lock = False)
    z_arr = Array("d", t*k*v, lock = False)
    
    p = Process(target=runPQG, args=(b_arr, x_arr, y_arr, z_arr))
    p.start()
    MainProgram(b_arr, x_arr, y_arr, z_arr)
    p.join()
    
 
