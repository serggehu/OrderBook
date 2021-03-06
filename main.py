from multiprocessing import Process, Value, Array
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
import sys
import copy

class Book():
    def __init__(self, array_size):
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
        
        self.array_size = array_size
        self.x_array = np.zeros([self.array_size])
        self.y_array = np.zeros([self.array_size])
        self.z_array = np.zeros([self.array_size])

        self.outlier_constant = 4

    def DataPrepare(self):
        data = pd.read_csv('/home/sergeu/Documents/env/DB/test2.csv', sep = ";")
        
        data.columns = ["1","2","3","4","5","6","7","8","9"]
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
    
    def StackArray(self):
                 
        elems_to_save = self.array_size - 2 
        x_arr_temp = self.x_array[-elems_to_save:]
        self.x_array[:elems_to_save] = x_arr_temp
        self.x_array[elems_to_save:] = 0

        y_arr_temp = self.y_array[-elems_to_save:]
        self.y_array[:elems_to_save] = y_arr_temp
        self.y_array[elems_to_save:] = 0 

        z_arr_temp = self.z_array[-elems_to_save:]
        self.z_array[:elems_to_save] = z_arr_temp
        self.z_array[elems_to_save:] = 0
        
        self.last_x_ind_ask = self.last_x_ind_ask - 2
        self.last_x_ind_bid = self.last_x_ind_bid - 2

        self.last_y_ind_ask = self.last_y_ind_ask - 2 
        self.last_y_ind_bid = self.last_y_ind_bid - 2 

        self.last_z_ind_ask = self.last_z_ind_ask - 2
        self.last_z_ind_bid = self.last_z_ind_bid - 2

    def ProcessOutliers(self):      # find outliers in y array and assigh to prev value
        
        if np.count_nonzero(self.y_array) < self.array_size-10:
            return

        upper_quantile = np.percentile(self.y_array, 75)
        lower_quantile = np.percentile(self.y_array, 25)
        IQR = (upper_quantile - lower_quantile) * self.outlier_constant
        quantile_set = (lower_quantile - IQR, upper_quantile + IQR)
        result_list = []
        y_list = self.y_array.tolist()
        for i in range(len(self.y_array)):

            if self.y_array[i]>0 and (self.y_array[i] < quantile_set[0] or self.y_array[i] > quantile_set[1]):
                if self.y_array[i-2] > 0:
                    self.y_array[i] = self.y_array[i-2]
                else:
               
                    self.y_array[i] = self.y_array[i+2]       
        
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
                        
                    last_x_value_ask = self.last_x_value_ask 
                    last_x_value_bid = self.last_x_value_bid
                    if self.data.loc[ind,'2'] == 0:
                        # fill the l2 array
                        if(self.ask[pos, 0] != price): np.delete(self.ask, 0)         # nicht verstaandbar                        
                        volume = self.ask[pos,1] - self.data.loc[ind,'9']
                        if volume < 0: np.delete(self.ask, pos)
                        self.ask[pos,1] = volume
                        
                        # fill the trades array
                            # initiate new price pair in array
                        if(self.x_array[-2]!=0 or self.x_array[-1]!=0): 

                            self.StackArray()

                        if  last_x_value_ask == 0 or self.y_array[self.last_x_ind_ask]!=price or last_x_value_ask < last_x_value_bid: 
                            # if the  x array is full delete first elemes

                            if last_x_value_ask == 0: # if the 1 elem of arra has 0 index
                                cur_x_value = self.last_x_value_ask +1
                                cur_x_ind = self.last_x_ind_ask
                                cur_y_ind = self.last_y_ind_ask
                                cur_z_ind  = self.last_z_ind_ask
                            else:
                                cur_x_value = self.last_x_value_ask +1
                                cur_x_ind = self.last_x_ind_ask + 2 
                                cur_y_ind = self.last_y_ind_ask + 2
                                cur_z_ind  = self.last_z_ind_ask + 2
                            self.x_array[cur_x_ind]= cur_x_value
                            self.y_array[cur_y_ind] = price
                            self.z_array[cur_z_ind] = self.data.loc[ind,'9']
                            self.last_x_ind_ask = cur_x_ind
                            self.last_y_ind_ask = cur_x_ind
                            self.last_z_ind_ask = cur_x_ind
                            self.last_x_value_ask = cur_x_value

                        else:
                            z_ind  = self.last_z_ind_ask
                            self.z_array[z_ind] = z_arr[z_ind] + self.data.loc[ind,'9']

                    else:                        
                        if(self.bid[pos, 0] != price): np.delete(self.bid, 0)   # fill the l2 array
                        volume = self.bid[pos,1] - self.data.loc[ind,'9']
                        if volume < 0: volume = 0
                        self.bid[pos,1] = volume
                       
                        # fill the trades array
                            # initiate new price pair in array

                        if(self.x_array[-1]!=0 or self.x_array[-2]!=0):       # if the  x array is full delete first elemes 
#                            print("before bid", self.x_array)
                            self.StackArray()
#                            print("after bid", self.x_array)

                        if last_x_value_bid == 0 or self.y_array[self.last_x_ind_bid]!=price or last_x_value_ask > last_x_value_bid: 
 
                            if last_x_value_bid == 0:   # if the 1 elem of arra has 0 index
                                cur_x_value = self.last_x_value_bid +1
                                cur_x_ind = self.last_x_ind_bid + 1
                                cur_y_ind = self.last_y_ind_bid + 1
                                cur_z_ind  = self.last_z_ind_bid + 1
                            else:
                                cur_x_value = self.last_x_value_bid +1
                                cur_x_ind = self.last_x_ind_bid + 2 
                                cur_y_ind = self.last_y_ind_bid + 2
                                cur_z_ind  = self.last_z_ind_bid + 2

                            self.x_array[cur_x_ind]= cur_x_value
                            self.y_array[cur_y_ind] = price
                            self.z_array[cur_z_ind] = self.data.loc[ind,'9']
                            self.last_x_ind_bid = cur_x_ind
                            self.last_y_ind_bid = cur_x_ind
                            self.last_z_ind_bid = cur_x_ind
                            self.last_x_value_bid = cur_x_value
                            
                        else:
                            z_ind  = self.last_z_ind_bid
                            self.z_array[z_ind] = z_arr[z_ind] + self.data.loc[ind,'9']

                else:                      #update OrderBook
                                            
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
                self.ProcessOutliers()
                for k in range(self.array_size-1):
                    x_arr[k] = self.x_array[k] 
                    y_arr[k] = self.y_array[k]
                    z_arr[k] = self.z_array[k]
#            print("x__arr ", x_arr[:], " y_arr ",y_arr[:])
            
def MainProgram(arr, x_arr, y_arr, z_arr):

    myBook.UpdateBook(arr, x_arr, y_arr, z_arr)
    sys.exit()
    
def runGraph(b_arr, x_arr, y_arr, z_arr):
    
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    # Create figure for plotting
    fig, ax = plt.subplots()
    num_bars = int(len(x_arr)/2)-2

    positions_ask = np.arange(1, num_bars, 1)
    positions_bid = positions_ask[:]
    height_list_ask = random.sample(range(30, 661), num_bars)
    height_list_bid = [-h for h in height_list_ask]
    positions_ask = random.sample(range(1, 662), num_bars)
    positions_bid = [-p for p in positions_ask]
    bottom_list_ask = random.sample(range(1, 601), num_bars)
    bottom_list_bid = [-b for b in bottom_list_ask]
    bars_ask = ax.bar(x=positions_ask, height=height_list_ask, width=0.8, bottom=bottom_list_ask)
    bars_bid = ax.bar(x=positions_bid, height=height_list_bid, width=0.8, bottom=bottom_list_bid)

    bs_ask = [b for b in bars_ask]
    bs_bid = [b for b in bars_bid]

    def animate(i, x_arr):

 #       if (x_arr[0] == 0 and x_arr[1]== 0):
 #           return    
        
        x_pos = np.array(x_arr[:])
        x_pos = x_pos[np.nonzero(x_pos)]
        x_pos_ask = x_pos[0::2]
        x_pos_bid = x_pos[1::2]
        x_pos_ask = x_pos_ask[-num_bars:]
        x_pos_bid = x_pos_bid[-num_bars:]

        y_pos = np.array(y_arr[:])
        y_pos = y_pos[np.nonzero(y_pos)]
        y_pos_ask = y_pos[0::2]
        y_pos_bid = y_pos[1::2]
        y_pos_ask = y_pos_ask[-num_bars:]
        y_pos_bid = y_pos_bid[-num_bars:]        
    
        z_size = np.array(z_arr[:])    
        z_size = z_size[np.nonzero(z_size)]
        z_size = [z/100 for z in z_size]
        z_size_ask = z_size[0::2]
        z_size_bid = z_size[1::2]
        z_size_ask = z_size_ask[-num_bars:]
        z_size_bid = z_size_bid[-num_bars:]
        

        for x, y, z, b in zip(x_pos_ask, y_pos_ask, z_size_ask, bs_ask):

            b.set_x(x)
            b.set_y(y)
            b.set_height(z/100)
            
        
        for x, y, z, b in zip(x_pos_bid, y_pos_bid, z_size_bid, bs_bid):

            b.set_x(x)
            b.set_y(y)
            b.set_height(-z/100)
            
        ax.relim()
        ax.autoscale_view()

        return bs_ask, bs_bid
        
    ani = animation.FuncAnimation(fig,
        animate, fargs=(x_arr, ), interval=200, blit=False)
    plt.show()
    

if __name__ == '__main__':
    
    t = 250
    v = 1
    k = 2
    myBook = Book(array_size = t*k*v)
    myBook.data=myBook.DataPrepare()
    myBook.init_time = int(str(myBook.data.loc[0, "3"]) + str(myBook.data.loc[0, "4"]))
    n = 128
    m = 2
    b_arr = Array('d', n*m)
    x_arr = Array("d", t*k*v, lock = False)
    y_arr = Array("d", t*k*v, lock = False)
    z_arr = Array("d", t*k*v, lock = False)
    
    p = Process(target=runGraph, args=(b_arr, x_arr, y_arr, z_arr))
    p.start()
    MainProgram(b_arr, x_arr, y_arr, z_arr)
    p.join()
    
 
