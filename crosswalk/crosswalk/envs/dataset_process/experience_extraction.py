#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
"""
    0   Track ID. All rows with the same ID belong to the same path.
    1   xmin. The top left x-coordinate of the bounding box.
    2   ymin. The top left y-coordinate of the bounding box.
    3   xmax. The bottom right x-coordinate of the bounding box.
    4   ymax. The bottom right y-coordinate of the bounding box.
    5   frame. The frame that this annotation represents.
    6   lost. If 1, the annotation is outside of the view screen.
    7   occluded. If 1, the annotation is occluded.
    8   generated. If 1, the annotation was automatically interpolated.
    9  label. The label for this annotation, enclosed in quotation marks.
"""
class ReadAnnotations():
    def __init__(self, test=True):
        header = ['Track_ID','x','y','v_x','v_y','frame','lost','occluded','generated','label']    
        if test:
            self.data = pd.read_csv('input/test.txt',sep=" ", names=header, index_col=0)
            filename = 'test-1.txt'
        else:
            self.data = pd.read_csv('input/annotations.txt',sep=" ",names=header, index_col=0)
            filename = 'output/annotations_method2.txt'
        self.createnew(filename)

    def cal_pos(self):
        self.x = (self.data.loc[:,'x'] + self.data.loc[:,'v_x']).div(2.0)
        self.y = (self.data.loc[:,'y'] + self.data.loc[:,'v_y']).div(2.0)

    def cal_speed(self):

        self.delta_x = self.data['v_x'].copy()
        self.delta_y = self.data['v_y'].copy()
        self.delta_y.iloc[1:] = self.y.values[1:] - self.y.values[:-1]
        self.delta_x.iloc[1:] = self.x.values[1:] - self.x.values[:-1]

        for track_id in list(set(self.delta_x.index)):
            self.delta_x.loc[track_id].iat[0] = 0.0
            self.delta_y.loc[track_id].iat[0] = 0.0

    def createnew(self,filename):
        self.cal_pos()        
        self.cal_speed()
        # print(self.x.loc[1])
        # self.complement()

        self.newdata = self.data.copy()


        self.newdata['x'] = self.x
        self.newdata['y'] = self.y

        self.newdata['v_x'] = self.delta_x
        self.newdata['v_y'] = self.delta_y 
        
        # for track_id in list(set(self.newdata.index)):
        #     self.newdata.loc[track_id,].iat[0, 2] = 0.0
        #     self.newdata.loc[track_id,].iat[0, 3] = 0.0
        self.newdata.to_csv('output/annotations_with_speed.txt', sep=' ')
        self.newdata = self.newdata.sort_values(by=['frame']) 

        self.newdata = self.newdata.reset_index()
        self.newdata = self.newdata.set_index(['frame'])  

        self.newdata.to_csv(filename, sep=' ')

    
    def complement(self):
        # plt.figure()

        for track_id in list(set(self.delta_x.index)):
            pd.rolling_mean(self.delta_x.loc[track_id],10)
            pd.rolling_mean(self.delta_y.loc[track_id],10)
            #self.x.loc[track_id].interpolate(method='polynomial', order=2)
        plt.plot(self.x.loc[0].values)
        plt.show()

        
        




        


if __name__ == "__main__":
    read = ReadAnnotations(test=False)
    