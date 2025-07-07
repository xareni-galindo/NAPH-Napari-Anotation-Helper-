# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:55:09 2022

@author: Xareni Galindo
"""
import sys
import functools
import napari
import random
import math 
import tifffile

from enum import Enum
from pathlib import Path
from time import sleep
from typing import List
from py3dbp import Packer, Bin, Item
from skimage import measure
from pathlib import Path

import numpy as np
import pandas as pd

import skimage.measure
from skimage.util import map_array

from napari.types import ImageData
from napari.types import LabelsData
from napari.layers import Labels

from qtpy.QtCore import Qt
from PyQt5.QtCore import pyqtSlot
from qtpy.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget, QPushButton
from superqt import (QLabeledDoubleRangeSlider, QLabeledDoubleSlider, QLabeledRangeSlider, QLabeledSlider)

from magicgui import magicgui
from magicgui import widgets
from magicgui.tqdm import trange
from magicgui.widgets import Table

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


viewer = napari.Viewer()

'''-----------------'''
class Color_Palettes(): 
    
    """Class to compute the color palettes use un the Napari labels layer 
    
    Parameters:
    ----------
    color : numpy array 
        default colors of the pre-segmented labels
    labels_2d :numpy array 
        segmented image
    labels_3d : numpy array
        3d labels to be added manually
    
    Methods:
    ----------
    
    color_CIELab_converter(color)
        converts RGB color coordinates to CIELab color coordinates
    
    color_difference(color_1, color_2)
        computes the color difference between two colors in the CIELab color space
    
    color_Palette_cv(color, labels_2d, labels_3d, color_dic ={0:'black'})
        computes the dolor dictionary use in the Napari labels layer
    
    colorBlind_Palette(color, labels_2d, labels_3d, color_dictionary={0:'black'})
        computes the dolor dictionary use in the Napari labels layer using color bling friendly colors 
    
    """
    def color_CIELab_converter(color): 
        
        """ Funtion to convert a aRGB valuet to CIELab values
        
        Parameters:
        ----------
        
        color : (numpy array)
            color must bee a set of 4 values where the first 3 define the parameters (red, green, and blue) 
            that defines the intensity of the color between 0 and 1. The 4th value defines the alpha value. 
            The alpha parameter is a number between 0.0 (fully transparent) and 1.0 (not transparent at all)  """
        
        color_rgb = sRGBColor(color[0], color[1], color[2]) # conver color value to sRGB color object
        color_Cielab = convert_color(color_rgb, LabColor) # Convert from RGB to Lab Color Space
        
        return color_Cielab
    
    def color_difference(color_1, color_2): 
        
        """ Function to calculate the color difference between to CIELab color values
        
        Parameters:
        ----------
        color_1, color_2 - must be color_object.LabColor   """
        
        color_diff = delta_e_cie2000(color_1, color_2) 
        
        return color_diff

 
    def color_Palette_cv(color, labels_2d, labels_3d, color_dic ={0:'black'}): 
        
        """ Function to generate a dictionnary that will be use as color palette in viewer.add_labels() method
        
        Parameters:
        ----------
        color : (numpy array)
            color must bee a set of 4 values where the first 3 define the parameters (red, green, and blue) 
            that defines the intensity of the color between 0 and 1. The 4th value defines the alpha value. 
            The alpha parameter is a number between 0.0 (fully transparent) and 1.0 (not transparent at all) 
            
        labels_ : (np.aray) 
            labels obtained from labeled images 
        new_labels : (np.aray)
            labels desired to add to our labeled image
        base_labels_color : (np.aray) 
            default color of the labels show by napari. 

        """

        base_CIELab = Color_Palettes.color_CIELab_converter(color) # Convert from RGB to Lab Color Space
        
        for label in labels_2d:
            color_dic[label] = color 
        
        color_diff = {} 
        for new in labels_3d:
            color_new_label =  np.random.rand(4) # generating a random numpy array with values between 0.0-1.0
            color_new_CIELab = Color_Palettes.color_CIELab_converter(color_new_label) #converting RGB color values to CIELab color values
            
            # Compute the color difference
            base_new_labels_diff = Color_Palettes.color_difference(base_CIELab, color_new_CIELab) 
            color_diff[new] = base_new_labels_diff
            
            while base_new_labels_diff <= 15:
                color_new_label =  np.random.rand(4) # generating a random numpy array with values between 0.0-1.0
                color_new_CIELab = Color_Palettes.color_CIELab_converter(color_new_label) #converting RGB color values to CIELab color values
                # Compute the color difference
                base_new_labels_diff = Color_Palettes.color_difference(base_CIELab, color_new_CIELab)             
                
                new +1

            color_new_label[3] = 1.0 #change last value to the define alpha value
            color_dic[new] =  color_new_label
        
        color_palette_ncv = color_dic 
        return color_palette_ncv
       
    
    #color_palette_cv=  color_Palette(base_labels_color, labels, new_labels) #generating the color palette 
    #print(len(color_palette_cv))
    #print(color_palette_cv[10])
    
    
    def colorBlind_Palette(color, labels_2d, labels_3d, color_dictionary={0:'black'}): 
        
        """ Function to generate a color blind friendly color palette.
        
        Parameters:
        ----------
        color : (numpy array)
            color must bee a set of 4 values where the first 3 define the parameters (red, green, and blue) 
            that defines the intensity of the color between 0 and 1. The 4th value defines the alpha value. 
            The alpha parameter is a number between 0.0 (fully transparent) and 1.0 (not transparent at all) 
            
        labels_ : (np.aray) 
            labels obtained from labeled images 
        new_labels : (np.aray)
            labels desired to add to our labeled image
        base_labels_color : (np.aray) 
            default color of the labels show by napari. 
         
        The colors use are the RGB representation of the next HEX color code     
        color_HEX_list =  ['#F5793A', '#A95AA1', '#85C0F9', '#0F2080', '#BDB8AD', '#44749D'] #, '#EBE7E0', '#C6D4E1'] 
        
        """
        
        #print("color blind palette")
        
        color_RGB_dic = {'1': np.asarray([0.96078431, 0.4745098 , 0.22745098, 1]), '2': np.asarray([0.6627451 , 0.35294118, 0.63137255, 1]), 
                         '3': np.asarray([0.52156863, 0.75294118, 0.97647059, 1]), '4': np.asarray([0.05882353, 0.1254902 , 0.50196078, 1]), 
                         '5': np.asarray([0.74117647, 0.72156863, 0.67843137, 1]), '6': np.asarray([0.26666667, 0.45490196, 0.61568627, 1])}  #'8': np.asarray([0.92156863, 0.90588235, 0.87843137]),
          
        for label in labels_2d:
            color_dictionary[label] = color 
        
        index = None 
        for label in labels_3d: 
            #print("label", label)
            index_ = str(random.randint(1, 6))
            #print("index_", index_)
            if  index_ == index :
                index_ = str(random.randint(1, 6))
                #print("new index_", index_)
            
            color_dictionary[label] =  color_RGB_dic[index_]  
            #print("color", color_RGB_dic[index_])              
            index = index_ 
        color_palette_cb = color_dictionary 
        
        return color_palette_cb
'''-----------------------------'''


'''-----------------------------'''
class Color_base_labels(Enum):
    """Enum for various media and their refractive indices."""
   
    Red =     [1.0, 0.0, 0.0, 1.0] # Funciona con las dos palettas de colores
    White =   [1.0, 1.0, 1.0, 1.0] # Funciona con las dos palettas de colores
    Yellow =  [1.0, 1.0, 0.0, 1.0] # Funciona con las dos palettas de colores
    Orange =  [1.0, 0.4, 0.0, 1.0] # Funciona con las dos palettas de colores
    Magenta = [1.0, 0.0, 1.0, 1.0] # Funciona con las dos palettas de colores
    
    '''
    Green1 =  [0.0, 1.0, 0.0, 1.0] # Funciona solo con la color blind palette 
    Blue =    [0.0, 0.0, 1.0, 1.0] # Funciona solo con la color blind palette
    Purple =  [0.5, 0.0, 1.0, 1.0] # Funciona solo con la color blind palette
    Cyan =    [0.0, 1.0, 1.0, 1.0] # Funciona solo con la color blind palette
    '''
'''-----------------------------'''


'''-----------------------------'''
def checkConsecutive(l):
    n = len(l) - 1
    return (sum(np.diff(sorted(l)) == 1) >= n) 
'''-----------------------------'''     


'''-----------------------------'''
def bin_size(nucei_dic:dict, padding_value: int):
    nucleus_width = [nucei_dic[nuclei].shape[2] for nuclei in nucei_dic]
    average_width = math.ceil(sum(nucleus_width) / len(nucleus_width))  

    nucleus_height = [nucei_dic[nuclei].shape[1] for nuclei in nucei_dic]
    average_height = math.ceil(sum(nucleus_height) / len(nucleus_height))  

    sqrt_no_elements =  math.ceil(math.sqrt(len(nucei_dic)))

    bin_width = math.ceil(sqrt_no_elements*average_width + sqrt_no_elements * padding_value *2)
    bin_height = math.ceil(sqrt_no_elements*average_height + sqrt_no_elements * padding_value *2)
    
    return bin_width, bin_height
'''-----------------------------'''


'''-------------'''


class Area_slider(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.slider_areas = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.slider_areas.setRange(0, 10000)#(min_area, max_area)
        self.slider_areas.setValue((0, 10000))#((min_area, max_area))
        self.slider_areas.setSingleStep(10)
        self.slider_areas.LabelPosition.LabelsAbove
        self.slider_areas.EdgeLabelMode.LabelIsValue
        self.genrate_image_area = QPushButton(' Generate area image ')
        self.genrate_image_area.clicked.connect(lambda: self.threshold_areas_color())
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.slider_areas)
        self.hbox.addWidget(self.genrate_image_area)        
        self.setLayout(self.hbox)
        self.setGeometry(300, 300, 550, 100)
        self.show()
        
        
    def threshold_areas_color(self, color_palette_area = None)-> napari.types.LayerDataTuple: #color_palette = {0:'black'})-> dict:
        
        image_label  = viewer.layers['Labels']
        labels_ = image_label.data
        
        if color_palette_area is None:
            color_palette_area = {0:'black'}
        
        values_areas = self.slider_areas._value
        area_lower_threhols = round(values_areas[0])
        area_high_threshold = round(values_areas[1])        
        print("lower:",area_lower_threhols, "higher:", area_high_threshold)
        
        props_areas = skimage.measure.regionprops_table(labels_, properties=('label','area'))
        props_areas_dF = pd.DataFrame.from_dict(props_areas)    
        props_dF_labels_list = props_areas_dF['label'].tolist()
        
        thresholded_area = props_areas_dF[(props_areas_dF['area']>area_lower_threhols) &
                                             (props_areas_dF['area']< area_high_threshold)]
       
        thresholded_label_list = thresholded_area['label'].tolist()
        
        color_base = 'magenta'
        print(color_base)
       
        for label in props_dF_labels_list:
            if label in thresholded_label_list:
                color_palette_area[label] = color_base
        colorareas = color_palette_area
        print(colorareas)
        
        image_threshold_area = viewer.add_labels(labels_, color = colorareas, name = "Area threshold", blending= "additive")
       
        return image_threshold_area  
        
    #area threshold
          

#app_area = QApplication(sys.argv)
widget_area_slider = Area_slider(viewer)
#sys.exit(app.exec_())
viewer.window.add_dock_widget(widget_area_slider , area= "top", name='Intensity threshold slider', add_vertical_stretch=True) 


class Intensity_slider(QWidget):
    def __init__(self, viewer):
        super().__init__()         
        self.slider_intensity = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.slider_intensity.setRange(0, 10000)#(min_int, max_int)
        self.slider_intensity.setValue((0, 10000))#((min_int, max_int))
        self.slider_intensity.setSingleStep(10)
        self.slider_intensity.LabelPosition.LabelsAbove
        self.slider_intensity.EdgeLabelMode.LabelIsValue
        self.genrate_image_intensity = QPushButton(' Generate intensity image ')
        self.genrate_image_intensity.clicked.connect(lambda: self.int_threshold())
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.slider_intensity)
        self.hbox.addWidget(self.genrate_image_intensity)        
        self.setLayout(self.hbox)
        self.setGeometry(300, 300, 550, 100)
        self.show()
        
    def int_threshold(self, color_palette_int = None): # color_palette_int = {0:'black'}
        
        image_label  = viewer.layers['Labels']
        labels_int = image_label.data        
        imageFluo = viewer.layers['Image'].data
        
        if color_palette_int is None:
            color_palette_int = {0:'black'}
            
        values_intensity = self.slider_intensity._value
        int_lower_threhols = round(values_intensity[0])
        int_high_threshold = round(values_intensity[1])
        print("lower:",int_lower_threhols, "higher:", int_high_threshold)
            
        image_props_int = skimage.measure.regionprops_table(labels_int , imageFluo, properties=('label','intensity_mean'))  
        image_props_int_DF = pd.DataFrame.from_dict(image_props_int)
        list_labels = image_props_int_DF['label'].tolist()
        print(list_labels)
        
        thresholded_int = image_props_int_DF[(image_props_int_DF['intensity_mean']>int_lower_threhols) &
                                             (image_props_int_DF['intensity_mean']< int_high_threshold) ]
        
        thresholded_int_labels = thresholded_int['label'].tolist()
        print(thresholded_int_labels)
        
        color_int_ = "yellow"

        for label_int in list_labels:
            if label_int in thresholded_int_labels: 
                color_palette_int[label_int] = color_int_ 
            
        colors = color_palette_int
        
        image_threshold_int = viewer.add_labels(labels_int , color = colors, name = "Intensity threshold", blending= "additive")
        
        return  image_threshold_int


#app_intensity = QApplication(sys.argv)
widget_intensity_slider = Intensity_slider(viewer)
#sys.exit(app.exec_())
viewer.window.add_dock_widget(widget_intensity_slider, area= "top", name='Intensity threshold slider', add_vertical_stretch=True)

@magicgui(filename2 =  {"label": "Flourescence:"}, layout="vertical")
def layer_image(filename2 = Path.home())-> napari.types.LayerDataTuple: # path objects are provided a file picker: 
    image_Fluo = tifffile.imread(filename2)
    image_ = np.array(image_Fluo)
    return (image_, {'name': 'Image'}, 'image') 


@magicgui(filename1 =  {"label": "Labels:"},
        LUT={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [("Color vision", 1), ("ColorBlind", 2)]},
        layout="vertical")
def layer_labels(Color_base= Color_base_labels.Red, LUT=1, filename1 = Path.home())-> napari.types.LayerDataTuple: # path objects are provided a file picker
    global image_Labels
    image_Labels = tifffile.imread(filename1)     
    min_val = np.amin(image_Labels)    
    image_Labels = np.where(image_Labels > min_val, image_Labels + 20000, 0).astype(int)
    
    labels_ = np.delete(np.unique(image_Labels), 0)
    new_labels = np.arange(1, labels_.shape[0] + 2000)
     
    base_labels_color = np.array(Color_base.value)
    
    if LUT == 1: 
        color_palette_dictionary = Color_Palettes.color_Palette_cv(base_labels_color, labels_, new_labels) #generating the color palette  
    elif LUT == 2:
        color_palette_dictionary = Color_Palettes.colorBlind_Palette(base_labels_color, labels_, new_labels) #generating the color palette  
        
    return (image_Labels, {'name': 'Labels', 'color':color_palette_dictionary ,  'opacity': 1.0, 'blending': 'additive'}, 'Labels') 


@magicgui(call_button="Print last Labels", result_widget=True,  layout="vertical")
def last_3Dlabels(image_labels: Labels) -> napari.types.LabelsData:
    labels_small= image_labels.data[viewer.dims.current_step[0]]  
    small_labels_list = [label for label in  np.unique(labels_small) if label < 20000] #list of unique labels above 10000
    result = small_labels_list[-1]
    return result

@magicgui(call_button="Print Big Labels",  result_widget=True, layout="vertical") 
def last_2Dlabels(image_labels: Labels)-> napari.types.LabelsData:   
    big_labels = image_labels.data[viewer.dims.current_step[0]]   
    big_labels_list = [label for label in  np.unique(big_labels) if label > 20000] #list of unique labels above 10000
    last_big = big_labels_list[-1]
    return last_big


@magicgui(call_button="Labels coordinates", layout="vertical", label={'max': 2**20})
def localize_labels(image_labels: Labels, label:int , x_coordinates="", y_coordinates="")-> str:
    frame= image_labels.data[viewer.dims.current_step[0]]  
    label_y_coor = np.argwhere(frame == label)[:, 0]
    label_x_coor = np.argwhere(frame == label)[:, 1]
    label_x_coor_print = label_x_coor[0:7].tolist()    
    label_y_coor_print =label_y_coor [0:7].tolist()
    str_x = ",".join(str(x) for x in label_x_coor_print)
    str_y = ",".join(str(y) for y in label_y_coor_print)
    
    localize_labels.x_coordinates.value = str_x
    localize_labels.y_coordinates.value = str_y
    return  str_x, str_y 


@magicgui(call_button="Labels_erros", layout="vertical")
def label_erros(image_labels: Labels, label_erros_list="", )-> str:
    labels_3D = image_labels.data
    unique_labels= np.delete(np.unique(labels_3D), 0).astype(int)
    
    errors = {}
    for label in unique_labels:
        z, y, x = np.where(labels_3D == label)
        
        single_nuclei = labels_3D[np.unique(z)[0]:np.unique(z)[-1] + 1, 
                                     np.unique(y)[0]:np.unique(y)[-1] + 1, 
                                     np.unique(x)[0]:np.unique(x)[-1] +1]
        
        single_nuclei_clean = np.where(single_nuclei == label, single_nuclei, single_nuclei* 0)    
        
        voxel = measure.label(single_nuclei_clean, background=0)
        num_of_comp = list(np.delete(np.unique(voxel), 0))
        
        if len(num_of_comp) > 1:  
            errors[label] = num_of_comp
    
    list_labels = list(errors.keys())  
    label_erros.label_erros_list.value = list_labels    
    return  list_labels


@magicgui(call_button="nuclei individualization", layout="horizontal", num_nuclei={'max': 2**20})
def nuclei_individualization(img_layer:"napari.layers.Image", image_labels: Labels, num_nuclei:int)-> napari.types.LayerDataTuple: 
    
    fluo_image = img_layer.data
    labels_3D = image_labels.data
    unique_labels= np.delete(np.unique(labels_3D), 0).astype(int)
    
    coordinates = {}
    single_nuclei_frames = {} 
    xy_padding_fluo= {}
    xy_padding = {}
    padding = 5
    
    for label in unique_labels:
        z, y, x = np.where(labels_3D == label)
        coordinates[label] = [np.unique(z), np.unique(y), np.unique(x)]
    
    random_indecex = random.sample(range(0, len(unique_labels)), num_nuclei) #[0, 1, 2, 563, 564, 565]#
    random_labels = [unique_labels[i] for i in random_indecex] 
    labes_indiv = {k:coordinates[k] for k in random_labels if k in coordinates}
    
    for label in random_labels: 
        z_index = labes_indiv[label][0]   
        y_index = labes_indiv[label][1]
        x_index = labes_indiv[label][2]
        
        single_nuclei = labels_3D[z_index[0]:z_index[-1] + 1, 
                                     y_index[0]: y_index[-1] + 1, 
                                     x_index[0]: x_index[-1] +1]
        
        fluo_single_nuclei =fluo_image[z_index[0]:z_index[-1] + 1, 
                                     y_index[0]: y_index[-1] + 1, 
                                     x_index[0]: x_index[-1] +1] 
        
        single_nuclei_clean = np.where(single_nuclei == label, single_nuclei, single_nuclei* 0)
        single_nuclei_frames[label] = single_nuclei_clean.shape[0]
        
        xy_zero_padding =  np.pad(single_nuclei_clean, [(0, 0), (padding, padding), (padding, padding)],
                               mode='constant', constant_values=0) 
        
        xy_zero_padding_fluo =  np.pad(fluo_single_nuclei, [(0, 0), (padding, padding), (padding, padding)],
                               mode='constant', constant_values=0) 
        
        xy_padding[label] = xy_zero_padding 
        xy_padding_fluo[label] = xy_zero_padding_fluo

    max_num_frames = max(single_nuclei_frames.values())


    width_bin, height_bin = bin_size(xy_padding, padding) #bin_width, bin_height #
    packer = Packer()
    images_bin =  Bin("Final Image", width_bin, height_bin, 1, 100000)  # my_bin = Bin(name, width, height, depth, max_weight)
    packer.add_bin(images_bin)

    individual_nuclei = {}
    individual_nuclei_fluo = {}
    for label in random_labels:
        z_padding = max_num_frames - single_nuclei_frames[label]
        z_zero_padding =  np.pad(xy_padding[label], [(0, z_padding), (0, 0), (0, 0)], mode='constant', constant_values=0)
        
        individual_nuclei[label] = z_zero_padding.astype(int)  
        individual_nuclei_fluo[label] = np.pad(xy_padding_fluo[label], [(0, z_padding), (0, 0), (0, 0)], mode='constant', constant_values=0).astype(int)  
        
        item = Item(str(label), z_zero_padding.shape[2], z_zero_padding.shape[1], 1, 1) # my_item = Item(name, width, height, depth, weight)
        packer.add_item(item) # packing the images in a contanainer 
        
    # packing the images in a contanainer placing the bigger package first 
    packer.pack(bigger_first=True, number_of_decimals=0)
    
    items_initial_position = {}
    items_final_position = {}
    bounding_box = {}
    
    
     
    for b in packer.bins: 
        for item in b.items:
            
            item_depth = int(item.depth)
            item_height = int(item.height)
            item_width = int(item.width)
            
            items_initial_position[int(item.name)] = [int(item.position[2]), int(item.position[1]), int(item.position[0])]

            z_final = int(item.position[2]) + max_num_frames
            y_final = int(item.position[1]) + item_height
            x_final = int(item.position[0]) + item_width
                           
            items_final_position[int(item.name)] = [z_final, y_final, x_final]
            
            bounding_box[int(item.name)] = np.array([[int(item.position[1]), int(item.position[0])], 
                                                    [int(item.position[1]), x_final], 
                                                    [y_final, x_final], 
                                                    [y_final, int(item.position[0])]])
    
    
    sorter =list(bounding_box.keys())           
    sorterIndex = dict(zip(sorter, range(len(sorter))))                                                  
    bunding_boxes_list = list(bounding_box.values())      

    depth_final =  max_num_frames #max(items_zfinal_position.values())
    height_final = max([element[1] for element in  list(items_final_position.values())])
    width_final  =  max([element[2] for element in  list(items_final_position.values())])

    final_image = np.empty((depth_final, height_final, width_final), dtype=int)
    final_image_fluo = np.empty((depth_final, height_final, width_final), dtype=int)


    for label in random_labels: 

        final_image[items_initial_position[label][0]:items_final_position[label][0],
               items_initial_position[label][1]: items_final_position[label][1], 
               items_initial_position[label][2]: items_final_position[label][2]] = individual_nuclei[label]
        
        final_image_fluo[items_initial_position[label][0]:items_final_position[label][0],
               items_initial_position[label][1]: items_final_position[label][1], 
               items_initial_position[label][2]: items_final_position[label][2]] = individual_nuclei_fluo[label]
    
    image_Labels_ind = final_image
    image_fluo_ind = final_image_fluo
    
    props = skimage.measure.regionprops_table(image_Labels_ind, properties=('label','area'))
    pos = np.argsort(props["area"])
    pos += 1 + math.ceil(pos.size / 5)   # to have distance from zero
    value_map2 = map_array(image_Labels_ind, props["label"], pos)


    p1 = pd.DataFrame.from_dict(props)
    p1['label_Rank'] = p1['label'].map(sorterIndex)
    p1.sort_values('label_Rank',
            ascending = True, inplace = True)
    p1.drop('label_Rank', 1, inplace = True)

    text_props = {
        'text': 'label: {label} volume:{area}',
        'anchor': 'upper_left',
        'translation': [5, 5],
        'size':10,
        'color': 'magenta',
    }
    '''
    #-----------------------
    steps=num_nuclei
    delay = 0.1
    for i in trange(steps):
        sleep(delay)
    #-----------------------
    '''
    return [(image_fluo_ind, {'name': 'Individualization Fluo', 'colormap':'gray', 'gamma':1, 'blending':'additive'}, 'image'),
            (value_map2, {'name':"volume heatmap", 'colormap':'inferno','opacity': 1.0, 'blending':'translucent'}, 'image'),
            (image_Labels_ind, {'name': 'Individualization','opacity': 1.0, 'blending':'translucent'}, 'Labels'), 
            (bunding_boxes_list, {'name': 'boundig box',  'properties': p1, 'text':text_props, 'shape_type':'rectangle', 
                                  'face_color':'transparent', 'edge_width':1, 'edge_color':'#ffffff', 'opacity': 1.0, 'blending':'translucent'}, 'Shapes') 
            ]


@magicgui(call_button="nuclei centroids", layout="horizontal")
def centroids(image_labels: Labels,  Coordinates:bool)-> napari.types.LayerDataTuple: 
    indivual_labels = image_labels.data
    props = skimage.measure.regionprops_table(indivual_labels, properties=('label','area', 'centroid'))
    
    props['centroid-0']=  np.around(props['centroid-0'], decimals=0)
    props['centroid-1']=  np.around(props['centroid-1'], decimals=0)
    props['centroid-2']=  np.around(props['centroid-2'], decimals=0)  
    
    points_z=  props['centroid-0']
    points_y=  props['centroid-1']
    points_x=  props['centroid-2'] 

    point_list = [np.array([pz, py, px]) for pz, py, px in zip(points_z, points_y, points_x)]
    
    if Coordinates == True: 
        
        text_points = {
            'text': 'x: {centroid-2}, y:{centroid-2}',
            'size': 10,
            'color': 'green',
            'translation': np.array([5, 5]),
            }
    elif Coordinates == False: 
        text_points = {}
    
    print(points_z, points_y, points_x)
    return (point_list, {'name': 'Centroids', 'size':1, 'properties': props, 'text':text_points}, 'points')



@magicgui(call_button="find nuclei", layout="horizontal")
def show_nuclei_ind(image_labels_ind: Labels, image_label_orig: "napari.layers.Labels")-> napari.types.LayerDataTuple:
    individualization_image = image_labels_ind.data
    image_Labels2= image_label_orig.data
    
    vals = np.delete(np.unique(individualization_image), 0)  
    mask = functools.reduce(np.logical_or, (image_Labels2==val for val in vals))
    masked = np.where(mask, image_Labels2, 0)

    return  (masked, {'name': 'Localization','opacity': 1.0, 'blending': 'translucent'}, 'Labels')


@magicgui(call_button="Properties Table", layout="horizontal")
def table_properties(image_labels: Labels, img_layer:"napari.layers.Image")-> dict:
    indivual_labels = image_labels.data
    image_fluo = img_layer.data
    props = skimage.measure.regionprops_table(indivual_labels, intensity_image= image_fluo, 
                                              properties=('label','area', 'centroid', 'intensity_min','intensity_mean','intensity_max', 'bbox'))
                                                                                                                                                        
    props['centroid-0']=  np.around(props['centroid-0'], decimals=0)
    props['centroid-1']=  np.around(props['centroid-1'], decimals=0)
    props['centroid-2']=  np.around(props['centroid-2'], decimals=0)  
    
    Table(props).show()
    return props



'''--------- adding widgets---------'''

viewer.window.add_dock_widget(layer_image, area= "right", name='Fluorescence Image', add_vertical_stretch=True)
viewer.window.add_dock_widget(layer_labels, area= "right", name='Labels', add_vertical_stretch=True)
viewer.window.add_dock_widget(last_3Dlabels, area= "right", name='Last label added', add_vertical_stretch=True)
viewer.window.add_dock_widget(last_2Dlabels, area= "right", name='2D labels remained', add_vertical_stretch=True)
viewer.window.add_dock_widget(localize_labels, area= "right", name='Label coordinates', add_vertical_stretch=True)
viewer.window.add_dock_widget(label_erros, area= "right", name='Label errors', add_vertical_stretch=True)
viewer.window.add_dock_widget(nuclei_individualization, area= "bottom", name='Nuclei individualization', add_vertical_stretch=True)
viewer.window.add_dock_widget(centroids, area= "bottom", name='Nuclei centroids', add_vertical_stretch=True)
viewer.window.add_dock_widget(show_nuclei_ind, area= "bottom", name='Nuclei localization', add_vertical_stretch=True)
viewer.window.add_dock_widget(table_properties, area= "bottom", name='Table properties', add_vertical_stretch=True)







