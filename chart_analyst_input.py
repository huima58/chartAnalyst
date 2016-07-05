# Copyright 2016 Hui Ma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the Chart Analysis Convolutional Neural Network.

Summary of available functions:

 # Draw charts and decide chart label.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For the chart ends on id, set it as 0 if within forward_bar_num, the low first drops to close - (5 days down median) * 1/3
# set it as 2 if within forward_bar_num, the high first rises to close + (5 days up median)  * 1/3
# otherwise, set it as 1
def ca_decide_chart_label(data, id, bar_num=128, forward_bar_num=5):
    data_shape = np.shape(data)
    num_limit = data_shape[0]
    # Cache the chart label for each id in static variable.
    # Should cache images for each id too, but model training takes much longer time than making chart image, 
    # so no images are cached for now.  
    if "chart_label" not in ca_decide_chart_label.__dict__: # cache the value in static variable  
        ca_decide_chart_label.chart_label = np.zeros(num_limit) - 1
    if ca_decide_chart_label.chart_label[id] != -1:
        return ca_decide_chart_label.chart_label[id]
    
    up_close, down_close = _get_close_change_median(data, id, bar_num, forward_bar_num)
    close = data['Close'][id]
    close_future = data['Close'][min(num_limit-1, id+forward_bar_num)]
    # Adjust rate is used to balance the percentage of the labels
    target_adjust_rate = 1. / 3.
    up_target = close * (1. + up_close * target_adjust_rate)
    down_target = close * (1. - down_close * target_adjust_rate)
    
    if close_future > up_target: 
        ca_decide_chart_label.chart_label[id] = 2
    elif close_future < down_target: 
        ca_decide_chart_label.chart_label[id] = 0
    else: 
        ca_decide_chart_label.chart_label[id] = 1
    return ca_decide_chart_label.chart_label[id]

# Draw a x_pix * y_pix line chart, with 3 lines for each column: high, low, and close.
# Similar functions can be developed to draw volume or technical indicators.
def ca_draw_line_chart(data, id, img, bar_num=128, x_pix_num=1280, y_pix_num=1280, line_width = 4):
    highest, lowest = _get_highest_lowest(data, id, bar_num)
    # x, y are related to col and row of the matrix
    pre_x_pix = 0
    pre_close = 0
    pre_high = 0
    pre_low = 0
    for i in range(1, bar_num + 1, 1):
        x_pix = (i - 1) * x_pix_num / bar_num;
        close = (1 - (data['Close'][id - bar_num + i] - lowest) / (highest - lowest)) * (y_pix_num - 1)
        low = (1 - (data['Low'][id - bar_num + i] - lowest) / (highest - lowest)) * (y_pix_num - 1)
        high = (1 - (data['High'][id - bar_num + i] - lowest) / (highest - lowest)) * (y_pix_num - 1)
        if i > 1:
            _draw_line(img, pre_x_pix, pre_close, x_pix, close, line_width)
            _draw_line(img, pre_x_pix, pre_high, x_pix, high, line_width)
            _draw_line(img, pre_x_pix, pre_low, x_pix, low, line_width)
        pre_x_pix = x_pix
        pre_close = close
        pre_high = high
        pre_low = low

def _draw_line(img, x1, y1, x2, y2, line_width = 4):
    if x2 == x1:
        if y2 >= y1:
            for y in range(y1, y2+1):
                _draw_dot_on_line(img, x1, y, line_width)
        else:
            for y in range(y2, y1 + 1):
                _draw_dot_on_line(img, x1, y, line_width)
        return
    if x2 < x1:
        x = x2
        x2 = x1
        x1 = x
        y = y2
        y2 = y1
        y1 = y
    pre_y = y1
    for x in range(x1, x2+1):
        y = (y2 - y1) * (x - x1) / (x2 - x1) + y1
        _draw_line_on_two_continuous_x(img, x-1, pre_y, x, y, line_width)
        pre_y = y
    return

# Try to fill two continuous dots with a line instead of just two dots.
def _draw_line_on_two_continuous_x(img, x1, y1, x2, y2, line_width = 4):
    if x2 != x1 + 1:
        return
    y_mid = (y1 + y2) / 2
    if y2 >= y1:
        for y in range(int(y1), int(y_mid)+1):
            _draw_dot_on_line(img, x1, y, line_width)
        for y in range(int(y_mid), int(y2) + 1):
            _draw_dot_on_line(img, x2, y, line_width)
    else:
        for y in range(int(y_mid), int(y1)+1):
            _draw_dot_on_line(img, x1, y, line_width)
        for y in range(int(y2), int(y_mid) + 1):
            _draw_dot_on_line(img, x2, y, line_width)

def _draw_dot_on_line(img, x, y, line_width = 4):
    y_limit, x_limit = np.shape(img)
    for i in range(0, line_width):
        if i + x < x_limit:
            img[y][x+i] = 1

def _get_highest_lowest(data, id, bar_num=128):
    highest = data['High'][id]
    lowest = data['Low'][id]
    for i in range(id-1, max(-1, id-bar_num), -1):
        if data['High'][i] > highest:
            highest = data['High'][i]
        if data['Low'][i] < lowest:
            lowest = data['Low'][i]
    return (highest, lowest)

def _get_close_change_median(data, id, bar_num=128, interval_bar_num=5):
    lst_up = []
    lst_down = []
    for i in range(id-interval_bar_num, id - bar_num, -interval_bar_num):
        if data['Close'][i+5] > data['Close'][i]:
            lst_up.append(data['Close'][i+5] / data['Close'][i] - 1)
        else:
            lst_down.append(1 - data['Close'][i+5] / data['Close'][i])
    return (np.median(np.array(lst_up)), np.median(np.array(lst_down)))

def cr_plot_image(img):
    plt.imshow(img, cmap='gray')

def read_csv_file(file_name):
    csv_data = pd.read_csv(file_name, delimiter=',')
    return csv_data