import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import pandas as pd
class PltHelper:
    def __init__(self):
        self.refresh()

    def refresh(self): 
#         sns.set()
        sns.set_style('darkgrid') #style
        sns.set_palette('Paired') # colors
        self.axis = { key : {} for key in ['color','fontsize']}
        self.axis['color'] = 'black'
        self.axis['fontsize'] = 14
        color = sns.color_palette()
        self.fig, self.ax = plt.subplots()

    
    ### AXIS setting###
    """ X-axis (num)"""
    def set_num_axis(self,name,_min,_max,axis_interval):
        self.ax.set_xlim(_min,_max)
        ticks = []
        while True:
            tick = _min+axis_interval*len(ticks)
            if tick < _max:
                ticks.append(tick)
            else:
                break
        
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(ticks, fontsize=self.axis['fontsize'], color=self.axis['color'])
        self.ax.set_xlabel(name,  fontsize=self.axis['fontsize'], color=self.axis['color'])
        
    """ X-axis (cat)"""
    def set_cat_axis(self,name,tick_names):
        tick_dict = {i : tick_name for i,tick_name in enumerate(tick_names)}
        self.ax.set_xticks(list(tick_dict.keys()))
#         print(list(tick_dict.values()))
        self.ax.set_xticklabels(list(tick_dict.values()), fontsize=self.axis['fontsize'], color=self.axis['color'])
        self.ax.set_xlabel(name,  fontsize=self.axis['fontsize'], color=self.axis['color'])
#         ax.plot(x,y, 'k-', lw=1.5, alpha=0.6, label='theory')
#         ax.plot(x_a, y_a, 'o', color='none', markersize=10, markeredgewidth=3, markeredgecolor='blue', alpha=0.8, label='experiment'


    """ bar plot """
    def accbarplot(self,x_list,y_list,bottom_y_list=None):
        if bottom_y_list == None:
            bottom_y_list = [ 0 for _ in range(len(y_list))]
        self.ax.bar(x_list, y_list, bottom=bottom_y_list)
    
    def boxplot(self, x_key=None, y_key=None,data=None,hue_key=None,is_swarm=False):
        self.ax = sns.boxplot(x = x_key, y = y_key, hue = hue_key, data = data)
        if is_swarm : 
            self.ax = sns.swarmplot(x = x_key, y = y_key, hue = hue_key, data = data)
            
    """ point  plot """
    def plot(self,x=0,y=0,label=None):
        self.ax.plot(x,y,  label=label,color='red')
        
 
    def distplot():
        sns.distplot(df_gs_result['rank_test_score'])
        plt.show()