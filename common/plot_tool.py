import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import drange
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns

class Fig:
    def __init__(self,**kwargs):
        self.fig = plt.figure(
            figsize=kwargs['figsize'],
            facecolor='w', #fixed
            tight_layout=True
        )

class Ax:
    
    
    def __init__(self,**kwargs):
        self.fig= kwargs['fig']     
        self.ax = kwargs['ax']
        self.default = {}
#         self.default['size'] = (7,4)
#         self.default['color'] = 'red'
#         self.default['alpha'] = 0.8
#         self.default['maker'] = 'o'
#         self.default['makersize'] = 2
#         self.default['markerfacecolor'] = 'red'
#         self.default['markeredgecolor'] = None
#         self.default['markeredgewidth'] = 0
#         self.default['linestyle'] = '-'
#         self.default['linewidth'] = 1
#         self.default['loc'] = 'upper right'
#         self.default['bbox_to_anchor'] = (1,1)
#         self.default['borderaxespad'] = 1
#         self.default['fontsize'] = 14
#         self.default['cmap'] = 'tab20'

#     def parse_kwargs(self,ls_config_item,kwargs):
#         dct_config = {}
#         for config_item in ls_config_item:
#             dct_config[config_item]=kwargs[config_item] if(config_item in kwargs.keys())else self.default[config_item]
        
#         return dct_config
    #
    ####
    #'maker' = 'o'
    #'makersize' = 2
    ####
    def plot_line(self, _ls_x, _ls_y, _label, **kwargs):
        self.ax.plot(
            _ls_x, _ls_y,
            label=_label,
            **kwargs
        )
    
    def plot_box(self, data, **kwargs):
        #1.5 iQRまで
        if 'has_mean' in  kwargs.keys():
            kwargs['showmeans'] =True
            kwargs['meanprops']=dict(marker='o',markeredgecolor='k', markerfacecolor='r')
            del kwargs['has_means']
                    
        ax = sns.boxplot(
            data=data, 
            ax=self.ax,
            **kwargs
        )
        
    
    def set_ylim(self,minmax):
        _min,_max = minmax
        ymin,ymax = self.ax.get_ylim()
        ymin = ymin if _min is None else _min
        ymax = ymax if _max is None else _max
        self.ax.ylim(ymin,ymax)
        
    def set_xlimDt(self,_min,_max):
        _min_dt = _min
        _max_dt = _max
        delta = datetime.timedelta(minutes=1)
        x = drange(_min_dt, _max_dt, delta)
        self.ax.set_xlim(x.min(),x.max())
        
    def legend(self,**kwargs):
        
#         ls_config_item=['loc','bbox_to_anchor','borderaxespad','fontsize']
#         dct_config = self.parse_kwargs(ls_config_item,kwargs)
#         'loc',#図上基準点を凡例のどこに合わせるか(UPPER RIGHTなど) #bestは自動
#         'bbox_to_anchor',#図上基準点；左下(0, 0), 右上(1, 1)
#         'borderaxespad',#図上基準点との距離
#         'fontsize'
        self.ax.legend(
            **kwargs
        )
        
    def set_ticks(self,axis, ls_use_data_num ,is_minor,**kwargs):
        # 軸 の目盛りを設定する。(データの値ではなく、データの順番で指定)
        if axis == 'x':
            self.ax.set_xticks(ls_use_data_num,is_minor)
        elif axis == 'y':
            self.ax.set_yticks(ls_use_data_num,is_minor)
            
    def set_title(title):
        self.ax.set_title(title)
    
    def set_title(title):
        self.ax.set_title(title)

    def xticklabelsDateFormatter(self,_format):
        xaxis_ = self.ax.xaxis
        xaxis_.set_major_formatter(DateFormatter(_format))

    def grid(self,axis,is_major,**kwargs):   
        # 軸に目盛線を設定
        is_major = 'major' if is_major else 'minor'
        self.ax.grid(
            which = "major",
            axis = axis, 
            **kwargs
        )

    def tab20(self,color_num):
        return cm.tab20(color_num)
    
    def tab10(self,color_num):
        return cm.tab10(color_num)
