import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import pandas as pd
from . import UsageFunctions as uf
from . import PltHelper as ph

uf = uf.UsageFunctions()
ph = ph.PltHelper()
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
pd.set_option('display.max_columns', 500)
color = sns.color_palette()
sns.set_style('darkgrid')

NUMERICS_COL_TYPE = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
STATISTIC_ORDER ={
    1: 'min',
    2: 'lower',
    3: '25%',
    4: '50%',
    5: '75%',
    6: 'upper',
    7: 'max',
    10:'mean',
    11:'std',
    12:'median',
    13:'riqr',
    14: 'count',
    
}

class DataFrameHandler:
    def __init__(self,df):
        self.df = df
        
    ### make Info df about cols ###
    """ numerical and str """
    def all_cols_info(self):
        uf.message('Detail descrption ALL cols',2)
        print('n_data',' : ',len(self.df))
        print('n_duplicated',' : ',self.df.duplicated().sum())
        self.get_info_df_numerical_cols()
        self.get_info_df_str_cols()

    """ Only numerical """
    def get_info_df_numerical_cols(self,col=None):
        dct_df_result = {}
        cols_dict = self.get_numerical_cols()
        if col is None:
            pass
        else:
            if col in cols_dict.keys():
                cols_dict = {col : cols_dict[col]}
            else:
                uf.message(col+'is not including in NUMRICAL COLS',10)
                return dct_df_result
        # get statistics
        df_statistics =  self.get_statistics_df_numerical_cols(cols_dict)
        dct_df_result['stat'] = df_statistics
        
        # get feature of each col
        df_isnull = pd.DataFrame(self.df[cols_dict.keys()].isnull().sum(),columns=['is_null']).T
        df_dtype =  pd.DataFrame({ key : [str(cols_dict[key])]  for key in cols_dict.keys() },index=['dtype'])
        df_merged = pd.concat([df_statistics,  df_isnull, df_dtype ], axis = 0).T
        df_merged['null_ratio']  = df_merged['is_null'].astype(float)/ df_merged['count'].astype(float)*100
        dct_df_result['feature'] = df_merged.T

        return dct_df_result
    
    """ numerical """
    def get_statistics_df_numerical_cols(self,cols_dict):
        
        df_quantile= self.df[cols_dict.keys()].describe().T
        uf.message('Get describe',20)
        
        # Cal for quantile
        df_quantile['riqr'] = (df_quantile['75%'] - df_quantile['25%'])*1.5
        df_quantile['upper'] = df_quantile['75%'] + df_quantile['riqr']
        df_quantile['lower'] = df_quantile['25%'] - df_quantile['riqr']
        df_quantile['upper'] = \
            df_quantile.apply(
                lambda _row : _row['upper'] if _row['upper'] < _row['max'] else _row['max'],
                axis=1
            )
            
        df_quantile['lower'] = \
            df_quantile.apply( 
                lambda _row : _row['lower'] if _row['lower'] > _row['min'] else _row['min'],
                axis=1
            )
        uf.message('Get qunatile',20)
            
        # Cal statistics which are not shown in dataframe.describe()
        df_median = pd.DataFrame(self.df[cols_dict.keys()].median(),columns=['median'])
        uf.message('Get median',20)
        
        # Concat
        df_stat = pd.concat([df_quantile, df_median ], axis = 1)
        
        # Change order for visibility
        df_stat = df_stat[[col for col in STATISTIC_ORDER.values()]].T
    
        return df_stat

    """ Only string """
    def get_info_df_str_cols(self):
        dct_df_result = {} 
        cols_dict = self.get_str_cols()
        
        # Cal info 
        df_n_unique =  pd.DataFrame(
            { key : len(self.get_unique_in_col(key)) for key in cols_dict.keys() },
            index=['n_unique']
        )
        uf.message('Get unique info',20)
        
        df_isnull = pd.DataFrame(
            self.df[cols_dict.keys()].isnull().sum(),
            columns=['is_null']
        ).T
        uf.message('Get null info',20)
        
        df_dtype =  pd.DataFrame(
            { key : [str(cols_dict[key])]  for key in cols_dict.keys() },
            index=['dtype']
        )
        uf.message('Get dtype info',20)
        # merge and be clearly
        df_merged = pd.concat([df_n_unique, df_isnull, df_dtype ], axis = 0).T
        df_merged['null_ratio']  = (df_merged['is_null'].astype(float)/ len(self.df))*100

        dct_df_result['feature'] = df_merged
        dct_df_result['cat'] = \
            pd.concat(
                [pd.DataFrame({ key : list(self.get_unique_in_col(key).index)}).T for key in cols_dict.keys()],axis=0
            ).fillna('-')
            
        return dct_df_result
        
    ### show info df ###
    """ only string """
    def get_str_col_info(self,col):
        df_count =self.get_unique_in_col(col).sort_values(by = col ,ascending=False)
        df_count['ratio'] = df_count.iloc[:,0]/sum(df_count.iloc[:,0])*100
    
    def show_str_col_info(self,col):
        df_count = get_str_col_info(col)
        display(df_count)
        
        ph.refresh()
        ph.set_cat_axis('Attr distribution',[col])
        bottom_y = 0     
        for i in range(len(df_count)):
            bottom_y += (df_count.iloc[i-1,0] if i>0 else 0)
            ph.accbarplot(0,list(df_count.iloc[i]),[bottom_y])

    
    """ only numerical """
    def show_numerical_col_info(self,col):
        cols_dict = self.get_numerical_cols()
        if not col in list(cols_dict.keys()) :
            uf.message('Not numerical col',10)
        df_describe =  self.get_info_df_numerical_cols()
        ph.refresh()
        ph.set_cat_axis('Attr distribution',[col])
        ph.boxplot(x_key=None, y_key =col,data = self.df,hue_key=None,is_swarm=False)
        ph.plot(0,df_describe[col].loc['mean'],'mean')
        display(df_describe[col])


    ### get df (target cols)###
    """ return type is colname : col's type """
    
    """ only numerical """
    def get_numerical_cols(self):
        return { col:self.df[col].dtype for col  in self.df.columns if self.df[col].dtype in NUMERICS_COL_TYPE} 

    """ only string """
    def get_str_cols(self):
        return { col:self.df[col].dtype for col  in self.df.columns if not self.df[col].dtype in NUMERICS_COL_TYPE} 
    
    ### set dtype  ###
    def set_cols_type(self,dct_convert):
        for col in dct_convert.keys():
            try:
                type_original = str(self.df[col].dtype)
                self.df[col] = self.df[col].astype(dct_convert[col])
                _message = 'Change dtype@',col,' : ',str(type_original) ,'->', str(dct_convert[col])
                uf.message(_message,20)
            except Exception as e:
                _message = 'Cannot change dtype@',col,' : ',type_original,'->', dct_convert[col]
                uf.message(_message,10)
                uf.message(e)

    """ only unique """
    def get_unique_in_col(self,col):
        return pd.DataFrame(self.df.reset_index().groupby([col]).count()['index']).rename(columns={'index': 'Count'})

    def get_unique_in_col_info(self,col):
        df_col_attr = pd.DataFrame(self.df.reset_index().groupby([col]).count()['index']).rename(columns={'index': 'Count'})
    