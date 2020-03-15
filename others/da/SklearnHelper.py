from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

class SklearnHelper:
    def __init__(self,df,n_fold):
        self.df = df
        self.param_list = {}
        self.clf = {}
        self.K = n_fold
        self.gscv = {}
    
    def set_CV(self,name):
        self.gscv[name] = GridSearchCV(
            self.clf[name],
            self.param_list[name],
            cv=n_fold,
            verbose=1
        )
        
    def do_CV_fit(self,name):
        self.gscv[name].fit(self.data['train']['X'], self.data['train']['y'])
        
        df_gs_result = pd.DataFrame.from_dict(elf.gscv[name].cv_results_)
        df_gs_result.to_csv('GsResult_'+name+'.csv')
        df_gs_result.sort_values(by='rank_test_score')
        sns.distplot(df_gs_result['rank_test_score'])
        plt.show()
        
    def do_CV_predict(self,name,use_clf =  None):
        # 最高性能のモデルを取得し、テストデータを分類 
        if use_clf is None:
            use_clf = self.gscv[name].best_estimator_
        pred = use_clf.predict(data['test']['X'])

        # 混同行列を出力
        print(confusion_matrix(data['test']['y'], pred))
        
    ### parameter　の意味を知るべき
    
    def set_DecisionTreeClassifier(self):
        self.param_list['DTC'] = {
            'criterion':['gini'],
            'max_depth':[2,4,6,8],
            'min_samples_split':[2,4,6,8],
            'min_samples_leaf':[1,2,3,4]
        }
        self.clf['DTC'] =  DecisionTreeClassifier()
        
    def set_RandomForestClassifier(self):
        self.param_list['RFC'] = {
            'criterion':['gini'],
            'max_depth':[2,4,6,8],
            'min_samples_split':[2,4,6,8],
            'min_samples_leaf':[1,2,3,4]
        }
        self.clf['RFC'] =  DecisionTreeClassifier()
        
    def set_RandomForestRegressor(self):
        self.param_list['RFR'] = {
            'criterion':['gini'],
            'max_depth':[2,4,6,8],
            'min_samples_split':[2,4,6,8],
            'min_samples_leaf':[1,2,3,4]
        }
        self.clf['RFR'] =  DecisionTreeClassifier()
        
    
    
    def set_split_datas(self,X,test_size=0.3,do_random=None):
        if do_random:
            do_random = 22
        data = { usage_type : { io_type : {}  } for usage_type in ['test','train'] for io_type in ['X','y']}
        data['train']['X'],data['test']['X'],data['train']['y'],data['test']['y'] = train_test_split(X, y, test_size = 0.3, random_state = do_random)
        