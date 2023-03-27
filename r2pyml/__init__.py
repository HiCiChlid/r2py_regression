from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import globalenv
from functools import wraps
import pandas as pd
import numpy as np
import geopandas as gpd
import sys
current = sys.stdout
importr("car")
pandas2ri.activate()

class ML(object):

    def __init__(self, path, encoding, dependent:str, independent_list: list):
        self.path=path
        self.encoding=encoding
        self.read_file()
        rdf = pandas2ri.py2rpy(self.df)
        globalenv['rdf'] = rdf
        self.dependent=dependent
        self.independent_list=independent_list
        
    def back2var(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs): 
            f = open('./default.log', 'w')
            sys.stdout = f
            func(self)
            sys.stdout = current
            with open('./default.log', 'r') as f:
                txt=f.readlines()
            return txt
        return wrapper

    def read_file(self):
        if isinstance(self.path, str):
            sux=self.path.split(".")[-1]
            if sux=='pkl':
                self.df=pd.read_pickle(self.path, encoding=self.encoding)
            elif sux=='csv':
                self.df=pd.read_csv(self.path, encoding=self.encoding)
            elif sux=='parquet':
                self.df=pd.read_parquet(self.path, encoding=self.encoding)
            elif sux=='.shp':
                self.df=gpd.read_file(self.path, encoding=self.encoding)
            else:
                print("please input a pandas dataframe !!")
        elif isinstance(self.path, pd.DataFrame):
            self.df=self.path
        else:
            print("please input a pandas dataframe !!")
        
    def run(self,step=False):
        temp=""
        for i in self.independent_list:
            temp+="%s +"%i
        independent_str=temp[:-1]
        if step==False:
            rscript = "test <- lm(%s ~ %s, data = rdf)"%(self.dependent, independent_str)
        else:
            rscript = "test <- step(lm(%s ~ %s, data = rdf))"%(self.dependent, independent_str)
        r(rscript)

    @back2var
    def summary(self):
        rscript = """
        summary(test)
        """
        print(r(rscript))
        print("VIF")
        self.get_vif()
    
    def get_vif(self):
        rscript = """
        print(vif(test))
        """
        r(rscript)
    
