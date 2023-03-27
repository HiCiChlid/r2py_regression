from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import globalenv
import pandas as pd
import numpy as np
pandas2ri.activate()
importr("nnet")
importr("DescTools")
importr("summarytools")
class MNlogit(object):

    def __init__(self, path, encoding, dependent:str, independent_list: list, reference:str):
        self.path=path
        self.encoding=encoding
        self.read_file()
        rdf = pandas2ri.py2rpy(self.df)
        globalenv['rdf'] = rdf
        self.dependent=dependent
        self.independent_list=independent_list
        self.reference=reference

        rscript = """
        rdf$%s <- relevel(as.factor(rdf$%s), ref='%s')
        """%(self.dependent, self.dependent, self.reference)
        r(rscript)


    def summary(self):
        self.run_OIM()
        self.run()
        self.get_coefficients()
        self.get_z()
        self.get_pvalue()
        self.get_std_error()
        self.get_log_likelihood()
        self.get_goodness_of_fit()
        self.get_pseudo_r_square()
        self.get_exp_coef()
        self.get_classification()
        self.get_zWald()

        # vcoefname
        rscript = """
        summary(test)$vcoefnames
        """
        names=r(rscript)

        # lab
        rscript = """
        summary(test)$lab
        """
        a=r(rscript)
        print(a)
        lab=a[1:]

        # AIC
        rscript = """
        summary(OIM)$AIC
        """
        aic=r(rscript)[0]
        print("null AIC: %s"%aic)

        rscript = """
        summary(test)$AIC
        """
        aic=r(rscript)[0]
        print("AIC: %s"%aic)

        sum_square=0
        for j in list(set(self.df[self.dependent])):
            perc=len(self.df[self.df[self.dependent]==j])/len(self.df)
            print("%s: %.3f"%(j, perc))
            sum_square+=perc**2

        for i in range(0, len(self.coefficients)):
            print(lab[i])
            dict1={}
            dict1['attr']=names
            dict1['coefficients']=self.coefficients[i]
            dict1['standard_error']=self.ste[i]
            dict1['z']=self.z[i]
            dict1['p']=self.p[i]
            dict1['exp(coef)']=self.exp_coef[i]
            output=pd.DataFrame(dict1)
            output['coefficients']=output.apply(lambda x: _mark(x['coefficients'],x['p']), axis=1)
            print(output[['attr','coefficients','exp(coef)']])
            #print("全的参数估计")
            #print(output)
        print("Hipotesis 测试:%s"%self.zWald)
        print("Model fit information \n")
        print(self.log_likelihood)
        print("\n")
        print("Test the goodness of fit \n")
        print(self.goodness_of_fit)
        print("\n")
        print("Pseudo R-Square \n")
        print("CoxSnell: %.3f, \nNagelkerke: %.3f, \nMcFadden: %.3f, \n"%tuple(self.pseudo_r_square))  
        print("Build a classification table \n")
        a=self.classification
        b=np.diagonal(a).sum()/a.sum()
        c=b/sum_square-1
        if c>0.25:
            print("good: 原来是：%.3f, 现在是：%.3f, 提高了：%.3f"%(sum_square,b,c))
        else:
            print("bad: 原来是：%.3f, 现在是：%.3f, 提高了：%.3f"%(sum_square,b,c))  
        


        
    def read_file(self):
        if isinstance(self.path, str):
            sux=self.path.split(".")[-1]
            if sux=='pkl':
                self.df=pd.read_pickle(self.path, encoding=self.encoding)
            elif sux=='csv':
                self.df=pd.read_csv(self.path, encoding=self.encoding)
            elif sux=='parquet':
                self.df=pd.read_parquet(self.path, encoding=self.encoding)
            else:
                print("please input a pandas dataframe !!")
        elif isinstance(self.path, pd.DataFrame):
            self.df=self.path
        else:
            print("please input a pandas dataframe !!")
        
    
    def run_OIM(self):
        rscript = """
        OIM <- multinom(%s ~ 1, data = rdf)
        """%self.dependent
        r(rscript)

    def run(self):
        temp=""
        for i in self.independent_list:
            temp+="%s +"%i
        independent_str=temp[:-1]
        rscript = "test <- multinom(%s ~ %s, data = rdf)"%(self.dependent, independent_str)
        r(rscript)

    def get_coefficients(self):
        rscript = 'summary(test)$coefficients'
        self.coefficients=r(rscript)
        
    def get_z(self):
        rscript = """
        z <- summary(test)$coefficients/summary(test)$standard.errors
        """
        self.z=r(rscript)

    def get_zWald(self):
        rscript = """
        zWald_test <- function(x){
            a <- t(apply(z, 1, function(x) {x < qnorm(0.025, lower.tail = FALSE)} ))
            b <- t(apply(z, 1, function(x) {x > -qnorm(0.025, lower.tail = FALSE)} ))
            ifelse(a==TRUE & b==TRUE, TRUE, FALSE)
            }
        zWald_test(test)
        """
        self.zWald=r(rscript)

    def get_pvalue(self):
        rscript = """
        p <- (1 - pnorm(abs(z), 0, 1)) * 2
        """
        self.p=r(rscript)

    def get_std_error(self):
        rscript = """
        summary(test)$standard.errors
        """
        self.ste=r(rscript)

    def get_log_likelihood(self):
        #detach("package:jmv", unload=TRUE)
        rscript = """
        log_likelihood <- anova(OIM,test)
        """
        self.log_likelihood=r(rscript)

    def get_goodness_of_fit(self):
        rscript = """
        chisq.test(rdf$%s,predict(test))
        """%self.dependent
        self.goodness_of_fit=r(rscript)

    def get_pseudo_r_square(self):
        rscript = """
        PseudoR2(test, which = c("CoxSnell","Nagelkerke","McFadden"))
        """
        self.pseudo_r_square=r(rscript)

    def get_exp_coef(self):
        rscript = """
        exp(coef(test))
        """
        self.exp_coef=r(rscript)       

    def get_classification (self):
        rscript = """
        ctable <- table(rdf$%s,predict(test))
        """%self.dependent
        self.classification =r(rscript)

def _mark(coef,p):
    if abs(coef)<0.001:
        coef=float("%.2f"%coef)
    else:
        coef=float("%.2f"%coef)
    if p<0.01:
        return "%s**"%coef
    elif p<0.05:
        return "%s*"%coef
    else:
        return "%s"%coef
    
