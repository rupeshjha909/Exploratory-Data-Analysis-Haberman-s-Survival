'''There are 4 features including class label/dependent variable.
1)age - It represents age of patient at the time of operation(numerical)
2)year_operation - It represents year of operation(numerical)
3)positive_lymph_nodes - It tells no of +ve auxillry node detected(numerical)
4)Survival status 1 = the patient survived 5 years or more
  Survival status 2 =  patients who survived less than 5 years
##############################################################################
OBJECTIVE:
    To classify/predict a patient survival who had undergone surgery for breast cancer.
'''
#importing the library
import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
import seaborn as sns

#importing the dataset
haberman=pd.read_csv('haberman1.csv')

#shape of data
haberman.shape

#columns of data
haberman.columns

#knowing first 7 data
haberman.head(7)

#knowing last 7 data
haberman.tail(7)

#knowing information summary
haberman.info()
#since no data is missing so we can go for further

#knowing STATISTICAL summary
haberman.describe()

#number of data of each category
haberman['survival_status'].value_counts()
#it is unbalance dataset 
#we will understand later how to handle unbalanced dataset
#lets start analysing our data

############## UNIVARIATE ANALYSIS ########################
#this analysis sumarizes data and finds patterns in the data
#it is done by histogram,pdf,cdf,box plot,violin plot

#HISTOGRAM
#age
sns.FacetGrid(haberman,hue="survival_status",size=5).map(sns.distplot,'age').add_legend()
plt.show()

#year_operation
sns.FacetGrid(haberman,hue="survival_status",size=5).map(sns.distplot,'year_operation').add_legend()
plt.show()

#positive_lymph_nodes
sns.FacetGrid(haberman,hue="survival_status",size=5).map(sns.distplot,'positive_lymph_nodes').add_legend()
plt.show()

# Here, Class 1 means survived
# Class 2 means not survived
#1)pdf(PROBABILITY DISTRIBUTION FUNCTION):
#pdf shows how many of points lies in some interval
class1=haberman.loc[haberman['survival_status']==1]
class2=haberman.loc[haberman['survival_status']==2]

#age
plt.title('class1')
plt.xlabel('age')
plt.ylabel('density')
count,bin_edges=np.histogram(class1['age'],bins=10,density=True)
pdf=count/(sum(count))
print(pdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.title('class1')
plt.xlabel('age')
plt.ylabel('density')
plt.plot(bin_edges[1:],cdf)
plt.show()

plt.title('class2')
plt.xlabel('age')
count,bin_edges=np.histogram(class2['age'],bins=10,density=True)
pdf=count/(sum(count))
print(pdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()

#year of operation
plt.title('class1')
plt.xlabel('year_operation')
count,bin_edges=np.histogram(class1['year_operation'],bins=10,density=True)
pdf=count/(sum(count))
print(pdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()

plt.title('class2')
plt.xlabel('year_operation')
count,bin_edges=np.histogram(class2['year_operation'],bins=10,density=True)
pdf=count/(sum(count))
print(pdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()

#positive_lymph_nodes
count,bin_edges=np.histogram(class1['positive_lymph_nodes'],bins=10,density=True)
plt.title('class1')
plt.xlabel('positive_lymph_nodes')
pdf=count/(sum(count))
print(pdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()

plt.title('class2')
plt.xlabel('positive_lymph_nodes')
count,bin_edges=np.histogram(class2['positive_lymph_nodes'],bins=10,density=True)
pdf=count/(sum(count))
print(pdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()

'''IMPORTANT POINTS(PDF):
***1)IN ALL THE PLOTS THE FEATURES ARE OVERLAPPING EACH OTHER MASSIVELY
***2)BUT PROBABILY 58% PEOPLE SURVIVED WHO HAD 0-5 AXLILARY_LYMPH_NODE AND 12% DIED AS WELL
***IMPORTANT POINTS(CDF):
   1)15% OF THE PEOPLE HAVE LESS THAN OR EQUAL T AGE 37 WHO SURVIVED
   2)PERSONS WHO HAVE MORE THAN 46 AUXILLARY_LYMPH_NODE NOT SURVIVED
    '''

'''BOX PLOT
    1) boxplot gives you the statistical summary of data
    2) Rectangle represent the 2nd and 3rd quartile (horizontal line either side of the rectangle)
    3) The horizontal line inside box represents median
'''
#age
sns.boxplot(x='survival_status',y='age',hue='survival_status',data=haberman).set_title("Box plot for survival_status and age")
plt.grid(color='r', linestyle='-', linewidth=2)
plt.show()
#year_operation
sns.boxplot(x='survival_status',y='year_operation',hue='survival_status',data=haberman).set_title("Box plot for survival_status and year_operation")
plt.grid(color='r', linestyle='-', linewidth=2)
plt.show()
#positive_lymph_nodes
sns.boxplot(x='survival_status',y='positive_lymph_nodes',hue='survival_status',data=haberman).set_title("Box plot for survival_status and positive_lymph_nodes")
plt.grid(color='r', linestyle='-', linewidth=2)
plt.show()


#VIOLIN PLOT
'''IMPORTANT POINT:
    1) The violin plot shows the full distribution of the data.
    2) It is combination of box plot and histogram
    3) central dot represents median
'''
#age
sns.violinplot(x='survival_status',y='age',hue = "survival_status",data=haberman,size=8)
#year_operation
sns.violinplot(x='survival_status',y='year_operation',hue = "survival_status",data=haberman,size=8)
#positive_lymph_nodes
sns.violinplot(x='survival_status',y='positive_lymph_nodes',hue = "survival_status",data=haberman,size=8)


############################# BIVARIATE ANALYSIS #########################
#scatter plot
haberman.plot.scatter(x='age',y='year_operation')
plt.show()
#any conclusion cant be drawn so let plot using color of survival status
'''if patients survived 5 years or more is represented as 1 and 
patients who survived less than 5 years is represented as 2'''
#plotting color CODING FOR EACH SURVIVAL SITUATION
sns.set_style("darkgrid")
sns.FacetGrid(haberman,hue='survival_status',size=4).map(plt.scatter,'age','year_operation').add_legend()
plt.title("2-D scatter plot for age and operation_year")
plt.show()

sns.set_style("darkgrid")
sns.FacetGrid(haberman,hue='survival_status',size=4).map(plt.scatter,'age','positive_lymph_nodes').add_legend()
plt.title("2-D scatter plot for age and positive_lymph_nodes")
plt.show()

#OBSERVATIONS FROM SCATTER PLOT:
'''
******1)In the above 2d scatter plot class label(i.e. a person died or survived) is not linearly seprable
******2)0-5 axillary_lymph_node person survived and died as well but the died ratio is less than survive ratio.
'''
#PAIR PLOT
sns.set_style("darkgrid")
sns.pairplot(haberman,hue='survival_status',size=4)

#OBSERVATIONS FROM SCATTER PLOT:
'''
******1)As we are unable to classify which is the most useful feature because of too much overlapping. But, Somehow we can say, In operation_year, 60-65 more person died who has less than 6 axillary_lymph_node.
******2)And hence, this plot is not much informative in this case.
'''
##########################** CONCLUSIONS **#############################
'''
1)The given dataset is imbalanced because:
    Survival status 1 = the patient survived 5 years or more is 225
  Survival status 2 =  patients who survived less than 5 years is 81
2)The given dataset is not linearly seprable form each class. 
  There are too much overlapping in the data-points and 
  hence it is very diffucult to classify.
3)somehow axillary_lymph_node is giving some intution in the dataset.
4)we can not build simple model using only if else condition 
  we need to have some more complex technique to handle this dataset.
'''








