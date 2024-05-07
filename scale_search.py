import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import sys

#read parameters
subset_name='art'
#subset_name=sys.argv[1]
cg_type='fft'
#cg_type=sys.argv[2]

#pickle load
handle=open('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.pickle','rb')
df=pickle.load(handle)

#remove outliers


idx=[]

if subset_name=='suprematism':
	idx=[]#[50, 95, 64, 65, 52, 53, 13, 48, 20]
elif subset_name=='advertisement':
	msk=df['ms_total']>0.3 
	idx = df.index[msk]
elif subset_name=='art':
	idx=[119,107,88,30,200, 225,380, 328, 286, 216, 316, 332, 86, 108, 73, 23, 285,31]
	#idx=[328, 80, 86, 225, 286, 352, 380, 119] for tasteless images
	#idx=[380, 225, 30, 88, 119, 286, 107] for broad lines
elif subset_name=='scenes':
	idx=[52, 22]
'''elif subset_name=='infographics':
	msk=df['ms_total']>0.5 
	idx = df.index[msk]'''



'''
x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['ms_total'].to_numpy()

y1o=outliers['gt'].to_numpy()
y2o=outliers['ms_total'].to_numpy()
'''

outliers=df.loc[idx]
df=df.drop(idx)

#separate df for statistics
#regression for each separate scale

#regression for each separate scale: loop
reg_results=[]
y=df['gt']
features = [f"s{i}" for i in range(42)]


for i in range(42):
	#regression
	x=df[features[i]]
	#xo=outliers['ms_total']
	#yo=outliers['gt']
	if x.any():
		slope, intercept, r, p, std_err = stats.linregress(x, y)
		reg_results.append([r,p])
	else:
		reg_results.append([0,0])

df_stats = pd.DataFrame(reg_results, columns=['r', 'p'])

df_stats.to_csv('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity_regression.csv', sep='\t')


df['frac']=df['s15']+df['s16']+df['s17']+df['s18']+df['s19']
outliers['frac']=outliers['s15']+outliers['s16']+df['s17']+df['s18']+df['s19']
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(subset_name+', 15-19: r=', str(r)+', p='+str(p))
freqrange='15-19'


#regression
x=df['frac']
y=df['gt']
xo=outliers['frac']
yo=outliers['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.xlabel('human ranking',fontsize=16)
plt.ylabel('multi-scale structural complexity',fontsize=16)
plt.scatter(xo,yo,color='red', linewidth=2, alpha=0.5)
plt.plot(x, y1, color='orange')
plt.title(subset_name+', linear regression', fontsize=20)
plt.title(cg_type+' cg, '+subset_name+' regression (middle scale only - broader).png')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression_'+freqrange+'.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_'+freqrange+'.eps', format='eps')

df_part=df[['gt','frac']]
df_part.to_csv('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity_mid.csv', sep='\t')


#regression full
x=df['ms_total']
y=df['gt']
xo=outliers['ms_total']
yo=outliers['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.xlabel('human ranking',fontsize=16)
plt.ylabel('multi-scale structural complexity',fontsize=16)
plt.scatter(xo,yo,color='red', linewidth=2, alpha=0.5)
plt.plot(x, y1, color='orange')
plt.title(subset_name+', linear regression', fontsize=20)
plt.title(cg_type+' cg, '+subset_name+' regression (total ms).png')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression.eps', format='eps')


'''df['frac']=df['s2']+df['s3']+df['s4']
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(subset_name+', 2-4: r=', str(r)+', p='+str(p))
df['frac']=df['s2']+df['s3']+df['s4']+df['s5']
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(subset_name+', 2-5: r=', str(r)+', p='+str(p))
df['frac']=df['s2']+df['s3']+df['s4']+df['s5']+df['s6']
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(subset_name+', 2-6: r=', str(r)+', p='+str(p))


df['frac']=df['s1']+df['s2']+df['s3']+df['s4']
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(subset_name+', 1-4: r=', str(r)+', p='+str(p))
df['frac']=df['s1']+df['s2']+df['s3']+df['s4']+df['s5']
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(subset_name+', 1-5: r=', str(r)+', p='+str(p))
df['frac']=df['s1']+df['s2']+df['s3']+df['s4']+df['s5']+df['s6']
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(subset_name+', 1-6: r=', str(r)+', p='+str(p))
'''
'''
df['frac']=df['s2']+df['s3']+df['s4']+df['s5']
outliers['frac']=outliers['s2']+outliers['s3']+outliers['s4']+outliers['s5']

x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['frac'].to_numpy()
'''
'''
y1o=outliers['gt'].to_numpy()
y2o=outliers['frac'].to_numpy()


#regression
x=df['frac']
y=df['gt']
xo=outliers['frac']
yo=outliers['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept

print(subset_name+', 2-5: r=', str(r)+', p='+str(p))


df['frac']=df['s1']+df['s2']+df['s3']+df['s4']+df['s5']+df['s6']+df['s7']
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(subset_name+', 1-7: r=', str(r)+', p='+str(p))

x=df['ms_total']
y=df['gt']
xo=outliers['ms_total']
yo=outliers['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept

print(subset_name+', total: r=', str(r)+', p='+str(p))
'''
'''
#filter outliers statistically

#z-scores give worse results

df['z_scores']=np.abs(stats.zscore(df['frac']))


msk=df['z_scores']>3
idx = df.index[msk]


#quantiles
data=df['frac']
first_quar = np.percentile(data, 20)
third_quar = np.percentile(data, 80)
IQR = third_quar - first_quar

lower_limit = first_quar - 1.5 * IQR
upper_limit = third_quar + 1.5 * IQR

outliers_iqr = stats.iqr(data, nan_policy='omit', axis=0, rng=(25, 75)) * 1.5

msk=(df['frac'] < lower_limit) | (df['frac'] > upper_limit)
idx = df.index[msk]
#outliers = np.where((data < lower_limit) | (data > upper_limit))[0]

outliers=df.loc[idx]
df=df.drop(idx)


x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['frac'].to_numpy()

y1o=outliers['gt'].to_numpy()
y2o=outliers['frac'].to_numpy()


#human vs calculated complexity
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.scatter(y1o,y2o,color='red', linewidth=2, alpha=0.5)
#plt.ylim(0,0.2)
plt.xlabel('human ranking',fontsize=16)
plt.ylabel('multi-scale structural complexity',fontsize=16)
plt.title(subset_name+', complexity (no outliers)',fontsize=20)
plt.legend(loc='lower right')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_complexity_noout.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_complexity_noout.eps', format='eps')



#regression
x=df['frac']
y=df['gt']
xo=outliers['frac']
yo=outliers['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.xlabel('human ranking',fontsize=16)
plt.ylabel('multi-scale structural complexity',fontsize=16)
plt.scatter(xo,yo,color='red', linewidth=2, alpha=0.5)
plt.plot(x, y1, color='orange')
plt.title(subset_name+', linear regression', fontsize=20)
plt.title(cg_type+' cg, '+subset_name+' regression (no outliers).png')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression_noout.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_noout.eps', format='eps')


f = open("mssc_figures/"+cg_type+'_'+subset_name+'_regression_noout.log', "w")
ttt=[slope, intercept, r, p, std_err]
print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
print(*ttt, sep='\t', end='\n', file=f)
f.close()

print(subset_name+', 2-5: r=', str(r)+', p='+str(p))

'''

#regression for centre - 2-6, 1-7
#regression for edges - 0-2+6-8

#write everything into a dataframe and then csv: r and p
#avg and std for complexity scales, redraw images
#entropy of scales

#s0-s9

