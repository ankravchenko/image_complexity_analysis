import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import sys

cg_type='fft'
subset_name='art'
nat=sys.argv[1] #natural or artificial
#nat='natural'


dset_natural=['interior_design', 'objects', 'scenes']
dset_artificial=['advertisement', 'art', 'infographics', 'suprematism']
colours_natural=['orange', '#4169e1', '#20b2aa']
colours_artificial=[	'#00bfff', "#db7093", '#2f4f4f', '#006400']



if nat == 'natural':
	dset=dset_natural
	colours=colours_natural
else:
	dset=dset_artificial
	colours=colours_artificial

plt.clf()
plt.hlines(0,0,40, color='red', linewidth=1)

#pickle load

cnt=0
for subset_name in dset:
	df = pd.read_csv('calculated_mssc_100/'+cg_type+'_'+subset_name+'_complexity_regression.csv', delimiter='\t')
	y=df['r'].to_numpy()
	x=range(y.size)
	y_max=np.max(y)
	plt.plot(x, y, color=colours[cnt], label=subset_name)
	plt.xlabel('spatial frequencies',fontsize=16)
	plt.ylabel('r-value',fontsize=16)
	plt.legend(loc='upper right')
	cnt=cnt+1

plt.title('correlation to human ranking by spatial scale', fontsize=20)
plt.savefig(nat+'_scale_impact_100.png')


#repeat for broader strides


plt.clf()
plt.hlines(0,0,10, color='red', linewidth=1)

#pickle load

cnt=0
for subset_name in dset:
	df = pd.read_csv('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity_regression.csv', delimiter='\t')
	y=df['r'].to_numpy()
	x=range(y.size)
	y_max=np.max(y)
	plt.plot(x, y, color=colours[cnt], label=subset_name)
	plt.xlabel('spatial frequencies',fontsize=16)
	plt.ylabel('r-value',fontsize=16)
	plt.legend(loc='upper right')
	cnt=cnt+1

plt.title('correlation to human ranking by spatial scale', fontsize=20)
plt.savefig(nat+'_scale_impact_10.png')
	

