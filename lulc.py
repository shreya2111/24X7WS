#from matplotlib import pyplot as plt
#from matplotlib import cm 
#from sobel import sobel
#, draw, plot, show
from itertools import izip as zip, count
from centralPixels import centralPixels 
import cv2
import numpy as np
import os
import json
import re
import math
import time 


def algo(img,name):
	edges = cv2.Canny(img,100,200)

	cv2.imshow('edges',edges)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

	(row,col)=img.shape
	
	rng_row=100
	rng_col=100
	ls=[]
	for i in range(0,row):
		for j in range(0,col):
			if edges[i,j]==255: #finding edge points
				#store i,j				
				ls.append([i,j])
	ls=np.array(ls,dtype=np.int64)
	name=re.sub('.png', '', name)
	# with open(str(name)+'Data.json','w') as f:
	# 	json.dump(ls,f,indent=2)
	print name,ls.shape

	#finding reference circles
	radius=[]
	for i in range(rng_row):
		t=time.clock()
		dia=[]
		for j in range(rng_col):
			
			tmp=np.zeros(ls.shape)
			
			tmp[:,0]=i
			tmp[:,1]=j
			dst=np.amin([round(np.linalg.norm(tmp[k,:]-ls[k,:])) for k in range(len(ls))])

			dia.append(dst) #radius of ref circle

		tmp=None	
		print "time: ",time.clock()-t	
		radius.append(dia)	
	print "radius done"
		
	add,radCP=centralPixels(rng_row,rng_col,radius)	
	print "out of if"		

	#print add,radCP


	#plot central pixels
	# new=np.full((rng_row,rng_col),255,dtype=np.uint8)
	# for i in add:
		
	# 	new[i[0]][i[1]]=0

	# cv2.imshow('dst',new)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()	

	#finding central pixel groups
	groupCPixel=[]

	for i in add:
		t=time.clock()
		ls=[]
		rad=radius[i[0]][i[1]]
		if i[0]-rad>=0 and i[1]-rad>=0 and i[0]+rad<=rng_row and i[1]+rad<=rng_col:

			for a in xrange(int(i[0]-rad),int(i[0]+rad+1)):
				for b in xrange(int(i[1]-rad),int(i[1]+rad+1)):
					if [a,b] in add:
						ls.append([a,b])
		if ls:
			groupCPixel.append(ls)				
		print "time groupCPixel: ",time.clock()-t	
	print len(groupCPixel),"group pixel"
	
	#print "length ",len(add), "group length ",len(groupCPixel)	
	#print len(groupCPixel[0]),"--------------------",len(groupCPixel[1]),"---------------",len(groupCPixel[35])
	
	# for i in groupCPixel:

	# 	#line fitting using group pixel points
	# 	centralPixel=np.array(i)
	# 	#print "centralPixel ",centralPixel
	# 	x=centralPixel[:,0]
	# 	y=centralPixel[:,1]
	# 	#print x

	# 	# calculate polynomial
	# 	z = np.polyfit(x, y, 1)
	# 	f = np.poly1d(z)

	# 	# calculate new x's and y's
	# 	x_new = np.linspace(x[0], x[-1], 50)
	# 	y_new = f(x_new)

	
	#print x_new, y_new


		# plt.plot(x,y,'o', x_new, y_new)
		# plt.xlim([x[0]-1, x[-1] + 1 ])
		# plt.show()

	print "line plotted"	

	#segmentation
	#ratio of total central pixels to the avg radius of those pixels
	#per region
	for i in groupCPixel:
		sum=0
		for j in i:
			
			sum+=radius[j[0]][j[1]]
		avg=sum/len(i)
		
		ratio=len(i)/avg

		print " the ratio ",ratio, " for i "
		print i	

os.chdir('../data')

l=['Barodi_Haryana.png']
#'Datarpur_Punjab.png','Barodi_Haryana.png','Gudhana_Husainka.png']

for i in l:

	img=cv2.imread(i,0)
	(row,col)=img.shape

	algo(img,i)


	# Neural net, classes like water resource, vegetation, households, roads, fields. around 50-60 images +plus test data (30). Land cover recognition algorithm.
