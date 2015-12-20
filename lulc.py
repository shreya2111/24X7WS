#from matplotlib import pyplot as plt
#from matplotlib import cm 
#from sobel import sobel
#, draw, plot, show
from itertools import izip as zip, count
import cv2
import numpy as np
import os
import json
import re
import math
import time 


def otsuFilter(color):
	img = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(img,(5,5),0)

	# find normalized_histogram, and its cumulative distribution function
	hist = cv2.calcHist([blur],[0],None,[256],[0,256])
	hist_norm = hist.ravel()/hist.max()
	Q = hist_norm.cumsum()

	bins = np.arange(256)

	fn_min = np.inf
	thresh = -1

	for i in xrange(1,256):
	    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
	    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
	    b1,b2 = np.hsplit(bins,[i]) # weights

	    # finding means and variances
	    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
	    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

	    # calculates the minimization function
	    fn = v1*q1 + v2*q2
	    if fn < fn_min:
		fn_min = fn
		thresh = i

	# find otsu's threshold value with OpenCV function
	ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	images=[img,otsu]
	#plt.imshow(otsu,cmap='gray',interpolation='bicubic')
	for i in range(2):
		plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')		
		#plt.imshow(thresh,cmap='gray',interpolation='bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()

#########################

def contour(img):
	#NOT WORKING
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	#return (ret,thresh)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	    #cv2.THRESH_BINARY,11,2)
	#th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	    #cv2.THRESH_BINARY,11,2)

	for h,cnt in enumerate(contours):
	    mask = np.zeros(gray.shape,np.uint8)
	    s=cv2.drawContours(mask,[cnt],0,255,-1)
	    mean = cv2.mean(img,mask = mask)
	
	for i in len(contours):
		plt.imshow(contours[i],'gray')		
		#plt.imshow(thresh,cmap='gray',interpolation='bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()
	images=[gray,thresh,contours]

	#for i in range(3):
	#	plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')		
		#plt.imshow(thresh,cmap='gray',interpolation='bicubic')
	#	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	#plt.show()

def centralPixels(rng_row,rng_col,radius):
	add=[]	
	radCP=[]

	for i in range(1,rng_row-3):
		for j in range(1,rng_col-3):
			
			maxRad=radius[i][j]
			centralPixel=[i,j]
			#case 1
			if i-1<0 and j-1<0:
				#print "case 1"
				#compare i,j to {i+1,i+2},{j+1,j+2}
				for a in [i,i+1,i+2]:
					for b in [j,j+1,j+2]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	
				

			#case 2
			elif i-1<0 and j-1>=0:
				#print "case 2"
				# i+1,i+2,j-1,j+1
				for a in [i,i+1,i+2]:
					for b in [j-1,j,j+1]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	
				
			#case 3
			elif i-1<0 and j+1==col:
				#print "case 3"
				# i+1,i+2,j-2,j-1
				for a in [i,i+1,i+2]:
					for b in [j-2,j-1,j]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	
			
			#case 4
			elif i-1>=0 and j-1<0:
				#print "case 4"
				# i-1,i+1,j+2,j+1
				for a in [i-1,i,i+1]:
					for b in [j,j+1,j+2]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	
			

			#case 5
			elif i-1>=0 and j-1>=0:
				#print "case 5"
				# i+1,i-1,j-1,j+1
				for a in [i-1,i,i+1]:
					for b in [j-1,j,j+1]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	
				
			#case 6
			elif i-1>=0 and j+1==col:
				#print "case 6"
				# i+1,i-1,j-2,j-1
				for a in [i-1,i,i+1]:
					for b in [j-2,j-1,j]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	
		
			#case 7
			elif i+1==row and j-1<0:
				#print "case 7"
				# i-1,i-2,j+2,j+1
				for a in [i-2,i-1,i]:
					for b in [j,j+1,j+2]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	

			#case 8	
			elif i+1==row and j-1>=0:
				#print "case 8"	
				# i-1,i-2,j-1,j-2
				for a in [i-2,i-1,i]:
					for b in [j-1,j,j-2]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	
				
			#case 9	
			elif i+1==row and j+1==col:
				#print "case 9"
				# i-1,i-2,j-1,j-2
				for a in [i-2,i-1,i]:
					for b in [j-2,j-1,j]:
						if radius[a][b]>maxRad:
							maxRad=radius[a][b]
							centralPixel=[a,b]
						else:
							continue	
			#print maxRad, " ",centralPixel	
			#print img[i,j], " i ",i," j ",j

			add.append(centralPixel)
			radCP.append(maxRad)	
	return (add,radCP)						

# def groupPixel(p,rad,row,col):
# 	#p[0]

def algo(img,name):
	edges = cv2.Canny(img,100,200)

	cv2.imshow('edges',edges)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

	#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#plt.subplot(121),plt.imshow(img,cmap = 'gray')
	#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	#plt.show()

	(row,col)=img.shape
	
	rng_row=row
	rng_col=col
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
	new=np.full((rng_row,rng_col),255,dtype=np.uint8)
	for i in add:
		
		new[i[0]][i[1]]=0

	cv2.imshow('dst',new)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

	#finding central pixel groups
	groupCPixel=[]

	for i in add:
		ls=[]
		rad=radius[i[0]][i[1]]
		if i[0]-rad>=0 and i[1]-rad>=0 and i[0]+rad<=rng_row and i[1]+rad<=rng_col:

			for a in xrange(int(i[0]-rad),int(i[0]+rad+1)):
				for b in xrange(int(i[1]-rad),int(i[1]+rad+1)):
					if [a,b] in add:
						ls.append([a,b])
		if ls:
			groupCPixel.append(ls)				

	print groupCPixel,"group pixel"
	
	#print "length ",len(add), "group length ",len(groupCPixel)	
	#print len(groupCPixel[0]),"--------------------",len(groupCPixel[1]),"---------------",len(groupCPixel[35])
	
	#print groupCPixel[0]
	
	for i in groupCPixel:

		#line fitting using group pixel points
		centralPixel=np.array(i)
		#print "centralPixel ",centralPixel
		x=centralPixel[:,0]
		y=centralPixel[:,1]
		#print x

		# calculate polynomial
		z = np.polyfit(x, y, 1)
		f = np.poly1d(z)

		# calculate new x's and y's
		x_new = np.linspace(x[0], x[-1], 50)
		y_new = f(x_new)

		print x_new, y_new
	# new=np.full((rng_row,rng_col),255,dtype=np.uint8)	
	# for i in add:
	# 	new[i[0]][i[1]]=0

	# cv2.imshow('dst',new)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()	

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




#def sobel(img):
#	img = cv2.GaussianBlur(img,(3,3),0)
#	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
#	dst=cv2.Sobel(gray,cv2.CV_16S,1,0,ksize=3)
	#print dst.shape
#	plt.imshow(dst,cmap=None)
#	plt.xticks([]),plt.yticks([])
#	plt.show()

os.chdir('../data')

l=['tmp.png']
#'Datarpur_Punjab.png','Barodi_Haryana.png','Gudhana_Husainka.png']

for i in l:

	img=cv2.imread(i,0)
	(row,col)=img.shape


	#(ret,thresh)=
	#watershed(img)
	#print ret,thresh
	
	#contour(img)
	#otsuFilter(img)
	algo(img,i)


	#sobel(img)

	# Neural net, classes like water resource, vegetation, households, roads, fields. around 50-60 images +plus test data (30). Land cover recognition algorithm.
