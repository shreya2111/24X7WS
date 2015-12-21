import numpy as np

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