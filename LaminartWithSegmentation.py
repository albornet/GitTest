# ! /usr/bin/env python

# NEST implementation of the LAMINART model of visual cortex.
# Created by Greg Francis (gfrancis@purdue.edu) as part of the Human Brain Project.
# 16 June 2014

# 8 July 2014: modified to use arrays for cells rather than nest.topology

# 15 July 2014: Implementing end-cutting

# 21 August 2014: Realised that end-cutting should reflect the offset nature of the boundary grid. Should be different for up/down and for left/right.

# 24 October 2014: Implementing diagonal orientations

# 03 December 2014: Implementing segmentation signals for surfaces

# 10 December 2014: Implementing segmentation signals for boundaries

# 18 December 2014: Introducing separate segmentation signals for multiple layers

# 3 February 2015: Diagonal grouping works properly. 

# 3 February 2015: All works, but wish did not need tonic input for boundary segmentation process (unable to make work an approach where boundary segmentation signals generate their own blocking signals)

# 5 March 2015: Realised that filling-in of brightness did not work properly and modified the circuit to send excitatory or inhibitory signals depending on the differences between neighbors.

# 6 March 2015: Filling-in works properly now, but adds little to offset discrimination that is not covered (more cleanly) by boundaries. Now measuring offset discrimination from boundaries. 

# 10 March 2015: Goes through multiple conditions with separate output directories.

# 11 March 2015: Runs multiple simulations with random noise added to location of segmentation signal. 

# 12 March 2015: Place location signal target in a good place relative to stimulus.

# 13 March 2015: Re-organising simulation so that create network only one time and then present stimuli for multiple trials

# 18 March 2015: Added reset signal for surface segmentation to prevent carry-over across trials.

# 24 March 2015: added across-orientation inhibition at V1 to sharpen up responses

# 20 April 2015: Noted that across trials the simulation takes increasing long to run. Seems to reset with a new stimulus condition. Maybe that holding all the spike counts in memory leads to memory swapping. Maybe try resetting the network in between trials.

# 20 August 2015: Upgrading to NEST 2.6.0 introduces an incompatibility with existing code, where input devices can only connect with one type of synapse model. Need an intermediate spiking neuron for LGN cells. 

# 08 September 2015: Intermediate spiking LGN cells requires other parameter changes for simple cell connections.

# 23 May 2016 : the model can read any bmp image as a stimulus, and neuron layers are defined into 1-D nest pop arrays (e.g : neuron[k][i][j] (k = ori, i = row, j = column) corresponds to neuron[k*NRows*NColumns + i*NColumns + j)

# Notes:
# nest.Connect(source, target)
# NEST simulation time is in milliseconds

import os, sys
import matplotlib
import random
from datetime import datetime
# startTime = datetime.now()

#matplotlib.use("macosx")
import pylab

import numpy  
from images2gif import writeGif
import scipy.io, scipy.misc, scipy.signal

#matplotlib.use("macosx")
import nest
import nest.voltage_trace

from readAndCropBMP import readAndCropBMP


# Stimuli have to be defined below according to the names present in the directory "Stimuli/"
ConditionNames = ["Test"]
ConditionNames = ["vernier"]
ConditionNames = ["malania short 1", "malania short 2", "malania short 4", "malania short 8", "malania short 16"]
ConditionNames = ["malania equal 1", "malania equal 2", "malania equal 4", "malania equal 8", "malania equal 16"]
ConditionNames = ["malania long 1", "malania long 2", "malania long 4", "malania long 8", "malania long 16"]
ConditionNames = ["boxes", "crosses", "boxes and crosses"]
ConditionNames = ["squares 1", "squares 2", "cuboids", "scrambled cuboids"]
ConditionNames = ["circles 1", "circles 2", "circles 3", "circles 4", "circles 5", "circles 6"]
ConditionNames = ["hexagons 1", "hexagons 2", "hexagons 3", "hexagons 4", "hexagons 5", "hexagons 6", "hexagons 7", "hexagons 8", "hexagons 9", "hexagons 10", "hexagons 11"]
ConditionNames = ["octagons 1", "octagons 2", "octagons 3", "octagons 4", "octagons 5", "octagons 6", "octagons 7", "octagons 8", "octagons 9", "octagons 10", "octagons 11"]
ConditionNames = ["stars 1", "stars 2", "stars 3", "stars 4", "stars 5", "stars 6", "stars 7", "stars 8", "stars 9"]
ConditionNames = ["irreg1 1", "irreg1 2", "irreg1 3", "irreg1 4", "irreg1 5", "irreg1 6", "irreg1 7", "irreg1 8", "irreg1 9", "irreg1 10", ]
ConditionNames = ["irreg2 1", "irreg2 2", "irreg2 3", "irreg2 4", "irreg2 5"]
ConditionNames = ["pattern2 1", "pattern2 2", "pattern2 3", "pattern2 4", "pattern2 5", "pattern2 6", "pattern2 7", "pattern2 8", "pattern2 9", "pattern2 10"]
ConditionNames = ["pattern irregular 1", "pattern irregular 2", "pattern irregular 3", "pattern irregular 4", "pattern irregular 5", "pattern irregular 6", "pattern irregular 7", "pattern irregular 8", "pattern irregular 9", "pattern irregular 10", "pattern irregular 11"]
ConditionNames = ["pattern stars 1", "pattern stars 2", "pattern stars 3", "pattern stars 4", "pattern stars 5", "pattern stars 6", "pattern stars 7", "pattern stars 8", "pattern stars 9", "pattern stars 10", "pattern stars 11", "pattern stars 12", "pattern stars 13", "pattern stars 14"]
ConditionNames = ["HalfLineFlanks1", "HalfLineFlanks2", "HalfLineFlanks3", "HalfLineFlanks5"] # stimuli used in the past

# Actual defining of the condition
ConditionNames = ["hexagons 4", "hexagons 5", "hexagons 6", "hexagons 7", "hexagons 8", "hexagons 9", "hexagons 10", "hexagons 11"]
numTrials=20  # within each condition

for thisConditionName in ConditionNames:

	## Rebuild network for each condition (some vary in image size, and NEST simulations have an upper time limit that might be exceeded)
	nest.ResetKernel()

	synapseCount=0
	nest.SetKernelStatus({'local_num_threads':8})

	# Read the image from Stimuli/, crop it until it contains only the stimulus image, and crop it even more, if wanted for computational purposes
	addCropX = addCropY = 0
	if thisConditionName == "Test":
		numTrials=1
	if thisConditionName == "vernier":
		[addCropX,addCropY] = [50,15]
	if thisConditionName in ["malania short 1", "malania short 2", "malania short 4", "malania short 8", "malania short 16",
							 "malania equal 1", "malania equal 2", "malania equal 4", "malania equal 8", "malania equal 16"]:
		[addCropX,addCropY] = [10,15]
	if thisConditionName in ["malania long 1", "malania long 2", "malania long 4", "malania long 8", "malania long 16", "cuboids"]:
		[addCropX,addCropY] = [10,10]
	if thisConditionName in ["boxes", "crosses", "boxes and crosses"]:
		[addCropX,addCropY] = [20,15]
	if thisConditionName == "scrambled cuboids":
		[addCropX,addCropY] = [15,20]
	if thisConditionName in ["squares 1", "circles 1", "circles 2", "pattern2 1", "pattern2 2", "stars 1", "stars 2", "stars 6",
							 "pattern stars 1", "pattern stars 2", "hexagons 1", "hexagons 2", "hexagons 7", "octagons 1", "octagons 2",
							 "octagons 7", "irreg1 1", "irreg1 2", "irreg2 1", "irreg2 2", "pattern irregular 1", "pattern irregular 2"]:
		[addCropX,addCropY] = [180,75]
	if thisConditionName in ["circles 3", "stars 3"]:
		[addCropX,addCropY] = [110,75]
	if thisConditionName in ["circles 4", "hexagons 3", "hexagons 4", "hexagons 5", "hexagons 6", "hexagons 8", "hexagons 9", "hexagons 10",
							 "hexagons 11", "octagons 3", "octagons 4", "octagons 5", "octagons 6", "octagons 8", "octagons 9", "octagons 10",
							 "octagons 11", "stars 4", "stars 7", "irreg1 5", "irreg1 6", "irreg1 7", "irreg1 9"]:
		[addCropX,addCropY] = [100,75]
	if thisConditionName in ["circles 5", "stars 8", "pattern2 3", "pattern2 4", "irreg2 3"]:
		[addCropX,addCropY] = [90,75]
	if thisConditionName in ["circles 6", "stars 5", "stars 9", "pattern irregular 3", "pattern irregular 5"]:
		[addCropX,addCropY] = [80,75]
	if thisConditionName == "irreg2 4":
		[addCropX,addCropY] = [70,75]
	if thisConditionName == "irreg2 5":
		[addCropX,addCropY] = [55,75]
	if thisConditionName in ["irreg1 8"]:
		[addCropX,addCropY] = [130,75]
	if thisConditionName == "squares 2":
		[addCropX,addCropY] = [160,75]
	if thisConditionName in ["irreg1 3"]:
		[addCropX,addCropY] = [120,75]
	if thisConditionName in ["irreg1 4", "irreg1 10"]:
		[addCropX,addCropY] = [110,75]
	if thisConditionName == "pattern stars 12":
		[addCropX,addCropY] = [180,50]
	if thisConditionName == "pattern stars 13":
		[addCropX,addCropY] = [130,50]
	if thisConditionName in ["pattern2 5", "pattern2 6", "pattern2 7", "pattern2 8", "pattern2 9", "pattern2 10", "pattern stars 5",
							 "pattern stars 6", "pattern stars 7", "pattern stars 8", "pattern stars 9", "pattern stars 14"]:
		[addCropX,addCropY] = [95,50]
	if thisConditionName in ["pattern irregular 4", "pattern irregular 6", "pattern irregular 7", "pattern irregular 8", "pattern irregular 9",
							 "pattern irregular 10"]:
		[addCropX,addCropY] = [90,40]
	if thisConditionName in ["pattern stars 10", "pattern stars 11"]:
		[addCropX,addCropY] = [95,5]
	if thisConditionName in ["pattern irregular 11"]:
		[addCropX,addCropY] = [92,5]
	pixels, ImageNumPixelRows, ImageNumPixelColumns = readAndCropBMP("Stimuli/"+thisConditionName+".bmp", onlyZerosAndOnes=2, addCropx=addCropX, addCropy=addCropY)

	# Boundary coordinates (boundaries exist at positions between retinotopic coordinates, so add extra pixel on each side to insure a boundary could exists for retinal pixel)
	numPixelRows=ImageNumPixelRows+2  # height
	numPixelColumns=ImageNumPixelColumns+2 # width

	print numPixelRows, numPixelColumns
	
	#=========================
	numSegmentationLayers=3  # one of these is the baseline layer (commonly use 3) minimum is 1
	UseSurfaceSegmentation=0
	UseBoundarySegmentation=1
	if numSegmentationLayers==1:
		UseSurfaceSegmentation=0
		UseBoundarySegmentation=0
	if thisConditionName == "Test":
		UseSurfaceSegmentation=0
		UseBoundarySegmentation=0		
	if thisConditionName == "FlankingRectsWithSurfaceSegmentation" or thisConditionName == "FlankingRectsWithXs":
		UseSurfaceSegmentation=1
		UseBoundarySegmentation=0	
	if numSegmentationLayers==1:
		UseSurfaceSegmentation=0
		UseBoundarySegmentation=0			
			
	#=========================		
	sys.stdout.write('\nSetting up orientation filters...\n')

	# Set up orientation kernels	
	numOrientations = 4
	size = 4 # better as an even number
	sigma2= 0.75
	Olambda = 4
	midSize = (size-1.)/2.

	# Orientations
	# For numOrientations=2, [horizontal, vertical]
	# For numOrientations=4 [ /, |, \, - ]

	# For two orientations
	if numOrientations ==2:
		OppositeOrientationIndex = [1, 0]  
	if numOrientations == 4:
		OppositeOrientationIndex = [2, 3, 0, 1]

	# Two filters for different polarities
	filters1 = numpy.zeros( (numOrientations,size,size) )
	filters2 = numpy.zeros( (numOrientations,size,size) )

	maxValue = -1
	for k in range(0, numOrientations):  
		theta = numpy.pi*(k+1)/numOrientations
		# Make gabor filter
		for i in range(0, size):
			for j in range (0, size):			
				x = (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
				y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
				filters1[k][i][j] = numpy.exp(- (x*x +y*y)/(2*sigma2)) * numpy.sin(2*numpy.pi*x/Olambda)	
				filters2[k][i][j] = -filters1[k][i][j]
		
	# Rescale filters so max value is 1.0
	for k in range(0, numOrientations):
		maxValue = numpy.amax(numpy.abs(filters1[k]))
		filters1[k] /= maxValue
		filters2[k] /= maxValue
		filters1[k][numpy.abs(filters1[k]) < 0.3] = 0.0
		filters2[k][numpy.abs(filters2[k]) < 0.3] = 0.0

	# Convert filters into synapses (referred to by a name (string) )
	OfilterSynapses1= []
	OfilterSynapses2= []
	OfilterWeight = 400  # originally 200, 370 too big, 300 too small, 310 too big
	for i in range(0, size):  # Rows
		OfilterSynapses1.append([]) # define a new row
		OfilterSynapses2.append([]) # define a new row
		for j in range(0, size):  # Columns
			OfilterSynapses1[i].append([]) # define a new orientation
			OfilterSynapses2[i].append([]) # define a new orientation
			for k in range(0, numOrientations):  # Columns
				fname = 'Polarity1Orientation'+str(k)+'-Position'+str(i)+'-'+str(j)
				OfilterSynapses1[i][j].append(fname)
				nest.CopyModel("static_synapse", fname, {"weight": OfilterWeight*filters1[k][i][j]})	  # Weighted by OfilterWeight
				fname = 'Polarity2Orientation'+str(k)+'-Position'+str(i)+'-'+str(j)
				OfilterSynapses2[i][j].append(fname)
				nest.CopyModel("static_synapse", fname, {"weight": OfilterWeight*filters2[k][i][j]})
		
	#==================================
	# Set up V1 layer23 pooling filters
	# Set up orientation kernels

	V1PoolSize =3 # better as an odd number
	sigma2=4.0
	Olambda = 5
	midSize = (V1PoolSize-1.)/2.

	V1poolingfilters = numpy.zeros( (numOrientations,V1PoolSize,V1PoolSize) )
	for k in range(0, numOrientations):  
		theta = numpy.pi*(k+1)/numOrientations
		# Make filter
		for i in range(0, V1PoolSize):
			for j in range (0, V1PoolSize):			
				x = (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
				y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)		
				V1poolingfilters[k][i][j] = numpy.exp(-(x*x+y*y)/(2*sigma2))*numpy.power(numpy.cos(2*numpy.pi*x/Olambda), 5)
				
	# Rescale filters so max value is 1.0
	maxValue = numpy.amax(numpy.abs(V1poolingfilters))
	V1poolingfilters /= maxValue
	V1poolingfilters[V1poolingfilters < 0.1] = 0.0
	V1poolingfilters[V1poolingfilters >= 0.1] = 1.0

	#==================================
	# Set up V1 layer23 pooling cell connections (connect to points at either extreme of pooling line)
	# Set up orientation kernels	

	# Two filters for different directions
	V1poolingconnections1 = numpy.zeros( (numOrientations,V1PoolSize,V1PoolSize) )
	V1poolingconnections2 = numpy.zeros( (numOrientations,V1PoolSize,V1PoolSize) )

	for k in range(0, numOrientations):  
		theta = numpy.pi*(k+1)/numOrientations

		# Make filter
		for i in range(0, V1PoolSize):
			for j in range (0, V1PoolSize):			
				x = (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
				y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
				V1poolingconnections1[k][i][j] = numpy.exp(- (x*x +y*y)/(2*sigma2))	* numpy.power(numpy.cos(2*numpy.pi*x/Olambda), 5)	
				V1poolingconnections2[k][i][j] = V1poolingconnections1[k][i][j]
				
		# Rescale filters so max value is 1.0
		maxValue = numpy.amax(numpy.abs(V1poolingconnections1[k]))
		for i in range(0, V1PoolSize):
			for j in range (0, V1PoolSize):			
				V1poolingconnections1[k][i][j] /= maxValue	
				V1poolingconnections2[k][i][j] /= maxValue	
				if V1poolingconnections1[k][i][j] <0.1:
					V1poolingconnections1[k][i][j]= 0.0
					V1poolingconnections2[k][i][j]= 0.0
				else:
					V1poolingconnections1[k][i][j]= 1.0 * (i+2)*(j+1)	# allows for discrimination of different ends
					V1poolingconnections2[k][i][j]= 1.0 * (V1PoolSize-i+2)*(V1PoolSize-j+1)	# allows for discrimination of different ends

		# Want only the end points of each filter line (remove all interior points)
		for i in range(1, V1PoolSize-1):
			for j in range (1, V1PoolSize-1):			
				V1poolingconnections1[k][i][j] = 0.0
				V1poolingconnections2[k][i][j] = 0.0

		maxValue1 = numpy.amax(numpy.abs(V1poolingconnections1[k]))
		maxValue2 = numpy.amax(numpy.abs(V1poolingconnections2[k]))
		for i in range(0, V1PoolSize):
			for j in range(0, V1PoolSize):
				if V1poolingconnections1[k][i][j] == maxValue1:
					V1poolingconnections1[k][i][j] = 1.0
				else:
					V1poolingconnections1[k][i][j] = 0.0

				if V1poolingconnections2[k][i][j] == maxValue2:
					V1poolingconnections2[k][i][j] = 1.0
				else:
					V1poolingconnections2[k][i][j] = 0.0

	#==================================
	# Set up V2 layer23 pooling filters
	# Set up orientation kernels

	V2PoolSize =7 # better as an odd number  7 is original
	sigma2=26.0  # 4
	Olambda = 9
	midSize = (V2PoolSize-1.)/2.

	V2poolingfilters = numpy.zeros( (numOrientations,V2PoolSize,V2PoolSize) )
	for k in range(0, numOrientations):  
		theta = numpy.pi*(k+1)/numOrientations

		# Make filter
		for i in range(0, V2PoolSize):
			for j in range (0, V2PoolSize):			
				x = (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
				y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
				V2poolingfilters[k][i][j] = numpy.exp(- (x*x +y*y)/(2*sigma2))	* numpy.power(numpy.cos(2*numpy.pi*x/Olambda), sigma2+1)
				
	# rescale filters so max value is 1.0
	maxValue = numpy.amax(numpy.abs(V2poolingfilters))
	V2poolingfilters /= maxValue
	V2poolingfilters[V2poolingfilters < 0.1] = 0.0
	V2poolingfilters[V2poolingfilters >= 0.1] = 1.0

	#==================================
	# Set up V2 layer23 pooling cell connections (connect to points at either extreme of pooling line)
	# Set up orientation kernels	

	# Two filters for different directions
	V2poolingconnections1 = numpy.zeros( (numOrientations,V2PoolSize,V2PoolSize) )
	V2poolingconnections2 = numpy.zeros( (numOrientations,V2PoolSize,V2PoolSize) )

	for k in range(0, numOrientations):
		theta = numpy.pi*(k+1)/numOrientations

		# Make filter
		for i in range(0, V2PoolSize):
			for j in range (0, V2PoolSize):			
				x = (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
				y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
				V2poolingconnections1[k][i][j] = numpy.exp(-(x*x +y*y)/(2*sigma2))*numpy.power(numpy.cos(2*numpy.pi*x/Olambda), sigma2+1)
				V2poolingconnections2[k][i][j] = V2poolingconnections1[k][i][j]
				
		# rescale filters so max value is 1.0
		maxValue = numpy.amax(numpy.abs(V1poolingconnections1[k]))
		for i in range(0, V2PoolSize):
			for j in range (0, V2PoolSize):			
				V2poolingconnections1[k][i][j] /= maxValue	
				V2poolingconnections2[k][i][j] /= maxValue	
				if V2poolingconnections1[k][i][j] <0.1:
					V2poolingconnections1[k][i][j]= 0.0
					V2poolingconnections2[k][i][j]= 0.0
				else:
					V2poolingconnections1[k][i][j]= 1.0 * (i+2)*(j+1)	# allows for discrimination of different ends
					V2poolingconnections2[k][i][j]= 1.0 * (V2PoolSize-i+2)*(V2PoolSize-j+1)	# allows for discrimination of different ends

		# want only the end points of each filter line (remove all interior points)
		for i in range(1, V2PoolSize-1):
			for j in range (1, V2PoolSize-1):			
				V2poolingconnections1[k][i][j] =0.0	
				V2poolingconnections2[k][i][j] =0.0	

		maxValue1 = numpy.amax(numpy.abs(V2poolingconnections1[k]))
		maxValue2 = numpy.amax(numpy.abs(V2poolingconnections2[k]))
		for i in range(0, V2PoolSize):
			for j in range (0, V2PoolSize):			
				if V2poolingconnections1[k][i][j] ==maxValue1:
					V2poolingconnections1[k][i][j]= 1.0
				else:
					V2poolingconnections1[k][i][j]= 0.0
	
				if V2poolingconnections2[k][i][j] ==maxValue2:
					V2poolingconnections2[k][i][j]= 1.0
				else:
					V2poolingconnections2[k][i][j]= 0.0

	# Switch V2 pooling and V2pooling connections for diagonals (weird bit for defining horizontal and vertical directions)
	if numOrientations==4 and 1==0 :  # not applied right now
		for i in range(0, V2PoolSize):
			for j in range (0, V2PoolSize):		
				temp = V2poolingfilters[0][i][j]
				V2poolingfilters[0][i][j] = V2poolingfilters[2][i][j]
				V2poolingfilters[2][i][j] = temp

				temp1 = V2poolingconnections1[0][i][j]
				temp2 = V2poolingconnections2[0][i][j]
				V2poolingconnections1[0][i][j] = V2poolingconnections1[2][i][j]
				V2poolingconnections2[0][i][j] = V2poolingconnections2[2][i][j]
				V2poolingconnections1[2][i][j] = temp1
				V2poolingconnections2[2][i][j] = temp2

	#================================
	# Set up filters for filling-in stage (spreads in various directions). Interneurons receive inhibitory input from all but boundary orientation that matches flow direction.
	# Up, Right (Down and Left are defined implicitly by these)

	# (brightness/darkness) right and down
	numFlows = 2
	flowFilter = [ [1,0], [0,1]]  #  down, right

	# For numOrientations=2, [horizontal, vertical]
	# For numOrientations=4 [ /, -, \, | ]

	# Specify flow orientation (all others block) and position of blockers
	# Different subarrays are for different flow directions
	# (1 added to each offset position because boundary grid is (1,1) offset from brightness grid)
	if numOrientations==2:
		BoundaryBlockFilter = [ [[0, 1, 1], [0, 1, 0]], [[1, 1, 1], [1, 0, 1]] ]
	if numOrientations==4:
		BoundaryBlockFilter = [ [[1, 1, 1], [1, 1, 0]], [[3, 1, 1], [3, 0, 1]] ]

	#=======================================
	# LGN
	sys.stdout.write('LGN,...')
	sys.stdout.flush()

	# LGNbright layer set up as a dc_generator
	LGNbrightInput = nest.Create("dc_generator", ImageNumPixelRows*ImageNumPixelColumns)
	LGNdarkInput = nest.Create("dc_generator", ImageNumPixelRows*ImageNumPixelColumns)

	# Neural LGN cells receive input values from LGNbrightInputs
	LGNbright = nest.Create("iaf_neuron", ImageNumPixelRows*ImageNumPixelColumns)
	LGNdark = nest.Create("iaf_neuron", ImageNumPixelRows*ImageNumPixelColumns)

	# Area V1
	sys.stdout.write('V1,...')
	sys.stdout.flush()

	# Simple oriented (H or V)
	layer4P1 = nest.Create("iaf_neuron", numOrientations*numPixelRows*numPixelColumns)
	layer4P2 = nest.Create("iaf_neuron", numOrientations*numPixelRows*numPixelColumns)
	layer6P1 = nest.Create("iaf_neuron", numOrientations*numPixelRows*numPixelColumns)
	layer6P2 = nest.Create("iaf_neuron", numOrientations*numPixelRows*numPixelColumns)

	# Complex cells
	layer23 = nest.Create("iaf_neuron", numOrientations*numPixelRows*numPixelColumns)
	layer23Pool = nest.Create("iaf_neuron", numOrientations*numPixelRows*numPixelColumns)
	layer23Inter1 = nest.Create("iaf_neuron", numOrientations*numPixelRows*numPixelColumns)
	layer23Inter2 = nest.Create("iaf_neuron", numOrientations*numPixelRows*numPixelColumns)

	###### All subsequent areas have multiple segmentation representations

	# Area V2
	sys.stdout.write('V2,...')
	sys.stdout.flush()

	V2layer4 = nest.Create("iaf_neuron", numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
	V2layer6 = nest.Create("iaf_neuron", numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
	V2layer23 = nest.Create("iaf_neuron", numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
	V2layer23Pool = nest.Create("iaf_neuron", numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
	V2layer23Inter1 = nest.Create("iaf_neuron", numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
	V2layer23Inter2 = nest.Create("iaf_neuron", numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)

	# Area V4
	sys.stdout.write('V4,...')
	sys.stdout.flush()

	V4brightness = nest.Create("iaf_neuron", numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns)
	V4InterBrightness1 = nest.Create("iaf_neuron", numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
	V4InterBrightness2 = nest.Create("iaf_neuron", numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
	V4darkness = nest.Create("iaf_neuron", numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns)
	V4InterDarkness1 = nest.Create("iaf_neuron", numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
	V4InterDarkness2 = nest.Create("iaf_neuron", numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns)

	if numSegmentationLayers>1:
		if UseSurfaceSegmentation==1:
			# Surface Segmentation cells
			sys.stdout.write('Surface,...')
			sys.stdout.flush()

			SurfaceSegmentationOn = nest.Create("iaf_neuron", (numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
			SurfaceSegmentationOnInter1 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			SurfaceSegmentationOnInter2 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			SurfaceSegmentationOff = nest.Create("iaf_neuron", (numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
			SurfaceSegmentationOffInter1 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			SurfaceSegmentationOffInter2 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			SurfaceSegmentationOnSignal = nest.Create("dc_generator", (numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
			SurfaceSegmentationOffSignal = nest.Create("dc_generator", (numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)

		if UseBoundarySegmentation==1:
			# Boundary Segmentation cells
			sys.stdout.write('Boundary,...')
			sys.stdout.flush()

			BoundarySegmentationOn = nest.Create("iaf_neuron", (numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
			BoundarySegmentationOnInter1 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			BoundarySegmentationOnInter2 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			BoundarySegmentationOnInter3 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			BoundarySegmentationOff = nest.Create("iaf_neuron", (numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
			BoundarySegmentationOffInter1 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			BoundarySegmentationOffInter2 = nest.Create("iaf_neuron", (numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
			BoundarySegmentationOnSignal = nest.Create("dc_generator", (numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
			BoundarySegmentationOffSignal = nest.Create("dc_generator", (numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)

	# Spike detectors
	sys.stdout.write('Spike Detectors,...')
	sys.stdout.flush()

	V2spikeDetectors = nest.Create("spike_detector", numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
	V4spikeDetectorsBrightness = nest.Create("spike_detector", numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns)
	V4spikeDetectorsDarkness = nest.Create("spike_detector", numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns)

	sys.stdout.write('done. \nDefining cells...')
	sys.stdout.flush()

	# Constant Input for interneuron 3 in the boundary segmentation layers, and Reset Cell to calm everything down between trials
	ConstantInput = nest.Create("dc_generator")
	ResetSegmentationCell = nest.Create("dc_generator")

	sys.stdout.write('done. \nSetting up spike detectors...')
	sys.stdout.flush()


	# =====================================  Neurons layers are defined, now set up connexions between them ====================================

	# ============================== Detectors =============

	# Spike retinotopic detectors
	nest.Connect(V4brightness, V4spikeDetectorsBrightness, 'one_to_one')
	nest.Connect(V4darkness, V4spikeDetectorsDarkness, 'one_to_one')

	# Spike cortical detectors
	nest.Connect(V2layer23, V2spikeDetectors, 'one_to_one')


	weightScale = 1.0
	# ============================== LGN ===================
	nest.CopyModel("static_synapse", "inputToLGN", {"weight": 700.0*weightScale}) #725	# Not just about making it bigger	# 800 too strong, 750 not quite strong enough (or too strong)
	nest.Connect(LGNbrightInput, LGNbright, 'one_to_one', syn_spec='inputToLGN')
	nest.Connect(LGNdarkInput, LGNdark, 'one_to_one', syn_spec='inputToLGN')


	# ============================== V1 ===================
	sys.stdout.write('done. \nSetting up V1, Layers 4 and 6...')
	sys.stdout.flush()

	orientPool=1
	for k in range(0, numOrientations): # Orientations
		for i2 in range(-size/2, size/2):  # Filter rows
			for j2 in range(-size/2, size/2):  # Filter columns
				source = []
				target = []
				source2 = []
				target2 = []
				for i in range(size/2, numPixelRows-size/2):  # Rows
					for j in range(size/2, numPixelColumns-size/2):  # Columns
						if i+i2 >=0 and i+i2<ImageNumPixelRows and j+j2>=0 and j+j2<ImageNumPixelColumns:
							# Dark inputs use reverse polarity filter
							if abs(filters1[k][i2+size/2][j2+size/2]) > 0.1:
								source.append((i+i2)*ImageNumPixelColumns + (j+j2))                   # neuron[i+i2][j+j2] of size ImageNumPixelColumns*Rows
								target.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j) # neuron[k][i][j] of size numOri*numPixelColumns*Rows
							if abs(filters2[k][i2+size/2][j2+size/2]) > 0.1:
								source2.append((i+i2)*ImageNumPixelColumns + (j+j2))
								target2.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j)

				# LGN -> Layer 6 (simple cells) connections (no connections at the edges, to avoid edge-effects)
				nest.Connect([LGNbright[s] for s in source], [layer6P1[t] for t in target], 'one_to_one', syn_spec=OfilterSynapses1[i2+size/2][j2+size/2][k])
				nest.Connect([LGNdark[s] for s in source], [layer6P2[t] for t in target], 'one_to_one', syn_spec=OfilterSynapses1[i2+size/2][j2+size/2][k])
				nest.Connect([LGNbright[s] for s in source2], [layer6P2[t] for t in target2], 'one_to_one', syn_spec=OfilterSynapses2[i2+size/2][j2+size/2][k])
				nest.Connect([LGNdark[s] for s in source2], [layer6P1[t] for t in target2], 'one_to_one', syn_spec=OfilterSynapses2[i2+size/2][j2+size/2][k])
				synapseCount += 2*(len(source)+len(source2))

				# LGN -> Layer 4 (simple cells) connections (no connections at the edges, to avoid edge-effects)
				nest.Connect([LGNbright[s] for s in source], [layer4P1[t] for t in target], 'one_to_one', syn_spec=OfilterSynapses1[i2+size/2][j2+size/2][k])
				nest.Connect([LGNdark[s] for s in source], [layer4P2[t] for t in target], 'one_to_one', syn_spec=OfilterSynapses1[i2+size/2][j2+size/2][k])
				nest.Connect([LGNbright[s] for s in source2], [layer4P2[t] for t in target2], 'one_to_one', syn_spec=OfilterSynapses2[i2+size/2][j2+size/2][k])
				nest.Connect([LGNdark[s] for s in source2], [layer4P1[t] for t in target2], 'one_to_one', syn_spec=OfilterSynapses2[i2+size/2][j2+size/2][k])
				synapseCount += 2*(len(source)+len(source2))

	# Layer 4 (simple cells) connections
	nest.CopyModel("static_synapse", "inhib6to4", {"weight": -1.0 * weightScale})
	nest.CopyModel("static_synapse", "excite6to4", {"weight": 1.0 * weightScale})

	# Excitatory connection from same orientation and polarity 1, input from layer 6
	nest.Connect(layer6P1, layer4P1, 'one_to_one', syn_spec='excite6to4')
	nest.Connect(layer6P2, layer4P2, 'one_to_one', syn_spec='excite6to4')
	synapseCount += (len(layer6P1)+len(layer6P2))

	source = []
	target = []
	for k in range(0, numOrientations): # Orientations
		for i in range(0, numPixelRows):  # Rows
			for j in range(0, numPixelColumns):  # Columns
				for i2 in range(-1,1):
					for j2 in range(-1,1):
						if i2!=0 or j2!=0:
							if i+i2 >=0 and i+i2 <numPixelRows and j+j2>=0 and j+j2<numPixelColumns:
								source.append(k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))
								target.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j)

	# Surround inhibition from layer 6 of same orientation and polarity
	nest.Connect([layer6P1[s] for s in source], [layer4P1[t] for t in target], 'one_to_one', syn_spec="inhib6to4")
	nest.Connect([layer6P2[s] for s in source], [layer4P2[t] for t in target], 'one_to_one', syn_spec="inhib6to4")
	synapseCount += 2*len(source)

	sys.stdout.write('done. \nSetting up V1, Layers 23 and 6 (feedback)...')
	sys.stdout.flush()
				
	# Layer 4 -> Layer23 (complex cell connections)
	nest.CopyModel("static_synapse", "complexExcit", {"weight": 500.0*weightScale})
	nest.CopyModel("static_synapse", "V1Feedback", {"weight": 500.0*weightScale}) # 500
	nest.CopyModel("static_synapse", "V1NegFeedback", {"weight": -1500.0*weightScale}) # -1500
	nest.CopyModel("static_synapse", "complexInhib", {"weight": -500.0*weightScale}) # -500
	nest.CopyModel("static_synapse", "interInhibV1", {"weight": -1500.0*weightScale}) # interneurons to main -1600
	nest.CopyModel("static_synapse", "endCutExcit", {"weight": 1500.0*weightScale}) # 1500
	nest.CopyModel("static_synapse", "crossInhib", {"weight": -1000.0*weightScale}) # - 1000
	nest.CopyModel("static_synapse", "23to6Excite", {"weight": 100.0*weightScale})  # 100

	nest.Connect(layer4P1, layer23, 'one_to_one', syn_spec="complexExcit")
	nest.Connect(layer4P2, layer23, 'one_to_one', syn_spec="complexExcit")
	synapseCount += (len(layer4P1)+len(layer4P2))

	source = []
	target = []
	for k in range(0, numOrientations):  # Orientations
		for i in range(0, numPixelRows):  # Rows
			for j in range(0, numPixelColumns):  # Columns
				for k2 in range(0, numOrientations):  # Orientations
					if k != k2:
						source.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j)
						target.append(k2*numPixelRows*numPixelColumns + i*numPixelColumns + j)

	# Cross-orientation inhibition
	nest.Connect([layer23[s] for s in source], [layer23[t] for t in target], 'one_to_one', syn_spec="crossInhib")
	synapseCount += len(source)

	source = []
	source2 = []
	source3 = []
	source4 = []
	source5 = []
	target = []
	target2 = []
	target3 = []
	target4 = []
	target5 = []
	for k in range(0, numOrientations):  # Orientations
		for i in range(0, numPixelRows):  # Rows
			for j in range(0, numPixelColumns):  # Columns
				for i2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):  # Filter rows (extra +1 to insure get top of odd-numbered filter)
					for j2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):  # Filter columns

						if V1poolingfilters[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
							if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
								source.append(k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))                             # [k][i+i2][j+j2]
								target.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j)                                       # [k][i][j]
								source2.append(OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))  # [opp[k]][i][j]
								target2.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j)                                      # [k][i][j]

						if V1poolingconnections1[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
							if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
								source3.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j)            # [k][i][j]
								target3.append(k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))  # [k][i+i2][j+j2]

						if V1poolingconnections2[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
							if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
								source4.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j)            # [k][i][j]
								target4.append(k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))  # [k][i+i2][j+j2]

				source5.append(OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j)  # [opp[k]][i][j]
				target5.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j)

	# Pooling neurons in Layer 23 (excitation from same orientation, inhibition from orthogonal)
	nest.Connect([layer23[s] for s in source], [layer23Pool[t] for t in target], 'one_to_one', syn_spec="complexExcit")
	nest.Connect([layer23[s] for s in source2], [layer23Pool[t] for t in target2], 'one_to_one', syn_spec="complexInhib")
	synapseCount += (len(source) + len(source2))

	# Pooling neurons back to Layer 23 and to interneurons
	nest.Connect([layer23Pool[s] for s in source3], [layer23[t] for t in target3], 'one_to_one', syn_spec="V1Feedback")
	nest.Connect([layer23Pool[s] for s in source3], [layer23Inter1[t] for t in target3], 'one_to_one', syn_spec="V1Feedback")
	nest.Connect([layer23Pool[s] for s in source3], [layer23Inter2[t] for t in target3], 'one_to_one', syn_spec="V1NegFeedback")
	nest.Connect([layer23Pool[s] for s in source4], [layer23[t] for t in target4], 'one_to_one', syn_spec="V1Feedback")
	nest.Connect([layer23Pool[s] for s in source4], [layer23Inter2[t] for t in target4], 'one_to_one', syn_spec="V1Feedback")
	nest.Connect([layer23Pool[s] for s in source4], [layer23Inter1[t] for t in target4], 'one_to_one', syn_spec="V1NegFeedback")
	synapseCount += 3*(len(source3) + len(source4))

	# Connect interneurons to complex cell and each other
	nest.Connect(layer23Inter1, layer23, 'one_to_one', syn_spec="interInhibV1")
	nest.Connect(layer23Inter2, layer23, 'one_to_one', syn_spec="interInhibV1")
	synapseCount += (len(layer23Inter1) + len(layer23Inter2))

	# End-cutting (excitation from orthogonal interneuron)
	nest.Connect([layer23Inter1[s] for s in source5], [layer23[t] for t in target5], 'one_to_one', syn_spec="endCutExcit")
	nest.Connect([layer23Inter2[s] for s in source5], [layer23[t] for t in target5], 'one_to_one', syn_spec="endCutExcit")
	synapseCount += 2*len(source5)

	# Connect Layer 23 cells to Layer 6 cells (folded feedback)
	nest.Connect(layer23, layer6P1, 'one_to_one', syn_spec="23to6Excite")
	nest.Connect(layer23, layer6P2, 'one_to_one', syn_spec="23to6Excite")
	synapseCount += 2*len(layer23)

	sys.stdout.write('done. \nSetting up V2, Layers 4 and 6...')
	sys.stdout.flush()


	###################### Area V2  #################
	nest.CopyModel("static_synapse", "V2toV1", {"weight": 10000.0*weightScale})		 #10000
	nest.CopyModel("static_synapse", "V2inhibit6to4", {"weight": -20.0*weightScale})		

	inhibRange64=1
	source = []
	source2 = []
	target = []
	target2 = []
	for h in range(0, numSegmentationLayers): # segmentation layers
		for k in range(0, numOrientations): # Orientations
			for i in range(0, numPixelRows):  # Rows
				for j in range(0, numPixelColumns):  # Columns
					source.append(k*numPixelRows*numPixelColumns + i*numPixelColumns + j) # [k][i][j]
					target.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j)

					for i2 in range(-inhibRange64, inhibRange64+1):
						for j2 in range(-inhibRange64, inhibRange64+1):
							if i+i2 >=0 and i+i2 <numPixelRows and i2!=0 and j+j2 >=0 and j+j2 <numPixelColumns and j2!=0:
								source2.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))
								target2.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j)

	# V2 Layers 4 and 6 connections
	nest.Connect([layer23[s] for s in source], [V2layer6[t] for t in target], 'one_to_one', syn_spec="V2toV1")
	nest.Connect([layer23[s] for s in source], [V2layer4[t] for t in target], 'one_to_one', syn_spec="V2toV1")
	nest.Connect(V2layer6, V2layer4, 'one_to_one', syn_spec="excite6to4")
	synapseCount += (2*len(source) + len(V2layer6))

	# Surround inhibition V2 Layer 6 -> 4
	nest.Connect([V2layer6[s] for s in source2], [V2layer4[t] for t in target2], 'one_to_one', syn_spec="V2inhibit6to4")
	synapseCount += len(source2)

	sys.stdout.write('done. \nSetting up V2, Layers 23 and 6 (feedback)...')
	sys.stdout.flush()

	# V2 Layer 4 -> V2 Layer23 (complex cell connections)
	V2poolRange = 6 # ?
	nest.CopyModel("static_synapse", "V2PoolInhib", {"weight": -1000.0 * weightScale})  # -1000	(orthogonal inhibition)
	nest.CopyModel("static_synapse", "V2PoolInhib2", {"weight": -100.0 * weightScale})  # -100	(non-orthogonal inhibition)
	nest.CopyModel("static_synapse", "V2OrientInhib", {"weight": -1200.0 * weightScale})  # -1000
	nest.CopyModel("static_synapse", "V2Feedback", {"weight": 500.0 * weightScale})  # 500
	nest.CopyModel("static_synapse", "V2NegFeedback", {"weight": -800.0 * weightScale})  # -1000	 Pools to interneurons
	nest.CopyModel("static_synapse", "interInhibV2", {"weight": -1500.0 * weightScale})  # -1000 interneurons to complex cell

	source = []
	source2 = []
	source3 = []
	source4 = []
	source5 = []
	source6 = []
	target = []
	target2 = []
	target3 = []
	target4 = []
	target5 = []
	target6 = []
	for h in range(0, numSegmentationLayers):  # segmentation layers
		for k in range(0, numOrientations):  # Orientations
			for i in range(0, numPixelRows):  # Rows
				for j in range(0, numPixelColumns):  # Columns
					source.append(h*numOrientations*numPixelRows*numPixelColumns + OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j)
					target.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j)

					for i2 in range(-V2PoolSize/2+1, V2PoolSize/2+1):  # Filter rows (extra +1 to insure get top of odd-numbered filter)
						for j2 in range(-V2PoolSize/2+1, V2PoolSize/2+1):  # Filter columns

							if V2poolingfilters[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
								if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
									source2.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))
									target2.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j)

									for k2 in range(0, numOrientations):
										if k2 != k:
											if k2 == OppositeOrientationIndex[k]:
												source3.append(h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))
												target3.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j)
											else:
												source4.append(h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))
												target4.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j)

							if V2poolingconnections1[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
								if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
									source5.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j)
									target5.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))

							if V2poolingconnections2[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
								if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
									source6.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j)
									target6.append(h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))


	# Connect layer 2 to layer 23
	nest.Connect(V2layer4, V2layer23, 'one_to_one', syn_spec="complexExcit")
	synapseCount += len(V2layer4)

	# Cross-orientation inhibition
	nest.Connect([V2layer23[s] for s in source], [V2layer23[t] for t in target], 'one_to_one', syn_spec="V2OrientInhib")
	synapseCount += len(source)

	# Pooling neurons in V2Layer 23 (excitation from same orientation, inhibition from orthogonal)
	nest.Connect([V2layer23[s] for s in source2], [V2layer23Pool[t] for t in target2], 'one_to_one', syn_spec="complexExcit")
	synapseCount += len(source2)

	# Inhibition from other orientations
	nest.Connect([V2layer23[s] for s in source3], [V2layer23Pool[t] for t in target3], 'one_to_one', syn_spec="V2PoolInhib")
	nest.Connect([V2layer23[s] for s in source4], [V2layer23Pool[t] for t in target4], 'one_to_one', syn_spec="V2PoolInhib2")
	synapseCount += (len(source3) + len(source4))

	# Pooling neurons back to Layer 23 and to interneurons
	nest.Connect([V2layer23Pool[s] for s in source5], [V2layer23[t] for t in target5], 'one_to_one', syn_spec="V2Feedback")
	nest.Connect([V2layer23Pool[s] for s in source5], [V2layer23Inter1[t] for t in target5], 'one_to_one', syn_spec="V2Feedback")
	nest.Connect([V2layer23Pool[s] for s in source5], [V2layer23Inter2[t] for t in target5], 'one_to_one', syn_spec="V2NegFeedback")
	nest.Connect([V2layer23Pool[s] for s in source6], [V2layer23[t] for t in target6], 'one_to_one', syn_spec="V2Feedback")
	nest.Connect([V2layer23Pool[s] for s in source6], [V2layer23Inter2[t] for t in target6], 'one_to_one', syn_spec="V2Feedback")
	nest.Connect([V2layer23Pool[s] for s in source6], [V2layer23Inter1[t] for t in target6], 'one_to_one', syn_spec="V2NegFeedback")
	synapseCount += (3*len(source5) + 3*len(source6))

	# Connect interneurons to complex cell
	nest.Connect(V2layer23Inter1, V2layer23, 'one_to_one', syn_spec="interInhibV2")
	nest.Connect(V2layer23Inter2, V2layer23, 'one_to_one', syn_spec="interInhibV2")
	synapseCount += (len(V2layer23Inter1) + len(V2layer23Inter2))

	# Connect Layer 23 cells to Layer 6 cells (folded feedback)
	nest.Connect(V2layer23, V2layer6, 'one_to_one', syn_spec="23to6Excite")
	synapseCount += len(V2layer23)

	sys.stdout.write('done. \nSetting up V4...')
	sys.stdout.flush()


	######################## Area V4 filling-in ##################

	# LGNbright -> V4  connections  and V4 <-> Interneurons
	nest.CopyModel("static_synapse", "LGNV4excit", {"weight": 280.0 * weightScale})  # 70
	nest.CopyModel("static_synapse", "V4betweenColorInhib", {"weight": -5000.0 * weightScale})  # -1000
	nest.CopyModel("static_synapse", "brightnessInhib", {"weight": -2000.0})  # 1000 (1000 is too small, 1250 is pretty good, 1750 is
	nest.CopyModel("static_synapse", "brightnessExcite", {"weight": 2000.0})  # 1000 (1000 is too small, 1500 is pretty good, 1750 is too big)
	nest.CopyModel("static_synapse", "boundaryInhib", {"weight": -5000.0})  # -5000
	nest.CopyModel("static_synapse", "V4inhib", {"weight": -200.0})  # -1000

	source = []
	source2 = []
	source3 = []
	source4 = []
	target = []
	target2 = []
	target3 = []
	target4 = []
	for h in range(0, numSegmentationLayers):  # Segmentation layers
		for i in range(0, ImageNumPixelRows):  # Rows
			for j in range(0, ImageNumPixelColumns):  # Columns
				source.append(i*ImageNumPixelColumns + j)
				target.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

				for k in range(0, numFlows):  # Flow directions
					source2.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)
					target2.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

					# set up flow indices
					i2 = flowFilter[k][0]
					j2 = flowFilter[k][1]
					if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
						source3.append(h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2))
						target3.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

					for k2 in range(0, len(BoundaryBlockFilter[k])):
						for k3 in range(0, numOrientations):
							if BoundaryBlockFilter[k][k2][0] != k3:
								i2 = BoundaryBlockFilter[k][k2][1]
								j2 = BoundaryBlockFilter[k][k2][2]
								if i + i2 < numPixelRows and j + j2 < numPixelColumns:
									source4.append(h*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))
									target4.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

	# Brightness and darkness at V4 compete
	nest.Connect(V4darkness, V4brightness, 'one_to_one', syn_spec="V4betweenColorInhib")
	nest.Connect(V4brightness, V4darkness, 'one_to_one', syn_spec="V4betweenColorInhib")
	synapseCount += (len(V4darkness) + len(V4brightness))

	# LGNbright->V4brightness and LGNdark->V4darkness
	nest.Connect([LGNbright[s] for s in source], [V4brightness[t] for t in target], 'one_to_one', syn_spec="LGNV4excit")
	nest.Connect([LGNdark[s] for s in source], [V4darkness[t] for t in target], 'one_to_one', syn_spec="LGNV4excit")
	synapseCount += 2*len(source)

	# V4brightness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2)
	nest.Connect([V4brightness[s] for s in source2], [V4InterBrightness1[t] for t in target2], 'one_to_one', syn_spec="brightnessExcite")
	nest.Connect([V4brightness[s] for s in source2], [V4InterBrightness2[t] for t in target2], 'one_to_one', syn_spec="brightnessInhib")
	nest.Connect([V4InterBrightness1[t] for t in target2], [V4brightness[s] for s in source2], 'one_to_one', syn_spec="brightnessInhib")
	nest.Connect([V4InterBrightness2[t] for t in target2], [V4brightness[s] for s in source2], 'one_to_one', syn_spec="brightnessExcite")
	synapseCount += 4*len(source2)

	# V4darkness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2)
	nest.Connect([V4darkness[s] for s in source2], [V4InterDarkness1[t] for t in target2], 'one_to_one', syn_spec="brightnessExcite")
	nest.Connect([V4darkness[s] for s in source2], [V4InterDarkness2[t] for t in target2], 'one_to_one', syn_spec="brightnessInhib")
	nest.Connect([V4InterDarkness1[t] for t in target2], [V4darkness[s] for s in source2], 'one_to_one', syn_spec="brightnessInhib")
	nest.Connect([V4InterDarkness2[t] for t in target2], [V4darkness[s] for s in source2], 'one_to_one', syn_spec="brightnessExcite")
	synapseCount += 4*len(source2)

	# V4brightness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2)
	nest.Connect([V4brightness[s] for s in source3], [V4InterBrightness2[t] for t in target3], 'one_to_one', syn_spec="brightnessExcite")
	nest.Connect([V4brightness[s] for s in source3], [V4InterBrightness1[t] for t in target3], 'one_to_one', syn_spec="brightnessInhib")
	nest.Connect([V4InterBrightness1[t] for t in target3], [V4brightness[s] for s in source3], 'one_to_one', syn_spec="brightnessExcite")
	nest.Connect([V4InterBrightness2[t] for t in target3], [V4brightness[s] for s in source3], 'one_to_one', syn_spec="brightnessInhib")
	synapseCount += 4*len(source3)

	# V4darkness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2)
	nest.Connect([V4darkness[s] for s in source3], [V4InterDarkness2[t] for t in target3], 'one_to_one', syn_spec="brightnessExcite")
	nest.Connect([V4darkness[s] for s in source3], [V4InterDarkness1[t] for t in target3], 'one_to_one', syn_spec="brightnessInhib")
	nest.Connect([V4InterDarkness1[t] for t in target3], [V4darkness[s] for s in source3], 'one_to_one', syn_spec="brightnessExcite")
	nest.Connect([V4InterDarkness2[t] for t in target3], [V4darkness[s] for s in source3], 'one_to_one', syn_spec="brightnessInhib")
	synapseCount += 4*len(source3)

	# V2layer23 -> V4 Interneurons (all boundaries block except for orientation of flow)
	nest.Connect([V2layer23[s] for s in source4], [V4InterBrightness1[t] for t in target4], 'one_to_one', syn_spec="boundaryInhib")
	nest.Connect([V2layer23[s] for s in source4], [V4InterBrightness2[t] for t in target4], 'one_to_one', syn_spec="boundaryInhib")
	nest.Connect([V2layer23[s] for s in source4], [V4InterDarkness1[t] for t in target4], 'one_to_one', syn_spec="boundaryInhib")
	nest.Connect([V2layer23[s] for s in source4], [V4InterDarkness2[t] for t in target4], 'one_to_one', syn_spec="boundaryInhib")
	synapseCount += 4*len(source4)

	# inhibition between segmentation layers
	if numSegmentationLayers>1:
		nest.CopyModel("static_synapse", "SegmentInhib", {"weight": -5000.0*weightScale})   # -5000 
		nest.CopyModel("static_synapse", "SegmentInhib2", {"weight": -20000.0*weightScale})   # -5000 

		source4 = []
		target4 = []

		for h in range(0, numSegmentationLayers-1):  # Num layers (not including baseline layer)
			for i in range(0, ImageNumPixelRows):  # Rows
				for j in range(0, ImageNumPixelColumns):  # Columns
					for k2 in range(0, numOrientations):
						for h2 in range(h, numSegmentationLayers-1):
							source4.append(h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j)
							target4.append((h2+1)*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j)

		# Boundaries in lower levels inhibit boundaries in higher segmentation levels (lower levels can be inhibited by segmentation signals)
		nest.Connect([V2layer23[s] for s in source4], [V2layer4[t] for t in target4], 'one_to_one', syn_spec="SegmentInhib2")
		synapseCount += len(source4)
										
	########### Surface segmentation network ############

	if numSegmentationLayers>1 and UseSurfaceSegmentation==1:
		sys.stdout.write('done. \nSetting up surface segmentation network...')
		sys.stdout.flush()
		nest.CopyModel("static_synapse", "SegmentExcite", {"weight": 1*weightScale})   # 1
		nest.CopyModel("static_synapse", "BoundaryToSegmentExcite", {"weight": 2000*weightScale})   # 1
		nest.CopyModel("static_synapse", "SegmentInhib2", {"weight": -20000.0*weightScale})   # -5000 
		nest.CopyModel("static_synapse", "brightnessExcite2", {"weight": 1000.0}) # 1000 (1000 is too small, 1500 is pretty good, 1750 is too big)
		nest.CopyModel("static_synapse", "ResetSegmentation", {"weight": -1000.0*weightScale})   # -1000

		source = []
		source2 = []
		source3 = []
		source4 = []
		source5 = []
		target = []
		target2 = []
		target3 = []
		target4 = []
		target5 = []
		for h in range(0, numSegmentationLayers-1):  # Num layers (not including baseline layer)
			for i in range(0, ImageNumPixelRows):  # Rows
				for j in range(0, ImageNumPixelColumns):  # Columns
					for k in range(0, numFlows):  # Flow directions
						source.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)
						target.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

						# set up flow indices
						i2 = flowFilter[k][0]
						j2 = flowFilter[k][1]
						if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
							source2.append(h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2))
							target2.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

						for k2 in range(0, len(BoundaryBlockFilter[k])):
							for k3 in range(0, numOrientations):
								if BoundaryBlockFilter[k][k2][0] != k3:
									i2 = BoundaryBlockFilter[k][k2][1]
									j2 = BoundaryBlockFilter[k][k2][2]
									if i + i2 < numPixelRows and j + j2 < numPixelColumns:
										for h2 in range(0, numSegmentationLayers):  # draw boundaries from all segmentation layers
											source3.append(h2*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))
											target3.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

					for k2 in range(0, numOrientations):
						for h2 in range(h, numSegmentationLayers-1):
							source4.append(h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j)
							target4.append((h2+1)*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j)

						for i2 in range(-2, 4):  # offset by (1,1) to reflect boundary grid is offset from surface grid
							for j2 in range(-2, 4):
								if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
									for h2 in range(0, h+1):
										source5.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)
										target5.append(h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))

		# Input from segmentation signals
		nest.Connect(SurfaceSegmentationOnSignal, SurfaceSegmentationOn, 'one_to_one', syn_spec="SegmentExcite")
		nest.Connect(SurfaceSegmentationOffSignal, SurfaceSegmentationOff, 'one_to_one', syn_spec="SegmentExcite")
		synapseCount += (len(SurfaceSegmentationOnSignal) + len(SurfaceSegmentationOffSignal))

		# Off signals inhibit on signals (they can be separated by boundaries)
		nest.Connect(SurfaceSegmentationOff, SurfaceSegmentationOn, 'one_to_one', syn_spec="SegmentInhib")
		synapseCount += len(SurfaceSegmentationOff)

		# Segmentation signals inhibited by reset signals (between trials)
		nest.Connect(ResetSegmentationCell, SurfaceSegmentationOn, 'all_to_all', syn_spec="ResetSegmentation")
		nest.Connect(ResetSegmentationCell, SurfaceSegmentationOff, 'all_to_all', syn_spec="ResetSegmentation")
		synapseCount += (len(SurfaceSegmentationOn) + len(SurfaceSegmentationOff))

		# SurfaceSegmentationOn/Off <-> Interneurons
		nest.Connect([SurfaceSegmentationOn[s] for s in source], [SurfaceSegmentationOnInter1[t] for t in target],'one_to_one', syn_spec="brightnessExcite2")
		nest.Connect([SurfaceSegmentationOnInter2[t] for t in target], [SurfaceSegmentationOn[s] for s in source],'one_to_one', syn_spec="brightnessExcite2")
		nest.Connect([SurfaceSegmentationOff[s] for s in source], [SurfaceSegmentationOffInter1[t] for t in target],'one_to_one', syn_spec="brightnessExcite2")
		nest.Connect([SurfaceSegmentationOffInter2[t] for t in target], [SurfaceSegmentationOff[s] for s in source],'one_to_one', syn_spec="brightnessExcite2")
		synapseCount += 4*len(source)

		# Mutual inhibition of interneurons
		nest.Connect(SurfaceSegmentationOffInter1, SurfaceSegmentationOffInter2, 'one_to_one', syn_spec="V4inhib")
		nest.Connect(SurfaceSegmentationOffInter2, SurfaceSegmentationOffInter1, 'one_to_one', syn_spec="V4inhib")
		nest.Connect(SurfaceSegmentationOnInter1, SurfaceSegmentationOnInter2, 'one_to_one', syn_spec="V4inhib")
		nest.Connect(SurfaceSegmentationOnInter2, SurfaceSegmentationOnInter1, 'one_to_one', syn_spec="V4inhib")
		synapseCount += (len(SurfaceSegmentationOffInter1) + len(SurfaceSegmentationOffInter2) + len(SurfaceSegmentationOnInter1) + len(SurfaceSegmentationOnInter2))

		# SurfaceSegmentationOn/Off <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2)
		nest.Connect([SurfaceSegmentationOn[s] for s in source2], [SurfaceSegmentationOnInter2[t] for t in target2], 'one_to_one', syn_spec="brightnessExcite2")
		nest.Connect([SurfaceSegmentationOnInter1[t] for t in target2], [SurfaceSegmentationOn[s] for s in source2], 'one_to_one', syn_spec="brightnessExcite2")
		nest.Connect([SurfaceSegmentationOff[s] for s in source2], [SurfaceSegmentationOffInter2[t] for t in target2], 'one_to_one', syn_spec="brightnessExcite2")
		nest.Connect([SurfaceSegmentationOffInter1[t] for t in target2], [SurfaceSegmentationOff[s] for s in source2], 'one_to_one', syn_spec="brightnessExcite2")
		synapseCount += 4*len(source2)

		# V2layer23 -> Segmentation Interneurons (all boundaries block except for orientation of flow)
		nest.Connect([V2layer23[s] for s in source3], [SurfaceSegmentationOnInter1[t] for t in target3], 'one_to_one', syn_spec="boundaryInhib")
		nest.Connect([V2layer23[s] for s in source3], [SurfaceSegmentationOnInter2[t] for t in target3], 'one_to_one', syn_spec="boundaryInhib")
		nest.Connect([V2layer23[s] for s in source3], [SurfaceSegmentationOffInter1[t] for t in target3], 'one_to_one', syn_spec="boundaryInhib")
		nest.Connect([V2layer23[s] for s in source3], [SurfaceSegmentationOffInter2[t] for t in target3], 'one_to_one', syn_spec="boundaryInhib")
		synapseCount += 4*len(source3)

		# Segmentation -> V2layer4 (gating)
		# Boundaries in lower levels inhibit boundaries in higher segmentation levels (lower levels can be inhibited by segmentation signals)
		nest.Connect([V2layer23[s] for s in source4], [V2layer4[t] for t in target4], 'one_to_one', syn_spec="SegmentInhib2")
		synapseCount += len(source4)
		# Segmentations inhibit boundaries at lower levels
		nest.Connect([SurfaceSegmentationOn[s] for s in source5], [V2layer4[t] for t in target5], 'one_to_one', syn_spec="SegmentInhib")
		synapseCount += len(source5)

	########### Boundary segmentation network ###################

	if numSegmentationLayers>1 and UseBoundarySegmentation==1:

		sys.stdout.write('done. \nSetting up boundary segmentation network...')
		sys.stdout.flush()	
	
		nest.CopyModel("static_synapse", "SegmentInhib3", {"weight": -150.0*weightScale})   # -5000 interneuron3 to segmentation activity
		nest.CopyModel("static_synapse", "SegmentInhib4", {"weight": -5000.0*weightScale})   # -5000  boundary to interneuron3
		nest.CopyModel("static_synapse", "SegmentInhib5", {"weight": -20000.0*weightScale})   # -5000 interneuron3 to segmentation activity
		nest.CopyModel("static_synapse", "SegmentExcite1", {"weight": 2000.0*weightScale}) # 1100  segmentation to interneurons 1-2
		nest.CopyModel("static_synapse", "SegmentExcite2", {"weight": 0.5*weightScale})   # 1  segmentation signal to network
		nest.CopyModel("static_synapse", "SegmentExcite4", {"weight": 500.0*weightScale}) # 500  segmentation to interneuron 3

		source = []
		source2 = []
		source3 = []
		source4 = []
		target = []
		target2 = []
		target3 = []
		target4 = []
		for h in range(0, numSegmentationLayers-1):  # Num layers (not including baseline layer)
			for i in range(0, ImageNumPixelRows):  # Rows
				for j in range(0, ImageNumPixelColumns):  # Columns
					for k in range(0, numFlows):  # Flow directions
						source.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)
						target.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

						# set up flow indices
						i2 = flowFilter[k][0]
						j2 = flowFilter[k][1]
						if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
							source2.append(h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2))
							target2.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

						for k2 in range(0, len(BoundaryBlockFilter[k])):
							#	if BoundaryBlockFilter[k][k2][0] != k:
							i2 = BoundaryBlockFilter[k][k2][1]
							j2 = BoundaryBlockFilter[k][k2][2]
							if i+i2 < numPixelRows and j+j2 < numPixelColumns:
								for k3 in range(0, numOrientations):
									for h2 in range(0, numSegmentationLayers):  # draw boundaries from all segmentation layers
										source3.append(h2*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))
										target3.append(h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

					for k2 in range(0, numOrientations):
						for i2 in range(-2, 4):  # offset by (1, 1) to reflect boundary grid is offset from surface grid
							for j2 in range(-2, 4):
								if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
									# segmentations inhibit boundaries at lower levels
									for h2 in range(0, h+1):
										source4.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)
										target4.append(h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2))

		# Input from segmentation signals
		nest.Connect(BoundarySegmentationOnSignal, BoundarySegmentationOn, 'one_to_one', syn_spec="SegmentExcite2")
		nest.Connect(BoundarySegmentationOffSignal, BoundarySegmentationOff, 'one_to_one', syn_spec="SegmentExcite2")
		# off signals inhibit on signals (they can be separated by boundaries)
		#	nest.Connect(BoundarySegmentationOff[h][i][j],BoundarySegmentationOn[h][i][j], syn_spec="SegmentInhib")
		synapseCount += (len(BoundarySegmentationOnSignal) + len(BoundarySegmentationOffSignal))

		# BoundarySegmentationOn<->Interneurons (flow out on interneuron 1 flow in on interneuron 2)
		nest.Connect([BoundarySegmentationOn[s] for s in source], [BoundarySegmentationOnInter1[t] for t in target], 'one_to_one', syn_spec="SegmentExcite1")
		nest.Connect([BoundarySegmentationOnInter2[t] for t in target], [BoundarySegmentationOn[s] for s in source], 'one_to_one', syn_spec="SegmentExcite1")

		# Mutual inhibition of interneurons (may not need this: flow only when Inter3 is inhibited - 19 Dec 2014)
		nest.Connect(BoundarySegmentationOnInter1, BoundarySegmentationOnInter2, 'one_to_one', syn_spec="V4inhib")
		nest.Connect(BoundarySegmentationOnInter2, BoundarySegmentationOnInter1, 'one_to_one', syn_spec="V4inhib")

		# Inhibition from third interneuron (itself inhibited by the presence of a boundary)
		nest.Connect(BoundarySegmentationOnInter3, BoundarySegmentationOnInter1, 'one_to_one', syn_spec="SegmentInhib5")
		nest.Connect(BoundarySegmentationOnInter3, BoundarySegmentationOnInter2, 'one_to_one', syn_spec="SegmentInhib5")
		synapseCount += (2*len(source) + 2*len(BoundarySegmentationOnInter1) + 2*len(BoundarySegmentationOnInter3))

		# Third interneuron receives constant excitation (that can be inhibited)
		nest.Connect(ConstantInput, BoundarySegmentationOnInter3, 'all_to_all', syn_spec="SegmentExcite2")
		synapseCount += len(BoundarySegmentationOnInter3)

		nest.Connect([BoundarySegmentationOn[s] for s in source2], [BoundarySegmentationOnInter2[t] for t in target2], 'one_to_one', syn_spec="SegmentExcite1")
		nest.Connect([BoundarySegmentationOnInter1[t] for t in target2], [BoundarySegmentationOn[s] for s in source2], 'one_to_one', syn_spec="SegmentExcite1")
		synapseCount += 2*len(source2)

		# V2layer23 -> Segmentation Interneurons (all boundaries open flow by inhibiting third interneuron)
		nest.Connect([V2layer23[s] for s in source3], [BoundarySegmentationOnInter3[t] for t in target3], 'one_to_one', syn_spec="SegmentInhib3")
		synapseCount += len(source3)

		# BoundarySegmentation -> V2layer4 (gating)
		nest.Connect([BoundarySegmentationOn[s] for s in source4], [V2layer4[t] for t in target4], 'one_to_one', syn_spec="SegmentInhib")
		synapseCount += len(source4)

	sys.stdout.write('done. \n'+str(synapseCount)+' network connections created.\n')
	sys.stdout.flush()		


	#=====================================  Network is defined, now set up stimuli ====================================


	# setupTime = datetime.now() - startTime
	# startTime = datetime.now()

	trialCounter=0  # goes across all trials for a condition (used to keep track to stimulus times)

	# subtracting this from the current value gives only the most recent spike counts (go across all trials)
	cumplotDensityOrientation=[[[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for h in range(numSegmentationLayers)] for k in range(numOrientations)]
	cumplotDensityBrightness=[[[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)] for h in range(numSegmentationLayers)]
	cumplotDensityDarkness=[[[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)] for h in range(numSegmentationLayers)]
			
	
	thisModelEvidence = numpy.zeros(numTrials)
	
	for trialCount in range(0, numTrials):
	
		print "\nSimulating trial "+str(trialCount)+" for condition "+str(thisConditionName)+"\n"	

		#----- Prep for plotting results -------

		newMax = 0  # for plotting
		newMaxBrightness = 0  # for plotting
		newMaxDarkness = 0  # for plotting
		outImages = []  # to create animated gifs for V2 boundaries of different segmentation layers
		outImagesBrightness = []  # to create animated gifs for V4 brightness of different segmentation layers
		outImagesStimulus = []

		for h in range(0, numSegmentationLayers):
			outImages.append([]) 
			outImagesBrightness.append([]) 
									
		# for plotting segmentation signals
		segmentationBoundarySignals = numpy.zeros((numPixelRows, numPixelColumns))
		segmentationSurfaceSignals = numpy.zeros((ImageNumPixelRows, ImageNumPixelColumns))
		
		############# Defining stimuli and segmentation strategy ###########################
		stimulusTimeSteps = 20.  # 40-- 2 seconds
		istimulusTimeSteps = 20
		timeChecks = range(0, istimulusTimeSteps)
		stepDuration = 50  # milliseconds
		InterTrialInterval = 20.0
		
		thisStimulusStartTime = trialCounter*(stimulusTimeSteps+InterTrialInterval)*stepDuration
		thisStimulusStopTime = thisStimulusStartTime + stimulusTimeSteps*stepDuration
		trialCounter+=1		
				
		numBars = 1
		vbarspace = 0
		vbarlength = 10  # best as an even number
		vbarwidth = 1  # produces size of 2*vbarwidth-1
		spaceFlankLines = 8  # 8
		squareSize = vbarlength + 6
		barshift = 1
		offsetSpace = 6  # 6
		
		startSegmentationSignal = 100  # 100 - milliseconds
		segmentationTargetLocationX = 25  # distance from target vernier (at center)
		segmentationTargetLocationY = 0
		segmentationTargetLocationSD = 8  # 8 standard deviation of where location actually ends up
		segmentSignalSize=20 # even number

		# Quick test
		if thisConditionName == "Test":
			stimulusTimeSteps = 20.
			istimulusTimeSteps = 20
			timeChecks = range(0, istimulusTimeSteps)
		# Vernier only
		if thisConditionName in ["vernier", "circles 1", "patterns2 1", "stars 1", "pattern stars 1", "hexagons 1",
								 "octagons 1", "irreg1 1", "irreg2 1", "pattern irregular 1"]:
			segmentationTargetLocationX = 2.5*segmentSignalSize  # far enough away that we do not care about it = 50
		# Only 1 flanking shape around vernier
		if thisConditionName in ["squares 1", "circles 2", "patterns2 2", "stars 2", "stars 6", "pattern stars 2",
								 "hexagons 2", "hexagons 7", "octagons 2", "octagons 7", "irreg1 2", "irreg2 2", "pattern irregular 2"]:
			segmentationTargetLocationX = 36  # aiming for outer edge of square
		# Flanking shape around vernier + 3 other flanking shapes by it (one or more lines of that flanking pattern)
		if thisConditionName in ["cuboids", "scrambled cuboids", "squares 2", "circles 3", "circles 4", "circles 5", "circles 6",
								 "pattern2 3", "pattern2 4", "pattern2 5", "pattern2 6", "pattern2 7", "pattern2 8", "pattern2 9", "pattern2 10",
								 "stars 3", "stars 4", "stars 5", "stars 7", "stars 8", "stars 9",
								 "pattern stars 3", "pattern stars 4", "pattern stars 5", "pattern stars 6", "pattern stars 7", "pattern stars 8",
								 "pattern stars 9", "pattern stars 10", "pattern stars 11", "pattern stars 13", "pattern stars 14",
								 "hexagons 3", "hexagons 4", "hexagons 5", "hexagons 6", "hexagons 8", "hexagons 9", "hexagons 10", "hexagons 11",
								 "octagons 3", "octagons 4", "octagons 5", "octagons 6", "octagons 8", "octagons 9", "octagons 10", "octagons 11",
								 "irreg1 3", "irreg1 4", "irreg1 5", "irreg1 6", "irreg1 7", "irreg1 8", "irreg1 9", "irreg1 10",
								 "irreg2 3", "irreg2 4", "irreg2 5",
								 "pattern irregular 3", "pattern irregular 4", "pattern irregular 5", "pattern irregular 6", "pattern irregular 7",
								 "pattern irregular 8", "pattern irregular 9", "pattern irregular 10", "pattern irregular 11"]:
			segmentationTargetLocationX = 45 # aiming for outer edge of outer square
		# Same as previous condition, but vertically shaped
		if thisConditionName == "pattern stars 12":
			segmentationTargetLocationX = 0
			segmentationTargetLocationY = 2*squareSize + 1*spaceFlankLines + segmentSignalSize  # aiming for outer edge of outer square ("2 *" originally) = 60
		# Box at both sides of the vernier
		if thisConditionName == "boxes":
			segmentationTargetLocationX = int(numpy.sqrt(2.0) * vbarlength) + spaceFlankLines + segmentSignalSize  # aiming for outer edge of rectangle = 42
		# Cross at both sides of the vernier
		if thisConditionName == "crosses":
			segmentationTargetLocationX = int(numpy.sqrt(2.0) * vbarlength) + segmentSignalSize  # aiming for a bit off center of rectangle (does not make much difference) = 34
		# Box with a cross within at both sides of the vernier
		if thisConditionName == "boxes and crosses":
			segmentationTargetLocationX = int(numpy.sqrt(2.0) * vbarlength) + spaceFlankLines/2 + segmentSignalSize  # aiming so edge of seg signal just covers region next to target = 38
		# One flanking line
		if thisConditionName in ["HalfLineFlanks1", "malania short 1", "malania equal 1", "malania long 1"]:
			segmentationTargetLocationX = spaceFlankLines + segmentSignalSize - 1  # aiming to just touch the flanker = 27
		# Two flanking lines
		if thisConditionName in ["HalfLineFlanks2", "malania short 2", "malania equal 2", "malania long 2"]:
			segmentationTargetLocationX = 2 * spaceFlankLines + segmentSignalSize - 2  # aiming to just touch the outer flanker = 34
		# More than two flanking lines
		if thisConditionName in ["HalfLineFlanks3", "HalfLineFlanks4", "HalfLineFlanks5", "HalfLineFlanks6", "HalfLineFlanks7", "HalfLineFlanks8", "HalfLineFlanks10",
								 "malania short 4", "malania short 8", "malania short 16",
								 "malania equal 4", "malania equal 8", "malania equal 16",
								 "malania long 4", "malania long 8", "malania long 16"]:
			segmentationTargetLocationX = 3 * spaceFlankLines + segmentSignalSize - 2  # aiming to just touch the outer flanker = 42

		print "segmentation target location X : " + str(segmentationTargetLocationX)

		# Pick central location of segmentation signal
		adjustLocationXLeft = int(round(random.gauss(segmentationTargetLocationX, segmentationTargetLocationSD)))  # center of boundary segmentation signal; normal is 10 (bigger numbers mean closer to center of image plane)
		adjustLocationYLeft = int(round(random.gauss(segmentationTargetLocationY, segmentationTargetLocationSD)))  # center of boundary segmentation signal; normal is 0 (bigger numbers mean closer to center of image plane)
		adjustLocationXRight = int(round(random.gauss(segmentationTargetLocationX, segmentationTargetLocationSD)))  # center of boundary segmentation signal; normal is 10 (bigger numbers mean closer to center of image plane)
		adjustLocationYRight = int(round(random.gauss(segmentationTargetLocationY, segmentationTargetLocationSD)))  # center of boundary segmentation signal; normal is 0 (bigger numbers mean closer to center of image plane)

		# ------------- Create templates (if necessary)
		UseTemplate=1
		HTemplate = numpy.zeros((numPixelRows, numPixelColumns))
		VTemplate = numpy.zeros((numPixelRows, numPixelColumns))

		# Offset illusory contours
		# Left
		for i in range(numPixelRows/4, 3*numPixelRows/4):  # Rows
			VTemplate[i][numPixelColumns/4] = 1.0
		# Right
		for i in range(numPixelRows/4, 3*numPixelRows/4):  # Rows
			VTemplate[i][3*numPixelColumns/4-1] = 1.0
		# Top
		for j in range(numPixelColumns/4, 3*numPixelColumns/4):  # Columns
			HTemplate[numPixelRows/4][j] = 1.0
		# Bottom
		for j in range(numPixelColumns/4, 3*numPixelColumns/4):  # Columns
			HTemplate[3*numPixelRows/4-1][j] = 1.0
	
		# V4 Brightness template for a vernier (left or right is for bottom line)
		RightVernierTemplate = numpy.zeros((ImageNumPixelRows, ImageNumPixelColumns))
		LeftVernierTemplate = numpy.zeros((ImageNumPixelRows, ImageNumPixelColumns))
		TemplateSize = 2*vbarlength - 2

		# Create a Vernier filter
		[Vx, Vy] = [2*vbarlength, 2*vbarwidth + barshift] # usually [2*10,3]
		vernierFilter = -numpy.ones([Vx, Vy])
		vernierFilter[0:vbarlength, 0] = 1.0  # takes 0 but not vbarlength
		vernierFilter[vbarlength:2 * vbarlength, 2] = 1.0  # takes vbarlength but not 2*vbarlength

		# Check for the Vernier location on the stimulus
		matchMax = 0
		[vernierLocX,vernierLocY] = [0, 0] # location of the top-left corner of the vernier on the stimulus
		for i in range(ImageNumPixelRows-Vx+1):
			for j in range(ImageNumPixelColumns-Vy+1):
				match = numpy.sum(vernierFilter * pixels[i:i + Vx, j:j + Vy])
				if match > matchMax:
					matchMax = match
					[vernierLocY, vernierLocX] = [i, j] # LocX corresponds to a column and LocY to a row

		# Bottom right
		firstRow = max(vernierLocY + vbarlength, 0)
		lastRow = min(firstRow + TemplateSize, ImageNumPixelRows)
		firstColumn = max(vernierLocX + 2, 0)
		lastColumn = min(firstColumn + TemplateSize, ImageNumPixelColumns)
		RightVernierTemplate[firstRow:lastRow, firstColumn:lastColumn] = 1.0

		# Top left
		lastRow = min(vernierLocY + vbarlength, ImageNumPixelRows)
		firstRow = max(lastRow - TemplateSize, 0)
		lastColumn = min(vernierLocX + 1, ImageNumPixelColumns)
		firstColumn = max(lastColumn - TemplateSize, 0)
		RightVernierTemplate[firstRow:lastRow, firstColumn:lastColumn] = 1.0

		# Bottom left
		firstRow = max(vernierLocY + vbarlength, 0)
		lastRow = min(firstRow + TemplateSize, ImageNumPixelRows)
		lastColumn = min(vernierLocX + 1, ImageNumPixelColumns)
		firstColumn = max(lastColumn - TemplateSize, 0)
		LeftVernierTemplate[firstRow:lastRow, firstColumn:lastColumn] = 1.0

		# Top right
		lastRow = min(vernierLocY + vbarlength, ImageNumPixelRows)
		firstRow = max(lastRow - TemplateSize, 0)
		firstColumn = max(vernierLocX + 2, 0)
		lastColumn = min(firstColumn + TemplateSize, ImageNumPixelColumns)
		LeftVernierTemplate[firstRow:lastRow, firstColumn:lastColumn] = 1.0

		# Different segmentation levels for template values and matchscores
		TemplateMatchScores = [[] for h in range(0, numSegmentationLayers)]
		TemplateRightValues = [[] for h in range(0, numSegmentationLayers)]
		TemplateLeftValues = [[] for h in range(0, numSegmentationLayers)]

		# Set which signals to use for template match
		TemplateUsesFillingInSignals=1
		TemplateUsesBoundarySignals=0

		# Set up stimulus inputs to the network
		for time in timeChecks:

			if time==0:  # draw initial stimulus (gray scale 0-255)
																																										
				sys.stdout.write('\nCreating gif of stimulus.\n')

				# create an gif of the stimulus image, as it first looks (gray scale images take values between 0 and 1)
				plotpixels = numpy.zeros((ImageNumPixelRows, ImageNumPixelColumns))
				for i in range(0, ImageNumPixelRows):  # Rows
					for j in range(0, ImageNumPixelColumns):  # Columns	
						plotpixels[i][j] = pixels[i][j]/254.0
				outImagesStimulus.append(plotpixels)
						
				directory1 = "SimFigures/"+thisConditionName
				if not os.path.exists(directory1):
					os.makedirs(directory1)
				directory = directory1+"/Trial"+str(trialCount)
				if not os.path.exists(directory):
					os.makedirs(directory)
					
				filename = directory+"/firstStimulus.GIF"
				writeGif(filename, outImagesStimulus, duration=1.0)
					
				# Set up LGN pattern by stimulating LGN layer
				for i in range(0, ImageNumPixelRows):  # Rows
					for j in range(0, ImageNumPixelColumns):  # Columns
						light = max((pixels[i][j])/127.0 - 1.0, 0.0)
						dark = max(1.0 - (pixels[i][j]/127.0), 0.0)

						nest.SetStatus([LGNbrightInput[i*ImageNumPixelColumns + j]],[{"amplitude":10.0*light, "start":thisStimulusStartTime, "stop":thisStimulusStopTime}])
						nest.SetStatus([LGNdarkInput[i*ImageNumPixelColumns + j]],[{"amplitude":10.0*dark, "start":thisStimulusStartTime, "stop":thisStimulusStopTime}])

				#---- Define surface segmentation signals
				nest.SetStatus(ConstantInput,[{"amplitude":1000.0, "start":thisStimulusStartTime, "stop":thisStimulusStopTime}])
				nest.SetStatus(ResetSegmentationCell,[{"amplitude":1000.0, "start":thisStimulusStopTime, "stop":(thisStimulusStopTime+InterTrialInterval*stepDuration/2)}])
				if UseSurfaceSegmentation==1:
					# Left side
					segmentLocationX = vernierLocX+vbarwidth+numpy.int((barshift-1)/2) - (adjustLocationXLeft-1) # ImageNumPixelColumns/2 - (adjustLocationXLeft-1)
					segmentLocationY = vernierLocY+vbarlength+numpy.int((vbarspace-1)/2) - adjustLocationYLeft # ImageNumPixelRows/2 - adjustLocationYLeft
			
					sys.stdout.write('\Left center surface segment signal = '+str(segmentLocationX)+", "+str(segmentLocationY))

					target = []
					target2 = []
					for i in range(0, ImageNumPixelRows):  # Rows
						for j in range(0, ImageNumPixelColumns):  # Columns	
							# off signals are at borders of image (will flow in to other locations unless stopped by boundaries)
							if i==0 or i==(ImageNumPixelRows-1) or j==0 or j==(ImageNumPixelColumns-1):
								for h in range(0, numSegmentationLayers-1):  # Segmentation layers (not including baseline)
									target.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

							# inhibit off signals during half of inter-trial interval to allow new surface signals to spread on following trial
					#		for h in range(0, numSegmentationLayers-1):  # Segmentation layers (not including baseline)										
					#			nest.SetStatus(SurfaceSegmentationOffSignal[h][i][j],[{"amplitude":-10.0, "start":thisStimulusStopTime, "stop":(thisStimulusStopTime+InterTrialInterval*stepDuration/2)}])

							distance = numpy.sqrt(numpy.power(segmentLocationX - j, 2.) + numpy.power(segmentLocationY - i, 2.))
							if distance < segmentSignalSize:
								target2.append(0*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j) # [0][i][j]
								segmentationSurfaceSignals[i][j] = 1

					nest.SetStatus([SurfaceSegmentationOffSignal[t] for t in target], [ {"amplitude": 1000.0, "start": thisStimulusStartTime, "stop": thisStimulusStopTime}])
					nest.SetStatus([SurfaceSegmentationOnSignal[t] for t in target2], [{"amplitude": 1000.0, "start": thisStimulusStartTime + startSegmentationSignal, "stop": thisStimulusStopTime}])

					if numSegmentationLayers>2:
						# Right side
						segmentLocationX = vernierLocX+vbarwidth+numpy.int((barshift-1)/2) + adjustLocationXRight # ImageNumPixelColumns/2 + adjustLocationXRight
						segmentLocationY = vernierLocY+vbarlength+numpy.int((vbarspace-1)/2) - adjustLocationYRight # ImageNumPixelRows/2 - adjustLocationYRight

						sys.stdout.write('\Right center surface segment signal = '+str(segmentLocationX)+", "+str(segmentLocationY))

						target = []
						for i in range(0, ImageNumPixelRows):  # Rows
							for j in range(0, ImageNumPixelColumns):  # Columns	
								distance = numpy.sqrt(numpy.power(segmentLocationX - j, 2.) + numpy.power(segmentLocationY - i, 2.) )
								if distance < segmentSignalSize:
									target.append(1*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j) # [1][i][j]
									segmentationSurfaceSignals[i][j] = 1

						nest.SetStatus([SurfaceSegmentationOnSignal[t] for t in target], [{"amplitude": 1000.0, "start": thisStimulusStartTime + startSegmentationSignal, "stop": thisStimulusStopTime}])

				# Define boundary segmentation signals
				if UseBoundarySegmentation==1:
					# left side
					segmentLocationX = vernierLocX+vbarwidth+numpy.int((barshift-1)/2) - (adjustLocationXLeft-1) # ImageNumPixelColumns/2 - (adjustLocationXLeft-1)
					segmentLocationY = vernierLocY+vbarlength+numpy.int((vbarspace-1)/2) - adjustLocationYLeft # ImageNumPixelRows/2 - adjustLocationYLeft
		
					sys.stdout.write('\nLeft center boundary segment signal = '+str(segmentLocationX)+', '+str(segmentLocationY))

					target = []
					for i in range(0, ImageNumPixelRows):  # Rows
						for j in range(0, ImageNumPixelColumns):  # Columns	
							distance = numpy.sqrt(numpy.power(segmentLocationX - j, 2.) + numpy.power(segmentLocationY - i, 2.) )
							if distance < segmentSignalSize:
								target.append(0*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j) # [0][i][j]
								segmentationBoundarySignals[i][j] = 1

					nest.SetStatus([BoundarySegmentationOnSignal[t] for t in target], [{"amplitude": 1000.0, "start": thisStimulusStartTime + startSegmentationSignal, "stop": thisStimulusStopTime}])

					if numSegmentationLayers>2:
						# right side
						segmentLocationX = vernierLocX+vbarwidth+numpy.int((barshift-1)/2) + adjustLocationXRight # ImageNumPixelColumns/2 + adjustLocationXRight
						segmentLocationY = vernierLocY+vbarlength+numpy.int((vbarspace-1)/2) - adjustLocationYRight # ImageNumPixelRows/2 - adjustLocationYRight

						sys.stdout.write('\nRight center boundary segment signal = '+str(segmentLocationX)+', '+str(segmentLocationY))

						target = []
						for i in range(0, ImageNumPixelRows):  # Rows
							for j in range(0, ImageNumPixelColumns):  # Columns	
								distance = numpy.sqrt(numpy.power(segmentLocationX - j, 2.) + numpy.power(segmentLocationY - i, 2.) )
								if distance <segmentSignalSize:
									target.append(1*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)  # [1][i][j]
									segmentationBoundarySignals[i][j] = 1

						nest.SetStatus([BoundarySegmentationOnSignal[t] for t in target], [{"amplitude": 1000.0, "start": thisStimulusStartTime + startSegmentationSignal, "stop": thisStimulusStopTime}])

		#--------Finished defining stimuli --------------------

		#	print 'Simulation turned off!, line 1430'
			nest.Simulate(stepDuration)

			# Store stimulus information to make an animated gif
			plotpixels = numpy.zeros((ImageNumPixelRows, ImageNumPixelColumns))
			if time>0 and time <stimulusTimeSteps:
				plotpixels = numpy.divide(pixels,254.0)
			if time>0:				
				outImagesStimulus.append(plotpixels)

			# Store results for later plotting
			plotDensityOrientation=[[[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for h in range(numSegmentationLayers)] for k in range(numOrientations)]

	
			# Orientations
			maxD=0
			for h in range(0, numSegmentationLayers):
				for i in range(0, numPixelRows):  # Rows
					for j in range(0, numPixelColumns):  # Columns
						for k in range(0, numOrientations): # Orientations
							plotDensityOrientation[k][h][i][j] +=  nest.GetStatus([V2spikeDetectors[h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j]],'n_events')[0] -cumplotDensityOrientation[k][h][i][j] # Plot only spikes from most recent simulation stepDuration
							cumplotDensityOrientation[k][h][i][j] += plotDensityOrientation[k][h][i][j] # update cumulative spikes
							if maxD< plotDensityOrientation[k][h][i][j]:
								maxD = plotDensityOrientation[k][h][i][j]

			# Set up images for boundaries
		#	maxD = max(max(max(max(plotDensityOrientation))))
			if maxD>newMax:
				newMax = maxD	
			for h in range(0, numSegmentationLayers):		
				data = numpy.zeros( (numPixelRows,numPixelColumns,3), dtype=numpy.uint8)
				if maxD>0:		
					for i in range(0, numPixelRows):  # Rows
						for j in range(0, numPixelColumns):  # Columns
							if numOrientations==2:  # vertical and horizontal
								data[i][j] = [plotDensityOrientation[0][h][i][j], plotDensityOrientation[1][h][i][j], 0] 
							if numOrientations==4:  # Vertical, horizontal, either diagonal
								temp = plotDensityOrientation[0][h][i][j]
								if temp <plotDensityOrientation[2][h][i][j] :
									temp = plotDensityOrientation[2][h][i][j]
								data[i][j] = [plotDensityOrientation[1][h][i][j], plotDensityOrientation[3][h][i][j],temp] 

						#	print 'Level '+str(h)+' '+str(data[i][j])+' '+str(maxD)+' '+str(newMax)+'\n'
				outImages[h].append(data)
			
			# Brightness filling-in		
			plotDensityBrightness=[[[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)] for h in range(numSegmentationLayers)]
			plotDensityDarkness=[[[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)] for h in range(numSegmentationLayers)]

			TemplateScore =0
			for h in range(0, numSegmentationLayers):				
				tempR=0
				tempL=0
				for i in range(0, ImageNumPixelRows):  # Rows
					for j in range(0, ImageNumPixelColumns):  # Columns
						plotDensityBrightness[h][i][j] =  nest.GetStatus([V4spikeDetectorsBrightness[h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j]],'n_events')[0] - cumplotDensityBrightness[h][i][j] # Plot only spikes from most recent simulation stepDuration
						cumplotDensityBrightness[h][i][j] += plotDensityBrightness[h][i][j] # update cumulative spikes
						plotDensityDarkness[h][i][j] =  nest.GetStatus([V4spikeDetectorsDarkness[h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j]],'n_events')[0] - cumplotDensityDarkness[h][i][j] # Plot only spikes from most recent simulation stepDuration
						cumplotDensityDarkness[h][i][j] += plotDensityDarkness[h][i][j] # update cumulative spikes
					
						# Template matches
						tempR+= plotDensityBrightness[h][i][j]*RightVernierTemplate[i][j]
						tempL+= plotDensityBrightness[h][i][j]*LeftVernierTemplate[i][j]
				TemplateScore = (tempR - tempL)/(100.0 + tempR + tempL)
			#	TemplateScore = tempL + tempR
				TemplateMatchScores[h].append(TemplateScore)
				TemplateRightValues[h].append(tempR)
				TemplateLeftValues[h].append(tempL)
				print 'Level '+str(h)+", "+str(TemplateScore)+" "+str(sum(sum(LeftVernierTemplate)))+" "+str(sum(sum(RightVernierTemplate)))
			
			# Set up images for brightness
			maxDBrightness = max(max(max(plotDensityBrightness)))
			if maxDBrightness>newMaxBrightness:
				newMaxBrightness = maxDBrightness
			maxDDarkness = max(max(max(plotDensityDarkness)))
			if maxDDarkness>newMaxBrightness:
				newMaxBrightness = maxDDarkness	
		
			for h in range(0, numSegmentationLayers):
				dataBrightness = numpy.zeros( (ImageNumPixelRows,ImageNumPixelColumns,3), dtype=numpy.uint8)

				if maxDBrightness>0 or maxDDarkness>0:		
					for i in range(0, ImageNumPixelRows):  # Rows
						for j in range(0, ImageNumPixelColumns):  # Columns
							dataBrightness[i][j] = [plotDensityBrightness[h][i][j], plotDensityDarkness[h][i][j],0] 
				outImagesBrightness[h].append(dataBrightness)

			print (time+1)*stepDuration
	
		#----- End of time step --------	

		sys.stdout.write('\nCreating animated gifs. NewMax='+str(newMax)+'\n')

		# Create animated gif of stimulus
		filename = directory+"/Stimulus.GIF"
		writeGif(filename, outImagesStimulus, duration=0.2)

		for h in range(0,numSegmentationLayers):
			# create an animated gif of the oriented output
			# rescale firing rates to max value
			newImages = []  # to create an animated gif
			if newMax==0:
				newMax = 1
			for data in outImages[h]:
				data1 = numpy.zeros( (numPixelRows,numPixelColumns,3), dtype=numpy.uint8)
				for i in range(0, numPixelRows):  # Rows
					for j in range(0, numPixelColumns):  # Columns
						for k in range(0,3):
							data1[i][j][k] = (255*data[i][j][k])/newMax		
				newImages.append(data1)	
				
						
			filename = directory+"/V2OrientationsSeg"+str(h)+".GIF"
			writeGif(filename, newImages, duration=0.2)

			# create an animated gif of the output (gray scale images take values between 0 and 1)
			outImagesDarkness = []
			if newMaxBrightness==0:
				newMaxBrightness= 0.5
			for data in outImagesBrightness[h]:
				plotpixels = numpy.zeros((ImageNumPixelRows, ImageNumPixelColumns))
				for i in range(0, ImageNumPixelRows):  # Rows
					for j in range(0, ImageNumPixelColumns):  # Columns	
						plotpixels[i][j] = (newMaxBrightness + data[i][j][0]- data[i][j][1])/(2.0*newMaxBrightness)	
				outImagesDarkness.append(plotpixels)
				
			filename = directory+"/V4BrightnessSeg"+str(h)+".GIF"
			writeGif(filename, outImagesDarkness, duration=0.2)

			# Create animated gif of combination (boundaries and features signals) image
			combinationImage = []
			# add in boundaries
			n=0
			for data in newImages:
				data1 = numpy.zeros( (4*numPixelRows,4*numPixelColumns,3), dtype=numpy.uint8)
				# set background to gray
				for i in range(0, 4*numPixelRows):  # Rows
					for j in range(0, 4*numPixelColumns):  # Columns
						for k in range(0,3):
							data1[i][j][k] = 128

				# superimpose original image
				bright = outImagesStimulus[0]
				for i in range(0, ImageNumPixelRows):  # Rows
					for j in range(0, ImageNumPixelColumns):  # Columns	
						for k in range(0,3):	
							data1[4*(i+1)][4*(j+1)][k] = bright[i][j] * 255		
											
				# superimpose orientation signals
				for i in range(0, numPixelRows):  # Rows
					for j in range(0, numPixelColumns):  # Columns
						if data[i][j][0] >0 or data[i][j][1] >0 or data[i][j][2] >0:
							for k in range(0,3):
								data1[2+4*i][2+4*j][k] = data[i][j][k]

				# draw Template matching filter			
				if UseTemplate==1 and 1==1:		
					for i in range(0, ImageNumPixelRows):  # Rows
						for j in range(0, ImageNumPixelColumns):  # Columns	
							if RightVernierTemplate[i][j] == 1 :  # Draw in pink/purple
								data1[4*(i+1)][4*(j+1)][0] = (255+255*bright[i][j])/2
								data1[4*(i+1)][4*(j+1)][1] = 0
								data1[4*(i+1)][4*(j+1)][2] = (255+255*bright[i][j])/2										
							if LeftVernierTemplate[i][j] == 1 :  #Draw in yellow
								data1[4*(i+1)][4*(j+1)][0] = (255+255*bright[i][j])/2
								data1[4*(i+1)][4*(j+1)][1] = (255+255*bright[i][j])/2
								data1[4*(i+1)][4*(j+1)][2] = 0		

				# draw surface segmentations in yellow (set blue term to zero)
				if UseSurfaceSegmentation==1:		
					for i in range(0, ImageNumPixelRows):  # Rows
						for j in range(0, ImageNumPixelColumns):  # Columns	
							if segmentationSurfaceSignals[i][j]==1:
								data1[4*(i+1)][4*(j+1)][0] = max(data1[4*(i+1)][4*(j+1)][0], 200)
								data1[4*(i+1)][4*(j+1)][1] = max(data1[4*(i+1)][4*(j+1)][1], 200)
								data1[4*(i+1)][4*(j+1)][2] = 0		

				# draw boundary segmentation signals in blue	
				if UseBoundarySegmentation==1:		
					for i in range(0, numPixelRows):  # Rows
						for j in range(0, numPixelColumns):  # Columns
							if segmentationBoundarySignals[i][j] == 1:	
								data1[2+4*i][2+4*j][2] = 255
						
							


									
				combinationImage.append(data1)	
				n = n+1
	
			
			filename = directory+"/ComboSeg"+str(h)+".GIF"
			writeGif(filename, combinationImage, duration=0.2)


			# --- Write template matching results to file for each time step for each segmentation level
			if UseTemplate==1:	
				fname = directory+'/TemplateMatch-Level'+str(h)+'.txt'
				f = open(fname, 'w')
				# Header info
				f.write('numPixelRows='+str(numPixelRows)+'\n')
				f.write('numPixelColumns='+str(numPixelColumns)+'\n')
				f.write('offsetSpace='+str(offsetSpace)+'\n')
				f.write('\n\nTime \t MatchScore \t RightTemplateValues \t LeftTemplateValues\n')

				tempTime = stepDuration
				for t in range(0,len(TemplateMatchScores[h])):
					temp1 = str(tempTime)
					temp2 = str(TemplateMatchScores[h][t])
					temp3 = str(TemplateRightValues[h][t])
					temp4 = str(TemplateLeftValues[h][t])
					f.write( temp1+' \t '+temp2+' \t '+temp3+' \t '+temp4+'\n')
					tempTime += stepDuration
	
				f.close()


		## Intertrial interval
		print "\nSimulating inter-trial interval of "+str(InterTrialInterval*stepDuration)+" milliseconds\n"	

		nest.Simulate(InterTrialInterval*stepDuration)

		# Update cumulative spike counts
		# Orientations
		for h in range(0, numSegmentationLayers):
			for i in range(0, numPixelRows):  # Rows
				for j in range(0, numPixelColumns):  # Columns
					for k in range(0, numOrientations): # Orientations
						plotDensityOrientation[k][h][i][j] +=  nest.GetStatus([V2spikeDetectors[h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j]],'n_events')[0] -cumplotDensityOrientation[k][h][i][j] # Plot only spikes from most recent simulation stepDuration
						cumplotDensityOrientation[k][h][i][j] += plotDensityOrientation[k][h][i][j] # update cumulative spikes
						
		# Brightness signals				
		for h in range(0, numSegmentationLayers):				
			tempR=0
			tempL=0
			for i in range(0, ImageNumPixelRows):  # Rows
				for j in range(0, ImageNumPixelColumns):  # Columns				
					plotDensityBrightness[h][i][j] =  nest.GetStatus([V4spikeDetectorsBrightness[h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j]],'n_events')[0] -cumplotDensityBrightness[h][i][j] # Plot only spikes from most recent simulation stepDuration
					cumplotDensityBrightness[h][i][j] += plotDensityBrightness[h][i][j] # update cumulative spikes
					plotDensityDarkness[h][i][j] =  nest.GetStatus([V4spikeDetectorsDarkness[h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j]],'n_events')[0] -cumplotDensityDarkness[h][i][j] # Plot only spikes from most recent simulation stepDuration
					cumplotDensityDarkness[h][i][j] += plotDensityDarkness[h][i][j] # update cumulative spikes
								

		# --- Compute model evidence (summing across segmentation level and averaging across time)
		if UseTemplate==1:	
			maxScore=-100
			for h2 in range(0, numSegmentationLayers):
				sumX = 0
				for t in range(0,len(TemplateMatchScores[h2])):
					sumX+= TemplateMatchScores[h2][t]
				mean = sumX/len(TemplateMatchScores[h2])
				if mean>maxScore:
					maxScore= mean
					
			thisModelEvidence[trialCount] = maxScore


	#-- Write average model evidence (across trials) to file
	if UseTemplate==1:	
		fname = directory1+'/averageModelEvidence.txt'
		f = open(fname, 'w')
		f.write('Trial \t ModelEvidence\n')
		sumX=0
		sumXX=0
		for trialCount in range(0,numTrials):
			sumX += thisModelEvidence[trialCount]
			sumXX += thisModelEvidence[trialCount]*thisModelEvidence[trialCount]
			f.write( str(trialCount)+' \t '+str(thisModelEvidence[trialCount])+'\n')
		
		sd = numpy.sqrt((sumXX - sumX*sumX/numTrials)/(numTrials-1) )
		f.write( '\n\n\nMean=\t'+str(sumX/numTrials)+'\n')
		f.write( 'sd=\t'+str(sd)+'\n')
		
		f.close()

	# print setupTime
	# print datetime.now() - startTime
					
				
			
