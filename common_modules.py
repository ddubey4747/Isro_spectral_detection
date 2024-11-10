'''
;====================================================================================================
;                              X2ABUNDANCE
;
; Package for determining elemental weight percentages from XRF line fluxes
;
; Algorithm developed by P. S. Athiray (Athiray et al. 2015)
; Codes in IDL written by Netra S Pillai
; Codes for XSPEC localmodel developed by Ashish Jacob Sam and Netra S Pillai
;
; Developed at Space Astronomy Group, U.R.Rao Satellite Centre, Indian Space Research Organisation
;
;====================================================================================================

This file contains the common functions/methods and class definitions used in the repository

The major library used for this repository is the open-source xraylib at https://github.com/tschoonj/xraylib which is available to be added as dependacy only via conda (Anaconda/miniconda). Thus it is essential to use conda virtual enviornment to execute the code.

The repository uses the following dependacies
	xraylib:    	installed by running conda install xraylib
	numpy: 		installed by running conda install numpy
	astropy: 	installed by running conda install astropy
'''
from typing import Any
import numpy as np

def n_elements(array)->int:
	if(type(array) is list):
		return array.__len__()
	else:#numpy array
		s = np.size(array)
		return s

def dblarr(*args:int) ->Any:
	return np.zeros(tuple(args))

def total(MyList:list) -> Any:
	total = 0
	if(type(MyList) is list):
		for i in MyList:
			total = total + i
	else:
		for i in np.nditer(MyList):
			total = total + i
	return total

def ChangeEveryElement(function,array:list) -> None:
	for i in range(0,array.__len__()):
		array[i]=function(array[i])

def readcol(filename:str, format:str=None)->tuple:
	if(format==None):
		rowformat=[]
	else: 
		rowformat=format.split(sep=',')
	TupleOfLists=[]
	for i in range(0,rowformat.__len__()):
		rowformat[i] = rowformat[i].capitalize()
		TupleOfLists.append([])
	import glob
	for filenames in glob.glob(filename):
		with open(filenames, 'r') as f:
			for line in f:
				inputstring=line.split(sep=None,maxsplit=-1)
				if(inputstring.__len__()==rowformat.__len__() and inputstring[0]):
					try:
						for i in range(0,inputstring.__len__()):
							if(i==TupleOfLists.__len__()):
								TupleOfLists.append([])
							if(rowformat[i]=='B' or rowformat[i]=='I' or rowformat[i]=='L' or rowformat[i]=='Z'):
								TupleOfLists[i].append(int(inputstring[i]))
							elif(rowformat[i]=='D' or rowformat[i]=='F'):
								TupleOfLists[i].append(float(inputstring[i]))
							elif(rowformat[i]!='X'):
								TupleOfLists[i].append(inputstring[i])
					except:
						continue

	TupleOfNpArray=[]
	for listitem in TupleOfLists:
		TupleOfNpArray.append(np.array(listitem))
	return tuple(TupleOfNpArray)


def SortVectors(TupleOfArrays:tuple, Reverse:bool = False) -> tuple:
	TupleOfLists = []
	for item in TupleOfArrays:
		myList = list(item)
		TupleOfLists.append(myList)
	ListOfTuples = []
	tuplelength = TupleOfLists.__len__()
	length = TupleOfLists[0].__len__()
	for w in TupleOfLists:
		if(w.__len__() != length):
			return ListOfTuples
	for i in range(0,length):
		tupleItem = []
		for k in TupleOfLists:
			tupleItem.append(k[i])
		ListOfTuples.append(tuple(tupleItem)) 
	ListOfTuples.sort(reverse=Reverse)
	for anylist in TupleOfLists:
		anylist.clear()
	for item in ListOfTuples:
		for i in range(0,tuplelength):
			TupleOfLists[i].append(item[i])
	TupleOfNpArray=[]
	for listitem in TupleOfLists:
		TupleOfNpArray.append(np.array(listitem))
	return tuple(TupleOfNpArray)


def totalLambda(function, array:list)->float:
	sum=0.0
	if(type(array) is list):
		for item in array:
			sum=sum+function(item)
	else:
		for item in np.nditer(array):
			sum=sum+function(item)
	return sum

def PRODUCT(array:list)->float:
	p=1
	if(type(array) is list):
		for item in array:
			p=p*item
	else:
		for item in np.nditer(array):
			p=p*item
	return p

def Tuple2String(MyRow:tuple)->str:
	output=""
	for x in MyRow:
		output=output+x.__str__()+" "
	return output

def file_lines(filename:str)->int:
	return sum(1 for _ in open(filename))

def fix(stuff):
	typeof = type(stuff)
	if(typeof is float):
		return int(stuff)
	elif typeof is list:
		intlist = []
		for f in stuff:
			intlist.append(int(f))
		return intlist
	else:#numpy array
		return stuff.astype(int)


def array_indices_custom(NpArray, *location)->list:
	ReturnValue = []
	if (type(location[0]) is list):
		location=list(location[0])
	else:
		location=list(location)
	for CurrentLocation in location:
		ArrayShape = list(np.shape(NpArray))
		prod = PRODUCT(ArrayShape)
		for i in range(0,ArrayShape.__len__()):
			item=ArrayShape[i]
			prod=prod/item
			ArrayShape[i]=int((CurrentLocation//prod)%item)
		ReturnValue.append(np.array(ArrayShape))
	return np.array(ReturnValue)

def strarr(count:int)->list:
	return ['']*count

class Xrf_Lines:
	def __init__(self, Edgeenergy, FluorYield, jumpfactor, RadRate, LineEnergy,Energy_nist, photoncs_nist, totalcs_nist,elename_string)  -> None:
		self.fluoryield = FluorYield
		self.edgeenergy = Edgeenergy
		self.jumpfactor = jumpfactor
		self.radrate = RadRate
		self.lineenergy = LineEnergy
		self.energy_nist=Energy_nist
		self.photoncs_nist = photoncs_nist
		self.totalcs_nist = totalcs_nist
		self.elename_string = elename_string


class Const_Xrf:
	def __init__(self,Musampletotal_echarline,Musampletotal_eincident,Muelementphoto_eincident) -> None:
		self.musampletotal_echarline=Musampletotal_echarline
		self.musampletotal_eincident = Musampletotal_eincident
		self.muelementphoto_eincident=Muelementphoto_eincident

class Xrf_Struc:
	def __init__(self,primary,secondary,total) -> None:
		self.primary_xrf=primary
		self.secondary_xrf=secondary
		self.total_xrf=total

class Scat_struc:
	def __init__(self,Coh,Incoh,Total) -> None:
		self.i_total=Total
		self.i_coh=Coh
		self.i_incoh=Incoh

class Spectrum:
	def __init__(self,energy,counts,xrf,scat_coh,scat_incoh,scat_total,xrf_lines_flux,xrf_lines_energy,pxrf,sxrf) -> None:
		self.energy=energy
		self.counts=counts
		self.xrf=xrf
		self.scat_coh=scat_coh
		self.scat_incoh=scat_incoh
		self.scat_total=scat_total
		self.xrf_lines_flux=xrf_lines_flux
		self.xrf_lines_energy+xrf_lines_energy
		self.pxrf=pxrf
		self.sxrf=sxrf

class Constant_scat:
	def __init__(self,energy_compton,mu_coh,mu_incoh,mu_photo,salpha_lamda) -> None:
		self.energy_compton= energy_compton
		self.mu_coh=mu_coh
		self.mu_incoh=mu_incoh
		self.mu_photo=mu_photo
		self.salpha_lamda=salpha_lamda
		pass