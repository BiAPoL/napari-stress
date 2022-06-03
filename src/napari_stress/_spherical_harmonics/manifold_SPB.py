# From https://github.com/campaslab/STRESS
#!!! WILL DEPEND ON DIFF GEO WHEN WE GO OFF SPHERE !!! #
from numpy  import *
import numpy as np
#import mpmath as mp

# for picking manny inv mats
#import cPickle as pkl # BJG: py2pt7 version
import pickle as pkl
import os, sys

from . import sph_func_SPB as sph_f

MY_DIR  = os.path.realpath(os.path.dirname(__file__)) #current folder
PICKLE_Manny_DIR = os.path.join(MY_DIR, 'Pickled_Manny_Inv_Mat_Files') #subfolder of manny_inv_matrix pickled files


# Use PREVIOUS definitions below to define x,y,z maps for SPB coors:
def Manny_Fn_Def(theta, phi, r_0, Manny_Name, IsRadial):

	x_val = Manny_Fn_Def_X(theta, phi, r_0, Manny_Name, IsRadial)
	y_val = Manny_Fn_Def_Y(theta, phi, r_0, Manny_Name, IsRadial)
	z_val = Manny_Fn_Def_Z(theta, phi, r_0, Manny_Name, IsRadial)

	return x_val, y_val, z_val


# We may want seperate defs for each coordinate:
def Manny_Fn_Def_X(theta, phi, r_0, Manny_Name, IsRadial):

	if(IsRadial == True):
		return Radial_Manifold_R_Def(theta, phi, r_0, Manny_Name)*np.cos(theta)*np.sin(phi)
	else:
		return Non_Radial_Manifold_X_Def(theta, phi, r_0, Manny_Name)

def Manny_Fn_Def_Y(theta, phi, r_0, Manny_Name, IsRadial):

	if(IsRadial == True):
		return Radial_Manifold_R_Def(theta, phi, r_0, Manny_Name)*np.sin(theta)*np.sin(phi)
	else:
		return Non_Radial_Manifold_Y_Def(theta, phi, r_0, Manny_Name)

def Manny_Fn_Def_Z(theta, phi, r_0, Manny_Name, IsRadial):

	if(IsRadial == True):
		return Radial_Manifold_R_Def(theta, phi, r_0, Manny_Name)*np.cos(phi)
	else:
		return Non_Radial_Manifold_Z_Def(theta, phi, r_0, Manny_Name)


# Non-Radial Manifold Name Defs:
def Non_Radial_Manifold_X_Def(theta, phi, r_0, Manny_Name):
	if(Manny_Name == "Cusp_Bowl"):
		return 2.*np.cos(theta)*np.sin(phi)

	elif(Manny_Name == "UFO"):
		return np.cos(theta)*(2.*np.sin(phi) + r_0*np.sin(phi)*np.cos(phi)**3 )
	elif(Manny_Name == "Ellipsoid_r0"):
		return (1. + r_0)*np.cos(theta)*np.sin(phi)
	elif(Manny_Name == "Gen_Pill_nr"):
		return np.cos(theta)*np.sin(phi)
	elif(Manny_Name == "Cone_Head"):
		return np.cos(theta)*np.sin(phi)
	elif(Manny_Name == "Cone_Head5x"):
		return 5.*np.cos(theta)*np.sin(phi)
	elif(Manny_Name  == "Cone_Head_Logis"):
		return np.cos(theta)*np.sin(phi)
	elif(Manny_Name  == "Fission_Yeast_R2" or Manny_Name  == "Fission_Yeast_R1pt2"):

		R = 0.
		if(Manny_Name  == "Fission_Yeast_R2"):
			R = 2. # fix for this geo
		elif(Manny_Name  == "Fission_Yeast_R1pt2"):
			R = 1.2 # fix for this geo
		else:
			print("\n"+"ERROR: Fission Yeast (NR) Geo Not Recognized"+"\n")

		phi_1 = np.pi*(np.pi*R/2.)/(r_0 + np.pi*R) #normalzied arclen of top hem
		phi_2 = np.pi*(r_0 + np.pi*R/2.)/(r_0 + np.pi*R) # normalized arcen up to bottom hemi
		omega = np.pi/(r_0 + np.pi*R) # normalized speed

		if(isscalar(phi)):
			if(phi < phi_1):
				return R*np.sin(phi/(R*omega))*np.cos(theta)
			elif(phi < phi_2):
				return R*np.cos(theta)
			else: # phi > phi_2, < pi
				return R*np.cos((phi - phi_2)/(R*omega))*np.cos(theta)
		else:
			return np.where(phi < phi_1, R*np.sin(phi/(R*omega))*np.cos(theta), np.where(phi < phi_2, R*np.cos(theta), R*np.cos((phi - phi_2)/(R*omega))*np.cos(theta)) )
	else:
		print("\n"+"ERROR: NON-radial Manifold Name: "+str(Manny_Name)+", (X-Coor) Not Recognized"+"\n")

def Non_Radial_Manifold_Y_Def(theta, phi, r_0, Manny_Name):
	if(Manny_Name == "Cusp_Bowl"):
		return 2.*np.sin(theta)*np.sin(phi)

	elif(Manny_Name == "UFO"):
		return np.sin(theta)*(2.*np.sin(phi) + r_0*np.sin(phi)*np.cos(phi)**3 )
	elif(Manny_Name == "Ellipsoid_r0"):
		return (1. + 2.*r_0)*np.sin(theta)*np.sin(phi)
	elif(Manny_Name == "Gen_Pill_nr"):
		return np.sin(theta)*np.sin(phi)
	elif(Manny_Name == "Cone_Head"):
		return np.sin(theta)*np.sin(phi)
	elif(Manny_Name == "Cone_Head5x"):
		return 5.*np.sin(theta)*np.sin(phi)
	elif(Manny_Name  == "Cone_Head_Logis"):
		return np.sin(theta)*np.sin(phi)
	elif(Manny_Name  == "Fission_Yeast_R2" or Manny_Name  == "Fission_Yeast_R1pt2"):

		R = 0.
		if(Manny_Name  == "Fission_Yeast_R2"):
			R = 2. # fix for this geo
		elif(Manny_Name  == "Fission_Yeast_R1pt2"):
			R = 1.2 # fix for this geo
		else:
			print("\n"+"ERROR: Fission Yeast (NR) Geo Not Recognized"+"\n")

		phi_1 = np.pi*(np.pi*R/2.)/(r_0 + np.pi*R) #normalzied arclen of top hem
		phi_2 = np.pi*(r_0 + np.pi*R/2.)/(r_0 + np.pi*R) # normalized arcen up to bottom hemi
		omega = np.pi/(r_0 + np.pi*R) # normalized speed

		if(isscalar(phi)):
			if(phi < phi_1):
				return R*np.sin(phi/(R*omega))*np.sin(theta)
			elif(phi < phi_2):
				return R*np.sin(theta)
			else: # phi > phi_2, < pi
				return R*np.cos((phi - phi_2)/(R*omega))*np.sin(theta)
		else:
			return np.where(phi < phi_1, R*np.sin(phi/(R*omega))*np.sin(theta), np.where(phi < phi_2, R*np.sin(theta), R*np.cos((phi - phi_2)/(R*omega))*np.sin(theta)) )
	else:
		print("\n"+"ERROR: NON-radial Manifold Name: "+str(Manny_Name)+", (Y-Coor) Not Recognized"+"\n")

def Non_Radial_Manifold_Z_Def(theta, phi, r_0, Manny_Name):
	if(Manny_Name == "Cusp_Bowl"):
		return np.cos(phi) - r_0*(np.cos(2.*phi)**2 - 1.)

	elif(Manny_Name == "UFO"):
		return (np.cos(phi) - r_0*np.cos(3.*phi))/(1. - r_0)
	elif(Manny_Name == "Ellipsoid_r0"):
		return (1. + 3.*r_0)*np.cos(phi)
	elif(Manny_Name == "Gen_Pill_nr"):
		return (1. + r_0)*np.cos(phi)
	elif(Manny_Name == "Cone_Head"):
		return np.cos(phi)*(1. + r_0*np.exp(-1.*4.*phi**2))
	elif(Manny_Name == "Cone_Head5x"):
		return 5.*np.cos(phi)*(1. + r_0*np.exp(-1.*4.*phi**2))
	elif(Manny_Name  == "Cone_Head_Logis"):
		return np.cos(phi)*(1. + r_0/(1. + np.exp(-20.*(np.pi/16 - phi))))
	elif(Manny_Name  == "Fission_Yeast_R2" or Manny_Name  == "Fission_Yeast_R1pt2"):

		R = 0.
		if(Manny_Name  == "Fission_Yeast_R2"):
			R = 2. # fix for this geo
		elif(Manny_Name  == "Fission_Yeast_R1pt2"):
			R = 1.2 # fix for this geo
		else:
			print("\n"+"ERROR: Fission Yeast (NR) Geo Not Recognized"+"\n")

		phi_1 = np.pi*(np.pi*R/2.)/(r_0 + np.pi*R) #normalzied arclen of top hem
		phi_2 = np.pi*(r_0 + np.pi*R/2.)/(r_0 + np.pi*R) # normalized arcen up to bottom hemi
		omega = np.pi/(r_0 + np.pi*R) # normalized speed

		if(isscalar(phi)):
			if(phi < phi_1):
				return R*np.cos(phi/(R*omega)) + r_0/2.
			elif(phi < phi_2):
				return r_0/2. - (phi-phi_1)/omega
			else: # phi > phi_2, < pi
				return -1.*r_0/2. - R*np.sin((phi - phi_2)/(R*omega))
		else:
			return np.where(phi < phi_1, R*np.cos(phi/(R*omega)) + r_0/2., np.where(phi < phi_2, r_0/2. - (phi-phi_1)/omega, -1.*r_0/2. - R*np.sin((phi - phi_2)/(R*omega))) )
	else:
		print("\n"+"ERROR: NON-radial Manifold Name: "+str(Manny_Name)+", (Z-Coor) Not Recognized"+"\n")


#!!!! USE TO DETERMINE WITHOUT SYM HARN !!!!#
def Man_Name_Radial(Man_Name):
	IsRadial = True

	if(Man_Name == "Cusp_Bowl" or Man_Name == "UFO" or Man_Name == "Ellipsoid_r0" or Man_Name == "Gen_Pill_nr" or Man_Name == "Cone_Head" or Man_Name == "Cone_Head_Logis" or Man_Name == "Cone_Head5x" or Man_Name == "Fission_Yeast_R2" or Man_Name  == "Fission_Yeast_R1pt2"):
		IsRadial = False
	return IsRadial


# Define RADIAL Manifolds ONCE here:
def Radial_Manifold_R_Def(theta, phi, r_0, Manny_Name):

	if(Manny_Name == "S2"): #Unit Sphere
		return 1.

	elif(Manny_Name == "Pear"):
		return 2. + np.cos(phi)

	elif(Manny_Name == "Pill"): #Ellipsoid: x^2 + y^2 + z^2/4 = 1
		return 2./np.sqrt(1. + 3.*np.sin(phi)**2)

	elif(Manny_Name == "Gen_Pill"):
		return (1. + r_0)/np.sqrt(1. + ((1. + r_0)**2 - 1.)*np.sin(phi)**2) #Ellipsoid: x^2 + y^2 + z^2/((1+r_0)^2) = 1

	elif(Manny_Name == "Oblate_Gen_Pill"):
		return (1. + r_0)/np.sqrt( ((1. + r_0)**2)*(np.cos(phi)**2) + np.sin(phi)**2 ) #Ellipsoid: (x^2 + y^2)/((1+r_0)^2) + z^2 = 1

	elif(Manny_Name == "Gen_S2"):
		return 1. + r_0 #radius: 1+r_0

	elif(Manny_Name  == "Chew_Toy"):
		return 1. + r_0*np.sin(3*phi)*np.cos(theta)

	elif(Manny_Name  == "Little_Five_Points"):
		return 1. + r_0*np.sin(5*phi)*np.cos(theta)

	elif(Manny_Name  == "Dog_Shit"):
		return 1. + r_0*np.sin(7*phi)*np.cos(theta)

	elif(Manny_Name  == "Quint_Spikes"):
		c_shape = .5 # distortion magnitude
		k_shape = 10. #inv. length-scale of distortion decay
		tol_shape = 1.e-3

		X_S2 = np.sin(phi)*np.cos(theta)
		Y_S2 = np.sin(phi)*np.sin(theta)
		Z_S2 = np.cos(phi)*np.ones_like(theta)

		d_0_sq = (X_S2-0.)**2 + (Y_S2 - 0.)**2 + (1. - Z_S2)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_0_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_0_sq )), 1.)

		d_1_sq = (X_S2-1.)**2 + (Y_S2 - 0.)**2 +  (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_1_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_1_sq )), R_shape)

		d_2_sq = (X_S2-0.)**2 + (Y_S2 - 1.)**2 +  (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_2_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_2_sq )), R_shape)

		d_3_sq = (X_S2-0.)**2 + (Y_S2 - 0.)**2 + (-1. - Z_S2)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_3_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_3_sq )), R_shape)

		d_4_sq = (X_S2- -1.)**2 + (Y_S2 - 0.)**2 + (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_4_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_4_sq )), R_shape)

		return R_shape

	elif(Manny_Name  == "Quad_Spikes"):
		# r0 =  distortion magnitude
		k_shape = 10. #inv. length-scale of distortion decay
		tol_shape = 1.e-3

		X_S2 = np.sin(phi)*np.cos(theta)
		Y_S2 = np.sin(phi)*np.sin(theta)
		Z_S2 = np.cos(phi)*np.ones_like(theta)

		d_1_sq = (X_S2 - 1.)**2 + (Y_S2 - 0.)**2 +  (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_1_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_1_sq )), 1.)

		d_2_sq = (X_S2 - 0.)**2 + (Y_S2 - 1.)**2 +  (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_2_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_2_sq )), R_shape)

		d_3_sq = (X_S2 - 0.)**2 + (Y_S2 + 1.)**2 + (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_3_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_3_sq )), R_shape)

		d_4_sq = (X_S2 + 1.)**2 + (Y_S2 - 0.)**2 + (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_4_sq ) > tol_shape,  (1. + r_0*np.exp( -1.*k_shape*d_4_sq )), R_shape)

		return R_shape

	elif(Manny_Name  == "Quad_Spikes_R2"):
		# r0 =  distortion magnitude
		k_shape = 5. #inv. length-scale of distortion decay
		tol_shape = 1.e-3

		R_S2 = 2.
		X_S2 = R_S2*np.sin(phi)*np.cos(theta)
		Y_S2 = R_S2*np.sin(phi)*np.sin(theta)
		Z_S2 = R_S2*np.cos(phi)*np.ones_like(theta)

		d_1_sq = (X_S2 - R_S2)**2 + (Y_S2 - 0.)**2 +  (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_1_sq ) > tol_shape,  (R_S2 + r_0*np.exp( -1.*k_shape*d_1_sq )), R_S2)

		d_2_sq = (X_S2 - 0.)**2 + (Y_S2 - R_S2)**2 +  (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_2_sq ) > tol_shape,  (R_S2 + r_0*np.exp( -1.*k_shape*d_2_sq )), R_shape)

		d_3_sq = (X_S2 - 0.)**2 + (Y_S2 + R_S2)**2 + (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_3_sq ) > tol_shape,  (R_S2 + r_0*np.exp( -1.*k_shape*d_3_sq )), R_shape)

		d_4_sq = (X_S2 + R_S2)**2 + (Y_S2 - 0.)**2 + (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_4_sq ) > tol_shape,  (R_S2 + r_0*np.exp( -1.*k_shape*d_4_sq )), R_shape)

		return R_shape

	elif(Manny_Name  == "Quad_Spikes_R3"):
		# r0 =  distortion magnitude
		k_shape = 2. #inv. length-scale of distortion decay
		tol_shape = 1.e-3

		R_S2 = 3
		X_S2 = R_S2*np.sin(phi)*np.cos(theta)
		Y_S2 = R_S2*np.sin(phi)*np.sin(theta)
		Z_S2 = R_S2*np.cos(phi)*np.ones_like(theta)

		d_1_sq = (X_S2 - R_S2)**2 + (Y_S2 - 0.)**2 +  (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_1_sq ) > tol_shape,  (R_S2 + r_0*np.exp( -1.*k_shape*d_1_sq )), R_S2)

		d_2_sq = (X_S2 - 0.)**2 + (Y_S2 - R_S2)**2 +  (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_2_sq ) > tol_shape,  (R_S2 + r_0*np.exp( -1.*k_shape*d_2_sq )), R_shape)

		d_3_sq = (X_S2 - 0.)**2 + (Y_S2 + R_S2)**2 + (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_3_sq ) > tol_shape,  (R_S2 + r_0*np.exp( -1.*k_shape*d_3_sq )), R_shape)

		d_4_sq = (X_S2 + R_S2)**2 + (Y_S2 - 0.)**2 + (Z_S2 - 0.)**2
		R_shape = np.where(r_0*np.exp( -1.*k_shape*d_4_sq ) > tol_shape,  (R_S2 + r_0*np.exp( -1.*k_shape*d_4_sq )), R_shape)

		return R_shape


	elif(Manny_Name  == "Cone_Head_Rad"):
		return 1. + r_0*np.exp(-10.*phi**2)

	elif(Manny_Name  == "Cone_Head_Logis_Rad"):
		return 1. + r_0/(1. + np.exp(-20.*(np.pi/16 - phi)))

	elif(Manny_Name == "Fission_Yeast_Rad_R2" or Manny_Name == "Fission_Yeast_Rad_R1pt2"):

		R=0.
		if(Manny_Name == "Fission_Yeast_Rad_R2"):
			R=2.
		elif(Manny_Name == "Fission_Yeast_Rad_R1pt2"):
			R=1.2
		else:
			print("\n"+"ERROR: Fission Yeast Rad Geo Not Recognized"+"\n")
		phi_1 = np.arctan( R/(r_0/2.) )
		phi_2 = np.pi - np.arctan( R/(r_0/2.) )

		if(isscalar(phi)):
			if(phi < phi_1):
				return (r_0/2.)*np.cos(phi) + np.sqrt( R**2 - ( (r_0/2)*np.sin(phi) )**2 )
			elif(phi< phi_2):
				return R/np.sin(phi)
			else:
				return (r_0/2.)*np.cos(np.pi - phi) + np.sqrt( R**2 - ( (r_0/2)*np.sin(np.pi - phi) )**2 )
		else:
			return np.where(phi < phi_1, (r_0/2.)*np.cos(phi) + np.sqrt( R**2 - ( (r_0/2)*np.sin(phi) )**2 ), np.where( phi< phi_2, R/np.sin(phi), (r_0/2.)*np.cos(np.pi - phi) + np.sqrt( R**2 - ( (r_0/2)*np.sin(np.pi - phi) )**2 ) ) )

	elif(Manny_Name == "Divet"):

		if(isscalar(phi)):
			if(np.cos(phi) > .9):
				return 1. - r_0*np.exp(-1.*(.19)/(np.cos(phi)**2 - (.9)**2) + 1.)

			else:
				return 1.

		else:
			# Vectorized for manifold functions
			return np.where(np.cos(phi) > .9,  1. - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1.), 1)

	elif(Manny_Name == "Double_Divet"):

		if(isscalar(phi)):
			if(np.cos(phi) > .9):
				return 1. - r_0*np.exp(-1.*(.19)/(np.cos(phi)**2 - (.9)**2) + 1.)

			elif(np.sin(phi)*np.cos(theta) > .9):
				return 1. - r_0*np.exp(-1.*(.19)/((np.sin(phi)*np.cos(theta))**2 - (.9)**2) + 1)

			else:
				return 1.

		else:
			# Vectorized for manifold functions
			return np.where(np.cos(phi) > .9,  1. - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1.),
				np.where(np.sin(phi)*np.cos(theta) > .9, 1. - r_0*np.exp(-1*(.19)/((np.sin(phi)*np.cos(theta))**2 - (.9)**2) + 1.), 1.))

	elif(Manny_Name == "Divets_Around_Pimple"):

		if(isscalar(phi)):
			if(abs(np.cos(phi)) > .9):
				return 1. - r_0*np.exp(-1.*(.19)/(np.cos(phi)**2 - (.9)**2) + 1.)

			elif(np.sin(phi)*np.cos(theta) > .9):
				return 1. + r_0*np.exp(-1*(.19)/((np.sin(phi)*np.cos(theta))**2 - (.9)**2) + 1)

			else:
				return 1.

		else:
			# Vectorized for manifold functions
			return np.where(abs(np.cos(phi)) > .9,  1. - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1),
				np.where(np.sin(phi)*np.cos(theta) > .9, 1. + r_0*np.exp(-1.*(.19)/((np.sin(phi)*np.cos(theta))**2 - (.9)**2) + 1.), 1.))

	elif(Manny_Name == "Hour_Glass"):
		# Note: (Previously) NOT A SPHERE for R_0 = 0
		return 1. - (6.*r_0)*np.exp(-1.0/(1.-np.cos(phi)**2)) # previously 2 + r_0, for r_0 = .4

	elif(Manny_Name == "Muffin_Top"):
		return 1. + (r_0)*np.exp(30.0)*np.exp(-30.0/(1-np.cos(phi)**2))

	else:
		print("\n"+"ERROR: RADIAL Manifold Name: "+str(Manny_Name)+", Not Recognized"+"\n")

# Takes an array of R0 Values and returns string format for Rayleigh Dissipation Studies (!!!!VTU NOT PICKLE!!!!)
def Max_Decimal_R0_Array(R0_Values_Array):

	max_dec_length = 0

	for R0_i in range(len(R0_Values_Array)):

		R0_Value_i = R0_Values_Array[R0_i]
		str_index = str(R0_Value_i)[::-1].find('.')

		if(max_dec_length < str_index and str_index != 0):
			max_dec_length = str_index

	new_R0_array_labels = []

	for R0_j in range(len(R0_Values_Array)):

		a, b  = str(R0_Values_Array[R0_j]).split(".", 1)
		b_adjusted = b.ljust(max_dec_length, "0")


		decimal_form_str = a + "." + b_adjusted
		alpha_numeric_form = decimal_form_str.replace(".", "pt")

		new_R0_array_labels.append(alpha_numeric_form)

	return new_R0_array_labels

#######################################################################################################################

class manifold(object): #This now represents geo of S^2, will later be adapted to other manifolds with same topology

	'''
	 Use Manifold name to automatically load/pickle manny inv mats:
	 Format: Maniold_Official_Name = Man_Shape_Name+"R_0_"+R_0_str+"_Pdeg_"+str(deg_basis)+"_Q"+str(num_quad_pts)
	 Filename "Manny_Inv_Mats_" +  Maniold_Official_Name + ".p", goes in 'Pickled_Manny_Inv_Mat_Files' sub-directory
	 Man_Shape_Name = "S2", "Chew_Toy", "Gen_R0_Pill", "Dog_Shit", etc
	 R_0_str = "0pt3", "0pt0" for example.
	'''

	def __init__(self, Manifold_Constr_Dict): #BJG: Need to add option to initialize from point cloud of lbdv point vals, and from named manifold
		# old constructor: (self, R_func, R_deg, lbdv, Maniold_Official_Name = [])

		#print("Constructing Manifold") # BJG: should add verbose option

		self.Tol = 1.e-3 # Threshold for considering a quantity as 0, in terms of where to switch charts (replaces using condtion: lbdv.Chart_of_Quad_Pts > 0 )
		self.pickling = Manifold_Constr_Dict['Pickle_Manny_Data'] #BJG: May not want to pickle for moving surface and debugging

		lbdv = Manifold_Constr_Dict['Maniold_lbdv']
		self.num_quad_pts = lbdv.lbdv_quad_pts
		self.Man_SPH_Deg = Manifold_Constr_Dict['Manifold_SPH_deg']

		self.Use_Man_Name = Manifold_Constr_Dict['use_manifold_name']
		self.Man_Shape_Dict = Manifold_Constr_Dict['Maniold_Name_Dict'] # this is a (possibly trivial) dictionary of manifold name and r_0 value, OR {x,y,z} at quad pts


		# BJG: given name and r0 value, we can use function to get points we need:
		if(self.Use_Man_Name == True):

			self.IsManRad = self.Man_Shape_Dict['Is_Manifold_Radial']
			self.Man_R0_Val = self.Man_Shape_Dict['Maniold_R0_Value']
			self.Man_Shape_Name = self.Man_Shape_Dict['Maniold_Shape_Name'] # name of manifold shape (i.e. spehre, ellipsoid, etc)

			R_0_str = str(self.Man_R0_Val).replace(".", "pt")
			self.Man_Official_Name = self.Man_Shape_Name + "_R_0_"+R_0_str+"_Pdeg_"+str(self.Man_SPH_Deg)+"_Q"+str(self.num_quad_pts) # str for picking, with R0, num_pts, deg of basis, etc

			self.X_A_Pts = Manny_Fn_Def_X(lbdv.theta_pts, lbdv.phi_pts, self.Man_R0_Val, self.Man_Shape_Name, self.IsManRad)
			self.Y_A_Pts = Manny_Fn_Def_Y(lbdv.theta_pts, lbdv.phi_pts, self.Man_R0_Val, self.Man_Shape_Name, self.IsManRad)
			self.Z_A_Pts = Manny_Fn_Def_Z(lbdv.theta_pts, lbdv.phi_pts, self.Man_R0_Val, self.Man_Shape_Name, self.IsManRad)

		# BJG: otherwise, we use input lebedev quad pts, in our shape dictionary:
		else:
			self.X_A_Pts = self.Man_Shape_Dict['coordinates'][:, 0]
			self.Y_A_Pts = self.Man_Shape_Dict['coordinates'][:, 1]
			self.Z_A_Pts = self.Man_Shape_Dict['coordinates'][:, 2]

			#print("self.X_A_Pts.shape = "+str(self.X_A_Pts.shape))


		# BJG: use rotation conversion between charts
		self.quad_pts = range(self.num_quad_pts) # list of quad pts, for vectorization:
		quad_pts_inv_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(self.quad_pts) #lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pts)

		self.X_B_Pts = self.X_A_Pts[quad_pts_inv_rot] #self.Z_A_Pts
		self.Y_B_Pts = self.Y_A_Pts[quad_pts_inv_rot] #self.Y_A_Pts
		self.Z_B_Pts = self.Z_A_Pts[quad_pts_inv_rot] #-1.*self.X_A_Pts

		self.Cart_Coors_A = np.hstack(( self.X_A_Pts, self.Y_A_Pts, self.Z_A_Pts )) # Cart Coors uses these
		self.Cart_Coors_B = np.hstack(( self.X_B_Pts, self.Y_B_Pts, self.Z_B_Pts ))

		self.X, self.X_Bar =  sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.X_A_Pts, self.Man_SPH_Deg, lbdv)
		self.Y, self.Y_Bar =  sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.Y_A_Pts, self.Man_SPH_Deg, lbdv)
		self.Z, self.Z_Bar =  sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.Z_A_Pts, self.Man_SPH_Deg, lbdv)


		# BJG: Lists of Quad Pt Vals, so we can evaluate, with necesary derivatives at these points:

		self.X_theta = self.X.Quick_Theta_Der()
		self.X_theta_A_Pts = self.X_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)
		self.X_phi_A_Pts = self.X.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.X_theta_phi_A_Pts = self.X_theta.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.X_phi_phi_A_Pts = self.X.Eval_SPH_Der_Phi_Phi_Coef(self.quad_pts, lbdv)

		self.X_theta_theta = self.X_theta.Quick_Theta_Der()
		self.X_theta_theta_A_Pts = self.X_theta_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)

		self.X_Bar_theta = self.X_Bar.Quick_Theta_Bar_Der()
		self.X_theta_B_Pts = self.X_Bar_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)
		self.X_phi_B_Pts = self.X_Bar.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.X_theta_phi_B_Pts = self.X_Bar_theta.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.X_phi_phi_B_Pts = self.X_Bar.Eval_SPH_Der_Phi_Phi_Coef(self.quad_pts, lbdv)

		self.X_Bar_theta_theta = self.X_Bar_theta.Quick_Theta_Bar_Der()
		self.X_theta_theta_B_Pts = self.X_Bar_theta_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)

		self.Y_theta = self.Y.Quick_Theta_Der()
		self.Y_theta_A_Pts = self.Y_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)
		self.Y_phi_A_Pts = self.Y.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.Y_theta_phi_A_Pts = self.Y_theta.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.Y_phi_phi_A_Pts = self.Y.Eval_SPH_Der_Phi_Phi_Coef(self.quad_pts, lbdv)

		self.Y_theta_theta = self.Y_theta.Quick_Theta_Der()
		self.Y_theta_theta_A_Pts = self.Y_theta_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)

		self.Y_Bar_theta = self.Y_Bar.Quick_Theta_Bar_Der()
		self.Y_theta_B_Pts = self.Y_Bar_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)
		self.Y_phi_B_Pts = self.Y_Bar.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.Y_theta_phi_B_Pts = self.Y_Bar_theta.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.Y_phi_phi_B_Pts = self.Y_Bar.Eval_SPH_Der_Phi_Phi_Coef(self.quad_pts, lbdv)

		self.Y_Bar_theta_theta = self.Y_Bar_theta.Quick_Theta_Bar_Der()
		self.Y_theta_theta_B_Pts = self.Y_Bar_theta_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)

		self.Z_theta = self.Z.Quick_Theta_Der()
		self.Z_theta_A_Pts = self.Z_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)
		self.Z_phi_A_Pts = self.Z.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.Z_theta_phi_A_Pts = self.Z_theta.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.Z_phi_phi_A_Pts = self.Z.Eval_SPH_Der_Phi_Phi_Coef(self.quad_pts, lbdv)

		self.Z_theta_theta = self.Z_theta.Quick_Theta_Der()
		self.Z_theta_theta_A_Pts = self.Z_theta_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)

		self.Z_Bar_theta = self.Z_Bar.Quick_Theta_Bar_Der()
		self.Z_theta_B_Pts = self.Z_Bar_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)
		self.Z_phi_B_Pts = self.Z_Bar.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.Z_theta_phi_B_Pts = self.Z_Bar_theta.Eval_SPH_Der_Phi_Coef(self.quad_pts, lbdv)
		self.Z_phi_phi_B_Pts = self.Z_Bar.Eval_SPH_Der_Phi_Phi_Coef(self.quad_pts, lbdv)

		self.Z_Bar_theta_theta = self.Z_Bar_theta.Quick_Theta_Bar_Der()
		self.Z_theta_theta_B_Pts = self.Z_Bar_theta_theta.Eval_SPH_Coef_Mat(self.quad_pts, lbdv)


		#print("Constucting Tensors For euc_k_form") # BJG: should add verbose option


		self.Metric_Factor_A_pts = np.sqrt( (self.X_theta_A_Pts*self.Y_phi_A_Pts - self.Y_theta_A_Pts*self.X_phi_A_Pts)**2 + (self.X_theta_A_Pts*self.Z_phi_A_Pts - self.Z_theta_A_Pts*self.X_phi_A_Pts)**2 + (self.Y_theta_A_Pts*self.Z_phi_A_Pts - self.Z_theta_A_Pts*self.Y_phi_A_Pts)**2 )
		self.Metric_Factor_B_pts = np.sqrt( (self.X_theta_B_Pts*self.Y_phi_B_Pts - self.Y_theta_B_Pts*self.X_phi_B_Pts)**2 + (self.X_theta_B_Pts*self.Z_phi_B_Pts - self.Z_theta_B_Pts*self.X_phi_B_Pts)**2 + (self.Y_theta_B_Pts*self.Z_phi_B_Pts - self.Z_theta_B_Pts*self.Y_phi_B_Pts)**2 )

		# For error metric on Manny:
		self.Metric_Factor_A_over_sin_phi_pts = np.where(np.sin(lbdv.phi_pts) > self.Tol, self.Metric_Factor_A_pts/np.sin(lbdv.phi_pts), 0)
		self.Metric_Factor_B_over_sin_phi_bar_pts = np.where(np.sin(lbdv.phi_pts) > self.Tol, self.Metric_Factor_B_pts/np.sin(lbdv.phi_pts), 0)


		### For Pointwise Explicit LB:
		self.E_A_pts = self.X_theta_A_Pts**2 + self.Y_theta_A_Pts**2 + self.Z_theta_A_Pts**2
		self.E_B_pts = self.X_theta_B_Pts**2 + self.Y_theta_B_Pts**2 + self.Z_theta_B_Pts**2

		self.F_A_pts = self.X_theta_A_Pts*self.X_phi_A_Pts + self.Y_theta_A_Pts*self.Y_phi_A_Pts + self.Z_theta_A_Pts*self.Z_phi_A_Pts
		self.F_B_pts = self.X_theta_B_Pts*self.X_phi_B_Pts + self.Y_theta_B_Pts*self.Y_phi_B_Pts + self.Z_theta_B_Pts*self.Z_phi_B_Pts

		self.G_A_pts = self.X_phi_A_Pts**2 + self.Y_phi_A_Pts**2 + self.Z_phi_A_Pts**2
		self.G_B_pts = self.X_phi_B_Pts**2 + self.Y_phi_B_Pts**2 + self.Z_phi_B_Pts**2

		# |g| = (\sqrt|g|)^2
		self.Metric_Factor_Squared_A = self.E_A_pts*self.G_A_pts - self.F_A_pts**2
		self.Metric_Factor_Squared_B = self.E_B_pts*self.G_B_pts - self.F_B_pts**2

		self.E_theta_A_pts = 2.*self.X_theta_A_Pts*self.X_theta_theta_A_Pts + 2.*self.Y_theta_A_Pts*self.Y_theta_theta_A_Pts + 2.*self.Z_theta_A_Pts*self.Z_theta_theta_A_Pts
		self.E_phi_A_pts = 2.*self.X_theta_A_Pts*self.X_theta_phi_A_Pts + 2.*self.Y_theta_A_Pts*self.Y_theta_phi_A_Pts + 2.*self.Z_theta_A_Pts*self.Z_theta_phi_A_Pts

		self.E_theta_B_pts = 2.*self.X_theta_B_Pts*self.X_theta_theta_B_Pts + 2.*self.Y_theta_B_Pts*self.Y_theta_theta_B_Pts + 2.*self.Z_theta_B_Pts*self.Z_theta_theta_B_Pts
		self.E_phi_B_pts = 2.*self.X_theta_B_Pts*self.X_theta_phi_B_Pts + 2.*self.Y_theta_B_Pts*self.Y_theta_phi_B_Pts + 2.*self.Z_theta_B_Pts*self.Z_theta_phi_B_Pts

		self.F_theta_A_pts = (self.X_theta_A_Pts*self.X_theta_phi_A_Pts + self.X_phi_A_Pts*self.X_theta_theta_A_Pts) + (self.Y_theta_A_Pts*self.Y_theta_phi_A_Pts + self.Y_phi_A_Pts*self.Y_theta_theta_A_Pts) + (self.Z_theta_A_Pts*self.Z_theta_phi_A_Pts + self.Z_phi_A_Pts*self.Z_theta_theta_A_Pts)
		self.F_phi_A_pts = (self.X_theta_A_Pts*self.X_phi_phi_A_Pts + self.X_phi_A_Pts*self.X_theta_phi_A_Pts) +  (self.Y_theta_A_Pts*self.Y_phi_phi_A_Pts + self.Y_phi_A_Pts*self.Y_theta_phi_A_Pts) +  (self.Z_theta_A_Pts*self.Z_phi_phi_A_Pts + self.Z_phi_A_Pts*self.Z_theta_phi_A_Pts)

		self.F_theta_B_pts = (self.X_theta_B_Pts*self.X_theta_phi_B_Pts + self.X_phi_B_Pts*self.X_theta_theta_B_Pts) + (self.Y_theta_B_Pts*self.Y_theta_phi_B_Pts + self.Y_phi_B_Pts*self.Y_theta_theta_B_Pts) + (self.Z_theta_B_Pts*self.Z_theta_phi_B_Pts + self.Z_phi_B_Pts*self.Z_theta_theta_B_Pts)
		self.F_phi_B_pts = (self.X_theta_B_Pts*self.X_phi_phi_B_Pts + self.X_phi_B_Pts*self.X_theta_phi_B_Pts) +  (self.Y_theta_B_Pts*self.Y_phi_phi_B_Pts + self.Y_phi_B_Pts*self.Y_theta_phi_B_Pts) +  (self.Z_theta_B_Pts*self.Z_phi_phi_B_Pts + self.Z_phi_B_Pts*self.Z_theta_phi_B_Pts)

		self.G_theta_A_pts = 2.*self.X_phi_A_Pts*self.X_theta_phi_A_Pts + 2.*self.Y_phi_A_Pts*self.Y_theta_phi_A_Pts + 2.*self.Z_phi_A_Pts*self.Z_theta_phi_A_Pts
		self.G_phi_A_pts =  2.*self.X_phi_A_Pts*self.X_phi_phi_A_Pts + 2.*self.Y_phi_A_Pts*self.Y_phi_phi_A_Pts + 2.*self.Z_phi_A_Pts*self.Z_phi_phi_A_Pts

		self.G_theta_B_pts = 2.*self.X_phi_B_Pts*self.X_theta_phi_B_Pts + 2.*self.Y_phi_B_Pts*self.Y_theta_phi_B_Pts + 2.*self.Z_phi_B_Pts*self.Z_theta_phi_B_Pts
		self.G_phi_B_pts =  2.*self.X_phi_B_Pts*self.X_phi_phi_B_Pts + 2.*self.Y_phi_B_Pts*self.Y_phi_phi_B_Pts + 2.*self.Z_phi_B_Pts*self.Z_phi_phi_B_Pts


		# For g^(ij)*sqrt(g):
		self.E_over_Metric_Factor_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.E_A_pts/self.Metric_Factor_A_pts, 0)
		self.F_over_Metric_Factor_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.F_A_pts/self.Metric_Factor_A_pts, 0)
		self.G_over_Metric_Factor_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.G_A_pts/self.Metric_Factor_A_pts, 0)

		self.E_over_Metric_Factor_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.E_B_pts/self.Metric_Factor_B_pts, 0)
		self.F_over_Metric_Factor_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.F_B_pts/self.Metric_Factor_B_pts, 0)
		self.G_over_Metric_Factor_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.G_B_pts/self.Metric_Factor_B_pts, 0)

		# d_i(sqrt(g)):
		self.Metric_Factor_dTheta_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (0.5)*(self.E_theta_A_pts*self.G_A_pts + self.E_A_pts*self.G_theta_A_pts- 2*self.F_A_pts*self.F_theta_A_pts)/self.Metric_Factor_A_pts, 0)
		self.Metric_Factor_dPhi_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (0.5)*(self.E_phi_A_pts*self.G_A_pts + self.E_A_pts*self.G_phi_A_pts- 2*self.F_A_pts*self.F_phi_A_pts)/self.Metric_Factor_A_pts, 0)

		self.Metric_Factor_dTheta_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (0.5)*(self.E_theta_B_pts*self.G_B_pts + self.E_B_pts*self.G_theta_B_pts- 2*self.F_B_pts*self.F_theta_B_pts)/self.Metric_Factor_B_pts, 0)
		self.Metric_Factor_dPhi_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (0.5)*(self.E_phi_B_pts*self.G_B_pts + self.E_B_pts*self.G_phi_B_pts- 2*self.F_B_pts*self.F_phi_B_pts)/self.Metric_Factor_B_pts, 0)

		# d_i(|g|):
		self.Metric_Factor_Squared_dTheta_A_pts = self.E_theta_A_pts*self.G_A_pts + self.E_A_pts*self.G_theta_A_pts - 2*self.F_A_pts*self.F_theta_A_pts
		self.Metric_Factor_Squared_dPhi_A_pts = self.E_phi_A_pts*self.G_A_pts + self.E_A_pts*self.G_phi_A_pts - 2*self.F_A_pts*self.F_phi_A_pts

		self.Metric_Factor_Squared_dTheta_B_pts = self.E_theta_B_pts*self.G_B_pts + self.E_B_pts*self.G_theta_B_pts - 2*self.F_B_pts*self.F_theta_B_pts
		self.Metric_Factor_Squared_dPhi_B_pts = self.E_phi_B_pts*self.G_B_pts + self.E_B_pts*self.G_phi_B_pts - 2*self.F_B_pts*self.F_phi_B_pts


		# d_i(sqrt(g))/sqrt(g): # for divergence expression:
		self.Metric_Factor_dTheta_over_Metric_Factor_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Metric_Factor_dTheta_A_pts/self.Metric_Factor_A_pts, 0)
		self.Metric_Factor_dPhi_over_Metric_Factor_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Metric_Factor_dPhi_A_pts/self.Metric_Factor_A_pts, 0)

		self.Metric_Factor_dTheta_over_Metric_Factor_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Metric_Factor_dTheta_B_pts/self.Metric_Factor_B_pts, 0)
		self.Metric_Factor_dPhi_over_Metric_Factor_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Metric_Factor_dPhi_B_pts/self.Metric_Factor_B_pts, 0)


		#d_k(g_ik/met_fac):
		#self.E_over_Metric_Factor_dTheta_A = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_A_pts*self.E_theta_A_pts - self.E_A_pts*self.Metric_Factor_dTheta_A_pts)/self.Metric_Factor_Squared_A, 0)
		self.F_over_Metric_Factor_dTheta_A = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_A_pts*self.F_theta_A_pts - self.F_A_pts*self.Metric_Factor_dTheta_A_pts)/self.Metric_Factor_Squared_A, 0)
		self.G_over_Metric_Factor_dTheta_A = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_A_pts*self.G_theta_A_pts - self.G_A_pts*self.Metric_Factor_dTheta_A_pts)/self.Metric_Factor_Squared_A, 0)

		self.E_over_Metric_Factor_dPhi_A = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_A_pts*self.E_phi_A_pts - self.E_A_pts*self.Metric_Factor_dPhi_A_pts)/self.Metric_Factor_Squared_A, 0)
		self.F_over_Metric_Factor_dPhi_A = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_A_pts*self.F_phi_A_pts - self.F_A_pts*self.Metric_Factor_dPhi_A_pts)/self.Metric_Factor_Squared_A, 0)
		#self.G_over_Metric_Factor_dPhi_A = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_A_pts*self.G_phi_A_pts - self.G_A_pts*self.Metric_Factor_dPhi_A_pts)/self.Metric_Factor_Squared_A, 0)


		#self.E_over_Metric_Factor_dTheta_B = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_B_pts*self.E_theta_B_pts - self.E_B_pts*self.Metric_Factor_dTheta_B_pts)/self.Metric_Factor_Squared_B, 0)
		self.F_over_Metric_Factor_dTheta_B = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_B_pts*self.F_theta_B_pts - self.F_B_pts*self.Metric_Factor_dTheta_B_pts)/self.Metric_Factor_Squared_B, 0)
		self.G_over_Metric_Factor_dTheta_B = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_B_pts*self.G_theta_B_pts - self.G_B_pts*self.Metric_Factor_dTheta_B_pts)/self.Metric_Factor_Squared_B, 0)

		self.E_over_Metric_Factor_dPhi_B = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_B_pts*self.E_phi_B_pts - self.E_B_pts*self.Metric_Factor_dPhi_B_pts)/self.Metric_Factor_Squared_B, 0)
		self.F_over_Metric_Factor_dPhi_B = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_B_pts*self.F_phi_B_pts - self.F_B_pts*self.Metric_Factor_dPhi_B_pts)/self.Metric_Factor_Squared_B, 0)
		#self.G_over_Metric_Factor_dPhi_B = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.Metric_Factor_B_pts*self.G_phi_B_pts - self.G_B_pts*self.Metric_Factor_dPhi_B_pts)/self.Metric_Factor_Squared_B, 0)
		###

		### From Sharp!/*(1-forms):
		self.g_Inv_Theta_Theta_A_Pts = self.G_A_pts/self.Metric_Factor_Squared_A
		self.g_Inv_Theta_Phi_A_Pts = -1.*self.F_A_pts/self.Metric_Factor_Squared_A
		self.g_Inv_Phi_Phi_A_Pts = self.E_A_pts/self.Metric_Factor_Squared_A

		self.g_Inv_Theta_Theta_B_Pts = self.G_B_pts/self.Metric_Factor_Squared_B
		self.g_Inv_Theta_Phi_B_Pts = -1.*self.F_B_pts/self.Metric_Factor_Squared_B
		self.g_Inv_Phi_Phi_B_Pts = self.E_B_pts/self.Metric_Factor_Squared_B

		self.Sigma_Theta_A_Pts = np.hstack(( self.X_theta_A_Pts, self.Y_theta_A_Pts, self.Z_theta_A_Pts ))
		self.Sigma_Theta_B_Pts = np.hstack(( self.X_theta_B_Pts, self.Y_theta_B_Pts, self.Z_theta_B_Pts ))

		self.Sigma_Phi_A_Pts = np.hstack(( self.X_phi_A_Pts, self.Y_phi_A_Pts, self.Z_phi_A_Pts ))
		self.Sigma_Phi_B_Pts = np.hstack(( self.X_phi_B_Pts, self.Y_phi_B_Pts, self.Z_phi_B_Pts ))

		# BJG: derivative vectors are easy in this framework:
		self.Sigma_Theta_Theta_A_Pts = np.hstack(( self.X_theta_theta_A_Pts, self.Y_theta_theta_A_Pts, self.Z_theta_theta_A_Pts ))
		self.Sigma_Theta_Theta_B_Pts = np.hstack(( self.X_theta_theta_B_Pts, self.Y_theta_theta_B_Pts, self.Z_theta_theta_B_Pts ))

		self.Sigma_Theta_Phi_A_Pts = np.hstack(( self.X_theta_phi_A_Pts, self.Y_theta_phi_A_Pts, self.Z_theta_phi_A_Pts ))
		self.Sigma_Theta_Phi_B_Pts = np.hstack(( self.X_theta_phi_B_Pts, self.Y_theta_phi_B_Pts, self.Z_theta_phi_B_Pts ))

		self.Sigma_Phi_Phi_A_Pts = np.hstack(( self.X_phi_phi_A_Pts, self.Y_phi_phi_A_Pts, self.Z_phi_phi_A_Pts ))
		self.Sigma_Phi_Phi_B_Pts = np.hstack(( self.X_phi_phi_B_Pts, self.Y_phi_phi_B_Pts, self.Z_phi_phi_B_Pts ))


		# Normal Vectors and II can be computed here as well (within Chart, for pickled fields below):
		self.Normal_Dir_X_A_Pts = self.Y_theta_A_Pts*self.Z_phi_A_Pts - self.Z_theta_A_Pts*self.Y_phi_A_Pts
		self.Normal_Dir_X_B_Pts = self.Y_theta_B_Pts*self.Z_phi_B_Pts - self.Z_theta_B_Pts*self.Y_phi_B_Pts

		self.Normal_Dir_Y_A_Pts = -1.*(self.X_theta_A_Pts*self.Z_phi_A_Pts - self.Z_theta_A_Pts*self.X_phi_A_Pts)
		self.Normal_Dir_Y_B_Pts = -1.*(self.X_theta_B_Pts*self.Z_phi_B_Pts - self.Z_theta_B_Pts*self.X_phi_B_Pts)

		self.Normal_Dir_Z_A_Pts = self.X_theta_A_Pts*self.Y_phi_A_Pts - self.Y_theta_A_Pts*self.X_phi_A_Pts
		self.Normal_Dir_Z_B_Pts = self.X_theta_B_Pts*self.Y_phi_B_Pts - self.Y_theta_B_Pts*self.X_phi_B_Pts

		self.Normal_Dirs_A_Pts = np.hstack(( self.Normal_Dir_X_A_Pts, self.Normal_Dir_Y_A_Pts, self.Normal_Dir_Z_A_Pts ))
		self.Normal_Dirs_B_Pts = np.hstack(( self.Normal_Dir_X_B_Pts, self.Normal_Dir_Y_B_Pts, self.Normal_Dir_Z_B_Pts ))

		self.Normal_Vec_X_A_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_X_A_Pts/self.Metric_Factor_A_pts, 0)
		self.Normal_Vec_X_B_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_X_B_Pts/self.Metric_Factor_B_pts, 0)

		self.Normal_Vec_Y_A_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_Y_A_Pts/self.Metric_Factor_A_pts, 0)
		self.Normal_Vec_Y_B_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_Y_B_Pts/self.Metric_Factor_B_pts, 0)

		self.Normal_Vec_Z_A_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_Z_A_Pts/self.Metric_Factor_A_pts, 0)
		self.Normal_Vec_Z_B_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_Z_B_Pts/self.Metric_Factor_B_pts, 0)

		self.Normal_Vecs_A_Pts = np.hstack(( self.Normal_Vec_X_A_Pts, self.Normal_Vec_Y_A_Pts, self.Normal_Vec_Z_A_Pts ))
		self.Normal_Vecs_B_Pts = np.hstack(( self.Normal_Vec_X_B_Pts, self.Normal_Vec_Y_B_Pts, self.Normal_Vec_Z_B_Pts ))

		self.L_A_Pts = self.Normal_Vec_X_A_Pts*self.X_theta_theta_A_Pts + self.Normal_Vec_Y_A_Pts*self.Y_theta_theta_A_Pts + self.Normal_Vec_Z_A_Pts*self.Z_theta_theta_A_Pts
		self.L_B_Pts = self.Normal_Vec_X_B_Pts*self.X_theta_theta_B_Pts + self.Normal_Vec_Y_B_Pts*self.Y_theta_theta_B_Pts + self.Normal_Vec_Z_B_Pts*self.Z_theta_theta_B_Pts

		self.M_A_Pts = self.Normal_Vec_X_A_Pts*self.X_theta_phi_A_Pts + self.Normal_Vec_Y_A_Pts*self.Y_theta_phi_A_Pts + self.Normal_Vec_Z_A_Pts*self.Z_theta_phi_A_Pts
		self.M_B_Pts = self.Normal_Vec_X_B_Pts*self.X_theta_phi_B_Pts + self.Normal_Vec_Y_B_Pts*self.Y_theta_phi_B_Pts + self.Normal_Vec_Z_B_Pts*self.Z_theta_phi_B_Pts

		self.N_A_Pts = self.Normal_Vec_X_A_Pts*self.X_phi_phi_A_Pts + self.Normal_Vec_Y_A_Pts*self.Y_phi_phi_A_Pts + self.Normal_Vec_Z_A_Pts*self.Z_phi_phi_A_Pts
		self.N_B_Pts = self.Normal_Vec_X_B_Pts*self.X_phi_phi_B_Pts + self.Normal_Vec_Y_B_Pts*self.Y_phi_phi_B_Pts + self.Normal_Vec_Z_B_Pts*self.Z_phi_phi_B_Pts

		# We can use this to compute entries of the Weingarten Map Directly, where W = [[W_11, W_12], [W_21, W_22]]:
		self.Wein_11_A_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.L_A_Pts*self.G_A_pts - self.M_A_Pts*self.F_A_pts)/self.Metric_Factor_Squared_A, 0)
		self.Wein_12_A_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.M_A_Pts*self.G_A_pts - self.N_A_Pts*self.F_A_pts)/self.Metric_Factor_Squared_A, 0)
		self.Wein_21_A_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.M_A_Pts*self.E_A_pts - self.L_A_Pts*self.F_A_pts)/self.Metric_Factor_Squared_A, 0)
		self.Wein_22_A_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.N_A_Pts*self.E_A_pts - self.M_A_Pts*self.F_A_pts)/self.Metric_Factor_Squared_A, 0)

		self.Wein_11_B_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.L_B_Pts*self.G_B_pts - self.M_B_Pts*self.F_B_pts)/self.Metric_Factor_Squared_B, 0)
		self.Wein_12_B_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.M_B_Pts*self.G_B_pts - self.N_B_Pts*self.F_B_pts)/self.Metric_Factor_Squared_B, 0)
		self.Wein_21_B_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.M_B_Pts*self.E_B_pts - self.L_B_Pts*self.F_B_pts)/self.Metric_Factor_Squared_B, 0)
		self.Wein_22_B_Pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.N_B_Pts*self.E_B_pts - self.M_B_Pts*self.F_B_pts)/self.Metric_Factor_Squared_B, 0)


		### From Two_Form_Conv_to_Euc_pt/ *(0-forms)!/ d(1-forms)/ *(1-forms)

		self.R_sq_A_Pts = self.X_A_Pts**2 +  self.Y_A_Pts**2 + self.Z_A_Pts**2
		self.R_sq_B_Pts = self.X_B_Pts**2 +  self.Y_B_Pts**2 + self.Z_B_Pts**2

		Denom_A = np.multiply((self.R_sq_A_Pts), np.sqrt(self.X_A_Pts**2 + self.Y_A_Pts**2))

		self.dx_dy_A_Vals_From_Polar_NEW = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_Z_A_Pts/self.Metric_Factor_Squared_A, 0)
		self.dx_dy_A_Vals_From_Polar = np.where(lbdv.Chart_of_Quad_Pts > 0, -self.Z_A_Pts/Denom_A, 0)
		self.dx_dz_A_Vals_From_Polar_NEW = np.where(lbdv.Chart_of_Quad_Pts > 0, -1.*self.Normal_Dir_Y_A_Pts/self.Metric_Factor_Squared_A, 0)
		self.dx_dz_A_Vals_From_Polar = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Y_A_Pts/Denom_A, 0)
		self.dy_dz_A_Vals_From_Polar_NEW = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_X_A_Pts/self.Metric_Factor_Squared_A, 0)
		self.dy_dz_A_Vals_From_Polar = np.where(lbdv.Chart_of_Quad_Pts > 0, -self.X_A_Pts/Denom_A, 0)

		#print("self.dx_dy_A_Vals_From_Polar_NEW = "+str(self.dx_dy_A_Vals_From_Polar_NEW))
		#print("self.dx_dy_A_Vals_From_Polar_OLD = "+str(self.dx_dy_A_Vals_From_Polar))

		Denom_B = np.multiply((self.R_sq_B_Pts), np.sqrt(self.Y_B_Pts**2 + self.Z_B_Pts**2))

		self.dx_dy_B_Vals_From_Polar_NEW = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_Z_B_Pts/self.Metric_Factor_Squared_B, 0)
		self.dx_dy_B_Vals_From_Polar = np.where(lbdv.Chart_of_Quad_Pts > 0, -self.Z_B_Pts/Denom_B, 0)
		self.dx_dz_B_Vals_From_Polar_NEW = np.where(lbdv.Chart_of_Quad_Pts > 0, -1.*self.Normal_Dir_Y_B_Pts/self.Metric_Factor_Squared_B, 0)
		self.dx_dz_B_Vals_From_Polar = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Y_B_Pts/Denom_B, 0)
		self.dy_dz_B_Vals_From_Polar_NEW = np.where(lbdv.Chart_of_Quad_Pts > 0, self.Normal_Dir_X_B_Pts/self.Metric_Factor_Squared_B, 0)
		self.dy_dz_B_Vals_From_Polar = np.where(lbdv.Chart_of_Quad_Pts > 0, -self.X_B_Pts/Denom_B, 0)


		# THIS IS THE MOST TIME CONSUMING PART WE NEED TO PRECOMPUTE/VECTORIZE!

		def zero_vector_of_basis_mats():
			return np.zeros((3,3, lbdv.lbdv_quad_pts))

		# If we are given a name for the manifold, we can use pickling
		Manny_Inv_Mats_filepath = []

		if( Manifold_Constr_Dict['use_manifold_name'] == True ):
			Inv_Mats_Name = "Manny_Inv_Mats_"+ self.Man_Official_Name +".p" #name of file we dump/load the inv_mats from
			Manny_Inv_Mats_filepath = os.path.join(PICKLE_Manny_DIR, Inv_Mats_Name)

		if( Manifold_Constr_Dict['use_manifold_name'] == False  or os.path.isfile(Manny_Inv_Mats_filepath) == False or self.Man_Official_Name == [] or self.pickling == False): # If we need to (re)generate these:

			self.rho_A_Mats = zero_vector_of_basis_mats() #rho = G*(A^-1)
			self.rho_B_Mats = zero_vector_of_basis_mats()

			self.rho_theta_A_Mats = zero_vector_of_basis_mats() #rho_i = G_i*(A^-1)
			self.rho_theta_B_Mats = zero_vector_of_basis_mats()

			self.rho_phi_A_Mats = zero_vector_of_basis_mats()
			self.rho_phi_B_Mats = zero_vector_of_basis_mats()

			self.xi_theta_A_Mats = zero_vector_of_basis_mats() #xi_i = -1*G*(A^-1)*(d_i[A^-1])*(A^-1)
			self.xi_theta_B_Mats = zero_vector_of_basis_mats()

			self.xi_phi_A_Mats = zero_vector_of_basis_mats()
			self.xi_phi_B_Mats = zero_vector_of_basis_mats()

			# Calculate K at quad pts, for Stokes Solver:
			self.K_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.L_A_Pts*self.N_A_Pts - self.M_A_Pts**2)/self.Metric_Factor_Squared_A, 0) #zeros_like(lbdv.X) #Note these are Vectors
			self.K_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.L_B_Pts*self.N_B_Pts - self.M_B_Pts**2)/self.Metric_Factor_Squared_B, 0) #zeros_like(lbdv.X)

			# Calculate H at quad pts, for Droplets:
			self.H_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.L_A_Pts*self.G_A_pts -2.*self.M_A_Pts*self.F_A_pts + self.N_A_Pts*self.E_A_pts)/(2.*self.Metric_Factor_Squared_A), 0)
			self.H_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, (self.L_B_Pts*self.G_B_pts -2.*self.M_B_Pts*self.F_B_pts + self.N_B_Pts*self.E_B_pts)/(2.*self.Metric_Factor_Squared_B), 0)

			#print("Computing Flat Tensors Needed") # BJG: should add verbose option

			for quad_pt in range(self.num_quad_pts):
				if(lbdv.Chart_of_Quad_Pts[quad_pt] > 0):

					# Change of Basis, metric tensor mats, and their first derivatives
					A_Mat_pt = self.Change_Basis_Mat(quad_pt, 'A')
					B_Mat_pt = self.Change_Basis_Mat(quad_pt, 'B')

					A_theta_Mat_pt = self.dChange_Basis_Mat_theta(quad_pt, 'A')
					B_theta_Mat_pt = self.dChange_Basis_Mat_theta(quad_pt, 'B')

					A_phi_Mat_pt = self.dChange_Basis_Mat_phi(quad_pt, 'A')
					B_phi_Mat_pt = self.dChange_Basis_Mat_phi(quad_pt, 'B')

					G_A_Mat_pt = self.G_Mat(quad_pt, 'A')
					G_B_Mat_pt = self.G_Mat(quad_pt, 'B')

					G_theta_A_Mat_pt = self.dG_Mat_theta(quad_pt, 'A')
					G_theta_B_Mat_pt = self.dG_Mat_theta(quad_pt, 'B')

					G_phi_A_Mat_pt = self.dG_Mat_phi(quad_pt, 'A')
					G_phi_B_Mat_pt = self.dG_Mat_phi(quad_pt, 'B')

					# use these to compute matrcies for flat, and derivative of cotangent components
					rho_A_pt = np.linalg.solve(A_Mat_pt.T, G_A_Mat_pt.T).T
					rho_B_pt = np.linalg.solve(B_Mat_pt.T, G_B_Mat_pt.T).T

					rho_theta_A_pt = np.linalg.solve(A_Mat_pt.T, G_theta_A_Mat_pt.T).T
					rho_theta_B_pt = np.linalg.solve(B_Mat_pt.T, G_theta_B_Mat_pt.T).T

					rho_phi_A_pt = np.linalg.solve(A_Mat_pt.T, G_phi_A_Mat_pt.T).T
					rho_phi_B_pt = np.linalg.solve(B_Mat_pt.T, G_phi_B_Mat_pt.T).T

					xi_theta_A_pt = np.linalg.solve(A_Mat_pt.T, np.dot(-1*A_theta_Mat_pt.T, rho_A_pt.T)).T
					xi_theta_B_pt = np.linalg.solve(B_Mat_pt.T, np.dot(-1*B_theta_Mat_pt.T, rho_B_pt.T)).T

					xi_phi_A_pt = np.linalg.solve(A_Mat_pt.T, np.dot(-1*A_phi_Mat_pt.T, rho_A_pt.T)).T
					xi_phi_B_pt = np.linalg.solve(B_Mat_pt.T, np.dot(-1*B_phi_Mat_pt.T, rho_B_pt.T)).T

					# Asign matricies to vectors for use in flat
					self.rho_A_Mats[:, :, quad_pt] = rho_A_pt
					self.rho_B_Mats[:, :, quad_pt] = rho_B_pt

					self.rho_theta_A_Mats[:, :, quad_pt] = rho_theta_A_pt
					self.rho_theta_B_Mats[:, :, quad_pt] = rho_theta_B_pt

					self.rho_phi_A_Mats[:, :, quad_pt] = rho_phi_A_pt
					self.rho_phi_B_Mats[:, :, quad_pt] = rho_phi_B_pt

					self.xi_theta_A_Mats[:, :, quad_pt] = xi_theta_A_pt
					self.xi_theta_B_Mats[:, :, quad_pt] = xi_theta_B_pt

					self.xi_phi_A_Mats[:, :, quad_pt] = xi_phi_A_pt
					self.xi_phi_B_Mats[:, :, quad_pt] = xi_phi_B_pt

					# Calculate VECTORS of Curvature
					#self.K_A_pts[quad_pt] = linalg.det(self.Wein_Map(quad_pt, 'A'))
					#self.K_B_pts[quad_pt] = linalg.det(self.Wein_Map(quad_pt, 'B'))

			# If we know the name, (and we allow pickling) we pickle inv_mats we just generated
			if(Manifold_Constr_Dict['use_manifold_name'] == True and self.Man_Official_Name != [] and self.pickling == True):

				print("pickling Manny Inv Mats for re-use"+"\n")

				# We save matricies as a list:
				To_Pickle_inv_Manny_Mats = np.zeros((3, 3, lbdv.lbdv_quad_pts, 10))

				To_Pickle_inv_Manny_Mats[:,:,:, 0] = self.rho_A_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 1] = self.rho_B_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 2] = self.rho_theta_A_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 3] = self.rho_theta_B_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 4] = self.rho_phi_A_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 5] = self.rho_phi_B_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 6] = self.xi_theta_A_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 7] = self.xi_theta_B_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 8] = self.xi_phi_A_Mats
				To_Pickle_inv_Manny_Mats[:,:,:, 9] = self.xi_phi_B_Mats


				#print("To_Pickle_inv_Manny_Mats.shape = "+str(To_Pickle_inv_Manny_Mats.shape))

				Manny_Info_Dict = {}
				Manny_Info_Dict['Inv_Mats'] = To_Pickle_inv_Manny_Mats
				Manny_Info_Dict['K_A'] = self.K_A_pts
				Manny_Info_Dict['K_B'] = self.K_B_pts

				Manny_Info_Dict['H_A'] = self.H_A_pts
				Manny_Info_Dict['H_B'] = self.H_B_pts

				with open(Manny_Inv_Mats_filepath, 'wb') as f_manny:
					pkl.dump(Manny_Info_Dict, f_manny)
			'''
			else:
				#print("NOT pickling Manny Inv Mats for later re-use"+"\n") # BJG: should add verbose option
			'''

		#If we have already pickled the above matricies, we load them:
		else:

			print("\n"+"loading pickled Manny Inv Mats"+"\n")

			#print("Pickled_Inverse_Mats.shape = "+str(Pickled_Inverse_Mats.shape))

			Pickled_Manny_Info_Dict = []

			with open(Manny_Inv_Mats_filepath, 'rb') as f_manny:
				Pickled_Manny_Info_Dict = pkl.load(f_manny)


			Pickled_inv_Manny_Mats = Pickled_Manny_Info_Dict['Inv_Mats']
			self.K_A_pts = Pickled_Manny_Info_Dict['K_A']
			self.K_B_pts = Pickled_Manny_Info_Dict['K_B']

			self.H_A_pts = Pickled_Manny_Info_Dict['H_A']
			self.H_B_pts = Pickled_Manny_Info_Dict['H_B']

			self.rho_A_Mats, self.rho_B_Mats, self.rho_theta_A_Mats, self.rho_theta_B_Mats, self.rho_phi_A_Mats, self.rho_phi_B_Mats, self.xi_theta_A_Mats, self.xi_theta_B_Mats, self.xi_phi_A_Mats, self.xi_phi_B_Mats = np.squeeze(np.split(Pickled_inv_Manny_Mats, 10, 3))


			'''
			self.rho_A_Mats = Pickled_Inverse_Mats[:,:,:, 0]
			self.rho_B_Mats = Pickled_Inverse_Mats[:,:,:, 1]
			self.rho_theta_A_Mats = Pickled_Inverse_Mats[:,:,:, 2]
			self.rho_theta_B_Mats = Pickled_Inverse_Mats[:,:,:, 3]
			self.rho_phi_A_Mats = Pickled_Inverse_Mats[:,:,:, 4]
			self.rho_phi_B_Mats = Pickled_Inverse_Mats[:,:,:, 5]
			self.xi_theta_A_Mats = Pickled_Inverse_Mats[:,:,:, 6]
			self.xi_theta_B_Mats = Pickled_Inverse_Mats[:,:,:, 7]
			self.xi_phi_A_Mats = Pickled_Inverse_Mats[:,:,:, 8]
			self.xi_phi_B_Mats = Pickled_Inverse_Mats[:,:,:, 9]
			'''

		#print("Done Computing Flat Tensors") # BJG: should add verbose option

		'''
		### From *(2-forms)
		inv_met_fac_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1/self.Manifold.Metric_Factor_Quad_Pt(self.quad_pts, lbdv, 'A'), 0)
		inv_met_fac_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1/self.Manifold.Metric_Factor_Quad_Pt(self.quad_pts, lbdv, 'B'), 0)

		dx_dy_to_dtheta_dphi_A_pts, dx_dz_to_dtheta_dphi_A_pts, dy_dz_to_dtheta_dphi_A_pts = Two_Form_Conv_to_Polar_pt(self.quad_pts, lbdv, self.Manifold, 'A')
		dx_dy_to_dtheta_dphi_B_pts, dx_dz_to_dtheta_dphi_B_pts, dy_dz_to_dtheta_dphi_B_pts = Two_Form_Conv_to_Polar_pt(self.quad_pts, lbdv, self.Manifold, 'B')
		'''

		#print("Manifold Constucted"+"\n") # BJG: should add verbose option


######### Vector (& Derivs) of Manifold, Normals, 2-form Convs ###################################################################################

	def Cart_Coors(quad_pt, Chart):
		if(Chart == 'A'):
			return self.Cart_Coors_A[quad_pt, :]
		if(Chart == 'B'):
			return self.Cart_Coors_B[quad_pt, :]

	def R_Sq_Val(quad_pt, Chart):
		if(Chart == 'A'):
			return self.R_sq_A_Pts[quad_pt, :]
		if(Chart == 'B'):
			return self.R_sq_B_Pts[quad_pt, :]


	def sigma_theta(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Sigma_Theta_A_Pts[quad_pt, :]
		if(Chart == 'B'):
			return self.Sigma_Theta_B_Pts[quad_pt, :]

	def sigma_phi(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Sigma_Phi_A_Pts[quad_pt, :]
		if(Chart == 'B'):
			return self.Sigma_Phi_B_Pts[quad_pt, :]

	def sigma_theta_phi(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Sigma_Theta_Phi_A_Pts[quad_pt, :]
		if(Chart == 'B'):
			return self.Sigma_Theta_Phi_B_Pts[quad_pt, :]

	def sigma_theta_theta(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Sigma_Theta_Theta_A_Pts[quad_pt, :]
		if(Chart == 'B'):
			return self.Sigma_Theta_Theta_B_Pts[quad_pt, :]

	def sigma_phi_phi(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Sigma_Phi_Phi_A_Pts[quad_pt, :]
		if(Chart == 'B'):
			return self.Sigma_Phi_Phi_B_Pts[quad_pt, :]

	def Normal_Dir(self, quad_pt, Chart): # for 2-form conversion, we need un-normalized:
		if(Chart == 'A'):
			return self.Normal_Dirs_A_Pts[quad_pt, :]
		if(Chart == 'B'):
			return self.Normal_Dirs_B_Pts[quad_pt, :]

	def Normal_Vec(self, quad_pt, Chart): # unit normals
		if(Chart == 'A'):
			return self.Normal_Vecs_A_Pts[quad_pt, :]
		if(Chart == 'B'):
			return self.Normal_Vecs_B_Pts[quad_pt, :]


	def Polar_Two_Form_to_Euc_dx_dy(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.dx_dy_A_Vals_From_Polar[quad_pt, :]
		if(Chart == 'B'):
			return self.dx_dy_B_Vals_From_Polar[quad_pt, :]

	def Polar_Two_Form_to_Euc_dx_dz(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.dx_dz_A_Vals_From_Polar[quad_pt, :]
		if(Chart == 'B'):
			return self.dx_dz_B_Vals_From_Polar[quad_pt, :]

	def Polar_Two_Form_to_Euc_dy_dz(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.dy_dz_A_Vals_From_Polar[quad_pt, :]
		if(Chart == 'B'):
			return self.dy_dz_B_Vals_From_Polar[quad_pt, :]


######### Elements of First Fundamental Form and Its Ders, and W: ###################################################################

	def E(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.E_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.E_B_pts[quad_pt, 0]

	def F(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.F_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.F_B_pts[quad_pt, 0]

	def G(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.G_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.G_B_pts[quad_pt, 0]


	## I derivs:
	def E_theta(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.E_theta_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.E_theta_B_pts[quad_pt, 0]

	def E_phi(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.E_phi_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.E_phi_B_pts[quad_pt, 0]

	def F_theta(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.F_theta_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.F_theta_B_pts[quad_pt, 0]

	def F_phi(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.F_phi_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.F_phi_B_pts[quad_pt, 0]

	def G_theta(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.G_theta_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.G_theta_B_pts[quad_pt, 0]

	def G_phi(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.G_phi_A_pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.G_phi_B_pts[quad_pt, 0]

	def W_11(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Wein_11_A_Pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.Wein_11_B_Pts[quad_pt, 0]

	def W_12(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Wein_12_A_Pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.Wein_12_B_Pts[quad_pt, 0]

	def W_21(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Wein_21_A_Pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.Wein_21_B_Pts[quad_pt, 0]

	def W_22(self, quad_pt, Chart):
		if(Chart == 'A'):
			return self.Wein_22_A_Pts[quad_pt, 0]
		if(Chart == 'B'):
			return self.Wein_22_B_Pts[quad_pt, 0]

####### Fundamental Forms and Weingarten Map ##########################################################################


	# First Fundamental Form
	def I_Mat(self, quad_pt, Chart):

		return np.array([[self.E(quad_pt, Chart), self.F(quad_pt, Chart)],[self.F(quad_pt, Chart), self.G(quad_pt, Chart)]])


	# First Fundamental Form
	def I_theta_Mat(self, quad_pt, Chart):

		return np.array([[self.E_theta(quad_pt, Chart), self.F_theta(quad_pt, Chart)],[self.F_theta(quad_pt, Chart), self.G_theta(quad_pt, Chart)]])


	# First Fundamental Form
	def I_phi_Mat(self, quad_pt, Chart):

		return np.array([[self.E_phi(quad_pt, Chart), self.F_phi(quad_pt, Chart)],[self.F_phi(quad_pt, Chart), self.G_phi(quad_pt, Chart)]])



##### Weingarten Map/ Change of BASIS #################################################################################

	# Weingarten Map, derivative of (inward) normal vector
	def Wein_Map(self, quad_pt, Chart):

		return np.array([[self.W_11(quad_pt, Chart), self.W_12(quad_pt, Chart)], [self.W_21(quad_pt, Chart), self.W_22(quad_pt, Chart)]])


	# Derivative of inward normal, wrt e_theta
	def Normal_vec_dtheta(self, quad_pt, Chart):

		W_map = self.Wein_Map(quad_pt, Chart)
		sigma_theta_C_pt = self.sigma_theta(quad_pt, Chart)
		sigma_phi_C_pt = self.sigma_phi(quad_pt, Chart)

	        # For Scalar Case, we use usual vectorization:
		if(isscalar(quad_pt)):
			return -1.*(W_map[0,0]*sigma_theta_C_pt + W_map[1,0]*sigma_phi_C_pt)

 		# For multiple quad pts, we use einstein sumation to output vector of solutions at each point:
		else:
			#print("W_map[0,0].shape = "+str( W_map[0,0].shape ))
			#print("sigma_theta_C_pt.shape = "+str( sigma_theta_C_pt.shape ))
			return -1.*( W_map[0,0].reshape( len(quad_pt), 1)*sigma_theta_C_pt + W_map[1,0].reshape( len(quad_pt), 1)*sigma_phi_C_pt )


	# Derivative of inward normal, wrt e_phi
	def Normal_vec_dphi(self, quad_pt, Chart):

		W_map = self.Wein_Map(quad_pt, Chart)
		sigma_theta_C_pt = self.sigma_theta(quad_pt, Chart)
		sigma_phi_C_pt = self.sigma_phi(quad_pt, Chart)

		 # For Scalar Case, we use usual vectorization:
		if(isscalar(quad_pt)):
			return -1.*(W_map[0,1]*sigma_theta_C_pt + W_map[1,1]*sigma_phi_C_pt)

		# For multiple quad pts, we use einstein sumation to output vector of solutions at each point:
		else:
			return -1.*( W_map[0,1].reshape( len(quad_pt), 1)*sigma_theta_C_pt + W_map[1,1].reshape( len(quad_pt), 1)*sigma_phi_C_pt )


	# Matrix Used to Convert to Polar
	def Change_Basis_Mat(self, quad_pt, Chart):

		Basis_mat = zeros(( 3, 3 ))
		'''
		if(Chart == 'A'):
			Basis_mat[:, 0] = self.Sigma_Theta_A_Pts[quad_pt, :].T #self.sigma_theta(theta_pt, phi_pt, Chart).T
			Basis_mat[:, 1] = self.Sigma_Phi_A_Pts[quad_pt, :].T #self.sigma_phi(theta_pt, phi_pt, Chart).T
			Basis_mat[:, 2] = self.Normal_Vecs_A_Pts[quad_pt, :].T #self.Normal_Vec(theta_pt, phi_pt, Chart).T

		if(Chart == 'B'):
			Basis_mat[:, 0] = self.Sigma_Theta_B_Pts[quad_pt, :].T
			Basis_mat[:, 1] = self.Sigma_Phi_B_Pts[quad_pt, :].T
			Basis_mat[:, 2] = self.Normal_Vecs_B_Pts[quad_pt, :].T
		'''

		Basis_mat[:, 0] = self.sigma_theta(quad_pt, Chart).T
		Basis_mat[:, 1] = self.sigma_phi(quad_pt, Chart).T
		Basis_mat[:, 2] = self.Normal_Vec(quad_pt, Chart).T

		return Basis_mat


	# Entry-wise deriv of Basis mat, dtheta
	def dChange_Basis_Mat_theta(self, quad_pt, Chart):

		dBasis_mat_theta = zeros(( 3, 3 ))

		dBasis_mat_theta[:, 0] = self.sigma_theta_theta(quad_pt, Chart).T
		dBasis_mat_theta[:, 1] = self.sigma_theta_phi(quad_pt, Chart).T
		dBasis_mat_theta[:, 2] = self.Normal_vec_dtheta(quad_pt, Chart).T

		return dBasis_mat_theta


	# Entry-wise deriv of Basis mat, dphi
	def dChange_Basis_Mat_phi(self, quad_pt, Chart):

		dBasis_mat_phi = zeros(( 3, 3 ))

		dBasis_mat_phi[:, 0] = self.sigma_theta_phi(quad_pt, Chart).T
		dBasis_mat_phi[:, 1] = self.sigma_phi_phi(quad_pt, Chart).T
		dBasis_mat_phi[:, 2] = self.Normal_vec_dphi(quad_pt, Chart).T

		return dBasis_mat_phi

	# I mat is embedded:
	def G_Mat(self, quad_pt, Chart):

		G_mat = zeros(( 3, 3 ))
		G_mat[0:2, 0:2] = self.I_Mat(quad_pt, Chart)

		return G_mat


	def dG_Mat_theta(self, quad_pt, Chart):

		G_theta_mat = zeros(( 3, 3 ))
		G_theta_mat[0:2, 0:2] = self.I_theta_Mat(quad_pt, Chart)

		return G_theta_mat


	def dG_Mat_phi(self, quad_pt, Chart):

		G_phi_mat = zeros(( 3, 3 ))
		G_phi_mat[0:2, 0:2] = self.I_phi_Mat(quad_pt, Chart)

		return G_phi_mat
