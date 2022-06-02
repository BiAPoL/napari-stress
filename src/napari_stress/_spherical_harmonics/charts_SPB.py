# From https://github.com/campaslab/STRESS
#from numpy  import *
import numpy as np
from scipy  import *
import mpmath
import cmath
from scipy.special import sph_harm

#####  Size of Charts: #######
Chart_Min_Polar = np.pi/4 # Minimum Phi Val in Chart
Chart_Max_Polar = 3*np.pi/4 # Maximum Phi Val in Chart

eta_Min_Polar = np.pi/8 # Min 1 Value in Cutoff fn
eta_Max_Polar = 7*np.pi/8 # Max 1 Value in Cutoff fn
eta_Polar_Decay = np.pi/8 # decay interval in cutoff fn
##############################


def Domain(Theta, Phi): #Which coordinates to use
	if(Chart_Min_Polar <= Phi and Phi <= Chart_Max_Polar): #Use Coordinates A

		Bar_Coors = Coor_A_To_B(Theta, Phi)
		Phi_Bar = Bar_Coors[1]

		if(Chart_Min_Polar <= Phi_Bar and Phi_Bar <= Chart_Max_Polar):   #Either could be used
			return 0

		else:	#Only A Can be used
			return 1


	else: #Use Coordinates B, Not A
		return -1

def Domain_Unaffected(Theta, Phi): #Where fns in each chart arent affected by eta fns
	if(eta_Min_Polar <= Phi and Phi <= eta_Max_Polar): #Use Coordinates A

		Bar_Coors = Coor_A_To_B(Theta, Phi)
		Phi_Bar = Bar_Coors[1]

		if(eta_Min_Polar <= Phi_Bar and Phi_Bar <= eta_Max_Polar):   #Either could be used
			return 0

		else:	#Only A Can be used
			return 1


	else: #Use Coordinates B, Not A
		return -1

def Bump_Fn(Eval_Pt, Max_Val, Start, Length): #can be evaluated on [start, start+Length)
	X_2 = ((Eval_Pt-Start)/Length)**2
	return (Max_Val*e)*e**(-1/(1-X_2))
	# B(E) = Me*e^(-1/(1-((E-S)/L)^2))

	###!! Linear Cutoff !!###
	#return Max_Val*(1-(Eval_Pt - Start)/Length)

	###!! 0-Cutoff !!###
	#return 0

def eta_A(func, Theta, Phi): #Cutoff fn for phi in [eta_Min_Polar, eta_Max_Polar]
	if(eta_Min_Polar <= Phi and Phi <=  eta_Max_Polar):
		return func(Theta, Phi)

	elif(eta_Max_Polar < Phi and Phi < eta_Max_Polar + eta_Polar_Decay):
		return Bump_Fn(Phi, 1, eta_Max_Polar, eta_Polar_Decay)*func(Theta, Phi)

	elif(eta_Min_Polar - eta_Polar_Decay < Phi and Phi < eta_Min_Polar):
		return Bump_Fn(Phi, 1, eta_Min_Polar, eta_Polar_Decay)*func(Theta, Phi)
		#Bump is SYMETRIC around start
	else:
		return 0 # if Phi=< eta_Min_Polar - eta_Polar_Decay or Phi=> eta_Max_Polar + eta_Polar_Decay


def eta_A_const(const, Theta, Phi): #Cutoff fn for phi in [eta_Min_Polar, eta_Max_Polar], where const = f(theta, phi)
	if(eta_Min_Polar <= Phi and Phi <=  eta_Max_Polar):
		return const

	elif(eta_Max_Polar < Phi and Phi < eta_Max_Polar + eta_Polar_Decay):
		return Bump_Fn(Phi, 1, eta_Max_Polar, eta_Polar_Decay)*const

	elif(eta_Min_Polar - eta_Polar_Decay < Phi and Phi < eta_Min_Polar):
		return Bump_Fn(Phi, 1, eta_Min_Polar, eta_Polar_Decay)*const
		#Bump is SYMETRIC around start
	else:
		return 0 # if Phi=< eta_Min_Polar - eta_Polar_Decay or Phi=> eta_Max_Polar + eta_Polar_Decay

def eta_B(func, Theta_Bar, Phi_Bar): #Cutoff fn for Phi_Bar in [eta_Min_Polar, eta_Max_Polar]

	#A_Coors = Coor_B_To_A(Theta_Bar, Phi_Bar)
	#print("Equiv to (Theta, Phi) = "+'('+str(A_Coors[0]/pi)+'pi, '+str(A_Coors[1]/pi)+'pi)')

	#func needs to be COUNTER rotated back to phi, theta, to be consistent
	def func_counter_rot(theta_bar, phi_bar):
		Orig_Coors = Coor_B_To_A(theta_bar, phi_bar)
		theta = Orig_Coors[0]
		phi = Orig_Coors[1]
		return func(theta, phi)

	return eta_A(func_counter_rot, Theta_Bar, Phi_Bar)


# NOTE: (Theta, Phi) -> (Theta_Bar, Phi_Bar) by rotating the north pole (0,0) to the east pole (0,pi/2)
def Coor_A_To_B(Theta, Phi): #Converts from A to B

	X1 = np.cos(Theta)*np.sin(Phi)
	Y1 = np.sin(Theta)*np.sin(Phi)
	Z1 = np.cos(Phi)

	# print("X1= "+str(X1))
	# print("Y1= "+str(Y1))
	# print("Z1= "+str(Z1))

	XYZ1 = [X1, Y1, Z1]

	#How Coordinate Axes are rotated
	X_Rot = -Z1
	Y_Rot = Y1
	Z_Rot = X1


	Bar_Coors = Cart_To_Coor_A(X_Rot, Y_Rot, Z_Rot)

	# Theta_Bar = Bar_Coors[:, 0]
	# Phi_Bar = Bar_Coors[:, 1]

	# XYZ2 = [cos(Phi_Bar) , sin(Theta_Bar)*sin(Phi_Bar), -1*cos(Theta_Bar)*sin(Phi_Bar)]

	#print("Orig Cart. = "+str(XYZ1))
	#print("New Cart. = "+str(XYZ2))

	return Bar_Coors


def Coor_B_To_A(Theta_Bar, Phi_Bar): #Converts from B back to A

	#X2 = cos(Theta_Bar)*sin(Phi_Bar)
	#Y2 = sin(Theta_Bar)*sin(Phi_Bar)
	#Z2 = cos(Phi_Bar)

	#XYZ2 = [X2, Y2, Z2]

	#How Coordinate Axes are rotated
	#X_Inv_Rot = Z2
	#Y_Inv_Rot = Y2
	#Z_Inv_Rot = -X2

	#A_Coors = Cart_To_Coor_A(X_Inv_Rot, Y_Inv_Rot, Z_Inv_Rot)

	X_pt = np.cos(Phi_Bar)
	Y_pt = np.sin(Theta_Bar)*np.sin(Phi_Bar)
	Z_pt = -1*np.cos(Theta_Bar)*np.sin(Phi_Bar)

	A_Coors = Cart_To_Coor_A(X_pt, Y_pt, Z_pt)


	# Theta = A_Coors[:, 0]
	# Phi = A_Coors[:, 1]

	# XYZ1 = [cos(Phi_Bar) , sin(Theta_Bar)*sin(Phi_Bar), -1*cos(Theta_Bar)*sin(Phi_Bar)]


	return A_Coors


# Uses F(theta, phi) to find F_{Rot}(theta_bar, phi_bar)
def Rotate_Fn(func, Theta_Bar, Phi_Bar):

	Theta, Phi = Coor_B_To_A(Theta_Bar, Phi_Bar)
	return func(Theta, Phi)

#(x,y,z) -> (theta, phi)
def Cart_To_Coor_A(x,y,z):

	r = np.sqrt(pow(x,2) + pow(y,2) + pow(z,2))

	##Calculate Theta

	theta = np.arctan2(y,x)  #Actually gives us unit circle angle for theta < pi, theta-2*pi for theta > pi

	theta = np.where(theta >= 0, theta, theta + 2*np.pi) # return: theta (where theta >= 0), return: theta + 2*pi (where theta < 0)

	phi = np.arccos(np.divide(z,r)) #numpy version has range [0, pi]

	return [theta, phi]


#(x,y,z) -> (theta_bar, phi_bar)
def Cart_To_Coor_B(x,y,z):

	r = np.sqrt(pow(x,2) + pow(y,2) + pow(z,2))

	##Calculate Theta_Bar

	theta_bar = np.arctan2(y,-1*z)  #Actually gives us unit circle angle for theta_bar < pi, theta_bar-2*pi for theta_bar > pi

	theta_bar = np.where(theta_bar >= 0, theta_bar, theta_bar + 2*np.pi) # return: theta_bar (where theta_bar >= 0), return: theta_bar + 2*pi (where theta_bar < 0)

	phi_bar = np.arccos(np.divide(x,r)) #numpy version has range [0, pi]

	return [theta_bar, phi_bar]


#######################################################################################################################
# Put This Here For Convenience:
#######################################################################################################################

#BJG: This is already in manifold_SPB.py

'''
def Manifold_Fn_Def(theta, phi, r_0, Shape_Name):

	if(Shape_Name == "S2"):
		return 1


	elif(Shape_Name  == "Chew_Toy"):
		return 1 + r_0*np.sin(3*phi)*np.cos(theta)


	elif(Shape_Name  == "Dog_Shit"):
		return 1 + r_0*np.sin(7*phi)*np.cos(theta)


	elif(Shape_Name == "Gen_Pill"):
		return (1 + r_0)/np.sqrt(1 + ((1 + r_0)**2 - 1)*np.sin(phi)**2)


	elif(Shape_Name == "Divet"):
		if(isscalar(phi)):
			if(np.cos(phi) > .9):
				return 1 - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1)

			else:
				return 1

		else:
			# Vectorized for manifold functions
			return np.where(np.cos(phi) > .9,  1 - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1), 1)


	elif(Shape_Name == "Double_Divet"):
		if(isscalar(phi)):
			if(np.cos(phi) > .9):
				return 1 - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1)

			elif(np.sin(phi)*np.cos(theta) > .9):
				return 1 - r_0*np.exp(-1*(.19)/((np.sin(phi)*np.cos(theta))**2 - (.9)**2) + 1)

			else:
				return 1

		else:
			# Vectorized for manifold functions
			return np.where(np.cos(phi) > .9,  1 - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1),
				np.where(np.sin(phi)*np.cos(theta) > .9, 1 - r_0*np.exp(-1*(.19)/((np.sin(phi)*np.cos(theta))**2 - (.9)**2) + 1), 1))


	elif(Shape_Name == "Divets_Around_Pimple"):
		if(isscalar(phi)):
			if(abs(np.cos(phi)) > .9):
				return 1 - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1)

			elif(np.sin(phi)*np.cos(theta) > .9):
				return 1 + r_0*np.exp(-1*(.19)/((np.sin(phi)*np.cos(theta))**2 - (.9)**2) + 1)

			else:
				return 1

		else:
			# Vectorized for manifold functions
			return np.where(abs(np.cos(phi)) > .9,  1 - r_0*np.exp(-1*(.19)/(np.cos(phi)**2 - (.9)**2) + 1),
				np.where(np.sin(phi)*np.cos(theta) > .9, 1 + r_0*np.exp(-1*(.19)/((np.sin(phi)*np.cos(theta))**2 - (.9)**2) + 1), 1))


	elif(Shape_Name == "Hour_Glass"):
		# Note: (Previously) NOT A SPHERE for R_0 = 0
		return 1 - (6*r_0)*np.exp(-1.0/(1-np.cos(phi)**2)) # previously 2 + r_0, for r_0 = .4


	else:
		print("\n"+"ERROR: Manifold Shape Name Not Recognized"+"\n")
'''
