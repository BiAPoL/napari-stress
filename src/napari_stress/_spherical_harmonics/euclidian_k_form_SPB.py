# From https://github.com/campaslab/STRESS
#!!! We Define K_Forms in Euclidean Space to improve convergence

import numpy as np
from .._spherical_harmonics import sph_func_SPB as sph_f



# Create Array of zeros to store vals at quad pts
def zero_quad_array(Q_val):
	return np.zeros(( Q_val, 1 ))


# Return list of quad vals at each quad pt, given func
def get_quadrature_points_from_function(Func, lbdv):

	Extracted_Quad_vals = Func(lbdv.theta_pts, lbdv.phi_pts)
	return Extracted_Quad_vals


# Return list of quad vals at each quad pt, given SPH_Func (CHART A order)
def get_quadrature_points_from_sh_function(SPH_Func, lbdv, Chart):

	Extracted_Quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)

	quad_pts = range(lbdv.lbdv_quad_pts)

	if(Chart == 'A'):
		Extracted_Quad_vals = SPH_Func.Eval_SPH_Coef_Mat(quad_pts, lbdv)

	if(Chart == 'B'): # The list needs to be in
		Extracted_Quad_vals = SPH_Func.Eval_SPH_Coef_Mat(lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pts), lbdv)

	return Extracted_Quad_vals


# Return list of quad vals at each quad pt within Chart (CHART A order)
def Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(sph_func, lbdv, Chart):

	Extracted_dPhi_Quad_Vals = zero_quad_array(lbdv.lbdv_quad_pts)

	quad_pts = range(lbdv.lbdv_quad_pts)

	if(Chart == 'A'):

		return np.where(lbdv.Chart_of_Quad_Pts > 0, sph_func.Eval_SPH_Der_Phi_Coef(quad_pts, lbdv), 0)

	if(Chart == 'B'):

		quad_pts_rot = lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pts)
		return np.where(lbdv.Chart_of_Quad_Pts[quad_pts_rot] > 0, sph_func.Eval_SPH_Der_Phi_Coef(quad_pts_rot, lbdv), 0 )


# Return list of quad vals at each quad pt within Chart (CHART A order)
def Extract_dPhi_Phi_Quad_Pt_Vals_From_SPH_Fn(sph_func, lbdv, Chart):

	Extracted_dPhi_Phi_Quad_Vals = zero_quad_array(lbdv.lbdv_quad_pts)

	quad_pts = range(lbdv.lbdv_quad_pts)

	if(Chart == 'A'):

		return np.where(lbdv.Chart_of_Quad_Pts > 0, sph_func.Eval_SPH_Der_Phi_Phi_Coef(quad_pts, lbdv), 0)

	if(Chart == 'B'):

		quad_pts_rot = lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pts)
		return np.where(lbdv.Chart_of_Quad_Pts[quad_pts_rot] > 0, sph_func.Eval_SPH_Der_Phi_Phi_Coef(quad_pts_rot, lbdv), 0 )


# Combines info from each chart to give good vals at every point
def Combine_Chart_Quad_Vals(Quad_Vals_A, Quad_Vals_B, lbdv):

	quad_pts = range(lbdv.lbdv_quad_pts)
	return np.where(lbdv.Chart_of_Quad_Pts > 0, Quad_Vals_A, Quad_Vals_B[lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pts)])


# Combines K_A and K_B into Chart A vector
def Combine_Manny_Gauss_Curvatures(Manny, lbdv, verbose=False):

	K_cominbed_A_pts = Combine_Chart_Quad_Vals(Manny.K_A_pts, Manny.K_B_pts, lbdv)

	if(verbose == True):
		print("test of int on manny of K (in euc_kf) = "+str((Integral_on_Manny(K_cominbed_A_pts, Manny, lbdv)- 4*np.pi)/(4*np.pi)) + " (Rel Error)" )

	return K_cominbed_A_pts


# Converts 1-form alpha, to vector field at quad pts:
def Sharp(dtheta_A_vals, dphi_A_vals, dtheta_B_vals, dphi_B_vals, p_deg, lbdv, Manny):

	# alpha_i = (dtheta_vals_i)dtheta + (d_phi_vals_i)dphi

	# Store x,y,z vals of sharped vector field at each quad_pt
	sharped_x_vals_pts_A = zero_quad_array(lbdv.lbdv_quad_pts)
	sharped_y_vals_pts_A = zero_quad_array(lbdv.lbdv_quad_pts)
	sharped_z_vals_pts_A = zero_quad_array(lbdv.lbdv_quad_pts)

	sharped_x_vals_pts_B = zero_quad_array(lbdv.lbdv_quad_pts)
	sharped_y_vals_pts_B = zero_quad_array(lbdv.lbdv_quad_pts)
	sharped_z_vals_pts_B = zero_quad_array(lbdv.lbdv_quad_pts)

	# Use appropriate Chart for computing sharp
	quad_pts = range(lbdv.lbdv_quad_pts)

	# Values of 1-form:
	d_theta_pts_A = np.where(lbdv.Chart_of_Quad_Pts > 0, dtheta_A_vals[quad_pts], 0)
	d_phi_pts_A = np.where(lbdv.Chart_of_Quad_Pts > 0, dphi_A_vals[quad_pts], 0)

	d_theta_pts_B = np.where(lbdv.Chart_of_Quad_Pts > 0, dtheta_B_vals[quad_pts], 0)
	d_phi_pts_B = np.where(lbdv.Chart_of_Quad_Pts > 0, dphi_B_vals[quad_pts], 0)

	# Manifold Data Needed:

	#theta_pts = lbdv.theta_pts
	#phi_pts = lbdv.phi_pts

	g_inv_theta_theta_A_pts = Manny.g_Inv_Theta_Theta_A_Pts #Manny.g_inv_theta_theta(theta_pts, phi_pts, 'A')
	g_inv_theta_phi_A_pts = Manny.g_Inv_Theta_Phi_A_Pts #Manny.g_inv_theta_phi(theta_pts, phi_pts, 'A')
	g_inv_phi_phi_A_pts = Manny.g_Inv_Phi_Phi_A_Pts #Manny.g_inv_phi_phi(theta_pts, phi_pts, 'A')

	g_inv_theta_theta_B_pts = Manny.g_Inv_Theta_Theta_B_Pts #Manny.g_inv_theta_theta(theta_pts, phi_pts, 'B')
	g_inv_theta_phi_B_pts = Manny.g_Inv_Theta_Phi_B_Pts #Manny.g_inv_theta_phi(theta_pts, phi_pts, 'B')
	g_inv_phi_phi_B_pts = Manny.g_Inv_Phi_Phi_B_Pts #Manny.g_inv_phi_phi(theta_pts, phi_pts, 'B')

	sigma_theta_A_pts = Manny.Sigma_Theta_A_Pts #np.squeeze(Manny.sigma_theta(theta_pts, phi_pts, 'A'))
	sigma_theta_B_pts = Manny.Sigma_Theta_B_Pts #np.squeeze(Manny.sigma_theta(theta_pts, phi_pts, 'B') )

	sigma_phi_A_pts = Manny.Sigma_Phi_A_Pts #np.squeeze(Manny.sigma_phi(theta_pts, phi_pts, 'A'))
	sigma_phi_B_pts = Manny.Sigma_Phi_B_Pts #np.squeeze(Manny.sigma_phi(theta_pts, phi_pts, 'B'))

	# Sharp in (theta, phi) Coors:
	alpha_sharp_theta_A_pts = np.multiply(d_theta_pts_A, g_inv_theta_theta_A_pts) + np.multiply(d_phi_pts_A, g_inv_theta_phi_A_pts)
	alpha_sharp_phi_A_pts = np.multiply(d_theta_pts_A, g_inv_theta_phi_A_pts) + np.multiply(d_phi_pts_A, g_inv_phi_phi_A_pts)

	alpha_sharp_theta_B_pts = np.multiply(d_theta_pts_B, g_inv_theta_theta_B_pts) + np.multiply(d_phi_pts_B, g_inv_theta_phi_B_pts)
	alpha_sharp_phi_B_pts = np.multiply(d_theta_pts_B, g_inv_theta_phi_B_pts) + np.multiply(d_phi_pts_B, g_inv_phi_phi_B_pts)

	# Convert Sharped Result to Euclidean Coors:
	alpha_sharp_A_pts = np.multiply(alpha_sharp_theta_A_pts, sigma_theta_A_pts) + np.multiply(alpha_sharp_phi_A_pts, sigma_phi_A_pts)
	alpha_sharp_B_pts = np.multiply(alpha_sharp_theta_B_pts, sigma_theta_B_pts) + np.multiply(alpha_sharp_phi_B_pts, sigma_phi_B_pts)

	sharped_x_vals_pts_A, sharped_y_vals_pts_A, sharped_z_vals_pts_A = np.hsplit(alpha_sharp_A_pts, 3)
	sharped_x_vals_pts_B, sharped_y_vals_pts_B, sharped_z_vals_pts_B = np.hsplit(alpha_sharp_B_pts, 3)


	sharped_x_vals_pts_Used = Combine_Chart_Quad_Vals(sharped_x_vals_pts_A, sharped_x_vals_pts_B, lbdv)
	sharped_y_vals_pts_Used = Combine_Chart_Quad_Vals(sharped_y_vals_pts_A, sharped_y_vals_pts_B, lbdv)
	sharped_z_vals_pts_Used = Combine_Chart_Quad_Vals(sharped_z_vals_pts_A, sharped_z_vals_pts_B, lbdv)

	# For 1-form constuction
	Sharped_Quad_Vals = np.hstack(( sharped_x_vals_pts_Used, sharped_y_vals_pts_Used, sharped_z_vals_pts_Used ))

	return euc_k_form(1, lbdv.lbdv_quad_pts, p_deg, Manny, Sharped_Quad_Vals)


# Converts 2-forms from Polar to Euclidean at a Quad Pt
def Two_Form_Conv_to_Euc_pt(quad_pt, lbdv, Chart, Manny):

	# BJG: NEED TO CHANGE TO TO SPB FRAME:
	'''
	x_pt, y_pt, z_pt = Manny.Cart_Coor(quad_pt, lbdv, Chart)
	r_sq_pt = Manny.R_Sq_Val(quad_pt, Chart)

	if(Chart == 'A'):

		denom_A = (r_pt**2)*np.sqrt(x_pt**2 + y_pt**2)

		dx_dy_comp = -z_pt/denom_A
		dx_dz_comp = y_pt/denom_A
		dy_dz_comp = -x_pt/denom_A

		return dx_dy_comp, dx_dz_comp, dy_dz_comp

	if(Chart == 'B'):

		denom_B = (r_pt**2)*np.sqrt(y_pt**2 + z_pt**2)

		dx_dy_comp = -z_pt/denom_B
		dx_dz_comp = y_pt/denom_B
		dy_dz_comp = -x_pt/denom_B

		return dx_dy_comp, dx_dz_comp, dy_dz_comp
	'''

	dx_dy_comp = Polar_Two_Form_to_Euc_dx_dy(quad_pt, Chart)
	dx_dz_comp = Polar_Two_Form_to_Euc_dx_dz(quad_pt, Chart)
	dy_dz_comp = Polar_Two_Form_to_Euc_dy_dz(quad_pt, Chart)

	return dx_dy_comp, dx_dz_comp, dy_dz_comp


# Converts 2-forms from Euclidean to Polar at a Quad Pt
def Two_Form_Conv_to_Polar_pt(quad_pt, lbdv, Manny, Chart):

	# BJG: DOES NOT NEED TO CHANGE FOR SPB FRAME:

	theta_pt = lbdv.theta_pts[quad_pt]
	phi_pt = lbdv.phi_pts[quad_pt]

	dy_dz_comp = []
	dx_dz_comp = []
	dx_dy_comp = []

	if(np.isscalar(quad_pt) == True):

		theta_pt = np.asscalar(theta_pt)
		phi_pt = np.asscalar(phi_pt)

		#Cross_of_Basis = Manny.Normal_Dir(theta_pt, phi_pt, Chart)

		# BJG: New Change:

		Cross_of_Basis = Manny.Normal_Dir(quad_pt, Chart)

		dy_dz_comp = Cross_of_Basis[0]
		dx_dz_comp = -1*Cross_of_Basis[1]
		dx_dy_comp = Cross_of_Basis[2]
	else:

		#Cross_of_Basis = Manny.Normal_Dir_Quad_Pts(lbdv, Chart) #Manny.Normal_Dir(theta_pt, phi_pt, Chart)

		# BJG: New Change:

		Cross_of_Basis = Manny.Normal_Dir(quad_pt, Chart)

		#print("quad_pt = "+str(quad_pt)+", Cross_of_Basis.shape = "+str(Cross_of_Basis.shape)+", Cross_of_Basis = "+str(Cross_of_Basis))

		dy_dz_comp, neg_dx_dz_comp, dx_dy_comp = np.hsplit(Cross_of_Basis, 3)
		dx_dz_comp = -1.*neg_dx_dz_comp

		#print("dy_dz_comp = "+str(dy_dz_comp)+"\n"+"dx_dz_comp = "+str(dx_dz_comp)+"\n"+"dx_dy_comp = "+str(dx_dy_comp)+"\n")

		#dy_dz_comp = Cross_of_Basis[:, :, 0].T
		#dx_dz_comp = -1*Cross_of_Basis[:, :, 1].T
		#dx_dy_comp = Cross_of_Basis[:, :, 2].T

	return dx_dy_comp, dx_dz_comp, dy_dz_comp


# computes (-*d) directly;
# Takes in 1-form (in LOCAL coordinates) -> 0-Form
def Gen_Curl_1(theta_C_A_vals, phi_C_A_vals, theta_C_B_vals, phi_C_B_vals, p_deg, lbdv, Manny):

	# Project into SPH within Charts:
	theta_C_A_SPH_Fn = sph_f.Faster_Double_Proj(theta_C_A_vals, p_deg, lbdv)
	phi_C_A_SPH_Fn = sph_f.Faster_Double_Proj(phi_C_A_vals, p_deg, lbdv)

	theta_C_B_SPH_Fn = sph_f.Faster_Double_Proj(theta_C_B_vals, p_deg, lbdv)
	phi_C_B_SPH_Fn = sph_f.Faster_Double_Proj(phi_C_B_vals, p_deg, lbdv)

	# take (local) theta derivative of (local) phi component
	dTheta_phi_C_A_SPH_Fn = phi_C_A_SPH_Fn.Quick_Theta_Der()
	dTheta_phi_C_B_SPH_Fn = phi_C_B_SPH_Fn.Quick_Theta_Der()

	# project this azmuthal derivative into local chart ('A' because we dont rotate)
	dTheta_phi_C_A_quad_pt_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(dTheta_phi_C_A_SPH_Fn, lbdv, 'A')
	dTheta_phi_C_B_quad_pt_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(dTheta_phi_C_B_SPH_Fn, lbdv, 'A')

	# Compute (local) phi derivative of (local) theta component:
	dPhi_theta_C_A_quad_pt_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(theta_C_A_SPH_Fn, lbdv, 'A')
	dPhi_theta_C_B_quad_pt_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(theta_C_B_SPH_Fn, lbdv, 'A')

	# inv metric factor, at each point, within Chart:
	q_val = lbdv.lbdv_quad_pts
	quad_pts = range(q_val)

	inv_met_fac_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1./Manny.Metric_Factor_A_pts, 0)
	inv_met_fac_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1./Manny.Metric_Factor_B_pts, 0)

	# Combine fields to produce gen_curl_1 in each chart:
	gen_curl_1_A_pts =  np.multiply(dPhi_theta_C_A_quad_pt_vals - dTheta_phi_C_A_quad_pt_vals, inv_met_fac_A_pts)
	gen_curl_1_B_pts =  np.multiply(dPhi_theta_C_B_quad_pt_vals - dTheta_phi_C_B_quad_pt_vals, inv_met_fac_B_pts)

	gen_curl_1_Used_pts = Combine_Chart_Quad_Vals(gen_curl_1_A_pts, gen_curl_1_B_pts , lbdv)

	return euc_k_form(0, q_val, p_deg, Manny, gen_curl_1_Used_pts)


# takes in quadrature points and weights them by metric:
def Integral_on_Manny_Eta_Z(vals_at_quad_pts, Manny, lbdv): #Orriginal Version

	met_fac_over_sin_phi_pts_A = Manny.Metric_Factor_A_over_sin_phi_pts
	met_fac_over_sin_phi_pts_B = Manny.Metric_Factor_B_over_sin_phi_bar_pts

	num_quad_pts = lbdv.lbdv_quad_pts

	# We integrate in local coordinates
	eta_Z_Chart_A = np.zeros(( num_quad_pts, 1 ))
	one_minus_eta_Z_Chart_B = np.zeros(( num_quad_pts, 1 ))

	# We need to rotate integrand into chart B
	rotated_vals_at_quad_pts = np.zeros(( num_quad_pts, 1 ))

	for quad_pt in range(num_quad_pts):

		eta_Z_pt = lbdv.eta_z(quad_pt)

		eta_Z_Chart_A[quad_pt, 0] = eta_Z_pt

		quad_pt_rot = lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pt)
		one_minus_eta_Z_Chart_B[quad_pt_rot, 0] = 1- eta_Z_pt
		rotated_vals_at_quad_pts[quad_pt_rot, 0] = vals_at_quad_pts[quad_pt, 0]

	Chart_A_integrand = np.multiply(np.multiply(met_fac_over_sin_phi_pts_A, eta_Z_Chart_A), vals_at_quad_pts)
	Chart_B_integrand = np.multiply(np.multiply(met_fac_over_sin_phi_pts_B, one_minus_eta_Z_Chart_B), rotated_vals_at_quad_pts)

	Int_On_Manny = sph_f.S2_Integral(Chart_A_integrand, lbdv) + sph_f.S2_Integral(Chart_B_integrand, lbdv)

	return Int_On_Manny


def Integral_on_Manny(vals_at_quad_pts, Manny, lbdv): # New Version

	vals_at_quad_pts = vals_at_quad_pts.reshape(len(vals_at_quad_pts),1) # ASSERT THIS IS THE RIGHT SIZE

	met_fac_over_sin_phi_pts_A = Manny.Metric_Factor_A_over_sin_phi_pts
	met_fac_over_sin_phi_pts_B = Manny.Metric_Factor_B_over_sin_phi_bar_pts

	num_quad_pts = lbdv.lbdv_quad_pts

	# We need to rotate integrand into chart B
	rotated_vals_at_quad_pts = np.zeros(( num_quad_pts, 1 ))


	quad_pts = range(num_quad_pts)
	quad_pts_rot = lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pts) #lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pts)
	rotated_vals_at_quad_pts = vals_at_quad_pts[quad_pts_rot]


	Jacobian_Integrand = np.multiply(Combine_Chart_Quad_Vals(met_fac_over_sin_phi_pts_A, met_fac_over_sin_phi_pts_B, lbdv), vals_at_quad_pts) #Combine_Chart_Quad_Vals(np.multiply(met_fac_over_sin_phi_pts_A, vals_at_quad_pts), np.multiply(met_fac_over_sin_phi_pts_B, rotated_vals_at_quad_pts), lbdv)

	Jacobian_Int_On_Manny = sph_f.S2_Integral(Jacobian_Integrand, lbdv)

	return Jacobian_Int_On_Manny

# get metric factor scaling of weights from S2 to Manny:
def lebedev_quad_adj_Manny(Manny, lbdv):
	met_fac_over_sin_phi_pts_A = Manny.Metric_Factor_A_over_sin_phi_pts
	met_fac_over_sin_phi_pts_B = Manny.Metric_Factor_B_over_sin_phi_bar_pts
	return Combine_Chart_Quad_Vals(met_fac_over_sin_phi_pts_A, met_fac_over_sin_phi_pts_B, lbdv)


# Inputs quad vals for f, f_approx, integrates on MANNY:
def Lp_Rel_Error_At_Quad_On_Manny(approx_f_vals, f_vals, lbdv, p, Manny): #Assumes f NOT 0

	Lp_Err = 0 # ||self - f||_p
	Lp_f = 0  # || f ||_p

	pointwise_errs_to_the_p = abs((approx_f_vals - f_vals)**p)

	Lp_Err_Manny = Integral_on_Manny(pointwise_errs_to_the_p, Manny, lbdv)
	Lp_f_Manny = Integral_on_Manny(abs(f_vals)**p, Manny, lbdv)

	return 	(Lp_Err_Manny/Lp_f_Manny)**(1./p) #||f_approx - f||_p / || f ||_p


# Get rid of normal components in a elegant way:
def Tangent_Projection(Vectors_at_Quad, lbdv, Manny):

	num_quad_pts = lbdv.lbdv_quad_pts
	Tangent_Vecs_at_Quad = np.zeros(( num_quad_pts, 3 ))

	sigma_theta_A_pts = Manny.Sigma_Theta_A_Pts
	sigma_theta_B_pts = Manny.Sigma_Theta_B_Pts

	sigma_phi_A_pts = Manny.Sigma_Phi_A_Pts
	sigma_phi_B_pts = Manny.Sigma_Phi_B_Pts


	for quad_pt in range(num_quad_pts):

		Vec_pt = Vectors_at_Quad[quad_pt, :].flatten()

		Normal_Vec_pt = []

		# Chart A:
		if(lbdv.Chart_of_Quad_Pts[quad_pt] > 0):

			Normal_Vec_pt = np.cross(sigma_theta_A_pts[quad_pt, :].flatten(), sigma_phi_A_pts[quad_pt, :].flatten())

			dot_theta_A = np.dot(sigma_theta_A_pts[quad_pt, :].flatten(), Normal_Vec_pt)
			dot_phi_A = np.dot(sigma_phi_A_pts[quad_pt, :].flatten(), Normal_Vec_pt)

			if(dot_theta_A > 1e-12):
				print("dot_theta_A = "+str(dot_theta_A))

			if(dot_phi_A > 1e-12):
				print("dot_phi_A = "+str(dot_phi_A))

		# Chart B:
		else:
			quad_pt_rot = lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pt)
			Normal_Vec_pt = np.cross(sigma_theta_B_pts[quad_pt_rot, :].flatten(), sigma_phi_B_pts[quad_pt_rot, :].flatten())


			dot_theta_B = np.dot(sigma_theta_B_pts[quad_pt_rot, :].flatten(), Normal_Vec_pt)
			dot_phi_B = np.dot(sigma_phi_B_pts[quad_pt_rot, :].flatten(), Normal_Vec_pt)

			if(dot_theta_B > 1e-12):
				print("dot_theta_B = "+str(dot_theta_B))

			if(dot_phi_B > 1e-12):
				print("dot_phi_B = "+str(dot_phi_B))

		Vec_pt_Tangent = Vec_pt - Normal_Vec_pt*(np.dot(Vec_pt, Normal_Vec_pt)/np.dot(Normal_Vec_pt, Normal_Vec_pt))

		Tangent_Vecs_at_Quad[quad_pt, :] = Vec_pt_Tangent.reshape(1,3)

	return Tangent_Vecs_at_Quad

'''
# Uses above code to get Normal direction in both charts:
def Normal_Dirs_Manny(lbdv, Manny):

	num_quad_pts = lbdv.lbdv_quad_pts
	Normal_Dirs_at_Quad = np.zeros(( num_quad_pts, 3 ))

	sigma_theta_A_pts = Manny.Sigma_Theta_A_Pts
	sigma_theta_B_pts = Manny.Sigma_Theta_B_Pts

	sigma_phi_A_pts = Manny.Sigma_Phi_A_Pts
	sigma_phi_B_pts = Manny.Sigma_Phi_B_Pts


	for quad_pt in range(num_quad_pts):

		Normal_Vec_pt = []

		# Chart A:
		if(lbdv.Chart_of_Quad_Pts[quad_pt] > 0):

			Normal_Vec_pt = np.cross(sigma_theta_A_pts[quad_pt, :].flatten(), sigma_phi_A_pts[quad_pt, :].flatten())

		# Chart B:
		else:
			quad_pt_rot = lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pt)
			Normal_Vec_pt = np.cross(sigma_theta_B_pts[quad_pt_rot, :].flatten(), sigma_phi_B_pts[quad_pt_rot, :].flatten())

		Normal_Vec_pt_Mag = np.sqrt( np.dot(Normal_Vec_pt, Normal_Vec_pt) )

		Normal_Dirs_at_Quad[quad_pt, :] = Normal_Vec_pt.reshape(1,3)/Normal_Vec_pt_Mag

	return  Normal_Dirs_at_Quad #np.hstack(( lbdv.X, lbdv.Y, lbdv.Z ))
'''

# Returns Riemannian Inner Product of 1-Forms input, returns Integral of
def Riemann_L2_Inner_Product_One_Form(One_Form_pts_1, One_Form_pts_2, lbdv, manny):

	One_Form_1 = euc_k_form(1, lbdv.lbdv_quad_pts, manny.Man_SPH_Deg, manny, One_Form_pts_1)
	One_Form_2 = euc_k_form(1, lbdv.lbdv_quad_pts, manny.Man_SPH_Deg, manny, One_Form_pts_2)

	Inner_Prod_quad_pts = One_Form_1.Riemann_Inner_Prouct_One_Form(One_Form_2, lbdv)

	return Integral_on_Manny(Inner_Prod_quad_pts, manny, lbdv)


# Return L2 norm of one form:
def Riemann_L2_Norm_One_Form(One_Form_pts, lbdv, manny):
	return np.sqrt( Riemann_L2_Inner_Product_One_Form(One_Form_pts, One_Form_pts, lbdv, manny) )


# return L2  norm of scalar field:
def Riemann_L2_Norm_Zero_Form(Zero_Form_pts, lbdv, manny):
	return np.sqrt( Integral_on_Manny( np.multiply(Zero_Form_pts, Zero_Form_pts) , manny, lbdv) )

# return L2 inner-product  of scalar field:
def Riemann_L2_Inner_Product_Zero_Form(Zero_Form_pts_1, Zero_Form_pts_2, lbdv, manny):
	return  Integral_on_Manny( np.multiply(Zero_Form_pts_1, Zero_Form_pts_2) , manny, lbdv)

###############################################################################

class euc_k_form(object):

	def __init__(self, k_val, Q_val, P_deg, Manny, array_of_quad_vals):

		self.k_value = k_val # k = 0, 1, 2
		self.Q_value = Q_val #lbdv.lbdv_quad_pts # Number of Quad Pts
		self.P_degree = P_deg # Degree of Basis
		#self.LBDV = Lbdv
		self.Manifold = Manny

		self.array_size = 0

		if(self.k_value == 0):
			self.array_size = 1
		else:
			self.array_size = 3 #{x,y,z} or {dx^dy, dx^dz, dy^dz}

		self.quad_pt_array = array_of_quad_vals # col_i = comp_i, row_j = quad_pt_j


	# K_form copy constuctor
	def copy(self):
		return euc_k_form(self.k_value, self.Q_value, self.P_degree, self.Manifold, self.quad_pt_array)


	# Adds k_forms together (of same degree):
	def linear_comb(self, euc_k_form_2, Const_1, Const_2):
		return euc_k_form(self.k_value, self.Q_value, self.P_degree, self.Manifold, Const_1*self.quad_pt_array + Const_2*euc_k_form_2.quad_pt_array)


	# Multiplies k_form by a constanct
	def times_const(self, Const):
		return euc_k_form(self.k_value, self.Q_value, self.P_degree, self.Manifold, Const*self.quad_pt_array)


	# Multiplies k_form by a function
	def times_fn(self, fn_vals_at_quad_pts):

		num_cols = self.array_size
		product_array = np.zeros(( self.Q_value, num_cols ))

		product_array = np.einsum('i, ij -> ij', np.squeeze(fn_vals_at_quad_pts), self.quad_pt_array)

		return euc_k_form(self.k_value, self.Q_value, self.P_degree, self.Manifold, product_array)


	# Compute 1-form at a pt, given vector field
	#def Flat(self, deriv, Matrix_Fn_Quad_pt, lbdv):
	def Flat(self, G_deriv, A_inv_deriv, lbdv):
		# Deriv = '', for G, 'theta' to use G_theta, 'phi' to use G_phi
		# Matrix_Fn_Quad_pt is a 3x3 matrix function of the quad_pt, Chart (Could be eye(3))


		# Store the value of one_form comps, before combining:
		One_Form_Theta_Comp_A_vals = zero_quad_array(self.Q_value)
		One_Form_Theta_Comp_B_vals = zero_quad_array(self.Q_value)

		One_Form_Phi_Comp_A_vals = zero_quad_array(self.Q_value)
		One_Form_Phi_Comp_B_vals = zero_quad_array(self.Q_value)


		for quad_pt in range(self.Q_value):

			# We use quad pts within each chart
			if(lbdv.Chart_of_Quad_Pts[quad_pt] > 0):

				quad_pt_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(quad_pt)

				alpha_sharp_X_A_pt = self.quad_pt_array[quad_pt, :].T
				alpha_sharp_X_B_pt = self.quad_pt_array[quad_pt_rot, :].T

				Flat_A_mat = []
				Flat_B_mat = []

				# Use G_theta
				if(G_deriv == 'theta'):
					Flat_A_mat = self.Manifold.rho_theta_A_Mats[:, :, quad_pt]
					Flat_B_mat = self.Manifold.rho_theta_B_Mats[:, :, quad_pt]

				# Use G_phi
				elif(G_deriv == 'phi'):
					Flat_A_mat = self.Manifold.rho_phi_A_Mats[:, :, quad_pt]
					Flat_B_mat = self.Manifold.rho_phi_B_Mats[:, :, quad_pt]

				# Use G
				elif(G_deriv == ''):
					if(A_inv_deriv == ''):
						Flat_A_mat = self.Manifold.rho_A_Mats[:, :, quad_pt]
						Flat_B_mat = self.Manifold.rho_B_Mats[:, :, quad_pt]

					elif(A_inv_deriv == 'theta'):
						Flat_A_mat = self.Manifold.xi_theta_A_Mats[:, :, quad_pt]
						Flat_B_mat = self.Manifold.xi_theta_B_Mats[:, :, quad_pt]

					elif(A_inv_deriv == 'phi'):
						Flat_A_mat = self.Manifold.xi_phi_A_Mats[:, :, quad_pt]
						Flat_B_mat = self.Manifold.xi_phi_B_Mats[:, :, quad_pt]
					else:
						print("Error in Flat A_inv_deriv")

				# DONT use G: (regular flat)
				elif(G_deriv == 'No_G'):
					Flat_A_mat = self.Manifold.Change_Basis_Mat(quad_pt, 'A')
					Flat_B_mat = self.Manifold.Change_Basis_Mat(quad_pt, 'B')
				else:
					print("Error in Flat G_deriv")

				one_form_comps_A_pt = np.dot(Flat_A_mat, alpha_sharp_X_A_pt)
				one_form_comps_B_pt = np.dot(Flat_B_mat, alpha_sharp_X_B_pt)

				# we need to apply A^-1, but cant calc directly:
				if(G_deriv == 'No_G'):
					one_form_comps_A_pt = np.linalg.solve(Flat_A_mat, alpha_sharp_X_A_pt)
					one_form_comps_B_pt = np.linalg.solve(Flat_B_mat, alpha_sharp_X_B_pt)

				One_Form_Theta_Comp_A_vals[quad_pt] = one_form_comps_A_pt[0]
				One_Form_Theta_Comp_B_vals[quad_pt] = one_form_comps_B_pt[0]

				One_Form_Phi_Comp_A_vals[quad_pt] = one_form_comps_A_pt[1]
				One_Form_Phi_Comp_B_vals[quad_pt] = one_form_comps_B_pt[1]


		return One_Form_Theta_Comp_A_vals, One_Form_Theta_Comp_B_vals, One_Form_Phi_Comp_A_vals, One_Form_Phi_Comp_B_vals


	# d: k_form -> (k+1)_form
	def Ext_Der(self, lbdv):

		if(self.k_value == 0):

			#print("Doing Ext_Der on 0-Forms")

			f_dtheta_A_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)
			f_dphi_A_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)

			f_dtheta_B_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)
			f_dphi_B_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)

			# Project function f into basis of each chart
			f_SPH_A, f_SPH_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.quad_pt_array, self.P_degree, lbdv)

			# Theta Ders Exact within basis
			f_SPH_A_dtheta = f_SPH_A.Quick_Theta_Der()
			f_SPH_B_dtheta = f_SPH_B.Quick_Theta_Bar_Der()

			quad_pts = range(self.Q_value)

			f_dtheta_A_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_A_dtheta.Eval_SPH_Coef_Mat(quad_pts, lbdv), 0)
			f_dphi_A_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_A.Eval_SPH_Der_Phi_Coef(quad_pts, lbdv), 0)

			f_dtheta_B_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_B_dtheta.Eval_SPH_Coef_Mat(quad_pts, lbdv), 0)
			f_dphi_B_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_B.Eval_SPH_Der_Phi_Coef(quad_pts, lbdv), 0)

			# Use sharp function above to convert 1-form components to euclidean vector field
			return Sharp(f_dtheta_A_quad_vals, f_dphi_A_quad_vals, f_dtheta_B_quad_vals, f_dphi_B_quad_vals, self.P_degree, lbdv, self.Manifold)


		if(self.k_value == 1):

			#print("Doing Ext_Der on 1-Forms")

			# Store values of Ext Det  of (1-forms) alpha, at quad pts in each Chart
			d_alpha_dx_dy_A_pts = zero_quad_array(self.Q_value)
			d_alpha_dx_dz_A_pts = zero_quad_array(self.Q_value)
			d_alpha_dy_dz_A_pts = zero_quad_array(self.Q_value)

			d_alpha_dx_dy_B_pts = zero_quad_array(self.Q_value)
			d_alpha_dx_dz_B_pts = zero_quad_array(self.Q_value)
			d_alpha_dy_dz_B_pts = zero_quad_array(self.Q_value)

			# Vals at quad pts needed for projection
			alpha_sharp_x_quad_vals, alpha_sharp_y_quad_vals, alpha_sharp_z_quad_vals = np.hsplit(self.quad_pt_array, 3)

			# Project euclidean comps of alpha_sharp into basis of each chart
			alpha_sharp_x_SPH_A, alpha_sharp_x_SPH_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(alpha_sharp_x_quad_vals, self.P_degree, lbdv)
			alpha_sharp_y_SPH_A, alpha_sharp_y_SPH_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(alpha_sharp_y_quad_vals, self.P_degree, lbdv)
			alpha_sharp_z_SPH_A, alpha_sharp_z_SPH_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(alpha_sharp_z_quad_vals, self.P_degree, lbdv)

			# SPH Theta Ders Exact within basis
			alpha_sharp_x_SPH_A_dtheta = alpha_sharp_x_SPH_A.Quick_Theta_Der()
			alpha_sharp_x_SPH_B_dtheta = alpha_sharp_x_SPH_B.Quick_Theta_Bar_Der()

			alpha_sharp_y_SPH_A_dtheta = alpha_sharp_y_SPH_A.Quick_Theta_Der()
			alpha_sharp_y_SPH_B_dtheta = alpha_sharp_y_SPH_B.Quick_Theta_Bar_Der()

			alpha_sharp_z_SPH_A_dtheta = alpha_sharp_z_SPH_A.Quick_Theta_Der()
			alpha_sharp_z_SPH_B_dtheta = alpha_sharp_z_SPH_B.Quick_Theta_Bar_Der()

			#print("d_1: finished projections")

			# Quad Vals of theta derivatives functions for alpha_sharp_X comps:
			alpha_sharp_x_A_dtheta_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_x_SPH_A_dtheta, lbdv, 'A')
			alpha_sharp_x_B_dtheta_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_x_SPH_B_dtheta, lbdv, 'B')

			alpha_sharp_y_A_dtheta_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_y_SPH_A_dtheta, lbdv, 'A')
			alpha_sharp_y_B_dtheta_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_y_SPH_B_dtheta, lbdv, 'B')

			alpha_sharp_z_A_dtheta_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_z_SPH_A_dtheta, lbdv, 'A')
			alpha_sharp_z_B_dtheta_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_z_SPH_B_dtheta, lbdv, 'B')


			# Quad Vals of phi derivatives functions for alpha_sharp_X comps (within Charts):
			alpha_sharp_x_A_dphi_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_x_SPH_A, lbdv, 'A')
			alpha_sharp_x_B_dphi_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_x_SPH_B, lbdv, 'B')

			alpha_sharp_y_A_dphi_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_y_SPH_A, lbdv, 'A')
			alpha_sharp_y_B_dphi_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_y_SPH_B, lbdv, 'B')

			alpha_sharp_z_A_dphi_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_z_SPH_A, lbdv, 'A')
			alpha_sharp_z_B_dphi_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(alpha_sharp_z_SPH_B, lbdv, 'B')


			# Create quad val vector Arrays:
			alpha_sharp_dTheta_A_quad_array = np.hstack(( alpha_sharp_x_A_dtheta_vals, alpha_sharp_y_A_dtheta_vals, alpha_sharp_z_A_dtheta_vals ))
			alpha_sharp_dPhi_A_quad_array = np.hstack(( alpha_sharp_x_A_dphi_vals, alpha_sharp_y_A_dphi_vals, alpha_sharp_z_A_dphi_vals ))

			alpha_sharp_dTheta_B_quad_array = np.hstack(( alpha_sharp_x_B_dtheta_vals, alpha_sharp_y_B_dtheta_vals, alpha_sharp_z_B_dtheta_vals ))
			alpha_sharp_dPhi_B_quad_array = np.hstack(( alpha_sharp_x_B_dphi_vals, alpha_sharp_y_B_dphi_vals, alpha_sharp_z_B_dphi_vals ))

			alpha_vec_dTheta_A = euc_k_form(1, self.Q_value, self.P_degree, self.Manifold, alpha_sharp_dTheta_A_quad_array)
			alpha_vec_dPhi_A = euc_k_form(1, self.Q_value, self.P_degree, self.Manifold, alpha_sharp_dPhi_A_quad_array)

			alpha_vec_dTheta_B = euc_k_form(1, self.Q_value, self.P_degree, self.Manifold, alpha_sharp_dTheta_B_quad_array)
			alpha_vec_dPhi_B = euc_k_form(1, self.Q_value, self.P_degree, self.Manifold, alpha_sharp_dPhi_B_quad_array)

			# Compute Y = -dA_i * A^(-1)
			def Y_mat(q_pt, Chart, deriv):

				Basis_Mat = self.Manifold.Change_Basis_Mat(q_pt, Chart)
				dBasis_Mat_deriv = []

				if(deriv == 'theta'):
					dBasis_Mat_deriv = self.Manifold.dChange_Basis_Mat_theta(q_pt, Chart)

				if(deriv == 'phi'):
					dBasis_Mat_deriv = self.Manifold.dChange_Basis_Mat_phi(q_pt, Chart)

				return np.linalg.solve(Basis_Mat.T, -1*dBasis_Mat_deriv.T).T

			# Values of components, in each chart:
			# alpha_Sharp_X.Flat() gives us: (alpha_theta_comp_A_vals, alpha_theta_comp_B_vals, alpha_phi_comp_A_vals, alpha_phi_comp_B_vals)

			#print("d_1: created needed 1-forms vecs")

			# dG_i * A^(-1) * alpha^X
			dG_theta_comps = self.Flat('theta', '', lbdv)
			dG_phi_comps = self.Flat('phi', '', lbdv)

			# G * A^(-1) * dalpha^X_i
			d_alpha_sharp_X_theta_A_comps = alpha_vec_dTheta_A.Flat('', '', lbdv)
			d_alpha_sharp_X_phi_A_comps = alpha_vec_dPhi_A.Flat('', '', lbdv)

			d_alpha_sharp_X_theta_B_comps = alpha_vec_dTheta_B.Flat('', '', lbdv)
			d_alpha_sharp_X_phi_B_comps = alpha_vec_dPhi_B.Flat('', '', lbdv)

			# G * A^(-1) * dA_i * A^(-1) * alpha^X
			d_Basis_mat_theta_comps = self.Flat('', 'theta', lbdv)
			d_Basis_mat_phi_comps = self.Flat('', 'phi', lbdv)

			#print("d_1: Flat operations done")

			# Values we need to compute d:
			d_theta_comp_A_dphi_vals = dG_phi_comps[0] + d_Basis_mat_phi_comps[0] + d_alpha_sharp_X_phi_A_comps[0]
			d_phi_comp_A_dtheta_vals = dG_theta_comps[2] + d_Basis_mat_theta_comps[2] + d_alpha_sharp_X_theta_A_comps[2]

			d_theta_comp_B_dphi_vals = dG_phi_comps[1] + d_Basis_mat_phi_comps[1] + d_alpha_sharp_X_phi_B_comps[1]
			d_phi_comp_B_dtheta_vals = dG_theta_comps[3] + d_Basis_mat_theta_comps[3] + d_alpha_sharp_X_theta_B_comps[3]

			# d_alpha polar vals
			d_alpha_A_Polar_vals = d_phi_comp_A_dtheta_vals - d_theta_comp_A_dphi_vals
			d_alpha_B_Polar_vals = d_phi_comp_B_dtheta_vals - d_theta_comp_B_dphi_vals



			quad_pts = range(self.Q_value)

			# Convert to Euclidean Coors:
			dx_dy_A_pts = self.Manifold.dx_dy_A_Vals_From_Polar
			dx_dz_A_pts = self.Manifold.dx_dz_A_Vals_From_Polar
			dy_dz_A_pts = self.Manifold.dy_dz_A_Vals_From_Polar

			dx_dy_B_pts = self.Manifold.dx_dy_B_Vals_From_Polar
			dx_dz_B_pts = self.Manifold.dx_dz_B_Vals_From_Polar
			dy_dz_B_pts = self.Manifold.dy_dz_B_Vals_From_Polar

			# We use quad pts within each chart
			d_alpha_A_Polar_pts = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, d_alpha_A_Polar_vals, 0)
			d_alpha_B_Polar_pts = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, d_alpha_B_Polar_vals, 0)

			d_alpha_dx_dy_A_pts = d_alpha_A_Polar_pts*dx_dy_A_pts
			d_alpha_dx_dz_A_pts = d_alpha_A_Polar_pts*dx_dz_A_pts
			d_alpha_dy_dz_A_pts = d_alpha_A_Polar_pts*dy_dz_A_pts

			d_alpha_dx_dy_B_pts = d_alpha_B_Polar_pts*dx_dy_B_pts
			d_alpha_dx_dz_B_pts = d_alpha_B_Polar_pts*dx_dz_B_pts
			d_alpha_dy_dz_B_pts = d_alpha_B_Polar_pts*dy_dz_B_pts



			#print("d_1: computed polar solutions in each chart")

			# Combine Values from Charts
			d_alpha_Used_dx_dy_pts = Combine_Chart_Quad_Vals(d_alpha_dx_dy_A_pts, d_alpha_dx_dy_B_pts, lbdv)
			d_alpha_Used_dx_dz_pts = Combine_Chart_Quad_Vals(d_alpha_dx_dz_A_pts, d_alpha_dx_dz_B_pts, lbdv)
			d_alpha_Used_dy_dz_pts = Combine_Chart_Quad_Vals(d_alpha_dy_dz_A_pts, d_alpha_dy_dz_B_pts, lbdv)

			#print("d_1: combined into euc solution")

			# Create array of 2-form values
			New_Quad_Pt_Array = np.hstack(( d_alpha_Used_dx_dy_pts, d_alpha_Used_dx_dz_pts, d_alpha_Used_dy_dz_pts ))

			return euc_k_form(2, self.Q_value, self.P_degree, self.Manifold, New_Quad_Pt_Array)


		if(self.k_value == 2): # ext der of of 2-form is 0:

			#print("Doing Ext_Der on 2-Forms")

			return euc_k_form(0, self.Q_value, self.P_degree, self.Manifold, zero_quad_array(self.Q_value))


	# Compute *: k_form -> (n-k)_form
	def Hodge_Star(self, lbdv):

		if(self.k_value == 0):

			#print("Doing Hodge Star on 0-Forms")

			# store values of Hodge star of fn (0-form) f, at quad pts in each Chart
			star_f_A_dx_dy_pts = zero_quad_array(self.Q_value)
			star_f_A_dx_dz_pts = zero_quad_array(self.Q_value)
			star_f_A_dy_dz_pts = zero_quad_array(self.Q_value)

			star_f_B_dx_dy_pts = zero_quad_array(self.Q_value)
			star_f_B_dx_dz_pts = zero_quad_array(self.Q_value)
			star_f_B_dy_dz_pts = zero_quad_array(self.Q_value)


			# Corresponding quad_pts in chart B (in Chart A Coors)
			quad_pts = range(self.Q_value)
			quad_pts_inv_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(quad_pts)

			# values at (theta_C, phi_C) = (theta_i, phi_i), given q_i, Difffernt points on manifold!
			f_A_pts = self.quad_pt_array[quad_pts]
			f_B_pts = self.quad_pt_array[quad_pts_inv_rot]

			# metric factor, at each point, within Chart
			met_fac_A_pts = self.Manifold.Metric_Factor_A_pts  #np.where(lbdv.Chart_of_Quad_Pts > 0, self.Manifold.Metric_Factor_Quad_Pt(quad_pts, lbdv, 'A'), 0)
			met_fac_B_pts = self.Manifold.Metric_Factor_B_pts  #np.where(lbdv.Chart_of_Quad_Pts > 0, self.Manifold.Metric_Factor_Quad_Pt(quad_pts, lbdv, 'B'), 0)

			# Do Hodge Star:
			star_f_dtheta_dphi_A_pts = np.multiply(f_A_pts, met_fac_A_pts)
			star_f_dtheta_dphi_B_pts = np.multiply(f_B_pts, met_fac_B_pts)

			# Convert to Euclidean Coors:
			#dx_dy_A_vals, dx_dz_A_vals, dy_dz_A_vals = Two_Form_Conv_to_Euc_pt(quad_pts, lbdv, 'A', self.Manifold)
			#dx_dy_B_vals, dx_dz_B_vals, dy_dz_B_vals = Two_Form_Conv_to_Euc_pt(quad_pts, lbdv, 'B', self.Manifold)

			dx_dy_A_vals = self.Manifold.dx_dy_A_Vals_From_Polar
			dx_dz_A_vals = self.Manifold.dx_dz_A_Vals_From_Polar
			dy_dz_A_vals = self.Manifold.dy_dz_A_Vals_From_Polar

			dx_dy_B_vals = self.Manifold.dx_dy_B_Vals_From_Polar
			dx_dz_B_vals = self.Manifold.dx_dz_B_Vals_From_Polar
			dy_dz_B_vals = self.Manifold.dy_dz_B_Vals_From_Polar

			star_f_A_dx_dy_pts = np.multiply(star_f_dtheta_dphi_A_pts, dx_dy_A_vals)
			star_f_A_dx_dz_pts = np.multiply(star_f_dtheta_dphi_A_pts, dx_dz_A_vals)
			star_f_A_dy_dz_pts = np.multiply(star_f_dtheta_dphi_A_pts, dy_dz_A_vals)

			star_f_B_dx_dy_pts = np.multiply(star_f_dtheta_dphi_B_pts, dx_dy_B_vals)
			star_f_B_dx_dz_pts = np.multiply(star_f_dtheta_dphi_B_pts, dx_dz_B_vals)
			star_f_B_dy_dz_pts = np.multiply(star_f_dtheta_dphi_B_pts, dy_dz_B_vals)


			# Combine Values from Charts
			star_f_Used_dx_dy_pts = Combine_Chart_Quad_Vals(star_f_A_dx_dy_pts, star_f_B_dx_dy_pts, lbdv)
			star_f_Used_dx_dz_pts = Combine_Chart_Quad_Vals(star_f_A_dx_dz_pts, star_f_B_dx_dz_pts, lbdv)
			star_f_Used_dy_dz_pts = Combine_Chart_Quad_Vals(star_f_A_dy_dz_pts, star_f_B_dy_dz_pts, lbdv)

			# Create array of 2-form values
			New_Quad_Pt_Array = np.hstack((star_f_Used_dx_dy_pts, star_f_Used_dx_dz_pts, star_f_Used_dy_dz_pts ))

			return euc_k_form(2, self.Q_value, self.P_degree, self.Manifold, New_Quad_Pt_Array)


		if(self.k_value == 1):

			#print("Doing Hodge Star on 1-Forms")

			# Store values of Hodge Star of (1-forms) alpha, at quad pts in each Chart
			star_alpha_A_theta_pts = zero_quad_array(self.Q_value)
			star_alpha_A_phi_pts = zero_quad_array(self.Q_value)

			star_alpha_B_theta_pts = zero_quad_array(self.Q_value)
			star_alpha_B_phi_pts = zero_quad_array(self.Q_value)

			# Values of components, in each chart
			alpha_theta_comp_A_vals, alpha_theta_comp_B_vals, alpha_phi_comp_A_vals, alpha_phi_comp_B_vals = self.Flat('', '', lbdv)

			met_fac_A_pts = self.Manifold.Metric_Factor_A_pts
			met_fac_B_pts = self.Manifold.Metric_Factor_B_pts

			g_inv_theta_theta_A_pts = self.Manifold.g_Inv_Theta_Theta_A_Pts
			g_inv_theta_phi_A_pts = self.Manifold.g_Inv_Theta_Phi_A_Pts
			g_inv_phi_phi_A_pts = self.Manifold.g_Inv_Phi_Phi_A_Pts

			g_inv_theta_theta_B_pts = self.Manifold.g_Inv_Theta_Theta_B_Pts
			g_inv_theta_phi_B_pts = self.Manifold.g_Inv_Theta_Phi_B_Pts
			g_inv_phi_phi_B_pts = self.Manifold.g_Inv_Phi_Phi_B_Pts


			# Take Hodge-Star of 1-Form:
			star_alpha_A_theta_pts = np.multiply(-1*met_fac_A_pts, (np.multiply(g_inv_theta_phi_A_pts, alpha_theta_comp_A_vals) + np.multiply(g_inv_phi_phi_A_pts, alpha_phi_comp_A_vals)))

			star_alpha_A_phi_pts = np.multiply(met_fac_A_pts, (np.multiply(g_inv_theta_theta_A_pts, alpha_theta_comp_A_vals) + np.multiply(g_inv_theta_phi_A_pts, alpha_phi_comp_A_vals)))

			star_alpha_B_theta_pts = np.multiply(-1*met_fac_B_pts, (np.multiply(g_inv_theta_phi_B_pts, alpha_theta_comp_B_vals) + np.multiply(g_inv_phi_phi_B_pts, alpha_phi_comp_B_vals)))

			star_alpha_B_phi_pts = np.multiply(met_fac_B_pts, (np.multiply(g_inv_theta_theta_B_pts, alpha_theta_comp_B_vals) + np.multiply(g_inv_theta_phi_B_pts, alpha_phi_comp_B_vals)))

			return Sharp(star_alpha_A_theta_pts, star_alpha_A_phi_pts, star_alpha_B_theta_pts, star_alpha_B_phi_pts, self.P_degree, lbdv, self.Manifold)

		if(self.k_value == 2):

			#print("Doing Hodge Star on 2-Forms")

			# store values of Hodge star of fn (2-form) beta, at quad pts in each Chart
			star_beta_from_A_pts = zero_quad_array(self.Q_value)
			star_beta_from_B_pts = zero_quad_array(self.Q_value)


			# Corresponding quad pt in chart B (in Chart A Coors)
			quad_pts = range(self.Q_value)
			quad_pts_inv_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(quad_pts)

			# inv metric factor, at each point, within Chart

			#inv_met_fac_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1/self.Manifold.Metric_Factor_Quad_Pt(quad_pts, lbdv, 'A'), 0)
			#inv_met_fac_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1/self.Manifold.Metric_Factor_Quad_Pt(quad_pts, lbdv, 'B'), 0)

			#BJG: new change:

			inv_met_fac_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1./self.Manifold.Metric_Factor_A_pts, 0)
			inv_met_fac_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1./self.Manifold.Metric_Factor_B_pts, 0)

			# values at (theta_C, phi_C) = (theta_i, phi_i), given q_i, Difffernt points on manifold!
			beta_dx_dy_A_pts, beta_dx_dz_A_pts, beta_dy_dz_A_pts = np.hsplit(self.quad_pt_array[quad_pts], 3)
			beta_dx_dy_B_pts, beta_dx_dz_B_pts, beta_dy_dz_B_pts = np.hsplit(self.quad_pt_array[quad_pts_inv_rot], 3)

			# 2-forms to cartesian conversions:
			dx_dy_to_dtheta_dphi_A_pts, dx_dz_to_dtheta_dphi_A_pts, dy_dz_to_dtheta_dphi_A_pts = Two_Form_Conv_to_Polar_pt(quad_pts, lbdv, self.Manifold, 'A')
			dx_dy_to_dtheta_dphi_B_pts, dx_dz_to_dtheta_dphi_B_pts, dy_dz_to_dtheta_dphi_B_pts = Two_Form_Conv_to_Polar_pt(quad_pts, lbdv, self.Manifold, 'B')

			# Do Hodge Star:
			beta_dtheta_dphi_A_pts = np.multiply(beta_dx_dy_A_pts, dx_dy_to_dtheta_dphi_A_pts) + np.multiply(beta_dy_dz_A_pts, dy_dz_to_dtheta_dphi_A_pts) +  np.multiply(beta_dx_dz_A_pts, dx_dz_to_dtheta_dphi_A_pts)

			beta_dtheta_dphi_B_pts =  np.multiply(beta_dx_dy_B_pts, dx_dy_to_dtheta_dphi_B_pts) + np.multiply(beta_dy_dz_B_pts, dy_dz_to_dtheta_dphi_B_pts) + np.multiply(beta_dx_dz_B_pts, dx_dz_to_dtheta_dphi_B_pts)

			star_beta_from_A_pts = np.multiply(beta_dtheta_dphi_A_pts, inv_met_fac_A_pts)
			star_beta_from_B_pts = np.multiply(beta_dtheta_dphi_B_pts, inv_met_fac_B_pts)


			# Combine Values from Charts
			star_beta_Used_pts = Combine_Chart_Quad_Vals(star_beta_from_A_pts , star_beta_from_B_pts , lbdv)

			return euc_k_form(0, self.Q_value, self.P_degree, self.Manifold, star_beta_Used_pts)



	# Computes -*d directly;
	# Takes 0-Forms -> 1-Forms (in local chart)
	def Gen_Curl_0(self, lbdv):

		if(self.k_value == 0):

			#print("Doing Ext_Der on 0-Forms")

			f_dtheta_A_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)
			f_dphi_A_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)

			f_dtheta_B_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)
			f_dphi_B_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)

			# Project function f into basis of each chart
			f_SPH_A, f_SPH_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.quad_pt_array, self.P_degree, lbdv)

			# Theta Ders Exact within basis
			f_SPH_A_dtheta = f_SPH_A.Quick_Theta_Der()
			f_SPH_B_dtheta = f_SPH_B.Quick_Theta_Bar_Der()

			quad_pts = range(self.Q_value)

			f_dtheta_A_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_A_dtheta.Eval_SPH_Coef_Mat(quad_pts, lbdv), 0)
			f_dphi_A_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_A.Eval_SPH_Der_Phi_Coef(quad_pts, lbdv), 0)

			f_dtheta_B_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_B_dtheta.Eval_SPH_Coef_Mat(quad_pts, lbdv), 0)
			f_dphi_B_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_B.Eval_SPH_Der_Phi_Coef(quad_pts, lbdv), 0)


			# fields needed for *_1 :
			met_fac_A_pts = self.Manifold.Metric_Factor_A_pts
			met_fac_B_pts = self.Manifold.Metric_Factor_B_pts

			g_inv_theta_theta_A_pts = self.Manifold.g_Inv_Theta_Theta_A_Pts
			g_inv_theta_phi_A_pts = self.Manifold.g_Inv_Theta_Phi_A_Pts
			g_inv_phi_phi_A_pts = self.Manifold.g_Inv_Phi_Phi_A_Pts

			g_inv_theta_theta_B_pts = self.Manifold.g_Inv_Theta_Theta_B_Pts
			g_inv_theta_phi_B_pts = self.Manifold.g_Inv_Theta_Phi_B_Pts
			g_inv_phi_phi_B_pts = self.Manifold.g_Inv_Phi_Phi_B_Pts


			# Take Hodge-Star of 1-Form:
			neg_star_df_A_theta_pts = (-1)*np.multiply(-1*met_fac_A_pts, (np.multiply(g_inv_theta_phi_A_pts, f_dtheta_A_quad_vals) + np.multiply(g_inv_phi_phi_A_pts, f_dphi_A_quad_vals)))

			neg_star_df_A_phi_pts = (-1)*np.multiply(met_fac_A_pts, (np.multiply(g_inv_theta_theta_A_pts, f_dtheta_A_quad_vals) + np.multiply(g_inv_theta_phi_A_pts, f_dphi_A_quad_vals)))

			neg_star_df_B_theta_pts = (-1)*np.multiply(-1*met_fac_B_pts, (np.multiply(g_inv_theta_phi_B_pts, f_dtheta_B_quad_vals) + np.multiply(g_inv_phi_phi_B_pts, f_dphi_B_quad_vals)))

			neg_star_df_B_phi_pts = (-1)*np.multiply(met_fac_B_pts, (np.multiply(g_inv_theta_theta_B_pts, f_dtheta_B_quad_vals) + np.multiply(g_inv_theta_phi_B_pts, f_dphi_B_quad_vals)))


			return neg_star_df_A_theta_pts, neg_star_df_A_phi_pts, neg_star_df_B_theta_pts, neg_star_df_B_phi_pts
			#Sharp(star_alpha_A_theta_pts, star_alpha_A_phi_pts, star_alpha_B_theta_pts, star_alpha_B_phi_pts, self.P_degree, lbdv, self.Manifold)


		else:
			print("Error: Zero Forms needed for this inner-product fn to work")


	# Computes -*d directly; Will give output k-forms
	def Gen_Curl_to_K_Form(self, lbdv):

		if(self.k_value == 0):

			#print("Doing Ext_Der on 0-Forms")

			f_dtheta_A_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)
			f_dphi_A_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)

			f_dtheta_B_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)
			f_dphi_B_quad_vals = zero_quad_array(lbdv.lbdv_quad_pts)

			# Project function f into basis of each chart
			f_SPH_A, f_SPH_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.quad_pt_array, self.P_degree, lbdv)

			# Theta Ders Exact within basis
			f_SPH_A_dtheta = f_SPH_A.Quick_Theta_Der()
			f_SPH_B_dtheta = f_SPH_B.Quick_Theta_Bar_Der()

			quad_pts = range(self.Q_value)

			f_dtheta_A_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_A_dtheta.Eval_SPH_Coef_Mat(quad_pts, lbdv), 0)
			f_dphi_A_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_A.Eval_SPH_Der_Phi_Coef(quad_pts, lbdv), 0)

			f_dtheta_B_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_B_dtheta.Eval_SPH_Coef_Mat(quad_pts, lbdv), 0)
			f_dphi_B_quad_vals = np.where(lbdv.Chart_of_Quad_Pts[quad_pts] > 0, f_SPH_B.Eval_SPH_Der_Phi_Coef(quad_pts, lbdv), 0)


			# fields needed for local expr of C0:
			inv_met_fac_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1./self.Manifold.Metric_Factor_A_pts, 0)
			inv_met_fac_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1./self.Manifold.Metric_Factor_B_pts, 0)

			Sigma_Theta_A_pts = self.Manifold.Sigma_Theta_A_Pts
			Sigma_Theta_B_pts =self.Manifold.Sigma_Theta_B_Pts

			Sigma_Phi_A_pts = self.Manifold.Sigma_Phi_A_Pts
			Sigma_Phi_B_pts = self.Manifold.Sigma_Phi_B_Pts

			C0_A_pts = np.multiply( np.multiply(f_dphi_A_quad_vals, Sigma_Theta_A_pts) - np.multiply(f_dtheta_A_quad_vals, Sigma_Phi_A_pts), inv_met_fac_A_pts)
			C0_B_pts = np.multiply( np.multiply(f_dphi_B_quad_vals, Sigma_Theta_B_pts) - np.multiply(f_dtheta_B_quad_vals, Sigma_Phi_B_pts), inv_met_fac_B_pts)

			C0_X_A_pts, C0_Y_A_pts, C0_Z_A_pts = np.hsplit(C0_A_pts, 3)
			C0_X_B_pts, C0_Y_B_pts, C0_Z_B_pts = np.hsplit(C0_B_pts, 3)

			C0_X_Used_pts = Combine_Chart_Quad_Vals(C0_X_A_pts, C0_X_B_pts, lbdv)
			C0_Y_Used_pts = Combine_Chart_Quad_Vals(C0_Y_A_pts, C0_Y_B_pts, lbdv)
			C0_Z_Used_pts = Combine_Chart_Quad_Vals(C0_Z_A_pts, C0_Z_B_pts, lbdv)

			# For 1-form constuction:
			C0_Used_pts = np.hstack(( C0_X_Used_pts, C0_Y_Used_pts, C0_Z_Used_pts ))
			return euc_k_form(1, self.Q_value, self.P_degree, self.Manifold, C0_Used_pts)

		else:
			print("Error: Zero Forms needed for this to work (Will implement 1-forms next)")


	# div(v) = -\delta v^{\flat}
	def Divergence_1_Form(self, lbdv, debug_mode = False):
		if(self.k_value == 1):


			# get X derivatives at quad pts:
			Vec_X_SPH_Fn_A, Vec_X_SPH_Fn_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.quad_pt_array[:, 0], self.P_degree, lbdv)

			Vec_X_SPH_Fn_dTheta_A = Vec_X_SPH_Fn_A.Quick_Theta_Der()
			Vec_X_A_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_X_SPH_Fn_dTheta_A, lbdv, 'A')
			Vec_X_A_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Vec_X_SPH_Fn_A, lbdv, 'A')


			# for debugging, should recover input
			quad_pts_inv_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(range(self.Manifold.num_quad_pts))
			Vec_X_SPH_Fn_A_quad_pts_from_SPH = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_X_SPH_Fn_A, lbdv, 'A')
			Vec_X_SPH_Fn_B_quad_pts_from_SPH = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_X_SPH_Fn_B, lbdv, 'A')
			norm_err_Vec_X_A =  sph_f.Lp_Rel_Error_At_Quad(Vec_X_SPH_Fn_A_quad_pts_from_SPH, self.quad_pt_array[:, 0], lbdv, 2)
			norm_err_Vec_X_B =  sph_f.Lp_Rel_Error_At_Quad(Vec_X_SPH_Fn_B_quad_pts_from_SPH, self.quad_pt_array[quad_pts_inv_rot, 0], lbdv, 2)
			if( (norm_err_Vec_X_A > 1.e-3 or norm_err_Vec_X_A > 1.e-3 ) and debug_mode == True):
				print("norm_err_Vec_X_A test failed, norm_err_Vec_X_A = "+str(norm_err_Vec_X_A))
				print("norm_err_Vec_X_B test failed, norm_err_Vec_X_B = "+str(norm_err_Vec_X_B))

			Vec_X_SPH_Fn_dTheta_B = Vec_X_SPH_Fn_B.Quick_Theta_Der()
			Vec_X_B_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_X_SPH_Fn_dTheta_B, lbdv, 'A')
			Vec_X_B_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Vec_X_SPH_Fn_B, lbdv, 'A')

			# get Y derivatives at quad pts:
			Vec_Y_SPH_Fn_A, Vec_Y_SPH_Fn_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.quad_pt_array[:, 1], self.P_degree, lbdv)

			Vec_Y_SPH_Fn_dTheta_A = Vec_Y_SPH_Fn_A.Quick_Theta_Der()
			Vec_Y_A_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_Y_SPH_Fn_dTheta_A, lbdv, 'A')
			Vec_Y_A_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Vec_Y_SPH_Fn_A, lbdv, 'A')

			Vec_Y_SPH_Fn_dTheta_B = Vec_Y_SPH_Fn_B.Quick_Theta_Der()
			Vec_Y_B_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_Y_SPH_Fn_dTheta_B, lbdv, 'A')
			Vec_Y_B_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Vec_Y_SPH_Fn_B, lbdv, 'A')


			# for debugging, should recover input
			quad_pts_inv_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(range(self.Manifold.num_quad_pts))
			Vec_Y_SPH_Fn_A_quad_pts_from_SPH = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_Y_SPH_Fn_A, lbdv, 'A')
			Vec_Y_SPH_Fn_B_quad_pts_from_SPH = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_Y_SPH_Fn_B, lbdv, 'A')
			norm_err_Vec_Y_A =  sph_f.Lp_Rel_Error_At_Quad(Vec_Y_SPH_Fn_A_quad_pts_from_SPH, self.quad_pt_array[:, 1], lbdv, 2)
			norm_err_Vec_Y_B =  sph_f.Lp_Rel_Error_At_Quad(Vec_Y_SPH_Fn_B_quad_pts_from_SPH, self.quad_pt_array[quad_pts_inv_rot, 1], lbdv, 2)
			if( (norm_err_Vec_Y_A > 1.e-3 or norm_err_Vec_Y_A > 1.e-3 ) and debug_mode == True):
				print("norm_err_Vec_Y_A test failed, norm_err_Vec_Y_A = "+str(norm_err_Vec_Y_A))
				print("norm_err_Vec_Y_B test failed, norm_err_Vec_Y_B = "+str(norm_err_Vec_Y_B))


			# get Z derivatives at quad pts:
			Vec_Z_SPH_Fn_A, Vec_Z_SPH_Fn_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.quad_pt_array[:, 2], self.P_degree, lbdv)

			Vec_Z_SPH_Fn_dTheta_A = Vec_Z_SPH_Fn_A.Quick_Theta_Der()
			Vec_Z_A_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_Z_SPH_Fn_dTheta_A, lbdv, 'A')
			Vec_Z_A_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Vec_Z_SPH_Fn_A, lbdv, 'A')

			Vec_Z_SPH_Fn_dTheta_B = Vec_Z_SPH_Fn_B.Quick_Theta_Der()
			Vec_Z_B_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_Z_SPH_Fn_dTheta_B, lbdv, 'A')
			Vec_Z_B_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Vec_Z_SPH_Fn_B, lbdv, 'A')


			# for debugging, should recover input
			quad_pts_inv_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(range(self.Manifold.num_quad_pts))
			Vec_Z_SPH_Fn_A_quad_pts_from_SPH = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_Z_SPH_Fn_A, lbdv, 'A')
			Vec_Z_SPH_Fn_B_quad_pts_from_SPH = Extract_Quad_Pt_Vals_From_SPH_Fn(Vec_Z_SPH_Fn_B, lbdv, 'A')
			norm_err_Vec_Z_A =  sph_f.Lp_Rel_Error_At_Quad(Vec_Z_SPH_Fn_A_quad_pts_from_SPH, self.quad_pt_array[:, 2], lbdv, 2)
			norm_err_Vec_Z_B =  sph_f.Lp_Rel_Error_At_Quad(Vec_Z_SPH_Fn_B_quad_pts_from_SPH, self.quad_pt_array[quad_pts_inv_rot, 2], lbdv, 2)
			if( (norm_err_Vec_Z_A > 1.e-3 or norm_err_Vec_Z_A > 1.e-3 ) and debug_mode == True):
				print("norm_err_Vec_Z_A test failed, norm_err_Vec_Z_A = "+str(norm_err_Vec_Z_A))
				print("norm_err_Vec_Z_B test failed, norm_err_Vec_Z_B = "+str(norm_err_Vec_Z_B))


			# combine vector dirs
			Vec_dTheta_A_pts = np.hstack(( Vec_X_A_dTheta_quad_vals, Vec_Y_A_dTheta_quad_vals, Vec_Z_A_dTheta_quad_vals ))
			Vec_dPhi_A_pts = np.hstack(( Vec_X_A_dPhi_quad_vals, Vec_Y_A_dPhi_quad_vals, Vec_Z_A_dPhi_quad_vals ))

			Vec_dTheta_B_pts = np.hstack(( Vec_X_B_dTheta_quad_vals, Vec_Y_B_dTheta_quad_vals, Vec_Z_B_dTheta_quad_vals ))
			Vec_dPhi_B_pts = np.hstack(( Vec_X_B_dPhi_quad_vals, Vec_Y_B_dPhi_quad_vals, Vec_Z_B_dPhi_quad_vals ))

			# derivatives of metric factor, over metric factor:
			dMet_Fac_dTheta_over_Met_Fac_A_pts = self.Manifold.Metric_Factor_dTheta_over_Metric_Factor_A_pts
			dMet_Fac_dPhi_over_Met_Fac_A_pts = self.Manifold.Metric_Factor_dPhi_over_Metric_Factor_A_pts

			dMet_Fac_dTheta_over_Met_Fac_B_pts = self.Manifold.Metric_Factor_dTheta_over_Metric_Factor_B_pts
			dMet_Fac_dPhi_over_Met_Fac_B_pts = self.Manifold.Metric_Factor_dPhi_over_Metric_Factor_B_pts


			# A^{-1} V
			V_local_coors_theta_A = np.zeros(( self.Q_value, 1 ))
			V_local_coors_phi_A = np.zeros(( self.Q_value, 1 ))

			V_local_coors_theta_B = np.zeros(( self.Q_value, 1 ))
			V_local_coors_phi_B = np.zeros(( self.Q_value, 1 ))

			# A^{-1}_i V = (-A^{-1} A_i A^{-1} ) V
			V_local_coors_theta_dtheta_Amat_A = np.zeros(( self.Q_value, 1 ))
			V_local_coors_phi_dphi_Amat_A = np.zeros(( self.Q_value, 1 ))

			V_local_coors_theta_dtheta_Bmat_B = np.zeros(( self.Q_value, 1 ))
			V_local_coors_phi_dphi_Bmat_B = np.zeros(( self.Q_value, 1 ))

			# A^{-1} V_i
			V_local_coors_theta_dtheta_V_A = np.zeros(( self.Q_value, 1 ))
			V_local_coors_phi_dphi_V_A = np.zeros(( self.Q_value, 1 ))

			V_local_coors_theta_dtheta_V_B = np.zeros(( self.Q_value, 1 ))
			V_local_coors_phi_dphi_V_B = np.zeros(( self.Q_value, 1 ))


			manny_MF_tol = self.Manifold.Tol
			met_fac_A_pts = self.Manifold.Metric_Factor_A_pts
			met_fac_B_pts = self.Manifold.Metric_Factor_B_pts


			# for testing A^-1:
			g_inv_theta_theta_A_pts = self.Manifold.g_Inv_Theta_Theta_A_Pts
			g_inv_theta_phi_A_pts = self.Manifold.g_Inv_Theta_Phi_A_Pts
			g_inv_phi_phi_A_pts = self.Manifold.g_Inv_Phi_Phi_A_Pts
			All_A_inv_successful = True

			g_inv_theta_theta_B_pts = self.Manifold.g_Inv_Theta_Theta_B_Pts
			g_inv_theta_phi_B_pts = self.Manifold.g_Inv_Theta_Phi_B_Pts
			g_inv_phi_phi_B_pts = self.Manifold.g_Inv_Phi_Phi_B_Pts
			All_B_inv_successful = True

			for quad_pt in range(self.Manifold.num_quad_pts):
				if(lbdv.Chart_of_Quad_Pts[quad_pt] > 0): ### !!!! USE METRIC FACTOR !!!!
					#if(met_fac_A_pts[quad_pt] > manny_MF_tol):

					Vec_X_A_pt = self.quad_pt_array[quad_pt, :].T
					dTheta_Vec_X_A_pt = Vec_dTheta_A_pts[quad_pt, :].T
					dPhi_Vec_X_A_pt = Vec_dPhi_A_pts[quad_pt, :].T

					# Change of Basis mats

					A_Mat_pt = self.Manifold.Change_Basis_Mat(quad_pt, 'A')
					#print("det(A_Mat_pt) = "+str( np.linalg.det(A_Mat_pt) )+", met_fac_A_pts[quad_pt] = "+str(met_fac_A_pts[quad_pt]) +", A_Mat_pt = "+str(A_Mat_pt) )
					A_theta_Mat_pt = self.Manifold.dChange_Basis_Mat_theta(quad_pt, 'A')
					A_phi_Mat_pt = self.Manifold.dChange_Basis_Mat_phi(quad_pt, 'A')

					# Test A^-1 = A^T G^-1:
					G_inv_pt = np.zeros(( 3, 3 ))
					G_inv_pt[2,2] = 1.
					G_inv_pt[0,0] = g_inv_theta_theta_A_pts[quad_pt, 0]
					G_inv_pt[0,1] = g_inv_theta_phi_A_pts[quad_pt, 0]
					G_inv_pt[1,0] = g_inv_theta_phi_A_pts[quad_pt, 0]
					G_inv_pt[1,1] = g_inv_phi_phi_A_pts[quad_pt, 0]
					A_inv_from_G_pt = np.dot(G_inv_pt, A_Mat_pt.T)
					Id_test = np.dot(A_inv_from_G_pt, A_Mat_pt)
					if( np.linalg.norm(Id_test - np.eye(3) , 2) > 1.e-8 and debug_mode == True):
						print("A_inv test failed at pt: "+str(quad_pt)+", Id_test = "+str(Id_test))
						All_A_inv_successful = False

					Vec_X_local_coors_A_pt = np.dot(A_inv_from_G_pt, Vec_X_A_pt) #np.linalg.solve(A_Mat_pt, Vec_X_A_pt)

					'''
					# test new A^-1:
					Vec_X_local_coors_A_test_pt = np.dot(A_inv_from_G_pt, Vec_X_A_pt)
					norm_err_A =  np.linalg.norm(Vec_X_local_coors_A_pt - Vec_X_local_coors_A_test_pt , 2)
					if( norm_err_A > 1.e-8 and debug_mode == True):
						print("A_inv test failed at pt: "+str(quad_pt)+", norm_err_A = "+str(norm_err_A))
					'''

					V_local_coors_theta_A[quad_pt, 0] = Vec_X_local_coors_A_pt[0]
					V_local_coors_phi_A[quad_pt, 0] = Vec_X_local_coors_A_pt[1]

					dTheta_Vec_X_local_coors_A_pt = np.dot(A_inv_from_G_pt, dTheta_Vec_X_A_pt)  #np.linalg.solve(A_Mat_pt, dTheta_Vec_X_A_pt)
					dPhi_Vec_X_local_coors_A_pt = np.dot(A_inv_from_G_pt, dPhi_Vec_X_A_pt)  #np.linalg.solve(A_Mat_pt, dPhi_Vec_X_A_pt)

					V_local_coors_theta_dtheta_V_A[quad_pt, 0] = dTheta_Vec_X_local_coors_A_pt[0]
					V_local_coors_phi_dphi_V_A[quad_pt, 0] = dPhi_Vec_X_local_coors_A_pt[1]

					neg_A_inv_A_theta = np.dot(A_inv_from_G_pt, A_theta_Mat_pt) #-1.*np.linalg.solve(A_Mat_pt, A_theta_Mat_pt)
					neg_A_inv_A_phi = np.dot(A_inv_from_G_pt, A_phi_Mat_pt) #-1.*np.linalg.solve(A_Mat_pt, A_phi_Mat_pt)

					V_local_coors_theta_dtheta_Amat_A[quad_pt, 0] = np.dot(neg_A_inv_A_theta, Vec_X_local_coors_A_pt)[0]
					V_local_coors_phi_dphi_Amat_A[quad_pt, 0] = np.dot(neg_A_inv_A_phi, Vec_X_local_coors_A_pt)[1]

					#if(met_fac_B_pts[quad_pt] > manny_MF_tol):

					quad_pt_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(quad_pt)

					Vec_X_B_pt = self.quad_pt_array[quad_pt_rot, :].T
					dTheta_Vec_X_B_pt = Vec_dTheta_B_pts[quad_pt, :].T # these are already in B-Coors
					dPhi_Vec_X_B_pt = Vec_dPhi_B_pts[quad_pt, :].T

					B_Mat_pt = self.Manifold.Change_Basis_Mat(quad_pt, 'B')
					B_theta_Mat_pt = self.Manifold.dChange_Basis_Mat_theta(quad_pt, 'B')
					B_phi_Mat_pt = self.Manifold.dChange_Basis_Mat_phi(quad_pt, 'B')

					# Test B^-1 = B^T G_B^-1:
					G_B_inv_pt = np.zeros(( 3, 3 ))
					G_B_inv_pt[2,2] = 1.
					G_B_inv_pt[0,0] = g_inv_theta_theta_B_pts[quad_pt, 0]
					G_B_inv_pt[0,1] = g_inv_theta_phi_B_pts[quad_pt, 0]
					G_B_inv_pt[1,0] = g_inv_theta_phi_B_pts[quad_pt, 0]
					G_B_inv_pt[1,1] = g_inv_phi_phi_B_pts[quad_pt, 0]
					B_inv_from_G_pt = np.dot(G_B_inv_pt, B_Mat_pt.T)
					Id_test_B = np.dot(B_inv_from_G_pt, B_Mat_pt)
					if( np.linalg.norm(Id_test_B - np.eye(3) , 2) > 1.e-8 and debug_mode == True):
						print("B_inv test failed at pt: "+str(quad_pt))
						All_B_inv_successful = False

					Vec_X_local_coors_B_pt = np.dot(B_inv_from_G_pt, Vec_X_B_pt) #np.linalg.solve(B_Mat_pt, Vec_X_B_pt)

					V_local_coors_theta_B[quad_pt, 0] = Vec_X_local_coors_B_pt[0]
					V_local_coors_phi_B[quad_pt, 0] = Vec_X_local_coors_B_pt[1]

					dTheta_Vec_X_local_coors_B_pt = np.linalg.solve(B_Mat_pt, dTheta_Vec_X_B_pt)
					dPhi_Vec_X_local_coors_B_pt = np.linalg.solve(B_Mat_pt, dPhi_Vec_X_B_pt)

					V_local_coors_theta_dtheta_V_B[quad_pt, 0] = dTheta_Vec_X_local_coors_B_pt[0]
					V_local_coors_phi_dphi_V_B[quad_pt, 0] = dPhi_Vec_X_local_coors_B_pt[1]

					neg_B_inv_B_theta = -1.*np.linalg.solve(B_Mat_pt, B_theta_Mat_pt)
					neg_B_inv_B_phi = -1.*np.linalg.solve(B_Mat_pt, B_phi_Mat_pt)

					V_local_coors_theta_dtheta_Bmat_B[quad_pt, 0] = np.dot(neg_B_inv_B_theta, Vec_X_local_coors_B_pt)[0]
					V_local_coors_phi_dphi_Bmat_B[quad_pt, 0] = np.dot(neg_B_inv_B_phi, Vec_X_local_coors_B_pt)[1]


			div_V_A_pts = dMet_Fac_dTheta_over_Met_Fac_A_pts*V_local_coors_theta_A + dMet_Fac_dPhi_over_Met_Fac_A_pts*V_local_coors_phi_A + V_local_coors_theta_dtheta_Amat_A + V_local_coors_phi_dphi_Amat_A + V_local_coors_theta_dtheta_V_A + V_local_coors_phi_dphi_V_A

			div_V_B_pts = dMet_Fac_dTheta_over_Met_Fac_B_pts*V_local_coors_theta_B + dMet_Fac_dPhi_over_Met_Fac_B_pts*V_local_coors_phi_B + V_local_coors_theta_dtheta_Bmat_B + V_local_coors_phi_dphi_Bmat_B + V_local_coors_theta_dtheta_V_B + V_local_coors_phi_dphi_V_B

			div_V_quad_pts = Combine_Chart_Quad_Vals(div_V_A_pts, div_V_B_pts, lbdv)

			Debug_Dict = {}
			Debug_Dict['V_local_coors_theta_A'] = V_local_coors_theta_A
			Debug_Dict['V_local_coors_phi_A'] = V_local_coors_phi_A
			Debug_Dict['V_local_coors_theta_B'] = V_local_coors_theta_B
			Debug_Dict['V_local_coors_phi_B'] = V_local_coors_phi_B

			Debug_Dict['dTheta_V_local_coors_theta_A'] =  V_local_coors_theta_dtheta_Amat_A + V_local_coors_theta_dtheta_V_A
			Debug_Dict['dphi_V_local_coors_phi_A'] =  V_local_coors_phi_dphi_Amat_A + V_local_coors_phi_dphi_V_A
			Debug_Dict['dTheta_V_local_coors_theta_B'] =  V_local_coors_theta_dtheta_Bmat_B + V_local_coors_theta_dtheta_V_B
			Debug_Dict['dphi_V_local_coors_phi_B'] =  V_local_coors_phi_dphi_Bmat_B + V_local_coors_phi_dphi_V_B

			Debug_Dict['der_met_fac_comps_A_pts'] = dMet_Fac_dTheta_over_Met_Fac_A_pts*V_local_coors_theta_A + dMet_Fac_dPhi_over_Met_Fac_A_pts*V_local_coors_phi_A
			Debug_Dict['der_met_fac_comps_B_pts'] =  dMet_Fac_dTheta_over_Met_Fac_B_pts*V_local_coors_theta_B + dMet_Fac_dPhi_over_Met_Fac_B_pts*V_local_coors_phi_B

			Debug_Dict['dTheta_V_Euc_In_A_pts'] = Vec_dTheta_A_pts
			Debug_Dict['dPhi_V_Euc_In_A_pts'] = Vec_dPhi_A_pts
			Debug_Dict['dTheta_V_Euc_In_B_pts'] = Vec_dTheta_B_pts
			Debug_Dict['dPhi_V_Euc_In_B_pts'] = Vec_dPhi_B_pts

			Debug_Dict['div_V_A_pts'] =  div_V_A_pts
			Debug_Dict['div_V_B_pts'] =  div_V_B_pts

			if(debug_mode == True):
				print("\n"+"All_A_inv_successful = "+str(All_A_inv_successful))
				print("All_B_inv_successful = "+str(All_B_inv_successful)+"\n")
				return Debug_Dict
			else:
				return euc_k_form(0, self.Q_value, self.P_degree, self.Manifold, div_V_quad_pts)

		else:
			print("\n"+"Error: Divergence only defined on 1-Forms"+"\n")

	# Write Expressions in 1 line (COPIED FROM LB_POINT_VALS_SPB):
	def Explicit_LB(self, lbdv):

		Y_mn_SPH_Fn_A, Y_mn_SPH_Fn_B = sph_f.Proj_Into_SPH_Charts_At_Quad_Pts(self.quad_pt_array, self.P_degree, lbdv)

		Y_mn_SPH_Fn_dTheta_A = Y_mn_SPH_Fn_A.Quick_Theta_Der()
		Y_mn_SPH_Fn_dTheta_Theta_A = Y_mn_SPH_Fn_A.Quick_Theta_Der().Quick_Theta_Der()

		Y_mn_A_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_dTheta_A, lbdv, 'A')
		Y_mn_A_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_A, lbdv, 'A')
		Y_mn_A_dTheta_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_dTheta_A, lbdv, 'A')
		Y_mn_A_dTheta_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_dTheta_Theta_A, lbdv, 'A')
		Y_mn_A_dPhi_dPhi_quad_vals = Extract_dPhi_Phi_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_A, lbdv, 'A')


		Y_mn_SPH_Fn_dTheta_B = Y_mn_SPH_Fn_B.Quick_Theta_Der()
		Y_mn_SPH_Fn_dTheta_Theta_B = Y_mn_SPH_Fn_B.Quick_Theta_Der().Quick_Theta_Der()

		Y_mn_B_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_dTheta_B, lbdv, 'A')
		Y_mn_B_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_B, lbdv, 'A')
		Y_mn_B_dTheta_dPhi_quad_vals = Extract_dPhi_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_dTheta_B, lbdv, 'A')
		Y_mn_B_dTheta_dTheta_quad_vals = Extract_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_dTheta_Theta_B, lbdv, 'A')
		Y_mn_B_dPhi_dPhi_quad_vals = Extract_dPhi_Phi_Quad_Pt_Vals_From_SPH_Fn(Y_mn_SPH_Fn_B, lbdv, 'A')


		LB_Basis_Fn_A_pts = 1.0/(self.Manifold.Metric_Factor_A_pts)*(self.Manifold.G_over_Metric_Factor_A_pts*Y_mn_A_dTheta_dTheta_quad_vals - 2*self.Manifold.F_over_Metric_Factor_A_pts*Y_mn_A_dTheta_dPhi_quad_vals + self.Manifold.E_over_Metric_Factor_A_pts*Y_mn_A_dPhi_dPhi_quad_vals + self.Manifold.G_over_Metric_Factor_dTheta_A*Y_mn_A_dTheta_quad_vals - self.Manifold.F_over_Metric_Factor_dTheta_A*Y_mn_A_dPhi_quad_vals - self.Manifold.F_over_Metric_Factor_dPhi_A*Y_mn_A_dTheta_quad_vals + self.Manifold.E_over_Metric_Factor_dPhi_A*Y_mn_A_dPhi_quad_vals)


		LB_Basis_Fn_B_pts = 1.0/(self.Manifold.Metric_Factor_B_pts)*(self.Manifold.G_over_Metric_Factor_B_pts*Y_mn_B_dTheta_dTheta_quad_vals - 2*self.Manifold.F_over_Metric_Factor_B_pts*Y_mn_B_dTheta_dPhi_quad_vals + self.Manifold.E_over_Metric_Factor_B_pts*Y_mn_B_dPhi_dPhi_quad_vals + self.Manifold.G_over_Metric_Factor_dTheta_B*Y_mn_B_dTheta_quad_vals - self.Manifold.F_over_Metric_Factor_dTheta_B*Y_mn_B_dPhi_quad_vals - self.Manifold.F_over_Metric_Factor_dPhi_B*Y_mn_B_dTheta_quad_vals + self.Manifold.E_over_Metric_Factor_dPhi_B*Y_mn_B_dPhi_quad_vals)


		LB_quad_pts = Combine_Chart_Quad_Vals(LB_Basis_Fn_A_pts, LB_Basis_Fn_B_pts, lbdv)
		return euc_k_form(0, self.Q_value, self.P_degree, self.Manifold, LB_quad_pts)


	# Uses curl operators to compute LB of Zero Forms, to test whether we get better convergence this way (by ignoring intermediate k-form stuctures)
	def LB_Zero_Form_From_Curl(self, lbdv):

		if(self.k_value != 0):
			print("Error: Zero Forms needed for this inner-product fn to work")

		else:
			# (-1)*LB = [(-*d)_1][(-*d)_0]:
			gen_curl0_f_A_theta_pts, gen_curl0_f_A_phi_pts, gen_curl0_f_B_theta_pts, gen_curl0_f_B_phi_pts = self.Gen_Curl_0(lbdv)
			neg_LB_f_0_Form =  Gen_Curl_1(gen_curl0_f_A_theta_pts, gen_curl0_f_A_phi_pts, gen_curl0_f_B_theta_pts, gen_curl0_f_B_phi_pts, self.P_degree, lbdv, self.Manifold)

			return neg_LB_f_0_Form.times_const(-1)

	'''
	# Compute Co-Differential: k_form -> (k-1)_form
	def Co_Diff(self, lbdv):
		n = 2 #2-D manifold
		k = self.k_value

		return self.Hodge_Star(lbdv).Ext_Der(lbdv).Hodge_Star(lbdv).times_const((-1)**(n*(k+1)+1)) #Multiply by - 1, result is (k-1)-form
	'''

	# LB: k_form -> k_form (d\delta + \delta d) = (d + \delta)^2
	def Laplace_Beltrami(self, lbdv):
		if(self.k_value != 2):
			First_Term =  self.Ext_Der(lbdv).Co_Diff(lbdv) #zero for K==2

			if(self.k_value == 0):
				return First_Term

			if(self.k_value == 1):
				Second_Term = self.Co_Diff(lbdv).Ext_Der(lbdv) #zero for K==0
				return linear_comb(First_Term, Second_Term, 1, 1)

		if(self.k_value == 2):
			Second_Term = self.Co_Diff(lbdv).Ext_Der(lbdv)
			return Second_Term


	# Returns Riemannian Inner Product of 1-Forms input, returns vector of scalar inner-products at each point
	def Riemann_Inner_Prouct_One_Form(self, One_Form_2, lbdv):

		# if we have faulty input
		if(self.k_value != 1 or One_Form_2.k_value != 1):

			print("Error: One Forms needed for this inner-product fn to work")

		else:
			Inner_Product_One_Form_Vals_pts =  np.zeros(( self.Q_value, 1 ))

			'''
			# Hard to vectorize matrix solves
			for quad_pt in range(self.Q_value):

				# If Chart A:
				if(lbdv.Chart_of_Quad_Pts[quad_pt] > 0):

					alpha_1_sharp_X_A_pt = self.quad_pt_array[quad_pt, :].T
					alpha_2_sharp_X_A_pt = One_Form_2.quad_pt_array[quad_pt, :].T

					Flat_A_mat = self.Manifold.rho_A_Mats[:, :, quad_pt]

					one_form_comps_A_pt = np.dot(Flat_A_mat, alpha_1_sharp_X_A_pt)

					alpha_1_flat_theta_A_pt = one_form_comps_A_pt[0]
					alpha_1_flat_phi_A_pt = one_form_comps_A_pt[1]

					A_Mat_pt = self.Manifold.Change_Basis_Mat(quad_pt, lbdv, 'A')
					alpha_2_sharp_A_pt = np.linalg.solve(A_Mat_pt, alpha_2_sharp_X_A_pt)

					alpha_2_sharp_theta_A_pt = alpha_2_sharp_A_pt[0]
					alpha_2_sharp_phi_A_pt = alpha_2_sharp_A_pt[1]

					#I_A_mat_pt = self.Manifold.I_Mat_Quad_Pt(quad_pt, lbdv, 'A')

					Inner_Product_A_pt = alpha_2_sharp_theta_A_pt*alpha_1_flat_theta_A_pt + alpha_2_sharp_phi_A_pt*alpha_1_flat_phi_A_pt

					Inner_Product_One_Form_Vals_pts[quad_pt, 0] = Inner_Product_A_pt

				# We fill in other pts with Chart B:
				else:

					quad_pt_rot = lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pt)

					#same index because we more efficiently split the computation, to avoid overlap in calculation
					alpha_1_sharp_X_B_pt = self.quad_pt_array[quad_pt, :].T
					alpha_2_sharp_X_B_pt = One_Form_2.quad_pt_array[quad_pt, :].T

					Flat_B_mat = self.Manifold.rho_B_Mats[:, :, quad_pt_rot]

					one_form_comps_B_pt = np.dot(Flat_B_mat, alpha_1_sharp_X_B_pt)

					alpha_1_flat_theta_B_pt = one_form_comps_B_pt[0]
					alpha_1_flat_phi_B_pt = one_form_comps_B_pt[1]

					B_Mat_pt = self.Manifold.Change_Basis_Mat(quad_pt_rot, lbdv, 'B')
					alpha_2_sharp_B_pt = np.linalg.solve(B_Mat_pt, alpha_2_sharp_X_B_pt)

					alpha_2_sharp_theta_B_pt = alpha_2_sharp_B_pt[0]
					alpha_2_sharp_phi_B_pt = alpha_2_sharp_B_pt[1]

					#I_B_mat_pt = self.Manifold.I_Mat_Quad_Pt(quad_pt_rot, lbdv, 'B')

					Inner_Product_B_pt = alpha_2_sharp_theta_B_pt*alpha_1_flat_theta_B_pt + alpha_2_sharp_phi_B_pt*alpha_1_flat_phi_B_pt

					Inner_Product_One_Form_Vals_pts[quad_pt, 0] = Inner_Product_B_pt
			'''

			Inner_Product_One_Form_Vals_pts = np.dot(self.quad_pt_array, One_Form_2.quad_pt_array.T).diagonal().reshape(( self.Q_value, 1 ))

		return Inner_Product_One_Form_Vals_pts


	# Returns Riemannian Inner Product of 2-Forms with ITSELF input, returns vector of scalar inner-products at each point
	def Riemann_Self_Inner_Prouct_Two_Form(self, lbdv):



		# Corresponding quad pt in chart B (in Chart A Coors)
		quad_pts = range(self.Q_value)
		quad_pts_inv_rot = lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(quad_pts)

		# inv metric factor, at each point, within Chart
		#inv_met_fac_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1/self.Manifold.Metric_Factor_Quad_Pt(quad_pts, lbdv, 'A'), 0)
		#inv_met_fac_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1/self.Manifold.Metric_Factor_Quad_Pt(quad_pts, lbdv, 'B'), 0)

		#BJG: New Change:

		inv_met_fac_A_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1./self.Manifold.Metric_Factor_A_pts, 0)
		inv_met_fac_B_pts = np.where(lbdv.Chart_of_Quad_Pts > 0, 1./self.Manifold.Metric_Factor_B_pts, 0)


		# values at (theta_C, phi_C) = (theta_i, phi_i), given q_i, Difffernt points on manifold!
		beta_dx_dy_A_pts, beta_dx_dz_A_pts, beta_dy_dz_A_pts = np.hsplit(self.quad_pt_array[quad_pts], 3)
		beta_dx_dy_B_pts, beta_dx_dz_B_pts, beta_dy_dz_B_pts = np.hsplit(self.quad_pt_array[quad_pts_inv_rot], 3)

		# 2-forms to cartesian conversions:
		dx_dy_to_dtheta_dphi_A_pts, dx_dz_to_dtheta_dphi_A_pts, dy_dz_to_dtheta_dphi_A_pts = Two_Form_Conv_to_Polar_pt(quad_pts, lbdv, self.Manifold, 'A')
		dx_dy_to_dtheta_dphi_B_pts, dx_dz_to_dtheta_dphi_B_pts, dy_dz_to_dtheta_dphi_B_pts = Two_Form_Conv_to_Polar_pt(quad_pts, lbdv, self.Manifold, 'B')

		# Do Hodge Star:
		beta_dtheta_dphi_A_pts = np.multiply(beta_dx_dy_A_pts, dx_dy_to_dtheta_dphi_A_pts) + np.multiply(beta_dy_dz_A_pts, dy_dz_to_dtheta_dphi_A_pts) +  np.multiply(beta_dx_dz_A_pts, dx_dz_to_dtheta_dphi_A_pts)

		beta_dtheta_dphi_B_pts =  np.multiply(beta_dx_dy_B_pts, dx_dy_to_dtheta_dphi_B_pts) + np.multiply(beta_dy_dz_B_pts, dy_dz_to_dtheta_dphi_B_pts) + np.multiply(beta_dx_dz_B_pts, dx_dz_to_dtheta_dphi_B_pts)

		inv_mf_beta_sq_from_A_pts = np.multiply(beta_dtheta_dphi_A_pts, inv_met_fac_A_pts)**2
		inv_mf_beta_sq_from_B_pts = np.multiply(beta_dtheta_dphi_B_pts, inv_met_fac_B_pts)**2


		# Combine Values from Charts
		return Combine_Chart_Quad_Vals(inv_mf_beta_sq_from_A_pts, inv_mf_beta_sq_from_B_pts , lbdv)




	# Pushes forward a 1-form onto Manny_2, using our chart convention
	def Push_Forward_One_Form(self, Manny_2, lbdv):

		if(self.k_value != 1):

			print("Error: One Form needed for this Push-Forward fn to work")

		else:

			push_forward_quad_vals = np.zeros(( self.Q_value, 3 ))

			sigma_theta_A_vecs = Manny_2.sigma_theta_Quad_Pts(lbdv, 'A')
			sigma_theta_B_vecs = Manny_2.sigma_theta_Quad_Pts(lbdv, 'B')

			sigma_phi_A_vecs = Manny_2.sigma_phi_Quad_Pts(lbdv, 'A')
			sigma_phi_B_vecs = Manny_2.sigma_phi_Quad_Pts(lbdv, 'B')

			for quad_pt in range(self.Q_value):

				#print("q = "+str(quad_pt))

				# If Chart A:
				if(lbdv.Chart_of_Quad_Pts[quad_pt] > 0):

					alpha_sharp_X_A_pt = self.quad_pt_array[quad_pt, :].T

					A_Mat_pt = self.Manifold.Change_Basis_Mat(quad_pt, 'A')
					alpha_sharp_A_pt = np.linalg.solve(A_Mat_pt, alpha_sharp_X_A_pt)

					alpha_sharp_theta_A_pt = alpha_sharp_A_pt[0]
					alpha_sharp_phi_A_pt = alpha_sharp_A_pt[1]

					pf_sigma_theta_A_pt = sigma_theta_A_vecs[:, quad_pt][0]
					pf_sigma_phi_A_pt = sigma_phi_A_vecs[:, quad_pt][0]

					#print("pf_sigma_phi_A_pt ="+str(pf_sigma_phi_A_pt))

					push_forward_vec_A_pt = alpha_sharp_theta_A_pt*pf_sigma_theta_A_pt + alpha_sharp_phi_A_pt*pf_sigma_phi_A_pt

					#print("push_forward_vec_A_pt = "+str(push_forward_vec_A_pt))

					push_forward_quad_vals[quad_pt, :] =push_forward_vec_A_pt.T

				else:

					quad_pt_rot = lbdv.Eval_Rot_Lbdv_Quad_vals(quad_pt)

					alpha_sharp_X_B_pt = self.quad_pt_array[quad_pt, :].T

					B_Mat_pt = self.Manifold.Change_Basis_Mat(quad_pt_rot, 'B')
					alpha_sharp_B_pt = np.linalg.solve(B_Mat_pt, alpha_sharp_X_B_pt)

					alpha_sharp_theta_B_pt = alpha_sharp_B_pt[0]
					alpha_sharp_phi_B_pt = alpha_sharp_B_pt[1]

					pf_sigma_theta_B_pt = sigma_theta_B_vecs[:, quad_pt_rot][0]
					pf_sigma_phi_B_pt = sigma_phi_B_vecs[:, quad_pt_rot][0]

					push_forward_vec_B_pt = alpha_sharp_theta_B_pt*pf_sigma_theta_B_pt + alpha_sharp_phi_B_pt*pf_sigma_phi_B_pt

					push_forward_quad_vals[quad_pt, :] = push_forward_vec_B_pt.T

			# return 1-form on Manny_2
			return euc_k_form(1, self.Q_value, self.P_degree, Manny_2, push_forward_quad_vals)
