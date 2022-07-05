# From https://github.com/campaslab/STRESS
#! We use spherical_harmonics_function to encapsulate data from SPH Basis of functions, regardless of chart

from numpy import *
import numpy as np
from scipy  import *
import mpmath
import cmath
from scipy.special import sph_harm

from .lebedev_info_SPB import *
from .charts_SPB import*

import pyshtools


# Diagonal of Mass matrix is 1, non-diagonal is 1/2 (due to |Re(Y^m_l)|=|Im(Y^m_l)|)
def Mass_Mat_Exact(m_coef,n_coef):
    if(m_coef==0):
        return 1
    else:
        return .5


# Inverse Mass Matrix on S2 pullback (NOT INCLUDING constant mode
def Inv_Mass_Mat_on_Pullback(SPH_Deg):

        mat_dim = (SPH_Deg+1)**2 -1
        inv_mass_mat = np.zeros((mat_dim, mat_dim ))

        diag_entry = 0

        for n in range(1,SPH_Deg+1):
                for m in range(-1*n, n+1):

                        inv_mass_mat[diag_entry, diag_entry] = 1.0/ Mass_Mat_Exact(m,n)

        return inv_mass_mat


# Creates a SPH representation of Y^m_n
def Create_Basis_Fn(m, n, basis_degree):
    Coef_Mat = zeros(( basis_degree+1, basis_degree+1))

    if(m >= 0):
        Coef_Mat[n-m][n] = 1.
    else: # If m<0
        Coef_Mat[n][n-(-1*m)] = 1.

    return spherical_harmonics_function(Coef_Mat, basis_degree)


#Approximate func(theta, phi) in FE space
def Proj_Func(func, SPH_Deg, lbdv):

    Proj_Coef = zeros([SPH_Deg+1, SPH_Deg+1])

    theta_quad_pts = lbdv.Lbdv_Sph_Pts_Quad[:, 0]
    phi_quad_pts = lbdv.Lbdv_Sph_Pts_Quad[:, 1]

    func_vals_at_quad_pts = func(theta_quad_pts, phi_quad_pts)
    #print("func_vals_at_quad_pts= "+str(func_vals_at_quad_pts))

    #Calculate lebedev points for quadrature

    #print(Lbdv_Sph_Pts_der_Quad)

    #Compute inner product of theta der with each basis elt
    for n in range(SPH_Deg+1):
        for m in range(-1*n, n+1):

            Proj_Coef_mn = sum(multiply(lbdv.Eval_SPH_Basis_Wt_M_N(m, n), func_vals_at_quad_pts))/Mass_Mat_Exact(m,n)

            if m>0:
                        Proj_Coef[n-m][n] = Proj_Coef_mn

            else: #m <= 0
                         Proj_Coef[n][n+m] = Proj_Coef_mn


    return spherical_harmonics_function(Proj_Coef, SPH_Deg)


# Projects function into both charts
def Proj_Into_SPH_Charts(func, Coef_Deg, lbdv):

    #In chart A
    #Coef_A = Proj_Func(lambda theta, phi: eta_A(func, theta, phi), Coef_Deg)
    sph_func_A = Proj_Func(func, Coef_Deg, lbdv) #using class structure

    #func for chart B
    #Coef_B = Proj_Func(lambda theta_bar, phi_bar: eta_B(func, theta_bar, phi_bar), Coef_Deg)

    # rotate func to Chart B Coors
    def func_rot(theta_bar, phi_bar):

        A_Coors = Coor_B_To_A(theta_bar, phi_bar)

        Theta_Vals = A_Coors[0]
        Phi_Vals = A_Coors[1]

        func_vals = func(Theta_Vals, Phi_Vals) #zeros(shape(theta_bar))

        return func_vals


    sph_func_B = Proj_Func(func_rot, Coef_Deg, lbdv) #using class structure

    return (sph_func_A, sph_func_B) #should agree where charts intersect


# Projects function into both charts using quad_pts
def Proj_Into_SPH_Charts_At_Quad_Pts(func_quad_vals, Proj_Deg, lbdv):

    #In chart A
    sph_func_A = Faster_Double_Proj(func_quad_vals, Proj_Deg, lbdv) #using class structure

    #vals for chart B

    quad_pt_vals_B = zeros(( lbdv.lbdv_quad_pts, 1))

    # rotate func to Chart B Coors
    for quad_pt in range(lbdv.lbdv_quad_pts):

        quad_pt_vals_B[quad_pt] = func_quad_vals[lbdv.Eval_Inv_Rot_Lbdv_Quad_vals(quad_pt)]


    sph_func_B = Faster_Double_Proj(quad_pt_vals_B, Proj_Deg, lbdv) #using class structure

    return sph_func_A, sph_func_B #should agree where charts intersect


#TAKES VALS AT QUAD PTS, to Approximate func(theta, phi) in SPH Basis
def Faster_Double_Proj(func_quad_vals, Proj_Deg, lbdv):

    Proj_Coef = np.zeros([Proj_Deg+1, Proj_Deg+1]) #Size of basis used to represent derivative

    #Compute inner product of theta der with each basis elt
    for n in range(Proj_Deg+1):
        for m in range(-1*n, n+1):

            I_mn = 0

            for quad_pt in range(lbdv.lbdv_quad_pts):
                #theta_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][0]
                #phi_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][1]

                I_mn += func_quad_vals[quad_pt]*lbdv.Eval_SPH_Basis_Wt_At_Quad_Pts(m,n, quad_pt)
                #Above fn sums basis vals for proj, times func,  times weight at each quad pt


            Proj_mn = I_mn/Mass_Mat_Exact(m,n)

            if m>0:
                    Proj_Coef[n-m][n] = Proj_mn

            else: #m <= 0
                    Proj_Coef[n][n+m] = Proj_mn

    return spherical_harmonics_function(Proj_Coef, Proj_Deg)


#TAKES VALS AT QUAD PTS, to Approximate func(theta, phi)*Coef_Mat(theta, phi) in SPH Basis
def Faster_Double_Proj_Product(func1_quad_vals, func2_quad_vals, Proj_Deg, lbdv):


    Proj_Product_Coef = zeros([Proj_Deg+1, Proj_Deg+1]) #Size of basis used to represent derivative

    #Compute inner product of theta der with each basis elt
    for n in range(Proj_Deg+1):
        for m in range(-1*n, n+1):

            I_mn = 0

            for quad_pt in range(lbdv.lbdv_quad_pts):
                #theta_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][0]
                #phi_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][1]

                I_mn += func1_quad_vals[quad_pt]*func2_quad_vals[quad_pt]*lbdv.Eval_SPH_Basis_Wt_At_Quad_Pts(m,n, quad_pt)
                #Above fn sums basis vals for proj, times func*Coef_Mat,  times weight at each quad pt


            Proj_Product_mn = I_mn/Mass_Mat_Exact(m,n)

            if m>0:
                    Proj_Product_Coef[n-m][n] = Proj_Product_mn

            else: #m <= 0
                    Proj_Product_Coef[n][n+m] = Proj_Product_mn


    return spherical_harmonics_function(Proj_Product_Coef, Proj_Deg)


# Inputs quad vals for f, f_approx, integrates on SPHERE:
def Lp_Rel_Error_At_Quad(approx_f_vals, f_vals, lbdv, p): #Assumes f NOT 0

    Lp_Err = 0 # ||self - f||_p
    Lp_f = 0  # || f ||_p

    for quad_pt in range(lbdv.lbdv_quad_pts):

        weight_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][2]

        Lp_Err_Pt = abs((approx_f_vals[quad_pt] - f_vals[quad_pt])**p)*weight_pt
        Lp_f_Pt = abs(f_vals[quad_pt]**p)*weight_pt

        '''
        if(Lp_Err_Pt < 0):
            print("Lp_Err_Pt = "+str(Lp_Err_Pt)+" at pt = "+str(quad_pt))
            print("(approx_f_vals[quad_pt] - f_vals[quad_pt])**p = "+str( (approx_f_vals[quad_pt] - f_vals[quad_pt])**p ))
            print("abs((approx_f_vals[quad_pt] - f_vals[quad_pt])**p) = "+str( abs((approx_f_vals[quad_pt] - f_vals[quad_pt])**p) ))
            print("weight_pt = "+str(weight_pt))
            print("\n")
        '''
        Lp_Err += Lp_Err_Pt
        Lp_f += Lp_f_Pt


    #print("Lp_Err = "+str(Lp_Err))
    #print("Lp_f = "+str(Lp_f))


    return     (Lp_Err/Lp_f)**(1./p) #||f_approx - f||_p / || f ||_p


# Inputs quad vals for f, f_approx, integrates on CHART:
def Lp_Rel_Error_At_Quad_In_Chart(approx_f_vals, f_vals, lbdv, p): #Assumes f NOT 0

    Lp_Err = 0 # ||self - f||_p
    Lp_f = 0  # || f ||_p

    for quad_pt in range(lbdv.lbdv_quad_pts):

        if(lbdv.Chart_of_Quad_Pts[quad_pt] > 0):
            weight_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][2]

            Lp_Err_Pt = abs((approx_f_vals[quad_pt] - f_vals[quad_pt])**p)*weight_pt
            Lp_f_Pt = abs(f_vals[quad_pt]**p)*weight_pt

            Lp_Err += Lp_Err_Pt
            Lp_f += Lp_f_Pt

    return     (Lp_Err/Lp_f)**(1./p) #||f_approx - f||_p / || f ||_p



# Computes Coef of constant mode of Fn
def Const_SPH_Mode_Of_Func(func, lbdv):

    theta_quad_pts = lbdv.Lbdv_Sph_Pts_Quad[:, 0]
    phi_quad_pts = lbdv.Lbdv_Sph_Pts_Quad[:, 1]

    func_vals_at_quad_pts = func(theta_quad_pts, phi_quad_pts)

    Proj_Coef_Const_Mode = sum(multiply(lbdv.Eval_SPH_Basis_Wt_M_N(0, 0), func_vals_at_quad_pts))/Mass_Mat_Exact(0, 0)

    return Proj_Coef_Const_Mode


# Computes Average of sph proj of quad pts
def Avg_of_SPH_Proj_of_Func(func_vals_at_quad_pts, lbdv):



    Proj_Coef_Const_Mode = sum(multiply(lbdv.Eval_SPH_Basis_Wt_M_N(0, 0), func_vals_at_quad_pts.flatten() ))/Mass_Mat_Exact(0, 0)
    Avg_of_SPH_Proj = Proj_Coef_Const_Mode*1./(2*np.sqrt(np.pi)) #multiply by height of Y^0_0

    return Avg_of_SPH_Proj


def Un_Flatten_Coef_Vec(coefficient_vector, basis_degree):
    """
    reshape a coefficient vector into a N+1 x N+1 coefficient matrix.
    Requires properly ordered vector of SPH Coef
    """

    coefficient_matrix = np.zeros((basis_degree+1, basis_degree+1 ))

    row = 0
    for n in range(basis_degree+1):
        for m in range(-1*n, n+1):

            if m>0:
                    coefficient_matrix[n-m][n] = coefficient_vector[row]

            else: #m <= 0
                    coefficient_matrix[n][n+m] = coefficient_vector[row]

            row = row + 1

    return coefficient_matrix

def convert_coefficients_pyshtools_to_stress(coefficients_pysh: pyshtools.SHCoeffs
                                             ) -> np.ndarray:
    """
    Convert a pyshtools-coefficient matrix to stress format.

    In stress format, the upper (np.triu) part of the coefficient matrix
    corresponds to the real part and the lower section (np.tril) correspond
    to the imaginary part. The diagonal parts are identical in both.

    Parameters
    ----------
    coefficients : pyshtools.SHCoeffs
    Returns
    -------
    stress_coefficients : np.ndarray

    """
    coefficients = coefficients_pysh.to_array()

    # Separate matrix into real/imag part. Note that pysh format leaves a
    # blank column for degree l=0 for the imaginary coefficients
    real = coefficients[0].transpose()
    imag = coefficients[1, 1:, 1:]
    imag_mask = np.tril(np.ones_like(imag)) == 1

    # create stress-compliant coefficient matrix
    stress_coefficients = real
    stress_coefficients[stress_coefficients == np.tril(stress_coefficients, -1)] = imag[imag_mask]

    return stress_coefficients


def convert_coeffcients_stress_to_pyshtools(coefficients_stress: np.ndarray
                                            )-> pyshtools.SHCoeffs:
    """
    Convert a stress-coefficient matrix (deg+1 x deg+1) to pyshtools format.

    The pyshtools format follows a (2, deg+1, deg+1) shape codex, whereas the
    first dimension refers to the cosine/sine parts of the expansion.
    """
    real = np.triu(coefficients_stress)
    imag = np.tril(coefficients_stress, -1)

    coefficients_pysh = np.zeros([2] + list(coefficients_stress.shape))
    coefficients_pysh[0] = real.transpose()
    coefficients_pysh[1, 1:, 1:] = imag[1:, :-1]

    return pyshtools.SHCoeffs.from_array(coefficients_pysh,
                                         lmax=coefficients_stress.shape[0]-1)


# Gives L_1 Integral on SPHERE pullback:
def L1_Integral(f_quad_vals, lbdv):

    L1_Int = 0

    for quad_pt in range(lbdv.lbdv_quad_pts):

        weight_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][2]
        L1_Int_Pt = abs(f_quad_vals[quad_pt])*weight_pt

        L1_Int += L1_Int_Pt

    return     L1_Int


# Gives Integral on SPHERE pullback:
def S2_Integral(f_quad_vals, lbdv):

    S2_Int = 0

    for quad_pt in range(lbdv.lbdv_quad_pts):

        weight_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][2]
        S2_Int_Pt = f_quad_vals[quad_pt,0]*weight_pt

        #print("weight_pt = "+str(weight_pt))
        #print("f_quad_vals[quad_pt,0] = "+str(f_quad_vals[quad_pt,0]))

        S2_Int += S2_Int_Pt

    return     S2_Int

#############################################################################################################################################################

class spherical_harmonics_function(object): #create class for Spherical Harmonics fn in our basis

    def __init__(self, SPH_Coef, SPH_Deg):
        self.sph_coef = SPH_Coef #Array Representation in SPH Basis
        self.sph_deg = SPH_Deg #maximum degree of SPH Basis (formely mislabeled as order)

    #Evaluates the Coef Matrix Representing SPH Basis of a Fn
    def Eval_SPH(self, Theta, Phi):

        SPH_Val = 0

        for L_Coef in range(0, self.sph_deg+1):
            for M_Coef in range(0, L_Coef+1):

                SPH_Val += self.sph_coef[L_Coef - M_Coef][L_Coef] *Eval_SPH_Basis(M_Coef, L_Coef, Theta, Phi)

                if(L_Coef >0 and M_Coef>0):
                    SPH_Val += self.sph_coef[L_Coef][L_Coef +(-1* M_Coef)] *Eval_SPH_Basis(-1*M_Coef, L_Coef, Theta, Phi)

        return SPH_Val

    #Evaluates the Phi Derivatibve of Coef Matrix Representing SPH Basis of a Fn
    def Eval_SPH_Der_Phi(self, Theta, Phi):

        Der_SPH_Val = 0

        for L_Coef in range(0, self.sph_deg+1):
            for M_Coef in range(0, L_Coef+1):

                Der_SPH_Val += self.sph_coef[L_Coef - M_Coef][L_Coef] *Der_Phi_Basis_Fn(M_Coef, L_Coef, Theta, Phi)

                if L_Coef >0 and M_Coef>0:
                    Der_SPH_Val += self.sph_coef[L_Coef][L_Coef +(-1* M_Coef)] *Der_Phi_Basis_Fn(-1*M_Coef, L_Coef, Theta, Phi)

        return Der_SPH_Val


    #Evaluates the Second Phi Derivatibve of Coef Matrix Representing SPH Basis of a Fn
    def Eval_SPH_Der_Phi_Phi(self, Theta, Phi):

        Sec_Der_SPH_Val = 0

        for L_Coef in range(0, self.sph_deg+1):
            for M_Coef in range(0, L_Coef+1):

                Sec_Der_SPH_Val += self.sph_coef[L_Coef - M_Coef][L_Coef] *Der_Phi_Phi_Basis_Fn(M_Coef, L_Coef, Theta, Phi)

                if L_Coef >0 and M_Coef>0:
                    Sec_Der_SPH_Val += self.sph_coef[L_Coef][L_Coef +(-1* M_Coef)] *Der_Phi_Phi_Basis_Fn(-1*M_Coef, L_Coef, Theta, Phi)

        return Sec_Der_SPH_Val


    #Should Evalaute Phi Der of Matrix at Quad_Pt (times weight), to be used in quadrature,
    def Eval_SPH_Der_Phi_Coef(self, Quad_Pt, lbdv):

        # For Scalar Case, we use usual vectorization:
        if(isscalar(Quad_Pt)):
            return sum(multiply(self.sph_coef, lbdv.Eval_SPH_Der_Phi_At_Quad_Pt_Mat(Quad_Pt)))

        # For multiple quad pts, we use einstein sumation to output vector of solutions at each point:
        else:
            return einsum('ij, ijk -> k', self.sph_coef, lbdv.Eval_SPH_Der_Phi_At_Quad_Pt_Mat(Quad_Pt)).reshape((len(Quad_Pt) ,1))


    #Should Evalaute 2nd Phi Der of Matrix at Quad_Pt (times weight), to be used in quadrature,
    def Eval_SPH_Der_Phi_Phi_Coef(self, Quad_Pt, lbdv):

        # For Scalar Case, we use usual vectorization:
        if(isscalar(Quad_Pt)):
            return sum(multiply(self.sph_coef, lbdv.Eval_SPH_Der_Phi_Phi_At_Quad_Pt_Mat(Quad_Pt)))

        # For multiple quad pts, we use einstein sumation to output vector of solutions at each point:
        else:
            return einsum('ij, ijk -> k', self.sph_coef, lbdv.Eval_SPH_Der_Phi_Phi_At_Quad_Pt_Mat(Quad_Pt)).reshape((len(Quad_Pt) ,1))



    # Evaluates A SPH fn at a quad pt(s)
    def Eval_SPH_Coef_Mat(self, Quad_Pt, lbdv):

        # For Scalar Case, we use usual vectorization:
        if(isscalar(Quad_Pt)):

            return sum(multiply(self.sph_coef, lbdv.Eval_SPH_At_Quad_Pt_Mat(Quad_Pt)))

        # For multiple quad pts, we use einstein sumation to output vector of solutions at each point:
        else:

            return einsum('ij, ijk -> k', self.sph_coef, lbdv.Eval_SPH_At_Quad_Pt_Mat(Quad_Pt)).reshape((len(Quad_Pt) ,1))


    #Use the fact that Theta Der Formula is EXACT for our basis
    def Quick_Theta_Der(self):

        Der_Coef = zeros(shape(self.sph_coef))

        for n_coef in range(self.sph_deg+1):
            for m_coef in range(1, n_coef+1): #Theta Der of Y^0_n = 0

                # D_theta X^m_n = -m*Z^m_n
                Der_Coef[n_coef-m_coef][n_coef] = m_coef*self.sph_coef[n_coef][n_coef-m_coef]

                # D_theta Z^m_n = m*X^m_n
                Der_Coef[n_coef][n_coef-m_coef] = -1*m_coef*self.sph_coef[n_coef-m_coef][n_coef]

        return spherical_harmonics_function(Der_Coef, self.sph_deg)

    def Quick_Theta_Bar_Der(self): #For rotated coordinate frame
        return self.Quick_Theta_Der()


    #Approximate func(theta, phi)*Coef_Mat(theta, phi) in SPH Basis
    def Fast_Proj_Product(self, func, Proj_Deg, lbdv):


        Proj_Product_Coef = zeros([Proj_Deg+1, Proj_Deg+1]) #Size of basis used to represent derivative

        #Compute inner product of theta der with each basis elt
        for n in range(Proj_Deg+1):
            for m in range(-1*n, n+1):

                I_mn = 0

                for quad_pt in range(lbdv.lbdv_quad_pts):
                    theta_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][0]
                    phi_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][1]

                    I_mn += func(theta_pt, phi_pt)*self.Eval_SPH_Coef_Mat(quad_pt, lbdv)*lbdv.Eval_SPH_Basis_Wt_At_Quad_Pts(m,n, quad_pt)
                    #Above fn sums basis vals for proj, times func*Coef_Mat,  times weight at each quad pt


                Proj_Product_mn = I_mn/Mass_Mat_Exact(m,n)

                if m>0:
                    Proj_Product_Coef[n-m][n] = Proj_Product_mn

                else: #m <= 0
                    Proj_Product_Coef[n][n+m] = Proj_Product_mn

        return spherical_harmonics_function(Proj_Product_Coef, Proj_Deg)


    #TAKES VALS AT QUAD PTS, to Approximate func(theta, phi)*Coef_Mat(theta, phi) in SPH Basis
    def Faster_Proj_Product(self, func_quad_vals, Proj_Deg, lbdv):


        Proj_Product_Coef = zeros([Proj_Deg+1, Proj_Deg+1]) #Size of basis used to represent derivative

        #Compute inner product of theta der with each basis elt
        for n in range(Proj_Deg+1):
            for m in range(-1*n, n+1):

                I_mn = 0

                for quad_pt in range(lbdv.lbdv_quad_pts):
                    #theta_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][0]
                    #phi_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][1]

                    I_mn += func_quad_vals[quad_pt]*self.Eval_SPH_Coef_Mat(quad_pt, lbdv)*lbdv.Eval_SPH_Basis_Wt_At_Quad_Pts(m,n, quad_pt)
                    #Above fn sums basis vals for proj, times func*Coef_Mat,  times weight at each quad pt


                Proj_Product_mn = I_mn/Mass_Mat_Exact(m,n)

                if m>0:
                    Proj_Product_Coef[n-m][n] = Proj_Product_mn

                else: #m <= 0
                    Proj_Product_Coef[n][n+m] = Proj_Product_mn


        return spherical_harmonics_function(Proj_Product_Coef, Proj_Deg)

    #Approximate Coef_Mat2(theta, phi)*Coef_Mat(theta, phi) in SPH Basis
    def Fast_Proj_Product_SPH(self, spherical_harmonics_function2, Proj_Deg, lbdv):


        Proj_Product_Coef = zeros([Proj_Deg+1, Proj_Deg+1]) #Size of basis used to represent derivative

        #Compute inner product of theta der with each basis elt
        for n in range(Proj_Deg+1):
            for m in range(-1*n, n+1):

                I_mn = 0

                for quad_pt in range(lbdv.lbdv_quad_pts):
                    theta_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][0]
                    phi_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][1]

                    I_mn += SPH_Func2.Eval_SPH_Coef_Mat(quad_pt, lbdv)*self.Eval_SPH_Coef_Mat(quad_pt, lbdv)*lbdv.Eval_SPH_Basis_Wt_At_Quad_Pts(m,n, quad_pt)
                    #Above fn sums basis vals for proj, times func*Coef_Mat,  times weight at each quad pt


                Proj_Product_mn = I_mn/Mass_Mat_Exact(m,n)

                if m>0:
                    Proj_Product_Coef[n-m][n] = Proj_Product_mn

                else: #m <= 0
                    Proj_Product_Coef[n][n+m] = Proj_Product_mn


        return spherical_harmonics_function(Proj_Product_Coef, Proj_Deg)


    def Inner_Product_SPH(self, Other_SPH):

        Vec1 = self.sph_coef.flatten()
        Vec2 = Other_SPH.sph_coef.flatten()

        I_Vec = eye((self.sph_deg + 1)**2)

        return sum(multiply(abs(Vec1), abs(Vec2)))/2 + sum(multiply(multiply(abs(Vec1), I_Vec), multiply(abs(Vec2), I_Vec)))/2


    def L2_Norm_SPH(self):
        return sqrt(self.Inner_Product_SPH(self))



    def Lp_Rel_Error_in_Chart(self, f, lbdv, p): #Assumes f NOT 0

        Lp_Err = 0 # ||self - f||_p
        Lp_f = 0  # || f ||_p

        for quad_pt in range(lbdv.lbdv_quad_pts):
            theta_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][0]
            phi_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][1]
            weight_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][2]

            if(Chart_Min_Polar < phi_pt and phi_pt < Chart_Max_Polar):
                Lp_Err_Pt = (abs(self.Eval_SPH_Coef_Mat(quad_pt, lbdv) - f(theta_pt, phi_pt))**p)*weight_pt
                Lp_f_Pt = (abs(f(theta_pt, phi_pt)**p))*weight_pt

                Lp_Err += Lp_Err_Pt
                Lp_f += Lp_f_Pt



        return     (Lp_Err/Lp_f)**(1./p) #||self - f||_p / || f ||_p


    # Looks at Rel Error in ALL of S^2
    def Lp_Rel_Error_in_S2(self, f, lbdv, p): #Assumes f NOT 0

        Lp_Err = 0 # ||self - f||_p
        Lp_f = 0  # || f ||_p

        for quad_pt in range(lbdv.lbdv_quad_pts):
            theta_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][0]
            phi_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][1]
            weight_pt = lbdv.Lbdv_Sph_Pts_Quad[quad_pt][2]

            Lp_Err_Pt = (abs(self.Eval_SPH_Coef_Mat(quad_pt, lbdv) - f(theta_pt, phi_pt))**p)*weight_pt

            Lp_f_Pt = (abs(f(theta_pt, phi_pt)**p))*weight_pt

            Lp_Err += Lp_Err_Pt
            Lp_f += Lp_f_Pt


        return     (Lp_Err/Lp_f)**(1./p) #||self - f||_p / || f ||_p


    # Returns properly ordered vector of SPH Coef
    def Flatten_Coef_Mat(self):

        coef_mat = self.sph_coef
        sph_degree =  self.sph_deg

        coef_vec = zeros(( (sph_degree + 1)**2, 1))
        row = 0

        for n in range(sph_degree+1):
            for m in range(-1*n, n+1):

                if m>0:
                    coef_vec[row, 0] = coef_mat[n-m][n]

                else: #m <= 0
                    coef_vec[row, 0] = coef_mat[n][n+m]

                row = row + 1

        return coef_vec



    # Take sum of SPH function of same order
    def plus_sph(self, other_SPH):
        return spherical_harmonics_function(self.sph_coef +other_SPH.sph_coef , self.sph_deg)

    # Take difference of SPH function of same order
    def minus_sph(self, other_SPH):
        return spherical_harmonics_function(self.sph_coef -other_SPH.sph_coef , self.sph_deg)

    # Multiplies SPH function by a const
    def sph_times(self, const):
        return spherical_harmonics_function(self.sph_coef*const, self.sph_deg)

    #Easier way to debug these objects
    def print_spherical_harmonics_function(self):
        print("spherical_harmonics_function has degree: "+str(self.sph_deg)+" and has coef mat: "+"\n"+str(self.sph_coef))
