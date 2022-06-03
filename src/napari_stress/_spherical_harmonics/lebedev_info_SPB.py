# From https://github.com/campaslab/STRESS
#! This class Generates All of the quadrature information that sph_func, k_form use

from numpy  import *
from scipy  import *
import mpmath
import cmath
from scipy.special import sph_harm

from .lebedev_write_SPB import * # lists all Lebdv quadratures
from .charts_SPB import * # For Lebedev Point Conversion

#for pickling:
#import cPickle as pkl # BJG: py2pt7 version
import pickle as pkl
import os, sys

# Allows us to find appropriate quadrature:
quad_deg_lookUp = {
         6: 2,
         14: 3,
         26: 4,
         38: 5,
         50: 6,
         74: 7,
         86: 8,
         110: 9,
         146: 10,
         170: 11,
         194: 12,
         230: 13,
         266: 14,
         302: 15,
         350: 16,
          434: 18,
          590: 21,
          770: 24,
          974: 27,
          1202: 30,
          1454: 33,
          1730: 36,
          2030: 39,
          2354: 42,
          2702: 45,
          3074: 48,
          3470: 51,
          3890: 54,
          4334: 57,
          4802: 60,
          5294: 63,
          5810: 66
    }


#ALLOWS us to find quad pts for hyper-interpolation:
pts_of_lbdv_lookup = {
    2:6,
    3:14,
    4:26,
    5:38,
    6:50,
    7:74,
    8:86,
    9:110,
    10:146,
    11:170,
    12:194,
    13:230,
    14:266,
    15:302,
    16:350,
    18:434,
    21:590,
    24:770,
    27:974,
    30:1202,
    33:1454,
    36:1730,
    39:2030,
    42:2354,
    45:2702,
    48:3074,
    51:3470,
    54:3890,
    57:4334,
    60:4802,
    63:5294,
    66:5810
}

# Finds Lowest order basis for degree:
def look_up_lbdv_pts(Degree):
    if(Degree > 66 or Degree < 2):
        print("Error: Cannot have Basis of Degree: "+str(Degree))

    elif(Degree in pts_of_lbdv_lookup):
        return pts_of_lbdv_lookup[Degree]

    else:
        return look_up_lbdv_pts(Degree+1)

def get_quad_degree(quad_pts):
    return     quad_deg_lookUp[quad_pts]

#######################################################################################################################

def Eval_SPH_Basis(M_Coef, N_Coef, Theta, Phi):
    if(M_Coef >= 0):
        return sph_harm(M_Coef, N_Coef, Theta, Phi).real

    else: # m<0, we use Y^(-m)_n.imag
        return sph_harm(-1*M_Coef, N_Coef, Theta, Phi).imag


#Evaluates d_phi(Y^m_n) at SINGLE PT
def Der_Phi_Basis_Fn(M_Coef,N_Coef, Theta, Phi): # M_Coef < 0 Corresonds to Z^|M|_N

    Der_Phi_Val = []

        # For Scalar Case, we use usual vectorization:
    if(isscalar(Theta)):
        #COPIED FROM SPH_DER_PHI_FN
        Der_Phi_Val = 0

        if(M_Coef == 0): #No Cotangent terms:
            if(N_Coef > 0):
                return sqrt((N_Coef)*(N_Coef+1))*(((e**(-1j*Theta))*sph_harm(1, N_Coef, Theta, Phi))).real
            else:
                return 0 #d_phi Y^0_0 = 0

        elif(M_Coef < 0):

            m_sph = -1*M_Coef
            Der_Phi_Val += (m_sph*mpmath.cot(Phi))*sph_harm(m_sph, N_Coef, Theta, Phi).imag

            if(m_sph < N_Coef):
                Der_Phi_Val += sqrt((N_Coef-m_sph)*(N_Coef+m_sph+1))*(((e**(-1j*Theta))*sph_harm(m_sph+1, N_Coef, Theta, Phi))).imag

        else: # M_Coef >= 0

            m_sph = M_Coef
            Der_Phi_Val += (m_sph*mpmath.cot(Phi))*sph_harm(m_sph, N_Coef, Theta, Phi).real

            if(m_sph < N_Coef):
                Der_Phi_Val += sqrt((N_Coef-m_sph)*(N_Coef+m_sph+1))*(((e**(-1j*Theta))*sph_harm(m_sph+1, N_Coef, Theta, Phi))).real

    else: # array input case:

        #COPIED FROM SPH_DER_PHI_FN
        Der_Phi_Val = np.zeros_like(Theta)

        if(M_Coef == 0): #No Cotangent terms:
            if(N_Coef > 0):
                return sqrt((N_Coef)*(N_Coef+1))*(((np.exp(-1j*Theta))*sph_harm(1, N_Coef, Theta, Phi))).real
            else:
                return 0 #d_phi Y^0_0 = 0

        elif(M_Coef < 0):

            m_sph = -1*M_Coef
            Der_Phi_Val += (m_sph*(1./np.tan(Phi)))*sph_harm(m_sph, N_Coef, Theta, Phi).imag

            if(m_sph < N_Coef):
                Der_Phi_Val += sqrt((N_Coef-m_sph)*(N_Coef+m_sph+1))*(((np.exp(-1j*Theta))*sph_harm(m_sph+1, N_Coef, Theta, Phi))).imag

        else: # M_Coef >= 0

            m_sph = M_Coef
            Der_Phi_Val += (m_sph*(1./np.tan(Phi)))*sph_harm(m_sph, N_Coef, Theta, Phi).real

            if(m_sph < N_Coef):
                Der_Phi_Val += sqrt((N_Coef-m_sph)*(N_Coef+m_sph+1))*(((np.exp(-1j*Theta))*sph_harm(m_sph+1, N_Coef, Theta, Phi))).real

    return Der_Phi_Val


#Evaluates d_phi(d_phi((Y^m_n)) at SINGLE PT
def Der_Phi_Phi_Basis_Fn(M_Coef,N_Coef, Theta, Phi): # M_Coef < 0 Corresonds to Z^|M|_N

    Der_Phi_Phi_Val = 0

    if(M_Coef < 0):

        m_sph = -1*M_Coef
        Der_Phi_Phi_Val += m_sph*(m_sph*(mpmath.cot(Phi)**2) - (mpmath.csc(Phi)**2))*sph_harm(m_sph, N_Coef, Theta, Phi).imag

        if(m_sph < N_Coef):
            Der_Phi_Phi_Val += sqrt((N_Coef-m_sph)*(N_Coef+m_sph+1))*(2*m_sph + 1)*mpmath.cot(Phi)*(((e**(-1j*Theta))*sph_harm(m_sph+1, N_Coef, Theta, Phi))).imag

        if(m_sph < (N_Coef -1) ):
            Der_Phi_Phi_Val += sqrt((N_Coef-m_sph)*(N_Coef-m_sph-1)*(N_Coef+m_sph+1)*(N_Coef+m_sph+2))*(((e**(-2j*Theta))*sph_harm(m_sph+2, N_Coef, Theta, Phi))).imag

    else: # M_Coef >= 0

        m_sph = M_Coef
        Der_Phi_Phi_Val +=  m_sph*(m_sph*(mpmath.cot(Phi)**2) - (mpmath.csc(Phi)**2))*sph_harm(m_sph, N_Coef, Theta, Phi).real

        if(m_sph < N_Coef):
            Der_Phi_Phi_Val += sqrt((N_Coef-m_sph)*(N_Coef+m_sph+1))*(2*m_sph + 1)*mpmath.cot(Phi)*(((e**(-1j*Theta))*sph_harm(m_sph+1, N_Coef, Theta, Phi))).real
        if(m_sph < (N_Coef -1) ):
            Der_Phi_Phi_Val += sqrt((N_Coef-m_sph)*(N_Coef-m_sph-1)*(N_Coef+m_sph+1)*(N_Coef+m_sph+2))*(((e**(-2j*Theta))*sph_harm(m_sph+2, N_Coef, Theta, Phi))).real

    return Der_Phi_Phi_Val


def Lbdv_Cart_To_Sph(Cart_Pts_Wts): #takes matrix with rows [x,y,z,w] -> [theta, phi, w] (r=1)

    num_lbdv_pts = shape(Cart_Pts_Wts)[0]

    Sph_Pts_Wts = zeros((num_lbdv_pts, 3))

    for pt in range(num_lbdv_pts):


        x = Cart_Pts_Wts[pt][0]
        y = Cart_Pts_Wts[pt][1]
        z = Cart_Pts_Wts[pt][2]

        # set quad_pt weight
        Sph_Pts_Wts[pt][2] = Cart_Pts_Wts[pt][3]

        #Conversion Fn now on its own
        Angles = Cart_To_Coor_A(x, y, z)
        Sph_Pts_Wts[pt][0] = Angles[0]
        Sph_Pts_Wts[pt][1] = Angles[1]

    return Sph_Pts_Wts

# Get max degree quad pts for plotting/interpolation:
def get_5810_quad_pts():

    euc_quad_pts = Lebedev(5810)
    sph_quad_pts = Lbdv_Cart_To_Sph(euc_quad_pts)

    x_pts, y_pts, z_pts, w_pts = hsplit(euc_quad_pts, 4)
    theta_pts, phi_pts, w_pts = hsplit(sph_quad_pts, 3)

    return x_pts, y_pts, z_pts, theta_pts, phi_pts


######################################################################################################################################################

class lbdv_info(object): #Generates (ONCE) and stores Lebedev Info

    def __init__(self, Max_SPH_Deg, Num_Quad_Pts): # This generates lebedev and quad pt info upon instantiation

        MY_DIR  = os.path.realpath(os.path.dirname(__file__)) #current folder
        PICKLE_DIR = os.path.join(MY_DIR, 'Pickled_LBDV_Files') # directory for pickled files
        LBDV_name = "_Deg_basis"+str(Max_SPH_Deg)+"_Quad_Pts"+str(Num_Quad_Pts) #name for these pickled files

        if not os.path.exists(PICKLE_DIR):
            os.makedirs(PICKLE_DIR)

        ### GENERATE 5810 Quadrature ONCE #######
        #print("generating quad pts") # BJG: only notify if NEW mats are needed (time-consuming)
        self.lbdv_quad_pts = Num_Quad_Pts #Needs to be appropriate number up to 5810

        #### To see if there are errors in assigning points ###############
        self.Lbdv_Cart_Pts_Quad = Lebedev(self.lbdv_quad_pts)

        self.X, self.Y, self.Z, self.W = hsplit(self.Lbdv_Cart_Pts_Quad, 4)
        ###################################################################

        self.Lbdv_Sph_Pts_Quad = Lbdv_Cart_To_Sph(self.Lbdv_Cart_Pts_Quad)
        self.theta_pts, self.phi_pts, self.weight_pts = hsplit(self.Lbdv_Sph_Pts_Quad, 3)

        #print("quad pts done") # BJG: only notify if NEW mats are needed (time-consuming)
        #########################################



        LBDV_Basis_at_Quad_Pts_Mats_filename = "LBDV_Basis_at_Quad_Pts"+LBDV_name
        LBDV_Basis_at_Quad_Pts_Mats_filepath =  os.path.join(PICKLE_DIR, LBDV_Basis_at_Quad_Pts_Mats_filename) # we store as a 4-dim array:

        if(os.path.isfile(LBDV_Basis_at_Quad_Pts_Mats_filepath)): # If already pickled, we load it, and split it into the needed arrays:

            #print("\n"+"Loading Pickled LBDV data Mats"+"\n") # BJG: only notify if NEW mats are needed (time-consuming)

            Pickled_LBDV_Basis_at_Quad_Pts_Mats = []

            with open(LBDV_Basis_at_Quad_Pts_Mats_filepath, 'rb') as f_lbdv_basis:
                Pickled_LBDV_Basis_at_Quad_Pts_Mats = pkl.load(f_lbdv_basis)

            # Split into needed matricies:
            self.SPH_Basis_Wt_At_Quad_Pts = Pickled_LBDV_Basis_at_Quad_Pts_Mats[:,:,:, 0]
            self.SPH_Basis_At_Quad_Pts = Pickled_LBDV_Basis_at_Quad_Pts_Mats[:,:,:, 1]
            self.SPH_Phi_Der_At_Quad_Pts = Pickled_LBDV_Basis_at_Quad_Pts_Mats[:,:,:, 2]
            self.SPH_Phi_Phi_Der_At_Quad_Pts = Pickled_LBDV_Basis_at_Quad_Pts_Mats[:,:,:, 3]


        else: #If not pickled, we generate and pickle these files:

            print("\n"+"Pickling This LBDV data:"+"\n")

            ### Generate EVAL_SPH Vals At Quad Pts ONCE  ########
            print("generating basis vals")

            # Store for W_pt*Y^m_n at each point, in same format as coef mat
            self.SPH_Basis_Wt_At_Quad_Pts = zeros((Max_SPH_Deg+1, Max_SPH_Deg+1, self.lbdv_quad_pts)) #includes weights

            self.SPH_Basis_At_Quad_Pts = zeros((Max_SPH_Deg+1, Max_SPH_Deg+1, self.lbdv_quad_pts)) #does NOT inculude weights


            for N_Coef in range(Max_SPH_Deg+1): # 0,...,Max_SPH_Deg
                for M_Coef in range(-1*N_Coef,N_Coef+1): #-N,...,N
                    for quad_pt in range(self.lbdv_quad_pts):

                        Theta_Quad_Pt = self.Lbdv_Sph_Pts_Quad[quad_pt][0]
                        Phi_Quad_Pt = self.Lbdv_Sph_Pts_Quad[quad_pt][1]
                        Weight_Quad_Pt = self.Lbdv_Sph_Pts_Quad[quad_pt][2]

                        if(M_Coef >= 0):
                            self.SPH_Basis_At_Quad_Pts[N_Coef - M_Coef][N_Coef][quad_pt] = Eval_SPH_Basis(M_Coef, N_Coef, Theta_Quad_Pt, Phi_Quad_Pt)
                            self.SPH_Basis_Wt_At_Quad_Pts[N_Coef - M_Coef][N_Coef][quad_pt] = Weight_Quad_Pt*self.SPH_Basis_At_Quad_Pts[N_Coef - M_Coef][N_Coef][quad_pt]


                        else: #M_Coef < 0
                            self.SPH_Basis_At_Quad_Pts[N_Coef][N_Coef -(-1*M_Coef)][quad_pt] = Eval_SPH_Basis(M_Coef, N_Coef, Theta_Quad_Pt, Phi_Quad_Pt)
                            self.SPH_Basis_Wt_At_Quad_Pts[N_Coef][N_Coef -(-1*M_Coef)][quad_pt] = Weight_Quad_Pt*self.SPH_Basis_At_Quad_Pts[N_Coef][N_Coef -(-1*M_Coef)][quad_pt]




            print("generated basis vals")
            #######################   Der Phi Fns To Speed Up Code    ##############################

            print("generating dphi/ dphi_phi vals")



            #Create matrix to store Phi Der of all degrees used
            self.SPH_Phi_Der_At_Quad_Pts = zeros((Max_SPH_Deg+1, Max_SPH_Deg+1, self.lbdv_quad_pts))

            #Create matrix to store 2nd Phi Der of all degrees used
            self.SPH_Phi_Phi_Der_At_Quad_Pts = zeros((Max_SPH_Deg+1, Max_SPH_Deg+1, self.lbdv_quad_pts))

            #eta_A(lambda Theta, Phi: SPH_Der_Phi_Fn(SPH_Deg, Coef_Mat, Theta, Phi), Theta, Phi)

            #Fill up matrix ONCE, with above function, composed with eta_A
            for N_Coef in range(Max_SPH_Deg+1): # 0,...,Max_SPH_Deg
                for M_Coef in range(-1*N_Coef,N_Coef+1): #-N,...,N
                    for quad_pt in range(self.lbdv_quad_pts):

                        Theta_Quad_Pt = self.Lbdv_Sph_Pts_Quad[quad_pt][0]
                        Phi_Quad_Pt = self.Lbdv_Sph_Pts_Quad[quad_pt][1]
                        # Dont need Weight, since Quadrature covers that

                        if(M_Coef == 0): #We Dont need eta_A for first der in this case

                            self.SPH_Phi_Der_At_Quad_Pts[N_Coef - M_Coef][N_Coef][quad_pt] = Der_Phi_Basis_Fn(M_Coef, N_Coef, Theta_Quad_Pt, Phi_Quad_Pt)

                            self.SPH_Phi_Phi_Der_At_Quad_Pts[N_Coef - M_Coef][N_Coef][quad_pt] = eta_A(lambda Theta_Quad_Pt, Phi_Quad_Pt: Der_Phi_Phi_Basis_Fn(M_Coef, N_Coef, Theta_Quad_Pt, Phi_Quad_Pt), Theta_Quad_Pt, Phi_Quad_Pt)

                        elif(M_Coef >= 0):


                            self.SPH_Phi_Der_At_Quad_Pts[N_Coef - M_Coef][N_Coef][quad_pt] = eta_A(lambda Theta_Quad_Pt, Phi_Quad_Pt: Der_Phi_Basis_Fn(M_Coef, N_Coef, Theta_Quad_Pt, Phi_Quad_Pt), Theta_Quad_Pt, Phi_Quad_Pt)

                            self.SPH_Phi_Phi_Der_At_Quad_Pts[N_Coef - M_Coef][N_Coef][quad_pt] = eta_A(lambda Theta_Quad_Pt, Phi_Quad_Pt: Der_Phi_Phi_Basis_Fn(M_Coef, N_Coef, Theta_Quad_Pt, Phi_Quad_Pt), Theta_Quad_Pt, Phi_Quad_Pt)

                        else: #M_Coef < 0
                            self.SPH_Phi_Der_At_Quad_Pts[N_Coef][N_Coef -(-1*M_Coef)][quad_pt] = eta_A(lambda Theta_Quad_Pt, Phi_Quad_Pt: Der_Phi_Basis_Fn(M_Coef, N_Coef, Theta_Quad_Pt, Phi_Quad_Pt), Theta_Quad_Pt, Phi_Quad_Pt)

                            self.SPH_Phi_Phi_Der_At_Quad_Pts[N_Coef][N_Coef -(-1*M_Coef)][quad_pt] = eta_A(lambda Theta_Quad_Pt, Phi_Quad_Pt: Der_Phi_Phi_Basis_Fn(M_Coef, N_Coef, Theta_Quad_Pt, Phi_Quad_Pt), Theta_Quad_Pt, Phi_Quad_Pt)


            print("done with dphi/ dphi_phi vals"+"\n")


            ###!!! PICLKLE RESULTS FOR FUTURE USE !!!###
            To_Pickle_LBDV_Basis_at_Quad_Pts_Mats = zeros((Max_SPH_Deg+1, Max_SPH_Deg+1, self.lbdv_quad_pts, 4))

            To_Pickle_LBDV_Basis_at_Quad_Pts_Mats[:,:,:, 0] = self.SPH_Basis_Wt_At_Quad_Pts
            To_Pickle_LBDV_Basis_at_Quad_Pts_Mats[:,:,:, 1] = self.SPH_Basis_At_Quad_Pts
            To_Pickle_LBDV_Basis_at_Quad_Pts_Mats[:,:,:, 2] = self.SPH_Phi_Der_At_Quad_Pts
            To_Pickle_LBDV_Basis_at_Quad_Pts_Mats[:,:,:, 3] = self.SPH_Phi_Phi_Der_At_Quad_Pts

            with open(LBDV_Basis_at_Quad_Pts_Mats_filepath, 'wb') as f_lbdv_basis:
                pkl.dump(To_Pickle_LBDV_Basis_at_Quad_Pts_Mats, f_lbdv_basis)


        ####################### LBDV Rotation To Speed Up Code ##############################


        LBDV_Chart_of_Quad_Pts_Mats_filename = "LBDV_Chart_of_Quad_Pts"+LBDV_name
        LBDV_Chart_of_Quad_Pts_Mats_filepath =  os.path.join(PICKLE_DIR, LBDV_Chart_of_Quad_Pts_Mats_filename) #store these in seperate file, but in same directory

        if(os.path.isfile(LBDV_Chart_of_Quad_Pts_Mats_filepath)): # If already pickled, we load it, and split it into the needed arrays:

            #print("\n"+"Loading Pickled LBDV Chart Mats"+"\n") # BJG: only notify if NEW mats are needed (time-consuming)

            Pickled_LBDV_Charts_Quad_Pts_Mats = []

            with open(LBDV_Chart_of_Quad_Pts_Mats_filepath, 'rb') as f_lbdv_chart:
                Pickled_LBDV_Charts_Quad_Pts_Mats = pkl.load(f_lbdv_chart)

            # Split into needed matricies:
            self.Rot_Lbdv_Quad_vals, self.Inv_Rot_Lbdv_Quad_vals, self.Chart_of_Quad_Pts = np.hsplit(Pickled_LBDV_Charts_Quad_Pts_Mats ,3)



        else: #If not pickled, we generate and pickle these files:

            print("\n"+"Pickling This LBDV  Chart data:"+"\n")

            ### Generate Chart LBDV Data ONCE  ########


            print("generating lbdv rotation vals")

            self.Rot_Lbdv_Quad_vals = zeros(( self.lbdv_quad_pts, 1 )) # Stores equivlent Quad_pt in Chart B, for input Quad_Pt in Chart A
            self.Inv_Rot_Lbdv_Quad_vals = zeros(( self.lbdv_quad_pts, 1 )) # Stores equivlent Quad_pt in Chart A, for input Quad_Pt in Chart B


            '''
            Example: if quad_pt = i corresponds to (theta, phi) = (0, pi/2), and quad_pt = j corresponds to (0, pi),
            then  self.Rot_Lbdv_Quad_vals[j] = i,
            and  self.Inv_Rot_Lbdv_Quad_vals[i] = j,
            since (theta, phi) = (0, pi) has the same euclidean coors as (theta_bar, phi_bar) = (0, pi/2)
            '''

            self.Chart_of_Quad_Pts = zeros(( self.lbdv_quad_pts, 1 )) # 1 if pt is in Chart A, -1 if not

            # Choose which values we use and where:
            for quad_pt in range(self.lbdv_quad_pts):

                theta_pt = self.theta_pts[quad_pt]
                phi_pt = self.phi_pts[quad_pt]

                # Determine which Chart Each Quad Pt is in:
                if(Domain(theta_pt, phi_pt) >= 0):

                    self.Chart_of_Quad_Pts[quad_pt] = 1

                else:
                    self.Chart_of_Quad_Pts[quad_pt] = -1

                #If we are able to identify rotated quad pt in Chart B
                rot_pt_found = False

                x_pt = self.X[quad_pt] #cos(theta_pt)*sin(phi_pt)
                y_pt = self.Y[quad_pt] #sin(theta_pt)*sin(phi_pt)
                z_pt = self.Z[quad_pt] #cos(phi_pt)

                # Find Rotated Quad Pt at same location:
                for quad_pt_rot in range(self.lbdv_quad_pts):

                    theta_bar_pt_rot = self.theta_pts[quad_pt_rot]
                    phi_bar_pt_rot = self.phi_pts[quad_pt_rot]

                    x_pt_rot = cos(phi_bar_pt_rot)
                    y_pt_rot = sin(theta_bar_pt_rot)*sin(phi_bar_pt_rot)
                    z_pt_rot = -1*cos(theta_bar_pt_rot)*sin(phi_bar_pt_rot)

                    if(abs(x_pt - x_pt_rot) < 1e-7):
                        if(abs(y_pt - y_pt_rot) < 1e-7):
                            if(abs(z_pt - z_pt_rot) < 1e-7):
                                if(rot_pt_found == False):

                                    rot_pt_found = True

                                    self.Rot_Lbdv_Quad_vals[quad_pt] = quad_pt_rot
                                    self.Inv_Rot_Lbdv_Quad_vals[quad_pt_rot] = quad_pt


                if(rot_pt_found == False):
                    print("!!ROTATED QUAD PT NOT FOUND!!")


            print("done with lbdv rotation vals"+"\n")

            ###!!! PICLKLE RESULTS FOR FUTURE USE !!!###
            To_Pickle_LBDV_Charts_at_Quad_Pts_Mats = np.hstack(( self.Rot_Lbdv_Quad_vals, self.Inv_Rot_Lbdv_Quad_vals, self.Chart_of_Quad_Pts ))

            with open(LBDV_Chart_of_Quad_Pts_Mats_filepath, 'wb') as f_lbdv_charts:
                pkl.dump(To_Pickle_LBDV_Charts_at_Quad_Pts_Mats, f_lbdv_charts)




    #Fn that will retrieve these values to be used in quadrature, in nice format
    def Eval_SPH_Basis_Wt_At_Quad_Pts(self, M,N, Quad_Pt):
        if(M >= 0):
            return self.SPH_Basis_Wt_At_Quad_Pts[N-M][N][Quad_Pt]
        else: # If M<0
            return self.SPH_Basis_Wt_At_Quad_Pts[N][N-(-1*M)][Quad_Pt]


    #Fn that will retrieve basis vals at quad pts for product projections
    def Eval_SPH_At_Quad_Pts(self, M,N, Quad_Pt):
        if(M >= 0):
            return self.SPH_Basis_At_Quad_Pts[N-M][N][Quad_Pt]
        else: # If M<0
            return self.SPH_Basis_At_Quad_Pts[N][N-(-1*M)][Quad_Pt]


    #Fn that will retrieve Der Phi values in nice format
    def Eval_SPH_Der_Phi_At_Quad_Pts(self, M,N, Quad_Pt):
        if(M >= 0):
            return self.SPH_Phi_Der_At_Quad_Pts[N-M][N][Quad_Pt]
        else: # If M<0
            return self.SPH_Phi_Der_At_Quad_Pts[N][N-(-1*M)][Quad_Pt]

    #Fn that will retrieve 2nd Der Phi values in nice format
    def Eval_SPH_Der_Phi_Phi_At_Quad_Pts(self, M,N, Quad_Pt):
        if(M >= 0):
            return self.SPH_Phi_Phi_Der_At_Quad_Pts[N-M][N][Quad_Pt]
        else: # If M<0
            return self.SPH_Phi_Phi_Der_At_Quad_Pts[N][N-(-1*M)][Quad_Pt]


    # Return in Matrix format for VECTORIZATION in Sph_Func:

    #Fn that will retrieve all values to be used in quadrature, in nice format (of All quad points, for Y^M_N)
    def Eval_SPH_Basis_Wt_M_N(self, M, N):

        if(M >= 0):
            return self.SPH_Basis_Wt_At_Quad_Pts[N-M, N, :]
        else: # If M<0
            return self.SPH_Basis_Wt_At_Quad_Pts[N, N-(-1*M), :]



    #Fn that will retrieve all basis vals at quad pt for product projections
    def Eval_SPH_At_Quad_Pt_Mat(self, Quad_Pt):

        return self.SPH_Basis_At_Quad_Pts[:, :, Quad_Pt]



    #Fn that will retrieve ALL Der Phi values in nice format
    def Eval_SPH_Der_Phi_At_Quad_Pt_Mat(self, Quad_Pt):

        return self.SPH_Phi_Der_At_Quad_Pts[:, :, Quad_Pt]



    #Fn that will retrieve ALL 2nd Der Phi values in nice format
    def Eval_SPH_Der_Phi_Phi_At_Quad_Pt_Mat(self, Quad_Pt):

        return self.SPH_Phi_Phi_Der_At_Quad_Pts[:, :, Quad_Pt]



    # Fn that will retrive vals of Rot_Lbdv_Quad_vals, USE astpye(int)!!
    def Eval_Rot_Lbdv_Quad_vals(self, Quad_Pt):
        return self.Rot_Lbdv_Quad_vals.astype(int)[Quad_Pt, 0]

    # Fn that will retrive vals of Inv_Rot_Lbdv_Quad_vals, USE astpye(int)!!
    def Eval_Inv_Rot_Lbdv_Quad_vals(self, Quad_Pt):
        return self.Inv_Rot_Lbdv_Quad_vals.astype(int)[Quad_Pt, 0]

    # Fn that will retrive vals of Chart_of_Quad_Pts
    def Eval_Chart_of_Quad_Pts(self, Quad_Pt):
        self.Chart_of_Quad_Pts[Quad_Pt, 0]

    # For Splitting Integration between charts:
    def eta_z(self, quad_pt):

        z_pt = self.Z[quad_pt, 0]

        if(abs(z_pt) <= .25):
            return 1

        elif(abs(z_pt) >= .75):
            return 0

        elif(z_pt >= .25 and z_pt <= .75):
            return np.exp((-1.0)/(1.0 - ((z_pt - .25)/(.5))**2))*np.exp(1.0)

        else: #(z_pt <= -.25 and z_pt >= -.75):
            return np.exp((-1.0)/(1.0 - ((z_pt + .25)/(-.5))**2))*np.exp(1.0)
