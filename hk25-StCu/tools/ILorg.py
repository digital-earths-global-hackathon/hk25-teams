# Function downloaded from https://github.com/giobiagioli/organization_indices


from scipy import spatial
from scipy.ndimage import label, center_of_mass
import numpy as np
import sys

#The following routine calculate_indices computes the theoretical and observed Besag's L-functions given a 2D binary field of convective/non-convective points and provides the cloud-to-cloud nearest-neighbor distances for the calculation of L_org/dL_org and I_org/RI_org.

#	INPUT PARAMETERS
#		dxy				grid resolution (assumed to be uniform in both directions)
#		cnv_idx				2D binary matrix, =1 in convective points, =0 elsewhere. The Python object class must be numpy.ndarray
#		rmax				maximum search radius (box size in the discrete case) for the neighbor counting
#		bins				distance/box size bands in which to evaluate the object counts. The Python object class must be numpy.ndarray
#		periodic_BCs			flag for the assignment of periodic (True)/open (False) boundary conditions
# 		periodic_zonal			flag for the assignment of periodic boundary conditions in the x-direction and open boundary conditions in the y-direction (True). False if domain is doubly periodic or open
#		clustering_algo			flag for the application (True) or not (False) of a four-connectivity clustering algorithm to merge aggregates
#		binomial_continuous		flag for binomial correction in case finite domains and Poisson model are assumed (True, False otherwise)
#		binomial_discrete		flag for the assumption of discrete binomial model as a reference for spatial randomness (True, False otherwise)
#		edge_mode			In case of open domains, this specifies the edge correction method to compensate the undercount bias (options 'none', 'besag', see manuscript and documentation for details, 'besag' only if binomial_discrete is True)

#	OUTPUT PARAMETERS
#		I_org				value of the I_org index as originally introduced by Tompkins and Semie (2017)
#		RI_org				value of the RI_org index (i.e., RI_org = I_org - 0.5)
#		L_org				value of the L_org/dL_org index (depends on whether continuous domains or discrete grids are considered)
#		NNCDF_theor			NNCDF theoretically expected in case the ncnv cloud entities were randomly distributed within the domain (Weibull)
#		NNCDF_obs			NNCDF derived from the distribution of the ncnv objects in the scene
#		Besag_theor			Besag's L-function theoretically expected in case the ncnv cloud entities were randomly distributed within the domain
#		Besag_obs			Besag's L-function derived from the distribution of the ncnv objects in the scene   	 	

def calculate_indices(dxy, cnv_idx, rmax, bins, periodic_BCs, periodic_zonal, clustering_algo, binomial_continuous, binomial_discrete, edge_mode):
	
	##EXCLUSION OF CASES FOR WHICH INPUT ARGUMENTS CONFLICT/ARE NOT ACCOUNTED FOR BY THE ROUTINE
	if (periodic_BCs and periodic_zonal) or (binomial_continuous and binomial_discrete):
		print('--------CONFLICTING INPUT OPTIONS--------')
		sys.exit()
	if not binomial_discrete and not periodic_BCs:
		print('--------CASE NOT EXAMINED BY THE PRESENT ROUTINE--------')
		#Built-in functions are available for edge corrections in case of random Poisson processes, see https://docs.astropy.org/en/stable/stats/ripley.html 
		sys.exit()
	
	#Calculation of width and height of the observation window (domain)
	nx = cnv_idx.shape[1]
	ny = cnv_idx.shape[0]
	domain_x = (nx-1)*dxy
	domain_y = (ny-1)*dxy
	
	##DETERMINATION OF CLOUD OBJECT NUMBER AND CENTROIDS
	
	#If four-connectivity clustering algorithms are applied, adjacent convective pixels (i.e., sharing a common side) are merged into a single one. If the domain is cyclic, aggregates on either sides of the domain are close to each other and identified as single ones if they are contiguous. If the domain is cyclic in the zonal but not in the meridional direction, this applies along the x axis only.
	if clustering_algo:
		if periodic_BCs:		
			#Periodic continuation of the domain
			mask = np.block([[cnv_idx, cnv_idx, cnv_idx],[cnv_idx, cnv_idx, cnv_idx],[cnv_idx, cnv_idx, cnv_idx]])		
			#Identification of the clusters and computation of their centers of mass. Only the centroids located within the original (inner) domain are retained. 
			labeled_array, num_features = label(mask)
			centroid = np.asarray(center_of_mass(mask, labeled_array, range(1,num_features+1)))
			centroids_updraft = centroid[np.where((centroid[:,0]>=ny) & (centroid[:,0]<2*ny) & (centroid[:,1]>=nx) & (centroid[:,1]<2*nx))]-[ny,nx]
		elif periodic_zonal:		
			#Periodic continuation of the domain along the zonal direction
			mask = np.block([cnv_idx, cnv_idx, cnv_idx])
			labeled_array, num_features = label(mask)
			centroid = np.asarray(center_of_mass(mask, labeled_array, range(1,num_features+1)))
			centroids_updraft = centroid[np.where((centroid[:,1]>=nx) & (centroid[:,1]<2*nx))]-[0,nx]		
		else:
			#Open boundary case (no periodic continuation of the domain)
			mask = cnv_idx
			labeled_array, num_features = label(mask)
			centroids_updraft = np.asarray(center_of_mass(mask, labeled_array, range(1,num_features+1)))	
	else:	
		#If no clustering algorithms are applied, each cloud object is treated as a single entity
		centroids_updraft = np.argwhere(cnv_idx)

	#Determination of the number of convective points both with and without the clustering algorithm applied. Their average spatial density is then computed. If the clustering algorithm is applied, the resulting number of convective objects is used in the calculation of density
	ncnv_no_algo = np.sum(cnv_idx)
	ncnv = len(centroids_updraft)
	lambd = ncnv/(domain_x*domain_y)
	
	##DETERMINATION OF NEAREST-NEIGHBOR AND ALL-NEIGHBOR DISTANCES AND COUNTING OF NEIGHBORS IN A RANGE OF DISTANCE/BOX SIZE BANDS FOR ESTIMATION OF OBSERVED L-FUNCTION
	
	#Construct the array of all possible points (including duplicates in case of periodic boundaries)
	if periodic_BCs:
		for xoff in [0,nx,-nx]:
			for yoff in [0,-ny,ny]:
				if xoff==0 and yoff==0:
					j9=centroids_updraft.copy()
				else:
					jo=centroids_updraft.copy()
					jo[:,0]+=yoff
					jo[:,1]+=xoff
					j9=np.vstack((j9,jo))
	elif periodic_zonal:
		for xoff in [0,nx,-nx]:
			if xoff==0:
				j9=centroids_updraft.copy()
			else:
				jo=centroids_updraft.copy()
				jo[:,1]+=xoff
				j9=np.vstack((j9,jo))
	else:	
		#If no periodicity is assumed, the array of possible points is just the original one
		j9=centroids_updraft.copy()
	
	#Initialization of the array of cloud-to-cloud nearest-neighbor distances
	NNdist = np.zeros(len(centroids_updraft))
		
	#Initialization of the array whose rows represent the neighbor counting over a range of distances/box sizes (binned) for each element of the pattern
	cum_counting = np.zeros((len(centroids_updraft), len(bins)))
	
	#Determination of all-neighbor distances from each point of the pattern in the original domain. In case of periodic boundaries, multiple counting is avoided.  
	for k in range(len(centroids_updraft)):
		hist = np.zeros(len(bins))
		
		#The k-th object is the base point and all its neighbors are considered. In case of cyclic boundaries, the possible neighbors are all the points in the periodically continued domain, except for the duplications of the base point itself.
		extra_pts = np.delete(j9, list(range(k, j9.shape[0], len(centroids_updraft))), axis=0)
		tree=spatial.cKDTree(extra_pts)
		if periodic_BCs:
			dist,ii=tree.query(centroids_updraft[k,:], 9*(ncnv-1))			
			#Prohibit multiple counting
			indexes = np.sort(np.unique(extra_pts[ii]%[ny,nx], return_index=True, axis = 0)[1])
			dist_new = dist[indexes]
			ii_new = ii[indexes]
			dist, ii = dist_new, ii_new
		elif periodic_zonal:
			#No periodic continuation of the domain along the y-axis, only along x-axis				
			dist,ii=tree.query(centroids_updraft[k,:], 3*(ncnv-1))
			indexes = np.sort(np.unique(extra_pts[ii]%[ny,nx], return_index=True, axis = 0)[1])
			dist_new = dist[indexes]
			ii_new = ii[indexes]
			dist, ii = dist_new, ii_new		
		else:		
			#In case of open domains, no duplications of the domain are performed
			dist,ii=tree.query(centroids_updraft[k,:], ncnv-1)
		
		#Unit conversion from grid pixels to meters
		dist*=dxy
		
		#Storage of nearest-neighbor distances
		NNdist[k] = dist[0]
				
		#If the discrete version of the Besag's function is to be determined, the distances have to be computed on the discrete grid and their zonal and meridional components are considered 
		if binomial_discrete:
			dist_binomial = dxy*np.abs((centroids_updraft[k,:]-tree.data[ii]))
			
			#The size of the box surrounding the k-th object and determined by its j-th neighbor is twice the maximum between the zonal and meridional components of the distance d_{kj}  
			size = 2*np.maximum(dist_binomial[:,0], dist_binomial[:,1])
			
			#Only the box sizes shorter than the maximum allowed size are retained
			size = size[size<=rmax]
			
			#For each object, perform the neighbor counting as a function of distance/box size (cumulative sum). The following procedure is adopted in order to have right-closed intervals, i.e., evaluation of the number of neighbors over boxes of size less or equal than a given value. Note that the bulit-in function numpy.histogram takes right-open bins by definition, with the exception of the last one, hence a different procedure is implemented here
			values,counts = np.unique(np.digitize(size, bins=bins, right=True),  return_counts=True)
			hist[values]=counts
			cum_hist = np.cumsum(hist)
			
			#Definition of edge correction strategies for open domains
			if not periodic_BCs and edge_mode == 'besag':			
				#With the area-based correction technique, the weight is applied to any possible distance (box size) off the base point
				weights = np.zeros(len(bins))
				if periodic_zonal:
					for i,ir in enumerate(bins/2.):
						if ir>0:
							#The boxes centered at the k-th object are clipped to the domain edges. If periodic_zonal is True, this occurs only along the meridional direction 
							ymax = np.min((centroids_updraft[k,0]*dxy+ir, domain_y))
							ymin = np.max((centroids_updraft[k,0]*dxy-ir, 0))	
							#For each distance ir off the k-th base point, computation of the weighting factor as the fractional area of the box of size 2*ir centered on it and contained within the domain								
							weights[i]=2*ir/(ymax-ymin)
				else:
					#Open domain in both directions
					for i,ir in enumerate(bins/2.):
						if ir>0:
							#The boxes are clipped to the domain edges in both the zonal and meridional directions 
							ymax, xmax = np.min(((centroids_updraft[k,:]*dxy+np.array(ir,ir)), np.array([domain_y, domain_x])), axis = 0)
							ymin, xmin = np.max(((centroids_updraft[k,:]*dxy-np.array(ir,ir)), np.array([0,0])), axis = 0)
							#Calculation of the weighting factor
							weights[i]=(2*ir)**2/((ymax-ymin)*(xmax-xmin))
								
				#For each possible size of search boxes centered on the k-th convective object, the weighting factors are assigned to the corresponding counting of neighbors contained within the boxes
				cum_hist = weights*cum_hist
		
		#Continuous (not discrete) domains 							
		else:
			#Only the inter-point distances smaller than the maximum allowed one are retained
			dist = dist[dist<rmax]		
			#For each object, the counting of neighbors is performed as a function of distance (binned) 
			values,counts = np.unique(np.digitize(dist, bins=bins, right=True),  return_counts=True)
			hist[values] = counts
			cum_hist = np.cumsum(hist)
		
		#Storage of the neighbor counting in terms of distance into the array C previously initialized		
		cum_counting[k,:] = cum_hist
	
	##DERIVATION OF THE THEORETICAL AND OBSERVED BESAG'S FUNCTIONS
	#Calculation of the mean number of neighbors off any typical point of the pattern as a function of distance/box size. This is by definition the quantity lambda K(r), lambda being the spatial density of points and K(r) the Ripley's function
	mean_count = np.mean(cum_counting, axis = 0)
	
	#Calculation of OBSERVED Besag's functions
	if binomial_discrete:
		#To get the simulated Besag's function, the square root of the Ripley's function has to be taken. Note that mean_count = lambda K(r), hence K(r) = mean_count/lambda, where lambda is estimated as (ncnv-1)/(domain_x*domain_y) in order to have an unbiased estimator. This is formula eqn. (20) in the paper
		Besag_obs = np.sqrt(mean_count*domain_x*domain_y/(ncnv-1))
	else:
		#Same as above, but with the factor 1/pi for the derivation of the Besag's function from the Ripley's function. This is formula eqn. (11) in the paper	
		Besag_obs = np.sqrt(1/np.pi*mean_count*domain_x*domain_y/(ncnv-1))
		
	#Calculation of THEORETICAL Besag's functions
	
	#Square domains
	if nx == ny:
		if periodic_BCs:
			if binomial_continuous:			
				#Distance beyond which the correction for multiple counting must be included (see Sections 4a and 4c in the manuscript)
				rcrit = domain_x/2.			
				#This is formula eqn. (18) in the paper, normalized by rmax. For reasonable sample sizes (ncnv > 15), the factor (ncnv-1)/ncnv can be dropped. See also code documentation section 2.1.2
				Besag_theor = np.piecewise(bins, [bins<=rcrit, bins>rcrit], [lambda bins: np.sqrt((ncnv-1)/ncnv)*bins, lambda bins: np.sqrt((ncnv-1)/ncnv*1./np.pi*(np.pi*bins**2-4*(bins**2*np.arccos(rcrit/bins)-rcrit*np.sqrt(bins**2-rcrit**2))))])
			else:		
				#This includes cases with periodic boundaries and Poisson and discrete binomial models for spatial randomness, eqns. (10) and (19) in the paper, normalized by rmax (r_max and ell_max in the text respectively). See also code documentation sections 2.1.1 and 2.1.3)
				Besag_theor = bins
		elif periodic_zonal:
			#This is eqn. (1) in the code documentation (section 2.1.5), with approximations already applied
			Besag_theor = np.piecewise(bins, [bins<=min(domain_x, domain_y), bins>min(domain_x, domain_y)], [lambda bins: bins, lambda bins: np.sqrt(bins*min(domain_x, domain_y))])
		else:
			#Open boundary case (see code documentation section 2.1.4)
			Besag_theor = bins
	
	#Non-square domains
	if nx!=ny:
		if periodic_BCs:
			if binomial_continuous:
				min_rcrit = min(domain_x, domain_y)/2.
				max_rcrit = max(domain_x, domain_y)/2.
				#This is formula eqn. (22) in the paper, normalized by rmax. For reasonable sample sizes (ncnv > 15), the factor (ncnv-1)/ncnv can be dropped. See also section 2.2.2 in the code documentation
				Besag_theor = np.piecewise(bins, [bins<=min_rcrit, np.logical_and(bins>min_rcrit, bins<=max_rcrit), bins>max_rcrit], [lambda bins: np.sqrt((ncnv-1)/ncnv)*bins, lambda bins: np.sqrt((ncnv-1)/ncnv*1./np.pi*(np.pi*bins**2-2*(bins**2*np.arccos(min_rcrit/bins)-min_rcrit*np.sqrt(bins**2-min_rcrit**2)))), lambda bins: np.sqrt((ncnv-1)/ncnv*1./np.pi*(np.pi*bins**2-2*(bins**2*np.arccos(min_rcrit/bins)-min_rcrit*np.sqrt(bins**2-min_rcrit**2))-2*(bins**2*np.arccos(max_rcrit/bins)-max_rcrit*np.sqrt(bins**2-max_rcrit**2))))])
			elif binomial_discrete:
				#This is formula eqn. (23) in the paper, normalized by rmax. A simplification similar to eqn. (19) has been performed (see also code documentation section 2.2.3)
				Besag_theor = np.piecewise(bins, [bins<=min(domain_x, domain_y), bins>min(domain_x, domain_y)], [lambda bins: bins, lambda bins: np.sqrt(bins*min(domain_x,domain_y))])
			else:
				#Case corresponding to Poisson distribution for complete spatial randomness and periodicity in both directions (see section 2.2.1 in the code documentation)
				Besag_theor = bins
		elif periodic_zonal:
			#Case of zonally cyclic domains, see code documentation section 2.2.5 
			if (domain_x<domain_y or domain_x<2*domain_y):
				#This is eqn. (2) in the code documentation, with simplifications already applied
				Besag_theor = np.piecewise(bins, [bins<=min(domain_x, 2*domain_y), bins>min(domain_x, 2*domain_y)], [lambda bins: bins, lambda bins: np.sqrt(bins*min(domain_x, 2*domain_y))])
			elif domain_x >= 2*domain_y:
				Besag_theor = bins
		else:			
			#Open boundary case (code documentation section 2.2.4)
			Besag_theor = bins
			
	#Normalization of L-functions is performed.
	Besag_obs=Besag_obs/rmax
	Besag_theor=Besag_theor/rmax
			
	##CALCULATION OF THE INDICES I_ORG/RI_ORG
	#Evaluation of theoretical NNCDF. A different (binned) range of distances is introduced to evaluate the theoretical Weibull NNCDF and construct the observed NNCDF. 
	if periodic_BCs:
		r_Iorg = np.arange(0,np.sqrt((domain_x**2+domain_y**2)/2)+dxy, dxy)
	else:
		r_Iorg = np.arange(0,np.sqrt(domain_x**2+domain_y**2)+dxy, dxy)
		
	bins_Iorg = r_Iorg
	NNCDF_theor = 1-np.exp(-lambd*np.pi*r_Iorg**2)
	
	#Calculation of the NNCDF of the given scene (observed NNCDF). The latter is not computed through the python built-in function numpy.histogram (see note above about the fact that the bins in numpy.histogram are not right-closed, which conflicts with the formal definition of cumulative distribution function)
	values,counts = np.unique(np.digitize(NNdist, bins=bins_Iorg, right=True), return_counts=True)
	hist_Iorg = np.zeros(len(bins_Iorg), dtype = int)
	hist_Iorg[values] = counts
	NNPDF = hist_Iorg/np.sum(hist_Iorg)
	NNCDF_obs = np.cumsum(NNPDF)
	
	#Integration of the joint CDFs to give I_org/RI_org
	I_org = np.trapz(NNCDF_obs, x = NNCDF_theor)
	RI_org = np.trapz(NNCDF_obs-NNCDF_theor, x = NNCDF_theor)
	
	##CALCULATION OF THE INDICES L_ORG/dL_ORG
	L_org = np.trapz(Besag_obs-Besag_theor, x = bins)/rmax
	
	return I_org, RI_org, L_org, NNCDF_theor, NNCDF_obs, Besag_theor, Besag_obs
	