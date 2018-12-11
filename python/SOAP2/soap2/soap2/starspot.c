/******************************************************************************/
/*** file:   starspot.c                                                     ***/
/*** author: X. Bonfils                                                     ***/
/***         Centro de Astronomia e Astrofisica da Universidade de Lisboa   ***/
/***                                                                        ***/
/*** version: 0.1 2006/08/29                                                ***/
/******************************************************************************/

#include <stdlib.h>
#include "starspot.h"
#include <stdio.h>
#include <string.h>
#include <math.h>



FILE *file;            /* declare the file pointer */

int read_rdb(char *filename, double *value)
{    
    char line[1000];
    char *str_split;
    int i=0, j=0;
    
    file = fopen (filename, "r");  /* open the file for reading */
    /* filename is the name of the file */
    /* "r" means open the file for reading text */
    
    if( file == NULL )
    {
        perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }
    
    while (!feof(file))
    {
        if (fgets(line, 1000, file)) // le if permet de ne rien faire si fgets tombe sur EOF (end of file), 
            // sinon il fait un boucle de plus avec les anciennes valeurs, ce qui pose prb
        {
            if (i>=2)
            {
                str_split = strtok (line," \t\n");
                while (str_split != NULL)
                {
                    //printf ("%s\n",str_split);
                    value[j] = atof(str_split);
                    str_split = strtok (NULL, " \t\n");
                    j++;
                }
            }
            i++;
        }
    }
    fclose(file);
    
    return j;
    
}



double rndup(double n,int nb_decimal)//round up a double type at nb_decimal
{
    double t;
    t=n*pow(10,nb_decimal) - floor(n*pow(10,nb_decimal));
    if (t>=0.5)    
    {
        n*=pow(10,nb_decimal);//where n is the multi-decimal double
        n = ceil(n);
        n/=pow(10,nb_decimal);
    }
    else 
    {
        n*=pow(10,nb_decimal);//where n is the multi-decimal double
        n = floor(n);
        n/=pow(10,nb_decimal);
    }
    return n;
}    

//Calculation of the offset (or widening) of a line taking into account the speed and the wavelength, in A
double Delta_lambda(double line_width, double lambda_line0)//line_width in RV [m/s]
{   
    double c=299792458.;
    double line_spread=0.;
	double beta;
	
	// relativistic and symmetric case see "functions.py"
	beta = line_width/c;
	line_spread = -1 * lambda_line0 * (1 - sqrt((1+beta)/(1-beta)));
	
	// v/c*wavelength, cas non relaticviste et non symetrique
    //line_spread = line_width/c*lambda_line0;
	
    return line_spread;
}

// ATTENTION MAUSVAIS CAR ON FAIT "- v_shift * spectrum_derivative[i]", LE MOINS EST DJA COMPRIS DANS LE SPECTRUM DERIVATIVE SI LA DERIVEE EST NEGATIVE
//void shifting_spectrum(double *lambda, double *spectrum, double *spectrum_blueshifted, double *depth_from_continuum, double *fit, int n1, int func_type )
//{
//    int i;
//    double *spectrum_derivative, lambda_shift, v_shift;
//    
//    spectrum_derivative = (double *) malloc(n1*sizeof(double));
//    for (i=0;i<n1;i++) 
//    {
//        spectrum_blueshifted[i] = 0.0;
//    }
//    spectrum_derivative[0] = 0.;
//    for (i=1;i<n1-1;i++)
//    {
//        if (spectrum[i+1]==0 || spectrum[i-1]==0) // avant il y avait spectrum[i+i]==0 a la place de spectrum[i+1]==0 mais je pense que c etait une erreur
//        {
//            spectrum_derivative[i] = 0;
//        }
//        else 
//        {
//            spectrum_derivative[i] = (spectrum[i+1] - spectrum[i-1])/(lambda[i+1]-lambda[i-1]);
//        }
//    }
//    spectrum_derivative[n1-1] = 0.;
//    
//    for (i=0;i<n1;i++)
//    {
//        //if (i%100 == 0) printf("%d over %d\n",i,n1);
//        //printf("%d   %.2f  %.2f  %.2f   %.2f   %.2f\n",i,spectrum[i+1],spectrum[i-1],lambda[i+1],lambda[i-1],spectrum_derivative[i]);
//        //v_shift = maximum_blueshift_conv*depth_from_continuum[i];
//        //See "shift_mask_holes_to_meet_conv_blueshift" for the estimation of these terms
//        //v_shift = 6719.15208437*pow(depth_from_continuum[i],4) - 16207.45372463*pow(depth_from_continuum[i],3) + 15054.13259329*pow(depth_from_continuum[i],2) - 6711.32809434*depth_from_continuum[i] + 862.14880259;
//        v_shift = fit[0]*pow(depth_from_continuum[i],2) + fit[1]*depth_from_continuum[i] + fit[2];
//        
//        if (func_type == 1) // 1 if blueshifting the CCF
//        {
//            spectrum_blueshifted[i] = spectrum[i] - v_shift * spectrum_derivative[i]; // dy/dx = derivative => dy = derivative*dx
//        }
//        else // if blueshifting a spectrum
//        {
//            lambda_shift = Delta_lambda(v_shift,lambda[i]);
//            spectrum_blueshifted[i] = spectrum[i] - lambda_shift * spectrum_derivative[i]; // dy/dx = derivative => dy = derivative*dx
//        }
//    }
//    
//    free(spectrum_derivative);
//}

// SEE FILE TEST_SHIFTING_SPECTRUM.TXT IN FILE PROGRAM / TEST /.
void shifting_spectrum(double *lambda, double *spectrum, double *spectrum_blueshifted, double *depth_from_continuum, double *fit, int n1, int func_type )
{
    int i;
    double *spectrum_derivative, lambda_shift, v_shift;
	
    spectrum_derivative = (double *) malloc(n1*sizeof(double));
    for (i=0;i<n1;i++) 
    {
		spectrum_derivative[i] = 0.0;
    }
	
	if (lambda[1]-lambda[0] == 0) spectrum_derivative[0] = 0;
	else spectrum_derivative[0] = (spectrum[1] - spectrum[0])/(lambda[1]-lambda[0]);
	
	for (i=1;i<n1-1;i++)
	{
		if (lambda[i+1]-lambda[i] == 0) spectrum_derivative[i] = spectrum_derivative[i-1];
		else
		{
			spectrum_derivative[i] = (spectrum[i+1] - spectrum[i])/(lambda[i+1]-lambda[i]);
		}
	}
	spectrum_derivative[i] = 0.;
	
	if (func_type == 1)
	{
		for (i=0;i<n1;i++)
		{
			v_shift = fit[0]*pow(depth_from_continuum[i],2) + fit[1]*depth_from_continuum[i] + fit[2];
			
			if ((v_shift >= 0) || (i == 0)) 
			{
				spectrum_blueshifted[i] = spectrum[i] - v_shift * spectrum_derivative[i]; // dy/dx = derivative => dy = derivative*dx, pour le "-" voir program/convetive_blueshift/compute_spectrum.c
			}
			else if (v_shift < 0)
			{
				spectrum_blueshifted[i] = spectrum[i] - v_shift * spectrum_derivative[i-1]; // dy/dx = derivative => dy = derivative*dx, pour le "-" voir program/convetive_blueshift/compute_spectrum.c
			}
		}
	}
    // if the last value is 2, the value "depth_from_continuum" is the difference between the sampling we want and the actual one
    // depth_from_continuum = freq_wanted - freq_reelle
    // MUST USE + depth_from_continuum = freq_wanted - freq_reelle because the CCF is the same, but we decale the points in speed
    // TO KEEP THE SAME CCF (OR SPECTRUM) GO TO SHIFT THE POINTS IN VELOCITY (OR WAVELENGTH)
	else if (func_type == 2)
	{
		for (i=0;i<n1;i++)
		{
			if ((depth_from_continuum[i] >= 0) || (i == 0)) 
			{
				spectrum_blueshifted[i] = spectrum[i] + depth_from_continuum[i] * spectrum_derivative[i]; // dy/dx = derivative => dy = derivative*dx
			}
			else if (depth_from_continuum[i] < 0)
			{
				spectrum_blueshifted[i] = spectrum[i] + depth_from_continuum[i] * spectrum_derivative[i-1]; // dy/dx = derivative => dy = derivative*dx
			}
        }
	}
    else 
	{
		for (i=0;i<n1;i++)
		{
			v_shift = fit[0]*pow(depth_from_continuum[i],2) + fit[1]*depth_from_continuum[i] + fit[2];
			
			lambda_shift = Delta_lambda(v_shift,lambda[i]);
            spectrum_blueshifted[i] = spectrum[i] - lambda_shift * spectrum_derivative[i];
			if ((lambda_shift >= 0) || (i == 0)) 
			{
                // I do not understand why we have to put [i-1] for "lambda_shift" positive and [i] for "lambda_shift" negative. But that seems very easy when we use the spectra
				spectrum_blueshifted[i] = spectrum[i] - lambda_shift * spectrum_derivative[i-1]; // dy/dx = derivative => dy = derivative*dx, pour le "-" voir program/convetive_blueshift/compute_spectrum.c
			}
			else if (lambda_shift < 0)
			{
				spectrum_blueshifted[i] = spectrum[i] - lambda_shift * spectrum_derivative[i]; // dy/dx = derivative => dy = derivative*dx, pour le "-" voir program/convetive_blueshift/compute_spectrum.c
			}
		}
	}
        
	free(spectrum_derivative);
}
	

double loi_Planck(double lambda0, double Temp)
{
    double c   = 299792458.;     // speed of light in m / s
    double h   = 6.62606896e-34; // Planck's constant
    double k_b = 1.380e-23;      // Boltzmann's constant
    return 2*h*pow(c,2)*1./pow(lambda0,5)*1./(exp((h*c)/(lambda0*k_b*Temp))-1); // Planck's law
}

void obtain_same_sampling(double *freq, int n1,double *freq_oversampled, int n2, double *flux_oversampled,double *freq_same, double *flux_same, double *index_same, double *diff)
{
    int i,j=0;
    double diff_min;
    
    for (i=0;i<n1;i++)
    {
        //if (i%1000==0) printf("%d over %d\n",i,n1);
        diff_min=1.e30;
        while((fabs(freq_oversampled[j]-freq[i]) < diff_min) && (j < n2))
        {   
            diff_min = fabs(freq_oversampled[j]-freq[i]);
            j += 1;
        }
		j -= 1;
        freq_same[i] = freq_oversampled[j];
        flux_same[i] = flux_oversampled[j];
        index_same[i] = j;
        diff[i] = freq[i]-freq_oversampled[j];
    }
}

/************************************************* ************************************** /
/*** itot measures the flow in each box of the position grid *** /
/*** Then the total flow of the star for a certain speed (= wavelength) *** /
/*** is the sum of each box in the grid. Then to come back to a CCF *** /
/*** normalized (= similar to the flow) we divide by the max of the CCF *** /
/* Google translate:
It then measures the flow of the grid position
total flow of the star for a certain speed (= wavelength) Is the sum of
each square of the grid. Then back to a Normalized CCF (= similar to the flow) is divided by the CCF * /
/ ************************************************* **************************************/
void itot(double v, double i, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid,
	  double *vrad_ccf, double *intensity_ccf, double v_interval, int n_v, int n,
	  double *f_star, double *sum_star)
{
  /*
   * v     [km/s]          stellar rotation
   * i     [degree]        inclination of the rotational axis / sky plane
   * limba [0-1.]          coefficient of the linear line-darkening law 
   * grid
   * intenisty_ccf [0-1.]  vecteur de l intensite de la CCF
   * v_intevral
   * n_v
   * f_star
   * sum_star
   */
  double omega, delta_grid, delta_v;
  double y, z, delta, r_cos, limb;
  int ii, iy, j, iz;
  int max_shift=2,min_wave=0,max_wave=10000000; // cut "max_shift" Angstrom on each side of the spectrum, so all the specrum can then be cut on the same pixel scale
  //double conv_blueshift_fit[3] = {-896.27246575,  1334.40727695,  -179.3847372};
  double *depth_from_continuum, *intensity_ccf_blueshifted, *intensity_ccf_blueshifted_shift,*vrad_ccf_select,*vrad_ccf_shift;
  double *freq_same,*flux_same,*index_same,*diff;
    
  intensity_ccf_blueshifted = (double *)malloc(sizeof(double)*n);
  depth_from_continuum      = (double *)malloc(sizeof(double)*n);
  vrad_ccf_shift            = (double *)malloc(sizeof(double)*n);
    
  for (ii=0;ii<n;ii++)
  {
      //printf("vrad_ccf[j] = %.2f, (vrad_ccf[j]-(vrad_ccf[0]+max_shift)) = %.2f\n",vrad_ccf[j],(vrad_ccf[j]-(vrad_ccf[0]+max_shift)));
      if ((vrad_ccf[ii]-(vrad_ccf[0]+max_shift))<0) min_wave = ii;
      if ((vrad_ccf[ii]-(vrad_ccf[n-1]-max_shift))<0) max_wave = ii;
  }
  int n_select = max_wave-min_wave; // len of the select wavelength
  vrad_ccf_select                 = (double *)malloc(sizeof(double)*n_select);
  intensity_ccf_blueshifted_shift = (double *)malloc(sizeof(double)*n_select);
    
  freq_same  = (double *)malloc(sizeof(double)*n_select);
  flux_same  = (double *)malloc(sizeof(double)*n_select);
  index_same = (double *)malloc(sizeof(double)*n_select);
  diff       = (double *)malloc(sizeof(double)*n_select);
    
  //initialisation
  for (ii=0;ii<n;ii++) 
  {
      depth_from_continuum[ii] = 1.0;
      intensity_ccf_blueshifted[ii] = 0.0;
      vrad_ccf_shift[ii] = 0.0;
  }
  for (ii=0;ii<n_select;ii++) 
  {
      vrad_ccf_select[ii] = vrad_ccf[ii+min_wave];
      intensity_ccf_blueshifted_shift[ii] = 0.0;
      freq_same[ii]  = 0.0;
      flux_same[ii]  = 0.0;
      index_same[ii] = 0.0;
      diff[ii]       = 0.0;
  }
    
  //printf("n =%i, n select =%i, min_wave = %i, max_wave = %i, min(vrad_ccf_select) = %.2f, max(vrad_ccf_select) = %.2f \n",n,n_select,min_wave,max_wave,vrad_ccf_select[0],vrad_ccf_select[n_select-1]);

  /* Conversions */
  i = i * pi/180. ; // [degree]       --> [radian]

  omega = v;
  delta_grid = 2./grid; // step of the grid. grid goes from -1 to 1, therefore 2 in total
  // v_interval is from the velocity 0 to the edge of the spectrum taking into account minimal or maximal rotation (width - v to 0 or 0 to width + v). 
  // n_v is the number of points for all the CCF from minimum rotation to maximum one (from width - v to width + v).
  // n_v represent therefore the double than v_intervel, we therefore have to multiply v_interval by 2.
  delta_v = 2.*v_interval/(n_v-1); //step in speed of the CCF. There is (n_v-1) intervals

  /* ..... Calculate the total stellar intensity (without spots) ...*/
  #if VERBOSE
  printf("Total stellar intensity \n");
  #endif
  // y-axis scan...
  for (iy=0; iy<=grid; iy++) 
  {
    //printf("%i over %i -----------\n",iy,grid);
    y = -1. + iy*delta_grid; // y entre -1 et 1
    delta = y * omega * sin(i);  // donne la vitesse de rotation (ce qui change le centre de la CCF donc la vitesse) en fonction de
                                 // y. Donc pour y=-1 => Vrot min et pour y=1 => Vrot max. C est juste, pour y=0 => Vrot = 0.
    // z-axis scan
    for (iz=0; iz<=grid; iz++) 
    {
      z = -1. + iz*delta_grid;// z entre -1 et 1
      if ((y*y+z*z)<=1.) 
      {
          //limb-darkening
          r_cos = pow(1.-(y*y+z*z),.5); //cos teta, cos teta = 1 au centre de l etoile, cos teta = 0 sur les bords
          //printf("%f   %f  diff=%f\n",r_cos,r_cos_tmp,fabs(r_cos - r_cos_tmp));
		  
          //limb = 1.-limba+limba*r_cos;  // linear limb-darkening law 
          limb =  1. - limba1*(1-r_cos) - limba2*(1-r_cos)*(1-r_cos); // voir SOAP-T paper, coming from Mandel & Agol 2002
          
		  // change the line bissector as a function of the limb to center position (r_cos)
		  //double fit_bis[3] = {756.47158996/1000.*(1-r_cos),-1159.69528179/1000.*(1-r_cos),67.38120081/1000.*(1-r_cos)}; //see create_Sun_ccf.py for these values
          if (modif_bis_cte != 0.0)
          {
              //printf("APPLYING THE BISECTOR SHAPE MODIFICATION WITH THE LIMB-CENTER ANGLE");
              double fit_bis[3] = {modif_bis_quad*(1-r_cos),modif_bis_lin*(1-r_cos),modif_bis_cte*(1-r_cos)}; //see create_Sun_ccf.py for these values
              shifting_spectrum(vrad_ccf, intensity_ccf, intensity_ccf_blueshifted,intensity_ccf,fit_bis,n,1);
		  }
          else 
          {
              for (j=0;j<n;j++)
              {
                  intensity_ccf_blueshifted[j] = intensity_ccf[j];
              }
          }
          // Shift the wavelength to the the velocity of the cell in which we are putting the spectrum
          for (j=0;j<n;j++)
          {
              vrad_ccf_shift[j] = vrad_ccf[j] + Delta_lambda(delta*1000.,vrad_ccf[j]);
          }
          // function to calculate that "vrad_ccf_shift" and "vrad_ccf_select" have similar pixel scale, see "compute_spectrum.c" et "soap_comparison_SOAP_T_and_modified_version_using_real_spectra"
          obtain_same_sampling(vrad_ccf_select,n_select,vrad_ccf_shift,n,intensity_ccf_blueshifted,freq_same,flux_same,index_same,diff);
          // si la derniere valeur est 2, la valeur "depth_from_continuum" correspond a l ecart entre le sampling reel et celui que l on veut
          double fit[3] = {0,0,0};
          shifting_spectrum(freq_same, flux_same, intensity_ccf_blueshifted_shift, diff, fit, n_select, 2);
          
          for (j=min_wave;j<=max_wave;j++) f_star[j] += intensity_ccf_blueshifted_shift[j-min_wave]*limb; // ATTENTION, intensity_ccf_blueshifted_shift va de 0 a max_wave-min_wave
          *sum_star += intensity_ccf_blueshifted_shift[n_select-1]*limb; 
      }
    }
    #if VERBOSE
    if ((iy%10)==0) printf(".");
    #endif
  }
  #if VERBOSE
  printf("OK\n");
  #endif
    
  free(intensity_ccf_blueshifted); free(intensity_ccf_blueshifted_shift);
  free(depth_from_continuum);
  free(vrad_ccf_select);free(vrad_ccf_shift);
  free(freq_same);free(flux_same);free(index_same);free(diff);
}

void starmap(double v, double i, double limba1, double limba2, int grid,
	double **Fmap, double **Vmap)
{
  /*
   * v     [km/s]          stellar rotation
   * i     [degree]        inclination of the rotational axis / sky plane
   * limba [0-1.]          coefficient of the linear line-darkening law 
   * grid
   * Fmap  [arb. unit]     Flux map (2D array)
   * Vmap  [km/s]          Velocity map (2D array)
   */
	
  double r_cos, limb, y, z, delta_grid, delta_v;
  int iy, iz;
  v = v*sin(i/180.*pi);
  delta_grid = 2./grid;
  delta_v    = 2.*v/grid;
	
  // y-axis scan...
  for (iy=0; iy<grid; iy++) {
    y = -1 + iy*delta_grid;
    // z-axis scan
	  
    for (iz=0; iz<grid; iz++) {
      z = -1 + iz*delta_grid;
		
      if ((y*y+z*z)<1) {
        //limb darkening
        r_cos = sqrt(1-(y*y+z*z)); //mu = cos teta = sqrt(1-r^2), ou r est la coordonee radial sur le disque stellaire (normalise a 1)
        //limb = 1.-limba+limba*r_cos;       // linear limb-darkening law 
        limb =  1. - limba1*(1-r_cos) - limba2*(1-r_cos)*(1-r_cos); // voir SOAP-T paper, coming from Mandel & Agol 2002
        Fmap[iy][iz]= limb;
        Vmap[iy][iz]= -v+iy*delta_v;
      }
      else {
		Fmap[iy][iz]=0;
		Vmap[iy][iz]=-9999;
	  }
    }
  }
	
//printf("OK\n");
	
}

void spot_init(double s, double longitude, double latitude, double inclination, 
	       int nrho, double **xyz)
{
  /* Position of the spot initialized at the equator 
     (star rotates around the z axis) 

     s [spot radius] 
     longitude [degree]
     latitude  [degree]
     inclination [degree] i=0  -> pole-on (North)
                          i=90 -> equator-on
     nrho : Spot circonference resolution
     xyz

   */
  // This procedure requires xyz, an array of pointers
  // on pointers of type double.
  // Memory has to be allocated as follow :
  //   xyz    = (double **)malloc(sizeof(double *)*nrho);
  //   for (j=0; j<nrho; j++) xyz[j] = (double *)malloc(sizeof(double)*3);
  double *rho, rho_step, **xyz2;
  int j;

  /* Convertions [deg] -> [rad] */
  longitude   *= pi/180.;
  latitude    *= pi/180.;
  inclination *= pi/180.;

  // In this initial position we calculate the coordinates (x,y,z) of 
  // points of the spot's circonference
  rho = (double *)malloc(sizeof(double)*nrho);
  xyz2 = (double **)malloc(sizeof(double *)*nrho);
  for (j=0; j<nrho; j++) xyz2[j] = (double *)malloc(sizeof(double)*3);
    rho_step = 2.*pi/(nrho-1); //a spot has a circumferential resolution of 20, so we have one point every 2 * pi / (nrho-1). -1 because there is (nrho-1) interval from 0 to nrho
  for (j=0; j<nrho; j++) rho[j] = -pi + j*rho_step; // rho goes from -pi a pi
  for (j=0; j<nrho; j++) xyz2[j][0] = pow(1-s*s,.5); // sqrt(r*r-s*s) with r=1 (radius of the star), s is in the same unit. Since the spot has a radius, the circonference 
                                                     // of the spot is not at 1, but if you image a triangle between the center of the spot, its circonference and the center of the star, the position in x is (sqrt(R^2-s^2))
  for (j=0; j<nrho; j++) xyz2[j][1] = s*cos(rho[j]); //projection sur l axe des y
  for (j=0; j<nrho; j++) xyz2[j][2] = s*sin(rho[j]); //projection sur l axe des z
  free(rho);

  // to account for the real projection of the spot, we rotate the star and look how the coordinates of the circumfernce of the spot change
  // Position according to latitude, longitude and inclination. It consists of 
  // three rotations.
  //
  // Conventions :
  // -when inclination=0 the star rotates around z axis
  // -line of sight is along x-axis
  // -and sky plane = yz-plane
  //
  // Be Rx(alpha), Ry(beta), Rz(gamma) the rotations around the x, y and z axis
  // respectively, with angles alpha, beta and gamma (counter-clockwise 
  // direction when looking toward the origin).
  //
  // The rotations to apply are:
  //   Ry(inclination) x Rz(longitude) x Ry(latitude) x X(x,y,z)
  // 
  //         |  cos(b)  0  sin(b) |                       | cos(g) -sin(g) 0 |
  // Ry(b) = |    0     1    0    |               Rz(g) = | sin(g)  cos(g) 0 |
  //         | -sin(b)  0  cos(b) |                       |   0       0    1 |
  //
  //
  // |x'|   |  cos(b2)cos(g)cos(b)-sin(b)sin(b2)  -sin(g)cos(b2)  cos(b2)cos(g)sin(b)+sin(b2)cos(b) |   |x|
  // |y'| = |              sin(g)cos(b)               cos(g)                   sin(g)sin(b)         | x |y|
  // |z'|   |  -sin(b2)cos(g)cos(b)-cos(b2)sin(b)  sin(b2)sin(g) -sin(b2)cos(g)sin(b)+cos(b2)cos(b) |   |z|

  double b  = -latitude;
  double g = longitude;
  double b2 = pi/2.-inclination;

  double R[3][3] = {{cos(b2)*cos(g)*cos(b)-sin(b)*sin(b2), -sin(g)*cos(b2), cos(b2)*cos(g)*sin(b)+sin(b2)*cos(b)},
                    {sin(g)*cos(b),                          cos(g),          sin(g)*sin(b)},
                    {-sin(b2)*cos(g)*cos(b)-cos(b2)*sin(b), sin(b2)*sin(g), -sin(b2)*cos(g)*sin(b)+cos(b2)*cos(b)}};
  for (j=0; j<nrho; j++) xyz[j][0] = R[0][0]*xyz2[j][0] + R[0][1]*xyz2[j][1] + R[0][2]*xyz2[j][2];
  for (j=0; j<nrho; j++) xyz[j][1] = R[1][0]*xyz2[j][0] + R[1][1]*xyz2[j][1] + R[1][2]*xyz2[j][2];
  for (j=0; j<nrho; j++) xyz[j][2] = R[2][0]*xyz2[j][0] + R[2][1]*xyz2[j][1] + R[2][2]*xyz2[j][2];

  // Unallocation
  for (j=0; j<nrho; j++) free(xyz2[j]);
  free(xyz2); 
}


void spot_phase(double **xyz, double inclination, int nrho, double phase, 
		double **xyz2)
{

  int i;
    double psi = -phase*(2*pi); //phase between
  inclination = inclination*pi/180.; 

  double axe[3]  = {cos(inclination),0,sin(inclination)};
  double R[3][3] = {{(1-cos(psi))*axe[0]*axe[0] + cos(psi), 
                     (1-cos(psi))*axe[0]*axe[1] + sin(psi)*axe[2],
                     (1-cos(psi))*axe[0]*axe[2] - sin(psi)*axe[1]},
                    {(1-cos(psi))*axe[1]*axe[0] - sin(psi)*axe[2],
                     (1-cos(psi))*axe[1]*axe[1] + cos(psi),
                     (1-cos(psi))*axe[1]*axe[2] + sin(psi)*axe[0]},
                    {(1-cos(psi))*axe[2]*axe[0] + sin(psi)*axe[1],
                     (1-cos(psi))*axe[2]*axe[1] - sin(psi)*axe[0],
                     (1-cos(psi))*axe[2]*axe[2] + cos(psi)}};

  for (i=0; i<nrho; i++) {
    xyz2[i][0] = R[0][0]*xyz[i][0] + R[0][1]*xyz[i][1] + R[0][2]*xyz[i][2];
    xyz2[i][1] = R[1][0]*xyz[i][0] + R[1][1]*xyz[i][1] + R[1][2]*xyz[i][2];
    xyz2[i][2] = R[2][0]*xyz[i][0] + R[2][1]*xyz[i][1] + R[2][2]*xyz[i][2];
  }
}


int spot_area(double **xlylzl, int nrho, int grid, int *iminy, int *iminz, 
	       int *imaxy, int *imaxz)
{
  /* Determine a smaller yz-area of the stellar disk, where the spot is.        */
  // The different cases are :
  // - the spot is completely visible (x always >=0)
  // - the spot is completely invisible (x always <0)
  // - the spot is on the disk edge and partially visible only
  int j, visible=0;
  double grid_step = 2./grid;
  double miny=1, minz=1, maxy=-1, maxz=-1; // init to 'oposite'-extream values
  int counton=0, countoff=0; // count how many points of the circonference are
                             // visible and how many are invisible
  for (j=0; j<nrho; j++)
    if (xlylzl[j][0]>=0) { // if x>=0
      counton += 1;
      // permet de selectionner les points extremaux de la circonference du spot
      if (xlylzl[j][1]<miny) miny = xlylzl[j][1];
      if (xlylzl[j][2]<minz) minz = xlylzl[j][2]; 
      if (xlylzl[j][1]>maxy) maxy = xlylzl[j][1]; 
      if (xlylzl[j][2]>maxz) maxz = xlylzl[j][2];
    }
    else countoff = 1;

  if ((counton>0)&&(countoff>0)) { // There are both visible and invisible points
                                   // --> spot is on the edge
    // In this situation there are cases where the yz-area define above is 
    // actually smaller than the real area of the spot on the stellar disk.
    // The minima/maxima are over/under-estimated if the spot is on one of the 
    // axis (y or z). Because if on the y axis, the minimum (or maximum) won t be on the circonference of the spot. Same for z axis
    if (miny*maxy<0) {       //spot on the z-axis
      if (minz<0) minz=-1;   //spot on the bottom-z axis (z<0)
      else maxz=1;}          //spot on the top-z axis (z>=0)
    if (minz*maxz<0) {      //spot on the y-axis
      if (miny<0) miny=-1;   //spot on the left hand-y axis (y<0)
      else maxy=1;}          //spot on the right hand-y axis (y>=0)
  };
  if (counton==0) visible = 0;
  else visible = 1;
  
  /* Indices of miny, minz,... on the grid */
  *iminy = floor((1.+miny)/grid_step); //floor(x) returns the largest integral value that is not greater than x. floor of 2.3 is 2.0, floor of 3.8 is 3.0, floor of -2.3 is -3.0, floor of -3.8 is -4.0
  *iminz = floor((1.+minz)/grid_step); //floor(x) returns the largest integral value that is not greater than x. floor of 2.3 is 2.0, floor of 3.8 is 3.0, floor of -2.3 is -3.0, floor of -3.8 is -4.0
  *imaxy = ceil((1.+maxy)/grid_step);  //ceil(x) returns the smallest integral value that is not less than x. ceil of 2.3 is 3, ceil of 3.8 is 4.0, ceil of -2.3 is -2.0, ceil of -3.8 is -3.0
  *imaxz = ceil((1.+maxz)/grid_step);  //ceil(x) returns the smallest integral value that is not less than x. ceil of 2.3 is 3, ceil of 3.8 is 4.0, ceil of -2.3 is -2.0, ceil of -3.8 is -3.0
  
  return visible;
}

void spot_scan(double v, double i, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, 
          double *vrad_ccf, double *intensity_ccf, double *intensity_ccf_spot, double v_interval, int n_v, int n,
          double s, double longitude, double phase, double latitude, 
	      int iminy, int iminz, int imaxy, 
	      int imaxz, double *f_spot_flux, double *f_spot_bconv, double *f_spot_tot, double *sum_spot,
          int magn_feature_type, int T_star, int T_diff_spot)
{
    /* Scan of the yz-area where the spot is.                                   */
    // For each grid-point (y,z) we need to check whether it belongs to the spot
    // or not. Sadly, we do not know the projected geometry of the spot in its
    // actual position. Thus, we have to do an inverse rotation to replace the
    // grid point where it would be in the initial configuration. Indeed, in the
    // initial configuration, the spot has a well known geometry of a circle 
    // centered on the x-axis.
    int j, iy, iz,ii,n_v_shifted_quotien;
    int max_shift=2,min_wave=0,max_wave=10000000; // cut "max_shift" Angstrom on each side of the spectrum, so all the specrum can then be cut on the same pixel scale
    double n_v_shifted, n_v_shifted_rest;
    double y, z;
    double delta_grid=2./grid, delta, r_cos;
    double limb, delta_v = 2.*v_interval/(n_v-1);
    double *xayaza; // actual coordinates
    double *xiyizi; // coordinates transformed back to the initial configuration
    double *depth_from_continuum, *intensity_ccf_spot_shift;
    double *intensity_ccf_blueshifted, *intensity_ccf_blueshifted_shift,*vrad_ccf_select,*vrad_ccf_shift;
    int T_spot,T_plage;
	double intensity,loi_Planck_star;
    double *freq_same,*flux_same,*index_same,*diff,*freq_same_spot,*flux_same_spot,*index_same_spot,*diff_spot;
    
    loi_Planck_star = loi_Planck(5293.4115e-10,T_star); //the wavelength of the Kitt peak strum go from 3921.2441+6665.5789, the mean being 5293.4115
    
    xayaza             = (double *)malloc(sizeof(double)*3);
    xiyizi             = (double *)malloc(sizeof(double)*3);

    intensity_ccf_blueshifted = (double *)malloc(sizeof(double)*n);
    depth_from_continuum     = (double *)malloc(sizeof(double)*n);
    vrad_ccf_shift            = (double *)malloc(sizeof(double)*n);
    
    for (ii=0;ii<n;ii++)
    {
        //printf("vrad_ccf[j] = %.2f, (vrad_ccf[j]-(vrad_ccf[0]+max_shift)) = %.2f\n",vrad_ccf[j],(vrad_ccf[j]-(vrad_ccf[0]+max_shift)));
        if ((vrad_ccf[ii]-(vrad_ccf[0]+max_shift))<0) min_wave = ii;
        if ((vrad_ccf[ii]-(vrad_ccf[n-1]-max_shift))<0) max_wave = ii;
    }
    int n_select = max_wave-min_wave; // len of the select wavelength
    vrad_ccf_select                 = (double *)malloc(sizeof(double)*n_select);
    intensity_ccf_blueshifted_shift = (double *)malloc(sizeof(double)*n_select);
    intensity_ccf_spot_shift = (double *)malloc(sizeof(double)*n_select);
    
    freq_same  = (double *)malloc(sizeof(double)*n_select);
    flux_same  = (double *)malloc(sizeof(double)*n_select);
    index_same = (double *)malloc(sizeof(double)*n_select);
    diff       = (double *)malloc(sizeof(double)*n_select);
    freq_same_spot  = (double *)malloc(sizeof(double)*n_select);
    flux_same_spot  = (double *)malloc(sizeof(double)*n_select);
    index_same_spot = (double *)malloc(sizeof(double)*n_select);
    diff_spot       = (double *)malloc(sizeof(double)*n_select);
    
    //initialisation
    for (ii=0;ii<n;ii++) 
    {
        depth_from_continuum[ii] = 1.0;
        intensity_ccf_blueshifted[ii] = 0.0;
        vrad_ccf_shift[ii] = 0.0;
    }
    for (ii=0;ii<n_select;ii++) 
    {
        vrad_ccf_select[ii] = vrad_ccf[ii+min_wave];
        intensity_ccf_blueshifted_shift[ii] = 0.0;
        intensity_ccf_spot_shift[ii] = 0.0;
        freq_same[ii]  = 0.0;
        flux_same[ii]  = 0.0;
        index_same[ii] = 0.0;
        diff[ii]       = 0.0;
        freq_same_spot[ii]  = 0.0;
        flux_same_spot[ii]  = 0.0;
        index_same_spot[ii] = 0.0;
        diff_spot[ii]       = 0.0;
    }
    
    //printf("n =%i, n select =%i, min_wave = %i, max_wave = %i, min(vrad_ccf_select) = %.2f, max(vrad_ccf_select) = %.2f \n",n,n_select,min_wave,max_wave,vrad_ccf_select[0],vrad_ccf_select[n_select-1]);
    

    //initialisation
    for (ii=0;ii<n;ii++) depth_from_continuum[ii] = 1;
        
    // y-scan
    for (iy=iminy; iy<imaxy; iy++) 
    {
        y = -1.+iy*delta_grid;
        delta = y * v * sin(i*pi/180.);
        xayaza[1] = y;

        // z-scan
        for (iz=iminz; iz<imaxz; iz++) 
        {
            z = -1.+iz*delta_grid;
            if (z*z+y*y<1.)     // z*z+y*y<1. si on est sur le disque de l etoile
            {
                xayaza[0] = pow(1.-(y*y+z*z),.5); //sqrt(abs(r*r-(y*y+z*z))); nous donne la position en x car , car r = rayon de l etoile
                xayaza[2] = z;
                // xayaza[1] = y, voir plus haut

                // xayaza --> xiyizi
                spot_inverse_rotation(xayaza,longitude,latitude,i,phase,xiyizi); 
				
				// On calcul le spectre de l etoile sans spot a cet endroit, puis on supprime la contribution du spot sur ce spectre.
                if (xiyizi[0]*xiyizi[0]>=1.-s*s) // si x^2 >= r^2-s^2, defini la region couverte par le spot
                { 
                    //limb-darkening
                    r_cos = pow(1.-(y*y+z*z),.5); //cos teta, cos teta = 1 au centre de l etoile donc teta = 0 au centre. cos teta = 0 sur les bords et donc cos teta = 90 sur les bords
                    //printf("%f   %f  diff=%f     stepy=%d  stepz=%d  jj=%d\n",r_cos,r_cos_tmp,fabs(r_cos - r_cos_tmp),imaxy-iminy,imaxz-iminz,jj);
                    //limb = 1.-limba+limba*r_cos;  // linear limb-darkening law 
                    limb =  1. - limba1*(1-r_cos) - limba2*(1-r_cos)*(1-r_cos); // voir SOAP-T paper, coming from Mandel & Agol 2002
										
                    if (magn_feature_type==0)
                    {
                        T_spot = T_star-T_diff_spot;
                        //printf("T_spot %i= \n",T_spot);
                        intensity = loi_Planck(5293.4115e-10,T_spot)/loi_Planck_star;  //the wavelength of the Kitt peak strum go from 3921.2441+6665.5789, the mean being 5293.4115
						//printf("intensity spot = %f \n",intensity);
                    }
                    else 
                    {
//                        double mu = cos(latitude*2*3.1415926535897931/360.);
                        T_plage = T_star+250.9-407.7*r_cos+190.9*pow(r_cos,2); //plages are brighter on the limb Meunier 2010
                        //printf("T_star, T_plage =  %i\t%i \n",T_star,T_plage);
                        intensity = loi_Planck(5293.4115e-10,T_plage)/loi_Planck_star; //the wavelength of the Kitt peak strum go from 3921.2441+6665.5789, the mean being 5293.4115
						//printf("intensity plage = %f   mu=%f\n",intensity,r_cos);
                    }      
                    
					// change the line bissector as a function of the limb to center position (r_cos)
                    //double fit_bis[3] = {756.47158996/1000.*(1-r_cos),-1159.69528179/1000.*(1-r_cos),67.38120081/1000.*(1-r_cos)}; //see create_Sun_ccf.py for these values
                    
                    if (modif_bis_cte != 0.0)
                    {
                        double fit_bis[3] = {modif_bis_quad*(1-r_cos),modif_bis_lin*(1-r_cos),modif_bis_cte*(1-r_cos)}; //see create_Sun_ccf.py for these values
                        shifting_spectrum(vrad_ccf, intensity_ccf, intensity_ccf_blueshifted,intensity_ccf,fit_bis,n,1);
                    }
                    else 
                    {
                        for (j=0;j<n;j++)
                        {
                            intensity_ccf_blueshifted[j] = intensity_ccf[j];
                        }
                    }
					
                    // Shift the wavelength to the the velocity of the cell in which we are putting the spectrum
                    for (j=0;j<n;j++)
                    {
                        vrad_ccf_shift[j] = vrad_ccf[j] + Delta_lambda(delta*1000.,vrad_ccf[j]);
                    }
                    // function to calculate that "vrad_ccf_shift" and "vrad_ccf_select" have similar pixel scale, see "compute_spectrum.c" et "soap_comparison_SOAP_T_and_modified_version_using_real_spectra"
                    obtain_same_sampling(vrad_ccf_select,n_select,vrad_ccf_shift,n,intensity_ccf_blueshifted,freq_same,flux_same,index_same,diff);
                    obtain_same_sampling(vrad_ccf_select,n_select,vrad_ccf_shift,n,intensity_ccf_spot,freq_same_spot,flux_same_spot,index_same_spot,diff_spot);
                    // if the last value is 2, the value "depth_from_continuum" is the difference between the actual sampling and the one we want
                    double fit[3] = {0,0,0};
                    shifting_spectrum(freq_same, flux_same, intensity_ccf_blueshifted_shift, diff, fit, n_select, 2);
                    shifting_spectrum(freq_same_spot, flux_same_spot, intensity_ccf_spot_shift, diff_spot, fit, n_select, 2);
                    
                    for (j=min_wave;j<=max_wave;j++)
                    {
                        // ATTENTION, intensity_ccf_blueshifted_shift goes from 0 to max_wave-min_wave
                        f_spot_flux[j]  += intensity_ccf_blueshifted_shift[j-min_wave]*limb - intensity * intensity_ccf_blueshifted_shift[j-min_wave]*limb;
                        f_spot_bconv[j] += intensity_ccf_blueshifted_shift[j-min_wave]*limb - intensity_ccf_spot_shift[j-min_wave]*limb;
                        f_spot_tot[j]   += intensity_ccf_blueshifted_shift[j-min_wave]*limb - intensity * intensity_ccf_spot_shift[j-min_wave]*limb;
                    }
                    *sum_spot += intensity_ccf_blueshifted_shift[n_select-1]*limb*(1.-intensity); 

                    
                    
//                    n_v_shifted = delta/delta_v; // by how much steps the CCF is shifted due to rotation
//                    n_v_shifted_quotien = rndup(n_v_shifted,0); // integer number of steps
//                    n_v_shifted_rest = (delta - n_v_shifted_quotien*delta_v); // rest of the division between delta and the integer number of steps
//                        
//                    double fit[3] = {0,0,n_v_shifted/10.}; // voir fonction "shifting_spectrum", cela permet de decaer toute la CCF de n_v_shifted_rest se depth_from_continuum est tjs = 1.
//                    //shifting the CCF with the rest of n_v_shifted, the quotien will be taken into account by shifting all the points of the spectrum
//                    for (j=0;j<n;j++)
//                    {
//                        intensity_ccf_blueshifted_shift[j] = intensity_ccf_blueshifted[j];
//                        intensity_ccf_spot_shift[j] = intensity_ccf_spot[j];
//                    }
//                    for (j=0;j<10;j++)
//                    {
//                        shifting_spectrum(vrad_ccf, intensity_ccf_blueshifted_shift, intensity_ccf_blueshifted_shift, depth_from_continuum, fit, n, 0);
//                        shifting_spectrum(vrad_ccf, intensity_ccf_spot_shift, intensity_ccf_spot_shift, depth_from_continuum, fit, n, 0);
//                    }   
//                    
//                    for (j=0;j<n;j++)
//                    {
//                        f_spot_flux[j]  += intensity_ccf_blueshifted_shift[j]*limb - intensity * intensity_ccf_blueshifted_shift[j]*limb;
//                        f_spot_bconv[j] += intensity_ccf_blueshifted_shift[j]*limb - intensity_ccf_spot_shift[j]*limb;
//                        f_spot_tot[j]   += intensity_ccf_blueshifted_shift[j]*limb - intensity * intensity_ccf_spot_shift[j]*limb;
//                    }
//                    *sum_spot += intensity_ccf_blueshifted_shift[0]*limb*(1.-intensity);
                }
            }
        }
    }
    free(xayaza); free(xiyizi);
    free(intensity_ccf_blueshifted); free(intensity_ccf_blueshifted_shift);
    free(depth_from_continuum); free(intensity_ccf_spot_shift);
    free(vrad_ccf_select);free(vrad_ccf_shift);
    free(freq_same);free(flux_same);free(index_same);free(diff);
    free(freq_same_spot);free(flux_same_spot);free(index_same_spot);free(diff_spot);
}

void spot_inverse_rotation(double *xyz, double longitude, double latitude, 
			   double inclination, double phase, double *xiyizi)
{
  /* 
   * Relocate a point (x,y,z) to the 'initiale' configuration
   * i.e. when the star is equator on.
   *
   * Thus it consists of rotating the point, according to latitude, longitude,
   *  inclination and phase, but in the reverse order.
   */

  //
  // Conventions :
  // -when inclination=0 the star rotates around z axis (i.e. rotation axis and
  // z axis are indinstincted), 
  // -line of sight is along x-axis
  // -and sky plane = yz-plane
  //

  double g2 = --phase*(2*pi);  // inverse phase ([0-1] -> [rad])
  double i = inclination * pi/180.;

  double b  =  latitude  * pi/180.;
  double g  = -longitude * pi/180.;
  double b2 = -(pi/2.-i);

  double R[3][3]  = {{ (1-cos(g2))*cos(i)*cos(i) + cos(g2), 
  	                   sin(g2)*sin(i), 
  	                   (1-cos(g2))*cos(i)*sin(i)},
                     {-sin(g2)*sin(i),
                       cos(g2),        
                       sin(g2)*cos(i)},
                     { (1-cos(g2))*sin(i)*cos(i),
                      -sin(g2)*cos(i), 
                       (1-cos(g2))*sin(i)*sin(i) + cos(g2)}};
  double R2[3][3] =  {{cos(b)*cos(g)*cos(b2)-sin(b2)*sin(b), -sin(g)*cos(b), cos(b)*cos(g)*sin(b2)+sin(b)*cos(b2)},
                    {sin(g)*cos(b2),                          cos(g),          sin(g)*sin(b2)},
                    {-sin(b)*cos(g)*cos(b2)-cos(b)*sin(b2), sin(b)*sin(g), -sin(b)*cos(g)*sin(b2)+cos(b)*cos(b2)}};
  //double R2[3][3] = {{ (cos(b)*cos(g)*cos(b2) - sin(b)*sin(b2)), 
  //	                   cos(b)*sin(g), 
  //	                   -(cos(b)*cos(g)*sin(b2)+sin(b)*cos(b2))},
  //	                 {-sin(g)*cos(b2), 
  //	                   cos(g), 
  //	                   sin(g)*sin(b2)},
  //	                 {(sin(b)*cos(g)*cos(b2) + cos(b)*sin(b2)), 
  //	                   sin(b)*sin(g), 
  //	                  (-sin(b)*sin(g)*sin(b2)+cos(b)*cos(b2))}};
  double R3[3][3] = {{R2[0][0]*R[0][0]+R2[0][1]*R[1][0]+R2[0][2]*R[2][0], 
  	                  R2[0][0]*R[0][1]+R2[0][1]*R[1][1]+R2[0][2]*R[2][1],
  	                  R2[0][0]*R[0][2]+R2[0][1]*R[1][2]+R2[0][2]*R[2][2]},
  	                 {R2[1][0]*R[0][0]+R2[1][1]*R[1][0]+R2[1][2]*R[2][0], 
  	                  R2[1][0]*R[0][1]+R2[1][1]*R[1][1]+R2[1][2]*R[2][1],
  	                  R2[1][0]*R[0][2]+R2[1][1]*R[1][2]+R2[1][2]*R[2][2]},
  	                 {R2[2][0]*R[0][0]+R2[2][1]*R[1][0]+R2[2][2]*R[2][0], 
  	                  R2[2][0]*R[0][1]+R2[2][1]*R[1][1]+R2[2][2]*R[2][1],
  	                  R2[2][0]*R[0][2]+R2[2][1]*R[1][2]+R2[2][2]*R[2][2]}};

  xiyizi[0] = R3[0][0]*xyz[0] + R3[0][1]*xyz[1] + R3[0][2]*xyz[2];
  xiyizi[1] = R3[1][0]*xyz[0] + R3[1][1]*xyz[1] + R3[1][2]*xyz[2];
  xiyizi[2] = R3[2][0]*xyz[0] + R3[2][1]*xyz[1] + R3[2][2]*xyz[2];

}



void spotmap(double v, double i, double limba1, double limba2, int grid, double s, 
	      double longitude, double phase, double latitude, 
	      int iminy, int iminz, int imaxy, 
	      int imaxz, double **f_map, double **v_map, int magn_feature_type, int T_star, int T_diff_spot)
{
  /* Same as spot_scan except that it returns flux and velocity maps         */
  int iy, iz;
  double y, z;
  double delta_grid=2./grid, delta, r_cos;
  double limb;
  double *xayaza; // actual coordinates
  double *xiyizi; // coordinates transformed back to the initial configuration
  int T_spot,T_plage;
  double intensity,loi_Planck_star;
  xayaza = (double *)malloc(sizeof(double)*3);
  xiyizi = (double *)malloc(sizeof(double)*3);
    
  loi_Planck_star = loi_Planck(5293.4115e-10,T_star);  //the wavelength of the Kitt peak strum go from 3921.2441+6665.5789, the mean being 5293.4115

  // y-scan
  //for (iy=0, y=miny; y<=maxy; y+=delta_grid, iy++) {
  for (iy=0; iy<(imaxy-iminy); iy++) {
  	y = -1.+(iy+iminy)*delta_grid;
    delta = y * v * sin(i*pi/180.);
    xayaza[1] = y;

    // z-scan
    //for (iz=0, z=minz; z<=maxz; z+=delta_grid, iz++) {
    for (iz=0; iz<(imaxz-iminz); iz++) {
      z = -1.+(iz+iminz)*delta_grid; 
      if (z*z+y*y<1.) {
        xayaza[0] = pow(1.-(y*y+z*z),.5);
        xayaza[2] = z;

        // xayaza --> xiyizi
        spot_inverse_rotation(xayaza,longitude,latitude,i,phase,xiyizi); 
        
        if (xiyizi[0]*xiyizi[0]>=1.-s*s) 
        {
            //limb-darkening
            r_cos = pow(1.-(y*y+z*z),.5); //cos teta, cos teta = 1 au centre de l etoile donc teta = 0 au centre. cos teta = 0 sur les bords et donc cos teta = 90 sur les bords
            //limb = 1.-limba+limba*r_cos;  // linear limb-darkening law 
            limb =  1. - limba1*(1-r_cos) - limba2*(1-r_cos)*(1-r_cos); // voir SOAP-T paper, coming from Mandel & Agol 2002

            if (magn_feature_type==0)
            {
              T_spot = T_star-T_diff_spot;
              intensity = loi_Planck(5293.4115e-10,T_spot)/loi_Planck_star;  //the wavelength of the Kitt peak strum go from 3921.2441+6665.5789, the mean being 5293.4115
            }
            else 
            {
                //double mu = cos(latitude*2*3.1415926535897931/360.);
                T_plage = T_star+250.9-407.7*r_cos+190.9*pow(r_cos,2); //plages are brighter on the limb Meunier 2010
                intensity = loi_Planck(5293.4115e-10,T_plage)/loi_Planck_star; //the wavelength of the Kitt peak strum go from 3921.2441+6665.5789, the mean being 5293.4115
            }            
            f_map[iy][iz] = limb*(1.-intensity);
            v_map[iy][iz] = 1.;
        }
        else {
      	  f_map[iy][iz] = 0.;
      	  v_map[iy][iz] = 1.;
        }
      }
    }
  }
  free(xayaza); free(xiyizi);
	
}



void spot_scan_npsi(double **xyz, int nrho, double *psi, int npsi, double v, double v_pole,
                   double inclination, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, 
				   double *vrad_ccf, double *intensity_ccf, double *intensity_ccf_spot, double v_interval, int n_v, int n,
				   double *spots, double longitude, double latitude,
	      		   double **f_spot_flux, double **f_spot_bconv, double **f_spot_tot, double *sum_spot,
                   int magn_feature_type, int T_star, int T_diff_spot)
{
    /* 
    * Scans the yz-area where the spot is for different phases (psi) and        
    * returns the spot's "non-contribution" to the total flux and its           
    * "non-contribution" to the ccf, for each phase.                            
    *                                                                           
    * Thus the result is to be substracted to the output of itot() function.    
    */

    //tbd before:
    //spot_init(s, longitude, latitude, inclination, nrho, xyz) #out: xyz
    int ipsi, j;
    int iminy, iminz, imaxy, imaxz, vis;
    double **xyz2 = (double **)malloc(sizeof(double *)*nrho);
    for (j=0; j<nrho; j++) xyz2[j] = (double *)malloc(sizeof(double)*3);

    for (ipsi=0; ipsi<npsi; ipsi++) 
    {
        // A.S. spot size as a function of phase
        double s = spots[ipsi];
        
        // A.S. hackish implementation of differential rotation - lag spot longitudes each phase
        // v = ang. vel. at equator, v_pole = ang. vel. at pole
        // https://en.wikipedia.org/wiki/Differential_rotation#Calculating_differential_rotation
        double d2r = pi/180.;   // degree->radian conversion
        double dpsi = (ipsi == 0) ? psi[1]-psi[0] : psi[ipsi]-psi[ipsi-1];
        double dv = (v_pole - v)*sin(latitude*d2r)*sin(latitude*d2r); // dv between given lat & eq.
        longitude -= dv * (dpsi/v) / d2r;
        while(longitude >= 360) longitude -= 360;
        while(longitude < 0) longitude += 360;
        spot_init(s, longitude, latitude, inclination, nrho, xyz); //out: xyz
        
        //printf("PHASE = %d over %d\n",ipsi,npsi);
        spot_phase(xyz, inclination, nrho, psi[ipsi], xyz2);
        vis = spot_area(xyz2, nrho, grid, &iminy, &iminz, &imaxy, &imaxz);
        if (vis==1)
        {
            spot_scan(v, inclination, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, grid,
                      vrad_ccf, intensity_ccf, intensity_ccf_spot,v_interval, n_v, n,
                      s, longitude, psi[ipsi], latitude, iminy, iminz, imaxy, imaxz,
                      f_spot_flux[ipsi], f_spot_bconv[ipsi], f_spot_tot[ipsi], &sum_spot[ipsi],
                      magn_feature_type, T_star,T_diff_spot);
        }  
    }
    for (j=0; j<nrho; j++) free(xyz2[j]);
    free(xyz2);
}


void spot_scan_vr_phot(double v, double i, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, double s, 
	      double longitude, double phase, double latitude, 
	      double intensity, int iminy, int iminz, int imaxy, 
	      int imaxz, double *f_spot, double *sum_spot)
{
  /* Scan of the yz-area where the spot is.                                   */
  // For each grid-point (y,z) we need to check whether it belongs to the spot
  // or not. Sadly, we do not know the projected geometry of the spot in its
  // actual position. Thus, we have to do an inverse rotation to replace the
  // grid point where it would be in the initial configuration. Indeed, in the
  // initial configuration, the spot has a well known geometry of a circle 
  // centered on the x-axis.
  int iy, iz;
  double y, z;
  double delta_grid=2./grid, delta, r_cos;
  double limb;
  double *xayaza; // actual coordinates
  double *xiyizi; // coordinates transformed back to the initial configuration
  xayaza = (double *)malloc(sizeof(double)*3);
  xiyizi = (double *)malloc(sizeof(double)*3);

  // y-scan
  for (iy=iminy; iy<imaxy; iy++) {
    y = -1.+iy*delta_grid;
    delta = y * v * sin(i*pi/180.);
    xayaza[1] = y;

    // z-scan
    for (iz=iminz; iz<imaxz; iz++) {
      z = -1.+iz*delta_grid;
      if (z*z+y*y<1.) {
        xayaza[0] = pow(1.-(y*y+z*z),.5); //sqrt(abs(r*r-(y*y+z*z))); 
        xayaza[2] = z;

        // xayaza --> xiyizi
        spot_inverse_rotation(xayaza,longitude,latitude,i,phase,xiyizi); 

        if (xiyizi[0]*xiyizi[0]>=1.-s*s) {
          //limb-darkening
            r_cos = pow(1.-(y*y+z*z),.5); //cos teta, cos teta = 1 au centre de l etoile donc teta = 0 au centre. cos teta = 0 sur les bords et donc cos teta = 90 sur les bords
          //limb = 1.-limba+limba*r_cos;  // linear limb-darkening law 
          limb =  1. - limba1*(1-r_cos) - limba2*(1-r_cos)*(1-r_cos); // voir SOAP-T paper, coming from Mandel & Agol 2002
      
	  // velocity-scan (non-contribution, to be substracted)
	  *f_spot  += delta * (1.-intensity)*limb;
	
	  // flux "non-contribution" to the total flux
	  *sum_spot += (1.-intensity)*limb;
        }
      }
    }
  }
  free(xayaza); free(xiyizi);
}


void spot_scan_npsi_vr_phot(double **xyz, int nrho, double *psi, int npsi, double v, 
                   double inclination, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, 
		   double s, double longitude, double latitude, 
	      	   double intensity, double *f_spot, double *sum_spot)
{
  /* 
   * Scans the yz-area where the spot is for different phases (psi) and        
   * returns the spot's "non-contribution" to the total flux and its           
   * "non-contribution" to the ccf, for each phase.                            
   *                                                                           
   * Thus the result is to be substracted to the output of itot() function.    
   */

  //tbd before:
  //spot_init(s, longitude, latitude, inclination, nrho, xyz) #out: xyz
  int ipsi, j;
  int iminy, iminz, imaxy, imaxz, vis;
  double **xyz2 = (double **)malloc(sizeof(double *)*nrho);
  for (j=0; j<nrho; j++) xyz2[j] = (double *)malloc(sizeof(double)*3);

  for (ipsi=0; ipsi<npsi; ipsi++) {
    spot_phase(xyz, inclination, nrho, psi[ipsi], xyz2);
    vis = spot_area(xyz2, nrho, grid, &iminy, &iminz, &imaxy, &imaxz);
    if (vis==1)
      spot_scan_vr_phot(v, inclination, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, grid, s, 
  		     	longitude, psi[ipsi], latitude, intensity, iminy, iminz, imaxy, 
  			    imaxz, &f_spot[ipsi], &sum_spot[ipsi]);
  }

  for (j=0; j<nrho; j++) free(xyz2[j]);
  free(xyz2);
}


