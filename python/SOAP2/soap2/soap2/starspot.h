/******************************************************************************/
/*** file:   starspot.h                                                     ***/
/*** author: X. Bonfils                                                     ***/
/***         Centro de Astronomia e Astrofisica da Universidade de Lisboa   ***/
/***                                                                        ***/
/*** version: 0.1 2006/08/29                                                ***/
/******************************************************************************/

#ifndef __PHASM_H__
#define __PHASM_H__

#include <stdlib.h>
#include <math.h>

#define DEBUG   0
#define VERBOSE 0
#define pi      3.14159265358

extern 
void itot(double v, double i, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, 
          double *vrad_ccf, double *intensity_ccf, double v_interval, int n_v, int n, 
          double *f_star, double *sum_star);

extern
void starmap(double v, double i, double limba1, double limba2, int grid,
	  double **Fmap, double **Vmap);

extern 
void spot_init(double s, double longitude, double latitude, double inclination, 
	       int nrho, double **xyz);

extern 
void spot_phase(double **xyz, double inclination, int nrho, double psi, 
		double **xyz2);
		
extern
int spot_area(double **xlylzl, int nrho, int grid, int *iminy, int *iminz, 
	       int *imaxy, int *imaxz);
	       
extern
void spot_scan(double v, double i, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, 
	      double *vrad_ccf, double *intensity_ccf, double *intensity_ccf_spot, double v_interval, int n_v, int n, 
          double s, double longitude, double phase, double latitude, 
	      int iminy, int iminz, int imaxy, 
	      int imaxz, double *f_spot_flux, double *f_spot_bconv, double *f_spot_tot, double *sum_spot,
          int magn_feature_type, int T_star, int T_diff_spot);
	      
extern
void spot_inverse_rotation(double *xyz, double longitude, double latitude, 
			   double inclination, double phase, double *xiyizi);

extern
void spotmap(double v, double i, double limba1, double limba2, int grid, double s, 
	      double longitude, double phase, double latitude, 
	      int iminy, int iminz, int imaxy, 
	      int imaxz, double **f_map, double **v_map, int magn_feature_type, int T_star, int T_diff_spot);

extern
void spot_scan_npsi(double **xyz, int nrho, double *psi, int npsi, double v, double v_pole,
                   double inclination, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, 
				   double *vrad_ccf, double *intensity_ccf, double *intensity_ccf_spot, double v_interval, int n_v, int n,
				   double *spots, double longitude, double latitude,
	      		   double **f_spot_flux, double **f_spot_bconv, double **f_spot_tot, double *sum_spot,
                   int magn_feature_type, int T_star, int T_diff_spot);

extern
void spot_scan_vr_phot(double v, double i, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, double s, 
	      double longitude, double phase, double latitude, 
	      double intensity, int iminy, int iminz, int imaxy, 
	      int imaxz, double *f_spot, double *sum_spot);
extern
void spot_scan_npsi_vr_phot(double **xyz, int nrho, double *psi, int npsi, double v, 
                   double inclination, double limba1, double limba2, double modif_bis_quad, double modif_bis_lin, double modif_bis_cte, int grid, 
		   double s, double longitude, double latitude, 
	      	   double intensity, double *f_spot, double *sum_spot);
#endif /*__PHASM_H__*/
