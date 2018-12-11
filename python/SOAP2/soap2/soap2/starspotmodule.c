/******************************************************************************/
/*** file:   starspotmodule.c                                               ***/
/*** author: X. Bonfils                                                     ***/
/***         Centro de Astronomia e Astrofisica da Universidade de Lisboa   ***/
/***                                                                        ***/
/*** version: 0.1 2006/08/29                                                ***/
/******************************************************************************/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "starspot.h"

static PyObject *phasm_itot(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *f_star2,*a, *b;
  double v, i, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, v_interval;
  int grid, n_v, n, j, dim[1];
  double *f_star,*vrad_ccf, *intensity_ccf, sum_star;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "dddddddiO!O!dii", &v, &i, &limba1, &limba2, &modif_bis_quad, &modif_bis_lin, &modif_bis_cte, &grid,
			&PyArray_Type, &a, &PyArray_Type, &b, &v_interval, &n_v, &n)) 
    return NULL;
    
  /* Allocations */
//  f_star2 = (PyArrayObject *) PyArray_FromDims(1, &n, PyArray_DOUBLE);
//  f_star  = (double *)f_star2->data;
  vrad_ccf = (double *) (a->data + 0*a->strides[0]);
  intensity_ccf = (double *) (b->data + 0*b->strides[0]);
  
  // Creation du vecteur f_star et association de la memoire
  f_star = (double *) malloc(n*sizeof(double));
    
  /* Initialisations */
  for (j=0; j<n; j++) f_star[j] = 0;
  sum_star = 0;

  /* Total intensity */
  itot(v,i,limba1,limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte,grid,vrad_ccf,intensity_ccf,v_interval,n_v,n,f_star,&sum_star);

  dim[0] = n;
  f_star2 = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  for (j = 0; j < n; j++) *(double *) (f_star2->data + j*f_star2->strides[0]) = f_star[j];
    
  free(f_star);
    
  return Py_BuildValue("Nd", f_star2, sum_star);
}

static PyObject *phasm_starmap(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *f_map2, *v_map2;
  double v, i, limba1, limba2;
  int grid, j, k, dimensions[2];
  double **Fmap, **Vmap;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "ddddi", &v, &i, &limba1, &limba2, &grid)) 
    return NULL;

  /* Allocations */
  Fmap = (double **)malloc(sizeof(double *)*grid);
  Vmap = (double **)malloc(sizeof(double *)*grid);
  for (j=0; j<grid; j++) Fmap[j] = (double *)malloc(sizeof(double)*grid);
  for (j=0; j<grid; j++) Vmap[j] = (double *)malloc(sizeof(double)*grid);
  
  /* Initialisations */
  for (j=0; j<grid; j++) for (k=0; k<grid; k++) Fmap[j][k] = 0.;
  for (j=0; j<grid; j++) for (k=0; k<grid; k++) Vmap[j][k] = 0.;

  /* Star Maps (Intensity & Velocity Maps) */
  starmap(v, i, limba1, limba2, grid, Fmap, Vmap);

  /* Creation of the returned python object (Numeric array)*/
  dimensions[0] = grid; dimensions[1] = grid;
	
  f_map2 = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
  v_map2 = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);

  for (j=0; j<dimensions[0]; j++) for (k=0; k<dimensions[1]; k++)
    *(double *)(f_map2->data + j*f_map2->strides[0] + k*f_map2->strides[1]) = Fmap[j][k];
  for (j=0; j<dimensions[0]; j++) for (k=0; k<dimensions[1]; k++)
    *(double *)(v_map2->data + j*v_map2->strides[0] + k*v_map2->strides[1]) = Vmap[j][k];

  for (j=0; j<grid; j++) free(Fmap[j]); free(Fmap);
  for (j=0; j<grid; j++) free(Vmap[j]); free(Vmap);

  return Py_BuildValue("NN", f_map2, v_map2);
}

static PyObject *phasm_spot_init(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *xyz2;
  double s, longitude, latitude, inclination;
  int nrho, i, j;
  double **xyz;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "ddddi", &s, &longitude, &latitude, &inclination, 
			&nrho))
    return NULL;
 
  /* Allocations & Init.*/
  xyz    = (double **)malloc(sizeof(double *)*nrho);
  for (j=0; j<nrho; j++) xyz[j] = (double *)malloc(sizeof(double)*3);

  /* Init Spot */
  spot_init(s, longitude, latitude, inclination, nrho, xyz);

  /* Creation of the returned python object (Numeric array)                    */
  int dimensions[2]={nrho,3};
  xyz2 = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
  for (i=0; i<dimensions[0]; i++) for (j=0; j<dimensions[1]; j++)
    *(double *)(xyz2->data + i*xyz2->strides[0] + j*xyz2->strides[1]) = xyz[i][j];

  for (j=0; j<nrho; j++) free(xyz[j]); free(xyz);  
  
  return Py_BuildValue("N", xyz2);
}

static PyObject *phasm_spot_phase(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *xyz_in, *xyz_out;
  double inclination, phase;
  int nrho, i, j;
  double **xyz, **xyz2;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "O!did", &PyArray_Type, &xyz_in, &inclination, 
			&nrho, &phase))
    return NULL;
 
  /* Allocations & Init.*/
  xyz    = (double **)malloc(sizeof(double *)*nrho);
  xyz2   = (double **)malloc(sizeof(double *)*nrho);
  for (j=0; j<nrho; j++) xyz[j]  = (double *)malloc(sizeof(double)*3);
  for (j=0; j<nrho; j++) xyz2[j] = (double *)malloc(sizeof(double)*3);
  for (i=0; i<nrho; i++) for (j=0; j<3; j++) 
    xyz[i][j] = *(double *)(xyz_in->data 
			    + i*xyz_in->strides[0] + j*xyz_in->strides[1]);

  /* Init Spot */
  spot_phase(xyz, inclination, nrho, phase, xyz2);

  /* Creation of the returned python object (Numeric array)                    */
  int dimensions[2]={nrho,3};
  xyz_out = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
  for (i=0; i<dimensions[0]; i++) for (j=0; j<dimensions[1]; j++)
    *(double *)(xyz_out->data + i*xyz_out->strides[0] + j*xyz_out->strides[1]) = xyz2[i][j];
    
  for (j=0; j<nrho; j++) free(xyz[j]); free(xyz);
  for (j=0; j<nrho; j++) free(xyz2[j]); free(xyz2);
  return Py_BuildValue("N", xyz_out);
}

static PyObject *phasm_spot_area(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *xlylzl_in;
  int nrho, i, j, vis, grid;
  int iminy, iminz, imaxy, imaxz;
  double **xlylzl;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &xlylzl_in, &nrho, &grid))
    return NULL;
 
  /* Allocations & Init.*/
  xlylzl    = (double **)malloc(sizeof(double *)*nrho);
  for (j=0; j<nrho; j++) xlylzl[j]  = (double *)malloc(sizeof(double)*3);
  for (i=0; i<nrho; i++) for (j=0; j<3; j++) 
    xlylzl[i][j] = *(double *)(xlylzl_in->data 
			    + i*xlylzl_in->strides[0] + j*xlylzl_in->strides[1]);

  /* Spot Area */
  vis = spot_area(xlylzl, nrho, grid, &iminy, &iminz, &imaxy, &imaxz);

  for (j=0; j<nrho; j++) free(xlylzl[j]); free(xlylzl);
  return Py_BuildValue("iiiii", vis, iminy, iminz, imaxy, imaxz);
}

static PyObject *phasm_spot_scan(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *a, *b, *c, *f_spot2, *f_spot3, *f_spot4;
  double v, i, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, v_interval;
  int grid, n_v, n, j;
  double s, longitude, phase, latitude;
  int iminy, iminz, imaxy, imaxz, magn_feature_type,T_star,T_diff_spot; 
  double *vrad_ccf, *intensity_ccf, *intensity_ccf_spot, *f_spot_flux,*f_spot_bconv,*f_spot_tot, sum_spot;

  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "dddddddiO!O!O!diiddddiiiiiii", &v, &i, &limba1, &limba2, &modif_bis_quad, &modif_bis_lin, &modif_bis_cte, &grid, 
      &PyArray_Type, &a,&PyArray_Type, &b,&PyArray_Type, &c, &v_interval, &n_v, &n, &s, &longitude, 
      &phase, &latitude, &iminy, &iminz, &imaxy, &imaxz,
      &magn_feature_type, &T_star, &T_diff_spot))
    return NULL;
 
  /* Allocations */
  vrad_ccf = (double *) (a->data + 0*a->strides[0]);
  intensity_ccf = (double *) (b->data + 0*b->strides[0]);
  intensity_ccf_spot = (double *) (c->data + 0*c->strides[0]);
  f_spot2 = (PyArrayObject *) PyArray_FromDims(1, &n, PyArray_DOUBLE);
  f_spot_flux = (double *)f_spot2->data;
  f_spot3 = (PyArrayObject *) PyArray_FromDims(1, &n, PyArray_DOUBLE);
  f_spot_bconv = (double *)f_spot3->data;
  f_spot4 = (PyArrayObject *) PyArray_FromDims(1, &n, PyArray_DOUBLE);
  f_spot_tot = (double *)f_spot4->data;

  /* Initialisations */
  for (j=0; j<n; j++) {
      f_spot_flux[j] = 0;
      f_spot_bconv[j] = 0;
      f_spot_tot[j] = 0;
  }
  sum_spot = 0;

  /* Scan Spot */
  spot_scan(v, i, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, grid, vrad_ccf, intensity_ccf, intensity_ccf_spot, v_interval, n_v, n, s, longitude, 
            phase, latitude, iminy, iminz, imaxy, imaxz, f_spot_flux, f_spot_bconv,f_spot_tot, 
            &sum_spot,magn_feature_type,T_star,T_diff_spot);

  /*return Py_BuildValue("Nd", f_spot2, sum_spot);*/
  return Py_BuildValue("NNNd", f_spot2, f_spot3,f_spot4, sum_spot);
}

static PyObject *phasm_spot_inverse_rotation(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *xyz_in, *xiyizi2;
  double longitude, latitude, inclination, phase;
  int i, dimensions[1];
  double *xyz, *xiyizi;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "O!dddd", &PyArray_Type, &xyz_in, &longitude, 
      &latitude, &inclination, &phase))
    return NULL;
 
  /* Allocations & Init.*/
  dimensions[0] = 3;
  xiyizi2 = (PyArrayObject *) PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
  xiyizi  = (double *)xiyizi2->data;
  xyz    = (double *)malloc(sizeof(double *)*3);
  for (i=0; i<3; i++) xyz[i] = *(double *)(xyz_in->data + i*xyz_in->strides[0]);
  for (i=0; i<3; i++) xiyizi[i] = 0.;
   
  /* Init Spot */
  spot_inverse_rotation(xyz, longitude, latitude, inclination, phase, xiyizi);

  free(xyz);
  return Py_BuildValue("N", xiyizi2);
}

static PyObject *phasm_spotmap(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *f_map2, *v_map2;
  double v, i, limba1, limba2, s, longitude, phase, latitude;
  int iminy, iminz, imaxy, imaxz, magn_feature_type, T_star, T_diff_spot;
  int grid, j, k, dimensions[2], M, N;
  double **Fmap, **Vmap;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "ddddiddddiiiiiii", &v, &i, &limba1, &limba2, &grid, &s, 
      &longitude, &phase, &latitude, &iminy, &iminz, &imaxy, &imaxz, &magn_feature_type, &T_star, &T_diff_spot))
    return NULL;

  /* Allocations */
  M = imaxy-iminy;
  N = imaxz-iminz;
  Fmap = (double **)malloc(sizeof(double *)*M);
  Vmap = (double **)malloc(sizeof(double *)*M);
  for (j=0; j<M; j++) Fmap[j] = (double *)malloc(sizeof(double)*N);
  for (j=0; j<M; j++) Vmap[j] = (double *)malloc(sizeof(double)*N);
  
  /* Initialisations */
  for (j=0; j<M; j++) for (k=0; k<N; k++) Fmap[j][k] = 0.;
  for (j=0; j<M; j++) for (k=0; k<N; k++) Vmap[j][k] = 0.;
  //printf("(M,N): %i %i\n", M, N);
  
  /* Star Maps (Intensity & Velocity Maps) */
  spotmap(v, i, limba1, limba2, grid, s, longitude, phase, latitude, 
        iminy, iminz, imaxy, imaxz, Fmap, Vmap, magn_feature_type, T_star, T_diff_spot);

  /* Creation of the returned python object (Numeric array)*/
  dimensions[0] = M; dimensions[1] = N;
  f_map2 = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
  v_map2 = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
  for (j=0; j<dimensions[0]; j++) for (k=0; k<dimensions[1]; k++)
    *(double *)(f_map2->data + j*f_map2->strides[0] + k*f_map2->strides[1]) = Fmap[j][k];
  for (j=0; j<dimensions[0]; j++) for (k=0; k<dimensions[1]; k++)
    *(double *)(v_map2->data + j*v_map2->strides[0] + k*v_map2->strides[1]) = Vmap[j][k];

  for (j=0; j<M; j++) free(Fmap[j]); free(Fmap); 
  for (j=0; j<M; j++) free(Vmap[j]); free(Vmap);
  return Py_BuildValue("NN", f_map2, v_map2);
}

static PyObject *phasm_spot_scan_npsi(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *a,*b,*c,*spots_in,*xyz_in, *psi_in, *f_spot2, *f_spot3, *f_spot4, *sum_spot2;
  int i,j;   int dimensions[2];
  int nrho, npsi, grid, n_v, n, magn_feature_type,T_star,T_diff_spot;
  double v, v_pole, inclination, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte;
  double longitude, latitude, *vrad_ccf, *intensity_ccf, *intensity_ccf_spot, v_interval;
  double **xyz, *psi, *spots, **f_spot_flux, **f_spot_bconv, **f_spot_tot, *sum_spot;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "O!iO!iddddddddiO!O!O!diiO!ddiii", &PyArray_Type, &xyz_in,
        &nrho, &PyArray_Type, &psi_in, &npsi, &v, &v_pole, &inclination, &limba1, &limba2, &modif_bis_quad, &modif_bis_lin, &modif_bis_cte, &grid,
		&PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &v_interval, &n_v, &n, &PyArray_Type, &spots_in, &longitude, &latitude,
	    &magn_feature_type,&T_star,&T_diff_spot))
    return NULL;
     
  /* Allocations */
  vrad_ccf = (double *) (a->data + 0*a->strides[0]);
  intensity_ccf = (double *) (b->data + 0*b->strides[0]);
  intensity_ccf_spot = (double *) (c->data + 0*c->strides[0]);
    
  sum_spot2 = (PyArrayObject *) PyArray_FromDims(1, &npsi, PyArray_DOUBLE);
  sum_spot  = (double *)sum_spot2->data;
  f_spot_flux    = (double **)malloc(sizeof(double *)*npsi);
  f_spot_bconv    = (double **)malloc(sizeof(double *)*npsi);
  f_spot_tot    = (double **)malloc(sizeof(double *)*npsi);
  for (j=0; j<npsi; j++) f_spot_flux[j] = (double *)malloc(sizeof(double)*n);
  for (j=0; j<npsi; j++) f_spot_bconv[j] = (double *)malloc(sizeof(double)*n);
  for (j=0; j<npsi; j++) f_spot_tot[j] = (double *)malloc(sizeof(double)*n);
  xyz    = (double **)malloc(sizeof(double *)*nrho);
  for (j=0; j<nrho; j++) xyz[j]  = (double *)malloc(sizeof(double)*3);
  for (i=0; i<nrho; i++) for (j=0; j<3; j++) 
    xyz[i][j] = *(double *)(xyz_in->data 
			    + i*xyz_in->strides[0] + j*xyz_in->strides[1]);
  psi = (double *)malloc(sizeof(double)*npsi);
  for (i=0; i<npsi; i++) psi[i] = *(double *)(psi_in->data + i*psi_in->strides[0]);

  //A.S.
  spots = (double *) (spots_in->data + 0*spots_in->strides[0]);
  //spots = (double *)malloc(sizeof(double)*npsi);
  //for (i=0; i<npsi; i++) spots[i] = *(double *)(spots_in->data + i*spots_in->strides[0]);


  /* Initialisations */
  for (i=0; i<npsi; i++) for (j=0; j<n; j++) f_spot_flux[i][j] = 0.;
  for (i=0; i<npsi; i++) for (j=0; j<n; j++) f_spot_bconv[i][j] = 0.;
  for (i=0; i<npsi; i++) for (j=0; j<n; j++) f_spot_tot[i][j] = 0.;
  for (i=0; i<npsi; i++) sum_spot[i] = 0.;

  /* Scan Spot */
  spot_scan_npsi(xyz, nrho, psi, npsi, v, v_pole, inclination, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, grid,
				 vrad_ccf, intensity_ccf, intensity_ccf_spot, v_interval, n_v, n, spots, longitude, latitude,
	      		 f_spot_flux, f_spot_bconv, f_spot_tot, sum_spot, magn_feature_type,T_star,T_diff_spot);

  /* Creation of the returned python object (Numeric array)*/
  dimensions[0] = npsi; dimensions[1] = n;
  f_spot2 = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
  f_spot3 = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
  f_spot4 = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
  for (j=0; j<dimensions[0]; j++) for (i=0; i<dimensions[1]; i++)
  {
    *(double *)(f_spot2->data + j*f_spot2->strides[0] + i*f_spot2->strides[1]) = f_spot_flux[j][i];
    *(double *)(f_spot3->data + j*f_spot3->strides[0] + i*f_spot3->strides[1]) = f_spot_bconv[j][i];
    *(double *)(f_spot4->data + j*f_spot4->strides[0] + i*f_spot4->strides[1]) = f_spot_tot[j][i];
  }
  for (j=0; j<npsi; j++) free(f_spot_flux[j]);
  for (j=0; j<npsi; j++) free(f_spot_bconv[j]);
  for (j=0; j<npsi; j++) free(f_spot_tot[j]);
  for (j=0; j<nrho; j++) free(xyz[j]);
  free(f_spot_flux); free(f_spot_bconv); free(f_spot_tot); free(xyz); free(psi);
  
  /*return Py_BuildValue("NN", f_spot2, sum_spot2);*/
  return Py_BuildValue("NNNN", f_spot2, f_spot3, f_spot4, sum_spot2);
}

static PyObject *phasm_spot_scan_vr_phot(PyObject *self, PyObject *args)
{
  /* Declarations */
  double v, i, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte;
  int grid;
  double s, longitude, phase, latitude, intensity;
  int iminy, iminz, imaxy, imaxz; 
  double f_spot, sum_spot;

  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "dddddddidddddiiii", &v, &i, &limba1, &limba2, &modif_bis_quad, &modif_bis_lin, &modif_bis_cte, &grid, 
      &s, &longitude, &phase, &latitude, &intensity, &iminy, &iminz, &imaxy, &imaxz))
    return NULL;
 
  /* Initialisations */
  f_spot = 0;
  sum_spot = 0;

  /* Scan Spot */
  spot_scan_vr_phot(v, i, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, grid, s, longitude, phase, latitude, intensity, 
  	    iminy, iminz, imaxy, imaxz, &f_spot, &sum_spot);

  return Py_BuildValue("dd", f_spot, sum_spot);
}

static PyObject *phasm_spot_scan_npsi_vr_phot(PyObject *self, PyObject *args)
{
  /* Declarations */
  PyArrayObject *xyz_in, *psi_in, *f_spot2, *sum_spot2;
  int i,j;
  int nrho, npsi, grid;
  double v, inclination, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, s, longitude;
  double latitude, intensity;
  double **xyz, *psi, *f_spot, *sum_spot;
  
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "O!iO!iddddddddddidddd", &PyArray_Type, &xyz_in, 
        &nrho, &PyArray_Type, &psi_in, &npsi, &v, &inclination, &limba1, &limba2, &modif_bis_quad, &modif_bis_lin, &modif_bis_cte, &grid,
		&s, &longitude, &latitude, &intensity))
    return NULL;
 
  /* Allocations */
  f_spot2   = (PyArrayObject *) PyArray_FromDims(1, &npsi, PyArray_DOUBLE);
  sum_spot2 = (PyArrayObject *) PyArray_FromDims(1, &npsi, PyArray_DOUBLE);
  f_spot   = (double *)f_spot2->data;
  sum_spot = (double *)sum_spot2->data;
  xyz      = (double **)malloc(sizeof(double *)*nrho);
  for (j=0; j<nrho; j++) xyz[j]  = (double *)malloc(sizeof(double)*3);
  for (i=0; i<nrho; i++) for (j=0; j<3; j++) 
    xyz[i][j] = *(double *)(xyz_in->data 
			    + i*xyz_in->strides[0] + j*xyz_in->strides[1]);
  psi = (double *)malloc(sizeof(double)*npsi);
  for (i=0; i<npsi; i++) psi[i] = *(double *)(psi_in->data + i*psi_in->strides[0]);

  /* Initialisations */
  for (i=0; i<npsi; i++) f_spot[i] = 0;
  for (i=0; i<npsi; i++) sum_spot[i] = 0;

  /* Scan Spot */
  spot_scan_npsi_vr_phot(xyz, nrho, psi, npsi, v, inclination, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, grid, 
		 s, longitude, latitude, intensity, f_spot, sum_spot);

  for (j=0; j<nrho; j++) free(xyz[j]); free(xyz); 
  free(psi);
  return Py_BuildValue("NN", f_spot2, sum_spot2);
}

static PyMethodDef PhasmMethods[] = {
    {"itot", phasm_itot, METH_VARARGS, 
     "Calculation of total intensity with no spot"},
    {"starmap", phasm_starmap, METH_VARARGS, 
     "Returns intensity & velocity maps of the star"},
    {"spot_init", phasm_spot_init, METH_VARARGS, "Init spot"},
    {"spot_phase", phasm_spot_phase, METH_VARARGS, "Phase spot"},
    {"spot_area", phasm_spot_area, METH_VARARGS, 
     "Determine a smaller area to locate the spot"},
    {"spot_scan", phasm_spot_scan, METH_VARARGS, 
     "Scan spot non-contribution to the flux and velocity field of the star"},
    {"spot_inverse_rotation", phasm_spot_inverse_rotation, METH_VARARGS, 
     "Inverse rotaion of xyz to check whether it belongs to the spot or not"},
    {"spotmap", phasm_spotmap, METH_VARARGS, 
     "Returns intensity & velocity maps of the area occupied by the spot"},
    {"spot_scan_npsi", phasm_spot_scan_npsi, METH_VARARGS, 
     "Same as spot_scan for an array of phases psi"},
    {"spot_scan_vr_phot", phasm_spot_scan_vr_phot, METH_VARARGS, 
     "Scan spot non-contribution to the flux and velocity field of the star"},
    {"spot_scan_npsi_vr_phot", phasm_spot_scan_npsi_vr_phot, METH_VARARGS, 
     "Same as spot_scan for an array of phases psi"},
    {NULL, NULL, 0, NULL}
};

void init_starspot(void){
    (void) Py_InitModule("_starspot", PhasmMethods);
    import_array();
}

	  
//int main(int argc, char **argv){
//    Py_SetProgramName(argv[0]);
//    Py_Initialize();
//    initphasm();
//}
