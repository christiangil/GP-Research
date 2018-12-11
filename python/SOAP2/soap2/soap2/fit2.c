#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlin.h>

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

struct gauss_params
	{
		double *x, *y, *err, *fit;
		int n, m;
	};



int gauss_f (const gsl_vector *v, void *p, gsl_vector *f)
	{
		double c = gsl_vector_get(v, 0);
		double k = gsl_vector_get(v, 1);
		double x0 = gsl_vector_get(v, 2);
		double fwhm = gsl_vector_get(v, 3);
		
		double *x = ((struct gauss_params *) p)->x;
		double *y = ((struct gauss_params *) p)->y;
		double *err = ((struct gauss_params *) p)->err;
		double *fit = ((struct gauss_params *) p)->fit;
		int n = ((struct gauss_params *) p)->n;
		
		int i;
		double sigma = fwhm/2/sqrt(2*log(2));
		
		for (i = 0; i < n; i++) {
			fit[i] = c + k * exp(-(x[i]-x0)*(x[i]-x0)/2/sigma/sigma);
			gsl_vector_set (f, i, (fit[i]-y[i])/err[i]);
		}
		
		return GSL_SUCCESS;
	}



int gauss_df (const gsl_vector *v, void *p, gsl_matrix *J)
	{
		double c = gsl_vector_get(v, 0);
		double k = gsl_vector_get(v, 1);
		double x0 = gsl_vector_get(v, 2);
		double fwhm = gsl_vector_get(v, 3);
		
		double *x = ((struct gauss_params *) p)->x;
		double *y = ((struct gauss_params *) p)->y;
		double *err = ((struct gauss_params *) p)->err;
		int n = ((struct gauss_params *) p)->n;
		
		int i;
		double sigma = fwhm/2/sqrt(2*log(2));
		
		for (i = 0; i < n; i++) {
			double e = exp(-(x[i]-x0)*(x[i]-x0)/2/sigma/sigma);
			gsl_matrix_set (J, i, 0, 1/err[i]);
			gsl_matrix_set (J, i, 1, e/err[i]);
			gsl_matrix_set (J, i, 2, k/err[i]*e*(x[i]-x0)/sigma/sigma);
			gsl_matrix_set (J, i, 3, k/err[i]*e*(x[i]-x0)*(x[i]-x0)/sigma/sigma/sigma/2/sqrt(2*log(2)));
		}
		
		return GSL_SUCCESS;
	}



int gauss_fdf (const gsl_vector *v, void *p, gsl_vector *f, gsl_matrix *J)
	{
		gauss_f (v, p, f);
		gauss_df (v, p, J);
		
		return GSL_SUCCESS;
	}



static PyObject *gauss (PyObject *self, PyObject *args)
	{
		PyArrayObject *aa, *bb, *cc;
		PyArrayObject *dd;
		
		struct gauss_params p;
		
		double *x;
		int i, j = 0, status, npar = 4;
		double c, k, x0, fwhm;
		double sig_c, sig_k, sig_x0, sig_fwhm;
		
		gsl_vector *v = gsl_vector_alloc (npar);
		gsl_matrix *covar = gsl_matrix_alloc (npar, npar);
		const gsl_multifit_fdfsolver_type *T;
		gsl_multifit_fdfsolver *s;
		gsl_multifit_function_fdf F;
		
		if (!PyArg_ParseTuple(args, "O!O!O!", 
			&PyArray_Type, &aa, &PyArray_Type, &bb, &PyArray_Type, &cc))
			return NULL;
		
		x = (double *) (aa->data + 0*aa->strides[0]);
		p.y = (double *) (bb->data + 0*bb->strides[0]);
		p.err = (double *) (cc->data + 0*cc->strides[0]);
		p.n = aa->dimensions[0];
		
		p.x = (double *) malloc(p.n*sizeof(double));
		p.fit = (double *) malloc(p.n*sizeof(double));
		
		for (i = 0; i < p.n; i++)
			p.x[i] = x[i]-x[0];
		
		c = (p.y[0]+p.y[p.n-1])/2;
		fwhm = (p.x[p.n-1]-p.x[0])/5.;
		k = 0.; x0 = p.x[p.n/2];
		
		for (i = 0; i < p.n; i++)
			if (abs(p.y[i]-c) > abs(k)) { k = p.y[i]-c; x0 = p.x[i]; }
		
		F.f = &gauss_f;
		F.df = &gauss_df;
		F.fdf = &gauss_fdf;
		F.n = p.n;
		F.p = npar;
		F.params = &p;
		
		T = gsl_multifit_fdfsolver_lmsder;
		s = gsl_multifit_fdfsolver_alloc (T, p.n, npar);
		gsl_vector_set(v, 0, c);
		gsl_vector_set(v, 1, k);
		gsl_vector_set(v, 2, x0);
		gsl_vector_set(v, 3, fwhm);
		gsl_multifit_fdfsolver_set (s, &F, v);
		
		do {
			j++;
			status = gsl_multifit_fdfsolver_iterate (s);
			status = gsl_multifit_test_delta(s->dx, s->x, 0.0, 1e-5);
		}
		
		while (status == GSL_CONTINUE && j < 10000);
		
		gsl_multifit_covar (s->J, 0.0, covar);
		
		c = gsl_vector_get(s->x, 0); sig_c = sqrt(gsl_matrix_get(covar,0,0));
		k = gsl_vector_get(s->x, 1); sig_k = sqrt(gsl_matrix_get(covar,1,1));
		x0 = gsl_vector_get(s->x, 2); sig_x0 = sqrt(gsl_matrix_get(covar,2,2));
		fwhm = gsl_vector_get(s->x, 3); sig_fwhm = sqrt(gsl_matrix_get(covar,3,3));
		
		dd = (PyArrayObject *) 
			PyArray_SimpleNew(aa->nd, aa->dimensions, PyArray_DOUBLE);
		
		for (i = 0; i < p.n; i++)
			*(double *) (dd->data + i*dd->strides[0]) = p.fit[i];
		
		return Py_BuildValue("Nddddddddi", dd, c, k, x0+x[0], fwhm, sig_c, sig_k, sig_x0, sig_fwhm, j);
	}



static PyObject *gauss_bis (PyObject *self, PyObject *args)
	{
		PyArrayObject *aa, *bb, *cc, *zz;
		PyArrayObject *dd,*ee,*ff;
		
		struct gauss_params p;
		
		double *x;
		int i, j = 0, status, npar = 4;
		double c, k, x0, fwhm;
		double sig_c, sig_k, sig_x0, sig_fwhm;
		
		gsl_vector *v = gsl_vector_alloc (npar);
		gsl_matrix *covar = gsl_matrix_alloc (npar, npar);
		const gsl_multifit_fdfsolver_type *T;
		gsl_multifit_fdfsolver *s;
		gsl_multifit_function_fdf F;
		
		if (!PyArg_ParseTuple(args, "O!O!O!O!", 
			&PyArray_Type, &aa, &PyArray_Type, &bb, &PyArray_Type, &cc, &PyArray_Type, &zz)) //le vecteur zz permet juste de donner la taille pour le retour du vecteur bissecteur et depth
			return NULL;
		
		x = (double *) (aa->data + 0*aa->strides[0]);
		p.y = (double *) (bb->data + 0*bb->strides[0]);
		p.err = (double *) (cc->data + 0*cc->strides[0]);
		p.n = aa->dimensions[0];
		
		p.x = (double *) malloc(p.n*sizeof(double));
		p.fit = (double *) malloc(p.n*sizeof(double));
		
		for (i = 0; i < p.n; i++)
			p.x[i] = x[i]-x[0];
		
		c = (p.y[0]+p.y[p.n-1])/2;
		fwhm = (p.x[p.n-1]-p.x[0])/5.;
		k = 0.; x0 = p.x[p.n/2];
		
		for (i = 0; i < p.n; i++)
			if (abs(p.y[i]-c) > abs(k)) { k = p.y[i]-c; x0 = p.x[i]; }
		
		F.f = &gauss_f;
		F.df = &gauss_df;
		F.fdf = &gauss_fdf;
		F.n = p.n;
		F.p = npar;
		F.params = &p;
		
		T = gsl_multifit_fdfsolver_lmsder;
		s = gsl_multifit_fdfsolver_alloc (T, p.n, npar);
		gsl_vector_set(v, 0, c);
		gsl_vector_set(v, 1, k);
		gsl_vector_set(v, 2, x0);
		gsl_vector_set(v, 3, fwhm);
		gsl_multifit_fdfsolver_set (s, &F, v);
		
		do {
			j++;
			status = gsl_multifit_fdfsolver_iterate (s);
			status = gsl_multifit_test_delta(s->dx, s->x, 0.0, 1e-5);
		}
		
		while (status == GSL_CONTINUE && j < 10000);
		
		gsl_multifit_covar (s->J, 0.0, covar);
		
		c = gsl_vector_get(s->x, 0); sig_c = sqrt(gsl_matrix_get(covar,0,0));
		k = gsl_vector_get(s->x, 1); sig_k = sqrt(gsl_matrix_get(covar,1,1));
		x0 = gsl_vector_get(s->x, 2); sig_x0 = sqrt(gsl_matrix_get(covar,2,2));
		fwhm = gsl_vector_get(s->x, 3); sig_fwhm = sqrt(gsl_matrix_get(covar,3,3));
		
		dd = (PyArrayObject *) 
			PyArray_SimpleNew(aa->nd, aa->dimensions, PyArray_DOUBLE);
		
		for (i = 0; i < p.n; i++)
			*(double *) (dd->data + i*dd->strides[0]) = p.fit[i];
		
		/* Bisector part : */
		// Declarations...
		int    nstep=100, margin=5, len_depth=nstep-2*margin+1;
		double sigma = fwhm/2./pow(2.*log(2.),.5), vr, v0=x0+x[0];
		double dCCFdRV, d2CCFdRV2, d2RVdCCF2;
		double *norm_CCF, *depth, *bis, *p0, *p1, *p2; 

		// Allocations...
		norm_CCF = (double *)malloc(sizeof(double)*p.n);
		depth    = (double *)malloc(sizeof(double)*len_depth);
		bis      = (double *)malloc(sizeof(double)*len_depth);
		p0       = (double *)malloc(sizeof(double)*p.n);
		p1       = (double *)malloc(sizeof(double)*p.n);
		p2       = (double *)malloc(sizeof(double)*p.n);

		// Initialization...
		for (i=0; i<p.n; i++) norm_CCF[i] = -c/k*(1.-p.y[i]/c);
		for (i=0; i<(nstep-2*margin+1); i++) depth[i] = (double )(i+margin)/nstep;
		
		
		for (i=0; i<p.n-1; i++) {
		  //if ((max(norm_CCF[i],norm_CCF[i+1]) >= depth[0]) &&
		  //   (min(norm_CCF[i],norm_CCF[i+1]) <= depth[p.n-1])){
		    vr = (x[i]+x[i+1])/2.;
		    dCCFdRV = -(vr-v0)/pow(sigma,2)*exp(-pow((vr-v0),2)/2./pow(sigma,2));
		    d2CCFdRV2 = (pow((vr-v0),2)/pow(sigma,2)-1)/pow(sigma,2)*exp(-pow((vr-v0),2)/2./pow(sigma,2));
		    d2RVdCCF2 = -d2CCFdRV2/pow(dCCFdRV,3);
		    p2[i] = d2RVdCCF2/2.;
		    p1[i] = (x[i+1]-x[i]-p2[i]*(pow(norm_CCF[i+1],2)-pow(norm_CCF[i],2)))/(norm_CCF[i+1]-norm_CCF[i]);
		    p0[i] = x[i]-p1[i]*norm_CCF[i]-p2[i]*pow(norm_CCF[i],2);
			   };//};
		
		int ind_max = 0, i_b, i_r;
		for (i=0; i<p.n; i++) if (norm_CCF[i]>norm_CCF[ind_max]) ind_max=i;
		for (i=0; i<len_depth; i++) {
		  i_b = ind_max; i_r = ind_max;
		  while ((norm_CCF[i_b] > depth[i]) && (i_b > 1)) i_b--;
		  while ((norm_CCF[i_r+1] > depth[i]) && (i_r < (p.n-2))) i_r++;
		  bis[i] = (p0[i_b]+p0[i_r]) + (p1[i_b]+p1[i_r])*depth[i] + (p2[i_b]+p2[i_r])*pow(depth[i],2);
		  bis[i] /= 2.;
		  //printf("%f\t%f\t%f\t%f\t%i\t%i\n", bis[i], depth[i], p0[i_b], p0[i_r], i_b, i_r);
		};
		
		
		int n1=0, n2=0;
		double RV_top=0., RV_bottom=0., span;
		for (i=0; i<len_depth; i++) {
		    if ((depth[i]>=0.1) && (depth[i] <= 0.4)) {
		        n1++; 
			RV_top += bis[i];};
		    if ((depth[i]>=0.6) && (depth[i] <= 0.9)) {
		        n2++;
			RV_bottom += bis[i];};
		};
		RV_top    /= n1;
		RV_bottom /= n2;
		span = RV_top-RV_bottom;
          
        ee = (PyArrayObject *)
        PyArray_SimpleNew(zz->nd,zz->dimensions, PyArray_DOUBLE);
        ff = (PyArrayObject *)
        PyArray_SimpleNew(zz->nd,zz->dimensions, PyArray_DOUBLE);
        
		for (i = 0; i < len_depth; i++)
			*(double *) (ee->data + i*ee->strides[0]) = depth[i];
        for (i = 0; i < len_depth; i++)
			*(double *) (ff->data + i*ff->strides[0]) = bis[i];
        
		//printf("%f\t%f\t%f\n", span, RV_top, RV_bottom);		
        return Py_BuildValue("NddddddddidNNi", dd, c, k, x0+x[0], fwhm, sig_c, sig_k, sig_x0, sig_fwhm, j, span,ff,ee,len_depth);
        //return Py_BuildValue("Oddddddddid", dd, c, k, x0+x[0], fwhm, sig_c, sig_k, sig_x0, sig_fwhm, j, span);
	}




int n_gauss_f (const gsl_vector *v, void *p, gsl_vector *f)
	{
		double *x = ((struct gauss_params *) p)->x;
		double *y = ((struct gauss_params *) p)->y;
		double *err = ((struct gauss_params *) p)->err;
		double *fit = ((struct gauss_params *) p)->fit;
		int n = ((struct gauss_params *) p)->n;
		int m = ((struct gauss_params *) p)->m;
		int i, j;
		
		double c = gsl_vector_get(v, 0);
		double *k = (double *) malloc(m*sizeof(double));
		double *x0 = (double *) malloc(m*sizeof(double));
		double *fwhm = (double *) malloc(m*sizeof(double));
		double *sigma = (double *) malloc(m*sizeof(double));
		
		for (j = 0; j < m; j++) {
			
			k[j] = gsl_vector_get(v, j+1);
			x0[j] = gsl_vector_get(v, m+j+1);
			fwhm[j] = gsl_vector_get(v, 2*m+j+1);
			sigma[j] = fwhm[j]/2/sqrt(2*log(2));
		}
		
		for (i = 0; i < n; i++) {
			
			fit[i] = c;
			
			for (j = 0; j < m; j++)
				fit[i] += k[j]*exp(-(x[i]-x0[j])*(x[i]-x0[j])/2/sigma[j]/sigma[j]);
						
			gsl_vector_set (f, i, (fit[i]-y[i])/err[i]);
		}
		
		free(k); free(x0); free(fwhm); free(sigma);
		
		return GSL_SUCCESS;
	}



int n_gauss_df (const gsl_vector *v, void *p, gsl_matrix *J)
	{
		double *x = ((struct gauss_params *) p)->x;
		double *y = ((struct gauss_params *) p)->y;
		double *err = ((struct gauss_params *) p)->err;
		double *fit = ((struct gauss_params *) p)->fit;
		int n = ((struct gauss_params *) p)->n;
		int m = ((struct gauss_params *) p)->m;
		int i, j;
		
		double c = gsl_vector_get(v, 0);
		double *k = (double *) malloc(m*sizeof(double));
		double *x0 = (double *) malloc(m*sizeof(double));
		double *fwhm = (double *) malloc(m*sizeof(double));
		double *sigma = (double *) malloc(m*sizeof(double));
		double *e = (double *) malloc(m*sizeof(double));
		
		for (j = 0; j < m; j++) {
			
			k[j] = gsl_vector_get(v, j+1);
			x0[j] = gsl_vector_get(v, m+j+1);
			fwhm[j] = gsl_vector_get(v, 2*m+j+1);
			sigma[j] = fwhm[j]/2/sqrt(2*log(2));
		}
		
		for (i = 0; i < n; i++) {
			
			gsl_matrix_set (J, i, 0, 1/err[i]);
			
			for (j = 0; j < m; j++) {
				
				e[j] = exp(-(x[i]-x0[j])*(x[i]-x0[j])/2/sigma[j]/sigma[j]);
				gsl_matrix_set (J, i, j+1, e[j]/err[i]);
				gsl_matrix_set (J, i, m+j+1, k[j]/err[i]*e[j]*(x[i]-x0[j])/sigma[j]/sigma[j]);
				gsl_matrix_set (J, i, 2*m+j+1, k[j]/err[i]*e[j]*(x[i]-x0[j])*(x[i]-x0[j])/sigma[j]/sigma[j]/sigma[j]/2/sqrt(2*log(2)));
			}
		}
		
		free(k); free(x0); free(fwhm); free(sigma); free(e);
		
		return GSL_SUCCESS;
	}



int n_gauss_fdf (const gsl_vector *v, void *p, gsl_vector *f, gsl_matrix *J)
	{
		n_gauss_f (v, p, f);
		n_gauss_df (v, p, J);
		
		return GSL_SUCCESS;
	}



static PyObject *n_gauss (PyObject *self, PyObject *args)
	{
		PyArrayObject *aa, *bb, *cc, *dd;
		PyArrayObject *ee, *ff, *gg, *hh, *ii, *jj, *kk;
		
		struct gauss_params p;
		
		int *n;
		double *x, *k, *x0, *fwhm, *sig_k, *sig_x0, *sig_fwhm;
		int i, j, status, iter = 0;
		double c, sig_c;
		
		gsl_vector *v;
		gsl_matrix *covar;
		const gsl_multifit_fdfsolver_type *T;
		gsl_multifit_fdfsolver *s;
		gsl_multifit_function_fdf F;
		
		if (!PyArg_ParseTuple(args, "O!O!O!O!", 
			&PyArray_Type, &aa, &PyArray_Type, &bb, &PyArray_Type, &cc, &PyArray_Type, &dd))
			return NULL;
		
		x = (double *) (aa->data + 0*aa->strides[0]);
		p.y = (double *) (bb->data + 0*bb->strides[0]);
		p.err = (double *) (cc->data + 0*cc->strides[0]);
		p.n = aa->dimensions[0];
		x0 = (double *) (dd->data + 0*dd->strides[0]);
		p.m = dd->dimensions[0];
		
		p.x = (double *) malloc(p.n*sizeof(double));
		p.fit = (double *) malloc(p.n*sizeof(double));
		k = (double *) malloc(p.m*sizeof(double));
		fwhm = (double *) malloc(p.m*sizeof(double));
		n = (int *) malloc(p.m*sizeof(int));
		sig_k = (double *) malloc(p.m*sizeof(double));
		sig_x0 = (double *) malloc(p.m*sizeof(double));
		sig_fwhm = (double *) malloc(p.m*sizeof(double));
		
		for (i = 0; i < p.n; i++)
			p.x[i] = x[i]-x[0];
		
		v = gsl_vector_alloc (p.m*3+1);
		covar = gsl_matrix_alloc (p.m*3+1, p.m*3+1);
		F.f = &n_gauss_f;
		F.df = &n_gauss_df;
		F.fdf = &n_gauss_fdf;
		F.n = p.n;
		F.p = p.m*3+1;
		F.params = &p;
		
		T = gsl_multifit_fdfsolver_lmsder;
		s = gsl_multifit_fdfsolver_alloc (T, p.n, p.m*3+1);
		
		c = p.y[0];
		for (i = 0; i < p.n; i++) if (p.y[i] < c) c = p.y[i];
		gsl_vector_set(v, 0, c);
		
		for (j = 0; j < p.m; j++) {
		
			x0[j] = x0[j]-x[0];
			fwhm[j] = (p.x[p.n-1]-p.x[0])/3./p.m;
			n[j] = x0[j]/(p.x[p.n-1]-p.x[0])*p.n;
			k[j] = p.y[n[j]]-c;
			gsl_vector_set(v, j+1, k[j]);
			gsl_vector_set(v, p.m+j+1, x0[j]);
			gsl_vector_set(v, 2*p.m+j+1, fwhm[j]);
		}
		
		gsl_multifit_fdfsolver_set (s, &F, v);
		
		do {
			iter++;
			status = gsl_multifit_fdfsolver_iterate (s);
			if (status) break;
			status = gsl_multifit_test_delta(s->dx, s->x, 0.0, 1e-5);
		}
		
		while (status == GSL_CONTINUE && iter < 1000);
		
		gsl_multifit_covar (s->J, 0.0, covar);
		
		ff = (PyArrayObject *) 
			PyArray_SimpleNew(dd->nd, dd->dimensions, PyArray_DOUBLE);
		gg = (PyArrayObject *) 
			PyArray_SimpleNew(dd->nd, dd->dimensions, PyArray_DOUBLE);
		hh = (PyArrayObject *) 
			PyArray_SimpleNew(dd->nd, dd->dimensions, PyArray_DOUBLE);
		ii = (PyArrayObject *) 
			PyArray_SimpleNew(dd->nd, dd->dimensions, PyArray_DOUBLE);
		jj = (PyArrayObject *) 
			PyArray_SimpleNew(dd->nd, dd->dimensions, PyArray_DOUBLE);
		kk = (PyArrayObject *) 
			PyArray_SimpleNew(dd->nd, dd->dimensions, PyArray_DOUBLE);
		
		c = gsl_vector_get(s->x, 0); sig_c = sqrt(gsl_matrix_get(covar,0,0));
		
		for (j = 0; j < p.m; j++) {
		
			k[j] = gsl_vector_get(s->x, j+1);
			sig_k[j] = sqrt(gsl_matrix_get(covar,j+1,j+1));
			x0[j] = gsl_vector_get(s->x, p.m+j+1) + x[0];
			sig_x0[j] = sqrt(gsl_matrix_get(covar,p.m+j+1,p.m+j+1));
			fwhm[j] = gsl_vector_get(s->x, 2*p.m+j+1);
			sig_fwhm[j] = sqrt(gsl_matrix_get(covar,2*p.m+j+1,2*p.m+j+1));
			
			*(double *) (ff->data + j*ff->strides[0]) = k[j];
			*(double *) (gg->data + j*gg->strides[0]) = sig_k[j];
			*(double *) (hh->data + j*hh->strides[0]) = x0[j];
			*(double *) (ii->data + j*ii->strides[0]) = sig_x0[j];
			*(double *) (jj->data + j*jj->strides[0]) = fwhm[j];
			*(double *) (kk->data + j*kk->strides[0]) = sig_fwhm[j];
		}
		
		ee = (PyArrayObject *) 
			PyArray_SimpleNew(aa->nd, aa->dimensions, PyArray_DOUBLE);
		
		for (i = 0; i < p.n; i++)
			*(double *) (ee->data + i*ee->strides[0]) = p.fit[i];
		
		return Py_BuildValue("OddOOOOOOi", ee, c, sig_c, ff, gg, hh, ii, jj, kk, iter);
	}




static PyObject *poly (PyObject *self, PyObject *args)
	{
		PyArrayObject *aa, *bb, *cc;
		PyArrayObject *dd, *ee, *ff;
		
		int i, j, n, deg, dim[1];
		double err, chisq;
		gsl_vector *x, *y, *w, *p;
		gsl_matrix *X, *covar;
		double *fit;
		gsl_multifit_linear_workspace *work;
		
		if (!PyArg_ParseTuple(args, "O!O!O!i", 
			&PyArray_Type, &aa, &PyArray_Type, &bb, &PyArray_Type, &cc, &deg))
			return NULL;
		
		n = aa->dimensions[0];
		x = gsl_vector_alloc (n);
		y = gsl_vector_alloc (n);
		w = gsl_vector_alloc (n);
		p = gsl_vector_alloc (deg);
		fit = (double *) malloc(n*sizeof(double));
		X = gsl_matrix_alloc (n, deg);
		covar = gsl_matrix_alloc (deg, deg);
		
		for (i = 0; i < n; i++) {
			gsl_vector_set (x, i, *(double *) (aa->data + i*aa->strides[0]));
			gsl_vector_set (y, i, *(double *) (bb->data + i*bb->strides[0]));
			err = *(double *) (cc->data + i*cc->strides[0]);
			gsl_vector_set (w, i, 1.0/err/err);
			
			for (j = 0; j < deg; j++)
				gsl_matrix_set (X, i, j, gsl_pow_int(gsl_vector_get(x, i), j));
		}
		
		work = gsl_multifit_linear_alloc (n, deg);
		gsl_multifit_wlinear (X, w, y, p, covar, &chisq, work);
		gsl_multifit_linear_free (work);
		
		for (i = 0; i < n; i++) {
			fit[i] = 0.;
			for (j = 0; j < deg; j++)
				fit[i] += gsl_matrix_get (X, i, j) * gsl_vector_get (p, j);
		}
		
		chisq = chisq/(n-deg);
		dim[0] = deg;
		
		dd = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
		ee = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
		ff = (PyArrayObject *) PyArray_SimpleNew(aa->nd, aa->dimensions, PyArray_DOUBLE);
		
		for (j = 0; j < deg; j++) {
			*(double *) (dd->data + j*dd->strides[0]) = gsl_vector_get (p, j);
			*(double *) (ee->data + j*ee->strides[0]) = sqrt(gsl_matrix_get (covar, j, j));
		}
		
		for (i = 0; i < n; i++)
			*(double *) (ff->data + i*ff->strides[0]) = fit[i];
		
		return Py_BuildValue("OOOd", dd, ee, ff, chisq);
	}



//p,sigma_p,mod,chisq = fit2.linlsq(A,b,sigma_b)
//
//
//       | 1 Teff_0 Teff_0^2 |
//       | 1 Teff_1 Teff_1^2 |
//   A = | 1 Teff_2 Teff_2^2 |
//       | 1 Teff_3 Teff_3^2 |
//       | . ...... ........ |
//
// b = Data
// sigma_b = Data error
//
//call in python
//
//Matrice = dstack([ones(len(vrad)),jdb,jdb**2])
//para,sig_para,mod,chisq = fit2.linlsq(Matrice[0],vrad,svrad)
//
//
static PyObject *linlsq (PyObject *self, PyObject *args)
	{
		PyArrayObject *aa, *bb, *cc;
		PyArrayObject *dd, *ee, *ff;
		
		int i, j, n, m, dim[1];
		double err, chisq;
		gsl_vector *y, *w, *p;
		gsl_matrix *A, *covar;
		double *fit;
		gsl_multifit_linear_workspace *work;
		
		if (!PyArg_ParseTuple(args, "O!O!O!", 
			&PyArray_Type, &aa, &PyArray_Type, &bb, &PyArray_Type, &cc))
			return NULL;
		
		n = aa->dimensions[0];
		m = aa->dimensions[1];
		y = gsl_vector_alloc (n);
		w = gsl_vector_alloc (n);
		p = gsl_vector_alloc (m);
		fit = (double *) malloc(n*sizeof(double));
		A = gsl_matrix_alloc (n, m);
		covar = gsl_matrix_alloc (m, m);
		
		for (i = 0; i < n; i++) {
			gsl_vector_set (y, i, *(double *) (bb->data + i*bb->strides[0]));
			err = *(double *) (cc->data + i*cc->strides[0]);
			gsl_vector_set (w, i, 1.0/err/err);
			
			for (j = 0; j < m; j++)
				gsl_matrix_set (A, i, j, *(double *) (aa->data + i*aa->strides[0] + j*aa->strides[1]));
		}
		
		work = gsl_multifit_linear_alloc (n, m);
		gsl_multifit_wlinear (A, w, y, p, covar, &chisq, work);
		gsl_multifit_linear_free (work);
		
		for (i = 0; i < n; i++) {
			fit[i] = 0.;
			for (j = 0; j < m; j++)
				fit[i] += gsl_matrix_get (A, i, j) * gsl_vector_get (p, j);
		}
		
		chisq = chisq/(n-m);
		dim[0] = m;
		
		dd = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
		ee = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
		ff = (PyArrayObject *) PyArray_SimpleNew(bb->nd, bb->dimensions, PyArray_DOUBLE);
		
		for (j = 0; j < m; j++) {
			*(double *) (dd->data + j*dd->strides[0]) = gsl_vector_get (p, j);
			*(double *) (ee->data + j*ee->strides[0]) = sqrt(gsl_matrix_get (covar, j, j));
		}
		
		for (i = 0; i < n; i++)
			*(double *) (ff->data + i*ff->strides[0]) = fit[i];
		
		return Py_BuildValue("OOOd", dd, ee, ff, chisq);
	}




struct model_params
	{
		double *vr, *ccf, *mean;
		int n1;
	};



void spline(double *x1, double *y1, int n1, double *x2, double *y2, int n2)
	{
		int i;
		gsl_interp_accel *acc = gsl_interp_accel_alloc ();
		gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, n1);
		gsl_spline_init (spline, x1, y1, n1);

		for (i = 0; i < n2; i++)
			y2[i] = gsl_spline_eval (spline, x2[i], acc);
		
		gsl_spline_free (spline);
		gsl_interp_accel_free(acc);
	}



double eval_model (double dv, void *p)
	{
		double *vr = ((struct model_params *) p)->vr;
		double *ccf = ((struct model_params *) p)->ccf;
		double *mean = ((struct model_params *) p)->mean;
		int n1 = ((struct model_params *) p)->n1;
		
		int i;
		double result = 0.;
		double *vr2 = (double *) malloc(n1*sizeof(double));
		double *mean2 = (double *) malloc(n1*sizeof(double));
		
		for (i = 0; i < n1; i++)
			vr2[i] = vr[i] + dv;
		
		spline(vr2, mean, n1, vr, mean2, n1);
		
		for (i = 0; i < n1; i++)
			result += (ccf[i]-mean2[i])*(ccf[i]-mean2[i]);	
		
		free(vr2); free(mean2);
		
		return result;
	}



static PyObject *model (PyObject *self, PyObject *args)
	{
		PyArrayObject *aa, *bb, *cc;
		PyArrayObject *dd;
		
		struct model_params p;
		
		int i, j, status;
		double vmin, vmax, dv, result;
		
		const gsl_min_fminimizer_type *T = gsl_min_fminimizer_brent;
		gsl_min_fminimizer *s = gsl_min_fminimizer_alloc (T);
		gsl_function F;
		
		if (!PyArg_ParseTuple(args, "O!O!O!", 
			&PyArray_Type, &aa, &PyArray_Type, &bb, &PyArray_Type, &cc))
			return NULL;
		
		p.vr = (double *) (aa->data + 0*aa->strides[0]);
		p.ccf = (double *) (bb->data + 0*bb->strides[0]);
		p.mean = (double *) (cc->data + 0*cc->strides[0]);
		
		p.n1 = aa->dimensions[0];
		
		F.function = &eval_model;
		F.params = &p;
		gsl_min_fminimizer_set (s, &F, 0., -0.1, 0.1);
		j = 0;
		
		do {
			j++;
			status = gsl_min_fminimizer_iterate(s);
			dv = gsl_min_fminimizer_x_minimum (s);
			vmin = gsl_min_fminimizer_x_lower (s);
			vmax = gsl_min_fminimizer_x_upper (s);
			/* printf("%1.5f %1.5f %1.5f\n", dv, vmin, vmax); */
			status = gsl_min_test_interval(vmin, vmax, 1e-4, 0.);
		}
		
		while (status == GSL_CONTINUE && j < 10000);
		
		result = eval_model (dv, &p);
		
		return Py_BuildValue("ddi", dv, result, j);
	}




static PyMethodDef fit_methods[] = {
	
	{"gauss", gauss, METH_VARARGS, "Fit a Gaussian function."},
	
	{"n_gauss", n_gauss, METH_VARARGS, "Fit N Gaussian functions."},
	
	{"poly", poly, METH_VARARGS, "Fit a polynomial function."},
	
	{"linlsq", linlsq, METH_VARARGS, "Linear least-squares."},
	
	{"model", model, METH_VARARGS, "Fit a model function."},
	
	{"gauss_bis", gauss_bis, METH_VARARGS, "Fit a Gaussian function and calculate the bisector."},
	
	{NULL, NULL, 0, NULL}
	
	};




void initfit2(void)
	{
		(void) Py_InitModule("fit2", fit_methods);
		
		import_array();
	}
