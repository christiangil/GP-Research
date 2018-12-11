#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#define NRANSI

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

void avevar(double *data, int n, double *ave)
{
	int j;
    
	for (*ave=0.0,j=0;j<n;j++) *ave += data[j];
	*ave /= n;
}

int compare (const void * a, const void * b)
{
    const double *da = (const double *) a;
    const double *db = (const double *) b;
    
    return (*da > *db) - (*da < *db);
}

void median(double *data, int n1, double *med)
{
    int mean_len;
    
    qsort(data, n1, sizeof(double), compare);
    mean_len = rndup(n1/2.,0)-5;
    *med = data[mean_len];
}



//Calcul le decalage (ou l elargissement) d une raie en tenant compte de la vitesse et de la longueur d onde, en angstrom
// line_width est la largeur de la raie en vitesse
double Delta_lambda(double line_width, double lambda_line0)
{   
    double c=299792458.;
    double line_spread=0.;
	double beta;
	
	// cas relativiste et symmetrique voir "functions.py"
	beta = line_width/c;
	line_spread = -1 * lambda_line0 * (1 - sqrt((1+beta)/(1-beta)));
	
	// v/c*wavelength, cas non relaticviste et non symetrique
    //line_spread = line_width/c*lambda_line0;

    return line_spread;
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


void compute_spectrum(double *lambda, int n1, double *planck_spectrum, double *spectrum,  double *lambda_line, int n2, double *contrast, double *line_width)
{
	int i,j;
	double line_spread;
    
    printf("************************* compute_spectrum *************************\n\n");
    
    //initialisation du spectre a la fonction de planck
    for (j=0;j<n1;j++)
    {
        spectrum[j] = planck_spectrum[j];
    }
    
	for (i=0;i<n2;i++) 
    {
        
        if (i%100 == 0) printf("%d over %d\n",i,n2);
        
        for (j=0;j<n1;j++)
        {
            //if only one line width is used and in m/s
            line_spread = Delta_lambda(line_width[i],lambda_line[i]);
            //line_spread = line_width[i];
            spectrum[j] = spectrum[j] * (1 - contrast[i]*exp(-pow((lambda[j]-lambda_line[i]),2)/(2*line_spread*line_spread)));
        }
    }
}

// VOIR FICHIER TEST_SHIFTING_SPECTRUM.TXT DANS DOSSIER PROGRAM/TEST/.
void shifting_spectrum(double *lambda, double *spectrum, double *spectrum_blueshifted, double *depth_from_continuum, double *fit, int n1, int n2, int func_type )
{
    int i,j;
    double *spectrum_derivative, lambda_shift, *a_quad, *b_quad, *c_quad;
	double x1,x2,x3;
	double y1,y2,y3;
    
    spectrum_derivative = (double *) malloc(n1*sizeof(double));
	a_quad = (double *) malloc(n1*sizeof(double));
	b_quad = (double *) malloc(n1*sizeof(double));
	c_quad = (double *) malloc(n1*sizeof(double));
    for (i=0;i<n1;i++) 
    {
		spectrum_derivative[i] = 0.0;
        spectrum_blueshifted[i] = 0.0;
		a_quad[i] = 0.0;
		b_quad[i] = 0.0;
		c_quad[i] = 0.0;
    }
	
	if (lambda[1]-lambda[0] == 0) spectrum_derivative[0] = 0;
	else spectrum_derivative[0] = (spectrum[1] - spectrum[0])/(lambda[1]-lambda[0]);
		
	for (i=1;i<n1-1;i++)
	{
//		if (spectrum[i+1]==0 || spectrum[i]==0) // avant il y avait spectrum[i+i]==0 a la place de spectrum[i+1]==0 mais je pense que c etait une erreur
//		{
//			spectrum_derivative[i] = 0;
//		}
//		else 
//		{
//			//spectrum_derivative[i] = (spectrum[i+1] - spectrum[i-1])/(2 * lambda_step);
//			spectrum_derivative[i] = (spectrum[i+1] - spectrum[i])/(lambda[i+1]-lambda[i]);
//			//printf("%.6f\t%.6f\t%.6f\n",spectrum[i+1] - spectrum[i],lambda[i+1]-lambda[i],spectrum_derivative[i]);
//		}
		if (lambda[i+1]-lambda[i] == 0) spectrum_derivative[i] = spectrum_derivative[i-1];
		else
		{
			//spectrum_derivative[i] = (spectrum[i+1] - spectrum[i-1])/(2 * lambda_step);
			spectrum_derivative[i] = (spectrum[i+1] - spectrum[i])/(lambda[i+1]-lambda[i]);
			//printf("%.6f\t%.6f\t%.6f\n",spectrum[i+1] - spectrum[i],lambda[i+1]-lambda[i],spectrum_derivative[i]);
		}
		
		//a*x**2 + b*x + c = y
		//a = ((y1-y2)/(x2-x1)*x3 - y1 + (-y1+y2)/(x2-x1)*x1 + y3) / (x3**2 + (x1**2-x2**2)/(x2-x1)*x3 - x1**2 - (x1**2-x2**2)/(x2-x1)*x1)
		//b = (a*(x1**2-x2**2) - y1 + y2) / (x2-x1)
		//c = y1 - a*x1**2 - b*x1		
		
		x1 = lambda[i-1],x2 = lambda[i],x3 = lambda[i+1];
		y1 = spectrum[i-1],y2 = spectrum[i],y3 = spectrum[i+1];
		if ((x2-x1 == 0) & (i==1))
		{
			a_quad[i] = 0;
			b_quad[i] = 0;
			c_quad[i] = 0;
		}
		else if ((x2-x1 == 0) || (x2-x3==0))
		{
			a_quad[i] = a_quad[i-1];
			b_quad[i] = b_quad[i-1];
			c_quad[i] = c_quad[i-1];
		}
		else
		{
			a_quad[i] = ((y1-y2)/(x2-x1)*x3 - y1 + (-y1+y2)/(x2-x1)*x1 + y3) / (pow(x3,2) + (pow(x1,2)-pow(x2,2))/(x2-x1)*x3 - pow(x1,2) - (pow(x1,2)-pow(x2,2))/(x2-x1)*x1);
			b_quad[i] = (a_quad[i]*(pow(x1,2)-pow(x2,2)) - y1 + y2) / (x2-x1);
			c_quad[i] = y1 - a_quad[i]*pow(x1,2) - b_quad[i]*x1;
		}
			
    }
	
	
	a_quad[0] = a_quad[1],b_quad[0] = b_quad[1],c_quad[0] = c_quad[1];
	a_quad[n1-1] = a_quad[n1-2],b_quad[n1-1] = b_quad[n1-2],c_quad[n1-1] = c_quad[n1-2];
	spectrum_derivative[i] = 0.;
	
	// TO SHIFT THE CCF GIVEN A VELOCITY OR THE SPECTRUM GIVEN A WAVELENGTH. IN THIS CASE V_SHIFT IS IN REALITY LAMBDA_SHIFT. THE POINTS IN VELOCITY (OR WAVELENGTH) ARE FIXED.
    // IL FAUT UTILISER -v_shift car on garde les meme point en vitesse mais on decale toute la CCF
	if (func_type == 1)
	{
		for (i=0;i<n1;i++)
		{
			//if (i%100 == 0) printf("%d over %d\n",i,n1);
			//printf("%d   %.2f  %.2f  %.2f   %.2f   %.2f\n",i,spectrum[i+1],spectrum[i-1],lambda[i+1],lambda[i-1],spectrum_derivative[i]);
			//v_shift = maximum_blueshift_conv*depth_from_continuum[i];
			//See "shift_mask_holes_to_meet_conv_blueshift" for the estimation of these terms
			//v_shift = 6719.15208437*pow(depth_from_continuum[i],4) - 16207.45372463*pow(depth_from_continuum[i],3) + 15054.13259329*pow(depth_from_continuum[i],2) - 6711.32809434*depth_from_continuum[i] + 862.14880259;
			double v_shift=0;
            for (j=0;j<n2;j++)
            {
                v_shift += fit[j]*pow(depth_from_continuum[i],(n2-1-j));
            }
                
			if ((v_shift >= 0) || (i == 0)) 
			{
				spectrum_blueshifted[i] = spectrum[i] - v_shift * spectrum_derivative[i]; // dy/dx = derivative => dy = derivative*dx
			}
			else if (v_shift < 0)
			{
				spectrum_blueshifted[i] = spectrum[i] - v_shift * spectrum_derivative[i-1]; // dy/dx = derivative => dy = derivative*dx
			}
			
			//spectrum_blueshifted[i] = spectrum[i] + v_shift * spectrum_derivative[i];
			//printf("%.6f\t%.6f\t%i\n",v_shift,spectrum_derivative[i],func_type);
			//printf("%.6f\t%.6f\n",spectrum[i],spectrum_blueshifted[i]);
        }
	}
		
	// si la derniere valeur est 2, la valeur "depth_from_continuum" correspond a l ecart entre le sampling que l on veut et celui reel
	// depth_from_continuum = freq_wanted - freq_reelle
    // IL FAUT UTILISER +depth_from_continuum = freq_wanted - freq_reelle car la CCF est la meme, mais on decale les points en vitesse
    // TO KEEP THE SAME CCF (OR SPECTRUM) BUT TO SHIFT THE POINTS IN VELOCITY (OR WAVELENGTH) 
	else if (func_type == 2)
	{
		for (i=0;i<n1;i++)
		{
			//if (i%100 == 0) printf("%d over %d\n",i,n1);
			//printf("%d   %.2f  %.2f  %.2f   %.2f   %.2f\n",i,spectrum[i+1],spectrum[i-1],lambda[i+1],lambda[i-1],spectrum_derivative[i]);
			
			if ((depth_from_continuum[i] >= 0) || (i == 0)) 
			{
				spectrum_blueshifted[i] = spectrum[i] + depth_from_continuum[i] * spectrum_derivative[i]; // dy/dx = derivative => dy = derivative*dx
				// spectrum_blueshifted[i] = spectrum[i] - dy
				//spectrum_blueshifted[i] = spectrum[i] + (a_quad[i]*pow(lambda[i]+depth_from_continuum[i],2) + b_quad[i]*(lambda[i]+depth_from_continuum[i]) + c_quad[i] - spectrum[i]);
			}
			else if (depth_from_continuum[i] < 0)
			{
				spectrum_blueshifted[i] = spectrum[i] + depth_from_continuum[i] * spectrum_derivative[i-1]; // dy/dx = derivative => dy = derivative*dx
				// spectrum_blueshifted[i] = spectrum[i] - dy
				//spectrum_blueshifted[i] = spectrum[i] + (a_quad[i-1]*pow(lambda[i]+depth_from_continuum[i],2) + b_quad[i-1]*(lambda[i]+depth_from_continuum[i]) + c_quad[i-1] - spectrum[i]);

			}
			//printf("%.2f\t%.2f\n",spectrum[i],spectrum[i] + depth_from_continuum[i] * spectrum_derivative[i]);
            
			//spectrum_blueshifted[i] = spectrum[i] + depth_from_continuum[i] * spectrum_derivative[i];
			
			// spectrum_blueshifted[i] = spectrum[i] - dy
			//spectrum_blueshifted[i] = spectrum[i] + (a_quad[i]*pow(lambda[i]+depth_from_continuum[i],2) + b_quad[i]*(lambda[i]+depth_from_continuum[i]) + c_quad[i] - spectrum[i]);
        }
	}
    // TO SHIFT THE SPECTRUM GIVEN A VELOCITY. THE POINTS IN WAVELENGTH ARE FIXED.
    // IL FAUT UTILISER -lambda_shift car on garde les meme point en vitesse mais on decale toute la CCF
	else 
	{
		for (i=0;i<n1;i++)
		{
			//if (i%100 == 0) printf("%d over %d\n",i,n1);
			//printf("%d   %.2f  %.2f  %.2f   %.2f   %.2f\n",i,spectrum[i+1],spectrum[i-1],lambda[i+1],lambda[i-1],spectrum_derivative[i]);
			//v_shift = maximum_blueshift_conv*depth_from_continuum[i];
			//See "shift_mask_holes_to_meet_conv_blueshift" for the estimation of these terms
			//v_shift = 6719.15208437*pow(depth_from_continuum[i],4) - 16207.45372463*pow(depth_from_continuum[i],3) + 15054.13259329*pow(depth_from_continuum[i],2) - 6711.32809434*depth_from_continuum[i] + 862.14880259;
			double v_shift=0;
            for (j=0;j<n2;j++)
            {
                v_shift += fit[j]*pow(depth_from_continuum[i],(n2-1-j));
            }
			
			lambda_shift = Delta_lambda(v_shift,lambda[i]);
			if ((lambda_shift >= 0) || (i == 0)) 
			{
				spectrum_blueshifted[i] = spectrum[i] - lambda_shift * spectrum_derivative[i]; // dy/dx = derivative => dy = derivative*dx
			}
			else if (lambda_shift < 0)
			{
				spectrum_blueshifted[i] = spectrum[i] - lambda_shift * spectrum_derivative[i-1]; // dy/dx = derivative => dy = derivative*dx
			}
        }
//        if (i<20) 
//        {
//            printf("%.8f   %.8f   %.8f   %.8f      %.8f      %.8f\n",depth_from_continuum[i],v_shift,lambda_shift,spectrum[i],spectrum_derivative[i],spectrum_blueshifted[i]);
//        }
    }
    
    free(spectrum_derivative);free(a_quad);free(b_quad);free(c_quad);
}

void lower_resolution(double *wavelength, double *spectrum, double *spectrum_low_reso, int n1,double sigma_resolution)
{
    int i,j;
    double sigma_wavelength,wavelength_step,Gaussian_low_reso;
    double window_width=50.,pi=3.14159265359; // size of the window in sigma (of the Gaussian) on which the Gaussian used to convolve is not zero
    int nb_pixel;
    
    // the wavelength step is smaller in the blue, so this is the minimum wavelength step
    // I calculate here the number of pixel to consider.
    wavelength_step=wavelength[1]-wavelength[0];
    nb_pixel = rndup((window_width*Delta_lambda(sigma_resolution,wavelength[0]))/wavelength_step,0); // I want to have always the same number of pixels and it will be larger in the blue, so I fix it in the blue
    
    for (i=0;i<n1;i++)
    {
        int min_pixel=0.,max_pixel=n1-1;
        double *wavelength_step,wavelength_step_median;

        //printf("i = %i over %i\n",i,n1);
        
        //I calculate min_pixel and max_pixel to select a small part of the spectrum
        sigma_wavelength = Delta_lambda(sigma_resolution,wavelength[i]);
        if (i-nb_pixel>=0) min_pixel = i-nb_pixel;
        if (i+nb_pixel<n1-1) max_pixel = i+nb_pixel;
        
        //I calculate what is the median wavelength step in this small part of the spectrum
        wavelength_step = (double *) malloc((max_pixel-min_pixel-1)*sizeof(double));
        for (j=min_pixel;j<max_pixel-1;j++) wavelength_step[j-min_pixel] = wavelength[j+1]-wavelength[j]; // "[j-min_pixel]" car "wavelength_step" commence a 0
        avevar(wavelength_step,max_pixel-min_pixel-1,&wavelength_step_median);
        //median(wavelength_step,max_pixel-min_pixel-1,&wavelength_step_median);
        free(wavelength_step);
        
        //I multiply the small part of the spectrum with a normalized Gaussian that have for width the resolution of the instrument
        for (j=min_pixel;j<max_pixel;j++)
        {
            // 1/(sigma*sqrt(2*pi)) allows the Gaussian to be normalized
            //Because the pixel scale is not constant, we normalized by it to account for the same area under the Gaussian
            Gaussian_low_reso = wavelength_step_median*1./(sigma_wavelength*sqrt(2.*pi))*exp(-(wavelength[j]-wavelength[i])*(wavelength[j]-wavelength[i])/(2*sigma_wavelength*sigma_wavelength));
            spectrum_low_reso[i] += spectrum[j] * Gaussian_low_reso;
        }
            
    }
}
        


static PyObject *get_spectrum(PyObject *self, PyObject *args)
{
	PyArrayObject *a, *b, *c, *d, *e;
	PyArrayObject *aa;
	
	int i, n1, n2, dim[1];
    double *line_width, *lambda, *spectrum, *planck_spectrum, *lambda_line, *contrast;
    
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &PyArray_Type, &d, &PyArray_Type, &e))
		return NULL;
	
	lambda = (double *) (a->data + 0*a->strides[0]);
	planck_spectrum = (double *) (b->data + 0*b->strides[0]);
	n1 = a->dimensions[0];
    lambda_line = (double *) (c->data + 0*c->strides[0]);
    contrast = (double *) (d->data + 0*d->strides[0]);
    line_width = (double *) (e->data + 0*e->strides[0]);
    n2 = c->dimensions[0];
    
    spectrum = (double *) malloc(n1*sizeof(double));

	compute_spectrum(lambda, n1, planck_spectrum, spectrum, lambda_line, n2, contrast, line_width);
	
	dim[0] = n1;
	aa = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
   
	for (i = 0; i < n1; i++) {
		*(double *) (aa->data + i*aa->strides[0]) = spectrum[i];
    }
    
    free(spectrum);
    
    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
	return Py_BuildValue("N",aa);
}

static PyObject *get_spectrum_blueshifted(PyObject *self, PyObject *args)
{
	PyArrayObject *a, *b, *c, *d;
	PyArrayObject *aa;
	
	int i, n1, n2, dim[1];
    double *lambda, *spectrum, *spectrum_blueshifted, *depth_from_continuum, *fit;
    int func_type;
    
    //printf("test\n");
    
	//if (!PyArg_ParseTuple(args, "O!O!O!|dd", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &lambda_step, &maximum_blueshift_conv))
	if (!PyArg_ParseTuple(args, "O!O!O!O!|i", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &PyArray_Type, &d, &func_type))
        return NULL;
    
	
    //printf("test\n");

    
	lambda = (double *) (a->data + 0*a->strides[0]);
	spectrum = (double *) (b->data + 0*b->strides[0]);
    depth_from_continuum = (double *) (c->data + 0*c->strides[0]);
    fit = (double *) (d->data + 0*d->strides[0]);
	n1 = a->dimensions[0];
    n2 = d->dimensions[0];
    
    //for (i=0; i<10; i++) {
    //    printf("spectrum[i] = %.2f\n",spectrum[i]);
    //    printf("depth_from_continuum[i] = %.2f\n",depth_from_continuum[i]);
    //    printf("lambda[i] = %.2f\n",lambda[i]);
    //}
    
    spectrum_blueshifted = (double *) malloc(n1*sizeof(double));
	    
	//shifting_spectrum(lambda, spectrum, spectrum_blueshifted, depth_from_continuum, n1, lambda_step, maximum_blueshift_conv);
    shifting_spectrum(lambda, spectrum, spectrum_blueshifted, depth_from_continuum, fit, n1, n2, func_type);
		
	dim[0] = n1;
	aa = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
    
	for (i = 0; i < n1; i++) {
		*(double *) (aa->data + i*aa->strides[0]) = spectrum_blueshifted[i];
    }
    free(spectrum_blueshifted);
    
    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
	return Py_BuildValue("N",aa);
}

static PyObject *get_same_sampling(PyObject *self, PyObject *args)
{
	PyArrayObject *a, *b, *c;
	PyArrayObject *aa, *bb, *cc, *dd;
	    
	int i, n1, n2, dim[1];
    double *freq,*freq_oversampled,*flux_oversampled,*freq_same,*flux_same,*index_same,*diff;
    
	if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c))
		return NULL;
	
	freq = (double *) (a->data + 0*a->strides[0]);
	freq_oversampled = (double *) (b->data + 0*b->strides[0]);
    flux_oversampled = (double *) (c->data + 0*c->strides[0]);
	n1 = a->dimensions[0];
    n2 = b->dimensions[0];
    
    freq_same  = (double *) malloc(n1*sizeof(double));
    flux_same  = (double *) malloc(n1*sizeof(double));
	index_same = (double *) malloc(n1*sizeof(double));
	diff       = (double *) malloc(n1*sizeof(double));
    
    obtain_same_sampling(freq,n1,freq_oversampled,n2,flux_oversampled,freq_same,flux_same,index_same,diff);
	
	dim[0] = n1;
	aa = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
    bb = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
	cc = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
	dd = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
    
	for (i = 0; i < n1; i++) {
		*(double *) (aa->data + i*aa->strides[0]) = freq_same[i];
        *(double *) (bb->data + i*bb->strides[0]) = flux_same[i];
		*(double *) (cc->data + i*cc->strides[0]) = index_same[i];
		*(double *) (dd->data + i*dd->strides[0]) = diff[i];
    }
    
    free(freq_same);
    free(flux_same);
	free(index_same);
	free(diff);
    
    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
	return Py_BuildValue("NNNN",aa,bb,cc,dd);
}

static PyObject *func_lower_resolution(PyObject *self, PyObject *args)
{
    PyArrayObject *a, *b;
    PyArrayObject *aa;

    int i, n1, dim[1];
    double *wavelength,*spectrum,*spectrum_low_reso;
    double sigma_resolution;

    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &a, &PyArray_Type, &b, &sigma_resolution))
    return NULL;

    wavelength = (double *) (a->data + 0*a->strides[0]);
    spectrum = (double *) (b->data + 0*b->strides[0]);
    n1 = a->dimensions[0];

    spectrum_low_reso  = (double *) malloc(n1*sizeof(double));
    
    //init
    for (i=0;i<n1;i++) spectrum_low_reso[i] = 0.0;
    
    lower_resolution(wavelength,spectrum,spectrum_low_reso,n1,sigma_resolution);

    dim[0] = n1;
    aa = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
    for (i = 0; i < n1; i++) 
    {
        *(double *) (aa->data + i*aa->strides[0]) = spectrum_low_reso[i];
    }

    free(spectrum_low_reso);
    
    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
	return Py_BuildValue("N",aa);
}

static PyMethodDef spectrum_methods[] = {
	
	{"get_spectrum", get_spectrum, METH_VARARGS,
 	"compute spectrum"},
    {"get_spectrum_blueshifted", get_spectrum_blueshifted, METH_VARARGS,
        "compute spectrum blueshifted"},
    {"get_same_sampling", get_same_sampling, METH_VARARGS,
        "get same sampling whencomparing 2 spectrum"},
    {"func_lower_resolution", func_lower_resolution, METH_VARARGS,
        "lower the resolution of a spectrum"},
 	
	{NULL, NULL, 0, NULL}
	
};


void initcalculate_spectrum(void)
{
	(void) Py_InitModule("calculate_spectrum", spectrum_methods);
	
	import_array();
}


#undef TWOPID
#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software 5.){2puDY5m.`. */
