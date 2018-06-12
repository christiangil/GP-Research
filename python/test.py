#importing necessary packages
import numpy as np #for vector manipulation
from numpy.random import normal, seed, uniform #for drawing random values and setting the random seed
from scipy.optimize import minimize #for optimizing the log likelihood (in the later regression steps)
from scipy.special import gamma, kv #for matern kernel
import matplotlib.pyplot as plt #for plotting

#Setting a random seed so final results can be reproduced
seed(2)

#Linear kernel
def linear_kernel(hyperparameters,x1,x2):

    sigma_b, sigma_a = hyperparameters
    return sigma_b**2*np.dot(x1,x2)+ sigma_a**2

#Radial basis function kernel(aka squared exonential, ~gaussian)
def RBF_kernel(hyperparameters,dif_sq):

    kernel_amplitude, kernel_length = hyperparameters
    return kernel_amplitude**2*np.exp(-dif_sq/(2*kernel_length**2))

#Ornstein–Uhlenbeck (Exponential) kernel
def OU_kernel(hyperparameters,dif):

    kernel_amplitude, kernel_length = hyperparameters
    return kernel_amplitude**2*np.exp(-dif/(kernel_length))

#Periodic kernel
def periodic_kernel(hyperparameters,dif):

    kernel_amplitude, kernel_length, kernel_period = hyperparameters
    return kernel_amplitude**2*np.exp(-2*np.sin(np.pi*dif/kernel_period)**2/(kernel_length**2))

#Matern kernel (not sure if I implemented this correctly. Gets nans when dif==0)
def Matern_kernel(hyperparameters,dif,nu):

    kernel_amplitude, kernel_length = hyperparameters
    if dif==0:
        return 0
    else:
        return kernel_amplitude**2*((2**(1-nu))/(gamma(nu)))*((np.sqrt(2*nu)*dif)/(kernel_length))**nu*kv(nu,(np.sqrt(2*nu)*dif)/(kernel_length))

#Rational Quadratic kernel (equivalent to adding together many SE kernels
#with different lengthscales. When α→∞, the RQ is identical to the SE.)
def RQ_kernel(hyperparameters,dif_sq):

    kernel_amplitude, kernel_length, alpha = hyperparameters
    alpha=max(alpha,0)
    return kernel_amplitude**2*(1+dif_sq/(2*alpha*kernel_length**2))**-alpha

#Creating a custom kernel (possibly by adding and multiplying other kernels?)
def kernel(hyperparameters,x1,x2):

    #converting inputs to arrays and finding their squared differences
    x1a=np.array(x1)
    x2a=np.array(x2)
    dif_vec=x1a-x2a
    dif_sq=np.dot(dif_vec,dif_vec)
    dif=np.sqrt(dif_sq)

    # final = RBF_kernel(hyperparameters,dif_sq)
    # final = OU_kernel(hyperparameters,dif)
    # final = linear_kernel(hyperparameters,x1a,x2a)
    final = periodic_kernel(hyperparameters,dif)
    # final = RQ_kernel(hyperparameters,dif_sq)
    # final = Matern_kernel(hyperparameters,dif,3/2) #doesn't seem to be working right now

    #example of adding kernels
    # final = periodic_kernel(hyperparameters[0:3],dif)+RBF_kernel(hyperparameters[3:5],dif_sq)

    #example of multiplying kernels
    # final = periodic_kernel(hyperparameters[0:3],dif)*RBF_kernel(hyperparameters[3:5],dif_sq)

    return final

    #Creates the covariance matrix by evaluating the kernel function for each pair of passed inputs
def covariance(x1list,x2list,hyperparameters):
    K=np.zeros((len(x1list),len(x2list)))
    for i, x1 in enumerate(x1list):
        for j, x2 in enumerate(x2list):
            K[i,j]=kernel(hyperparameters,x1,x2)
    return K

    #kernel hyper parameters (AFFECTS SHAPE OF GP's, make sure you have the right amount!)
#just some starting values, see each kernel for what these mean
# hyperparameters = [0.8, 0.3, 2 ,0.5, .3]
hyperparameters = [0.25, 0.3, 4]

#how finely to sample the domain
GP_sample_amount=500+1

#creating many inputs to sample the eventual gaussian process on
domain=4 #how wide the measurement domain is
x_samp=np.linspace(0,domain,GP_sample_amount)

#Finding how correlated the sampled inputs are to each other
#(aka getting the covariance matrix by evaluating the kernel function at all pairs of points)
K_samp=covariance(x_samp,x_samp,hyperparameters)

#plot the evaluated covariance matrix (colors show how correlated points are to each other)
plt.figure(figsize=(8,8))
plt.imshow(K_samp)
plt.colorbar()
plt.show()

#noise to add to gaussians (to simulate observations)
GP_noise=.1

#how many GPs to plot
amount_of_GPs=10

#plotting amount_of_GPs randomly drawn Gaussian processes using the kernel function to correlate points
plt.figure(figsize=(10,6))
for i in range(amount_of_GPs):

    #sampled possible GP function values
    GP_func=np.matmul(K_samp,normal(size=GP_sample_amount))
    plt.plot(x_samp,GP_func, zorder=2)

    #sampled possible GP observations (that have noise)
    GP_obs=GP_func+GP_noise*normal(size=GP_sample_amount)
    plt.plot(x_samp,GP_obs, alpha=0.5, zorder=1)

plt.xlabel("x (time)")
plt.ylabel("y (flux or something lol)")
plt.title("Gaussian Processes (with noise?)")
plt.show()

#"true" underlying function for the fake observations
def observations(x,measurement_noise):

    #a phase shifted sine curve with noise
    shift = uniform(high=2*np.pi)
    return np.sin(np.pi/2*x+shift)+np.multiply(np.square(measurement_noise),normal(size=len(x)))

#creating observations to test methods on
amount_of_measurements=9

#Uncertainty in the data (AFFECTS SPREAD OF DATA AND HOW TIGHTLY THE GP's WILL TRY TO HUG THE DATA)
#aka how much noise is added to measurements and measurement covariance function
#can be a list (length=amount_of_measurements) or single number
measurement_noise=0.3

#make measurement noise into an array (for my convenience)
measurement_noise = np.ndarray.flatten(np.array([measurement_noise]))
if len(measurement_noise)==1:
    measurement_noise=np.ones(amount_of_measurements)*measurement_noise[0]

x_obs=np.linspace(0,domain,amount_of_measurements) #observation inputs
y_obs=observations(x_obs,measurement_noise) #observation outputs

#plotting some more randomly drawn Gaussian processes before data influences the posterior
plt.figure(figsize=(10,6))
draws=5000
show=50
storage=np.zeros((draws,GP_sample_amount))
#drawing lots of GPs (to get a good estimate of 5-95th percentile.
#Haven't found the analytical way to do it
for i in range(draws):
    storage[i,:]=np.matmul(K_samp,normal(size=GP_sample_amount))
    # plt.plot(x_samp,test[i,:], alpha=0.5, zorder=1)
#only showing some
for i in range(show):
    plt.plot(x_samp,storage[i,:], alpha=0.5, zorder=1)
#filling the 5-95th percentile with a transparent orange
storage=np.sort(storage,axis=0)
plt.fill_between(x_samp, storage[int(0.95*draws),:], storage[int(0.05*draws),:], alpha=0.3, color='orange')
#plt.plot(x_samp,np.matmul(K_post,np.ones(GP_sample_amount)*1.95)+mean_post, color="black", zorder=2)
#plt.plot(x_samp,np.matmul(K_post,np.ones(GP_sample_amount)*-1.95)+mean_post, color="black", zorder=2)
plt.scatter(x_obs,y_obs,color="black",zorder=2)
plt.xlabel("x (time)")
plt.ylabel("y (flux or something lol)")
plt.title("Gaussian Processes (Prior)")
plt.show()

#creating additional covariance matrices (as defined on pages 13-15 of Eric Schulz's tutorial)
noise_I=np.square(measurement_noise)*np.identity(amount_of_measurements)
K_obs=covariance(x_obs,x_obs,hyperparameters)+noise_I
K_samp_obs=covariance(x_samp,x_obs,hyperparameters)
K_obs_samp=covariance(x_obs,x_samp,hyperparameters)

#mean of the conditional distribution p(f_samp|x_obs,t_obs,x_samp) (as defined on page 14)
mean_post=np.matmul(K_samp_obs,np.matmul(np.linalg.inv(K_obs),y_obs))
#covariance matrix of the conditional distribution p(f_samp|x_obs,t_obs,x_samp) (as defined on page 14)
K_post=K_samp-np.matmul(K_samp_obs,np.matmul(np.linalg.inv(K_obs),K_obs_samp))

#plot the posterior covariance matrix (colors show how correlated points are to each other)
plt.figure(figsize=(8,8))
plt.imshow(K_post)
plt.colorbar()
plt.show()

#plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
#much closer to the data, no?
plt.figure(figsize=(10,6))
draws=5000
show=50
storage=np.zeros((draws,GP_sample_amount))
#drawing lots of GPs (to get a good estimate of 5-95th percentile.
#Haven't found the analytical way to do it
for i in range(draws):
    storage[i,:]=np.matmul(K_post,normal(size=GP_sample_amount))+mean_post
    # plt.plot(x_samp,test[i,:], alpha=0.5, zorder=1)
#only showing some
for i in range(show):
    plt.plot(x_samp,storage[i,:], alpha=0.5, zorder=1)
#filling the 5-95th percentile with a transparent orange
storage=np.sort(storage,axis=0)
plt.plot(x_samp,mean_post, color="black", zorder=2)
plt.fill_between(x_samp, storage[int(0.95*draws),:], storage[int(0.05*draws),:], alpha=0.3, color='orange')
plt.plot(x_samp,np.matmul(K_post,np.ones(GP_sample_amount)*1.645/np.sqrt(GP_sample_amount))+mean_post, color="black", zorder=2)
plt.plot(x_samp,np.matmul(K_post,np.ones(GP_sample_amount)*-1.645/np.sqrt(GP_sample_amount))+mean_post, color="black", zorder=2)
plt.scatter(x_obs,y_obs,color="black",zorder=2)
#plt.ylim(ymin, ymax)
plt.xlabel("x (time)")
plt.ylabel("y (flux or something lol)")
plt.title("Gaussian Processes (Posterior)")
plt.show()

#negative log likelihood of the data given the current kernel parameters (as seen on page 19)
#(negative because scipy has a minimizer instead of a maximizer)
def nlogL(hyperparameters):
    n=len(y_obs)

    noise_I=np.square(measurement_noise)*np.identity(amount_of_measurements)
    K_obs=covariance(x_obs,x_obs,hyperparameters)+noise_I

    #goodness of fit term
    data_fit=-1/2*np.matmul(np.transpose(y_obs),np.matmul(np.linalg.inv(K_obs),y_obs))
    #complexity penalization term
    penalty=-1/2*np.log(np.linalg.det(K_obs))
    #normalization term (functionally useless)
    normalization=-n/2*np.log(2*np.pi)
    return -1*(data_fit+penalty+normalization)

#numerically maximize the likelihood to find the best hyperparameters
result=minimize(nlogL, hyperparameters)
# print(result) #uncomment this for more details
print(result.message)

#reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
hyperparameters=result.x

print("'Best-fit' hyperparameters")
print(hyperparameters)

#recalculating covariance matrices
K_samp=covariance(x_samp,x_samp,hyperparameters)
noise_I=np.square(measurement_noise)*np.identity(amount_of_measurements)
K_obs=covariance(x_obs,x_obs,hyperparameters)+noise_I
K_samp_obs=covariance(x_samp,x_obs,hyperparameters)
K_obs_samp=covariance(x_obs,x_samp,hyperparameters)

#mean of the conditional distribution p(f_samp|x_obs,t_obs,x_samp) (as defined on page 14)
mean_post=np.matmul(K_samp_obs,np.matmul(np.linalg.inv(K_obs),y_obs))
#cavariance matrix of the conditional distribution p(f_samp|x_obs,t_obs,x_samp) (as defined on page 14)
K_post=K_samp-np.matmul(K_samp_obs,np.matmul(np.linalg.inv(K_obs),K_obs_samp))

#plot the posterior covariance of the "most likely" posterior matrix
#(colors show how correlated points are to each other)
plt.figure(figsize=(8,8))
plt.imshow(K_post)
plt.colorbar()
plt.show()

#plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
#much closer to the data, no?
plt.figure(figsize=(10,6))
draws=5000
show=50
storage=np.zeros((draws,GP_sample_amount))
#drawing lots of GPs (to get a good estimate of 5-95th percentile.
#Haven't found the analytical way to do it
for i in range(draws):
    storage[i,:]=np.matmul(K_post,normal(size=GP_sample_amount))+mean_post
    # plt.plot(x_samp,test[i,:], alpha=0.5, zorder=1)
#only showing some
for i in range(show):
    plt.plot(x_samp,storage[i,:], alpha=0.5, zorder=1)
#filling the 5-95th percentile with a transparent orange
storage=np.sort(storage,axis=0)
plt.plot(x_samp,mean_post, color="black", zorder=2)
plt.fill_between(x_samp, storage[int(0.95*draws),:], storage[int(0.05*draws),:], alpha=0.3, color='orange')
#plt.plot(x_samp,np.matmul(K_post,np.ones(GP_sample_amount)*1.95)+mean_post, color="black", zorder=2)
#plt.plot(x_samp,np.matmul(K_post,np.ones(GP_sample_amount)*-1.95)+mean_post, color="black", zorder=2)
plt.scatter(x_obs,y_obs,color="black",zorder=2)
#plt.ylim(ymin, ymax)
plt.xlabel("x (time)")
plt.ylabel("y (flux or something lol)")
plt.title("Gaussian Processes (Posterior)")
plt.show()
