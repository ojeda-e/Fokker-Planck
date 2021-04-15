#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "Random123/philox.h"
#include "Random123/u01.h"
#include "histo.h"



typedef r123::Philox2x32 RNG; // un counter-based RNG

// check times
 #ifdef OMP
 #include "cpu_timer.h"
 #else
 #include "gpu_timer.h"
 #endif


/*Random number (gaussian distribution) - generated using philox*/
__device__
float box_muller(RNG::ctr_type r_philox)
{
	// transforma el philox number a dos uniformes en (0,1]
 	float u1 = u01_open_closed_32_53(r_philox[0]);
  float u2 = u01_open_closed_32_53(r_philox[1]);

  float r = sqrtf( -2.0*logf(u1) );
  float theta = 2.0*M_PI*u2;
	
  return r*sinf(theta);    			
}


#define Dt		0.001  // paso temporal
#define NROBINS		100   // numero de bins
#define NROPARTS	1000000 // numero de particulas	
#define SEED		98273614 // una semilla global 

// force
__device__ float Fuerza(float x)
{
	// return -d/dx [sinf(2 M_PI x) + 0.25 sinf(4 M_PI x)]
	return -M_PI*(2.0f*cosf(2.0f*M_PI*x) + cosf(4.0f*M_PI*x));
}


// this functor to evolve the dynamic of the brownian particle
// trun steps at temperature T
struct dinamica
{ // time steps
  /* iteration - histogram */
  unsigned int trun; // iterations in histogram 
  /* Iteraciones total */
  unsigned int t;    // total iterations ~ time 

  float T;	      // temperature

  /* fac --> sqrt ( 2*T*delta t )*/
  float fac;         // aux variable
  dinamica(float _T, unsigned int _trun, unsigned int _t):T(_T),trun(_trun),t(_t)
  {
	fac=sqrtf(2.0f*T*Dt);
  };	

  // tid as counter to identify the particle, and position x
  // update x
  __device__
  float operator()(unsigned int tid, float x)
  {
    // keys and counters 
    RNG philox; 	
    RNG::ctr_type c={{}};
    RNG::key_type k={{}};
    RNG::ctr_type r;

    // tid que es el contador
    c[1]=SEED;
    c[0]=tid+SEED;
    k[0]=tid; 

    //generate random seq - SEED
    

    for(unsigned int i = 0; i < trun; ++i){
      c[0]=i+t;
      r = philox(c, k);
      
      //gaussian generator
      float randGauss=box_muller(r);
      
      //dynamics equation
      x = x + Fuerza(x)*Dt+ fac*randGauss;
      
      //periodic boundary conditions
      if(x<0.0f) x+=1.0;  // + 1
      if(x>1.0f) x-=1.0;  // - 1   --> x in range (0,1]
    }
    return x; 	
  }
};



#include <omp.h>

// to avoid "thrust::"
using namespace thrust;	


int main(void)
{

  #ifdef OMP
  // print max num threads in version openMP
  std::cout << "#host OMP threads = " << omp_get_max_threads() << std::endl;
  cpu_timer timer;
  #else
  int card;
  cudaGetDevice(&card);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, card);
  std::cout << "\nDevice Selected " << card << " " << deviceProp.name << "\n";
  gpu_timer timer; 
  #endif	

  // declare vector X (particle #) single precision
  device_vector<float> X(NROPARTS);

  // Initialize:

  // (1) Origin all:
  thrust::fill(X.begin(), X.end(), 0.);

  // (2) Uniform in range x = [0,1):
  //thrust::sequence(X.begin(), X.end());

  //(3) Random
  //thrust::generate(X.begin(), X.end(), rand);

  //declare Histogram in device, single precision
  device_vector<float> Histogram(NROBINS);

  float Temp=0.75f; // temperature
  unsigned trun=10; // step - histograms
  unsigned tiempo=0; // time (abs)

  // save to file
  std::ofstream histout("histograms.dat");

  // counter
  timer.tic();

  for(unsigned n=0;n<1000;n++,tiempo+=trun)
  {
    dense_histogram_data_on_device(X, Histogram, float(0.0), float(1.0));
	print_histograma(Histogram, float(0.0), float(1.0), histout);

    transform(
              counting_iterator<int>(0),
              counting_iterator<int>(NROPARTS),
              X.begin(),
              X.begin(),
              dinamica(Temp,trun,tiempo)
    );

  timer.tac();
  std::cout  << timer.ms_elapsed << "ms \n";

  }
  return 0;
}
