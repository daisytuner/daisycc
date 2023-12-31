
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include "pnetcdf.h"
#include <chrono>

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

constexpr double pi        = 3.14159265358979323846264338327;   //Pi
constexpr double grav      = 9.8;                               //Gravitational acceleration (m / s^2)
constexpr double cp        = 1004.;                             //Specific heat of dry air at constant pressure
constexpr double cv        = 717.;                              //Specific heat of dry air at constant volume
constexpr double rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
constexpr double p0        = 1.e5;                              //Standard pressure at the surface in Pascals
constexpr double C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
constexpr double gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
constexpr double xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
constexpr double zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
constexpr double hv_beta   = 0.05;    //How strong to diffuse the solution: hv_beta \in [0:1]
constexpr double cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
constexpr double max_speed = 450;     //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
constexpr int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
constexpr int sten_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
constexpr int NUM_VARS = 4;           //Number of fluid state variables
constexpr int ID_DENS  = 0;           //index for density ("rho")
constexpr int ID_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
constexpr int ID_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
constexpr int ID_RHOT  = 3;           //index for density * potential temperature ("rho * theta")
constexpr int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
constexpr int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction
constexpr int DATA_SPEC_COLLISION       = 1;
constexpr int DATA_SPEC_THERMAL         = 2;
constexpr int DATA_SPEC_GRAVITY_WAVES   = 3;
constexpr int DATA_SPEC_DENSITY_CURRENT = 5;
constexpr int DATA_SPEC_INJECTION       = 6;

constexpr int nqpoints = 3;
constexpr double qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
constexpr double qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };

///////////////////////////////////////////////////////////////////////////////////////
// BEGIN USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////
//The x-direction length is twice as long as the z-direction length
//So, you'll want to have nx_glob be twice as large as nz_glob
int    constexpr nx_glob       = _NX;            //Number of total cells in the x-direction
int    constexpr nz_glob       = _NZ;            //Number of total cells in the z-direction
double constexpr sim_time      = _SIM_TIME;      //How many seconds to run the simulation
double constexpr output_freq   = _OUT_FREQ;      //How frequently to output data to file (in seconds)
int    constexpr data_spec_int = _DATA_SPEC;     //How to initialize the data
double constexpr dx            = xlen / nx_glob; // grid spacing in the x-direction
double constexpr dz            = zlen / nz_glob; // grid spacing in the x-direction
///////////////////////////////////////////////////////////////////////////////////////
// END USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////

int    constexpr nx       = _NX;            //Number of total cells in the x-direction
int    constexpr nz       = _NZ;            //Number of total cells in the z-direction

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double dt;                    //Model time step (seconds)
// int    nx, nz;                //Number of local grid cells in the x- and z- dimensions for this MPI task
int    nranks, myrank;        //Number of MPI ranks and my rank id
int    left_rank, right_rank; //MPI Rank IDs that exist to my left and right in the global domain
int    mainproc;            //Am I the main process (rank == 0)?

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double etime;                 //Elapsed model time
double output_counter;        //Helps determine when it's time to do output
int    num_out = 0;           //The number of outputs performed so far
int    direction_switch = 1;
double mass0, te0;            //Initial domain totals for mass and total energy  
double mass , te ;            //Domain totals for mass and total energy  

//How is this not in the standard?!
double dmin( double a , double b ) { if (a<b) {return a;} else {return b;} };

//Declaring the functions defined after "main"
void   init                 (
  int *argc ,
  char ***argv,
  double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
  double state_tmp[NUM_VARS][nz+2*hs][nx+2*hs] ,
  double flux[NUM_VARS][nz+1][nx+1] ,
  double tend[NUM_VARS][nz][nx] ,
  double hy_dens_cell[nz+2*hs] ,
  double hy_dens_theta_cell[nz+2*hs] ,
  double hy_dens_int[nz+1] ,
  double hy_dens_theta_int[nz+1] ,
  double hy_pressure_int[nz+1]
 );
void   finalize             ( );
void   injection            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   density_current      ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   gravity_waves        ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   thermal              ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   collision            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   hydro_const_theta    ( double z                   , double &r , double &t );
void   hydro_const_bvfreq   ( double z , double bv_freq0 , double &r , double &t );
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad );
void   output               (
    double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double etime,
    double hy_dens_cell[nz+2*hs] ,
    double hy_dens_theta_cell[nz+2*hs]
);
void   ncwrap               ( int ierr , int line );
void   perform_timestep     (
    double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double state_tmp[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double flux[NUM_VARS][nz+1][nx+1] ,
    double tend[NUM_VARS][nz][nx] ,
    double dt,
    double hy_dens_cell[nz+2*hs],
    double hy_dens_theta_cell[nz+2*hs],
    double hy_dens_int[nz+1] ,
    double hy_dens_theta_int[nz+1] ,
    double hy_pressure_int[nz+1]
);
void   semi_discrete_step   (
    double state_init[NUM_VARS][nz+2*hs][nx+2*hs]  ,
    double state_forcing[NUM_VARS][nz+2*hs][nx+2*hs]  ,
    double state_out[NUM_VARS][nz+2*hs][nx+2*hs]  ,
    double dt ,
    int dir ,
    double flux[NUM_VARS][nz+1][nx+1] ,
    double tend[NUM_VARS][nz][nx],
    double hy_dens_cell[nz+2*hs],
    double hy_dens_theta_cell[nz+2*hs],
    double hy_dens_int[nz+1] ,
    double hy_dens_theta_int[nz+1] ,
    double hy_pressure_int[nz+1]
);
void   compute_tendencies_x (
    double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double flux[NUM_VARS][nz+1][nx+1] ,
    double tend[NUM_VARS][nz][nx] ,
    double dt,
    double hy_dens_cell[nz+2*hs],
    double hy_dens_theta_cell[nz+2*hs] 
);
void   compute_tendencies_z (
    double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double flux[NUM_VARS][nz+1][nx+1] ,
    double tend[NUM_VARS][nz][nx] ,
    double dt,
    double hy_dens_int[nz+1] ,
    double hy_dens_theta_int[nz+1] ,
    double hy_pressure_int[nz+1]
);
void   set_halo_values_x    ( double state[NUM_VARS][nz+2*hs][nx+2*hs], double hy_dens_cell[nz+2*hs], double hy_dens_theta_cell[nz+2*hs] );
void   set_halo_values_z    ( double state[NUM_VARS][nz+2*hs][nx+2*hs], double hy_dens_cell[nz+2*hs] );
void   reductions           (
  double &mass ,
  double &te,
  double state[NUM_VARS][nz+2*hs][nx+2*hs],
  double hy_dens_cell[nz+2*hs] ,
  double hy_dens_theta_cell[nz+2*hs]
);

///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  //Declare arrays
  double (*state)[NUM_VARS][nz+2*hs][nx+2*hs];        //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  double (*state_tmp)[NUM_VARS][nz+2*hs][nx+2*hs];    //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  double (*flux)[NUM_VARS][nz+1][nx+1];               //Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
  double (*tend)[NUM_VARS][nz][nx];                   //Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)

  double (*hy_dens_cell)[nz+2*hs];         //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
  double (*hy_dens_theta_cell)[nz+2*hs];   //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
  double (*hy_dens_int)[nz+1];          //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
  double (*hy_dens_theta_int)[nz+1];    //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
  double (*hy_pressure_int)[nz+1];      //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

  //Allocate the model data
  state              = (double (*)[NUM_VARS][nz+2*hs][nx+2*hs]) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  state_tmp          = (double (*)[NUM_VARS][nz+2*hs][nx+2*hs]) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  flux               = (double (*)[NUM_VARS][nz+1][nx+1]) malloc( (nx+1)*(nz+1)*NUM_VARS*sizeof(double) );
  tend               = (double (*)[NUM_VARS][nz][nx]) malloc( nx*nz*NUM_VARS*sizeof(double) );
  hy_dens_cell       = (double (*)[nz+2*hs]) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_theta_cell = (double (*)[nz+2*hs]) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_int        = (double (*)[nz+1]) malloc( (nz+1)*sizeof(double) );
  hy_dens_theta_int  = (double (*)[nz+1]) malloc( (nz+1)*sizeof(double) );
  hy_pressure_int    = (double (*)[nz+1]) malloc( (nz+1)*sizeof(double) );

  init( 
    &argc , 
    &argv,
    *state,
    *state_tmp,
    *flux,
    *tend,
    *hy_dens_cell,
    *hy_dens_theta_cell,
    *hy_dens_int,
    *hy_dens_theta_int,
    *hy_pressure_int
  );

  //Initial reductions for mass, kinetic energy, and total energy
  reductions(mass0,te0,*state,*hy_dens_cell,*hy_dens_theta_cell);

  //Output the initial state
  output(*state,etime,*hy_dens_cell,*hy_dens_theta_cell);

  ////////////////////////////////////////////////////
  // MAIN TIME STEP LOOP
  ////////////////////////////////////////////////////
  auto t1 = std::chrono::steady_clock::now();
  while (etime < sim_time) {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt > sim_time) { dt = sim_time - etime; }
    //Perform a single time step
    perform_timestep(*state,*state_tmp,*flux,*tend,dt,*hy_dens_cell,*hy_dens_theta_cell,*hy_dens_int,*hy_dens_theta_int,*hy_pressure_int);
    //Inform the user
#ifndef NO_INFORM
    if (mainproc) { printf( "Elapsed Time: %lf / %lf\n", etime , sim_time ); }
#endif
    //Update the elapsed time and output counter
    etime = etime + dt;
    output_counter = output_counter + dt;
    //If it's time for output, reset the counter, and do output
    if (output_counter >= output_freq) {
      output_counter = output_counter - output_freq;
      output(*state,etime,*hy_dens_cell,*hy_dens_theta_cell);
    }
  }
  auto t2 = std::chrono::steady_clock::now();
  if (mainproc) {
    std::cout << "CPU Time: " << std::chrono::duration<double>(t2-t1).count() << " sec\n";
  }

  //Final reductions for mass, kinetic energy, and total energy
  reductions(mass,te,*state,*hy_dens_cell,*hy_dens_theta_cell);

  if (mainproc) {
    printf( "d_mass: %le\n" , (mass - mass0)/mass0 );
    printf( "d_te:   %le\n" , (te   - te0  )/te0   );
  }

  free( state );
  free( state_tmp );
  free( flux );
  free( tend );
  free( hy_dens_cell );
  free( hy_dens_theta_cell );
  free( hy_dens_int );
  free( hy_dens_theta_int );
  free( hy_pressure_int );

  finalize();
}


//Performs a single dimensionally split time step using a simple low-storage three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void perform_timestep(
    double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double state_tmp[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double flux[NUM_VARS][nz+1][nx+1] ,
    double tend[NUM_VARS][nz][nx] ,
    double dt,
    double hy_dens_cell[nz+2*hs],
    double hy_dens_theta_cell[nz+2*hs],
    double hy_dens_int[nz+1] ,
    double hy_dens_theta_int[nz+1] ,
    double hy_pressure_int[nz+1]
) {
  if (direction_switch) {
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
  } else {
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend , hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int );
  }
  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}


//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step(
    double state_init[NUM_VARS][nz+2*hs][nx+2*hs]  ,
    double state_forcing[NUM_VARS][nz+2*hs][nx+2*hs]  ,
    double state_out[NUM_VARS][nz+2*hs][nx+2*hs]  ,
    double dt ,
    int dir ,
    double flux[NUM_VARS][nz+1][nx+1] ,
    double tend[NUM_VARS][nz][nx],
    double hy_dens_cell[nz+2*hs],
    double hy_dens_theta_cell[nz+2*hs],
    double hy_dens_int[nz+1] ,
    double hy_dens_theta_int[nz+1] ,
    double hy_pressure_int[nz+1]
) {
  int i, k, ll;
  //  int inds, indt, indw;
  double x, z, wpert, dist, x0, z0, xrad, zrad, amp;
  if        (dir == DIR_X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    set_halo_values_x(state_forcing, hy_dens_cell, hy_dens_theta_cell);
    //Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(state_forcing,flux,tend,dt,hy_dens_cell,hy_dens_theta_cell);
  } else if (dir == DIR_Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    set_halo_values_z(state_forcing, hy_dens_cell);
    //Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(state_forcing,flux,tend,dt,hy_dens_int,hy_dens_theta_int,hy_pressure_int);
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Apply the tendencies to the fluid state
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
          x = (0 + i+0.5)*dx;
          z = (0 + k+0.5)*dz;
          // Using sample_ellipse_cosine requires "acc routine" in OpenACC and "declare target" in OpenMP offload
          // Neither of these are particularly well supported. So I'm manually inlining here
          wpert = sample_ellipse_cosine( x,z , 0.01 , xlen/8,1000., 500.,500. );
          // {
          //   x0   = xlen/8;
          //   z0   = 1000;
          //   xrad = 500;
          //   zrad = 500;
          //   amp  = 0.01;
          //   //Compute distance from bubble center
          //   dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
          //   //If the distance from bubble center is less than the radius, create a cos**2 profile
          //   if (dist <= pi / 2.) {
          //      wpert = amp * pow(cos(dist),2.);
          //   } else {
          //      wpert = 0.;
          //   }
          // }
          //  indw = ID_WMOM*nz*nx + k*nx + i;
          tend[ID_WMOM][k][i] += wpert*hy_dens_cell[hs+k];
        }
        // inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        // indt = ll*nz*nx + k*nx + i;
        state_out[ll][k+hs][i+hs] = state_init[ll][k+hs][i+hs] + dt * tend[ll][k][i];
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_x(
    double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double flux[NUM_VARS][nz+1][nx+1] ,
    double tend[NUM_VARS][nz][nx] ,
    double dt,
    double hy_dens_cell[nz+2*hs],
    double hy_dens_theta_cell[nz+2*hs] 
) {
  int    i,k,ll,s;
  // int inds,indf1,indf2,indt;
  double r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coefficient
  hv_coef = -hv_beta * dx / (16*dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nz; k++) {
    for (i=0; i<nx+1; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s < sten_size; s++) {
          // inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+s;
          stencil[s] = state[ll][k+hs][i+s];
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_cell[k+hs];
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = ( vals[ID_RHOT] + hy_dens_theta_cell[k+hs] ) / r;
      p = C0*pow((r*t),gamm);

      //Compute the flux vector
      // flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u     - hv_coef*d3_vals[ID_DENS];
      // flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
      // flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*w   - hv_coef*d3_vals[ID_WMOM];
      // flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*t   - hv_coef*d3_vals[ID_RHOT];
      flux[ID_DENS][k][i] = r*u     - hv_coef*d3_vals[ID_DENS];
      flux[ID_UMOM][k][i] = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
      flux[ID_WMOM][k][i] = r*u*w   - hv_coef*d3_vals[ID_WMOM];
      flux[ID_RHOT][k][i] = r*u*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        // indt  = ll* nz   * nx    + k* nx    + i  ;
        // indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i  ;
        // indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
        tend[ll][k][i] = -( flux[ll][k][i+1] - flux[ll][k][i] ) / dx;
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_z(
    double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
    double flux[NUM_VARS][nz+1][nx+1] ,
    double tend[NUM_VARS][nz][nx] ,
    double dt,
    double hy_dens_int[nz+1] ,
    double hy_dens_theta_int[nz+1] ,
    double hy_pressure_int[nz+1]
) {
  int    i,k,ll,s;
  //int inds, indf1, indf2, indt;
  double r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coefficient
  hv_coef = -hv_beta * dz / (16*dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nz+1; k++) {
    for (i=0; i<nx; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s<sten_size; s++) {
          // inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
          stencil[s] = state[ll][k+s][i+hs];
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative of the state
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_int[k];
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r;
      p = C0*pow((r*t),gamm) - hy_pressure_int[k];

      //Enforce vertical boundary condition and exact mass conservation
      // if (k == 0 || k == nz) {
      //   w                = 0;
      //   d3_vals[ID_DENS] = 0;
      // }

      //Compute the flux vector with hyperviscosity
      //   flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - hv_coef*d3_vals[ID_DENS];
      //   flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
      //   flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
      //   flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
      flux[ID_DENS][k][i] = r*w     - hv_coef*d3_vals[ID_DENS];
      flux[ID_UMOM][k][i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
      flux[ID_WMOM][k][i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
      flux[ID_RHOT][k][i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  k = 0;
  for (i=0; i<nx; i++) {
    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (ll=0; ll<NUM_VARS; ll++) {
      for (s=0; s<sten_size; s++) {
        // inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
        stencil[s] = state[ll][k+s][i+hs];
      }
      //Fourth-order-accurate interpolation of the state
      vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
      //First-order-accurate interpolation of the third spatial derivative of the state
      d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    r = vals[ID_DENS] + hy_dens_int[k];
    u = vals[ID_UMOM] / r;
    w = vals[ID_WMOM] / r;
    t = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r;
    p = C0*pow((r*t),gamm) - hy_pressure_int[k];

    //Enforce vertical boundary condition and exact mass conservation
    w                = 0;
    d3_vals[ID_DENS] = 0;

    //Compute the flux vector with hyperviscosity
    //   flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - hv_coef*d3_vals[ID_DENS];
    //   flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
    //   flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
    //   flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    flux[ID_DENS][k][i] = r*w     - hv_coef*d3_vals[ID_DENS];
    flux[ID_UMOM][k][i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
    flux[ID_WMOM][k][i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
    flux[ID_RHOT][k][i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
  }

  k = nz;
  for (i=0; i<nx; i++) {
    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (ll=0; ll<NUM_VARS; ll++) {
      for (s=0; s<sten_size; s++) {
        // inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
        stencil[s] = state[ll][k+s][i+hs];
      }
      //Fourth-order-accurate interpolation of the state
      vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
      //First-order-accurate interpolation of the third spatial derivative of the state
      d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    r = vals[ID_DENS] + hy_dens_int[k];
    u = vals[ID_UMOM] / r;
    w = vals[ID_WMOM] / r;
    t = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r;
    p = C0*pow((r*t),gamm) - hy_pressure_int[k];

    //Enforce vertical boundary condition and exact mass conservation
    w                = 0;
    d3_vals[ID_DENS] = 0;

    //Compute the flux vector with hyperviscosity
    //   flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - hv_coef*d3_vals[ID_DENS];
    //   flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
    //   flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
    //   flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    flux[ID_DENS][k][i] = r*w     - hv_coef*d3_vals[ID_DENS];
    flux[ID_UMOM][k][i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
    flux[ID_WMOM][k][i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
    flux[ID_RHOT][k][i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        // indt  = ll* nz   * nx    + k* nx    + i  ;
        // indf1 = ll*(nz+1)*(nx+1) + (k  )*(nx+1) + i;
        // indf2 = ll*(nz+1)*(nx+1) + (k+1)*(nx+1) + i;
        tend[ll][k][i] = -( flux[ll][k+1][i] - flux[ll][k][i] ) / dz;
        if (ll == ID_WMOM) {
          // inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
          tend[ll][k][i] = tend[ll][k][i] - state[ID_DENS][k+hs][i+hs]*grav;
        }
      }
    }
  }
}



//Set this MPI task's halo values in the x-direction. This routine will require MPI
void set_halo_values_x( double state[NUM_VARS][nz+2*hs][nx+2*hs], double hy_dens_cell[nz+2*hs], double hy_dens_theta_cell[nz+2*hs] ) {
  int k, ll, i;
  // int ind_r, ind_u, ind_t;
  double z;
  ////////////////////////////////////////////////////////////////////////
  // TODO: EXCHANGE HALO VALUES WITH NEIGHBORING MPI TASKS
  // (1) give    state(1:hs,1:nz,1:NUM_VARS)       to   my left  neighbor
  // (2) receive state(1-hs:0,1:nz,1:NUM_VARS)     from my left  neighbor
  // (3) give    state(nx-hs+1:nx,1:nz,1:NUM_VARS) to   my right neighbor
  // (4) receive state(nx+1:nx+hs,1:nz,1:NUM_VARS) from my right neighbor
  ////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////
  // DELETE THE SERIAL CODE BELOW AND REPLACE WITH MPI
  //////////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      state[ll][k+hs][0      ] = state[ll][k+hs][nx+hs-2];
      state[ll][k+hs][1      ] = state[ll][k+hs][nx+hs-1];
      state[ll][k+hs][nx+hs  ] = state[ll][k+hs][hs     ];
      state[ll][k+hs][nx+hs+1] = state[ll][k+hs][hs+1   ];
    }
  }
  ////////////////////////////////////////////////////

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (myrank == 0) {
      for (k=0; k<nz; k++) {
        for (i=0; i<hs; i++) {
          z = (0 + k+0.5)*dz;
          if (fabs(z-3*zlen/4) <= zlen/16) {
            // ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            // ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            // ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            state[ID_UMOM][k+hs][i] = (state[ID_DENS][k+hs][i]+hy_dens_cell[k+hs]) * 50.;
            state[ID_RHOT][k+hs][i] = (state[ID_DENS][k+hs][i]+hy_dens_cell[k+hs]) * 298. - hy_dens_theta_cell[k+hs];
          }
        }
      }
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void set_halo_values_z( double state[NUM_VARS][nz+2*hs][nx+2*hs], double hy_dens_cell[nz+2*hs] ) {
  int          i, ll;
  const double mnt_width = xlen/8;
  double       x, xloc, mnt_deriv;
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (i=0; i<nx+2*hs; i++) {
      if (ll == ID_WMOM) {
        state[ll][(0      )][i] = 0.;
        state[ll][(1      )][i] = 0.;
        state[ll][(nz+hs  )][i] = 0.;
        state[ll][(nz+hs+1)][i] = 0.;
      } else if (ll == ID_UMOM) {
        state[ll][(0      )][i] = state[ll][(hs     )][i] / hy_dens_cell[hs     ] * hy_dens_cell[0      ];
        state[ll][(1      )][i] = state[ll][(hs     )][i] / hy_dens_cell[hs     ] * hy_dens_cell[1      ];
        state[ll][(nz+hs  )][i] = state[ll][(nz+hs-1)][i] / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs  ];
        state[ll][(nz+hs+1)][i] = state[ll][(nz+hs-1)][i] / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs+1];
      } else {
        state[ll][(0      )][i] = state[ll][(hs     )][i];
        state[ll][(1      )][i] = state[ll][(hs     )][i];
        state[ll][(nz+hs  )][i] = state[ll][(nz+hs-1)][i];
        state[ll][(nz+hs+1)][i] = state[ll][(nz+hs-1)][i];
      }
    }
  }
}


void init(
  int *argc ,
  char ***argv ,
  double state[NUM_VARS][nz+2*hs][nx+2*hs] ,
  double state_tmp[NUM_VARS][nz+2*hs][nx+2*hs] ,
  double flux[NUM_VARS][nz+1][nx+1] ,
  double tend[NUM_VARS][nz][nx] ,
  double hy_dens_cell[nz+2*hs] ,
  double hy_dens_theta_cell[nz+2*hs] ,
  double hy_dens_int[nz+1] ,
  double hy_dens_theta_int[nz+1] ,
  double hy_pressure_int[nz+1]
) {
  int    i, k, ii, kk, ll, ierr, inds;
  double x, z, r, u, w, t, hr, ht;

  ierr = MPI_Init(argc,argv);

  /////////////////////////////////////////////////////////////
  // BEGIN MPI DUMMY SECTION
  // TODO: (1) GET NUMBER OF MPI RANKS
  //       (2) GET MY MPI RANK ID (RANKS ARE ZERO-BASED INDEX)
  //       (3) COMPUTE MY BEGINNING "I" INDEX (1-based index)
  //       (4) COMPUTE HOW MANY X-DIRECTION CELLS MY RANK HAS
  //       (5) FIND MY LEFT AND RIGHT NEIGHBORING RANK IDs
  /////////////////////////////////////////////////////////////
  nranks = 1;
  myrank = 0;
  // nx = nx_glob;
  left_rank = 0;
  right_rank = 0;
  //////////////////////////////////////////////
  // END MPI DUMMY SECTION
  //////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  // nz = nz_glob;
  mainproc = (myrank == 0);

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = dmin(dx,dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  //If I'm the main process in MPI, display some grid information
  if (mainproc) {
    printf( "nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    printf( "dx,dz: %lf %lf\n",dx,dz);
    printf( "dt: %lf\n",dt);
  }
  //Want to make sure this info is displayed before further output
  ierr = MPI_Barrier(MPI_COMM_WORLD);

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (k=0; k<nz+2*hs; k++) {
    for (i=0; i<nx+2*hs; i++) {
      //Initialize the state to zero
      for (ll=0; ll<NUM_VARS; ll++) {
        // inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state[ll][k][i] = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (0 + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (0 + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

          //Store into the fluid state array
          // inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[ID_DENS][k][i] = state[ID_DENS][k][i] + r * qweights[ii]*qweights[kk];
          // inds = ID_UMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[ID_UMOM][k][i] = state[ID_UMOM][k][i] + (r+hr)*u * qweights[ii]*qweights[kk];
          // inds = ID_WMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[ID_WMOM][k][i] = state[ID_WMOM][k][i] + (r+hr)*w * qweights[ii]*qweights[kk];
          // inds = ID_RHOT*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[ID_RHOT][k][i] = state[ID_RHOT][k][i] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        // inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state_tmp[ll][k][i] = state[ll][k][i];
      }
    }
  }
  //Compute the hydrostatic background state over vertical cell averages
  for (k=0; k<nz+2*hs; k++) {
    hy_dens_cell      [k] = 0.;
    hy_dens_theta_cell[k] = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (0 + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
      hy_dens_cell      [k] = hy_dens_cell      [k] + hr    * qweights[kk];
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  //Compute the hydrostatic background state at vertical cell interfaces
  for (k=0; k<nz+1; k++) {
    double z = (0 + k)*dz;
    if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
    hy_dens_int      [k] = hr;
    hy_dens_theta_int[k] = hr*ht;
    hy_pressure_int  [k] = C0*pow((hr*ht),gamm);
  }
}

//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void injection( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
}


//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void density_current( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.);
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void gravity_waves( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_bvfreq(z,0.02,hr,ht);
  r = 0.;
  t = 0.;
  u = 15.;
  w = 0.;
}


//Rising thermal
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void thermal( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.);
}


//Colliding thermals
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void collision( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t = t + sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}


//Establish hydrostatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_theta( double z , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p,exner,rt;
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}


//Establish hydrostatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_bvfreq( double z , double bv_freq0 , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  double wpert = amp * pow(cos(dist),2.);
  return (dist <= pi / 2.) ? wpert  : 0.0;
}


//Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
//The file I/O uses parallel-netcdf, the only external library required for this mini-app.
//If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
void output( double state[NUM_VARS][nz+2*hs][nx+2*hs] , double etime, double hy_dens_cell[nz+2*hs] , double hy_dens_theta_cell[nz+2*hs] ) {
  int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid, dimids[3];
  int i, k, ind_r, ind_u, ind_w, ind_t;
  MPI_Offset st1[1], ct1[1], st3[3], ct3[3];
  //Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
  double *dens, *uwnd, *wwnd, *theta;
  double *etimearr;
  //Inform the user
  if (mainproc) { printf("*** OUTPUT ***\n"); }
  //Allocate some (big) temp arrays
  dens     = (double *) malloc(nx*nz*sizeof(double));
  uwnd     = (double *) malloc(nx*nz*sizeof(double));
  wwnd     = (double *) malloc(nx*nz*sizeof(double));
  theta    = (double *) malloc(nx*nz*sizeof(double));
  etimearr = (double *) malloc(1    *sizeof(double));

  //If the elapsed time is zero, create the file. Otherwise, open the file
  if (etime == 0) {
    //Create the file
    ncwrap( ncmpi_create( MPI_COMM_WORLD , "output.nc" , NC_CLOBBER , MPI_INFO_NULL , &ncid ) , __LINE__ );
    //Create the dimensions
    ncwrap( ncmpi_def_dim( ncid , "t" , (MPI_Offset) NC_UNLIMITED , &t_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "x" , (MPI_Offset) nx_glob      , &x_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "z" , (MPI_Offset) nz_glob      , &z_dimid ) , __LINE__ );
    //Create the variables
    dimids[0] = t_dimid;
    ncwrap( ncmpi_def_var( ncid , "t"     , NC_DOUBLE , 1 , dimids ,     &t_varid ) , __LINE__ );
    dimids[0] = t_dimid; dimids[1] = z_dimid; dimids[2] = x_dimid;
    ncwrap( ncmpi_def_var( ncid , "dens"  , NC_DOUBLE , 3 , dimids ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "uwnd"  , NC_DOUBLE , 3 , dimids ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "wwnd"  , NC_DOUBLE , 3 , dimids ,  &wwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "theta" , NC_DOUBLE , 3 , dimids , &theta_varid ) , __LINE__ );
    //End "define" mode
    ncwrap( ncmpi_enddef( ncid ) , __LINE__ );
  } else {
    //Open the file
    ncwrap( ncmpi_open( MPI_COMM_WORLD , "output.nc" , NC_WRITE , MPI_INFO_NULL , &ncid ) , __LINE__ );
    //Get the variable IDs
    ncwrap( ncmpi_inq_varid( ncid , "dens"  ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "uwnd"  ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "wwnd"  ,  &wwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "theta" , &theta_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "t"     ,     &t_varid ) , __LINE__ );
  }

  //Store perturbed values in the temp arrays for output
  for (k=0; k<nz; k++) {
    for (i=0; i<nx; i++) {
      // ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      // ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      // ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      // ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      dens [k*nx+i] = state[ID_DENS][k+hs][i+hs];
      uwnd [k*nx+i] = state[ID_UMOM][k+hs][i+hs] / ( hy_dens_cell[k+hs] + state[ID_DENS][k+hs][i+hs] );
      wwnd [k*nx+i] = state[ID_WMOM][k+hs][i+hs] / ( hy_dens_cell[k+hs] + state[ID_DENS][k+hs][i+hs] );
      theta[k*nx+i] = ( state[ID_RHOT][k+hs][i+hs] + hy_dens_theta_cell[k+hs] ) / ( hy_dens_cell[k+hs] + state[ID_DENS][k+hs][i+hs] ) - hy_dens_theta_cell[k+hs] / hy_dens_cell[k+hs];
    }
  }

  //Write the grid data to file with all the processes writing collectively
  st3[0] = num_out; st3[1] = 0; st3[2] = 0;
  ct3[0] = 1      ; ct3[1] = nz   ; ct3[2] = nx   ;
  ncwrap( ncmpi_put_vara_double_all( ncid ,  dens_varid , st3 , ct3 , dens  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  uwnd_varid , st3 , ct3 , uwnd  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  wwnd_varid , st3 , ct3 , wwnd  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid , theta_varid , st3 , ct3 , theta ) , __LINE__ );

  //Only the main process needs to write the elapsed time
  //Begin "independent" write mode
  ncwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
  //write elapsed time to file
  if (mainproc) {
    st1[0] = num_out;
    ct1[0] = 1;
    etimearr[0] = etime; ncwrap( ncmpi_put_vara_double( ncid , t_varid , st1 , ct1 , etimearr ) , __LINE__ );
  }
  //End "independent" write mode
  ncwrap( ncmpi_end_indep_data(ncid) , __LINE__ );

  //Close the file
  ncwrap( ncmpi_close(ncid) , __LINE__ );

  //Increment the number of outputs
  num_out = num_out + 1;

  //Deallocate the temp arrays
  free( dens     );
  free( uwnd     );
  free( wwnd     );
  free( theta    );
  free( etimearr );
}


//Error reporting routine for the PNetCDF I/O
void ncwrap( int ierr , int line ) {
  if (ierr != NC_NOERR) {
    printf("NetCDF Error at line: %d\n", line);
    printf("%s\n",ncmpi_strerror(ierr));
    exit(-1);
  }
}


void finalize() {
  int ierr;
  ierr = MPI_Finalize();
}


//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
void reductions(
  double &mass ,
  double &te,
  double state[NUM_VARS][nz+2*hs][nx+2*hs],
  double hy_dens_cell[nz+2*hs] ,
  double hy_dens_theta_cell[nz+2*hs]
) {
  mass = 0;
  te   = 0;
  for (int k=0; k<nz; k++) {
    for (int i=0; i<nx; i++) {
      // int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      // int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      // int ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      // int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      double r  =   state[ID_DENS][k+hs][i+hs] + hy_dens_cell[hs+k];             // Density
      double u  =   state[ID_UMOM][k+hs][i+hs] / r;                              // U-wind
      double w  =   state[ID_WMOM][k+hs][i+hs] / r;                              // W-wind
      double th = ( state[ID_RHOT][k+hs][i+hs] + hy_dens_theta_cell[hs+k] ) / r; // Potential Temperature (theta)
      double p  = C0*pow(r*th,gamm);                               // Pressure
      double t  = th / pow(p0/p,rd/cp);                            // Temperature
      double ke = r*(u*u+w*w);                                     // Kinetic Energy
      double ie = r*cv*t;                                          // Internal Energy
      mass += r        *dx*dz; // Accumulate domain mass
      te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    }
  }
  double glob[2], loc[2];
  loc[0] = mass;
  loc[1] = te;
  int ierr = MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  mass = glob[0];
  te   = glob[1];
}
