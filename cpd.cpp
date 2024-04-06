//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//  \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//  spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../orbital_advection/orbital_advection.hpp"

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real PoverR(const Real rad, const Real phi, const Real z);
//Real PreessProfile(const Real rad, const Real phi, const Real z);
void VelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3);
Real RotOrbitalVelocity(OrbitalAdvection *porb, Real x_, Real y_, Real z_);
Real RotOrbitalVelocity_r(OrbitalAdvection *porb, Real x_, Real y_, Real z_);
Real RotOrbitalVelocity_t(OrbitalAdvection *porb, Real x_, Real y_, Real z_);
void SubtractionOrbitalVelocity(OrbitalAdvection *porb, Coordinates *pco,
                                Real &v1, Real &v2, Real &v3, int i, int j, int k);

void AlphaVis(HydroDiffusion *phdif, MeshBlock *pmb,const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,int is, int ie, int js, int je, int ks, int ke);
// problem parameters which are useful to make global to this file
static Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, omegarot, Omega0, sound_speed, luminosity, r_radiation, gamma_rho;
static Real alpha, alphaa, aslope;
static int visflag;
static Real tcool;
static Real dfloor;
static Real rindamp, routdamp;
static Real insert_time;
static Real gauss_width;
static Real mass_hill, Rh, L_hill;
//Real voluume;
//----------------------------------------
// class for planetary system including mass, position, velocity

class PlanetarySystem
{
public:
  int np;
  int ind;
  Real rsoft2;
  double *mass;
  double *massset;
  double *xp, *yp, *zp;         // position in Cartesian coord.
  PlanetarySystem(int np);
  ~PlanetarySystem();
public:
  void orbit(double dt);      // circular planetary orbit
};

//------------------------------------------
// constructor for planetary system for np planets

PlanetarySystem::PlanetarySystem(int np0)
{
  np   = np0;
  ind  = 1;
  rsoft2 = 0.0;
  mass = new double[np];
  massset = new double[np];
  xp   = new double[np];
  yp   = new double[np];
  zp   = new double[np];
};

//---------------------------------------------
// destructor for planetary system

PlanetarySystem::~PlanetarySystem()
{
  delete[] mass;
  delete[] massset;
  delete[] xp;
  delete[] yp;
  delete[] zp;
};


// Planet Potential
Real grav_pot_car_btoa(const Real xca, const Real yca, const Real zca,
        const Real xcb, const Real ycb, const Real zcb, const Real gb, const Real rsoft2);
Real grav_pot_car_ind(const Real xca, const Real yca, const Real zca,
        const Real xpp, const Real ypp, const Real zpp, const Real gmp);
// Force on the planet
Real PlanetForce(MeshBlock *pmb, int iout);

// planetary system
static PlanetarySystem *psys;

/*void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);*/

void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);

void Cooling(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

// Functions for Planetary Source terms
void PlanetarySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

// User-defined boundary conditions for disk simulations

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    gm0 = pin->GetOrAddReal("problem","GMS",0.0);
  }
  r0 = pin->GetOrAddReal("problem","r0",1.0);
  omegarot = pin->GetOrAddReal("problem","omegarot",0.0);
  // Omega0 = pin->GetOrAddReal("problem","Omega0",0.0); 
  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);// - commented by Sabina

  if(Omega0!=0.0&&omegarot!=0.0){
    std::stringstream msg;
    msg << "omegarot and Omega0 cannot be non-zero at the same tiime"<<std::endl;
    ATHENA_ERROR(msg);
  }

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // viscosity
  alpha = pin->GetOrAddReal("problem","nu_iso",0.0);
  alphaa = pin->GetOrAddReal("problem","nu_aniso",0.0);
  aslope = pin->GetOrAddReal("problem","aslope",0.0); 
  visflag = pin->GetOrAddInteger("problem","visflag",0);

  // initial Gaussian
  gauss_width =  pin->GetOrAddReal("problem","gauss_width",0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
    gamma_rho = pin->GetOrAddReal("problem","gamma_rho", 1.4);
    tcool = pin->GetOrAddReal("problem","tcool",0.0);
    sound_speed = pin->GetReal("hydro","iso_sound_speed"); //added by Sabina
    //std::cout<<"Sabina, IT IS NON-BAROTROPIC"<<std::endl;  // Sabina needs to print
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
    sound_speed = pin->GetReal("hydro","iso_sound_speed");
    pslope = 0.0;
    // std::cout << "Sabina, it is BAROTROPIC"<<std::endl;
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  // Get boundary condition parameters
  rindamp = pin->GetOrAddReal("problem","rindamp",0.0);
  routdamp = pin->GetOrAddReal("problem","routdamp",HUGE_NUMBER);

  insert_time= pin->GetOrAddReal("problem","insert_time",5.0);
  luminosity = pin->GetOrAddReal("problem", "luminosity", 0.0); // luminosity 
  r_radiation = pin->GetOrAddReal("problem", "r_radiation", 0.1); // radius over which to deposit energy
  // set up the planetary system
  Real np = pin->GetOrAddInteger("planets","np",0);
  psys = new PlanetarySystem(np);
  psys->ind = pin->GetOrAddInteger("planets","ind",1);
  psys->rsoft2 = pin->GetOrAddReal("planets","rsoft2",0.0);

  // set initial planet properties
  for(int ip=0; ip<psys->np; ++ip){
    char pname[10];
    sprintf(pname,"mass%d",ip);
    psys->massset[ip]=pin->GetOrAddReal("planets",pname,0.0);
    psys->mass[ip]=0.0;
    sprintf(pname,"x%d",ip);
    psys->xp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"y%d",ip);
    psys->yp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"z%d",ip);
    psys->zp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    /*std::cout<<pname<<std::endl;
    std::cout<<"ip is "<<ip<<std::endl;
    std::cout<<"xp[ip] is "<<psys->xp[ip]<<std::endl;
    std::cout<<"yp[ip] is "<<psys->yp[ip]<<std::endl;
    std::cout<<"zp[ip] is "<<psys->zp[ip]<<std::endl;*/
  }

  // Enroll Fargo function
  if(omegarot!=0){
    EnrollOrbitalVelocity(RotOrbitalVelocity);
    EnrollOrbitalVelocityDerivative(0, RotOrbitalVelocity_r);
    EnrollOrbitalVelocityDerivative(1, RotOrbitalVelocity_t);
  }

  // enroll planetary potential
  EnrollUserExplicitSourceFunction(AllSourceTerms);
  AllocateUserHistoryOutput(15);
  EnrollUserHistoryOutput(0, PlanetForce, "fr");
  EnrollUserHistoryOutput(1, PlanetForce, "ft");
  EnrollUserHistoryOutput(2, PlanetForce, "fp");
  EnrollUserHistoryOutput(3, PlanetForce, "fxpp");
  EnrollUserHistoryOutput(4, PlanetForce, "fypp");
  EnrollUserHistoryOutput(5, PlanetForce, "fzpp");
  EnrollUserHistoryOutput(6, PlanetForce, "torque");
  EnrollUserHistoryOutput(7, PlanetForce, "xpp");
  EnrollUserHistoryOutput(8, PlanetForce, "ypp");
  EnrollUserHistoryOutput(9, PlanetForce, "zpp");
  EnrollUserHistoryOutput(10, PlanetForce, "rpp");
  EnrollUserHistoryOutput(11, PlanetForce, "tpp");
  EnrollUserHistoryOutput(12, PlanetForce, "ppp");
  EnrollUserHistoryOutput(13, PlanetForce, "mp");
  EnrollUserHistoryOutput(14, PlanetForce, "vol");
  // EnrollUserHistoryOutput(15, PlanetForce, "Rh");
  // EnrollUserHistoryOutput(16, PlanetForce, "mhill");
  // EnrollUserHistoryOutput(17, PlanetForce, "Lhill");

  EnrollViscosityCoefficient(AlphaVis);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  //std::cout << "Sabina, haven't reached if yet"<<std::endl; // Sabina needs to print
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        phydro->u(IDN,k,j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        
        if(porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(porb,pcoord,v1,v2,v3,i,j,k);
          // std::cout << "Sabina, ORBITAL ADVECTION IS DEFINED"<<std::endl; // Sabina needs to print
        }

        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
        phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          //Real p_over_r = sound_speed*sound_speed;
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          /*std::cout << "the energy is: " <<phydro->u(IEN,k,PI,i) <<std::endl; // Sabina needs to print
          std::cout << "the density is: " <<phydro->u(IDN,k,PI,i) <<std::endl; // Sabina needs to print
          std::cout << "the adiabatic pressure is: " <<phydro->u(IEN,k,PI,i)*(gamma_gas - 1.0) <<std::endl; // Sabina needs to print
          std::cout << "the isothermal pressure is: " <<phydro->u(IDN,k,PI,i)*iso_sound_speed*iso_sound_speed <<std::endl; // Sabina needs to print*/
        }
      }
    }
  }


  return;
}

//----------------------------------------------------------------------------------------
//!\f transform to cylindrical coordinate

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::fabs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(i);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \f  computes density in cylindrical coordinates

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS){ 
  p_over_r = PoverR(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  den = denmid*exp(-z*z/(2*sound_speed*sound_speed)) * (1 - exp(-pow(rad - r0, 2) / pow(0.1 * r0, 2)));
}
else{
  Real diff = std::pow(rad/r0,dslope*(gamma_rho-1))-gm0*(gamma_rho-1)/(sound_speed*sound_speed)*(1./rad-1./std::sqrt(SQR(rad)+SQR(z)));
  // den = rho0*std::pow(diff,(1/(gamma_gas-1)));
  // adding gaussian gap
  den = rho0*std::pow(diff,(1/(gamma_rho-1))) * (1 - exp(-pow(rad - r0, 2) / pow(0.1 * r0, 2)));
}
  if(std::isnan(den) == 1){
    den = dfloor;
  }

  //std::cout << "the density is "<< std::max(den,dfloor) <<std::endl;
  return std::max(den,dfloor);
}


//----------------------------------------------------------------------------------------
//! \f  computes isentropic pressure

/*Real PreessProfile(const Real rad, const Real phi, const Real z) {
  Real pre;
  Real den = DenProfileCyl(rad,phi,z);
  pre = sound_speed*sound_speed*rho0*std::pow(den,gamma_gas)/std::pow(rho0,gamma_gas)/gamma_gas;
  return pre;
}*/

//----------------------------------------------------------------------------------------
//! \f  computes pressure/density in cylindrical coordinates

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  Real den = DenProfileCyl(rad,phi,z);
  //poverr = p0_over_r0*std::pow(rad/r0, pslope);
  poverr = sound_speed*sound_speed*std::pow(den,gamma_gas-1)/gamma_gas;
  //std::cout << "the pres_over_r is "<< poverr <<std::endl;
  return poverr;
}

//----------------------------------------------------------------------------------------
//! \f  computes rotational velocity in cylindrical coordinates

void VelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3) {
  Real p_over_r = PoverR(rad, phi, z);
  /*Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel);*/

  Real den, vel;
  den = DenProfileCyl(rad,phi,z);
  Real gradpre = sound_speed*sound_speed*dslope*std::pow(rho0,1+gamma_gas-1*(dslope-1))*std::pow(rad,dslope*gamma_gas-1);
  Real diff = gradpre+den*gm0/(rad*rad);
  vel = std::sqrt(diff/(den*rad));

  if ( std::isnan(vel) == 1) {
    vel = std::sqrt((den*gm0/(rad*rad))/(den*rad));
  } 
  

  vel = vel*rad;
  //std::cout << "the velocity is "<< vel <<std::endl;

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1=0.0;
    v2=vel;
    v3=0.0;
    if(omegarot!=0.0) v2-=omegarot*rad;
    if(Omega0!=0.0) v2-=Omega0*rad;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1=0.0;
    v2=0.0;
    v3=vel;
    if(omegarot!=0.0) v3-=omegarot*rad;
    if(Omega0!=0.0) v3-=Omega0*rad;
  }
  return;
}

//-----------------------------------------------------------------------
//! \f fargo scheme to substract orbital velocity from v2

void SubtractionOrbitalVelocity(OrbitalAdvection *porb, Coordinates *pco,
                                Real &v1, Real &v2, Real &v3, int i, int j, int k) {
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  Real x1 = pco->x1v(i);
  Real x2 = pco->x2v(j);
  Real x3 = pco->x3v(k);
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v2 -= vK(porb,x1,x2,x3);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v3 -= vK(porb,x1,x2,x3);
  }
}

Real RotOrbitalVelocity(OrbitalAdvection *porb, Real x_, Real y_, Real z_){
  return std::sqrt(gm0/(x_*std::sin(y_)))-omegarot*x_*std::sin(y_);
//  return std::sqrt(gm0/x_)-omegarot*std::sin(y_)*x_;
}

Real RotOrbitalVelocity_r(OrbitalAdvection *porb, Real x_, Real y_, Real z_){
  return -0.5*std::sqrt(gm0/(std::sin(y_)*x_*x_*x_))-omegarot*std::sin(y_);
//  return -0.5*std::sqrt(gm0/x_)/x_-omegarot*std::sin(y_);
}

Real RotOrbitalVelocity_t(OrbitalAdvection *porb, Real x_, Real y_, Real z_){
  return -0.5*std::sqrt(gm0/(std::sin(y_)*x_))*std::cos(y_)/std::sin(y_)
         -omegarot*x_*std::cos(y_);
//  return -omegarot*std::cos(y_)*x_;
}

void AlphaVis(HydroDiffusion *phdif, MeshBlock *pmb,const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,int is, int ie, int js, int je, int ks, int ke) {
  // std::cout << "I am in alphavis!! "<< std::endl;
  Real rad,phi,z;
  Coordinates *pcoord = pmb->pcoord;
  if (phdif->nu_iso > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          GetCylCoord(pcoord,rad,phi,z,i,j,k);
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha*std::pow(rad/r0,aslope)*PoverR(rad, phi, z)/(sqrt(gm0/rad)/rad);
          // Real omega = sqrt(gm0/(rad*rad*rad));
          // phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha*sound_speed*sound_speed/omega;
        }
      }
    }
  }
 if (phdif->nu_aniso > 0.0) {
    for (int ij=0; ij<=2; ij++) {
      for (int ii=0; ii<=2; ii++) {
       //phdif->ani(ij,ii)=0.;
      }
    }
    //phdif->ani(0,2)=1.;
    //phdif->ani(2,0)=1.;  //SABINA COMMENTED

/*    phdif->ani(0,0)=1.;
      phdif->ani(1,1)=1.;
      phdif->ani(2,2)=1.;
      phdif->ani(0,1)=1.;
      phdif->ani(1,0)=1.;
      phdif->ani(1,2)=1.;
      phdif->ani(2,1)=1.;
*/
    if(visflag==0){
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            GetCylCoord(pcoord,rad,phi,z,i,j,k);
            phdif->nu(HydroDiffusion::DiffProcess::aniso,k,j,i) = alphaa*std::pow(rad/r0,aslope)*PoverR(rad, phi, z)/(sqrt(gm0/rad)/rad);
          }
        }
      }
    }
    if(visflag==1){
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            GetCylCoord(pcoord,rad,phi,z,i,j,k);
            Real height=sqrt(PoverR(rad, phi, z))/(sqrt(gm0/rad)/rad);
            Real ch=3.;
            if(std::fabs(z)<=ch*height){
              phdif->nu(HydroDiffusion::DiffProcess::aniso,k,j,i) = 
                alphaa*std::pow(rad/r0,aslope)*PoverR(rad, phi, z)/(sqrt(gm0/rad)/rad)*sqrt(2.*PI)/2./ch*exp(z*z/height/height/2.);
            }           
          }
        }
      }
    }
  }

  return;
}



//--------------------------------------------------------------------------
//!\f: User-work in loop: add damping boundary condition at the inner and outer boundaries
//

void MeshBlock::UserWorkInLoop() {
    Real smooth, tau ;
    Real rad(0.0), phi(0.0), z(0.0);
    Real den(0.0), v1(0.0), v2(0.0), v3(0.0), pre(0.0), tote(0.0);
    Real x1min = pmy_mesh->mesh_size.x1min;
    Real x1max = pmy_mesh->mesh_size.x1max;
    Real dt = pmy_mesh->dt;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          if (pcoord->x1v(i)<rindamp||pcoord->x1v(i)>routdamp){
            if (pcoord->x1v(i)<rindamp) {
              smooth = 1.- SQR(std::sin(PI/2.*(pcoord->x1v(i)-x1min)/(rindamp-x1min)));
              tau = 0.1*2.*PI*std::sqrt(x1min/gm0)*x1min;
            }
            if (pcoord->x1v(i)>routdamp){
              smooth = SQR(std::sin(PI/2.*(pcoord->x1v(i)-routdamp)/(x1max-routdamp)));
              tau = 0.1*2.*PI*std::sqrt(x1max/gm0)*x1max;
            } 
            GetCylCoord(pcoord,rad,phi,z,i,j,k);
            den = DenProfileCyl(rad,phi,z);
            VelProfileCyl(rad,phi,z,v1,v2,v3);       
            if(porb->orbital_advection_defined) {
              SubtractionOrbitalVelocity(porb,pcoord,v1,v2,v3,i,j,k);
            }
            if (NON_BAROTROPIC_EOS){
              Real gam = peos->GetGamma();
              

              pre = PoverR(rad, phi, z)*den;
              tote = 0.5*den*(SQR(v1)+SQR(v2)+SQR(v3)) + pre/(gam-1.0);
            }
            Real taud = tau/smooth;
            phydro->u(IDN,k,j,i) = (phydro->u(IDN,k,j,i)*taud + den*dt)/(dt + taud);            
            phydro->u(IM1,k,j,i) = (phydro->u(IM1,k,j,i)*taud + den*v1*dt)/(dt + taud);
            phydro->u(IM2,k,j,i) = (phydro->u(IM2,k,j,i)*taud + den*v2*dt)/(dt + taud);    
            phydro->u(IM3,k,j,i) = (phydro->u(IM3,k,j,i)*taud + den*v3*dt)/(dt + taud);
            if (NON_BAROTROPIC_EOS)
        phydro->u(IEN,k,j,i) = (phydro->u(IEN,k,j,i)*taud + tote*dt)/(dt + taud);         
          }
        }
      }
    }

    
    //luminosity:
     // 1D array of volumes

    /*if (luminosity != 0.0) { //luminosity per volume (defined in input file)
    Real Etot = luminosity*dt; // e, which is what cons(IEN) is.

    int ncells = 0;

    for (int k=ks; k<=ke; k++){
      x3=pco->x3v(k);
      if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){  
        cosx3=cos(x3);
        sinx3=sin(x3);
      }
      for (int j=js; j<=je; j++){
        pco->CellVolume(k, j, is, ie, volume);
        x2=pco->x2v(j);
        cosx2=cos(x2);
        sinx2=sin(x2);
        for (int i=is; i<=ie; i++){
          if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
            xcar = pco->x1v(i)*sinx2*cosx3;
            ycar = pco->x1v(i)*sinx2*sinx3;
            zcar = pco->x1v(i)*cosx2;
          }
          if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
            xcar = pco->x1v(i)*cosx2;
            ycar = pco->x1v(i)*sinx2;
            zcar = x3;
          }
          Real xpp=-1; 
          Real ypp=0;
          Real zpp=0;
          // get distance to planet
          Real dist = sqrt((xcar-xpp)*(xcar-xpp) + (ycar-ypp)*(ycar-ypp) + (zcar-zpp)*(zcar-zpp));
          //Real dist = std::sqrt(SQR(pmb->pcoord->x1v(i) - xp) + SQR(pmb->pcoord->x2v(j)) + SQR(pmb->pcoord->x3v(k)));
          if (dist <= r_radiation ){ //radius over which to deposit energy: 2*(r2-r1)/N (defined in input file)
            //cons(IEN,k,j,i) += E_per_vol;
            ncells++;
            E_per_vol = Etot/(volume(i)*ncells);
            phydro->u(IEN,k,j,i) += E_per_vol;
            //user_out_var(varnum,k,j,i) = E_per_vol;
            
          }
          //Etot = cons(IEN,k,j,i);
          //else {
            //user_out_var(varnum,k,j,i) = 0.;
          //}
           
            }
          }
        }
      }*/


    Real xcar, ycar, zcar;
    Real cosx3, sinx3, x3, cosx2, sinx2, x2;
    Real Etot;
    Coordinates *pco = pcoord;
    Real volume;
    int ncells;


    AthenaArray<Real> vol_;
    AthenaArray<Real> &volu = vol_;
    int nc1 = ncells1;
    vol_.NewAthenaArray(nc1);


    if (psys->np > 0) {
    psys->orbit(pmy_mesh->time);
    if (luminosity != 0.0) { //luminosity per volume (defined in input file)
        Etot = luminosity*dt; // e, which is what cons(IEN) is.
      //AthenaArray<Real> volu;
      //volu.NewAthenaArray((ie-is)+1+2*(NGHOST));
      ncells = 0;
      volume = 0;
      if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
        for (int ip=0; ip< psys->np; ++ip){
          Real xpp=psys->xp[ip]; //setting xpp to -1 for now
          Real ypp=psys->yp[ip];
          Real zpp=psys->zp[ip];
          for (int k=ks; k<=ke; ++k) {
            x3=pco->x3v(k);
            if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){  
              cosx3=cos(x3);
              sinx3=sin(x3);
            }
            for (int j=js; j<=je; ++j) {
              pco->CellVolume(k,j,is,ie,volu);
              x2=pco->x2v(j);
              cosx2=cos(x2);
              sinx2=sin(x2);
              for (int i=is; i<=ie; ++i) {  
                //volume = volu(i);
                if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
                  xcar = pco->x1v(i)*sinx2*cosx3;
                  ycar = pco->x1v(i)*sinx2*sinx3;
                  zcar = pco->x1v(i)*cosx2;
                }
                if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
                  xcar = pco->x1v(i)*cosx2;
                  ycar = pco->x1v(i)*sinx2;
                  zcar = x3;
                }
                Real dist = sqrt((xcar-xpp)*(xcar-xpp) + (ycar-ypp)*(ycar-ypp) + (zcar-zpp)*(zcar-zpp));
                if (dist <= r_radiation ){ //radius over which to deposit energy: 2*(r2-r1)/N (defined in input file)
                  ncells++;
                  //std::cout << "the number of cells is "<<ncells<<std::endl;
                }
                if (ncells == 8){
                  volume+=volu(i);
                  //std::cout << "the volume is "<<volume<<std::endl;
                  //std::cout << "the number of cells is "<<ncells<<std::endl;
                  Real E_per_vol = Etot/volume;
                  //std::cout << "the Energy per volume is "<<E_per_vol<<std::endl;
                  phydro->u(IEN,k,j,i) += E_per_vol;
                }
              }
            }
          }
        }
      }
    }
  }

  //std::cout << "the volume is "<<volume<<std::endl;
  //std::cout << "the number of cells is "<<ncells<<std::endl;

  

//std::cout << "I'm in UserWorkInLoop!"<<std::endl;

  return;
}

/*void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)*/
void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar)
{
  PlanetarySourceTerms(pmb,time,dt,prim,bcc,cons);
  if(NON_BAROTROPIC_EOS&&tcool>0.0) Cooling(pmb,time,dt,prim,bcc,cons);
  return;
}

void Cooling(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  if(tcool>0.0) {
    Coordinates *pco = pmb->pcoord;
    Real rad,phi,z;
    for(int k=pmb->ks; k<=pmb->ke; ++k){
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,k);
          Real eint = cons(IEN,k,j,i)-0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                                          +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
          Real pres_over_r=eint*(gamma_gas-1.0)/cons(IDN,k,j,i);
          Real p_over_r = PoverR(rad, phi, z);
          Real dtr = std::max(tcool*2.*PI/sqrt(gm0/rad/rad/rad),dt);
          Real dfrac=dt/dtr;
          Real dE=eint-p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);
          cons(IEN,k,j,i) -= dE*dfrac; 
        }
      }
    }
  }
}

void PlanetarySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  const AthenaArray<Real> *flux=pmb->phydro->flux;
  Real cosx3, sinx3, x3, cosx2, sinx2, x2;
  Real xcar, ycar, zcar;
  psys->orbit(time);
 
  Coordinates *pco = pmb->pcoord;
  AthenaArray<Real> vol;
  vol.NewAthenaArray((pmb->ie-pmb->is)+1+2*(NGHOST));

  Real src[NHYDRO];
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    x3=pco->x3v(k);
    if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){  
      cosx3=cos(x3);
      sinx3=sin(x3);
    }
    for (int j=pmb->js; j<=pmb->je; ++j) {
      x2=pco->x2v(j);
      cosx2=cos(x2);
      sinx2=sin(x2);
      Real sm = std::fabs(std::sin(pco->x2f(j  )));
      Real sp = std::fabs(std::sin(pco->x2f(j+1)));
      Real cmmcp = std::fabs(std::cos(pco->x2f(j  )) - std::cos(pco->x2f(j+1)));
      Real coord_src1_j=(sp-sm)/cmmcp;
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real rm = pco->x1f(i  );
        Real rp = pco->x1f(i+1);
        Real coord_src1_i=1.5*(rp*rp-rm*rm)/(rp*rp*rp-rm*rm*rm);
        Real drs = pco->dx1v(i) / 10000.;
        if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
          xcar = pco->x1v(i)*sinx2*cosx3;
          ycar = pco->x1v(i)*sinx2*sinx3;
          zcar = pco->x1v(i)*cosx2;
        }
        if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
          xcar = pco->x1v(i)*cosx2;
          ycar = pco->x1v(i)*sinx2;
          zcar = x3;
        }
        Real f_x1 = 0.0;
        Real f_x2 = 0.0;
        Real f_x3 = 0.0;
        for (int ip=0; ip< psys->np; ++ip){
          //Real xpp=psys->xp[ip];
          Real xpp=psys->xp[ip]; // setting this to 1 for now
          Real ypp=psys->yp[ip];
          Real zpp=psys->zp[ip];
          Real mp=psys->mass[ip];
          Real rsoft2=psys->rsoft2;
          Real f_xca = -1.0* (grav_pot_car_btoa(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp,rsoft2)
                                -grav_pot_car_btoa(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
          Real f_yca = -1.0* (grav_pot_car_btoa(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp,rsoft2)
                                -grav_pot_car_btoa(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
          Real f_zca = -1.0* (grav_pot_car_btoa(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp,rsoft2)
                                -grav_pot_car_btoa(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
          if(psys->ind!=0){
            f_xca += -1.0* (grav_pot_car_ind(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
            f_yca += -1.0* (grav_pot_car_ind(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
            f_zca += -1.0* (grav_pot_car_ind(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp))/(2.0*drs);
          }
          if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
            f_x1 += f_xca*sinx2*cosx3+f_yca*sinx2*sinx3+f_zca*cosx2;
            f_x2 += f_xca*cosx2*cosx3+f_yca*cosx2*sinx3-f_zca*sinx2;
            f_x3 += f_xca*(-sinx3) + f_yca*cosx3;
          }
          if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
            f_x1 += f_xca*cosx2 + f_yca*sinx2;
            f_x2 += -f_xca*sinx2 + f_yca*cosx2;
            f_x3 += f_zca;
          }

          // Calculating mass and angular momentum falling into the CPD

          // Rh = pow(mp/3,1/3); // Hill's sphere

          // if (rp<Rh){

          //   mass_hill = vol(i)*prim(IDN,k,j,i);

          //   double ome = sqrt(mass_hill/Rh/Rh/Rh)*dt;

          //   double b = Rh*Rh+1.*1.+2*Rh*1.*cos(ome);

          //   Real vppphi = sqrt(mass_hill/b/b/b)*dt*b; 

          //   L_hill = cons(IM3,k,j,i) - vppphi*prim(IDN,k,j,i);
          // } 
        }
        if(omegarot!=0.0) {
        /* centrifugal force */
          f_x1 += omegarot*omegarot*pco->x1v(i)*pco->x1v(i)*sinx2*sinx2*coord_src1_i;
          f_x2 += omegarot*omegarot*pco->x1v(i)*pco->x1v(i)*sinx2*sinx2*coord_src1_i*coord_src1_j;
        }

        src[IM1] = dt*prim(IDN,k,j,i)*f_x1;
        src[IM2] = dt*prim(IDN,k,j,i)*f_x2;
        src[IM3] = dt*prim(IDN,k,j,i)*f_x3;

        cons(IM1,k,j,i) += src[IM1];
        cons(IM2,k,j,i) += src[IM2];
        cons(IM3,k,j,i) += src[IM3];



        if(NON_BAROTROPIC_EOS) {
          src[IEN] = f_x1*dt*0.5*(flux[X1DIR](IDN,k,j,i)+flux[X1DIR](IDN,k,j,i+1))+f_x2*dt*0.5*(flux[X2DIR](IDN,k,j,i)+flux[X2DIR](IDN,k,j+1,i))+f_x3*dt*0.5*(flux[X3DIR](IDN,k,j,i)+flux[X3DIR](IDN,k+1,j,i));
//          src[IEN] = src[IM1]*prim(IM1,k,j,i)+ src[IM2]*prim(IM2,k,j,i) + src[IM3]*prim(IM3,k,j,i);
          if(pmb->porb->orbital_advection_defined&&std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
            Real vadv=pmb->porb->OrbitalVelocity(pmb->porb,pco->x1v(i),pco->x2v(j),pco->x3v(k));
            src[IEN] += dt*coord_src1_i*(0.5*(flux[X1DIR](IDN,k,j,i)+flux[X1DIR](IDN,k,j,i+1))*2.*omegarot*pco->x1v(i)*sinx2*vadv
                                        +0.5*(flux[X2DIR](IDN,k,j,i)+flux[X2DIR](IDN,k,j+1,i))*2.*omegarot*pco->x1v(i)*sinx2*vadv*coord_src1_j);  
//            src[IEN] += dt*coord_src1_i*(prim(IM1,k,j,i)*prim(IDN,k,j,i)*2.*omegarot*pco->x1v(i)*sinx2*vadv
//                                        +prim(IM2,k,j,i)*prim(IDN,k,j,i)*2.*omegarot*pco->x1v(i)*sinx2*vadv*coord_src1_j);                      
          }
          cons(IEN,k,j,i) += src[IEN];
        }
        if(omegarot!=0.0) {
        /* Coriolis force */
          cons(IM1,k,j,i) += omegarot*sinx2*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))*pco->x1v(i)*coord_src1_i*dt;
          cons(IM2,k,j,i) += omegarot*sinx2*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))*pco->x1v(i)*coord_src1_i*coord_src1_j*dt;
          cons(IM3,k,j,i) -= (omegarot*3.*(sp+sm)/(rp+rm)/(rp*rp+rp*rm+rm*rm)*(rp*rp*(3.*rp+rm)/4.*flux[X1DIR](IDN,k,j,i+1)+rm*rm*(3.*rm+rp)/4.*flux[X1DIR](IDN,k,j,i))+omegarot*3.*(rp+rm)*(rp+rm)*(sp-sm)/2./(sp+sm)/(rp*rp+rp*rm+rm*rm)/cmmcp*((3.*sp+sm)/4.*sp*flux[X2DIR](IDN,k,j+1,i)+(3.*sm+sp)/4.*sm*flux[X2DIR](IDN,k,j,i)))*dt;
          if(pmb->porb->orbital_advection_defined&&std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
            Real vadv=pmb->porb->OrbitalVelocity(pmb->porb,pco->x1v(i),pco->x2v(j),pco->x3v(k));
            cons(IM1,k,j,i) += omegarot*sinx2*2.*prim(IDN,k,j,i)*vadv*pco->x1v(i)*coord_src1_i*dt;
            cons(IM2,k,j,i) += omegarot*sinx2*2.*prim(IDN,k,j,i)*vadv*pco->x1v(i)*coord_src1_i*coord_src1_j*dt;
          }
/*
          Real dt_den_over_r = dt*prim(IDN,k,j,i)*coord_src1_i;
          Real rv  = pmb->pcoord->x1v(i);
          Real vc  = rv*std::sin(pmb->pcoord->x2v(j))*omegarot;
          Real cv1 = coord_src1_j;
          Real cv3 = coord_src1_j;

          Real src_i1 = 2.0*dt_den_over_r*vc*prim(IVZ,k,j,i);
          if(pmb->porb->orbital_advection_defined){
            Real vadv=pmb->porb->OrbitalVelocity(pmb->porb,pco->x1v(i),pco->x2v(j),pco->x3v(k));
            src_i1 += 2.0*dt_den_over_r*vc*vadv;
          }
          cons(IM1,k,j,i) += src_i1;
          cons(IM2,k,j,i) += src_i1*cv1;
*/
//          cons(IM3,k,j,i) += -2.0*dt_den_over_r*vc*(prim(IVX,k,j,i)+cv3*prim(IVY,k,j,i));

/* the following is identical to the implementation of the fargo scheme */
//          cons(IM3,k,j,i) += -2.0*dt*coord_src1_i*vc*
//                             (0.5*(flux[X1DIR](IDN,k,j,i)+flux[X1DIR](IDN,k,j,i+1))+
//                              cv3*0.5*(flux[X2DIR](IDN,k,j,i)+flux[X2DIR](IDN,k,j+1,i)));
        } 
      }
    }
  }
}

Real PlanetForce(MeshBlock *pmb, int iout)
{
  //std::cout << "I'm in PlanetForce!"<<std::endl;
  if (psys->np > 0) {
    psys->orbit(pmb->pmy_mesh->time);
    Coordinates *pco = pmb->pcoord;
    AthenaArray<Real> vol;
    vol.NewAthenaArray((pmb->ie-pmb->is)+1+2*(NGHOST));
    Real dphi=1.e-3;
    if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
      for (int ip=0; ip< psys->np; ++ip){
        Real f_xpp = 0.0;
        Real f_ypp = 0.0;
        Real f_zpp = 0.0;
        Real torque = 0.0;
        Real voluume = 0.0;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);


        //Real xpp=psys->xp[ip];
        Real xpp=psys->xp[ip]; //setting xpp to -1 for now
        Real ypp=psys->yp[ip];
        Real zpp=psys->zp[ip];
        Real rpp=sqrt(xpp*xpp+ypp*ypp+zpp*zpp);
        Real thepp=acos(zpp/rpp);
        Real phipp=atan2(ypp,xpp);
        Real mp=psys->mass[ip];
        Real rsoft2=psys->rsoft2;
        for (int k=pmb->ks; k<=pmb->ke; ++k) {
          Real x3=pco->x3v(k);
          Real cosx3=cos(x3);
          Real sinx3=sin(x3);
          Real x3p=pco->x3v(k)+dphi;
          Real cosx3p=cos(x3p);
          Real sinx3p=sin(x3p);
          Real x3m=pco->x3v(k)-dphi;
          Real cosx3m=cos(x3m);
          Real sinx3m=sin(x3m);
          for (int j=pmb->js; j<=pmb->je; ++j) {
            pco->CellVolume(k,j,pmb->is,pmb->ie,vol);
            Real x2=pco->x2v(j);
            Real cosx2=cos(x2);
            Real sinx2=sin(x2);
            for (int i=pmb->is; i<=pmb->ie; ++i) {
              Real drs = pco->dx1v(i) / 10000.;
              Real xcar = pco->x1v(i)*sinx2*cosx3;
              Real ycar = pco->x1v(i)*sinx2*sinx3;
              Real zcar = pco->x1v(i)*cosx2;
              f_xpp += vol(i)*pmb->phydro->u(IDN,k,j,i)*(grav_pot_car_btoa(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp,rsoft2)
                                  -grav_pot_car_btoa(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
              f_ypp += vol(i)*pmb->phydro->u(IDN,k,j,i)*(grav_pot_car_btoa(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp,rsoft2)
                                  -grav_pot_car_btoa(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
              f_zpp += vol(i)*pmb->phydro->u(IDN,k,j,i)*(grav_pot_car_btoa(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp,rsoft2)
                                  -grav_pot_car_btoa(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
              Real xpcar = pco->x1v(i)*sinx2*cosx3p;
              Real ypcar = pco->x1v(i)*sinx2*sinx3p;
              Real zpcar = pco->x1v(i)*cosx2;
              Real xmcar = pco->x1v(i)*sinx2*cosx3m;
              Real ymcar = pco->x1v(i)*sinx2*sinx3m;
              Real zmcar = pco->x1v(i)*cosx2;
              torque += -vol(i)*pmb->phydro->u(IDN,k,j,i)*(grav_pot_car_btoa(xpcar,ypcar,zpcar,xpp,ypp,zpp,mp,rsoft2)
                                  -grav_pot_car_btoa(xmcar,ymcar,zmcar,xpp,ypp,zpp,mp,rsoft2))/2.0/dphi;     

              voluume = vol(i);
            }
          }
        }
        
        Real f_r = f_xpp*sin(thepp)*cos(phipp) + f_ypp*sin(thepp)*sin(phipp) + f_zpp*cos(thepp);
        Real f_t = f_xpp*cos(thepp)*cos(phipp) + f_ypp*cos(thepp)*sin(phipp) - f_zpp*sin(thepp);
        Real f_p = f_xpp*(-sin(phipp)) + f_ypp*cos(phipp);

        

        if (iout==0&&ip==0) return f_r;
        if (iout==1&&ip==0) return f_t;
        if (iout==2&&ip==0) return f_p;
        if (iout==3&&ip==0) return f_xpp;
        if (iout==4&&ip==0) return f_ypp;
        if (iout==5&&ip==0) return f_zpp;
        if (iout==6&&ip==0) return torque;
        if (iout==7&&ip==0) return xpp;
        if (iout==8&&ip==0) return ypp;
        if (iout==9&&ip==0) return zpp;
        if (iout==10&&ip==0) return rpp;
        if (iout==11&&ip==0) return thepp;
        if (iout==12&&ip==0) return phipp;
        if (iout==13&&ip==0) return mp;
        if (iout==14&&ip==0) return voluume;
        if (rank == 0) {
            if (iout==7&&ip==0) return xpp;
            if (iout==8&&ip==0) return ypp;
            if (iout==9&&ip==0) return zpp;
            if (iout==10&&ip==0) return rpp;
            if (iout==11&&ip==0) return thepp;
            if (iout==12&&ip==0) return phipp;
            if (iout==13&&ip==0) return mp;
            if (iout==14&&ip==0) return voluume;
            // if (iout==15&&ip==0) return Rh;
            // if (iout==16&&ip==0) return mass_hill;
            // if (iout==17&&ip==0) return L_hill;
        }
      }
    }
  }
  return 0;
}     


//----------------------------------------------------------------------------------------
//!\f: Use grav potential to calculate forces from b to a
//

Real grav_pot_car_btoa(const Real xca, const Real yca, const Real zca,
        const Real xcb, const Real ycb, const Real zcb, const Real gb, const Real rsoft2)
{
  Real dist=sqrt((xca-xcb)*(xca-xcb) + (yca-ycb)*(yca-ycb) + (zca-zcb)*(zca-zcb));
  Real rsoft=sqrt(rsoft2);
  Real dos=dist/rsoft;
  Real pot;
  if(dist>=rsoft){
     pot=-gb/dist;
  }else{
     pot=-gb/dist*(dos*dos*dos*dos-2.*dos*dos*dos+2*dos);
  }
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    Real dists=sqrt(xca*xca+yca*yca+zca*zca);
    pot+=-1./dists;
  }
//  dist2 = (xca-xcb)*(xca-xcb) + (yca-ycb)*(yca-ycb) + (zca-zcb)*(zca-zcb);
//  Real pot = -gb*(dist2+1.5*rsoft2)/(dist2+rsoft2)/sqrt(dist2+rsoft2);
  return(pot);
}

//----------------------------------------------------------------------------------------
//!\f: Use grav potential to calculate indirect forces due to gmp 
//

Real grav_pot_car_ind(const Real xca, const Real yca, const Real zca,
        const Real xpp, const Real ypp, const Real zpp, const Real gmp)
{
  Real pdist=sqrt(xpp*xpp+ypp*ypp+zpp*zpp);
  Real pot = gmp/pdist/pdist/pdist*(xca*xpp+yca*ypp+zca*zpp);
  return(pot);
}

//----------------------------------------------------------------------------------------
//!\f: Fix planetary orbit
//
void PlanetarySystem::orbit(double time)
{
  int i;
  for(i=0; i<np; ++i){
    if(time<insert_time*2.*PI) {
      mass[i]=massset[i]*sin(time/insert_time/4.)*sin(time/insert_time/4.);
    }else{
      mass[i]=massset[i];
    }
    double dis=sqrt(xp[i]*xp[i]+yp[i]*yp[i]);
    double ome=(sqrt((gm0+mass[i])/dis/dis/dis)-omegarot-Omega0);
    double ang=acos(xp[i]/dis);
    ang = ome*time;
    // xp[i]=dis*cos(ang);
    // yp[i]=dis*sin(ang);
  }
  return;
}

//----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
//

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco,rad,phi,z,il-i,j,k);
        prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,il-i,j,k);
        }
        prim(IM1,k,j,il-i) = v1;
        prim(IM2,k,j,il-i) = v2;
        prim(IM3,k,j,il-i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
      }
    }
  }
}

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco,rad,phi,z,iu+i,j,k);
        prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,iu+i,j,k);
        }
        prim(IM1,k,j,iu+i) = v1;
        prim(IM2,k,j,iu+i) = v2;
        prim(IM3,k,j,iu+i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
      }
    }
  }
}

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        GetCylCoord(pco,rad,phi,z,i,jl-j,k);
        prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,i,jl-j,k);
        }
        prim(IM1,k,jl-j,i) = v1;
        prim(IM2,k,jl-j,i) = v2;
        prim(IM3,k,jl-j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
      }
    }
  }
}

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        GetCylCoord(pco,rad,phi,z,i,ju+j,k);
        prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,i,ju+j,k);
        }
        prim(IM1,k,ju+j,i) = v1;
        prim(IM2,k,ju+j,i) = v2;
        prim(IM3,k,ju+j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
      }
    }
  }
}

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        GetCylCoord(pco,rad,phi,z,i,j,kl-k);
        prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,i,j,kl-k);
        }
        prim(IM1,kl-k,j,i) = v1;
        prim(IM2,kl-k,j,i) = v2;
        prim(IM3,kl-k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
      }
    }
  }
}

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        GetCylCoord(pco,rad,phi,z,i,j,ku+k);
        prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,i,j,ku+k);
        }
        prim(IM1,ku+k,j,i) = v1;
        prim(IM2,ku+k,j,i) = v2;
        prim(IM3,ku+k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
      }
    }
  }
}
