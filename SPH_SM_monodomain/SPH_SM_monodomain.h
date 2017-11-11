/*
This class implements the SPH [5] and SM [2] algorithms with the needed modifications to correct velocity as presented in [1]. 

@author Octavio Navarro
@version 1.0
*/
#pragma once
#ifndef __SPH_SM_monodomain_H__
#define __SPH_SM_monodomain_H__

#include <m3Vector.h>
#include <m3Bounds.h>
#include <m3Real.h>
#include <m3Matrix.h>
#include <m9Matrix.h>

#include <Particle.h>

#include <vector>
#include <map>
#include <chrono>

#include <helper_cuda.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#define PI 3.141592f
#define INF 1E-12f

typedef std::chrono::system_clock::time_point 	tpoint;
typedef std::chrono::duration<double> 			duration_d;

class SPH_SM_monodomain
{
	private:
	
		Particles 	*particles;

		// grid data for sorting method
        uint  *dGridParticleHash; 		// grid hash value for each particle
        uint  *dGridParticleIndex;		// particle index for each particle
        uint  *dCellStart;        		// index of start of each cell in sorted list
        uint  *dCellEnd;          		// index of end of cell

		/// Particle system parameters

		m3Real 		kernel;					// kernel or h in kernel function
		int 		Max_Number_Paticles;	// initial array for particles
		int 		Number_Particles;		// paticle number

		m3Vector 	Grid_Size;				// Size of a size of each grid voxel
		m3Vector 	World_Size;				// screen size
		m3Real 		Cell_Size;				// Size of the divisions in the grid; used to determine the cell position for the has grid; kernel or h
		int 		Number_Cells;			// Number of cells in the hash grid

		m3Vector 	Gravity;
		m3Real 		K;						// ideal pressure formulation k; Stiffness of the fluid
											// The lower the value, the stiffer the fluid
		m3Real 		Stand_Density;			// ideal pressure formulation p0
		m3Real 		Time_Delta;			
		m3Real 		Wall_Hit;				// To manage collisions with the environment.
		m3Real 		mu;						// Viscosity.
		m3Real		velocity_mixing;		// Velocity mixing parameter for the intermediate velocity.

		// SM Parameters
		m3Bounds 	bounds;					// Controls the bounds of the simulation.
		m3Real 		alpha;					// alpha[0...1] Simulates stiffness.
		m3Real 		beta;					// Not entirely sure about this parameter, but the larger, the less rigid the object.
		bool 		quadraticMatch;			// Linear transformations can only represent shear and stretch. To extend the range of motion by twist and bending modes, w move from linear to quadratic transformations.
		bool 		volumeConservation;		// Allows the object to conserve its volume.
		bool 		allowFlip;

		// Monodomain parameters
		m3Real    	Cm;                   	// Membrane capacitance per unit area
		m3Real    	Beta;                 	// Surface volume ratio
		m3Real		sigma;					// Value for the conductivity tensor (a constant in this case)
		bool		isStimOn;				// Checks if the stimulation current is still enabled
		m3Real		stim_strength;

		//membrane model parameters
		m3Real 		FH_Vt=-75.0;
		m3Real 		FH_Vp=15.0;
		m3Real 		FH_Vr=-85.0;

		m3Real 		C1 = 0.175;
		m3Real 		C2 = 0.03;

		m3Real 		C3 = 0.011;
		m3Real 		C4 = 0.55;

		// Max velocity allowed for a particle.
		m3Vector 	max_vel = m3Vector(INF, INF, INF);	

		m3Real Poly6_constant;
		m3Real Spiky_constant;
		m3Real B_spline_constant;

	public:
		SPH_SM_monodomain();
		~SPH_SM_monodomain();

		m3Real voltage_constant = 50;
		m3Real max_pressure = 100000;
		m3Real max_voltage = 500;

		// Variables to meassure time spent in each step
		tpoint t_start_find_neighbors, t_start_corrected_velocity, t_start_intermediate_velocity, t_start_Density_SingPressure, t_start_cell_model, t_start_compute_Force, t_start_Update_Properties;

		duration_d d_find_neighbors,d_corrected_velocity,d_intermediate_velocity,d_Density_SingPressure,d_cell_model,d_compute_Force,d_Update_Properties;

		int total_time_steps;

		void Init_Fluid(std::vector<m3Vector> positions);	// initialize fluid
		// void Init_Particle(m3Vector pos, m3Vector vel);		// initialize particle system
		
		// void print_report(double avg_fps = 0.0f, double avg_step_d = 0.0f);
		void add_viscosity(float value);

		/// Hashed the particles into a grid
		void calcHash(uint *gridParticleHash, uint * gridParticleIndex, m3Vector *pos, int numberParticles);
		
		void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, int numParticles);

		void reorderDataAndFindCellStart(m3Vector *sortedPos_d, m3Vector *pos_d,
		m3Vector *sortedVel_d, m3Vector *vel_d,
		m3Vector *sorted_corr_vel_d, m3Vector *corrected_vel_d,
		m3Vector *sortedAcc_d, m3Vector *acc_d,
		m3Vector *sorted_int_vel_d, m3Vector *inter_vel_d,
		m3Real *sortedMass_d, m3Real *mass_d,
		bool *sorted_mFixed_d, bool *mFixed_d,
		m3Real *sorted_dens_d, m3Real *dens_d,
		m3Real *sorted_pres_d, m3Real *pres_d,
		m3Real *sorted_Vm_d, m3Real *Vm_d,
		m3Real *sorted_Inter_Vm_d, m3Real *Inter_Vm_d,
		m3Real *sorted_Iion_d, m3Real *Iion_d,
		m3Real *sorted_stim_d, m3Real *stim_d,
		m3Real *sorted_w_d, m3Real *w_d,
		uint *cellStart, uint *cellEnd, uint *gridParticleHash, uint *gridParticleIndex, uint numParticles, uint numCells);

		/// SM methods
		/// Calculates the predicted velocity, and the corrected velocity using SM, in order to
		/// obtain the intermediate velocity that is input to SPH. Taken from 2014 - A velocity correcting method
		/// for volume preserving viscoelastic fluids
		void calculate_corrected_velocity();
		/// Applies external forces for F-adv, including gravity
		void apply_external_forces(m3Vector* forcesArray, int* indexArray, int size);
		/// Obtains the corrected velocity of the particles using Shape Matching
		void projectPositions();

		/// Monodomain methods
		void calculate_cell_model();											// Updates the ionic current and recovery variable with the cell model
		void set_stim(m3Vector center, m3Real radius, m3Real stim_strength);	// Turns the stimulation on at point center, around a given radius
		void turnOnStim_Cube(std::vector<m3Vector> positions);
		void turnOnStim_Mesh(std::vector<m3Vector> positions);
		void turnOffStim();					// Turns the stimulation off for all particles

		/// SPH Methods
		void calculate_intermediate_velocity();		
		void Compute_Density_SingPressure();
		void Compute_Force();
		void Update_Properties();					// Updates Position and velocity for SPH, voltage for monodomain

		void compute_SPH_SM_monodomain();
		void Animation();

		inline int Get_Particle_Number() { return Number_Particles; }
		inline m3Vector Get_World_Size() { return World_Size; }
		inline Particles* Get_Paticles() { return particles; }
		inline m3Real Get_stand_dens()	 { return Stand_Density;}		 

		inline bool flip_quadratic()	{ quadraticMatch = !quadraticMatch; return quadraticMatch; }
		inline bool flip_volume()		{ volumeConservation = !volumeConservation; return volumeConservation; }

		uint iDivUp(uint a, uint b)
		{
			return (a % b != 0) ? (a / b + 1) : (a / b);
		}

		// compute grid and thread block size for a given number of elements
		void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
		{
			numThreads = min(blockSize, n);
			numBlocks = iDivUp(n, numThreads);
		}
};


#endif