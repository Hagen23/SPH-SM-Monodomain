#ifndef __SPH_SM_monodomain_CPP__
#define __SPH_SM_monodomain_CPP__

#include <SPH_SM_monodomain.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

SPH_SM_monodomain::SPH_SM_monodomain()
{
	float sigma_i = 0.893, sigma_e = 0.67;

	kernel = 0.04f;

	Max_Number_Paticles = 50000;
	total_time_steps = 0;
	Number_Particles = 0;

	Cm = 1.f;
    Beta = 50;
	isStimOn = false;
	sigma = sigma_i * sigma_e / ( sigma_i + sigma_e); //1.0f;
	stim_strength = 30000.0f;
	
	World_Size = m3Vector(1.5f, 1.5f, 1.5f);

	Cell_Size = 0.04;
	Grid_Size = World_Size / Cell_Size;
	Grid_Size.x = (int)ceil(Grid_Size.x);
	Grid_Size.y = (int)ceil(Grid_Size.y);
	Grid_Size.z = (int)ceil(Grid_Size.z);

	Number_Cells = (int)Grid_Size.x * (int)Grid_Size.y * (int)Grid_Size.z;

	Gravity.set(0.0f, -9.8f, 0.0f);
	K = 0.8f;
	Stand_Density = 1112.0f;
	max_vel = m3Vector(3.0f, 3.0f, 3.0f);
	velocity_mixing = 1.0f;

	/// Time step is calculated as in 2016 - Divergence-Free SPH for Incompressible and Viscous Fluids.
	/// Then we adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition [6] ∆t ≤ 0.4 * d / (||vmax||)
	Time_Delta = 0.4 * kernel / sqrt(max_vel.magnitudeSquared());
	Wall_Hit = -1.0f; //0.05f;
	mu = 100.0f;

	Particles = new Particle[Max_Number_Paticles];
	Cells = new Cell[Number_Cells];

	Poly6_constant = 315.0f/(64.0f * m3Pi * pow(kernel, 9));
	Spiky_constant = 45.0f/(m3Pi * pow(kernel, 6));

	B_spline_constant = 1.0f / (m3Pi*kernel*kernel*kernel);

	// SM initializations
	bounds.min.zero();
	bounds.max.set(1.5f, 1.5f, 1.5f);

	// Beta has to be larger than alfa
	alpha = 0.3f;
	beta = 0.4f;

	quadraticMatch = false;
	volumeConservation = true;
	allowFlip = true;

	cout<<"SPHSystem"<<endl;
	cout<<"Grid_Size_X : " << Grid_Size.x << endl;
	cout<<"Grid_Size_Y : " << Grid_Size.y << endl;
	cout<<"Grid_Size_Z : " << Grid_Size.y << endl;
	cout<<"Alpha :" << alpha << " Beta :" << beta << endl;
	cout<<"Volume conservation :" << volumeConservation << " Quadratic match : "<<quadraticMatch << endl;
	cout<<"Cell Number : "<<Number_Cells<<endl;
	cout<<"Time Delta : "<<Time_Delta<<endl;
}

SPH_SM_monodomain::~SPH_SM_monodomain()
{
	delete[] Particles;
	delete[] Cells;
}

void SPH_SM_monodomain::add_viscosity(float value)
{
	mu += (mu + value) >= 0 ? value : 0;
	// cout << "Viscosity: " << mu  << endl;
}

void SPH_SM_monodomain::Init_Fluid(vector<m3Vector> positions)
{
	for(m3Vector pos : positions)
		Init_Particle(pos, m3Vector(0.f, 0.f, 0.f));

	cout<<"Number of Paticles : "<<Number_Particles<<endl;
}

void SPH_SM_monodomain::Init_Particle(m3Vector pos, m3Vector vel)
{
	if(Number_Particles + 1 > Max_Number_Paticles)
		return;

	Particle *p = &(Particles[Number_Particles]);
	
	p->pos = pos;
	p->mOriginalPos = pos;
	p->mGoalPos = pos;
	p->mFixed = false;

	p->vel = vel;
	p->acc = m3Vector(0.0f, 0.0f, 0.0f);
	p->dens = Stand_Density;
	p->mass = 0.2f;

	p->Inter_Vm = 0.0f;
	p->Vm = 0.0f;
	p->Iion = 0.0f;
	p->stim = 0.0f;
	p->w = 0.0f;

	Number_Particles++;
}

m3Vector SPH_SM_monodomain::Calculate_Cell_Position(m3Vector pos)
{
	m3Vector cellpos = pos / Cell_Size;
	cellpos.x = (int)cellpos.x;
	cellpos.y = (int)cellpos.y;
	cellpos.z = (int)cellpos.z;
	return cellpos;
}

int SPH_SM_monodomain::Calculate_Cell_Hash(m3Vector pos)
{
	if((pos.x < 0)||(pos.x >= Grid_Size.x)||(pos.y < 0)||(pos.y >= Grid_Size.y)||
	(pos.z < 0)||(pos.z >= Grid_Size.z))
		return -1;

	int hash = pos.x + Grid_Size.x * (pos.y + Grid_Size.y * pos.z);
	if(hash > Number_Cells)
		cout<<"Error";
	return hash;
}

/// For density computation
float SPH_SM_monodomain::Poly6(float r2)
{
	return (r2 >= 0 && r2 <= kernel*kernel) ? Poly6_constant * pow(kernel * kernel - r2, 3) : 0;
}

/// For force of pressure computation
float SPH_SM_monodomain::Spiky(float r)
{
	return (r >= 0 && r <= kernel ) ? -Spiky_constant * (kernel - r) * (kernel - r) : 0;
}

/// For viscosity computation
float SPH_SM_monodomain::Visco(float r)
{
	return (r >= 0 && r <= kernel ) ? Spiky_constant * (kernel - r) : 0;
}

m3Real SPH_SM_monodomain::B_spline(m3Real r)
{
	m3Real q = r / kernel;
	if (q >= 0 && q < 1)
		return B_spline_constant * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (0.25*pow(2 - q, 3));
	else
		return 0;
}

m3Real SPH_SM_monodomain::B_spline_1(m3Real r)
{
	m3Real q = r / kernel;
	if (q >= 0 && q < 1)
		return B_spline_constant * (-3.0f * q + 2.25f * q * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (-0.75 * pow(2 - q, 2));
	else
		return 0;
}

m3Real SPH_SM_monodomain::B_spline_2(m3Real r)
{
	m3Real q = r / kernel;
	if (q >= 0 && q < 1)
		return B_spline_constant * (-3 + 4.5 * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (1.5 * (2 - q));
	else
		return 0;
}

void SPH_SM_monodomain::Find_neighbors()
{
	int hash;
	Particle *p;

	for(int i = 0; i < Number_Cells; i++)
		Cells[i].contained_particles.clear();

	for(int i = 0; i < Number_Particles; i ++)
	{
		p = &Particles[i];
		hash = Calculate_Cell_Hash(Calculate_Cell_Position(p->pos));
		Cells[hash].contained_particles.push_back(p);
	}
}

void SPH_SM_monodomain::apply_external_forces(m3Vector* forcesArray = NULL, int* indexArray = NULL, int size = 0)
{
	/// External forces
	for (int i = 0; i < size; i++)
	{
		int j = indexArray[i];
		if (Particles[j].mFixed) continue;
		Particles[j].predicted_vel += (forcesArray[i] * Time_Delta) / Particles[j].mass;
	}

	/// Gravity
	for (int i = 0; i < Number_Particles; i++)
	{
		if (Particles[i].mFixed) continue;
		Particles[i].predicted_vel = Particles[i].vel + (Gravity * Time_Delta) / Particles[i].mass;
		// Particles[i].mGoalPos = Particles[i].mOriginalPos;
	}
}

void SPH_SM_monodomain::projectPositions()
{
	if (Number_Particles <= 1) return;
	int i, j, k;

	// center of mass
	m3Vector cm, originalCm;
	cm.zero(); originalCm.zero();
	float mass = 0.0f;

	for (i = 0; i < Number_Particles; i++)
	{
		m3Real m = Particles[i].mass;
		if (Particles[i].mFixed) m *= 100.0f;
		mass += m;
		cm += Particles[i].pos * m;
		originalCm += Particles[i].mOriginalPos * m;
	}

	cm /= mass;
	originalCm /= mass;

	m3Vector p, q;

	m3Matrix Apq, Aqq;

	Apq.zero();
	Aqq.zero();

	for (i = 0; i < Number_Particles; i++)
	{
		p = Particles[i].pos - cm;
		q = Particles[i].mOriginalPos - originalCm;
		m3Real m = Particles[i].mass;

		Apq.r00 += m * p.x * q.x;
		Apq.r01 += m * p.x * q.y;
		Apq.r02 += m * p.x * q.z;

		Apq.r10 += m * p.y * q.x;
		Apq.r11 += m * p.y * q.y;
		Apq.r12 += m * p.y * q.z;

		Apq.r20 += m * p.z * q.x;
		Apq.r21 += m * p.z * q.y;
		Apq.r22 += m * p.z * q.z;

		Aqq.r00 += m * q.x * q.x;
		Aqq.r01 += m * q.x * q.y;
		Aqq.r02 += m * q.x * q.z;

		Aqq.r10 += m * q.y * q.x;
		Aqq.r11 += m * q.y * q.y;
		Aqq.r12 += m * q.y * q.z;

		Aqq.r20 += m * q.z * q.x;
		Aqq.r21 += m * q.z * q.y;
		Aqq.r22 += m * q.z * q.z;
	}

	if (!allowFlip && Apq.determinant() < 0.0f)
	{  	// prevent from flipping
		Apq.r01 = -Apq.r01;
		Apq.r11 = -Apq.r11;
		Apq.r22 = -Apq.r22;
	}

	m3Matrix R, S;
	m3Matrix::polarDecomposition(Apq, R, S);

	if (!quadraticMatch)
	{	// --------- linear match

		m3Matrix A = Aqq;
		A.invert();
		A.multiply(Apq, A);

		if (volumeConservation)
		{
			m3Real det = A.determinant();
			if (det != 0.0f)
			{
				det = 1.0f / sqrt(fabs(det));
				if (det > 2.0f) det = 2.0f;
				A *= det;
			}
		}

		m3Matrix T = R * (1.0f - beta) + A * beta;

		for (i = 0; i < Number_Particles; i++)
		{
			if (Particles[i].mFixed) continue;
			q = Particles[i].mOriginalPos - originalCm;
			Particles[i].mGoalPos = T.multiply(q) + cm;
		}
	}
	else
	{	// -------------- quadratic match---------------------

		m3Real A9pq[3][9];

		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 9; j++)
			A9pq[i][j] = 0.0f;

		m9Matrix A9qq;
		A9qq.zero();

		for (int i = 0; i < Number_Particles; i++)
		{
			p = Particles[i].pos - cm;
			q = Particles[i].mOriginalPos - originalCm;

			m3Real q9[9];
			q9[0] = q.x; q9[1] = q.y; q9[2] = q.z; q9[3] = q.x*q.x; q9[4] = q.y*q.y; q9[5] = q.z*q.z;
			q9[6] = q.x*q.y; q9[7] = q.y*q.z; q9[8] = q.z*q.x;

			m3Real m = Particles[i].mass;
			A9pq[0][0] += m * p.x * q9[0];
			A9pq[0][1] += m * p.x * q9[1];
			A9pq[0][2] += m * p.x * q9[2];
			A9pq[0][3] += m * p.x * q9[3];
			A9pq[0][4] += m * p.x * q9[4];
			A9pq[0][5] += m * p.x * q9[5];
			A9pq[0][6] += m * p.x * q9[6];
			A9pq[0][7] += m * p.x * q9[7];
			A9pq[0][8] += m * p.x * q9[8];

			A9pq[1][0] += m * p.y * q9[0];
			A9pq[1][1] += m * p.y * q9[1];
			A9pq[1][2] += m * p.y * q9[2];
			A9pq[1][3] += m * p.y * q9[3];
			A9pq[1][4] += m * p.y * q9[4];
			A9pq[1][5] += m * p.y * q9[5];
			A9pq[1][6] += m * p.y * q9[6];
			A9pq[1][7] += m * p.y * q9[7];
			A9pq[1][8] += m * p.y * q9[8];

			A9pq[2][0] += m * p.z * q9[0];
			A9pq[2][1] += m * p.z * q9[1];
			A9pq[2][2] += m * p.z * q9[2];
			A9pq[2][3] += m * p.z * q9[3];
			A9pq[2][4] += m * p.z * q9[4];
			A9pq[2][5] += m * p.z * q9[5];
			A9pq[2][6] += m * p.z * q9[6];
			A9pq[2][7] += m * p.z * q9[7];
			A9pq[2][8] += m * p.z * q9[8];

			for (j = 0; j < 9; j++)
			for (k = 0; k < 9; k++)
				A9qq(j, k) += m * q9[j] * q9[k];
		}

		A9qq.invert();

		m3Real A9[3][9];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				A9[i][j] = 0.0f;
				for (k = 0; k < 9; k++)
					A9[i][j] += A9pq[i][k] * A9qq(k, j);

				A9[i][j] *= beta;
				if (j < 3)
					A9[i][j] += (1.0f - beta) * R(i, j);
			}
		}

		m3Real det =
			A9[0][0] * (A9[1][1] * A9[2][2] - A9[2][1] * A9[1][2]) -
			A9[0][1] * (A9[1][0] * A9[2][2] - A9[2][0] * A9[1][2]) +
			A9[0][2] * (A9[1][0] * A9[2][1] - A9[1][1] * A9[2][0]);

		if (!allowFlip && det < 0.0f) {         		// prevent from flipping
			A9[0][1] = -A9[0][1];
			A9[1][1] = -A9[1][1];
			A9[2][2] = -A9[2][2];
		}

		if (volumeConservation)
		{
			if (det != 0.0f)
			{
				det = 1.0f / sqrt(fabs(det));
				if (det > 2.0f) det = 2.0f;

				for (int i = 0; i < 3; i++)
				for (int j = 0; j < 9; j++)
					A9[i][j] *= det;
			}
		}

		for (int i = 0; i < Number_Particles; i++)
		{
			if (Particles[i].mFixed) continue;
			q = Particles[i].mOriginalPos - originalCm;

			Particles[i].mGoalPos.x = A9[0][0] * q.x + A9[0][1] * q.y + A9[0][2] * q.z + A9[0][3] * q.x*q.x + A9[0][4] * q.y*q.y +
				A9[0][5] * q.z*q.z + A9[0][6] * q.x*q.y + A9[0][7] * q.y*q.z + A9[0][8] * q.z*q.x;

			Particles[i].mGoalPos.y = A9[1][0] * q.x + A9[1][1] * q.y + A9[1][2] * q.z + A9[1][3] * q.x*q.x + A9[1][4] * q.y*q.y +
				A9[1][5] * q.z*q.z + A9[1][6] * q.x*q.y + A9[1][7] * q.y*q.z + A9[1][8] * q.z*q.x;

			Particles[i].mGoalPos.z = A9[2][0] * q.x + A9[2][1] * q.y + A9[2][2] * q.z + A9[2][3] * q.x*q.x + A9[2][4] * q.y*q.y +
				A9[2][5] * q.z*q.z + A9[2][6] * q.x*q.y + A9[2][7] * q.y*q.z + A9[2][8] * q.z*q.x;

			Particles[i].mGoalPos += cm;
		}
	}
}

void SPH_SM_monodomain::Compute_Density_SingPressure()
{
	Particle *p;
	m3Vector CellPos;
	m3Vector NeighborPos;
	int hash;

	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];
		p->dens = 0;
		p->pres = 0;
		CellPos = Calculate_Cell_Position(p->pos);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos);

			if(hash == -1)
				continue;

			/// Calculates the density, Eq.3
			for(Particle* np : Cells[hash].contained_particles)
			{
				m3Vector Distance;
				Distance = p->pos - np->pos;
				float dis2 = (float)Distance.magnitudeSquared();
				// p->dens += np->mass * B_spline(Distance.magnitude());
				p->dens += np->mass * Poly6(dis2);
			}
		}

		 p->dens += p->mass * Poly6(0.0f);

		/// Calculates the pressure, Eq.12
		p->pres = K * (p->dens - Stand_Density);

		/// Testing if voltage can be used as a pressure
		// if(isStimOn)
			// m3Real inter_pressure_voltage = (p->Vm * voltage_constant);
			p->pres -= (p->Vm * voltage_constant);

			if(p->pres < -max_pressure)
				p->pres = -max_pressure;
			else if(p->pres > max_pressure)
				p->pres = max_pressure;
			// p->pres = p->pres < 0.0f? 0.0f : p->pres;

		// if(p->pos.x > 0.5 && p->pos.y > 0.2 && p->pos.z > 0.5 )
		// {
		// 	p->pres += 20.0 * Stand_Density;
			// cout << p->pres<< " " << p->Vm << " " << inter_pressure_voltage << endl;
		// }
	}
}

void SPH_SM_monodomain::Compute_Force()
{
	Particle *p;
	m3Vector CellPos;
	m3Vector NeighborPos;
	int hash;

	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];

		p->acc = m3Vector(0.0f, 0.0f, 0.0f);
		p->Inter_Vm = 0.0f;

		CellPos = Calculate_Cell_Position(p->pos);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos);
			if(hash == -1)
				continue;

			for(Particle* np : Cells[hash].contained_particles)
			{
				m3Vector Distance;
				Distance = p->pos - np->pos;
				float dis2 = (float)Distance.magnitudeSquared();

				if(dis2 > INF)
				{
					float dis = sqrt(dis2);

					/// Calculates the force of pressure, Eq.10
					float Volume = np->mass / np->dens;
					// float Force_pressure = Volume * (p->pres+np->pres)/2 * B_spline_1(dis);
					float Force_pressure = Volume * (p->pres+np->pres)/2 * Spiky(dis);
					p->acc -= Distance * Force_pressure / dis;

					/// Calculates the relative velocity (vj - vi), and then multiplies it to the mu, volume, and viscosity kernel. Eq.14
					// m3Vector RelativeVel = np->corrected_vel - p->corrected_vel;
					m3Vector RelativeVel = np->inter_vel - p->inter_vel;
					float Force_viscosity = Volume * mu * Visco(dis);
					p->acc += RelativeVel * Force_viscosity;

					/// Calculates the intermediate voltage needed for the monodomain model
					p->Inter_Vm += (np->Vm - p->Vm) * Volume * B_spline_2(dis);
				}
			}
		}
		/// Sum of the forces that make up the fluid, Eq.8
		p->acc = p->acc/p->dens;

		/// Adding the currents, and time integration for the intermediate voltage
		p->Inter_Vm += (sigma / (Beta * Cm)) * p->Inter_Vm - ((p->Iion - p->stim * Time_Delta / p->mass) / Cm);
	}
}

void SPH_SM_monodomain::calculate_cell_model()
{
	Particle *p;

	m3Real denom = (FH_Vp - FH_Vr);
	m3Real asd = (FH_Vt - FH_Vr)/denom;
	m3Real u = 0.0;

	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];

		u = (p->Vm - FH_Vr) / denom;

		p->Iion += Time_Delta * (C1*u*(u - asd)*(u - 1.0) + C2* p->w) / p->mass ;
		
		p->w += Time_Delta * C3*(u - C4*p->w) / p->mass;
	}
}


/// Time integration as in 2016 - Fluid simulation by the SPH Method, a survey.
/// Eq.13 and Eq.14
void SPH_SM_monodomain::Update_Properties()
{
	Particle *p;

	for(int i=0; i < Number_Particles; i++)
	{
		p = &Particles[i];

		if(!p->mFixed)
		{
			p->vel = p->inter_vel + (p->acc*Time_Delta/p->mass);
			p->pos = p->pos + (p->vel*Time_Delta);
		}

		p->Vm += p->Inter_Vm * Time_Delta / p->mass;
		if(p->Vm > max_voltage)
			p->Vm = max_voltage;
		else if(p->Vm < -max_voltage)
			p->Vm = -max_voltage;

		if(p->pos.x < 0.0f)
		{
			p->vel.x = p->vel.x * Wall_Hit;
			p->pos.x = 0.0f;
		}
		if(p->pos.x >= World_Size.x)
		{
			p->vel.x = p->vel.x * Wall_Hit;
			p->pos.x = World_Size.x - 0.0001f;
		}
		if(p->pos.y < 0.0f)
		{
			p->vel.y = p->vel.y * Wall_Hit;
			p->pos.y = 0.0f;
		}
		if(p->pos.y >= World_Size.y)
		{
			p->vel.y = p->vel.y * Wall_Hit;
			p->pos.y = World_Size.y - 0.0001f;
		}
		if(p->pos.z < 0.0f)
		{
			p->vel.z = p->vel.z * Wall_Hit;
			p->pos.z = 0.0f;
		}
		if(p->pos.z >= World_Size.z)
		{
			p->vel.z = p->vel.z * Wall_Hit;
			p->pos.z = World_Size.z - 0.0001f;
		}

		bounds.clamp(Particles[i].pos);
	}
}

void SPH_SM_monodomain::calculate_corrected_velocity()
{
	/// Computes predicted velocity from forces except viscoelastic and pressure
	apply_external_forces();

	/// Calculates corrected velocity
	projectPositions();

	m3Real time_delta_1 = 1.0f / Time_Delta;

	for (int i = 0; i < Number_Particles; i++)
	{
		Particles[i].corrected_vel = Particles[i].predicted_vel + (Particles[i].mGoalPos - Particles[i].pos) * time_delta_1 * alpha;
	}
}

void SPH_SM_monodomain::calculate_intermediate_velocity()
{
	Particle *p;
	m3Vector CellPos;
	m3Vector NeighborPos;
	int hash;

	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];
		CellPos = Calculate_Cell_Position(p->pos);
		m3Vector partial_velocity(0.0f, 0.0f, 0.0f);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos);
			if(hash == -1)
				continue;

			for(Particle* np : Cells[hash].contained_particles)
			{
				m3Vector Distance;
				Distance = p->pos - np->pos;
				float dis2 = (float)Distance.magnitudeSquared();
				partial_velocity += (np->corrected_vel - p->corrected_vel) * Poly6(dis2) * (np->mass / np->dens);
			}
		}
		p->inter_vel = p->corrected_vel + partial_velocity * velocity_mixing;
	}
}


void SPH_SM_monodomain::set_stim(m3Vector center, m3Real radius, m3Real stim_strength)
{
	Particle *p;
	isStimOn = true;
	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];
		m3Vector position = p->pos;
		if (((position.x-center.x)*(position.x-center.x)+(position.y-center.y)*(position.y-center.y)+(position.z-center.z)*(position.z-center.z)) <= radius)
		{
			p->stim = stim_strength;
		}
	}
}

void SPH_SM_monodomain::turnOnStim_Cube(std::vector<m3Vector> positions)
{
	m3Vector cm;
	Particle *p;

	for(m3Vector pos : positions)
	{
		cm += pos;
		if( (pos.x >= 0.45 && pos.x <= 0.48) || (pos.x > 1.0 && pos.z <= 1.05f) )
			set_stim(pos, 0.001f, stim_strength);
	}
	cm /= Number_Particles;
	// set_stim(m3Vector(0.3,0.0,0.7), 0.001f, stim_strength);
	// set_stim(cm, 0.001f, stim_strength);
	
	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];
		m3Vector position = p->pos;
		if ((position.y == 0.0f && position.x <= 0.48f) || (position.y == 0.0f && position.x >= 1.0))
			p->mFixed = true;
	}

	// cout<<"Particles stimulated."<<endl;
}

void SPH_SM_monodomain::turnOnStim_Mesh(std::vector<m3Vector> positions)
{
	m3Vector cm;
	Particle *p;

	for(m3Vector pos : positions)
	{
		// if( (pos.x >= 0.3 && pos.x <= 0.5) || (pos.x > 1.1f && pos.x <= 1.29f) )
		set_stim(pos, 0.01f, stim_strength);
	}

	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];
		if ((p->pos.x >= 0.05 && p->pos.x <= 0.1) || (p->pos.x >= 0.94 && p->pos.y >= 0.75))
			p->mFixed = true;
	}
}

void SPH_SM_monodomain::turnOffStim()
{
	Particle *p;
	isStimOn = false;
	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];
		p->stim = -1000.0f;
		p->Vm = 0.0f;
		p->Inter_Vm = 0.0f;
		p->Iion = 0.0f;
		p->pres = 0.0f;
		p->w = 0.0f;

		// if(p->stim > 0.0f)
		// {
		// 	p->stim = 0.0f;
		// }
	}
}

void SPH_SM_monodomain::print_report(double avg_fps, double avg_step_d)
{
	// cout << "Avg FPS ; Avg Step Duration ; Time Steps ; Find neighbors ; Corrected Velocity ; Intermediate Velocity ; Density-Pressure ; Cell model ; Compute Force ; Update Properties ; K ; Alpha ; Beta ; Mu ; sigma ; Stim strength ; FH_VT ; FH_VP ; FH_VR ; C1 ; C2 ; C3 ; C4" << endl;
	
	cout << avg_fps << ";" << avg_step_d << ";" << total_time_steps << ";" << d_find_neighbors.count() / total_time_steps << ";" << d_corrected_velocity.count() / total_time_steps << ";" << d_intermediate_velocity.count() / total_time_steps << ";" << d_Density_SingPressure.count() / total_time_steps << ";" << d_cell_model.count() / total_time_steps << ";" << d_compute_Force.count() / total_time_steps << ";" << d_Update_Properties.count() / total_time_steps << ";";

	cout << K << ";" << alpha << ";" << beta  << ";" << mu << ";" << sigma << ";" << stim_strength << ";" << FH_Vt << ";" << FH_Vp << ";" << FH_Vr << ";" << C1 << ";" << C2 << ";" << C3 << ";" << C4 << endl;
}

void SPH_SM_monodomain::compute_SPH_SM_monodomain()
{
	tpoint tstart = std::chrono::system_clock::now();
	Find_neighbors();
	d_find_neighbors += std::chrono::system_clock::now() - tstart;

	tstart = std::chrono::system_clock::now();
	calculate_corrected_velocity();
	d_corrected_velocity += std::chrono::system_clock::now() - tstart;
	
	tstart = std::chrono::system_clock::now();
	calculate_intermediate_velocity();
	d_intermediate_velocity += std::chrono::system_clock::now() - tstart;
	
	tstart = std::chrono::system_clock::now();
	Compute_Density_SingPressure();
	d_Density_SingPressure += std::chrono::system_clock::now() - tstart;
	
	tstart = std::chrono::system_clock::now();
	calculate_cell_model();
	d_cell_model += std::chrono::system_clock::now() - tstart;

	tstart = std::chrono::system_clock::now();
	Compute_Force();
	d_compute_Force += std::chrono::system_clock::now() - tstart;

	tstart = std::chrono::system_clock::now();
	Update_Properties();
	d_Update_Properties += std::chrono::system_clock::now() - tstart;

	total_time_steps++;
}

void SPH_SM_monodomain::Animation()
{
	compute_SPH_SM_monodomain();
}


#endif