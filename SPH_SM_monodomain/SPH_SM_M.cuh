extern "C"
{
	/// Calculates the cell position for each particle
	void calcHash(uint *gridParticleHash, uint * gridParticleIndex, m3Vector *pos, int numberParticles);

	/// Sorts the hashes and indices
	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numberParticles);

	/// Reorders the particles based on the hashes and indices. Also finds the start and end cells.
	void reorderDataAndFindCellStart(Particles *p, uint *dCellStart, uint *dCellEnd, uint *dGridParticleHash, uint *dGridParticleIndex, uint Number_Particles, uint Number_Cells);

	void calculate_cell_model(Particles *particles, int Number_Particles, m3Real Time_Delta)

	void calculate_intermediate_velocity(Particles *particles, uint *dCellStart, uint *dCellEnd, uint *dGridParticleHash, uint *dGridParticleIndex, uint Number_Particles, uint Number_Cells)
}