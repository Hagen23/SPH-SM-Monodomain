#pragma once
#ifndef M3_VECTOR_H
#define M3_VECTOR_H

//---------------------------------------------------------------------------

#include "m3Real.h"
#include <cassert>

//---------------------------------------------------------------------------
class m3Vector
	//---------------------------------------------------------------------------
{
public:
	__host__ __device__ inline m3Vector() { zero(); }
	__host__ __device__ inline m3Vector(const m3Vector& v0) :x(v0.x), y(v0.y), z(v0.z) { }//*this = v0; }
	__host__ __device__ inline m3Vector(m3Real x0, m3Real y0, m3Real z0) { x = x0; y = y0; z = z0; }
	__host__ __device__ inline void set(m3Real x0, m3Real y0, m3Real z0) { x = x0; y = y0; z = z0; }
	__host__ __device__ inline void zero() { x = 0.0; y = 0.0; z = 0.0; }
	__host__ __device__ inline bool isZero() { return x == 0.0 && y == 0.0 && z == 0.0; }

	__host__ __device__ m3Real & operator[] (int i) {
		assert(i >= 0 && i <= 2);
		return (&x)[i];
	}

	__host__ __device__ m3Vector& operator = (m3Vector rhs)
	{
		x = rhs.x; y = rhs.y; z = rhs.z;
		return *this;
	}

	__host__ __device__ bool operator == (const m3Vector &v) const {
		return (x == v.x) && (y == v.y) && (z ==v.z);
	}

	__host__ __device__ m3Vector operator + (const m3Vector &v) const {
		m3Vector r; 
		r.x = x + v.x; r.y = y + v.y; r.z = z + v.z;
		return r;
	}

	__host__ __device__ m3Vector operator - (const m3Vector &v) const {
		m3Vector r; r.x = x - v.x; r.y = y - v.y; r.z = z - v.z;
		return r;
	}
	__host__ __device__ void operator += (const m3Vector &v) {
		x += v.x; y += v.y; z += v.z;
	}
	__host__ __device__ void operator -= (const m3Vector &v) {
		x -= v.x; y -= v.y; z -= v.z;
	}
	__host__ __device__ void operator *= (const m3Vector &v) {
		x *= v.x; y *= v.y; z *= v.z;
	}
	__host__ __device__ void operator /= (const m3Vector &v) {
		x /= v.x; y /= v.y; z /= v.z;
	}
	__host__ __device__ m3Vector operator -() const {
		m3Vector r; 
		r.x = -x; r.y = -y; r.z = -z;
		return r;
	}
	__host__ __device__ m3Vector operator * (const m3Real f) const {
		m3Vector r; 
		r.x = x*f; r.y = y*f; r.z = z*f;
		return r;
	}
	__host__ __device__ m3Vector operator / (const m3Real f) const {
		m3Vector r;
		r.x = x / f; r.y = y / f; r.z = z / f;
		return r;
	}
	
	__host__ __device__ m3Vector cross(const m3Vector &v1, const m3Vector &v2) const {
		return m3Vector(v1.y*v2.z - v1.z *v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
	}

	__host__ __device__ inline m3Real dot(const m3Vector &v) const {
		return x*v.x + y*v.y + z*v.z;
	}

	__host__ __device__ inline void minimum(const m3Vector &v) {
		if (v.x < x) x = v.x;
		if (v.y < y) y = v.y;
		if (v.z < z) z = v.z;
	}
	__host__ __device__ inline void maximum(const m3Vector &v) {
		if (v.x > x) x = v.x;
		if (v.y > y) y = v.y;
		if (v.z > z) z = v.z;
	}

	__host__ __device__ inline m3Real magnitudeSquared() const { return x*x + y*y + z*z; }
	__host__ __device__ inline m3Real magnitude() const { return sqrt(x*x + y*y + z*z); }

	__host__ __device__ inline m3Real distanceSquared(const m3Vector &v) const {
		m3Real dx, dy, dz; 
		dx = v.x - x; dy = v.y - y; dz = v.z - z;
		return dx*dx + dy*dy + dz*dz;
	}

	__host__ __device__ inline m3Real distance(const m3Vector &v) const {
		m3Real dx, dy, dz; dx = v.x - x; dy = v.y - y; dz = v.z - z;
		return sqrt(dx*dx + dy*dy + dz*dz);
	}

	__host__ __device__ void operator *=(m3Real f) { x *= f; y *= f; z *= f; }
	__host__ __device__ void operator /=(m3Real f) { x /= f; y /= f; z /= f; }

	__host__ __device__ void normalize() 
	{
		m3Real l = magnitude();
		m3Vector v;
		if (l != 0.0f)
		{
			m3Real l1 = 1.0f / l; 
			x *= l1; y *= l1; z*=l1;
		}
	}

	// ------------------------------
	m3Real x, y, z;
};


#endif