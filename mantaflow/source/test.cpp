/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Use this file to test new functionality
 *
 ******************************************************************************/

#include "levelset.h"
#include "commonkernels.h"
#include "particle.h"
#include <cmath>

using namespace std;

namespace Manta {

// two simple example kernels

KERNEL(idx, reduce=+) returns (double sum=0)
double reductionTest(const Grid<Real>& v)
{
	sum += v[idx];
}

KERNEL(idx, reduce=min) returns (double sum=0)
double minReduction(const Grid<Real>& v)
{
	if (sum < v[idx])
		sum = v[idx];
}



//! remove particles at top
PYTHON() void deleteTopParts( BasicParticleSystem& parts, Real maxHeight, LevelsetGrid* phi=NULL)
{
	for (IndexInt idx=0; idx<(IndexInt)parts.size(); idx++) {
		if (!parts.isActive(idx)) continue;

		if (parts.getPos(idx)[1]>maxHeight) {
			parts.kill(idx); // out of domain, remove
		}
	}

	// fix levelset as well...
	if(phi) {
		FOR_IJK(*phi) { 
			if(j>maxHeight) {
				if((*phi)(i,j,k)<0.) {
					(*phi)(i,j,k) = (j-maxHeight-1) + 0.5;
				}
			}
		} 
	}
}



// ... add more test code here if necessary ...

} //namespace

