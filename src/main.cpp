#include "global_setup.h"
#include "global_class.h"

// Data struct
using Device = Kokkos::DefaultExecutionSpace;
using D_Data2d = Kokkos::View<real_t **, Device>;
using H_Data2d = D_Data2d::HostMirror;
using D_Data3d = Kokkos::View<real_t ***, Device>;
using H_Data3d = D_Data3d::HostMirror;

enum KokkosLayout
{
    KOKKOS_LAYOUT_LEFT = 0,
    KOKKOS_LAYOUT_RIGHT = 1
};

// Transfer id and i,j,k
/* 2D */
KOKKOS_INLINE_FUNCTION void index2coord(int index, int &i, int &j, int Nx, int Ny)
{
#ifdef KOKKOS_ENABLE_CUDA // left layout
    j = index / Nx;
    i = index - j * Nx;
#else // right layout
    i = index / Ny;
    j = index - i * Ny;
#endif
}

KOKKOS_INLINE_FUNCTION int coord2index(int i, int j, int Nx, int Ny)
{
#ifdef KOKKOS_ENABLE_CUDA // left layout
    return i + Nx * j;
#else // right layout
    return j + Ny * i;
#endif
}

/* 3D */
KOKKOS_INLINE_FUNCTION void index2coord(int index, int &i, int &j, int &k, int Nx, int Ny, int Nz)
{
#ifdef KOKKOS_ENABLE_CUDA // left layout
    int NxNy = Nx * Ny;
    k = index / NxNy;
    j = (index - k * NxNy) / Nx;
    i = index - j * Nx - k * NxNy;
#else // right layout
    int NyNz = Ny * Nz;
    i = index / NyNz;
    j = (index - i * NyNz) / Nz;
    k = index - j * Nz - i * NyNz;
#endif
}

KOKKOS_INLINE_FUNCTION int coord2index(int i, int j, int k, int Nx, int Ny, int Nz)
{
#ifdef KOKKOS_ENABLE_CUDA // left layout
    return i + Nx * j + Nx * Ny * k;
#else // right layout
    return k + Nz * j + Nz * Ny * i;
#endif
}

int main(int argc, char *argv[])
{
}