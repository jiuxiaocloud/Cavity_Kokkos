#pragma once
#include <ctime>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "global_setup.h"
#include "../external/mpiUtils/MpiCommCart.h"
using namespace mpiUtils;

#define DIM_X 1
#define DIM_Y 1

// All kinds of Boundary Conditions
enum BConditions
{
    Inflow = 0,
    Outflow = 1,
    Symmetry = 2,
    Periodic = 3,
    Wall = 4,
    BC_COPY = 5,
    BC_UNDEFINED = 6
};

struct Block
{
    //--for-Mesh-----------------------------
    int Xmax, Ymax;
    //--for-Mpi: mx means number of ranks in x direction, myMpiPos_x means location of this rank in x-dir ranks(from 0 to mx-1)
    int mx, my, myMpiPos_x, myMpiPos_y;
    // constructor
    Block(){};
    Block(int xmax, int ymax, int mx, int my) : Xmax(xmax), Ymax(ymax), mx(mx), my(my){};
};

/**
 * Attention: MpiTrans only allocate memory automatically and transfer messages when the trans-function be called,
 * you must assignment the values in buffer youself
 */
struct MpiTrans
{
    // runtime determination if we are using float or double (for MPI communication)
    int data_type;
    // MPI rank of current process
    int myRank;
    // number of MPI processes
    int nProcs;
    // number of MPI process neighbors (4 in 2D and 6 in 3D)
    int nNeighbors;
    // MPI rank of adjacent MPI processes
    int neighborsRank[6];
    // Global MPI Communicating Group
    MPI_Group comm_world;
    // MPI communicator in a cartesian virtual topology
    MpiCommCart *communicator;
    // Boundary condition type with adjacent domains (corresponding to neighbor MPI processes)
    BConditions neighborsBC[6];
    // Data buffer for aware or nan-aware-mpi : aware-mpi trans device buffer directly but nan-aware-mpi trans host buffer only

    MpiTrans(){};
    MpiTrans(Block &bl, BConditions const Boundarys[6]);
};

void getInput(std::string filename, real_t &Re, real_t &U, real_t &rho0, int &NX, int &NY);

class BaseSolver
{
protected:
    size_t NX, NY;
    real_t Re, U, rho0, nu, tau, omega, O_rhoba, O_Fd, O_cd, L2Error;
    D_Data1d ex, ey, w;
    D_IData1d iex, iey;
    D_Data2d rho, ux, uy, ux0, uy0, stream_func, vorticity;
    H_Data2d h_rho, h_ux, h_uy, h_stream_func, h_vorticity;
    D_Data3d f, fb;
    // TransSize
    // CellSz: total number of the cell points transferred by Mpi ranks, needed by mpisendrecv-function
    // DataSz: total sizeof-data(bytes) of all physical arguments in these cell points used for malloc memory
    int Ghost_CellSz_x, Ghost_DataSz_x, Ghost_CellSz_y, Ghost_DataSz_y;
    // TransBuffer
    D_Data2d d_TransBufSend_xmin, d_TransBufSend_xmax, d_TransBufRecv_xmin, d_TransBufRecv_xmax;
    D_Data2d d_TransBufSend_ymin, d_TransBufSend_ymax, d_TransBufRecv_ymin, d_TransBufRecv_ymax;
    H_Data2d h_TransBufSend_xmin, h_TransBufSend_xmax, h_TransBufRecv_xmin, h_TransBufRecv_xmax;
    H_Data2d h_TransBufSend_ymin, h_TransBufSend_ymax, h_TransBufRecv_ymin, h_TransBufRecv_ymax;
    // used for MPI
    Block bl;
    int rank, nranks, MX, MY;
    const BConditions BCs[6] = {Wall, Wall, Wall, Wall, Wall, Wall};

public:
    MpiTrans mpiTrans;

    BaseSolver(size_t nx, size_t ny, real_t Re_, real_t U_, real_t rho_, int mx);
    void Initialize();
    void Evolution(int tn);
    void Collision();
    void Propagation(int tn);
    void Boundary();
    void Update();
    real_t Error();
    void StreamFunction();
    void VorticityFunction();
    void Cal_cd_Fd();
    void MpiTransBufX();
    void MpiTransBufY();
    long double AllocMemory(const int N);
    void Output(const int m, std::string Path);
};

class Timer
{
private:
    std::chrono::high_resolution_clock::time_point global_start_time, global_end_time;

public:
    float time;
    Timer() { global_start_time = std::chrono::high_resolution_clock::now(); };
    float OutTime()
    {
        global_end_time = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<float, std::milli>(global_end_time - global_start_time).count();
        return time;
    };
};