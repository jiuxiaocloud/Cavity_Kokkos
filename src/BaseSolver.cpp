#include "global_class.h"

void getInput(std::string filename, real_t &Re, real_t &U, real_t &rho0, int &NX, int &NY)
{
    std::ifstream inFile(filename);
    std::string tmp;
    std::getline(inFile, tmp);
    int idx = tmp.rfind(' ');
    Re = std::stod(tmp.substr(idx + 1));

    std::getline(inFile, tmp);
    idx = tmp.rfind(' ');
    U = std::stod(tmp.substr(idx + 1));

    std::getline(inFile, tmp);
    idx = tmp.rfind(' ');
    rho0 = std::stod(tmp.substr(idx + 1));

    std::getline(inFile, tmp);
    idx = tmp.rfind(' ');
    NX = std::stoi(tmp.substr(idx + 1));

    std::getline(inFile, tmp);
    idx = tmp.rfind(' ');
    NY = std::stoi(tmp.substr(idx + 1));

    inFile.close();
}

KOKKOS_INLINE_FUNCTION real_t feq(int k, real_t rho, real_t ux, real_t uy, D_Data1d ex, D_Data1d ey, D_Data1d w)
{
    real_t eu = ex(k) * ux + ey(k) * uy;
    real_t uv = ux * ux + uy * uy;
    real_t FEQ = w(k) * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
    return FEQ;
}

BaseSolver::BaseSolver(size_t nx, size_t ny, real_t Re_, real_t U_, real_t rho_, int mx) : NX(nx), NY(ny), Re(Re_), U(U_), rho0(rho_), MX(mx)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MY = nranks / MX;
    bl = Block(nx, ny, MX, MY);
    mpiTrans = MpiTrans(bl, BCs), AllocMemory(1);

    nu = U * NX / Re, tau = 3.0 * nu + 0.5, omega = 1.0 / tau;
    iex = D_IData1d("iex", Q), iey = D_IData1d("iey", Q), ex = D_Data1d("ex", Q), ey = D_Data1d("ey", Q), w = D_Data1d("w", Q);
    ux = D_Data2d("ux", NX, NY), uy = D_Data2d("uy", NX, NY), ux0 = D_Data2d("ux0", NX, NY), uy0 = D_Data2d("uy0", NX, NY);
    rho = D_Data2d("rho", NX, NY), stream_func = D_Data2d("stream_func", NX, NY), vorticity = D_Data2d("vorticity", NX, NY);
    f = D_Data3d("f", NX, NY, Q), fb = D_Data3d("fb", NX, NY, Q);
    Kokkos::deep_copy(rho, rho0);
    h_rho = Kokkos::create_mirror_view(rho);
    h_ux = Kokkos::create_mirror_view(ux), h_uy = Kokkos::create_mirror_view(uy);
    h_stream_func = Kokkos::create_mirror_view(stream_func), h_vorticity = Kokkos::create_mirror_view(vorticity);
    Initialize();
}

void BaseSolver::Initialize()
{
    Kokkos::parallel_for(
        1, KOKKOS_CLASS_LAMBDA(const int64_t id) {
            // const int iex[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
            iex(0) = 0, iex(1) = 1, iex(2) = 0, iex(3) = -1, iex(4) = 0, iex(5) = 1, iex(6) = -1, iex(7) = -1, iex(8) = 1;
            // const real_t ex[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
            ex(0) = 0, ex(1) = 1, ex(2) = 0, ex(3) = -1, ex(4) = 0, ex(5) = 1, ex(6) = -1, ex(7) = -1, ex(8) = 1;
            // const int iey[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
            iey(0) = 0, iey(1) = 0, iey(2) = 1, iey(3) = 0, iey(4) = -1, iey(5) = 1, iey(6) = 1, iey(7) = -1, iey(8) = -1;
            // const real_t ey[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
            ey(0) = 0, ey(1) = 0, ey(2) = 1, ey(3) = 0, ey(4) = -1, ey(5) = 1, ey(6) = 1, ey(7) = -1, ey(8) = -1;
            // const real_t w[Q] = {w0, w1, w1, w1, w1, w2, w2, w2, w2};
            w(0) = w0, w(1) = w1, w(2) = w1, w(3) = w1, w(4) = w1, w(5) = w2, w(6) = w2, w(7) = w2, w(8) = w2;
        });
    Kokkos::fence();

    Kokkos::parallel_for(
        NX * NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {
            int i, j;
            index2coord(id, i, j, NX, NY);
            // rho(i, j) = rho0;
            // ux(i, j) = sqrt(real_t(i * i * j) / NX), uy(i, j) = sqrt(real_t(i * j * j) / NY);
            ux(i, j) = 0.0, uy(i, j) = 0.0;
            ux0(i, j) = 0.0, uy0(i, j) = 0.0;
            for (size_t k = 0; k < Q; k++)
                f(i, j, k) = feq(k, rho(i, j), ux(i, j), uy(i, j), ex, ey, w); // FEQ; //
        });
    Kokkos::fence();

    // // 左右边界
    // if (mpiTrans.neighborsBC[X_MIN] != BC_COPY)
    // {
    //     Kokkos::parallel_for(
    //         NY, KOKKOS_CLASS_LAMBDA(const int64_t j) { rho(0, j) = -1; });
    // }
    // if (mpiTrans.neighborsBC[X_MAX] != BC_COPY)
    // {
    //     Kokkos::parallel_for(
    //         NY, KOKKOS_CLASS_LAMBDA(const int64_t j) { rho(NX - 1, j) = -1; });
    // }
    // Kokkos::fence();
    // // 上下边界
    // if (mpiTrans.neighborsBC[Y_MIN] != BC_COPY)
    // {
    //     Kokkos::parallel_for(
    //         NX, KOKKOS_CLASS_LAMBDA(const int64_t i) { rho(i, 0) = -1; });
    // }
    // if (mpiTrans.neighborsBC[Y_MAX] != BC_COPY)
    // {
    //     Kokkos::parallel_for(
    //         NX, KOKKOS_CLASS_LAMBDA(const int64_t i) { rho(i, NY - 1) = -1; });
    // }
    // Kokkos::fence();
}

void BaseSolver::Evolution()
{
    Collision();
    Propagation();
    Boundary();
    Update();
}

void BaseSolver::Collision()
{
    Kokkos::parallel_for(
        NX * NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {
            int i, j;
            index2coord(id, i, j, NX, NY);
            for (size_t k = 0; k < Q; k++)
                fb(i, j, k) = (1.0 - omega) * f(i, j, k) + omega * feq(k, rho(i, j), ux(i, j), uy(i, j), ex, ey, w); //* FEQ; //
        });

    Kokkos::fence();
}

void BaseSolver::Propagation()
{
    Kokkos::parallel_for(
        NX * NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {
    int i, j;
    index2coord(id, i, j, NX, NY);
    for (size_t k = 0; k < Q; k++)
    {
        int newi = (i + iex(k) + NX) % NX;
        int newj = (j + iey(k) + NY) % NY;
        f(newi, newj, k) = fb(i, j, k);
    } });
    Kokkos::fence();

    if (MX > 1)
    {
        Kokkos::parallel_for( // // Xmin Copy
            NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {
                for (size_t k = 0; k < Q; k++)
                    d_TransBufSend_xmin(id, k) = fb(0, id, k);
            });
        Kokkos::parallel_for( // // Xmax Copy
            NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {
                for (size_t k = 0; k < Q; k++)
                    d_TransBufSend_xmax(id, k) = fb(NX - 1, id, k);
            });
        Kokkos::fence();

        MpiTransBufX();

        Kokkos::parallel_for( // // Xmin Copy Back
            NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {
                for (size_t k = 0; k < Q; k++)
                    if (1 == iex(k))
                        f(0, id, k) = d_TransBufRecv_xmin(id, k);
            });
        Kokkos::parallel_for( // // Xmax Copy Back
            NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {
                for (size_t k = 0; k < Q; k++)
                    if (-1 == iex(k))
                        f(NX - 1, id, k) = d_TransBufRecv_xmax(id, k);
            });
        Kokkos::fence();
    }

    if (MY > 1)
    {
        Kokkos::parallel_for( // // Ymin Copy
            NX, KOKKOS_CLASS_LAMBDA(const int64_t id) {
                for (size_t k = 0; k < Q; k++)
                    d_TransBufSend_ymin(id, k) = fb(id, 0, k);
            });
        Kokkos::parallel_for( // // Ymax Copy
            NX, KOKKOS_CLASS_LAMBDA(const int64_t id) {
                for (size_t k = 0; k < Q; k++)
                    d_TransBufSend_ymax(id, k) = fb(id, NY - 1, k);
            });
        Kokkos::fence();

        MpiTransBufY();

        Kokkos::parallel_for( // // Ymin Copy Back
            NX, KOKKOS_CLASS_LAMBDA(const int64_t id) {
                for (size_t k = 0; k < Q; k++)
                    if (1 == iey(k))
                        f(id, 0, k) = d_TransBufRecv_ymin(id, k);
            });
        Kokkos::parallel_for( // // Ymax Copy Back
            NX, KOKKOS_CLASS_LAMBDA(const int64_t id) {
                for (size_t k = 0; k < Q; k++)
                    if (-1 == iey(k))
                        f(id, NY - 1, k) = d_TransBufRecv_ymax(id, k);
            });
        Kokkos::fence();
    }
}

void BaseSolver::Boundary()
{
    // 左右边界
    if (mpiTrans.neighborsBC[X_MIN] != BC_COPY)
    {
        Kokkos::parallel_for(
            NY, KOKKOS_CLASS_LAMBDA(const int64_t j) {
        f(0,j,1) = fb(0,j,3);
        f(0,j,5) = fb(0,j,7);
        f(0,j,8) = fb(0,j,6); });
    }
    if (mpiTrans.neighborsBC[X_MAX] != BC_COPY)
    {
        Kokkos::parallel_for(
            NY, KOKKOS_CLASS_LAMBDA(const int64_t j) {
        f(NX - 1,j,3) = fb(NX - 1,j,1);
        f(NX - 1,j,6) = fb(NX - 1,j,8);
        f(NX - 1,j,7) = fb(NX - 1,j,5); });
    }
    Kokkos::fence();

    // 上下边界
    if (mpiTrans.neighborsBC[Y_MIN] != BC_COPY)
    {
        Kokkos::parallel_for(
            NX, KOKKOS_CLASS_LAMBDA(const int64_t i) {
            f(i,0,2) = fb(i,0,4);
            f(i,0,5) = fb(i,0,7);
            f(i,0,6) = fb(i,0,8); });
    }
    if (mpiTrans.neighborsBC[Y_MAX] != BC_COPY)
    {
        Kokkos::parallel_for(
            NX, KOKKOS_CLASS_LAMBDA(const int64_t i) {
            f(i,NY - 1,4) = fb(i,NY - 1,2);
            f(i,NY - 1,7) = fb(i,NY - 1,5) - rho0 * U / 6.0;
            f(i,NY - 1,8) = fb(i,NY - 1,6) + rho0 * U / 6.0; });
    }
    Kokkos::fence();
}

void BaseSolver::Update()
{
    Kokkos::parallel_for(
        NX * NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {
    int i, j;
    index2coord(id, i, j, NX, NY);
			rho(i,j) = f(i,j,0) + f(i,j,1) + f(i,j,2) + f(i,j,3) + f(i,j,4) + f(i,j,5) + f(i,j,6) + f(i,j,7) + f(i,j,8);
			ux(i,j) = ((f(i,j,1) + f(i,j,5) + f(i,j,8)) - (f(i,j,3) + f(i,j,6) + f(i,j,7))) / rho(i,j);
			uy(i,j) = ((f(i,j,2) + f(i,j,5) + f(i,j,6)) - (f(i,j,4) + f(i,j,7) + f(i,j,8))) / rho(i,j); });

    Kokkos::fence();
}

real_t BaseSolver::Error()
{
    D_Data temp1("temp1"), temp2("temp2");
    Kokkos::Sum<real_t, MemSpace> redutm1(temp1), redutm2(temp2);
    Kokkos::parallel_reduce(
        NX * NY, KOKKOS_CLASS_LAMBDA(const int64_t id, real_t &ltemp1, real_t &ltemp2) {
            int i, j;
            index2coord(id, i, j, NX, NY);
            ltemp1 += (ux(i, j) - ux0(i, j)) * (ux(i, j) - ux0(i, j)) + (uy(i, j) - uy0(i, j)) * (uy(i, j) - uy0(i, j));
            ltemp2 += ux(i, j) * ux(i, j) + uy(i, j) * uy(i, j);
            ux0(i, j) = ux(i, j);
            uy0(i, j) = uy(i, j);
        },
        redutm1, redutm2);
    real_t O_tm1, O_tm2, MPI_tm1, MPI_tm2;
    Kokkos::fence(), Kokkos::deep_copy(O_tm1, temp1), Kokkos::deep_copy(O_tm2, temp2), Kokkos::fence();
    mpiTrans.communicator->synchronize();
    mpiTrans.communicator->allReduce(&O_tm1, &MPI_tm1, 1, mpiTrans.data_type, mpiUtils::MpiComm::SUM);
    mpiTrans.communicator->allReduce(&O_tm2, &MPI_tm2, 1, mpiTrans.data_type, mpiUtils::MpiComm::SUM);
    mpiTrans.communicator->synchronize();
    O_tm1 = sqrt(MPI_tm1);
    O_tm2 = sqrt(MPI_tm2);
    L2Error = O_tm1 / (O_tm2 + 1e-30);

    return L2Error;
}

void BaseSolver::StreamFunction()
{
    // 利用拉格朗日三点插值，插值出曲线，之后积分即可
    Kokkos::parallel_for(
        NY, KOKKOS_CLASS_LAMBDA(const int64_t j) {
            stream_func(0, j) = -(0.291667 * uy(0, j) - 0.0138889 * uy(1, j));
            stream_func(1, j) = -(1.125 * uy(0, j) + 0.375 * uy(1, j));
        });

    Kokkos::fence();

    Kokkos::parallel_for(
        "label", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2, 0}, {NX, NY}), KOKKOS_CLASS_LAMBDA(const int64_t i, const int64_t j) { stream_func(i, j) = stream_func(i - 2, j) - (uy(i - 2, j) + 4.0 * uy(i - 1, j) + uy(i, j)) / 3.0; });

    Kokkos::fence();

    // 无量纲化
    Kokkos::parallel_for(
        NX * NY, KOKKOS_CLASS_LAMBDA(const int64_t id) {int i, j;
    index2coord(id, i, j, NX, NY);
        stream_func(i,j) /= (NX * U); });

    Kokkos::fence();
}

void BaseSolver::VorticityFunction()
{
    Kokkos::parallel_for(
        "label", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {NX - 1, NY - 1}), KOKKOS_CLASS_LAMBDA(const int64_t i, const int64_t j) { vorticity(i, j) = ((uy(i + 1, j) - uy(i - 1, j)) * 0.5 - (ux(i, j + 1) - ux(i, j - 1)) * 0.5) * NX / U; });

    Kokkos::fence();

    // 左右边界
    Kokkos::parallel_for(
        NY - 2, KOKKOS_CLASS_LAMBDA(const int64_t jj) {
            int j = jj + 1;
            vorticity(0, j) = ((uy(0, j) + uy(1, j) / 3.0) - (ux(0, j + 1) - ux(0, j - 1)) * 0.5) * NX / U;                           // 左边界
            vorticity(NX - 1, j) = (-(uy(NX - 1, j) + uy(NX - 2, j) / 3.0) - (ux(NX - 1, j + 1) - ux(NX - 1, j - 1)) * 0.5) * NX / U; // 右边界
        });

    Kokkos::fence();

    Kokkos::parallel_for(
        NX - 2, KOKKOS_CLASS_LAMBDA(const int64_t ii) {
            int i = ii + 1;
            vorticity(i, 0) = ((uy(i + 1, 0) - uy(i - 1, 0)) * 0.5 - (ux(i, 0) + ux(i, 1) / 3.0)) * NX / U;                                          // 下边界
            vorticity(i, NY - 1) = ((uy(i + 1, NY - 1) - uy(i - 1, NY - 1)) * 0.5 - (4.0 * U - 3.0 * ux(i, NY - 1) - ux(i, NY - 2)) / 3.0) * NX / U; // 上边界
        });

    Kokkos::fence();

    // 四个角点
    Kokkos::parallel_for(
        1, KOKKOS_CLASS_LAMBDA(const int64_t id) {
            vorticity(0, 0) = ((uy(0, 0) + uy(1, 0) / 3.0) - (ux(0, 0) + ux(0, 1) / 3.0)) * NX / U;
            vorticity(0, NY - 1) = ((uy(0, 0) + uy(1, 0) / 3.0) - (4.0 * U - 3.0 * ux(0, NY - 1) - ux(0, NY - 2)) / 3.0) * NX / U;
            vorticity(NX - 1, 0) = (-(uy(NX - 1, 0) + uy(NX - 2, 0) / 3.0) - (ux(NX - 1, 0) + ux(NX - 1, 1) / 3.0)) * NX / U;
            vorticity(NX - 1, NY - 1) = (-(uy(NX - 1, NY - 1) + uy(NX - 2, NY - 1) / 3.0) - (4.0 * U - 3.0 * ux(NX - 1, NY - 1) - ux(NX - 1, NY - 2)) / 3.0) * NX / U;
        });

    Kokkos::fence();
}

void BaseSolver::Cal_cd_Fd()
{
    // // 首先计算rho_ba
    D_Data rhoba("rhoba"), Fd("rhoba");
    Kokkos::Sum<real_t, MemSpace> redurho(rhoba), reduFd(Fd);
    Kokkos::parallel_reduce(
        NX * NY, KOKKOS_CLASS_LAMBDA(const int64_t id, real_t &lrho_ba) {
        int i, j;
        index2coord(id, i, j, NX, NY);
        lrho_ba += rho(i, j); },
        redurho);
    // // 计算Fd
    Kokkos::parallel_reduce(
        NX, KOKKOS_CLASS_LAMBDA(const int64_t i, real_t &lFd) { lFd += (U - ux(i, NY - 1)) * 2.0; },
        reduFd);

    Kokkos::fence();
    Kokkos::deep_copy(O_rhoba, rhoba);
    Kokkos::deep_copy(O_Fd, Fd);
    O_rhoba /= real_t(NX * NY), O_Fd *= rho0 * nu;
    O_cd = O_Fd / (O_rhoba * U * U * NX);
}

void BaseSolver::Output(const int m, std::string Path)
{
    StreamFunction();
    VorticityFunction();
    Cal_cd_Fd();
    H_Data3d h_f = Kokkos::create_mirror_view(f), h_fb = Kokkos::create_mirror_view(fb);
    Kokkos::deep_copy(h_f, f);
    Kokkos::deep_copy(h_fb, fb);
    Kokkos::deep_copy(h_ux, ux);
    Kokkos::deep_copy(h_uy, uy);
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_stream_func, stream_func);
    Kokkos::deep_copy(h_vorticity, vorticity);
    Kokkos::fence();
    std::string name = Path + "/cavity_" + std::to_string(m) + "_rank_" + std::to_string(nranks) + "_" + std::to_string(rank) + ".plt";
    std::ofstream out(name.c_str());
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(6);
    out << "Title=\"Lid Driven Flow\"\n"
        << "VARIABLES = \"X\",\"Y\",\"U\",\"V\",\"rho\",\"p\",\"PSI\",\"Vorticity\"\n"
        << "\nZONE T= \"BOX_Fd_" << O_Fd << "_cd_" << O_cd << "_Error_" << L2Error << "\",I= " << NX << ",J = " << NY << ",F = POINT" << std::endl;
    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
        {
            out << ((real_t(i) + 0.5) / (NX * MX) + real_t(1.0) / MX * bl.myMpiPos_x) << " " //* 1
                << ((real_t(j) + 0.5) / (NY * MY) + real_t(1.0) / MY * bl.myMpiPos_y) << " " //* MY
                << h_ux(i, j) / U << " " << h_uy(i, j) / U << " " << h_rho(i, j) / rho0 << " " << (h_rho(i, j) - rho0) / 3.0 << " ";
            out << h_stream_func(i, j) << " ";
            out << h_vorticity(i, j) << " ";
            // for (size_t k = 0; k < Q; k++)
            //     out << h_f(i, j, k) << " ";
            // for (size_t k = 0; k < Q; k++)
            //     out << h_fb(i, j, k) << " ";
            out << std::endl;
        }
}

// =======================================================
// =======================================================
long double BaseSolver::AllocMemory(const int N)
{
    long double temp = 0.0;
    if (N != 0)
    {
#if DIM_X
        Ghost_CellSz_x = bl.Ymax * N, temp += Ghost_CellSz_x * sizeof(real_t) * 4.0 / 1024.0 / 1024.0;
        // =======================================================
        d_TransBufSend_xmin = D_Data2d("d_TransBufSend_xmin", Ghost_CellSz_x, Q);
        d_TransBufRecv_xmin = D_Data2d("d_TransBufRecv_xmin", Ghost_CellSz_x, Q);
        d_TransBufRecv_xmax = D_Data2d("d_TransBufRecv_xmax", Ghost_CellSz_x, Q);
        d_TransBufSend_xmax = D_Data2d("d_TransBufSend_xmax", Ghost_CellSz_x, Q);
        // =======================================================
        h_TransBufSend_xmin = Kokkos::create_mirror_view(d_TransBufSend_xmin);
        h_TransBufRecv_xmin = Kokkos::create_mirror_view(d_TransBufRecv_xmin);
        h_TransBufRecv_xmax = Kokkos::create_mirror_view(d_TransBufRecv_xmax);
        h_TransBufSend_xmax = Kokkos::create_mirror_view(d_TransBufSend_xmax);
        // =======================================================
#endif // end DIM_X

#if DIM_Y
        Ghost_CellSz_y = bl.Xmax * N, temp += Ghost_CellSz_y * sizeof(real_t) * 4.0 / 1024.0 / 1024.0;
        // =======================================================
        d_TransBufSend_ymin = D_Data2d("d_TransBufSend_ymin", Ghost_CellSz_y, Q);
        d_TransBufRecv_ymin = D_Data2d("d_TransBufRecv_ymin", Ghost_CellSz_y, Q);
        d_TransBufRecv_ymax = D_Data2d("d_TransBufRecv_ymax", Ghost_CellSz_y, Q);
        d_TransBufSend_ymax = D_Data2d("d_TransBufSend_ymax", Ghost_CellSz_y, Q);
        // =======================================================
        h_TransBufSend_ymin = Kokkos::create_mirror_view(d_TransBufSend_ymin);
        h_TransBufRecv_ymin = Kokkos::create_mirror_view(d_TransBufRecv_ymin);
        h_TransBufRecv_ymax = Kokkos::create_mirror_view(d_TransBufRecv_ymax);
        h_TransBufSend_ymax = Kokkos::create_mirror_view(d_TransBufSend_ymax);
        // =======================================================
#endif // end DIM_Y
    }
    return temp;
} // MpiTrans::AllocMem
// =======================================================
// =======================================================
void BaseSolver::MpiTransBufX()
{
// =======================================================
#ifndef AWARE_MPI
    Kokkos::deep_copy(h_TransBufSend_xmin, d_TransBufSend_xmin);
    Kokkos::deep_copy(h_TransBufRecv_xmin, d_TransBufRecv_xmin);
    Kokkos::deep_copy(h_TransBufRecv_xmax, d_TransBufRecv_xmax);
    Kokkos::deep_copy(h_TransBufSend_xmax, d_TransBufSend_xmax);
    Kokkos::fence();
    real_t *inptr_TransBufSend_xmin = h_TransBufSend_xmin.data();
    real_t *inptr_TransBufSend_xmax = h_TransBufSend_xmax.data();
    real_t *inptr_TransBufRecv_xmin = h_TransBufRecv_xmin.data();
    real_t *inptr_TransBufRecv_xmax = h_TransBufRecv_xmax.data();
#else
    real_t *inptr_TransBufSend_xmin = d_TransBufSend_xmin.data();
    real_t *inptr_TransBufSend_xmax = d_TransBufSend_xmax.data();
    real_t *inptr_TransBufRecv_xmin = d_TransBufRecv_xmin.data();
    real_t *inptr_TransBufRecv_xmax = d_TransBufRecv_xmax.data();
#endif // end AWARE_MPI

    mpiTrans.communicator->sendrecv(inptr_TransBufSend_xmin, Ghost_CellSz_x * Q, mpiTrans.data_type, mpiTrans.neighborsRank[X_MIN], 100,
                                    inptr_TransBufRecv_xmax, Ghost_CellSz_x * Q, mpiTrans.data_type, mpiTrans.neighborsRank[X_MAX], 100);
    mpiTrans.communicator->sendrecv(inptr_TransBufSend_xmax, Ghost_CellSz_x * Q, mpiTrans.data_type, mpiTrans.neighborsRank[X_MAX], 200,
                                    inptr_TransBufRecv_xmin, Ghost_CellSz_x * Q, mpiTrans.data_type, mpiTrans.neighborsRank[X_MIN], 200);

// =======================================================
#ifndef AWARE_MPI
    Kokkos::deep_copy(d_TransBufSend_xmin, h_TransBufSend_xmin);
    Kokkos::deep_copy(d_TransBufRecv_xmin, h_TransBufRecv_xmin);
    Kokkos::deep_copy(d_TransBufRecv_xmax, h_TransBufRecv_xmax);
    Kokkos::deep_copy(d_TransBufSend_xmax, h_TransBufSend_xmax);
    Kokkos::fence();
#endif
}
// =======================================================
// =======================================================
void BaseSolver::MpiTransBufY()
{
// =======================================================
#ifndef AWARE_MPI
    Kokkos::deep_copy(h_TransBufSend_ymin, d_TransBufSend_ymin);
    Kokkos::deep_copy(h_TransBufRecv_ymin, d_TransBufRecv_ymin);
    Kokkos::deep_copy(h_TransBufRecv_ymax, d_TransBufRecv_ymax);
    Kokkos::deep_copy(h_TransBufSend_ymax, d_TransBufSend_ymax);
    Kokkos::fence();
    real_t *inptr_TransBufSend_ymin = h_TransBufSend_ymin.data();
    real_t *inptr_TransBufSend_ymax = h_TransBufSend_ymax.data();
    real_t *inptr_TransBufRecv_ymin = h_TransBufRecv_ymin.data();
    real_t *inptr_TransBufRecv_ymax = h_TransBufRecv_ymax.data();
#else
    real_t *inptr_TransBufSend_ymin = d_TransBufSend_ymin.data();
    real_t *inptr_TransBufSend_ymax = d_TransBufSend_ymax.data();
    real_t *inptr_TransBufRecv_ymin = d_TransBufRecv_ymin.data();
    real_t *inptr_TransBufRecv_ymax = d_TransBufRecv_ymax.data();
#endif // end AWARE_MPI

    mpiTrans.communicator->sendrecv(inptr_TransBufSend_ymin, Ghost_CellSz_y * Q, mpiTrans.data_type, mpiTrans.neighborsRank[Y_MIN], 100,
                                    inptr_TransBufRecv_ymax, Ghost_CellSz_y * Q, mpiTrans.data_type, mpiTrans.neighborsRank[Y_MAX], 100);
    mpiTrans.communicator->sendrecv(inptr_TransBufSend_ymax, Ghost_CellSz_y * Q, mpiTrans.data_type, mpiTrans.neighborsRank[Y_MAX], 200,
                                    inptr_TransBufRecv_ymin, Ghost_CellSz_y * Q, mpiTrans.data_type, mpiTrans.neighborsRank[Y_MIN], 200);

// =======================================================
#ifndef AWARE_MPI
    Kokkos::deep_copy(d_TransBufSend_ymin, h_TransBufSend_ymin);
    Kokkos::deep_copy(d_TransBufRecv_ymin, h_TransBufRecv_ymin);
    Kokkos::deep_copy(d_TransBufRecv_ymax, h_TransBufRecv_ymax);
    Kokkos::deep_copy(d_TransBufSend_ymax, h_TransBufSend_ymax);
    Kokkos::fence();
#endif
}

MpiTrans::MpiTrans(Block &bl, BConditions const Boundarys[6])
{
    int mx = bl.mx, my = bl.my, mz = 1;
    // runtime determination if we are using float ou double (for MPI communication)
    data_type = typeid(double).name() == typeid(real_t).name() ? mpiUtils::MpiComm::DOUBLE : mpiUtils::MpiComm::FLOAT;
    // check that parameters are consistent
    bool error = false;
    error |= (mx < 1), error |= (my < 1), error |= (mz < 1);
    // get world communicator size and check it is consistent with mesh grid sizes
    MPI_Comm_group(MPI_COMM_WORLD, &comm_world);
    nProcs = MpiComm::world().getNProc();
    if (nProcs != mx * my * mz)
    {
        std::cerr << "ERROR: mx * my = " << mx * my << " must match with nRanks given to mpirun !!!\n";
        abort();
    }
    communicator = new MpiCommCart(mx, my, 1, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);

    // get my MPI rank inside topology    // get my MPI rank inside topology
    myRank = communicator->getRank();
    {
        // get coordinates of myRank inside topology
        // myMpiPos[0] is between 0 and mx-1 // myMpiPos[1] is between 0 and my-1 // myMpiPos[2] is between 0 and mz-1
        int mpiPos[3] = {0, 0, 0};
        communicator->getMyCoords(&mpiPos[0]);
        bl.myMpiPos_x = mpiPos[0], bl.myMpiPos_y = mpiPos[1];
    }

    // compute MPI ranks of our neighbors and set default boundary condition types
    nNeighbors = 2 * (DIM_X + DIM_Y);
    neighborsRank[X_MIN] = DIM_X ? communicator->getNeighborRank<X_MIN>() : 0;
    neighborsRank[X_MAX] = DIM_X ? communicator->getNeighborRank<X_MAX>() : 0;
    neighborsRank[Y_MIN] = DIM_Y ? communicator->getNeighborRank<Y_MIN>() : 0;
    neighborsRank[Y_MAX] = DIM_Y ? communicator->getNeighborRank<Y_MAX>() : 0;
    neighborsRank[Z_MIN] = 0;
    neighborsRank[Z_MAX] = 0;

    neighborsBC[X_MIN] = DIM_X ? BC_COPY : BC_UNDEFINED;
    neighborsBC[X_MAX] = DIM_X ? BC_COPY : BC_UNDEFINED;
    neighborsBC[Y_MIN] = DIM_Y ? BC_COPY : BC_UNDEFINED;
    neighborsBC[Y_MAX] = DIM_Y ? BC_COPY : BC_UNDEFINED;
    neighborsBC[Z_MIN] = BC_UNDEFINED;
    neighborsBC[Z_MAX] = BC_UNDEFINED;

    // Identify outside boundaries for mpi rank at edge of each direction in all mpi nRanks world
    if (bl.myMpiPos_x == 0) // X_MIN boundary
        neighborsBC[X_MIN] = Boundarys[X_MIN];
    if (bl.myMpiPos_x == mx - 1) // X_MAX boundary
        neighborsBC[X_MAX] = Boundarys[X_MAX];
    if (bl.myMpiPos_y == 0) // Y_MIN boundary
        neighborsBC[Y_MIN] = Boundarys[Y_MAX];
    if (bl.myMpiPos_y == my - 1) // Y_MAX boundary
        neighborsBC[Y_MAX] = Boundarys[Y_MAX];

    for (int i = 0; i < nProcs; i++)
    {
        communicator->synchronize();
        if (myRank == i)
            std::cout << "For rank:" << myRank << " at MPI position(myMpiPos_x, myMpiPos_y)=(" << bl.myMpiPos_x << "," << bl.myMpiPos_y << ")\n"
                      << "  NeighborRanks(X_MIN, X_MAX, Y_MIN, Y_MAX)=(" << neighborsRank[X_MIN] << ", " << neighborsRank[X_MAX] << ", "
                      << neighborsRank[Y_MIN] << ", " << neighborsRank[Y_MAX] << ").\n";
        communicator->synchronize();
    }
} // MpiTrans::MpiTrans