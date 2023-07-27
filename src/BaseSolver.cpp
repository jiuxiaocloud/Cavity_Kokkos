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

BaseSolver::BaseSolver(size_t nx, size_t ny, real_t Re_, real_t U_, real_t rho_) : NX(nx), NY(ny), Re(Re_), U(U_), rho0(rho_)
{
    nu = U * NX / Re, tau = 3.0 * nu + 0.5, omega = 1.0 / tau;

    iex = D_IData1d("iex", Q), iey = D_IData1d("iey", Q);
    ex = D_Data1d("ex", Q), ey = D_Data1d("ey", Q), w = D_Data1d("w", Q);
    rho = D_Data2d("rho", NX, NY), ux = D_Data2d("ux", NX, NY), uy = D_Data2d("uy", NX, NY);
    ux0 = D_Data2d("ux0", NX, NY), uy0 = D_Data2d("uy0", NX, NY);
    stream_func = D_Data2d("stream_func", NX, NY), vorticity = D_Data2d("vorticity", NX, NY);
    f = D_Data3d("f", NX, NY, Q), fb = D_Data3d("fb", NX, NY, Q);
    Kokkos::deep_copy(rho, rho0);
    h_ux = Kokkos::create_mirror_view(ux);
    h_uy = Kokkos::create_mirror_view(uy);
    h_rho = Kokkos::create_mirror_view(rho);
    h_stream_func = Kokkos::create_mirror_view(stream_func);
    h_vorticity = Kokkos::create_mirror_view(vorticity);
    Initialize();
}

void BaseSolver::Initialize()
{
    Kokkos::parallel_for(
        1, KOKKOS_CLASS_LAMBDA(const int64_t id) {
            // const real_t ex[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
            iex(0) = 0, iex(1) = 1, iex(2) = 0, iex(3) = -1, iex(4) = 0, iex(5) = 1, iex(6) = -1, iex(7) = -1, iex(8) = 1;
            ex(0) = 0, ex(1) = 1, ex(2) = 0, ex(3) = -1, ex(4) = 0, ex(5) = 1, ex(6) = -1, ex(7) = -1, ex(8) = 1;
            // const real_t ey[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
            iey(0) = 0, iey(1) = 0, iey(2) = 1, iey(3) = 0, iey(4) = -1, iey(5) = 1, iey(6) = 1, iey(7) = -1, iey(8) = -1;
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
}

void BaseSolver::Boundary()
{
    // 左右边界
    Kokkos::parallel_for(
        NY, KOKKOS_CLASS_LAMBDA(const int64_t j) {
        f(0,j,1) = fb(0,j,3);
        f(0,j,5) = fb(0,j,7);
        f(0,j,8) = fb(0,j,6);

        f(NX - 1,j,3) = fb(NX - 1,j,1);
        f(NX - 1,j,6) = fb(NX - 1,j,8);
        f(NX - 1,j,7) = fb(NX - 1,j,5); });

    Kokkos::fence();

    // 上下边界
    Kokkos::parallel_for(
        NX, KOKKOS_CLASS_LAMBDA(const int64_t i) {
            f(i,0,2) = fb(i,0,4);
            f(i,0,5) = fb(i,0,7);
            f(i,0,6) = fb(i,0,8);

            f(i,NY - 1,4) = fb(i,NY - 1,2);
            f(i,NY - 1,7) = fb(i,NY - 1,5) - rho0 * U / 6.0;
            f(i,NY - 1,8) = fb(i,NY - 1,6) + rho0 * U / 6.0; });

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

    // Kokkos::parallel_reduce(
    //     NX * NY, KOKKOS_CLASS_LAMBDA(const int64_t id, real_t &ltemp2) {
    //     int i, j;
    //     index2coord(id, i, j, NX, NY);
    //     ltemp2 += ux(i, j) * ux(i, j) + uy(i, j) * uy(i, j);
    //     ux0(i,j) = ux(i,j);
    //     uy0(i,j) = uy(i,j); },
    //     redutm2);

    Kokkos::fence();
    real_t O_tm1, O_tm2;
    Kokkos::deep_copy(O_tm1, temp1);
    Kokkos::deep_copy(O_tm2, temp2);
    O_tm1 = sqrt(O_tm1);
    O_tm2 = sqrt(O_tm2);

    return O_tm1 / (O_tm2 + 1e-30);
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
    std::string name = Path + "/cavity_" + std::to_string(m) + ".plt";
    std::ofstream out(name.c_str());
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(6);
    out << "Title=\"Lid Driven Flow\"\n"
        << "VARIABLES = \"X\",\"Y\",\"U\",\"V\",\"rho\",\"p\",\"PSI\",\"Vorticity\",\"Fd\",\"cd\"\n"
        << "ZONE T= \"BOX_Fd_" << O_Fd << "_cd_" << O_cd << "\",I= " << NX << ",J = " << NY << ",F = POINT" << std::endl;
    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
        {
            out << i << " " << j << " "
                << (real_t(i) + 0.5) / NX << " " << (real_t(j) + 0.5) / NY << " "
                << h_ux(i, j) / U << " " << h_uy(i, j) / U << " "
                << h_rho(i, j) / rho0 << " ";
            out << (h_rho(i, j) - rho0) / 3.0 << " ";
            // out << h_stream_func(i, j) << " ";
            out << h_vorticity(i, j) << " ";
            for (size_t k = 0; k < Q; k++)
                out << h_f(i, j, k) << " ";
            for (size_t k = 0; k < Q; k++)
                out << h_fb(i, j, k) << " ";
            // out << O_Fd << " " << O_cd;
            out << std::endl;
        }
}