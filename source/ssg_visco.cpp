#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <numbers>
#include <cmath>
#include <omp.h>
#include <vector>
#include <highfive/H5Easy.hpp> // для записи результатов рабты численной схемы
#include <toml.hpp> // для считывания файла конфигурации

int main() {
  std::ostream &log(std::cout);

  const auto config = toml::parse("./assets/config.toml");

  const auto general_config = toml::find(config, "general");

  const auto tasks = toml::find(config, "tasks");

  const int snapshot_freq = toml::find<int>(general_config, "snapshot_freq");
  const std::string dataset_path = toml::find<std::string>(general_config, "hdf5_filename");
  HighFive::File dataset(dataset_path, HighFive::File::Truncate);


  for (const auto &task : tasks.as_array()) {
    std::string task_name = toml::find<std::string>(task, "name");
    std::string task_descr = toml::find<std::string>(task, "descr");

    log << "Start processing task " << " : " << task_name  << '\n';
    log << "\t description: " << task_descr << '\n';
    // dataset.createDataSet("tasks/" + task_name + "/", data);

    const unsigned int dimx = toml::find<unsigned int>(task, "dimx");
    const unsigned int dimy = toml::find<unsigned int>(task, "dimy");
    const unsigned int dimz = toml::find<unsigned int>(task, "dimz");


    const double dx = toml::find<double>(task, "dx");
    const double dy = toml::find<double>(task, "dy");
    const double dz = toml::find<double>(task, "dz");

    const double rho = toml::find<double>(task, "rho");

    const double tau_p = toml::find<double>(task, "tau_p");
    const double tau_s = toml::find<double>(task, "tau_s");
    const double tau = toml::find<double>(task, "tau");

    const double freqM = toml::find<double>(task, "freqM");

    const double lambda = toml::find<double>(task, "lambda");
    const double mu = toml::find<double>(task, "mu");

    const double a1P = (lambda + 2 * mu) * (1 + tau_p);
    const double a1S = mu * (1 + tau_s);

    const double a2P = (lambda + 2 * mu) * tau_p;
    const double a2S = mu * tau_s;

    const double a1d = a1P - 2 * a1S;
    const double a2d = a2P - 2 * a2S;

    // Размер вычислительной области
    // Eigen::Vector3i corner[8];
    int pml_domain_size = 3;
    // corner[0] = {pml_domain_size ,pml_domain_size ,pml_domain_size };
    // corner[1] = {dimx - pml_domain_size ,pml_domain_size ,pml_domain_size };
    // corner[2] = {dimx - pml_domain_size ,dimy -
    // pml_domain_size,pml_domain_size }; corner[3] = {pml_domain_size, dimy -
    // pml_domain_size, pml_domain_size }; corner[4] = {pml_domain_size, dimy -
    // pml_domain_size, dimz - pml_domain_size }; corner[5] = {pml_domain_size,
    // pml_domain_size, dimz - pml_domain_size }; corner[6] = {dimx -
    // pml_domain_size, pml_domain_size, dimz - pml_domain_size }; corner[7] =
    // {dimx - pml_domain_size, dimy - pml_domain_size, dimz - pml_domain_size
    // };

    const double tfin = toml::find<double>(task, "tfin");
    const double dt = toml::find<double>(task, "dt");
    const unsigned int dimt = (unsigned int)tfin / dt;
  
    std::string dump_folder = "/" + task_name;

    const unsigned int micro_x = toml::find<unsigned int>(task, "micro_x");
    const unsigned int micro_y = toml::find<unsigned int>(task, "micro_y");
    const unsigned int micro_z = toml::find<unsigned int>(task, "micro_z");

    // std::vector<double> micro_storage(3 * (dimt / snapshot_freq + 1));
    
    // Eigen::TensorMap<Eigen::Tensor<double, 2>> micro_data(&micro_storage.front(), 3, dimt / snapshot_freq + 1);

    std::vector<double> v_x_prev_storage(dimx * dimx * dimx);
    std::vector<double> v_y_prev_storage(dimx * dimx * dimx);
    std::vector<double> v_z_prev_storage(dimx * dimx * dimx);

    std::vector<double> v_x_next_storage(dimx * dimx * dimx);
    std::vector<double> v_y_next_storage(dimx * dimx * dimx);
    std::vector<double> v_z_next_storage(dimx * dimx * dimx);

    std::vector<double> sigma_xx_prev_storage(dimx * dimx * dimx);
    std::vector<double> sigma_yy_prev_storage(dimx * dimx * dimx);
    std::vector<double> sigma_zz_prev_storage(dimx * dimx * dimx);
    std::vector<double> sigma_yz_prev_storage(dimx * dimx * dimx);
    std::vector<double> sigma_xz_prev_storage(dimy * dimy * dimy);
    std::vector<double> sigma_xy_prev_storage(dimy * dimy * dimy);

    std::vector<double> sigma_xx_next_storage(dimy * dimy * dimy);
    std::vector<double> sigma_yy_next_storage(dimy * dimy * dimy);
    std::vector<double> sigma_zz_next_storage(dimy * dimy * dimy);
    std::vector<double> sigma_yz_next_storage(dimy * dimy * dimy);
    std::vector<double> sigma_xz_next_storage(dimy * dimy * dimy);
    std::vector<double> sigma_xy_next_storage(dimy * dimy * dimy);

    std::vector<double> r_xx_prev_storage(dimy * dimy * dimy);
    std::vector<double> r_yy_prev_storage(dimy * dimy * dimy);
    std::vector<double> r_zz_prev_storage(dimz * dimz * dimz);
    std::vector<double> r_yz_prev_storage(dimz * dimz * dimz);
    std::vector<double> r_xz_prev_storage(dimz * dimz * dimz);
    std::vector<double> r_xy_prev_storage(dimz * dimz * dimz);

    std::vector<double> r_xx_next_storage(dimz * dimz * dimz);
    std::vector<double> r_yy_next_storage(dimz * dimz * dimz);
    std::vector<double> r_zz_next_storage(dimz * dimz * dimz);
    std::vector<double> r_yz_next_storage(dimz * dimz * dimz);
    std::vector<double> r_xz_next_storage(dimz * dimz * dimz);
    std::vector<double> r_xy_next_storage(dimz * dimz * dimz);

    Eigen::TensorMap<Eigen::Tensor<double, 3>> vXprev(&v_x_prev_storage.front(),
                                                      dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> vYprev(&v_y_prev_storage.front(),
                                                      dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> vZprev(&v_z_prev_storage.front(),
                                                      dimx, dimy, dimz);

    Eigen::TensorMap<Eigen::Tensor<double, 3>> vXnext(&v_x_next_storage.front(),
                                                      dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> vYnext(&v_y_next_storage.front(),
                                                      dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> vZnext(&v_z_next_storage.front(),
                                                      dimx, dimy, dimz);

    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaXXprev(
        &sigma_xx_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaYYprev(
        &sigma_yy_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaZZprev(
        &sigma_zz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaYZprev(
        &sigma_yz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaXZprev(
        &sigma_xz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaXYprev(
        &sigma_xy_next_storage.front(), dimx, dimy, dimz);

    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaXXnext(
        &sigma_xx_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaYYnext(
        &sigma_yy_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaZZnext(
        &sigma_zz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaYZnext(
        &sigma_yz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaXZnext(
        &sigma_xz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> sigmaXYnext(
        &sigma_xy_next_storage.front(), dimx, dimy, dimz);

    Eigen::TensorMap<Eigen::Tensor<double, 3>> rXXprev(
        &sigma_xx_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rYYprev(
        &sigma_yy_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rZZprev(
        &sigma_zz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rYZprev(
        &sigma_yz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rXZprev(
        &sigma_xz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rXYprev(
        &sigma_xy_next_storage.front(), dimx, dimy, dimz);

    Eigen::TensorMap<Eigen::Tensor<double, 3>> rXXnext(
        &r_xx_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rYYnext(
        &r_yy_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rZZnext(
        &r_zz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rYZnext(
        &r_yz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rXZnext(
        &r_xz_next_storage.front(), dimx, dimy, dimz);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> rXYnext(
        &r_xy_next_storage.front(), dimx, dimy, dimz);

#define Diff_x(f, I, J, K, dx)                                                 \
  1. / dx *                                                                    \
      (27. / 24. * (f(I + 1, J, K) - f(I - 1, J, K)) -                         \
       1. / 24. * (f(I + 2, J, K) - f(I - 2, J, K)))
#define Diff_y(f, I, J, K, dy)                                                 \
  1. / dy *                                                                    \
      (27. / 24. * (f(I, J + 1, K) - f(I, J - 1, K)) -                         \
       1. / 24. * (f(I, J + 2, K) - f(I, J + 2, K)))
#define Diff_z(f, I, J, K, dz)                                                 \
  1. / dz *                                                                    \
      (27. / 24. * (f(I, J, K + 1) - f(I, J, K - 1)) -                         \
       1. / 24. * (f(I, J, K + 2) - f(I, J, K - 2)))

    const double memory_var_coef = (1 + dt / tau / 2);

    auto ricker_wavelet = [] (double freqM, double t) {
      return (1 - 2 * std::numbers::pi * std::numbers::pi * freqM * freqM * t * t) * std::exp(-std::numbers::pi * std::numbers::pi * freqM * freqM * t * t); 
    };

    Eigen::Vector3i source_pos = {dimx / 2, dimy / 2, dimz / 2};

    for (int n = 0; n < dimt; ++n) {
      log << "layer " << n << "/" << dimt << "\n";
      // update velocities
      for (int i = pml_domain_size; i < dimx - pml_domain_size; i += 2) {
        for (int j = pml_domain_size; j < dimy - pml_domain_size; j += 2) {
          for (int k = pml_domain_size; k < dimz - pml_domain_size; k += 2) {
            vXnext(i + 1, j, k) = vXprev(i + 1, j, k) +
                                  dt / rho *
                                      (Diff_x(sigmaXXprev, i + 1, j, k, dx) +
                                       Diff_y(sigmaXYprev, i + 1, j, k, dy) +
                                       Diff_z(sigmaXZprev, i + 1, j, k, dz));

            vYnext(i, j + 1, k) = vYprev(i, j + 1, k) +
                                  dt / rho *
                                      (Diff_x(sigmaXYprev, i, j + 1, k, dx) +
                                       Diff_y(sigmaYYprev, i, j + 1, k, dy) +
                                       Diff_z(sigmaYZprev, i, j + 1, k, dz));

            vZnext(i, j, k + 1) = vZprev(i, j, k + 1) +
                                  dt / rho *
                                      (Diff_x(sigmaXZprev, i, j, k + 1, dx) +
                                       Diff_y(sigmaYZprev, i, j, k + 1, dy) +
                                       Diff_z(sigmaZZprev, i, j, k + 1, dz));
          }
        }
      }

      // initialize source wavelet
      vXnext(source_pos(0) + 1, source_pos(1) + 1, source_pos(2) + 1) = ricker_wavelet(freqM, n * dt);
      vYnext(source_pos(0) + 1, source_pos(1) + 1, source_pos(2) + 1) = ricker_wavelet(freqM, n * dt);
      vZnext(source_pos(0) + 1, source_pos(1) + 1, source_pos(2) + 1) = ricker_wavelet(freqM, n * dt);

      // update memory variables
      for (int i = pml_domain_size; i < dimx - pml_domain_size; i += 2) {
        for (int j = pml_domain_size; j < dimy - pml_domain_size; j += 2) {
          for (int k = pml_domain_size; k < dimz - pml_domain_size; k += 2) {
            rXXnext(i, j, k) = rXXprev(i, j, k) + dt / tau *
                    (-a2P * Diff_x(vXnext, i, j, k, dx) -
                     a2d * Diff_y(vYnext, i, j, k, dy) -
                     a2d * Diff_z(vZnext, i, j, k, dz) - rXXprev(i, j, k) / 2.);

            rYYnext(i, j, k) = rYYprev(i, j, k) + dt / tau *
                    (-a2d * Diff_x(vXnext, i, j, k, dx) -
                     a2P * Diff_y(vYnext, i, j, k, dy) -
                     a2d * Diff_z(vZnext, i, j, k, dz) - rYYprev(i, j, k) / 2.);

            rZZnext(i, j, k) = rZZprev(i, j, k) + dt / tau *
                    (-a2d * Diff_x(vXnext, i, j, k, dx) -
                     a2d * Diff_y(vYnext, i, j, k, dy) -
                     a2P * Diff_z(vZnext, i, j, k, dz) - rZZprev(i, j, k) / 2.);

            rYZnext(i, j + 1, k + 1) = rYZprev(i, j + 1, k + 1) + dt / tau *
                    (- a2S * Diff_z(vYnext, i, j + 1, k + 1, dz)
                     - a2S * Diff_y(vZnext, i, j + 1, k + 1, dy)
                     - rYZprev(i, j + 1, k + 1) / 2);
            
            rXZnext(i + 1, j, k + 1) = rXZprev(i + 1, j, k + 1) + dt / tau *
                    (-a2S * Diff_x(vZnext, i + 1, k, k + 1, dx)
                     -a2S * Diff_z(vXnext, i + 1, j, k + 1, dz)
                     - rXZprev(i + 1, j, k + 1));
            
            rXYnext(i + 1, j + 1, k) = rXYprev(i + 1, j + 1, k) + dt / tau *
                    (-a2S * Diff_y(vXnext, i + 1, j + 1, k, dy)
                     -a2S * Diff_x(vYnext, i + 1, j + 1, k, dx) 
                     - rXYprev(i + 1, j + 1, k));
            
            rXXnext(i, j, k) /= memory_var_coef;
            rYYnext(i, j, k) /= memory_var_coef;
            rZZnext(i, j, k) /= memory_var_coef;

            rYZnext(i, j + 1, k + 1) /= memory_var_coef;
            rXZnext(i + 1, j, k + 1) /= memory_var_coef;
            rXYnext(i + 1, j + 1, k) /= memory_var_coef;
          }
        }
      }

      // update strain variables
      for (int i = pml_domain_size; i < dimx - pml_domain_size; i += 2) {
        for (int j = pml_domain_size; j < dimy - pml_domain_size; j += 2) {
          for (int k = pml_domain_size; k < dimz - pml_domain_size; k += 2) {
            sigmaXXnext(i, j, k) = sigmaXXprev(i, j, k) + dt * ( 
                a1P * Diff_x(vXnext, i, j, k, dx)
              + a1d * Diff_y(vYnext, i, j, k, dy)
              + a1d * Diff_z(vZnext, i, j, k, dz)
              + 0.5 * (rXXnext(i, j, k) + rXXprev(i, j, k))
            );

            sigmaYYnext(i, j, k) = sigmaYYprev(i, j, k) + dt * ( 
                a1d * Diff_x(vXnext, i, j, k, dx)
              + a1P * Diff_y(vYnext, i, j, k, dy)
              + a1d * Diff_z(vZnext, i, j, k, dz)
              + 0.5 * (rYYnext(i, j, k) + rYYprev(i, j, k))
            );

            sigmaZZnext(i, j, k) = sigmaZZprev(i, j, k) + dt * ( 
                a1d * Diff_x(vXnext, i, j, k, dx)
              + a1d * Diff_y(vYnext, i, j, k, dy)
              + a1P * Diff_z(vZnext, i, j, k, dz)
              + 0.5 * (rZZnext(i, j, k) + rZZprev(i, j, k))
            );

            sigmaYZnext(i, j + 1, k + 1) = sigmaYZprev(i, j + 1, k + 1) + dt * (
                a1S * Diff_z(vYnext, i, j + 1, k + 1, dz)
              + a1S * Diff_y(vZnext, i, j + 1, k + 1, dy)
              + 0.5 * (rYZnext(i, j + 1, k + 1) + rYZprev(i, j + 1, k + 1))
            );

            sigmaXZnext(i + 1, j, k + 1) = sigmaXZprev(i + 1, j, k + 1)  + dt * (
                a1S * Diff_z(vXnext, i + 1, j, k + 1, dz)
              + a1S * Diff_x(vZnext, i + 1, j, k + 1, dx)
              + 0.5 * (rXZnext(i + 1, j, k + 1) + rXZprev(i + 1, j, k + 1) )
            );

            sigmaXYnext(i + 1, j + 1, k) = sigmaXYprev(i + 1, j + 1, k) + dt * (
                a1S * Diff_y(vYnext, i + 1, j + 1, k, dy)
              + a1S * Diff_x(vZnext, i + 1, j + 1, k, dx)
              + 0.5 * (rXYnext(i + 1, j + 1, k) + rXYprev(i + 1, j + 1, k))
            );
          }
        }
      }

      if (n % snapshot_freq == 0) {
         
          H5Easy::dump(dataset, dump_folder + "/" + std::to_string(n) + "/vx", v_x_prev_storage, H5Easy::DumpMode::Overwrite);
          H5Easy::dump(dataset, dump_folder + "/" + std::to_string(n) + "/vy", v_y_prev_storage, H5Easy::DumpMode::Overwrite);
          H5Easy::dump(dataset, dump_folder + "/" + std::to_string(n) + "/vz", v_z_prev_storage, H5Easy::DumpMode::Overwrite);
          
          H5Easy::dump(dataset, dump_folder + "/" + std::to_string(n) + "/microx", vXnext(micro_x, micro_y, micro_z), H5Easy::DumpMode::Overwrite);
          H5Easy::dump(dataset, dump_folder + "/" + std::to_string(n) + "/microz", vYnext(micro_x, micro_y, micro_z), H5Easy::DumpMode::Overwrite);
          H5Easy::dump(dataset, dump_folder + "/" + std::to_string(n) + "/microy", vZnext(micro_x, micro_y, micro_z), H5Easy::DumpMode::Overwrite);
      }
    }

    // H5Easy::dumpAttribute(dataset, dump_folder, "dimt", dimt);
  }
  return 0;
}