#include <iostream>
#include <omp.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <highfive/H5Easy.hpp> // для записи результатов рабты численной схемы
#include <toml.hpp> // для считывания файла конфигурации


int main() {
  std::ostream& log(std::cout);

  const auto config = toml::parse("./assets/config.toml");
  const auto tasks  = toml::find(config, "tasks");

  for(const auto& task : tasks.as_array()) {
      log << "Start processing task " << " : " << toml::find<std::string>(task, "name") << '\n';
      log<< "\t description: " << toml::find<std::string>(task, "descr") << '\n';

      const unsigned int dimx = toml::find<unsigned int>(task, "dimx");
      const unsigned int dimy = toml::find<unsigned int>(task, "dimy");
      const unsigned int dimz = toml::find<unsigned int>(task, "dimz");

      const double dx = toml::find<double>(task, "dx");
      const double dy = toml::find<double>(task, "dy");
      const double dz = toml::find<double>(task, "dz");
      
      const double rho        = toml::find<double >(task, "rho");

      const double tau_p      = toml::find<double>(task, "tau_p");
      const double tau_s      = toml::find<double>(task, "tau_s");
      const double tau        = toml::find<double>(task, "tau");
      
      const double lambda     = toml::find<double>(task, "lambda");
      const double mu         = toml::find<double>(task, "mu");

      const double a1P = (lambda + 2 * mu) * (1 + tau_p);
      const double a1S = mu * (1 + tau_s);

      const double a2P = (lambda + 2 * mu) * tau_p;
      const double a2S = mu * tau_s;    

      const double a1d = a1P - 2 * a1S;
      const double a2d = a2P - 2 * a2S;  
      

      const double tfin = toml::find<double>(task, "tfin");
      const double dt = toml::find<double>(task, "dt");
      const unsigned int dimt = (unsigned int) tfin / dt;

      std::vector<double> v_x_prev_storage      (dimx * dimx * dimx);
      std::vector<double> v_y_prev_storage      (dimx * dimx * dimx);
      std::vector<double> v_z_prev_storage      (dimx * dimx * dimx);

      std::vector<double> v_x_next_storage      (dimx * dimx * dimx);
      std::vector<double> v_y_next_storage      (dimx * dimx * dimx);
      std::vector<double> v_z_next_storage      (dimx * dimx * dimx);


      std::vector<double> sigma_xx_prev_storage (dimx * dimx * dimx);
      std::vector<double> sigma_yy_prev_storage (dimx * dimx * dimx);
      std::vector<double> sigma_zz_prev_storage (dimx * dimx * dimx);
      std::vector<double> sigma_yz_prev_storage (dimx * dimx * dimx);
      std::vector<double> sigma_xz_prev_storage (dimy * dimy * dimy);
      std::vector<double> sigma_xy_prev_storage (dimy * dimy * dimy);

      std::vector<double> sigma_xx_next_storage (dimy * dimy * dimy);
      std::vector<double> sigma_yy_next_storage (dimy * dimy * dimy);
      std::vector<double> sigma_zz_next_storage (dimy * dimy * dimy);
      std::vector<double> sigma_yz_next_storage (dimy * dimy * dimy);
      std::vector<double> sigma_xz_next_storage (dimy * dimy * dimy);
      std::vector<double> sigma_xy_next_storage (dimy * dimy * dimy);


      std::vector<double> r_xx_prev_storage     (dimy * dimy * dimy);
      std::vector<double> r_yy_prev_storage     (dimy * dimy * dimy);
      std::vector<double> r_zz_prev_storage     (dimz * dimz * dimz);
      std::vector<double> r_yz_prev_storage     (dimz * dimz * dimz);
      std::vector<double> r_xz_prev_storage     (dimz * dimz * dimz);
      std::vector<double> r_xy_prev_storage     (dimz * dimz * dimz);

      std::vector<double> r_xx_next_storage     (dimz * dimz * dimz);
      std::vector<double> r_yy_next_storage     (dimz * dimz * dimz);
      std::vector<double> r_zz_next_storage     (dimz * dimz * dimz);
      std::vector<double> r_yz_next_storage     (dimz * dimz * dimz);
      std::vector<double> r_xz_next_storage     (dimz * dimz * dimz);
      std::vector<double> r_xy_next_storage     (dimz * dimz * dimz);


      Eigen::TensorMap<Eigen::Tensor<double, 3>> vXprev (&v_x_prev_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> vYprev (&v_y_prev_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> vZprev (&v_z_prev_storage.front(), dimx, dimy, dimz);

      Eigen::TensorMap<Eigen::Tensor<double, 3>> vXnext (&v_x_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> vYnext (&v_y_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> vZnext (&v_z_next_storage.front(), dimx, dimy, dimz);


      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaXXprev(&sigma_xx_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaYYprev(&sigma_yy_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaZZprev(&sigma_zz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaYZprev(&sigma_yz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaXZprev(&sigma_xz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaXYprev(&sigma_xy_next_storage.front(), dimx, dimy, dimz);

      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaXXnext(&sigma_xx_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaYYnext(&sigma_yy_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaZZnext(&sigma_zz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaYZnext(&sigma_yz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaXZnext(&sigma_xz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> SigmaXYnext(&sigma_xy_next_storage.front(), dimx, dimy, dimz);


      Eigen::TensorMap<Eigen::Tensor<double, 3>> rXXprev(&sigma_xx_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rYYprev(&sigma_yy_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rZZprev(&sigma_zz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rYZprev(&sigma_yz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rXZprev(&sigma_xz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rXYprev(&sigma_xy_next_storage.front(), dimx, dimy, dimz);

      Eigen::TensorMap<Eigen::Tensor<double, 3>> rXXnext(&r_xx_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rYYnext(&r_yy_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rZZnext(&r_zz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rYZnext(&r_yz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rXZnext(&r_xz_next_storage.front(), dimx, dimy, dimz);
      Eigen::TensorMap<Eigen::Tensor<double, 3>> rXYnext(&r_xy_next_storage.front(), dimx, dimy, dimz);

      for (size_t n = 1; n < dimt; ++n) {
        
      }
  }
  return 0;
}