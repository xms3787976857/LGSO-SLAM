/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/gaussian_mapper.h"
#include "FullSystem/ResidualProjections.h"
#include <fstream>
#include <sstream>
#include <functional>
#include <unordered_map>
// 在 include 之后添加
//extern int wG[0], hG[0];


// 根据 incoming_id 查找 FrameHessian
dso::FrameHessian* findFrameHessianById(std::shared_ptr<dso::FullSystem> pSLAM, int incoming_id) {
    for (dso::FrameHessian* fh : pSLAM->frameHessians) {
        if (fh->shell->incoming_id == incoming_id) return fh;
    }
    return nullptr;
}

GaussianMapper::GaussianMapper(
    std::shared_ptr<dso::FullSystem> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // Random seed
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // Initialize scene and model
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

    // Mode
    if (!pSLAM) {
        // NO SLAM
        return;
    }

    // Cameras
    // TODO: not only monocular (from photo-slam)
    cv::Size SLAM_im_size(image_width, image_height);

    std::vector<float> vPinHoleDistorsion1(5);
    vPinHoleDistorsion1[0] = distortion_k1; // k1
    vPinHoleDistorsion1[1] = distortion_k2; // k2
    vPinHoleDistorsion1[2] = distortion_p1; // p1
    vPinHoleDistorsion1[3] = distortion_p2; // p2
    vPinHoleDistorsion1[4] = distortion_k3; // k3
    std::cout << "k1 k2 p1 p2 k3 " << distortion_k1 << distortion_k2 << distortion_p1 << distortion_p2 << distortion_k3 << std::endl;
    cv::Mat camera1DistortionCoef = cv::Mat(vPinHoleDistorsion1.size(),1,CV_32F,vPinHoleDistorsion1.data());
    UndistortParams undistort_params(
        SLAM_im_size,
        // settings->camera1DistortionCoef()
        camera1DistortionCoef
    );
    
    if (distortion_k1 != 0 || distortion_k2 != 0 || distortion_p1 != 0 || distortion_p2 != 0 || distortion_k3 != 0) {
        need_distortion = true;
    }

    Camera camera;
    camera.camera_id_ = 0;
    camera.setModelId(Camera::CameraModelType::PINHOLE);
    
    // float SLAM_fx = pSLAM_->Hcalib.fxl();
    // float SLAM_fy = pSLAM_->Hcalib.fyl();
    // float SLAM_cx = pSLAM_->Hcalib.cxl();
    // float SLAM_cy = pSLAM_->Hcalib.cyl();
    float SLAM_fx = mapping_fx;
    float SLAM_fy = mapping_fy;
    float SLAM_cx = mapping_cx;
    float SLAM_cy = mapping_cy;

    // Old K, i.e. K in SLAM
    cv::Mat K = (
        cv::Mat_<float>(3, 3)
            << SLAM_fx, 0.f, SLAM_cx,
                0.f, SLAM_fy, SLAM_cy,
                0.f, 0.f, 1.f
    );

    camera.width_ = undistort_params.old_size_.width;
    float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;

    camera.height_ = undistort_params.old_size_.height;
    float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

    camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
    camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
    camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
        camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
    }

    camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
    camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
    camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
    camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

    cv::Mat K_new = (
        cv::Mat_<float>(3, 3)
            << camera.params_[0], 0.f, camera.params_[2],
                0.f, camera.params_[1], camera.params_[3],
                0.f, 0.f, 1.f
    );

    // Undistortion
    // if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
    undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

    camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

    undistort_mask_[camera.camera_id_] =
        tensor_utils::cvMat2TorchTensor_Float32(
            camera.undistort_mask, device_type_);

    cv::Mat viewer_sub_undistort_mask;
    int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
    int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
    cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                cv::Size(viewer_image_width_, viewer_image_height_));
    viewer_sub_undistort_mask_[camera.camera_id_] =
        tensor_utils::cvMat2TorchTensor_Float32(
            viewer_sub_undistort_mask, device_type_);

    cv::Mat viewer_main_undistort_mask;
    int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
    int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
    cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                cv::Size(viewer_image_width_main_, viewer_image_height_main_));
    viewer_main_undistort_mask_[camera.camera_id_] =
        tensor_utils::cvMat2TorchTensor_Float32(
            viewer_main_undistort_mask, device_type_);

    if (!viewer_camera_id_set_) {
        viewer_camera_id_ = camera.camera_id_;
        viewer_camera_id_set_ = true;
    }

    this->scene_->addCamera(camera);
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // SLAM parameters
    image_width = 
        settings_file["SLAM.image_width"].operator int();
    image_height = 
        settings_file["SLAM.image_height"].operator int();
    distortion_k1 =
        settings_file["SLAM.distortion_k1"].operator float();
    distortion_k2 =
        settings_file["SLAM.distortion_k2"].operator float();
    distortion_p1 =
        settings_file["SLAM.distortion_p1"].operator float();
    distortion_p2 =
        settings_file["SLAM.distortion_p2"].operator float();
    distortion_k3 =
        settings_file["SLAM.distortion_k3"].operator float();
    // depth_scale =
    //     settings_file["SLAM.depth_scale"].operator float();
    mapping_fx =
        settings_file["Mapper.fx"].operator float();
    mapping_fy =
        settings_file["Mapper.fy"].operator float();
    mapping_cx =
        settings_file["Mapper.cx"].operator float();
    mapping_cy =
        settings_file["Mapper.cy"].operator float();

    // added
    lambda_sparse_depth =
        settings_file["Mapper.lambda_sparse_depth"].operator float();

    // Loss-weighted KF sampling thresholds
    loss_low_thresh_  = settings_file["Mapper.loss_low_thresh"].operator float();

    // Model parameters
    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;

    // Pipeline Parameters
    z_near_ =
        settings_file["Camera.z_near"].operator float();
    z_far_ =
        settings_file["Camera.z_far"].operator float();

    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    record_depth_image_ = 
        (settings_file["Record.record_depth_image"].operator int()) != 0;
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;

    // Optimization Parameters
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter_"].operator int();
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();

    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();

    // Viewer Parameters
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();

    // Localization
    loc_camera_cfg_path =
        settings_file["Localization.camera_cfg_path"].operator std::string();
}

void GaussianMapper::run()
{
    std::cout << "!!! GaussianMapper::run" << std::endl;
    // First loop: Initial gaussian mapping
    while (!isStopped()) {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions()) {
            // Get initial map
            std::vector<float> new_points;
            std::vector<float> new_colors;
            std::vector<float> new_scales;
            std::vector<float> new_rots;

            {   // Locked block
                boost::unique_lock<boost::mutex> allKFLock(pSLAM_->frozenMapMutex);
                std::vector<dso::FrozenFrameHessian*>* keyframesFromDso = &pSLAM_->newframeHessians;

                int j = 0;
                int num_kf = keyframesFromDso->size();
                for (int idx=0; idx < num_kf; idx++) {
                    if (keyframesFromDso->empty()) break;
                    // dso::FrozenFrameHessian* fh = keyframesFromDso.at(idx);
                    dso::FrozenFrameHessian* fh = keyframesFromDso->front();

                    if (fh->pointsInWorld.empty()) {
                        keyframesFromDso->erase(keyframesFromDso->begin());
                        continue;
                    }

                    float fx = pSLAM_->Hcalib.fxl();
                    float fy = pSLAM_->Hcalib.fyl();
                    float cx = pSLAM_->Hcalib.cxl();
                    float cy = pSLAM_->Hcalib.cyl();

                    Sophus::SE3d c2w = fh->camToWorld;
                    
                    // for (dso::PointHessian* ph : fh->pointHessians) {
                    //     if (ph->idepth > 0) {
                    //         float depth = 1.0f / ph->idepth;
                    //         float x = ((ph->u - cx) / fx) * depth;
                    //         float y = ((ph->v - cy) / fy) * depth;
                    //         float z = depth;

                    //         // TODO: change to matrix calculation
                    //         Eigen::Vector3d point_in_camera(x, y, z);
                    //         Eigen::Vector3d point_in_world = c2w * point_in_camera;

                    //         Point3D point3D;
                    //         point3D.xyz_ = point_in_world;

                    //         point3D.color_(0) = ph->kf_color.at(0);
                    //         point3D.color_(1) = ph->kf_color.at(1);
                    //         point3D.color_(2) = ph->kf_color.at(2);

                    //         scene_->cachePoint3D(j, point3D);
                    //     }
                    //     ++j;
                    // }

                    new_points.reserve(new_points.size() + fh->pointsInWorld.size());
                    new_points.insert(new_points.end(), fh->pointsInWorld.begin(), fh->pointsInWorld.end());

                    new_colors.reserve(new_colors.size() + fh->colors.size());
                    new_colors.insert(new_colors.end(), fh->colors.begin(), fh->colors.end());

                    new_scales.reserve(new_scales.size() + fh->scales.size());
                    new_scales.insert(new_scales.end(), fh->scales.begin(), fh->scales.end());

                    new_rots.reserve(new_rots.size() + fh->rots.size());
                    new_rots.insert(new_rots.end(), fh->rots.begin(), fh->rots.end());

                    std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(fh->incomingID, getIteration());
                    // std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(scene_->keyframes().size(), getIteration());
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;
                    // Pose
                    // auto pose = pKF->GetPose();
                    {
                        auto q_before = c2w.unit_quaternion();
                        std::cout << "[INIT KF DEBUG] fid=" << fh->incomingID
                                  << " camToWorld_q(wxyz)=" << q_before.w()<<" "<<q_before.x()<<" "<<q_before.y()<<" "<<q_before.z()
                                  << " camToWorld_t=" << c2w.translation().transpose() << std::endl;
                    }
                    c2w = c2w.inverse();
                    {
                        auto q_after = c2w.unit_quaternion();
                        std::cout << "[INIT KF DEBUG] fid=" << fh->incomingID
                                  << " after_inv_q(wxyz)=" << q_after.w()<<" "<<q_after.x()<<" "<<q_after.y()<<" "<<q_after.z()
                                  << " after_inv_t=" << c2w.translation().transpose() << std::endl;
                    }
                    new_kf->setPose(
                        // pose.unit_quaternion().cast<double>(),
                        // pose.translation().cast<double>()
                        c2w.unit_quaternion(),
                        c2w.translation());
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;

                    try {
                        // Add first keyframe to the scene
                        Camera& camera = scene_->cameras_.at(0);
                        new_kf->setCameraParams(camera);

                        // Image (left if STEREO)
                        dso::MinimalImageB3* img = fh->kfImg;
                        cv::Mat imgRGB(img->h, img->w, CV_8UC3, img->data);
                        if (need_distortion) {
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                        } else {
                            imgRGB_undistorted = imgRGB;
                        }
                        // imgRGB_undistorted = imgRGB;

                        imgAux_undistorted = imgRGB_undistorted;
                        imgRGB_undistorted.convertTo(imgRGB_undistorted, CV_32FC3, 1.0 / 255.0);
                        new_kf->original_image_ =
                            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                        
                        // Depth image
                        // dso::MinimalImage<unsigned short>* depth_img = fh->kf_depth;
                        // cv::Mat imgDepth(depth_img->h, depth_img->w, CV_16UC1, depth_img->data);
                        // cv::Mat imgDepthFloat;
                        // imgDepth.convertTo(imgDepthFloat, CV_32FC1, 1.0 / depth_scale);
                        // new_kf->original_depth_ =
                        //     tensor_utils::cvMat2TorchTensor_Float32(imgDepthFloat, device_type_);

                        // Sparse Depth
                        new_kf->sparse_depth_ =
                            tensor_utils::cvMat2TorchTensor_Float32(fh->kfSparseDepth, device_type_);

                        // TODO: add if needed
                        new_kf->img_filename_ = result_dir_ / "img" / (std::to_string(fh->incomingID) + ".png");
                        new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                        new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                        new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                    }
                    catch (std::out_of_range) {
                        throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
                    }

                    new_kf->computeTransformTensors();
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);
                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // Features
                    // std::vector<float> pixels;
                    // std::vector<float> pointsLocal;
                    // pKF->GetKeypointInfo(pixels, pointsLocal);
                    // new_kf->kps_pixel_ = std::move(pixels);
                    // new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                    
                    if (inserted_kf_cursor < fh->incomingID)
                        inserted_kf_cursor = fh->incomingID;

                    keyframesFromDso->erase(keyframesFromDso->begin());
                }
            }   // Lock resolved

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }
            
            // Prepare for training
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                // gaussians_->createFromPcd(new_points, new_colors, scene_->cameras_extent_);
                gaussians_->createFromPcd(new_points, new_colors, new_scales, new_rots, scene_->cameras_extent_);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                gaussians_->trainingSetup(opt_params_);
            }
            // Invoke training once
            trainForOneIteration();
            // Finish initial mapping loop
            initial_mapped_ = true;
            break;
        }
        // else if (pSLAM_->isRunMapping()) {
        //     break;
        // }
        else {
            // Initial conditions not satisfied
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental gaussian mapping
    std::cout << "start training loop" << std::endl;
    int SLAM_stop_iter = 0;
    while (!isStopped()) {
        // Update DSO keyframes
        if (checkKFupdateFromGSRequested()) {
            // std::cout << "Keyframe Update Requested" << std::endl;
            if (scene_->keyframes_.size() > 50) {
                // updateKeyFramesFromGS();
            }
            // std::cout << "Update End" << std::endl;
            pSLAM_->isDoneKFUpdateFromGS = true;
        }

        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            // if (cull_keyframes_)
            //     cullKeyframes();
        }

        // Invoke training once
        trainForOneIteration();

        if (!SLAM_ended_ && !pSLAM_->isRunMapping()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
            // ablation study!!!!!!!!!!!
            // additional_training_start = std::chrono::steady_clock::now();
            // additional_training_start_iter = getIteration();
        }

        
        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;

        // ablation study!!!!!!!!!!!
        // if (getIteration() - additional_training_start_iter > 13000)
        //     break;

        // if (getIteration() >= opt_params_.iterations_)
        //     break;
    }

    // Third loop: Tail gaussian optimization
    // int densify_interval = densifyInterval();
    // int n_delay_iters = densify_interval * 0.8;
    // while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
    //     trainForOneIteration();
    //     densify_interval = densifyInterval();
    //     n_delay_iters = densify_interval * 0.8;
    // }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / "gs_map");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
    {
        std::ofstream f(result_dir_ / "kf_filtered_count.txt");
        f << "Total KFs ever filtered (loss < " << loss_low_thresh_ << "): "
          << filtered_fids_.size() << " / " << scene_->keyframes().size() << std::endl;
        for (std::size_t fid : filtered_fids_) {
            auto it = kf_recent_loss_.find(fid);
            float loss = (it != kf_recent_loss_.end()) ? it->second : -1.0f;
            f << "  fid=" << fid << " loss=" << loss << std::endl;
        }
    }

    signalStop();
}

void GaussianMapper::trainColmap()
{
    // Prepare multi resolution images for training
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }

    // Prepare for training
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    // Main loop: gaussian splatting training
    while (!isStopped()) {
        // Invoke training once
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

/**
 * @brief The training iteration body
 * 
 */
void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    // std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

    // writeKeyframeUsedTimes(result_dir_ / "used_times");

    // if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask, gt_depth, sparse_depth;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];

        // gt_depth = viewpoint_cam->original_depth_.cuda();
        sparse_depth = viewpoint_cam->sparse_depth_.cuda();
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // DEBUG: track crash location
    {
        static int debug_iter = 0;
        std::cout << "[DEBUG] trainForOneIteration start, iter=" << getIteration()
                  << " fid=" << viewpoint_cam->fid_ << " gaussians=" << gaussians_->xyz_.size(0) << std::endl << std::flush;
        debug_iter++;
    }

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // 在 trainForOneIteration 开头附近添加
//    extern int wG[0], hG[0];
//    printf("===== DSO real image size =====\n");
//    printf("wG[0]=%d, hG[0]=%d\n", wG[0], hG[0]);
    // printf("DSO Hcalib: fx=%f fy=%f cx=%f cy=%f\n",
    //        pSLAM_->Hcalib.fxl(), pSLAM_->Hcalib.fyl(),
    //        pSLAM_->Hcalib.cxl(), pSLAM_->Hcalib.cyl());
    // printf("GaussianMapper intr: fx=%f fy=%f cx=%f cy=%f\n",
    //        viewpoint_cam->intr_[0], viewpoint_cam->intr_[1],
    //        viewpoint_cam->intr_[2], viewpoint_cam->intr_[3]);

    // Update learning rate
    if (pSLAM_) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
        float position_lr = gaussians_->updateLearningRate(step);
        setPositionLearningRateInit(position_lr);
    }
    else {
        gaussians_->updateLearningRate(getIteration());
    }

    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render
    // Outputs:
    // rgb_img, viewspace_points, visibility_filter, radii, render_alpha, render_normal, render_dist, surf_depth, surf_normal
    { std::cout << "[DEBUG] before render, iter=" << getIteration() << " gaussians=" << gaussians_->xyz_.size(0) << std::endl << std::flush; }
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    { std::cout << "[DEBUG] after render, iter=" << getIteration() << std::endl << std::flush; }
    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);
    // 2DGS
    // auto rend_alpha = std::get<4>(render_pkg);
    auto rend_normal = std::get<5>(render_pkg);
    auto rend_dist = std::get<6>(render_pkg);
    auto surf_depth = std::get<7>(render_pkg);
    auto surf_normal = std::get<8>(render_pkg);
    auto median_depth = std::get<9>(render_pkg);
    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;

    // Loss
    // original Gaussian Splatting
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();

    // 2DGS
    float lambda_normal = 0.01f;
    float lambda_dist = 0.0f;
    auto normal_error = (1 - (rend_normal * surf_normal)).sum(0).unsqueeze(0);
    auto normal_loss = (normal_error).mean();
    // auto dist_loss = (rend_dist).mean();

    auto ssim_val = loss_utils::ssim(masked_image, gt_image, device_type_);
    auto photo_normal_loss = (1.0 - lambda_dssim) * Ll1
                           + lambda_dssim * (1.0 - ssim_val)
                           + lambda_normal * normal_loss;
    auto loss = photo_normal_loss.clone();
    float depth_L1 = 0.0f;   // <--- 提前声明，默认值 0
    if (training_level == num_gaus_pyramid_sub_levels_) {
        torch::Tensor masked_depth = surf_depth * mask;
        // auto depth_L1 = loss_utils::l1_loss(masked_depth, gt_depth);
        // loss += 0.5 * depth_L1;

        auto sparse_depth_nonzero_mask = (sparse_depth != 0).to(masked_depth.dtype());
        auto valid_masked_depth = masked_depth * sparse_depth_nonzero_mask;
        auto valid_sparse_depth = sparse_depth * sparse_depth_nonzero_mask;
        auto depth_L1_tensor = torch::abs(valid_masked_depth - valid_sparse_depth).sum() / sparse_depth_nonzero_mask.sum();
        depth_L1 = depth_L1_tensor.item().toFloat();   // 在这里赋值
        loss += 5000 * depth_L1;

        // 8×8 block smoothness: uniform-color → normal + plane-depth (GPU) — DISABLED
        if (getIteration() > 1000) {
            float lambda_smooth = 1.0f;
            int BS = 8, H = gt_image.size(1), W = gt_image.size(2);
            int BH = H/BS, BW = W/BS;
            if (BH >= 1 && BW >= 1) {
                int hc = BH*BS, wc = BW*BS;
                auto rgb = gt_image.index({torch::indexing::Slice(), torch::indexing::Slice(0,hc), torch::indexing::Slice(0,wc)});
                auto dpt = surf_depth.index({torch::indexing::Slice(), torch::indexing::Slice(0,hc), torch::indexing::Slice(0,wc)});
                auto rgb_b = rgb.reshape({3, BH, BS, BW, BS}).permute({0,1,3,2,4}).reshape({3, BH*BW, BS*BS});
                auto dpt_b = dpt.reshape({1, BH, BS, BW, BS}).permute({0,1,3,2,4}).reshape({1, BH*BW, BS*BS});
                auto rgb_var = (rgb_b - rgb_b.mean(2,true)).pow(2).sum(0).mean(1);
                auto uniform = rgb_var < 0.0002f;
                if (uniform.any().item().toBool()) {
                    auto du = dpt_b.index({torch::indexing::Slice(), uniform, torch::indexing::Slice()});
                    auto md = du.mean(2, true);
                    auto dv = (du - md).abs().mean();
                    loss += lambda_smooth * dv;

                    // Visualize uniform blocks every 100 iters
                    // if (getIteration() % 100 == 0) {
                    // auto vis = gt_image.clone().detach();
                    // auto um = uniform.reshape({BH, BW}).cpu();
                    // auto ua = um.accessor<bool, 2>();
                    // // Yellow overlay on uniform blocks
                    // for (int by = 0; by < BH; by++)
                    //     for (int bx = 0; bx < BW; bx++)
                    //         if (ua[by][bx])
                    //             vis.index_put_({torch::indexing::Slice(),
                    //                 torch::indexing::Slice(by*BS, (by+1)*BS),
                    //                 torch::indexing::Slice(bx*BS, (bx+1)*BS)}, 0.7f);
                    // auto vis_img = vis.permute({1,2,0}).contiguous().cpu().mul(255).clamp(0,255).to(torch::kByte);
                    // cv::Mat cv_img(vis_img.size(0), vis_img.size(1), CV_8UC3, vis_img.data_ptr());
                    // cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);
                    // auto vis_dir = result_dir_ / "smooth_vis";
                    // std::filesystem::create_directories(vis_dir);
                    // cv::imwrite((vis_dir / (std::to_string(getIteration())+"_"+std::to_string(viewpoint_cam->fid_)+".jpg")).string(), cv_img);
                    // }
                }
            }
        }
    }
    // ==================================================================
    // 2.5. Old-frame photometric BA (DISABLED)
    // ==================================================================
    // if (ba_accum_iter_ >= 10 && (int)ba_points_.size() >= 50000 && (int)ba_frame_fids_.size() >= 4) {
    //     runMultiFrameBA();
    // }

    // ==================================================================
    // 3. 执行梯度反向传播
    // ==================================================================
    {
        float loss_val = loss.item().toFloat();
        std::cout << "[DEBUG] before backward, iter=" << getIteration()
                  << " loss=" << loss_val << " isnan=" << std::isnan(loss_val) << std::endl << std::flush;
        if (std::isnan(loss_val)) {
            std::cerr << "[FATAL] NaN loss detected, skipping backward" << std::endl;
            increaseIteration(-1);
            return;
        }
    }
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[DEBUG] CUDA error before backward: " << cudaGetErrorString(err) << std::endl << std::flush;
        }
    }
    loss.backward();

    torch::cuda::synchronize();
    // loss.backward();

    // torch::cuda::synchronize();
    { std::cout << "[DEBUG] after backward, iter=" << getIteration() << std::endl << std::flush; }
    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;
        kf_recent_loss_[viewpoint_cam->fid_] = photo_normal_loss.item().toFloat();
        { static std::ofstream llog("kf_loss_log.txt", std::ios::app);
          static int lc=0;
          if (lc == 0) { llog << "#iter fid L1 SSIM loss" << std::endl; lc++; }
          if (getIteration() % 100 == 0) {
              llog<<getIteration()<<" "<<viewpoint_cam->fid_<<" "
                  <<Ll1.item().toFloat()<<" "<<ssim_val.item().toFloat()<<" "
                  <<photo_normal_loss.item().toFloat()<<std::endl; llog.flush();
          }
        }
        { std::cout << "[DEBUG] E1: ema done, gaussians=" << gaussians_->xyz_.size(0) << std::endl << std::flush; }

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, surf_depth, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_, result_dir_);

        { std::cout << "[DEBUG] E2: record done" << std::endl << std::flush; }

        // Densification
        if (getIteration() < opt_params_.densify_until_iter_) {
            // Keep track of max radii in image-space for pruning
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));

            { std::cout << "[DEBUG] E3: radii updated" << std::endl << std::flush; }

            gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            { std::cout << "[DEBUG] E4: dens stats added" << std::endl << std::flush; }

            int effective_interval = (getIteration() < 3200) ? 100 : 50;
            if ((getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % effective_interval == 0)) {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 3000000 : 0;   // 20
                gaussians_->densifyAndPrune(
                    densifyGradThreshold(),
                    densify_min_opacity_,//0.005,//
                    scene_->cameras_extent_,
                    size_threshold
                );
                { std::cout << "[DEBUG] E5: densifyAndPrune done, gaussians=" << gaussians_->xyz_.size(0) << std::endl << std::flush; }
            }

            { std::cout << "[DEBUG] E6: densification block done" << std::endl << std::flush; }

            if (opacityResetInterval()
                && (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();
        }

        { std::cout << "[DEBUG] E7: training loop bottom, about to return" << std::endl << std::flush; }

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),
                opt_params_.iterations_,
                Ll1,
                loss,
                ema_loss_for_log_,
                loss_utils::l1_loss,
                iter_time,
                *gaussians_,
                *scene_,
                pipe_params_,
                background_
            );
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        // ablation study!!!!!!!!!!!
        // if (SLAM_ended_ && (getIteration() - additional_training_start_iter) % 100 == 0)
        // {
        //     int additional_training_iter = getIteration() - additional_training_start_iter;
        //     std::chrono::steady_clock::time_point start_saving = std::chrono::steady_clock::now();
        //     double duration = std::chrono::duration_cast<std::chrono::duration<double>>(start_saving - additional_training_start).count();
        //     std::ofstream outFile(result_dir_ / "additional_training_time.txt", std::ios::app);

        //     // outFile << "saving " << std::to_string(additional_training_iter) << " iteration time: " << duration << std::endl;
        //     outFile << duration << std::endl;
        //     savePly(result_dir_ / ("additional_" + std::to_string(additional_training_iter)) / "ply");

        //     std::chrono::steady_clock::time_point end_saving = std::chrono::steady_clock::now();
        //     duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_saving - additional_training_start).count();
        //     // outFile << "saving end" << std::to_string(additional_training_iter) << " iteration time: " << duration << std::endl;
        //     outFile << duration << std::endl;

        //     outFile.close();
        // }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();

            gaussians_->optimizer_->zero_grad(true);
        }
    }
    { std::cout << "[DEBUG] EXIT trainForOneIteration, iter=" << getIteration() << " gaussians=" << gaussians_->xyz_.size(0) << std::endl << std::flush; }
}

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

bool GaussianMapper::hasMetInitialMappingConditions()
{
    if (checkKFupdateFromGSRequested()) {
        pSLAM_->isDoneKFUpdateFromGS = true;
    }
    // if (pSLAM_->allKeyframeHessians.size() >= min_num_initial_map_kfs_)
    if (pSLAM_->initialized && pSLAM_->newframeHessians.size() >= min_num_initial_map_kfs_)
        return true;

    return false;
}

bool GaussianMapper::hasMetIncrementalMappingConditions()
{
    // if (inserted_kf_cursor < pSLAM_->lastKFIncomingId)
    if (pSLAM_->newframeHessians.size() > 0)
        return true;

    // inserted_kf_cursor

    return false;
}

bool GaussianMapper::checkKFupdateFromGSRequested()
{
    return pSLAM_->callKFUpdateFromGS;
}

void GaussianMapper::combineMappingOperations()
{
    std::vector<float> new_points;
    std::vector<float> new_colors;
    std::vector<float> new_scales;
    std::vector<float> new_rots;

    {
        boost::unique_lock<boost::mutex> allKFLock(pSLAM_->frozenMapMutex);
        std::vector<dso::FrozenFrameHessian*>* keyframesFromDso = &pSLAM_->newframeHessians;

        // while (keyframesFromDso.size() > 0) {
        int num_kf = keyframesFromDso->size();
        for (int idx=0; idx < num_kf; idx++) {
            // dso::FrozenFrameHessian* fh = keyframesFromDso.at(idx);
            dso::FrozenFrameHessian* fh = keyframesFromDso->front();
            
            if ( fh->pointsInWorld.empty())
                continue;

            std::cout << "New KF from DSO to 2DGS, incoming_id: " << fh->incomingID << std::endl;

            float fx = pSLAM_->Hcalib.fxl();
            float fy = pSLAM_->Hcalib.fyl();
            float cx = pSLAM_->Hcalib.cxl();
            float cy = pSLAM_->Hcalib.cyl();

            Sophus::SE3d c2w = fh->camToWorld;

            // int j = 0;
            // for (dso::PointHessian* ph : fh->pointHessians) {
            //     if (ph->idepth > 0) {
            //         float depth = 1.0f / ph->idepth;
            //         float x = ((ph->u - cx) / fx) * depth;
            //         float y = ((ph->v - cy) / fy) * depth;
            //         float z = depth;

            //         Eigen::Vector3d point_in_camera(x, y, z);
            //         Eigen::Vector3d point_in_world = c2w * point_in_camera;

            //         new_points.push_back(point_in_world.x());
            //         new_points.push_back(point_in_world.y());
            //         new_points.push_back(point_in_world.z());

            //         new_colors.push_back(ph->kf_color.at(0));
            //         new_colors.push_back(ph->kf_color.at(1));
            //         new_colors.push_back(ph->kf_color.at(2));
            //     }
            //     ++j;
            // }

            // get initial GS from DSO
            new_points.reserve(new_points.size() + fh->pointsInWorld.size());
            new_points.insert(new_points.end(), fh->pointsInWorld.begin(), fh->pointsInWorld.end());

            new_colors.reserve(new_colors.size() + fh->colors.size());
            new_colors.insert(new_colors.end(), fh->colors.begin(), fh->colors.end());

            new_scales.reserve(new_scales.size() + fh->scales.size());
            new_scales.insert(new_scales.end(), fh->scales.begin(), fh->scales.end());

            new_rots.reserve(new_rots.size() + fh->rots.size());
            new_rots.insert(new_rots.end(), fh->rots.begin(), fh->rots.end());

            // std::cout << "Num_points " << new_points.size()/3 << " Points" << std::endl;

            std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(fh->incomingID, getIteration());
            new_kf->zfar_ = z_far_;
            new_kf->znear_ = z_near_;
            // Pose
            // auto pose = pKF->GetPose();
            // Sophus::SE3d c2w = fh->shell->camToWorld; declared before
            {
                auto q_before = c2w.unit_quaternion();
                std::cout << "[KF DEBUG] fid=" << fh->incomingID
                          << " camToWorld_q(wxyz)=" << q_before.w()<<" "<<q_before.x()<<" "<<q_before.y()<<" "<<q_before.z()
                          << " camToWorld_t=" << c2w.translation().transpose() << std::endl;
            }
            c2w = c2w.inverse();
            {
                auto q_after = c2w.unit_quaternion();
                std::cout << "[KF DEBUG] fid=" << fh->incomingID
                          << " after_inv_q(wxyz)=" << q_after.w()<<" "<<q_after.x()<<" "<<q_after.y()<<" "<<q_after.z()
                          << " after_inv_t=" << c2w.translation().transpose() << std::endl;
            }
            new_kf->setPose(
                // pose.unit_quaternion().cast<double>(),
                // pose.translation().cast<double>()
                c2w.unit_quaternion(),
                c2w.translation());
            cv::Mat imgRGB_undistorted, imgAux_undistorted;
            try {
                // Add first keyframe to the scene
                Camera& camera = scene_->cameras_.at(0);
                new_kf->setCameraParams(camera);

                // Image (left if STEREO)
                dso::MinimalImageB3* img = fh->kfImg;
                cv::Mat imgRGB(img->h, img->w, CV_8UC3, img->data);

                if (need_distortion) {
                    camera.undistortImage(imgRGB, imgRGB_undistorted);
                } else {
                    imgRGB_undistorted = imgRGB;
                }
                // camera.undistortImage(imgRGB, imgRGB_undistorted);
                // imgRGB_undistorted = imgRGB;

                imgAux_undistorted = imgRGB_undistorted;
                imgRGB_undistorted.convertTo(imgRGB_undistorted, CV_32FC3, 1.0 / 255.0);
                new_kf->original_image_ =
                    tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);

                // Depth image
                // dso::MinimalImage<unsigned short>* depth_img = fh->kf_depth;
                // cv::Mat imgDepth(depth_img->h, depth_img->w, CV_16UC1, depth_img->data);
                // cv::Mat imgDepthFloat;
                // imgDepth.convertTo(imgDepthFloat, CV_32FC1, 1.0 / depth_scale);
                // new_kf->original_depth_ =
                //     tensor_utils::cvMat2TorchTensor_Float32(imgDepthFloat, device_type_);

                // Sparse Depth
                new_kf->sparse_depth_ =
                    tensor_utils::cvMat2TorchTensor_Float32(fh->kfSparseDepth, device_type_);

                new_kf->img_filename_ = result_dir_ / "img" / (std::to_string(fh->incomingID) + ".png");
                new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
            }
            catch (std::out_of_range) {
                throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
            }
            new_kf->computeTransformTensors();
            scene_->addKeyframe(new_kf, &kfid_shuffled_);
            increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());
            // Features
            // std::vector<float> pixels;
            // std::vector<float> pointsLocal;
            // pKF->GetKeypointInfo(pixels, pointsLocal);
            // new_kf->kps_pixel_ = std::move(pixels);
            // new_kf->kps_point_local_ = std::move(pointsLocal);
            new_kf->img_undist_ = imgRGB_undistorted;
            new_kf->img_auxiliary_undist_ = imgAux_undistorted;
            // Prepare multi resolution images for training

            // 新增：拷贝观测信息
            // ====== 新增：拷贝观测信息 ======
            new_kf->observations.resize(fh->observations.size());
            for (size_t pt_idx = 0; pt_idx < fh->observations.size(); ++pt_idx) {
                const auto& obs_list_src = fh->observations[pt_idx];
                auto& obs_list_dst = new_kf->observations[pt_idx];
                obs_list_dst.resize(obs_list_src.size());
                for (size_t obs_idx = 0; obs_idx < obs_list_src.size(); ++obs_idx) {
                    const auto& src = obs_list_src[obs_idx];
                    obs_list_dst[obs_idx].target_fid = src.target_frame_id;
                    obs_list_dst[obs_idx].observed_pixel = cv::Point2f(src.observed_u, src.observed_v);
                }
            }

            // ====== 填充本帧像素坐标和深度（用于反投影） ======
            if (fh->points_u.size() == fh->observations.size() &&
                fh->points_v.size() == fh->observations.size()) {
                new_kf->point_pixels.reserve(fh->points_u.size());
                new_kf->point_depths.reserve(fh->points_u.size());
                for (size_t pt_idx = 0; pt_idx < fh->points_u.size(); ++pt_idx) {
                    float u = fh->points_u[pt_idx];
                    float v = fh->points_v[pt_idx];
                    new_kf->point_pixels.emplace_back(u, v);

                    // 深度从 fh->kfSparseDepth（cv::Mat）获取
                    float depth = fh->kfSparseDepth.at<float>(v, u);
                    new_kf->point_depths.push_back(depth);
                }
            } else {
                std::cerr << "[Warning] point coordinates size mismatch for keyframe " << fh->incomingID << std::endl;
            }

            if (device_type_ == torch::kCUDA) {
                cv::cuda::GpuMat img_gpu;
                img_gpu.upload(new_kf->img_undist_);
                new_kf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                    cv::cuda::GpuMat img_resized;
                    cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(new_kf->gaus_pyramid_width_[l], new_kf->gaus_pyramid_height_[l]));
                    new_kf->gaus_pyramid_original_image_[l] =
                        tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                }
            }
            else {
                new_kf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                    cv::Mat img_resized;
                    cv::resize(new_kf->img_undist_, img_resized,
                                cv::Size(new_kf->gaus_pyramid_width_[l], new_kf->gaus_pyramid_height_[l]));
                    new_kf->gaus_pyramid_original_image_[l] =
                        tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                }
            }

            inserted_kf_cursor = fh->incomingID;

            keyframesFromDso->erase(keyframesFromDso->begin());
        }
    }

    // Add new gaussians
    if (!new_points.empty() and initial_mapped_) {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // gaussians_->increasePcd(new_points, new_colors, getIteration());
        gaussians_->increasePcd(new_points, new_colors, new_scales, new_rots, getIteration());
        // std::cout << "Inserted " << new_points.size()/3 << " Points to the GS Scene" << std::endl;
        // std::cout << "add new points to the scene" << std::endl;
    }

    updateGSKeyFramesFromDSO();

    // if (scene_->keyframes_.size() > 10)
    //     updateKeyFramesFromGS();
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>> GaussianMapper::renderKFDepths(std::vector<Sophus::SE3d> kf_poses)
{
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> depth_maps_out;

    // render depth imgs
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    for (Sophus::SE3d kfPose : kf_poses)
    {
        kfPose = kfPose.inverse();
        std::tuple<torch::Tensor, torch::Tensor> renderedDepth = renderDepthFromPose(kfPose, image_width, image_height, false);
        depth_maps_out.push_back(renderedDepth);
    }

    return depth_maps_out;
}

void GaussianMapper::handleNewKeyframe(
    std::tuple< unsigned long/*Id*/,
                unsigned long/*CameraId*/,
                Sophus::SE3f/*pose*/,
                cv::Mat/*image*/,
                bool/*isLoopClosure*/,
                cv::Mat/*auxiliaryImage*/,
                std::vector<float>,
                std::vector<float>,
                std::string> &kf)
{
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        // Image (left if STEREO)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
            
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    // Add the new keyframe to the scene
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }
}

void GaussianMapper::generateKfidRandomShuffle()
{
    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    // std::shuffle(kfid_in_dso_active_window.begin(), kfid_in_dso_active_window.end(), g);

    // std::cout << "Shuffled: " << kfid_in_dso_active_window << std::endl;

    kfid_training = kfid_in_dso_active_window.size() - 1;
    kfid_shuffle_idx_ = 0;
    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
    if (scene_->keyframes().empty() || kfid_in_dso_active_window.empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

	std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
	if (kfid_shuffled_) {
		int start_shuffle_idx = kfid_shuffle_idx_;
		do {
			++kfid_shuffle_idx_;
			if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
				kfid_shuffle_idx_ = 0;
			if (kfid_shuffle_idx_ == start_shuffle_idx)
				for (auto& kfit : scene_->keyframes())
					increaseKeyframeTimesOfUse(kfit.second, 1);
			int random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
			auto random_cam_it = scene_->keyframes().begin();
			for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
				++random_cam_it;
			viewpoint_cam = (*random_cam_it).second;
		} while (viewpoint_cam->remaining_times_of_use_ <= 0);
	}

	auto fid = viewpoint_cam->fid_;
	if (kfs_used_times_.find(fid) == kfs_used_times_.end())
		kfs_used_times_[fid] = 1;
	else
		++kfs_used_times_[fid];
	--(viewpoint_cam->remaining_times_of_use_);
	return viewpoint_cam;

}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    --(viewpoint_cam->remaining_times_of_use_);

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes()
{
    // TODO
    // std::unordered_set<unsigned long> kfids =
    //     pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
    // std::vector<unsigned long> kfids_to_erase;
    // std::size_t nkfs = scene_->keyframes().size();
    // kfids_to_erase.reserve(nkfs);
    // for (auto& kfit : scene_->keyframes()) {
    //     unsigned long kfid = kfit.first;
    //     if (kfids.find(kfid) == kfids.end()) {
    //         kfids_to_erase.emplace_back(kfid);
    //     }
    // }

    // for (auto& kfid : kfids_to_erase) {
    //     scene_->keyframes().erase(kfid);
    // }
}

void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf)
{
// auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(), {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(), {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);

        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor, colors,
                monocular_inactive_geo_densify_max_pixel_dist_, pkf->intr_, pkf->image_width_);
        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid = std::get<1>(result);
        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);
        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case STEREO:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // From CV_32FC3 to CV_32FC1
        cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // From CV_32FC1 to CV_8UC1
        gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to get 3D points
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();
        torch::Tensor points3D = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();
    
        // Clear undisired and unreliable stereo points
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
            // if (u < 0.3 * width || u > 0.7 * width)
            point_valid_flags[idx] = true;
            // idx += width;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx - 3] = true;
            //     point_valid_flags[idx - 2] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx] = true;
            // }
            // idx -= (2 * width);
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            //     point_valid_flags[idx + 3] = true;
            // }
            // idx += width;
            // idx += 3;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx - 2] = true;
            // }
            // idx -= 6;
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            // }
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()), true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()), true, false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid = colors.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case RGBD:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        // To clear undisired and unreliable depth
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)}, false/*true*/, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));

        torch::Tensor colors_valid = rgb.index({point_valid_flags});

        // Reproject to get 3D points
        torch::Tensor points3D_valid;
        Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_valid = reprojectDepthPinhole(
                depth, point_valid_flags, pkf->intr_, pkf->image_width_);
        }
        break;
        case Camera::FISHEYE:
        {
            //TODO: support fisheye camera?
            throw std::runtime_error("[Gaussian Mapper]Fisheye cameras are not supported currently!");
        }
        break;
        default:
        {
            throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        }
        break;
        }
        points3D_valid = points3D_valid.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;
        // Add new points to the model
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_, getIteration());
    }

// auto end_timing = std::chrono::steady_clock::now();
// auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                 end_timing - start_timing).count();
// std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify() takes "
//             << completion_time
//             << " ms"
//             << std::endl;
}

// bool GaussianMapper::needInterruptTraining()
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     return this->interrupt_training_;
// }

// void GaussianMapper::setInterruptTraining(const bool interrupt_training)
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     this->interrupt_training_ = interrupt_training;
// }

void GaussianMapper::recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        torch::Tensor &depth_map,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::filesystem::path result_depth_dir,
        std::string name_suffix)
{
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
        // cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
        // cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        // cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }

    if (record_depth_image_) {
        depth_map = depth_map.squeeze(0);
        depth_map = depth_map.to(torch::kCPU);
        depth_map = depth_map.contiguous();
        // std::cout << depth_map.sizes() << std::endl;
        cv::Mat depth_cv(depth_map.size(0), depth_map.size(1), CV_32FC1, depth_map.data_ptr<float>());

        // auto depth_cv = tensor_utils::torchTensor2CvMat_Float32(depth_map);
        cv::Mat depth_cv_normalized, depth_clipped;
        // clip depth
        cv::threshold(depth_cv, depth_clipped, 10, 10, cv::THRESH_TRUNC);
        cv::threshold(depth_clipped, depth_clipped, 0, 0, cv::THRESH_TOZERO);

        // cv::normalize(depth_clipped, depth_cv_normalized, 0, 255, cv::NORM_MINMAX);
        // depth_cv_normalized.convertTo(depth_cv_normalized, CV_8UC1);
        // cv::imwrite(result_depth_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".png"), depth_cv_normalized);
    
        depth_clipped *= 6553.5;
        depth_clipped.convertTo(depth_clipped, CV_16U);
        cv::imwrite(result_depth_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".png"), depth_clipped);
    }
}

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision) {
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
        masked_image = masked_image.index_select(0, torch::tensor({2, 1, 0}).to(device_type_)); // bgr rgb convert
        // std::cout << masked_image.size(0) << " " << masked_image.size(1) << " " << masked_image.size(2) << std::endl;
    }
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

std::tuple<torch::Tensor, torch::Tensor> GaussianMapper::renderDepthFromPose(
    const Sophus::SE3d &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    torch::NoGradGuard no_grad;

    // if (!initial_mapped_ || getIteration() <= 0)
    //     return cv::Mat(height, width, CV_32FC1, 0.0f);

    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion(),
        Tcw.translation());

    try {
        // Camera
        Camera& camera = scene_->cameras_.at(0);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        // std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    // torch::Tensor masked_image;
    // if (main_vision) {
    //     masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    //     masked_image = masked_image.index_select(0, torch::tensor({2, 1, 0}).to(device_type_)); // bgr rgb convert
    //     // std::cout << masked_image.size(0) << " " << masked_image.size(1) << " " << masked_image.size(2) << std::endl;
    // }
    // else
    //     masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    
    torch::Tensor masked_depth, masked_alpha;
    // median depth
    // masked_depth = std::get<9>(render_pkg) * undistort_mask_[0];
    // masked_depth = std::get<9>(render_pkg);
    // expected depth
    masked_depth = std::get<7>(render_pkg) * undistort_mask_[0];
    // masked_depth = std::get<7>(render_pkg);

    masked_alpha = std::get<4>(render_pkg) * undistort_mask_[0];
    // masked_alpha = std::get<4>(render_pkg);

    std::tuple<torch::Tensor, torch::Tensor> out_info = std::make_tuple(masked_depth, masked_alpha);

    return out_info;
    // return tensor_utils::torchTensor2CvMat_Float32(masked_depth);
}

cv::Mat GaussianMapper::rendercvMatDepthFromPose(
    const Eigen::Matrix4f &Tcw_)
{
    torch::NoGradGuard no_grad;
    const int width = image_width;
    const int height = image_height;

    Sophus::SE3d Tcw(Tcw_.inverse().matrix().cast<double>());

    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion(),
        Tcw.translation());

    try {
        // Camera
        Camera& camera = scene_->cameras_.at(0);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        // std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    
    torch::Tensor masked_depth;
    // median depth
    // masked_depth = std::get<9>(render_pkg) * undistort_mask_[pkf->camera_id_];
    // masked_depth = std::get<9>(render_pkg);
    // expected depth
    masked_depth = std::get<7>(render_pkg);

    // return masked_depth;
    return tensor_utils::torchTensor2CvMat_Float32(masked_depth);
}

void GaussianMapper::updateGSKeyFramesFromDSO()
{
    // -- DSO to 2DGS -- //
    // update sparse depth and keyframe poses
    boost::unique_lock<boost::mutex> frozenKFLock(pSLAM_->frozenMapMutex);
    
    kfid_in_dso_active_window.clear();

    for (dso::FrozenFrameHessian* kfs : pSLAM_->frameHessiansFrozen)
    {
        Sophus::SE3d c2w = kfs->camToWorld;
        auto gsKeyframe = scene_->getKeyframe(kfs->incomingID);

        if (gsKeyframe != nullptr) {
            // update pose: camToWorld → worldToCam
            c2w = c2w.inverse();
            gsKeyframe->setPose(
                c2w.unit_quaternion(),
                c2w.translation()
            );

            // update sparse depth
            gsKeyframe->sparse_depth_ =
                tensor_utils::cvMat2TorchTensor_Float32(kfs->kfSparseDepth, device_type_);

            kfid_in_dso_active_window.push_back(kfs->incomingID);
        }
    }
}

void GaussianMapper::updateKeyFramesFromGS()
{
    // -- GS depth → DSO ImmaturePoint prior -- //
	boost::unique_lock<boost::mutex> lock(pSLAM_->mapMutex);

	std::vector<Sophus::SE3d> kf_poses;
	// get keyframe poses
	// for(dso::FrameHessian* kfs : pSLAM_->frameHessians)
	// {
	// 	Sophus::SE3d c2w = kfs->shell->camToWorld;
	// 	kf_poses.push_back(c2w);
	// }

    for (int i=0; i < pSLAM_->frameHessians.size()-2; i++)
	{
        auto kf = pSLAM_->frameHessians[i];
        Sophus::SE3d c2w = kf->shell->camToWorld;
        kf_poses.push_back(c2w);
    }

	// render depths (with keyframe poses)
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> depth_maps = renderKFDepths(kf_poses);
    c10::cuda::CUDACachingAllocator::emptyCache();
	// get inverse depth points from depths & update current keyframe's idepth values
    for (int i=0; i < pSLAM_->frameHessians.size()-2; i++)
	{
        auto kf = pSLAM_->frameHessians[i];

        // std::cout << kf->pointHessians.size() << std::endl;

        // if (kf->pointHessians.empty())
        //     continue;

        torch::Tensor& renderedDepth = std::get<0>(depth_maps[i]);
        torch::Tensor& renderedAlpha = std::get<1>(depth_maps[i]);

        static std::ofstream iplog(result_dir_ / "ip_depth_log.txt", std::ios::app);
        for (size_t j = 0; j < kf->immaturePoints.size(); ++j) {
            dso::ImmaturePoint* ih = kf->immaturePoints[j];
            if (!ih) continue;
            // DSO activation pre-conditions: only help points worth keeping
            int tr = (int)ih->lastTraceStatus; // IPS_GOOD=0, SKIP=1, BADCONDITION=2, OOB=3, OUTLIER=4, UNINIT=5
            bool trace_ok = (tr >= 0 && tr <= 3);                     // not outlier/uninit
            bool quality_ok = ih->quality > 3.0f;                     // unique enough
            bool depth_pos = (ih->idepth_min + ih->idepth_max) > 0;  // positive depth
            if (!trace_ok || !quality_ok || !depth_pos) continue;     // skip unpromising points
            float ra = renderedAlpha.index({0, int(ih->v), int(ih->u)}).item().toFloat();
            float rd = renderedDepth.index({0, int(ih->v), int(ih->u)}).item().toFloat();
            bool need_help = !std::isfinite(ih->idepth_max) || (ih->lastTracePixelInterval > 8.0f);
            if (need_help && ra > 0.95 && rd > 0.1 && rd < 10.0) {
                // ============================================================
                //  GS depth → idepth range, using DSO's own uncertainty width
                //
                //  Key insight: DSO already computes a reliable uncertainty range
                //  [idepth_min, idepth_max] from multi-frame epipolar search.
                //  We keep DSO's width, but center it at the GS-estimated depth.
                //  GS provides the best-guess CENTER, DSO provides the WIDTH.
                // ============================================================
                float idepth_gs = 1.0f / rd;
                float idm, idx;

                if (std::isfinite(ih->idepth_max)) {
                    // DSO has traced this point → use DSO's own range width
                    float dso_half_width = (ih->idepth_max - ih->idepth_min) * 0.5f;
                    idm = idepth_gs - dso_half_width;
                    idx = idepth_gs + dso_half_width;
                } else {
                    // Never traced → DSO gives [0, inf]. Use a tight default around GS.
                    // 5% is a reasonable first guess; DSO's traceOn will tighten it.
                    idm = 1.0f / (rd * 1.05f);
                    idx = 1.0f / (rd * 0.95f);
                }

                float old_m = ih->idepth_min, old_x = ih->idepth_max;
                if (!std::isfinite(ih->idepth_max) || idm > ih->idepth_min) ih->idepth_min = idm;
                if (!std::isfinite(ih->idepth_max) || idx < ih->idepth_max) ih->idepth_max = idx;
                if (ih->idepth_min > ih->idepth_max) std::swap(ih->idepth_min, ih->idepth_max);
                iplog << " fid=" << kf->shell->incoming_id
                      << " u=" << ih->u << " v=" << ih->v << " gs_d=" << rd
                      << " dso_w=" << ((std::isfinite(old_x)) ? (old_x - old_m) : -1.0f)
                      << " old=[" << old_m << "," << old_x << "]"
                      << " gs=[" << idm << "," << idx << "]"
                      << " final=[" << ih->idepth_min << "," << ih->idepth_max << "]" << std::endl;
            }
        }
        iplog.flush();
        // }
	}
}
void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float &dssim,
    float &psnr,
    float &psnr_gs,
    double &render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::filesystem::path result_depth_dir,
    std::string name_suffix)
{
    auto start_timing = std::chrono::steady_clock::now();
    auto render_pkg = GaussianRenderer::render(
        pkf,
        pkf->image_height_,
        pkf->image_width_,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get<0>(render_pkg);
    // expected depth
    auto surf_depth = std::get<7>(render_pkg);
    // median depth
    // auto surf_depth = std::get<9>(render_pkg);
    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    // torch::Tensor masked_dist = surf_depth * undistort_mask_[pkf->camera_id_];
    torch::Tensor masked_dist = surf_depth;
    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;
    auto gt_image = pkf->original_image_;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    // std::cout << masked_image.sizes() << std::endl;
    // std::cout << gt_image.sizes() << std::endl;
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, gt_image, masked_dist, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, result_depth_dir, name_suffix);    
}

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path depth_dir = result_dir / "depth";
    if (record_depth_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(depth_dir);
    }

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir, depth_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
    }
}

void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    gaussians_->savePly(result_dir / "point_cloud.ply");
    gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    this->gaussians_->loadPly(ply_path);

    // Camera
    if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if(!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        Camera camera;
        camera.camera_id_ = 0;
        camera.width_ = camera_file["Camera.w"].operator int();
        camera.height_ = camera_file["Camera.h"].operator int();

        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole") {
            camera.setModelId(Camera::CameraModelType::PINHOLE);

            float fx = camera_file["Camera.fx"].operator float();
            float fy = camera_file["Camera.fy"].operator float();
            float cx = camera_file["Camera.cx"].operator float();
            float cy = camera_file["Camera.cy"].operator float();

            float k1 = camera_file["Camera.k1"].operator float();
            float k2 = camera_file["Camera.k2"].operator float();
            float p1 = camera_file["Camera.p1"].operator float();
            float p2 = camera_file["Camera.p2"].operator float();
            float k3 = camera_file["Camera.k3"].operator float();

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << fx, 0.f, cx,
                        0.f, fy, cy,
                        0.f, 0.f, 1.f
            );

            camera.params_[0] = fx;
            camera.params_[1] = fy;
            camera.params_[2] = cx;
            camera.params_[3] = cy;

            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

        }
        else {
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    // Ready
    this->initial_mapped_ = true;
    increaseIteration();
}

void GaussianMapper::runMultiFrameBA()
{
    if (ba_points_.empty() || ba_frame_id_to_idx_.empty()) return;
    int N_frames = (int)ba_frame_fids_.size();
    if (N_frames < 3) return;

    // Build camera→world poses from GS keyframes
    std::vector<Sophus::SE3d> poses_c2w(N_frames);
    for (int i = 0; i < N_frames; ++i) {
        auto kf = scene_->getKeyframe(ba_frame_fids_[i]);
        if (!kf) { ba_points_.clear(); ba_frame_id_to_idx_.clear(); ba_frame_fids_.clear(); ba_accum_iter_=0; return; }
        Sophus::SE3d w2c(kf->R_quaternion_, kf->t_);
        poses_c2w[i] = w2c.inverse();
    }

    // Convert X_ref_cam → X_world, skip failed
    for (int pi = 0; pi < (int)ba_points_.size(); ++pi) {
        auto it = ba_frame_id_to_idx_.find(ba_points_[pi].ref_frame_id);
        if (it == ba_frame_id_to_idx_.end()) { ba_points_[pi].X_world.setZero(); continue; }
        Sophus::SE3d w2c = poses_c2w[it->second].inverse();
        ba_points_[pi].X_world = w2c.rotationMatrix().transpose() * (ba_points_[pi].X_ref_cam - w2c.translation());
        if (!std::isfinite(ba_points_[pi].X_world.squaredNorm())) ba_points_[pi].X_world.setZero();
    }

    // Precompute grayscale images
    struct FrameImg { cv::Mat I; cv::Mat dx, dy; int W, H; };
    std::map<int, FrameImg> fimgs;
    for (int i = 0; i < N_frames; ++i) {
        auto kf = scene_->getKeyframe(ba_frame_fids_[i]);
        if (!kf) continue;
        FrameImg fi;
        cv::Mat gray; cv::cvtColor(kf->img_undist_, gray, cv::COLOR_RGB2GRAY);
        gray.copyTo(fi.I);
        cv::Sobel(fi.I, fi.dx, CV_32F, 1, 0, 3);
        cv::Sobel(fi.I, fi.dy, CV_32F, 0, 1, 3);
        fi.W = kf->image_width_; fi.H = kf->image_height_;
        fimgs[i] = fi;
    }

    // DSO 8-point pattern
    static const int photo_pat[8][2] = {{-3,0},{0,-3},{3,0},{0,3},{2,2},{-2,2},{2,-2},{-2,-2}};

    // Merge points sharing a frame AND close pixel → more observations per point
    {
        std::unordered_map<uint64_t, std::vector<int>> pixel_groups;
        for (int pi = 0; pi < (int)ba_points_.size(); ++pi) {
            if (ba_points_[pi].X_world.squaredNorm() < 1e-10) continue;
            for (auto& o : ba_points_[pi].obs) {
                uint64_t key = ((uint64_t)o.frame_id << 32) | (((int)o.u & 0xFFFF) << 16) | ((int)o.v & 0xFFFF);
                pixel_groups[key].push_back(pi);
            }
        }
        double merge_dist2 = 0.01 * 0.01;  // 1cm²
        std::vector<int> parent(ba_points_.size());
        for (int i = 0; i < (int)parent.size(); ++i) parent[i] = i;
        std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : (parent[x] = find(parent[x])); };
        for (auto& kv : pixel_groups) {
            auto& ids = kv.second;
            for (size_t a = 0; a < ids.size(); ++a)
                for (size_t b = a+1; b < ids.size(); ++b)
                    if ((ba_points_[ids[a]].X_world - ba_points_[ids[b]].X_world).squaredNorm() < merge_dist2)
                        parent[find(ids[a])] = find(ids[b]);
        }
        std::unordered_map<int, std::vector<int>> groups;
        for (int pi = 0; pi < (int)ba_points_.size(); ++pi) groups[find(pi)].push_back(pi);
        std::vector<BAPoint> merged;
        for (auto& g : groups) {
            if (g.second.size() == 1) { merged.push_back(ba_points_[g.second[0]]); continue; }
            BAPoint mp;
            mp.X_world.setZero(); mp.ref_frame_id = (unsigned long)-1;
            std::map<int, std::pair<double,double>> obs_map;
            for (int pi : g.second) {
                mp.X_world += ba_points_[pi].X_world;
                for (auto& o : ba_points_[pi].obs)
                    if (obs_map.find(o.frame_id) == obs_map.end())
                        obs_map[o.frame_id] = {o.u, o.v};
            }
            mp.X_world /= (double)g.second.size();
            if (obs_map.size() < 3) { for (int pi : g.second) merged.push_back(ba_points_[pi]); }
            else { for (auto& kv : obs_map) { BAObservation o; o.frame_id=kv.first; o.u=(float)kv.second.first; o.v=(float)kv.second.second; mp.obs.push_back(o); } merged.push_back(mp); }
        }
        ba_points_ = merged;
    }

    // Subsampling: max 30K points
    // DSO-style: project each point to all BA frames, add observations
    int expanded = 0;
    for (int pi = 0; pi < (int)ba_points_.size(); ++pi) {
        if (ba_points_[pi].X_world.squaredNorm() < 1e-10) continue;
        // Collect existing frame IDs for this point
        std::set<int> existing_fids;
        for (auto& o : ba_points_[pi].obs) existing_fids.insert(o.frame_id);
        for (int fi = 0; fi < N_frames; ++fi) {
            if (existing_fids.count(ba_frame_fids_[fi])) continue;  // already observing
            auto kf = scene_->getKeyframe(ba_frame_fids_[fi]);
            if (!kf) continue;
            double fx=kf->intr_[0],fy=kf->intr_[1],cx=kf->intr_[2],cy=kf->intr_[3];
            Sophus::SE3d w2c = poses_c2w[fi].inverse();
            Eigen::Vector3d Xc = w2c.rotationMatrix() * ba_points_[pi].X_world + w2c.translation();
            if (Xc.z() < 0.01) continue;
            double u = fx*Xc.x()/Xc.z() + cx;
            double v = fy*Xc.y()/Xc.z() + cy;
            if (u<4||u>kf->image_width_-5||v<4||v>kf->image_height_-5) continue;
            BAObservation o; o.frame_id = ba_frame_fids_[fi]; o.u = (float)u; o.v = (float)v;
            ba_points_[pi].obs.push_back(o);
            expanded++;
        }
    }
    // DSO-style: filter points with <3 good photometric observations
    {
        float q_th = 0.1f; // intensity threshold
        int filtered = 0;
        for (int pi = 0; pi < (int)ba_points_.size(); ++pi) {
            auto& bp = ba_points_[pi];
            if (bp.X_world.squaredNorm() < 1e-10 || bp.obs.size() < 2) { bp.obs.clear(); filtered++; continue; }
            auto rit = ba_frame_id_to_idx_.find(bp.obs[0].frame_id);
            if (rit == ba_frame_id_to_idx_.end()) { bp.obs.clear(); filtered++; continue; }
            auto rfi = fimgs.find(rit->second);
            if (rfi == fimgs.end()) { bp.obs.clear(); filtered++; continue; }
            float rru=bp.obs[0].u, rrv=bp.obs[0].v;
            int rui=(int)rru, rvi=(int)rrv;
            if (rui<1||rui>=rfi->second.W-1||rvi<1||rvi>=rfi->second.H-1) { bp.obs.clear(); filtered++; continue; }
            float I_ref = rfi->second.I.ptr<float>(rvi)[rui];
            int good = 0;
            for (int oi = 1; oi < (int)bp.obs.size(); ++oi) {
                auto oit = ba_frame_id_to_idx_.find(bp.obs[oi].frame_id);
                if (oit == ba_frame_id_to_idx_.end()) continue;
                auto ofi = fimgs.find(oit->second);
                if (ofi == fimgs.end()) continue;
                int oui=(int)bp.obs[oi].u, ovi=(int)bp.obs[oi].v;
                if (oui<1||oui>=ofi->second.W-1||ovi<1||ovi>=ofi->second.H-1) continue;
                float In = ofi->second.I.ptr<float>(ovi)[oui];
                if (std::abs(In - I_ref) < q_th) good++;
            }
            if (good < 3) { bp.obs.clear(); filtered++; }
        }
        ba_points_.erase(std::remove_if(ba_points_.begin(), ba_points_.end(),
            [](const BAPoint& bp){ return bp.obs.empty(); }), ba_points_.end());
    }
    int N_points_before = (int)ba_points_.size();
    int top_n = std::min(N_points_before, 30000);
    if (N_points_before > top_n) {
        std::vector<std::pair<int,int>> idx_obs(N_points_before);
        for (int i = 0; i < N_points_before; ++i) idx_obs[i] = {i, (int)ba_points_[i].obs.size()};
        std::partial_sort(idx_obs.begin(), idx_obs.begin()+top_n, idx_obs.end(),
            [](auto& a, auto& b){ return a.second > b.second; });
        std::vector<BAPoint> sampled;
        for (int i = 0; i < top_n; ++i) sampled.push_back(ba_points_[idx_obs[i].first]);
        ba_points_ = sampled;
    }
    int N_points = (int)ba_points_.size();
    int n_merged = 0, n_solo = 0;
    for (int pi = 0; pi < N_points; ++pi) {
        if ((int)ba_points_[pi].obs.size() > 2) n_merged++; else n_solo++;
    }

    const double photo_huber = 0.05;
    double lambda = 1e-3;
    // Save original DSO poses before optimization
    std::vector<Sophus::SE3d> poses_original = poses_c2w;

    // DSO-style: store and initialize inverse depth per point
    std::vector<double> idepth_i(N_points);
    for (int pi = 0; pi < N_points; ++pi) {
        unsigned long host_fid = ba_points_[pi].ref_frame_id;
        if (host_fid == (unsigned long)-1 && !ba_points_[pi].obs.empty()) host_fid = ba_points_[pi].obs[0].frame_id;
        if (ba_points_[pi].X_world.squaredNorm() < 1e-10) { idepth_i[pi] = 0.1; continue; }
        auto hit = ba_frame_id_to_idx_.find(host_fid);
        if (hit == ba_frame_id_to_idx_.end()) { idepth_i[pi] = 0.1; continue; }
        Sophus::SE3d hw2c = poses_c2w[hit->second].inverse();
        double z = (hw2c.rotationMatrix() * ba_points_[pi].X_world + hw2c.translation()).z();
        idepth_i[pi] = (z > 1e-6) ? 1.0 / z : 0.1;
    }

    std::vector<double> b_frames(N_frames, 0.0);

    double init_rmse = 0;
    int init_cnt = 0;

    for (int iter = 0; iter < 10; ++iter) {
        const int DS = 7; // 6 pose + 1 brightness
        int dim_pose = N_frames * DS;
        Eigen::MatrixXd H_pp = Eigen::MatrixXd::Zero(dim_pose, dim_pose);
        Eigen::VectorXd b_p  = Eigen::VectorXd::Zero(dim_pose);
        double total_err = 0;
        int total_obs = 0;
        // DSO-style: per-point 1DOF inverse depth accumulators
        std::vector<double> H_dd_i(N_points, 0.0);    // scalar idepth Hessian
        std::vector<double> b_d_i(N_points, 0.0);     // scalar idepth gradient
        std::vector<std::vector<std::pair<int,Eigen::Matrix<double,6,1>>>> H_pd_i(N_points); // 7×1 per obs [pose6, b=0]

        for (int pi = 0; pi < N_points; ++pi) {
            const BAPoint& bp = ba_points_[pi];
            if (bp.X_world.squaredNorm() < 1e-10 || bp.obs.size() < 2) continue;
            // Ref intensity
            auto ref_it = ba_frame_id_to_idx_.find(bp.obs[0].frame_id);
            if (ref_it == ba_frame_id_to_idx_.end()) continue;
            int ref_fi = ref_it->second;
            auto rf = fimgs.find(ref_fi);
            if (rf == fimgs.end()) continue;
            float rru=bp.obs[0].u, rrv=bp.obs[0].v;
            int rui=(int)rru, rvi=(int)rrv;
            // DSO-style: 8 pattern intensities from ref frame
            float I_ref_pat[8];
            bool ref_ok = true;
            for (int k=0; k<8; ++k) {
                int rpx = rui+photo_pat[k][0], rpy = rvi+photo_pat[k][1];
                if (rpx<0||rpx>=rf->second.W||rpy<0||rpy>=rf->second.H) { ref_ok=false; break; }
                I_ref_pat[k] = rf->second.I.ptr<float>(rpy)[rpx];
            }
            if (!ref_ok) continue;

            for (int oi = 1; oi < (int)bp.obs.size(); ++oi) {
                auto it = ba_frame_id_to_idx_.find(bp.obs[oi].frame_id);
                if (it == ba_frame_id_to_idx_.end()) continue;
                int fi = it->second;
                auto fm = fimgs.find(fi);
                if (fm == fimgs.end()) continue;
                auto kf = scene_->getKeyframe(bp.obs[oi].frame_id);
                if (!kf) continue;
                double fx=kf->intr_[0],fy=kf->intr_[1],cx=kf->intr_[2],cy=kf->intr_[3];

                Sophus::SE3d w2c = poses_c2w[fi].inverse();
                Eigen::Vector3d Xc = w2c.rotationMatrix() * bp.X_world + w2c.translation();
                if (Xc.z() <= 1e-6 || !std::isfinite(Xc.squaredNorm())) continue;
                // DSO-style: 8 pattern points — correct Rplane = K_near * R_rel * K_ref^{-1}
                auto ref_kf = scene_->getKeyframe(bp.obs[0].frame_id);
                double rfx=600,rfy=600,rcx=599.5,rcy=339.5;
                if (ref_kf) { rfx=ref_kf->intr_[0]; rfy=ref_kf->intr_[1]; rcx=ref_kf->intr_[2]; rcy=ref_kf->intr_[3]; }
                Eigen::Matrix3d Kn, Kr;
                Kn << fx,0,cx, 0,fy,cy, 0,0,1;
                Kr << rfx,0,rcx, 0,rfy,rcy, 0,0,1;
                Eigen::Matrix3d KRKi = Kn * (w2c.rotationMatrix() * poses_c2w[ref_fi].rotationMatrix().transpose()) * Kr.inverse();
                Eigen::Matrix2d Rplane = KRKi.topLeftCorner<2,2>();
                double pz = Xc.z(), iz = 1.0/pz;

                bool near_ok = true;
                Eigen::Matrix<double,6,1> H_pm_obs = Eigen::Matrix<double,6,1>::Zero();
                for (int k=0; k<8; ++k) {
                    Eigen::Vector2d pp = Rplane * Eigen::Vector2d(photo_pat[k][0], photo_pat[k][1]);
                    double u_center = fx*Xc.x()*iz + cx;
                    double v_center = fy*Xc.y()*iz + cy;
                    double up = u_center + pp.x();
                    double vp = v_center + pp.y();
                    int ui=(int)up, vi=(int)vp;
                    if (ui<0||ui>=fm->second.W-1||vi<0||vi>=fm->second.H-1) { near_ok=false; break; }

                    float I_near = fm->second.I.ptr<float>(vi)[ui];
                    float dx = fm->second.dx.ptr<float>(vi)[ui];
                    float dy = fm->second.dy.ptr<float>(vi)[ui];
                    if (!std::isfinite(dx)||!std::isfinite(dy)) { near_ok=false; break; }

                    double res = (double)(I_near - I_ref_pat[k]) - (b_frames[fi] - b_frames[ref_fi]);
                    double hub = (std::abs(res)>photo_huber) ? photo_huber/std::abs(res) : 1.0;
                    // DSO variance weight: downweight high-gradient observations
                    double var_w = std::sqrt(2500.0 / (2500.0 + dx*dx + dy*dy));
                    hub *= var_w;
                    total_err += hub*res*res*(2.0-hub);
                    total_obs++;
                    if (iter==0) { init_rmse += res*res; init_cnt++; }

                    double px = Xc.x(), py = Xc.y();
                    Eigen::Matrix<double,2,6> Jg;
                    Jg << fx*iz, 0, -fx*px*iz*iz, -fx*px*py*iz*iz, fx*(1+px*px*iz*iz), -fx*py*iz,
                          0, fy*iz, -fy*py*iz*iz, -fy*(1+py*py*iz*iz), fy*px*py*iz*iz, fy*px*iz;
                    Eigen::Matrix<double,1,6> Jp = dx*Jg.row(0) + dy*Jg.row(1);

                    int ps = fi * DS;   // target frame
                    int rs = ref_fi * DS; // ref frame
                    H_pp.block<6,6>(ps, ps) += hub * Jp.transpose() * Jp;
                    b_p.segment<6>(ps) -= hub * Jp.transpose() * res;
                    H_pp(ps+6, ps+6) += hub;       // b_target Hessian
                    b_p(ps+6) += hub * res;         // b_target gradient (-J^T*r, J_b=-1 → +r)
                    H_pp(rs+6, rs+6) += hub;       // b_ref Hessian  
                    b_p(rs+6) -= hub * res;         // b_ref gradient (-J^T*r, J_b=+1 → -r)
                    // DSO-style 1DOF inverse depth Jacobian
                    // ∂X_host/∂idepth = -z² * K⁻¹[u,v,1] where z=1/idepth
                    // Use host frame intrinsics (fallback for merged points)
                    unsigned long hfid = (bp.ref_frame_id != (unsigned long)-1) ? bp.ref_frame_id : bp.obs[0].frame_id;
                    auto hkf = scene_->getKeyframe(hfid);
                    double hfx=hkf->intr_[0], hfy=hkf->intr_[1], hcx=hkf->intr_[2], hcy=hkf->intr_[3];
                    double idepth = idepth_i[pi];
                    double hz = 1.0 / idepth;
                    double didz = -hz * hz;  // d(1/idepth)/d(idepth) = -1/idepth² = -z²
                    double dXh_did_x = didz * (bp.obs[0].u - hcx) / hfx;
                    double dXh_did_y = didz * (bp.obs[0].v - hcy) / hfy;
                    double dXh_did_z = didz;
                    Eigen::Vector3d dXh_did(dXh_did_x, dXh_did_y, dXh_did_z);
                    Sophus::SE3d hw2c; { auto hit=ba_frame_id_to_idx_.find(hfid);
                      if (hit==ba_frame_id_to_idx_.end()) { near_ok=false; break; }
                      hw2c = poses_c2w[hit->second].inverse(); }
                    Eigen::Vector3d dXw_did = hw2c.rotationMatrix().transpose() * dXh_did;
                    Eigen::Vector3d dXt_did = w2c.rotationMatrix() * dXw_did; // same for all pattern points
                    double dud = (fx*iz*dXt_did.x() - fx*Xc.x()*iz*iz*dXt_did.z());
                    double dvd = (fy*iz*dXt_did.y() - fy*Xc.y()*iz*iz*dXt_did.z());
                    double Jpid = dx*dud + dy*dvd;
                    H_dd_i[pi] += hub * Jpid * Jpid;
                    b_d_i[pi] -= hub * Jpid * res;
                    H_pm_obs += hub * Jp.transpose() * Jpid;
                }
                if (!near_ok) continue;
                H_pd_i[pi].push_back({fi, H_pm_obs});
            }
        }
        if (iter==0) init_rmse = (init_cnt>0) ? std::sqrt(init_rmse/init_cnt) : 0;

        for (int i=0; i<dim_pose; ++i) H_pp(i,i) += lambda;
        // Schur complement: DSO-style 1DOF inverse depth elimination
        std::vector<double> idepth_backup(N_points);
        for (int pi=0; pi<N_points; ++pi) idepth_backup[pi] = idepth_i[pi];
        for (int pi=0; pi<N_points; ++pi) {
            if (H_pd_i[pi].empty()) continue;
            if (H_dd_i[pi] < 100.0) continue;  // DSO: setting_minIdepthH_act = 100
            double pt_lambda = lambda * (1.0 + std::abs(H_dd_i[pi]) * 0.01);
            double H_inv = 1.0 / (H_dd_i[pi] + pt_lambda);
            for (auto& pa : H_pd_i[pi]) {
                int fa = pa.first;
                b_p.segment<6>(fa*DS) -= pa.second * H_inv * b_d_i[pi];
                for (auto& pb : H_pd_i[pi]) {
                    int fb = pb.first;
                    H_pp.block<6,6>(fa*DS, fb*DS) -= pa.second * H_inv * pb.second.transpose();
                }
            }
        }
        for (int i=0; i<N_frames; ++i)
            for (int j=0; j<i; ++j)
                H_pp.block<7,7>(i*DS, j*DS) = H_pp.block<7,7>(j*DS, i*DS).transpose();

        auto llt = H_pp.ldlt();
        Eigen::VectorXd dx = llt.solve(b_p);
        if (dx.hasNaN() || llt.info() != Eigen::Success) { lambda *= 10; continue; }

        std::vector<Sophus::SE3d> poses_new = poses_c2w;
        std::vector<double> b_new = b_frames;
        for (int i=0; i<N_frames; ++i) {
            Sophus::SE3d w2c = poses_c2w[i].inverse();
            w2c = Sophus::SE3d::exp(dx.segment<6>(i*DS)) * w2c;
            poses_new[i] = w2c.inverse();
            b_new[i] = b_frames[i] + dx(i*DS+6);
        }

        // Back-substitute: idepth update, then recompute X_world
        std::vector<Eigen::Vector3d> X_new(N_points);
        for (int pi=0; pi<N_points; ++pi) {
            if (H_pd_i[pi].empty()) { X_new[pi] = ba_points_[pi].X_world; continue; }
            unsigned long hfid2 = ba_points_[pi].ref_frame_id;
            if (hfid2 == (unsigned long)-1 && !ba_points_[pi].obs.empty()) hfid2 = ba_points_[pi].obs[0].frame_id;
            auto hit = ba_frame_id_to_idx_.find(hfid2);
            if (hit == ba_frame_id_to_idx_.end()) { X_new[pi]=ba_points_[pi].X_world; continue; }
            auto hkf = scene_->getKeyframe(hfid2);
            if (!hkf) { X_new[pi]=ba_points_[pi].X_world; continue; }
            double hfx=hkf->intr_[0],hfy=hkf->intr_[1],hcx=hkf->intr_[2],hcy=hkf->intr_[3];
            double pt_lambda = lambda * (1.0 + std::abs(H_dd_i[pi]) * 0.01);
            double H_inv = 1.0 / (H_dd_i[pi] + pt_lambda);
            double d_id = H_inv * b_d_i[pi];
            for (auto& pa : H_pd_i[pi])
                if (pa.first >= 0)
                    d_id -= H_inv * pa.second.dot(dx.segment<6>(pa.first*DS));
            double idepth_new = idepth_backup[pi] + d_id;
            if (idepth_new < 0.001) idepth_new = 0.001;
            idepth_i[pi] = idepth_new;
            // Recompute X_world
            Eigen::Vector3d Xh_norm((ba_points_[pi].obs[0].u - hcx)/hfx, (ba_points_[pi].obs[0].v - hcy)/hfy, 1.0);
            X_new[pi] = poses_c2w[hit->second].rotationMatrix() * (Xh_norm / idepth_new) + poses_c2w[hit->second].translation();
        }

        if (dx.norm() < 1e-6) { poses_c2w = poses_new; for (int pi=0; pi<N_points; ++pi) ba_points_[pi].X_world = X_new[pi]; break; }

        // Check new energy
        double new_err = 0;
        for (int pi=0; pi<N_points; ++pi) {
            const BAPoint& bp = ba_points_[pi];
            if (bp.X_world.squaredNorm() < 1e-10 || bp.obs.size() < 2) continue;
            auto rit=ba_frame_id_to_idx_.find(bp.obs[0].frame_id);
            if (rit==ba_frame_id_to_idx_.end()) continue;
            auto rf2=fimgs.find(rit->second);
            if (rf2==fimgs.end()) continue;
            float ru2=bp.obs[0].u, rv2=bp.obs[0].v;
            int ri2=(int)ru2, rj2=(int)rv2;
            if (ri2<0||ri2>=rf2->second.W-1||rj2<0||rj2>=rf2->second.H-1) continue;
            float Ir2 = 0.f;
            { const int pr2=1; int npx=(2*pr2+1)*(2*pr2+1);
              for (int dy=-pr2; dy<=pr2; ++dy) {
                  const float* rr2=rf2->second.I.ptr<float>(rj2+dy);
                  for (int dx=-pr2; dx<=pr2; ++dx) Ir2 += rr2[ri2+dx];
              }
              Ir2 /= npx; }
            for (int oi=1; oi<(int)bp.obs.size(); ++oi) {
                auto it2=ba_frame_id_to_idx_.find(bp.obs[oi].frame_id);
                if (it2==ba_frame_id_to_idx_.end()) continue;
                auto fm2=fimgs.find(it2->second);
                if (fm2==fimgs.end()) continue;
                auto kf2=scene_->getKeyframe(bp.obs[oi].frame_id);
                if (!kf2) continue;
                double fx2=kf2->intr_[0],fy2=kf2->intr_[1],cx2=kf2->intr_[2],cy2=kf2->intr_[3];
                Sophus::SE3d w2c2=poses_new[it2->second].inverse();
                Eigen::Vector3d Xc2=w2c2.rotationMatrix()*X_new[pi]+w2c2.translation();
                if (Xc2.z()<=1e-6) continue;
                double z2=Xc2.z(), iz2=1.0/z2;
                double u2=fx2*Xc2.x()*iz2+cx2, v2=fy2*Xc2.y()*iz2+cy2;
                if (u2<1||u2>fm2->second.W-2||v2<1||v2>fm2->second.H-2) continue;
                int ui2=(int)u2, vi2=(int)v2;
                const float* n02=fm2->second.I.ptr<float>(vi2);
                const float* n12=fm2->second.I.ptr<float>(vi2+1);
                float In2 = 0.f;
                { const int pr2=1; int npx=(2*pr2+1)*(2*pr2+1);
                  for (int dy=-pr2; dy<=pr2; ++dy) {
                      const float* nn2=fm2->second.I.ptr<float>(vi2+dy);
                      for (int dx=-pr2; dx<=pr2; ++dx) In2 += nn2[ui2+dx];
                  }
                  In2 /= npx; }
                double rs2=(double)(In2-Ir2);
                double hb2=(std::abs(rs2)>photo_huber)?photo_huber/std::abs(rs2):1.0;
                new_err += hb2*rs2*rs2*(2.0-hb2);
            }
        }

        if (new_err < total_err) {
            poses_c2w = poses_new;
            b_frames = b_new;
            for (int pi=0; pi<N_points; ++pi) ba_points_[pi].X_world = X_new[pi];
            lambda *= 0.5;
        } else {
            for (int pi=0; pi<N_points; ++pi) idepth_i[pi] = idepth_backup[pi];
            lambda *= 10;
        }
    }

    // Write back to GS keyframes
    for (int i = 0; i < N_frames; ++i) {
        auto kf = scene_->getKeyframe(ba_frame_fids_[i]);
        if (kf) { Sophus::SE3d w2c = poses_c2w[i].inverse(); kf->setPose(w2c.unit_quaternion(), w2c.translation()); }
    }
    // Write back to DSO FrozenFrameHessians
    {
        boost::unique_lock<boost::mutex> fl(pSLAM_->frozenMapMutex);
        for (int i = 0; i < N_frames; ++i)
            for (dso::FrozenFrameHessian* ffh : pSLAM_->frameHessiansFrozen)
                if (ffh->incomingID == (int)ba_frame_fids_[i]) { ffh->camToWorld = poses_c2w[i]; break; }
    }

    // Compute final photometric RMSE
    double final_rmse = 0; int fc = 0;
    for (int pi=0; pi<N_points; ++pi) {
        const BAPoint& bp = ba_points_[pi];
        if (bp.X_world.squaredNorm() < 1e-10 || bp.obs.size() < 2) continue;
        auto rit=ba_frame_id_to_idx_.find(bp.obs[0].frame_id);
        if (rit==ba_frame_id_to_idx_.end()) continue;
        auto rf=fimgs.find(rit->second);
        if (rf==fimgs.end()) continue;
        float ru=bp.obs[0].u, rv=bp.obs[0].v;
        int ri=(int)ru, rj=(int)rv;
        if (ri<0||ri>=rf->second.W-1||rj<0||rj>=rf->second.H-1) continue;
        float Ir = 0.f;
        { const int pr2=1; int npx=(2*pr2+1)*(2*pr2+1);
          for (int dy=-pr2; dy<=pr2; ++dy) {
              const float* rpr = rf->second.I.ptr<float>(rj+dy);
              for (int dx=-pr2; dx<=pr2; ++dx) Ir += rpr[ri+dx];
          }
          Ir /= npx; }
        for (int oi=1; oi<(int)bp.obs.size(); ++oi) {
            auto it=ba_frame_id_to_idx_.find(bp.obs[oi].frame_id);
            if (it==ba_frame_id_to_idx_.end()) continue;
            auto fm=fimgs.find(it->second);
            if (fm==fimgs.end()) continue;
            auto kf=scene_->getKeyframe(bp.obs[oi].frame_id);
            if (!kf) continue;
            double fx=kf->intr_[0],fy=kf->intr_[1],cx=kf->intr_[2],cy=kf->intr_[3];
            Sophus::SE3d w2c=poses_c2w[it->second].inverse();
            Eigen::Vector3d Xc=w2c.rotationMatrix()*bp.X_world+w2c.translation();
            if (Xc.z()<=1e-6) continue;
            double z=Xc.z(), iz=1.0/z;
            double u=fx*Xc.x()*iz+cx, v=fy*Xc.y()*iz+cy;
            if (u<1||u>fm->second.W-2||v<1||v>fm->second.H-2) continue;
            int ui=(int)u, vi=(int)v;
            float In = 0.f;
            { const int pr2=1; int npx=(2*pr2+1)*(2*pr2+1);
              for (int dy=-pr2; dy<=pr2; ++dy) {
                  const float* nnr=fm->second.I.ptr<float>(vi+dy);
                  for (int dx=-pr2; dx<=pr2; ++dx) In += nnr[ui+dx];
              }
              In /= npx; }
            double r=(double)(In-Ir); fc++;
            final_rmse += r*r;
        }
    }
    final_rmse = (fc>0) ? std::sqrt(final_rmse/fc) : 0;

    std::ofstream blog(result_dir_ / "ba_log.txt", std::ios::app);
    blog << getIteration() << " frames=" << N_frames << " points=" << N_points
         << " merged=" << n_merged << " solo=" << n_solo << " expanded_obs=" << expanded
         << " init_rmse=" << init_rmse << " final_rmse=" << final_rmse;
    // GT comparison via relative poses (coordinate-invariant)
    {
        static std::vector<Eigen::Matrix4d> gt_traj;
        static bool gt_loaded = false;
        if (!gt_loaded) {
            for (auto& p : std::vector<std::string>{
                (result_dir_ / "../../../../dataset/Replica/room0/traj.txt").string(),
                (result_dir_ / "../../../dataset/Replica/room0/traj.txt").string(),
                "../../dataset/Replica/room0/traj.txt"}) {
                std::ifstream f(p);
                if (f.is_open()) {
                    std::string line;
                    while (std::getline(f, line)) {
                        std::istringstream ss(line);
                        Eigen::Matrix4d m;
                        for (int r=0;r<4;++r) for (int c=0;c<4;++c) ss >> m(r,c);
                        gt_traj.push_back(m);
                    }
                    gt_loaded = true; break;
                }
            }
        }

        double dso_err_sum = 0, ba_err_sum = 0;
        double dso_trans_sum = 0, ba_trans_sum = 0;
        int pair_cnt = 0;
        if (gt_loaded) {
            for (int pi = 0; pi < N_points; ++pi) {
                if (ba_points_[pi].obs.size() < 2) continue;
                unsigned long fid0 = ba_points_[pi].obs[0].frame_id;
                if (fid0 >= gt_traj.size()) continue;
                auto i0 = ba_frame_id_to_idx_.find(fid0);
                if (i0 == ba_frame_id_to_idx_.end()) continue;
                for (int oi = 1; oi < (int)ba_points_[pi].obs.size(); ++oi) {
                    unsigned long fid1 = ba_points_[pi].obs[oi].frame_id;
                    if (fid1 >= gt_traj.size()) continue;
                    auto i1 = ba_frame_id_to_idx_.find(fid1);
                    if (i1 == ba_frame_id_to_idx_.end()) continue;

                    // GT: R_ref→near = R_near_c2w^T * R_ref_c2w
                    Eigen::Matrix3d Rgt = gt_traj[fid1].topLeftCorner<3,3>().transpose() * gt_traj[fid0].topLeftCorner<3,3>();
                    // DSO & BA: same formula, c2w matrices
                    Eigen::Matrix3d Rdso = poses_original[i1->second].rotationMatrix().transpose() * poses_original[i0->second].rotationMatrix();
                    Eigen::Matrix3d Rba  = poses_c2w[i1->second].rotationMatrix().transpose() * poses_c2w[i0->second].rotationMatrix();

                    double dso_err=0, ba_err=0, dso_trans=0, ba_trans=0;
                    { Eigen::Matrix3d R=Rgt.transpose()*Rdso; double c=std::max(-1.0,std::min(1.0,(R.trace()-1.0)*0.5)); dso_err=std::acos(c)*180.0/M_PI; }
                    { Eigen::Matrix3d R=Rgt.transpose()*Rba;  double c=std::max(-1.0,std::min(1.0,(R.trace()-1.0)*0.5)); ba_err=std::acos(c)*180.0/M_PI; }
                    // Translation direction error (scale-invariant)
                    { Eigen::Vector3d t_gt_rel = gt_traj[fid1].topLeftCorner<3,3>().transpose() * (gt_traj[fid0].topRightCorner<3,1>() - gt_traj[fid1].topRightCorner<3,1>());
                      Eigen::Vector3d t_dso_rel = poses_original[i1->second].rotationMatrix().transpose() * (poses_original[i0->second].translation() - poses_original[i1->second].translation());
                      Eigen::Vector3d t_ba_rel  = poses_c2w[i1->second].rotationMatrix().transpose() * (poses_c2w[i0->second].translation() - poses_c2w[i1->second].translation());
                      auto trans_angle = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b)->double {
                          double na=a.norm(), nb=b.norm();
                          if (na<1e-10||nb<1e-10) return 0;
                          double cs=a.dot(b)/(na*nb);
                          return std::acos(std::max(-1.0,std::min(1.0,cs)))*180.0/M_PI;
                      };
                      dso_trans = trans_angle(t_gt_rel, t_dso_rel);
                      ba_trans  = trans_angle(t_gt_rel, t_ba_rel); }

                    dso_err_sum += dso_err; ba_err_sum += ba_err; pair_cnt++;
                    dso_trans_sum += dso_trans; ba_trans_sum += ba_trans;
            }
        }
        double dso_avg = pair_cnt>0 ? dso_err_sum/pair_cnt : -1;
        double ba_avg  = pair_cnt>0 ? ba_err_sum/pair_cnt : -1;
        double dso_tavg = pair_cnt>0 ? dso_trans_sum/pair_cnt : -1;
        double ba_tavg  = pair_cnt>0 ? ba_trans_sum/pair_cnt : -1;
        blog << " dso_rot=" << dso_avg << " ba_rot=" << ba_avg
             << " dso_trans=" << dso_tavg << " ba_trans=" << ba_tavg
             << " npairs=" << pair_cnt;
        }
    }
    blog << std::endl;
    blog.close();

    ba_points_.clear();
    ba_frame_id_to_idx_.clear();
    ba_frame_fids_.clear();
    ba_accum_iter_ = 0;
}

