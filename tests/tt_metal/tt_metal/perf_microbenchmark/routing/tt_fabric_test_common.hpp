// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace fabric_tests {

struct TestFabricFixture {
    tt::ARCH arch_;
    std::vector<chip_id_t> physical_chip_ids_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> devices_map_;
    bool slow_dispatch_;

    void setup_devices() {
        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch_) {
            tt::log_info(tt::LogTest, "Running fabric tests with slow dispatch");
        } else {
            tt::log_info(tt::LogTest, "Running fabric tests with fast dispatch");
        }

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        size_t chip_id_offset = 0;
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG) {
            chip_id_offset = 4;
        }
        const auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        this->physical_chip_ids_.resize(num_devices);
        std::iota(this->physical_chip_ids_.begin(), this->physical_chip_ids_.end(), chip_id_offset);

        // todo: check if and where we need to configure the routing tables
    }

    void open_devices(tt::tt_metal::FabricConfig fabric_config) {
        tt::tt_metal::detail::InitializeFabricConfig(fabric_config);
        this->devices_map_ = tt::tt_metal::detail::CreateDevices(this->physical_chip_ids_);
    }

    std::vector<chip_id_t> get_available_chip_ids() { return this->physical_chip_ids_; }

    tt::tt_metal::IDevice* get_device_handle(chip_id_t physical_chip_id) {
        if (this->devices_map_.find(physical_chip_id) == this->devices_map_.end()) {
            tt::log_fatal(tt::LogTest, "Unknown physical chip id: {}", physical_chip_id);
            throw std::runtime_error("Unexpected physical chip id for device handle lookup");
        }
        return this->devices_map_.at(physical_chip_id);
    }

    void run_program_non_blocking(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::LaunchProgram(device, program, false);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueProgram(cq, program, false);
        }
    }

    void wait_for_program_done(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            // Wait for the program to finish
            tt::tt_metal::detail::WaitProgramDone(device, program);
        } else {
            // Wait for all programs on cq to finish
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::Finish(cq);
        }
    }

    void close_devices() {
        tt::tt_metal::detail::CloseDevices(this->devices_map_);
        tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
    }
};

struct TestPhysicalMeshes {
    static constexpr uint8_t NUM_DIMS = 2;
    static constexpr uint8_t ROW_IDX = 0;
    static constexpr uint8_t COL_IDX = 1;

public:
    void setup_physical_meshes();
    void print_meshes();
    std::vector<chip_id_t> get_other_chips_on_same_row(chip_id_t physical_chip_id);
    std::vector<chip_id_t> get_other_chips_on_same_col(chip_id_t physical_chip_id);

private:
    // map of 2D meshes (mesh id is logical, but chip ids are physical)
    std::unordered_map<mesh_id_t, std::vector<std::vector<chip_id_t>>> physical_chip_view_;
    // map of chip coords per mesh (mesh id is logical, chip ids are physical)
    std::unordered_map<mesh_id_t, std::unordered_map<chip_id_t, std::pair<uint32_t, uint32_t>>> physical_chip_coords_;
    // map of number of rows/columns per mesh
    std::unordered_map<mesh_id_t, std::array<uint32_t, NUM_DIMS>> physical_mesh_dims_;
    // map of size along each dim
    std::unordered_map<mesh_id_t, std::array<uint32_t, NUM_DIMS>> physical_mesh_dims_size_;

    tt::tt_fabric::ControlPlane* control_plane_;

    void validate_mesh_id(mesh_id_t mesh_id);
    void validate_physical_chip_id(mesh_id_t mesh_id, chip_id_t physical_chip_id);
    void set_mesh_dims_and_size(mesh_id_t mesh_id, std:: : array<uint32_t, NUM_DIMS> dims);
    void generate_physical_chip_view(mesh_id_t mesh_id);
    std::pair<uint32_t, uint32_t> get_chip_coords_from_physical_id(mesh_id_t mesh_id, chip_id_t physical_chip_id);
    std::pair<uint32_t, uint32_t> get_chip_coords_from_logical_id(mesh_id_t mesh_id, chip_id_t logical_chip_id);
};

inline void TestPhysicalMeshes::validate_mesh_id(mesh_id_t mesh_id) {
    // TODO: take in a string param for debug/log strings
    if (this->physical_chip_coords_.find(mesh_id) == this->physical_chip_coords_.end()) {
        tt::log_fatal(tt::LogTest, "Unknown mesh id: {}", mesh_id);
        throw std::runtime_error("Unexpected mesh_id");
    }
}

inline void TestPhysicalMeshes::validate_physical_chip_id(mesh_id_t mesh_id, chip_id_t physical_chip_id) {
    if (this->physical_chip_coords_[mesh_id].find(physical_chip_id) == this->physical_chip_coords_[mesh_id].end()) {
        tt::log_fatal(tt::LogTest, "Unknown chip id: {} for mesh id: {}", physical_chip_id, mesh_id);
        throw std::runtime_error("Unexpected chip_id");
    }
}

inline void TestPhysicalMeshes::set_mesh_dims_and_size(mesh_id_t mesh_id, std:: : array<uint32_t, NUM_DIMS> dims) {
    this->physical_mesh_dims_[mesh_id][ROW_IDX] = dims[ROW_IDX];
    this->physical_mesh_dims_[mesh_id][COL_IDX] = dims[COL_IDX];

    this->physical_mesh_dims_size_[mesh_id][ROW_IDX] = dims[COL_IDX];
    this->physical_mesh_dims_size_[mesh_id][COL_IDX] = dims[ROW_IDX];
}

inline void TestPhysicalMeshes::generate_physical_chip_view(mesh_id_t mesh_id) {
    // for now assume its a 2D mesh
    const auto& mesh_shape = this->control_plane_ptr_->get_physical_mesh_shape(mesh_id);
    const auto num_rows = mesh_shape[0];
    const auto num_cols = mesh_shape[1];

    this->set_mesh_dims_and_size(mesh_id, {num_rows, num_cols});

    this->physical_chip_view_[mesh_id].resize(num_rows, std::vector<chip_id_t>(num_cols));
    chip_id_t logical_chip_id = 0;
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = 0; j < num_cols; j++) {
            FabricNodeId fabric_node_id{mesh_id, logical_chip_id};
            chip_id_t physical_chip_id =
                this->control_plane_ptr_->get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            this->physical_chip_view_[mesh_id][i][j] = physical_chip_id;
            this->physical_chip_coords_[mesh_id][physical_chip_id] = std::make_pair(j, i);
            logical_chip_id++;
        }
    }
}

inline std::pair<uint32_t, uint32_t> TestPhysicalMeshes::get_chip_coords_from_physical_id(
    mesh_id_t mesh_id, chip_id_t physical_chip_id) {
    this->validate_mesh_id(mesh_id);
    this->validate_physical_chip_id(mesh_id, physical_chip_id);
    return this->physical_chip_coords_[mesh_id][physical_chip_id];
}

inline std::pair<uint32_t, uint32_t> TestPhysicalMeshes::get_chip_coords_from_logical_id(
    mesh_id_t mesh_id, chip_id_t logical_chip_id) {
    FabricNodeId fabric_node_id{mesh_id, logical_chip_id};
    const auto physical_chip_id = this->control_plane_ptr_->get_physical_chip_id_from_fabric_node_id(fabric_node_id);
    return this->get_chip_coords_from_physical_id(mesh_id, physical_chip_id);
}

inline void TestPhysicalMeshes::setup_physical_meshes() {
    this->control_plane_ptr_ = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    const auto user_meshes = this->control_plane_ptr_->get_user_physical_mesh_ids();

    for (const auto& mesh_id : user_meshes) {
        this->generate_physical_chip_view(mesh_id);
    }
}

inline void TestPhysicalMeshes::print_meshes() {
    tt::log_info(tt::LogTest, "Printing physical meshes, (total: {})", this->physical_chip_view_.size());
    for (const auto& [mesh_id, mesh] : this->physical_chip_view_) {
        tt::log_info(tt::LogTest, "Mesh id: {}", mesh_id);
        for (const auto& row : mesh) {
            tt::log_info(tt::LogTest, "{}", row);
        }
    }
}

inline std::vector<chip_id_t> TestPhysicalMeshes::get_other_chips_on_same_row(chip_id_t physical_chip_id) {
    const auto& fabric_node_id = this->control_plane_->get_fabric_node_id_from_physical_chip_id(physical_chip_id);
    const auto mesh_id = fabric_node_id.mesh_id;

    this->validate_mesh_id(mesh_id);
    this->validate_physical_chip_id(mesh_id, physical_chip_id);

    const auto [chip_row, chip_col] = this->get_chip_location_from_physical_id(mesh_id, physical_chip_id);
    std::vector<chip_id_t> chips = this->physical_chip_view_[mesh_id][chip_row];
    chips.erase(std::remove(chips.begin(), chips.end(), physical_chip_id), chips.end());

    return chips;
}

inline std::vector<chip_id_t> TestPhysicalMeshes::get_other_chips_on_same_col(chip_id_t physical_chip_id) {
    const auto& fabric_node_id = this->control_plane_->get_fabric_node_id_from_physical_chip_id(physical_chip_id);
    const auto mesh_id = fabric_node_id.mesh_id;

    this->validate_mesh_id(mesh_id);
    this->validate_physical_chip_id(mesh_id, physical_chip_id);

    const auto [chip_row, chip_col] = this->get_chip_location_from_physical_id(mesh_id, physical_chip_id);
    std::vector<chip_id_t> chips;
    for (uint32_t i = 0; i < this->physical_mesh_dims_[mesh_id][ROW_IDX]; i++) {
        chips.push_back(this->physical_chip_view_[mesh_id][i][chip_col]);
    }
    chips.erase(std::remove(chips.begin(), chips.end(), physical_chip_id), chips.end());

    return chips;
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
