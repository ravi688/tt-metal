// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>
#include <unordered_map>

#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_device_setup.hpp"

using TestPhysicalMeshes = tt::tt_fabric::fabric_tests::TestPhysicalMeshes;
using TestFabricFixture = tt::tt_fabric::fabric_tests::TestFabricFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;

class TestContext {
public:
    void init();
    void handle_test_config();
    void open_devices(tt::tt_metal::FabricConfig fabric_config);
    void handle_traffic_config(TestTrafficConfig traffic_config);
    void close_devices();

private:
    TestPhysicalMeshes physical_meshes_;
    TestFabricFixture fixture_;
    std::vector<chip_id_t> available_physical_chip_ids_;
    std::unordered_map<chip_id_t, TestDevice> test_devices_;
};

void TestContext::init() {
    this->physical_meshes_.setup_physical_meshes();
    this->fixture_.setup_devices();
    this->available_physical_chip_ids_ = this->fixture_.get_available_chip_ids();
    this->physical_meshes_.print_meshes();
}

void TestContext::open_devices(tt::tt_metal::FabricConfig fabric_config) {
    this->fixture_.open_devices(fabric_config);
    for (const auto& chip_id : this->available_physical_chip_ids_) {
        auto* device_handle = this->fixture_.get_device_handle(chip_id);
        this->test_devices_.emplace_back(chip_id, device_handle);
    }
}

void TestContext::handle_traffic_config(TestTrafficConfig traffic_config) {
    const auto src_chip_id = traffic_config.src_phys_chip_id;

    auto test_device_it = this->test_devices_.find(src_chip_id);
    if (test_device_it == this->test_devices_.end()) {
        tt::log_fatal(tt::LogTest, "Unknown src physical chip id: {}", src_chip_id);
        throw std::runtime_error("Unexpected physical chip id for test device lookup");
    }
    auto& test_device = it->second;

    // check if dst or hops are specified -> error if none specified
    std::vector<chip_id_t> dst_phys_chip_ids;
    std::unordered_map<RoutingDirection, uint32_t> num_hops;
    if (!traffic_config.dst_phys_chip_ids.has_value() && !traffic_config.num_hops.has_value()) {
        tt::log_fatal(tt::LogTest, "One of dst phys chip id or num hops should be present, none specified");
        throw std::runtime_error("Unexpected test config");
    } else if (traffic_config.dst_phys_chip_ids.has_value() && traffic_config.num_hops.has_value()) {
        tt::log_fatal(tt::LogTest, "Only one of dst phys chip id or num hops should be present, both specified");
        throw std::runtime_error("Unexpected test config");
    } else if (traffic_config.dst_phys_chip_ids.has_value()) {
        dst_phys_chip_ids = traffic_config.dst_phys_chip_ids.value();
        // TODO: fetch the number of hops b/w chips from the mesh
    } else if (traffic_config.num_hops.has_value()) {
        // TODO: fetch the dst chip id based on number of hops
    }

    TestTrafficSenderConfig traffic_sender_config;
    traffic_sender_config.data_config = traffic_config.data_config;

    // create sender config for the src device

    // create receiver config

    // allocate sender

    // allocate receiver

    // additional handling - if mcast mode then add receivers for every chip in the route

    // if bidirectional - add sender for every receiver
}

void TestContext::close_devices() { this->fixture_.close_devices(); }

// TODO: method to get random chip send type
// TODO: method to get random noc send type
// TODO: method to get random hops (based on mode - 1D/2D)
// TODO: method to get random dest chip (based on mode - 1D/2D)

void setup_fabric(
    tt::tt_fabric::fabric_tests::TestFabricSetup fabric_setup_config, std::vector<TestDevice>& test_devices) {}

void setup_traffic_config(TestTrafficDataConfig data_config, chip_id_t src_phys_chip_id);

void setup_traffic_config(TestTrafficDataConfig data_config, chip_id_t src_phys_chip_id);

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    tt::tt_fabric::fabric_tests::parse_config(input_args);

    TestContext test_context;
    test_context.init();

    test_context.open_devices(tt::tt_metal::FabricConfig::FABRIC_1D);

    // fabric setup
    // setup_fabric()

    // all-to-all mode

    // workers setup

    // launch programs

    //

    test_context.close_devices();

    return 0;
}
