// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/mesh_graph.hpp>

namespace tt::tt_fabric {
namespace fabric_tests {

// create memory maps

// this also has to consider the memory map which has addresses for synchronization etc

struct TestTrafficDataConfig {
    ChipSendType chip_send_type;
    NocSendType noc_send_type;
    size_t num_packets;
    size_t payload_size_bytes;
};

struct TestTrafficConfig {
    TestTrafficDataConfig data_config;
    chip_id_t src_phys_chip_id;
    std::optional<std::vector<chip_id_t>> dst_phys_chip_ids;
    std::optional<std::unordered_map<RoutingDirection, uint32_t>> num_hops;
    std::optional<CoreCoord> src_logical_core;
    std::optional<CoreCoord> dst_logical_core;
    std::optional<std::string_view> sender_kernel_src;
    std::optional<std::string_view> receiver_kernel_src;
    // TODO: add later
    // mode - BW, latency etc
};

struct TestTrafficSenderConfig {
    TestTrafficDataConfig data_config;
    chip_id_t dst_phys_chip_id;
    std::unordered_map<RoutingDirection, uint32_t> num_hops;
    size_t target_address;
    uint32_t receiver_noc_xy_encoding;
};

struct TestTrafficReceiverConfig {
    TestTrafficDataConfig data_config;
    uint32_t sender_id;
    size_t target_address;
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
