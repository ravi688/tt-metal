// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <string_view>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/mesh_graph.hpp>

#include "impl/context/metal_context.hpp"

#include "tt_fabric_test_worker_setup.hpp"

namespace tt::tt_fabric {
namespace fabric_tests {

// forward declaration
struct TestDevice;

struct TestFabricBuilderConfigs {
private:
    // collection of all the builder configs we may need
    // TODO: maybe supply a hash function to avoid nesting
    std::unordered_map<
        Topology,
        std::unordered_map<
            tt::tt_metal::FabricConfig,
            std::unordered_map<bool, std::unique_ptr<FabricEriscDatamoverConfig>>>>
        builder_configs_{};

    TestFabricBuilderConfigs() {}

public:
    TestFabricBuilderConfigs(const TestFabricBuilderConfigs&) = delete;
    TestFabricBuilderConfigs& operator=(const TestFabricBuilderConfigs&) = delete;

    static TestFabricBuilderConfigs& get_instance() {
        static TestFabricBuilderConfigs instance;
        return instance;
    }

    size_t get_packet_header_size_bytes(Topology topology, tt::tt_metal::FabricConfig config) const;
    size_t get_max_payload_size_bytes(Topology topology) const;
    FabricEriscDatamoverConfig& get_builder_config(
        Topology topology, tt::tt_metal::FabricConfig config, bool is_dateline);
};

struct TestFabricBuilder {
private:
    // hack to cache the builder
    // TODO: find a better way?
    std::vector<FabricEriscDatamoverBuilder> builder_;

public:
    void build();
    void connect_to_downstream_builder();
};

// per router config
struct TestFabricRouter {
private:
    chan_id_t my_chan_;
    chip_id_t remote_chip_id_;
    std::vector<chan_id_t> downstream_router_chans_;
    bool is_on_dateline_connection_;
    bool is_loopback_router_;

    TestFabricBuilder fabric_builder_;

    // my_direction_ / eth_direction_;
    // my_link_idx_;
    TestDevice* test_device_context_;

public:
    TestFabricRouter(chan_id_t chan, chip_id_t remote_chip_id);
    void add_downstream_router(chan_id_t downstream_router_chan);
    void set_dateline_flag();
    void set_loopback_flag();
    void set_remote_chip();
    void connect_to_downstream_routers();
};

struct TestDeviceFabricRouters {
private:
    std::unordered_map<RoutingDirection, std::vector<chan_id_t>> direction_and_chans_{};
    // chip_neighbors_in_directions_;
    std::unordered_map<chan_id_t, TestFabricRouter> routers_{};
    chan_id_t master_router_chan_{};
    std::optional<Topology> topology_;
    TestDevice* test_device_context_;

public:
    TestDeviceFabricRouters();
    void set_topology(Topology topology);
    void add_active_router();
    std::vector<RoutingDirection> get_downstream_directions(RoutingDirection direction, Topology topology);
    void connect_routing_planes();
    void setup_topology();
    void build_fabric();
    void wait_for_router_sync();
    void notify_routers();
    void terminate_routers();
};

// for now keep the memory map same for both senders and receivers
struct TestWorkerMemoryMap {
    uint32_t worker_usable_address;
};

struct TestWorker {
public:
    TestWorker(CoreCoord logical_core, TestDevice* test_device_context);
    TestWorker(CoreCoord logical_core, TestDevice* test_device_context, const std::string& kernel_src);
    void set_kernel_src();
    uint32_t get_worker_id();
    uint32_t get_worker_noc_encoding();
    void create_kernel();
    void collect_results();
    void validate_results();
    void dump_results();

private:
    CoreCoord logical_core_;
    uint32_t noc_xy_encoding_;
    std::string_view kernel_src_;
    TestDevice* test_device_context_;
};

struct TestSender : TestWorker {
public:
    TestSender(CoreCoord logical_core, TestDevice* test_device_context);
    TestSender(CoreCoord logical_core, TestDevice* test_device_context, const std::string& kernel_src);
    void add_receiver_config();
    void connect_to_fabric_router();

private:
    TestWorkerMemoryMap memory_map_;
    // for now assume that a worker can handle multiple test configs in the same direction
    std::vector<TestSenderConfig> sender_configs_;
};

struct TestReceiverAllocator {
public:
    TestReceiverAllocator(uint32_t start_address, uint32_t end_address, uint32_t chunk_size);
    uint32_t allocate();

private:
    uint32_t start_address_;
    uint32_t end_address_;
    uint32_t total_size_;
    uint32_t chunk_size_;
    std::vector<uint32_t> available_addresses_;
};

struct TestReceiver : TestWorker {
public:
    TestReceiver(CoreCoord logical_core, TestDevice* test_device_context, bool is_shared);
    TestReceiver(
        CoreCoord logical_core, TestDevice* test_device_context, bool is_shared, const std::string& kernel_src);
    void add_sender_config();
    bool is_shared_receiver();
    uint32_t allocate_space_for_sender();

private:
    TestWorkerMemoryMap memory_map_;
    TestReceiverAllocator allocator_;
    bool is_shared_;
    std::vector<TestReceiverConfig> receiver_configs_;
};

struct TestDevice {
private:
    tt::tt_metal::IDevice* IDevice_handle_;
    chip_id_t physical_chip_id_;
    mesh_id_t mesh_id_;
    chip_id_t logical_chip_id_;

    tt_metal::Program program_handle_;

    std::vector<CoreCoord> avaialble_worker_logical_cores_;  // TODO: rename to available only?

    // For now instead of moving to a new worker for the receiver every time, just pick one and exhaust it
    // if can no longer allocate, fetch a new core (as per policy)
    std::optional<CoreCoord> current_receiver_logical_core_;

    std::unordered_map<CoreCoord, TestSender> senders_;
    std::unordered_map<CoreCoord, TestReceiver> receivers_;

    // controller?

    TestDeviceFabricRouters fabric_routers_;

    CoreCoord reserve_worker_core();

public:
    TestDevice(tt::tt_metal::IDevice* IDevice_handle);
    tt::tt_metal::IDevice* get_device_handle();
    tt::tt_metal::Program& get_program_handle();
    chip_id_t get_physical_chip_id() const;
    void add_sender(TestSenderConfig config);
    void add_sender(CoreCoord logical_core, TestSenderConfig config);
    void update_sender(CoreCoord logical_core, TestSenderConfig config);

    // update_sender_config -> only update when there is a receiver avaialble

    void allocate_receiver_for_sender(CoreCoord receiver_core, bool share_receiver);
};

/* **********************************
 * TestFabricBuilderConfigs Methods *
 ************************************/
inline size_t TestFabricBuilderConfigs::get_packet_header_size_bytes(
    Topology topology, tt::tt_metal::FabricConfig config) const {
    if (topology == Topology::Mesh) {
        return (config == tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC) ? sizeof(MeshPacketHeader)
                                                                         : sizeof(LowLatencyMeshPacketHeader);
    } else {
        return sizeof(PacketHeader);
    }
}

inline size_t TestFabricBuilderConfigs::get_max_payload_size_bytes(Topology topology) const {
    if (topology == Topology::Mesh) {
        return FabricEriscDatamoverBuilder::default_mesh_packet_payload_size_bytes;
    } else {
        return FabricEriscDatamoverBuilder::default_packet_payload_size_bytes;
    }
}

inline FabricEriscDatamoverConfig& TestFabricBuilderConfigs::get_builder_config(
    Topology topology, tt::tt_metal::FabricConfig config, bool is_dateline) {
    bool found = true;
    const auto topology_it = this->builder_configs_.find(topology);
    if (topology_it == this->builder_configs_.end()) {
        found = false;
    }

    if (found) {
        const auto config_it = this->builder_configs_.at(topology).find(config);
        if (config_it == this->builder_configs_.at(topology).end()) {
            found = false;
        }

        if (found) {
            const auto dateline_it = this->builder_configs_.at(topology).at(config).find(is_dateline);
            if (dateline_it == this->builder_configs_.at(topology).at(config).end()) {
                found = false;
            }
        }
    }

    if (!found) {
        const auto packet_header_size_bytes = this->get_packet_header_size_bytes(topology, config);
        const auto max_payload_size_bytes = this->get_max_payload_size_bytes(topology);
        const auto channel_buffer_size_bytes = packet_header_size_bytes + max_payload_size_bytes;
        this->builder_configs_[topology][config][is_dateline] =
            std::make_unique<FabricEriscDatamoverConfig>(channel_buffer_size_bytes, topology, is_dateline);
    }

    return *this->builder_configs_.at(topology).at(config).at(is_dateline).get();
}

/* **************************
 * TestFabricRouter Methods *
 ****************************/
inline TestFabricRouter::TestFabricRouter(chan_id_t chan, chip_id_t remote_chip_id) {
    this->my_chan_ = chan;
    this->remote_chip_id_ = remote_chip_id;
}

inline void TestFabricRouter::add_downstream_router(chan_id_t downstream_router_chan) {
    this->downstream_router_chans_.push_back(downstream_router_chan);
}

inline void TestFabricRouter::set_dateline_flag() { this->is_on_dateline_connection_ = true; }

inline void TestFabricRouter::set_loopback_flag() {
    this->is_loopback_router_ = true;

    // for loopback mode, set the downstream router chan to be itself
    if (!this->downstream_router_chans_.empty()) {
        tt::log_fatal(
            tt::LogTest,
            "For chip: {}, chan: {}, downstream router is already set to: {}, but tried to set as loopback",
            this->test_device_context_->get_physical_chip_id(),
            this->my_chan_,
            this->downstream_router_chans_);
        throw std::runtime_error("Tried to set loopback when downstream router chan is already set");
    }
    this->downstream_router_chans_ = {this->my_chan_};
}

inline void TestFabricRouter::set_remote_chip() { this->remote_chip_id_ = true; }

/*
    void build(tt::tt_metal::IDevice* IDevice_handle, tt_metal::Program& program_handle) {
        this->builder_ = FabricEriscDatamoverBuilder::build(
            IDevice_handle,
            program_handle,
            fabric_eth_chan_to_logical_core.at(eth_chan),
            physical_chip_id,
            remote_chip_id,
            edm_config,
            true,
            false,
            is_dateline);
        this->builder_.set_wait_for_host_signal(true);
    }
*/
inline void TestFabricRouter::connect_to_downstream_routers() {
    // this->builder_.connect_to_downstream_edm(downstream_router.builder_);
}

/* *********************************
 * TestDeviceFabricRouters Methods *
 ***********************************/
inline TestDeviceFabricRouters::TestDeviceFabricRouters() {
    // TODO: take in the control plane ptr as a part of setup
    // since we may be setting up a custom control plane
    const auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    // TODO update chip neighbors
}

inline void TestDeviceFabricRouters::set_topology(Topology topology) { this->topology_ = topology; }

inline void TestDeviceFabricRouters::add_active_router() {}

inline std::vector<RoutingDirection> TestDeviceFabricRouters::get_downstream_directions(
    RoutingDirection direction, Topology topology) {
    if (topology == Topology::Linear) {
        switch (direction) {
            case RoutingDirection::N: return {RoutingDirection::S};
            case RoutingDirection::S: return {RoutingDirection::N};
            case RoutingDirection::E: return {RoutingDirection::W};
            case RoutingDirection::W: return {RoutingDirection::E};
            default: throw std::runtime_error("Invalid direction for quering forwarding directions");
        }
    } else if (topology == Topology::Mesh) {
        // return all the directions except itself
        std::vector<RoutingDirection> directions = FabricContext::routing_directions;
        directions.erase(std::remove(directions.begin(), directions.end(), direction), directions.end());
        return directions;
    } else {
        // throw error?
    }

    return {};
}

// blanket method/preset mode method
inline void TestDeviceFabricRouters::connect_routing_planes() {
    for (const auto& [direction, router_chans] : this->direction_and_chans_) {
        for (auto routing_plane_id = 0; routing_plane_id < router_chans.size(); routing_plane_id++) {
            const auto channel = router_chans[routing_plane_id];
            if (this->routers_.find(channel) == this->routers_.end()) {
                // no router is active for this channel
                continue;
            }

            const auto& downstream_directions = get_downstream_directions(direction, this->topology_.value());
            for (const auto& downstream_dir : downstream_directions) {
                auto it = this->direction_and_chans_.find(downstream_dir);
                if (it == this->direction_and_chans_.end()) {
                    continue;
                }

                const auto& downstream_channels = it->second;
                if (routing_plane_id >= downstream_channels.size()) {
                    // no downstream channel for this routing plane
                    continue;
                }

                this->routers_.at(channel).add_downstream_router(downstream_channels[routing_plane_id]);
            }
        }
    }
}

inline void TestDeviceFabricRouters::setup_topology() {
    // TODO: add checks for some of the router configs that may have already been setup

    if (!this->topology_.has_value()) {
        return;
    }

    this->connect_routing_planes();
}

inline void TestDeviceFabricRouters::build_fabric() {
    setup_topology();

    // by now we assume that each individual router has been setup

    // first invoke individual builders
    for (const auto& [_, router] : this->routers_) {
        // router.build()
    }

    // connect downstream routers
    for (const auto& [_, router] : this->routers_) {
        // router.connect_to_downstream_routers();
    }

    // create and compile kernels
}

inline void TestDeviceFabricRouters::wait_for_router_sync() {}

inline void TestDeviceFabricRouters::notify_routers() {}

inline void TestDeviceFabricRouters::terminate_routers() {}

/* ********************
 * TestWorker Methods *
 **********************/
inline TestWorker::TestWorker(CoreCoord logical_core, TestDevice* test_device_context) {
    this->logical_core_ = logical_core;
    this->noc_xy_encoding_ = 0; /* set from metal context/soc desc/hal */
    this->test_device_context_ = test_device_context;
}

inline TestWorker::TestWorker(CoreCoord logical_core, TestDevice* test_device_context, const std::string& kernel_src) {
    TestWorker(logical_core, test_device_context);
    this->kernel_src_ = kernel_src;
}

inline void TestWorker::set_kernel_src(const std::string& kernel_src) { this->kernel_src_ = kernel_src; }

inline uint32_t TestWorker::get_worker_id() { /*return some combination of logical core and phys chip id*/ }

inline uint32_t TestWorker::get_worker_noc_encoding() { return this->noc_xy_encoding_; }

inline void TestWorker::create_kernel(
    const std::vector<uint32_t>& ct_args,
    const std::vector<uint32_t>& rt_args,
    const std::vector<std::pair<size_t, size_t>>& addresses_and_size_to_clear) {
    auto kernel_handle = tt::tt_metal::CreateKernel(
        this->test_device_context_->get_program_handle(),
        std::string(this->kernel_src_),
        {this->logical_core_},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
    tt::tt_metal::SetRuntimeArgs(
        this->test_device_context_->get_program_handle(), kernel_handle, this->logical_core_, rt_args);

    for (const auto& [address, num_bytes] : addresses_and_size_to_clear) {
        std::vector<uint32_t> zero_vec((num_bytes / sizeof(uint32_t)), 0);
        tt::tt_metal::detail::WriteToDeviceL1(
            this->test_device_context_->get_device_handle(), this->logical_core_, address, zero_vec);
    }
}

/* ********************
 * TestSender Methods *
 **********************/
inline TestSender::TestSender(CoreCoord logical_core, TestDevice* test_device_context) :
    TestWorker(logical_core, test_device_context) {
    // TODO: init mem map?
}

inline TestSender::TestSender(CoreCoord logical_core, TestDevice* test_device_context, const std::string& kernel_src) {
    TestSender(logical_core, test_device_context);
    this->set_kernel_src(kernel_src);
}

inline void TestSender::add_receiver_config() {}

inline void TestSender::connect_to_fabric_router() {}

/* *******************************
 * TestReceiverAllocator Methods *
 *********************************/
inline TestReceiverAllocator::TestReceiverAllocator(uint32_t start_address, uint32_t end_address, uint32_t chunk_size) {
    // TODO: sanity check on start and end addresses

    this->start_address_ = start_address;
    this->end_address_ = end_address;
    this->total_size_ = end_address - start_address;
    this->chunk_size_ = chunk_size;

    // for now initialize the addresses in a contiguous fashion
    uint32_t next_avaialable_address = this->start_address_;
    while (next_avaialable_address + this->chunk_size_ <= this->end_address_) {
        this->available_addresses_.push_back(next_avaialable_address);
        next_avaialable_address += this->chunk_size_;
    }
}

inline uint32_t TestReceiverAllocator::allocate() {
    // TODO: can have a policy for allocator as well. For now, just allocate the next available
    if (this->available_addresses_.empty()) {
        return 0;
    }

    uint32_t avaialble_address = this->available_addresses_.back();
    this->available_addresses_.pop_back();
    return avaialble_address;
}

/* **********************
 * TestReceiver Methods *
 ************************/
inline TestReceiver::TestReceiver(CoreCoord logical_core, TestDevice* test_device_context, bool is_shared) :
    TestWorker(logical_core, test_device_context) {
    this->is_shared_ = true;
    // TODO: init mem map?
    // TODO: get these from the mem map
    uint32_t start_address = 0;
    uint32_t end_address = 0;
    uint32_t chunk_size = 0;  // TODO: get this from the config/settings
    this->allocator_ = TestReceiverAllocator(start_address, end_address, chunk_size);
}

inline TestReceiver::TestReceiver(
    CoreCoord logical_core, TestDevice* test_device_context, bool is_shared, const std::string& kernel_src) {
    TestReceiver(logical_core, test_device_context, is_shared);
    this->set_kernel_src(kernel_src);
}

inline bool TestReceiver::is_shared_receiver() { return this->is_shared_; }

inline uint32_t TestReceiver::allocate_space_for_sender() { return this->allocator_.allocate(); }

/* ********************
 * TestDevice Methods *
 **********************/
inline CoreCoord TestDevice::reserve_worker_core() {
    // currently pick from last -> can be configured as a policy (random, optimized etc)
    if (this->avaialble_worker_logical_cores_.empty()) {
        tt::log_fatal(tt::LogTest, "On chip: {}, no more worker cores avaialble", this->physical_chip_id_);
        throw std::runtime_error("Failed to allocate worker core");
    }

    CoreCoord worker_core = this->avaialble_worker_logical_cores_.back();
    this->avaialble_worker_logical_cores_.pop_back();
    return worker_core;
}

inline TestDevice::TestDevice(tt_metal::IDevice* IDevice_handle) {
    const auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    this->IDevice_handle_ = IDevice_handle;

    this->physical_chip_id_ = this->IDevice_handle_->id();
    const auto& fabric_node_id = control_plane->get_fabric_node_id_from_physical_chip_id(this->physical_chip_id_);
    this->mesh_id_ = fabric_node_id.mesh_id;
    this->logical_chip_id_ = fabric_node_id.chip_id;

    this->program_handle_ = tt::tt_metal::CreateProgram();

    const auto grid_size = this->IDevice_handle_->compute_with_storage_grid_size();
    for (auto i = 0; i < grid_size.x; i++) {
        for (auto j = 0; j < grid_size.y; j++) {
            this->avaialble_worker_logical_cores_.push_back(CoreCoord({i, j}));
        }
    }

    this->current_receiver_logical_core_ = std::nullopt;

    // TODO: init routers
}

inline tt::tt_metal::IDevice* TestDevice::get_device_handle() { return this->IDevice_handle_; }

inline tt::tt_metal::Program& TestDevice::get_program_handle() { return this->program_handle_; }

inline chip_id_t TestDevice::get_physical_chip_id() const { return this->physical_chip_id_; }

inline void TestDevice::allocate_receiver_for_sender(CoreCoord receiver_core, bool share_receiver) {
    TestReceiver* receiver_ptr;
    auto reciever_it = this->receivers_.find(receiver_core);
    if (receiver_it == this->receivers_.end()) {
        this->receivers_.emplace(receiver_core, receiver_core, this, share_receiver);
    }

    receiver_ptr = this->receivers_.at(receiver_core);
    if (share_receiver != receiver_ptr->is_shared_receiver()) {
        tt::log_fatal(
            tt::LogTest,
            "On chip: {}, logical core: {} has shared flag set as: {}, but requested: {}",
            this->physical_chip_id_,
            receiver_core,
            !share_receiver,
            share_receiver);
        throw std::runtime_error("Conflicting configurations for sharing the receiver");
    }

    uint32_t address = receiver_ptr->allocate_space_for_sender();
    tt::log_fatal(
        tt::LogTest,
        "On chip: {}, logical core: {} unable to allocate space for sender",
        this->physical_chip_id_,
        receiver_core);
    throw std::runtime_error("Unable to allocate space for sender");
}

inline void TestDevice::allocate_receiver_for_sender(bool share_receiver) {
    if (!this->current_receiver_logical_core_.has_value()) {
        this->current_receiver_logical_core_ = this->reserve_worker_core();
    }

    this->allocate_receiver_for_sender(this->current_receiver_logical_core_.value(), share_receiver);
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
