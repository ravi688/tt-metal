function(setup_ttnn_test_target target_name)
    target_link_libraries(
        ${target_name}
        PUBLIC
            test_common_libs
            ttnn
            Metalium::Metal
            GTest::gmock_main
    )
    target_include_directories(
        ${target_name}
        PRIVATE
            ${UMD_HOME}
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tt_metal
            ${PROJECT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}
            "$<TARGET_PROPERTY:TT::NN::CPP,INCLUDE_DIRECTORIES>"
    )
    set_target_properties(
        ${target_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/ttnn
    )
endfunction()

set(TTNN_CCL_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_erisc_data_mover_with_workers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_fabric_erisc_data_mover_loopback_with_workers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_ccl_commands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_ccl_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_ccl_tensor_slicers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_sharded_address_generators_new.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_sharded_address_generators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_ccl_reduce_scatter_host_helpers.cpp
)

set(TTNN_CCL_OP_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_persistent_fabric_ccl_ops.cpp)

set(TTNN_FABRIC_EDM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_fabric_edm.cpp)
set(TTNN_CCL_MULTI_TENSOR_UNIT_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_multi_tensor_ccl.cpp)
set(TTNN_1D_FABRIC_LATENCY_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_1d_fabric_loopback_latency.cpp)

set(TTNN_ACCESSOR_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/accessor/test_sharded_accessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/accessor/test_sharded_accessor_on_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/accessor/test_accessor_benchmarks.cpp
)

set(TTNN_TENSOR_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/common_tensor_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor_with_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_distributed_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_mesh_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_sharding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_nd_sharding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_vector_conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_xtensor_conversion.cpp
)

set(EMITC_UNIT_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/emitc/test_sanity.cpp)

add_library(unit_tests_ttnn_smoke OBJECT)
add_library(TT::NN::Test::Smoke ALIAS unit_tests_ttnn_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn_smoke)
target_sources(
    unit_tests_ttnn_smoke
    PRIVATE
        test_add.cpp
        test_graph_add.cpp
        test_graph_basic.cpp
        test_async_runtime.cpp
        test_matmul_benchmark.cpp
        test_multiprod_queue.cpp
        test_multi_cq_multi_dev.cpp
        test_launch_operation.cpp
        test_graph_capture_arguments_morehdot.cpp
        test_graph_capture_arguments_transpose.cpp
        test_graph_query_op_constraints.cpp
        test_reflect.cpp
        test_to_and_from_json.cpp
        test_broadcast_to.cpp
        test_sliding_window_infra.cpp
        common_test_utils.cpp
        test_generic_op.cpp
        test_conv2d.cpp
        test_graph_query_op_runtime.cpp
)
target_include_directories(unit_tests_ttnn_smoke PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_smoke
    PRIVATE
        test_common_libs
        TT::NN::CPP
)

#[[
add_library(unit_tests_ttnn_basic OBJECT)
add_library(TT::NN::Test::Basic ALIAS unit_tests_ttnn_basic)
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn_basic)
target_sources(
    unit_tests_ttnn_basic
    PRIVATE
        test_conv2d.cpp
        test_graph_query_op_runtime.cpp
)
target_include_directories(unit_tests_ttnn_basic PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_basic
    PRIVATE
        test_common_libs
        TT::NN::CPP
)
#]]

add_executable(unit_tests_ttnn)
set_target_properties(
    unit_tests_ttnn
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/ttnn
)
target_link_libraries(
    unit_tests_ttnn
    PRIVATE
        TT::NN::Test::Smoke
    # TT::NN::Test::Basic
)

add_executable(unit_tests_ttnn_ccl ${TTNN_CCL_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_ccl_ops ${TTNN_CCL_OP_TESTS_SRC})
add_executable(unit_tests_ttnn_fabric_edm ${TTNN_FABRIC_EDM_SRC})
add_executable(unit_tests_ttnn_ccl_multi_tensor ${TTNN_CCL_MULTI_TENSOR_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_1d_fabric_latency ${TTNN_1D_FABRIC_LATENCY_TEST_SRC})
add_executable(unit_tests_ttnn_accessor ${TTNN_ACCESSOR_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_tensor ${TTNN_TENSOR_UNIT_TESTS_SRC})
target_link_libraries(unit_tests_ttnn_tensor PRIVATE xtensor)
add_executable(
    test_ccl_multi_cq_multi_device
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_thread/test_ccl_multi_cq_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_thread/test_utils.cpp
)
add_executable(unit_tests_ttnn_emitc ${EMITC_UNIT_TESTS_SRC})

target_link_libraries(
    test_ccl_multi_cq_multi_device
    PRIVATE
        Boost::asio
        Boost::lockfree
)

# Set up properties for all targets
target_include_directories(unit_tests_ttnn_ccl PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_ccl
    PRIVATE
        test_common_libs
        TT::NN::CPP
)
setup_ttnn_test_target(unit_tests_ttnn_ccl_ops)
setup_ttnn_test_target(unit_tests_ttnn_fabric_edm)
setup_ttnn_test_target(unit_tests_ttnn_ccl_multi_tensor)
setup_ttnn_test_target(unit_tests_ttnn_1d_fabric_latency)
setup_ttnn_test_target(unit_tests_ttnn_accessor)
setup_ttnn_test_target(unit_tests_ttnn_tensor)
setup_ttnn_test_target(test_ccl_multi_cq_multi_device)
setup_ttnn_test_target(unit_tests_ttnn_emitc)
