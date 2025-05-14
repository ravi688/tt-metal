// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "watcher_common.h"

// We don't control the stack size for active erisc, and share the stack with base FW, so don't implement for ERISC.
#if defined(WATCHER_ENABLED) && !(defined(WATCHER_DISABLE_STACK_USAGE) || defined(COMPILE_FOR_ERISC) || \
                                  defined(FORCE_WATCHER_OFF))

#define STACK_DIRTY_PATTERN 0xBABABABA

uint32_t get_stack_size() {
#if defined(COMPILE_FOR_BRISC)
    return MEM_BRISC_STACK_SIZE;
#elif defined(COMPILE_FOR_NCRISC)
    return MEM_NCRISC_STACK_SIZE;
#elif defined(COMPILE_FOR_IDLE_ERISC)
#if COMPILE_FOR_IDLE_ERISC == 0
    return MEM_IERISC_STACK_SIZE;
#elif COMPILE_FOR_IDLE_ERISC == 1
    return MEM_SLAVE_IERISC_STACK_SIZE;
#else
#error "idle erisc get_stack_size unknown"
#endif
#elif defined(COMPILE_FOR_TRISC)
#if COMPILE_FOR_TRISC == 0
    return MEM_TRISC0_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 1
    return MEM_TRISC1_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 2
    return MEM_TRISC2_STACK_SIZE;
#else
#error "trisc get_stack_size unknown"
#endif
#else
#error "get_stack_size unknown"
#endif
}

uint32_t get_stack_top() {
#if defined(COMPILE_FOR_BRISC)
    return MEM_BRISC_STACK_TOP;
#elif defined(COMPILE_FOR_NCRISC)
    return MEM_NCRISC_STACK_TOP;
#elif defined(COMPILE_FOR_IDLE_ERISC)
#if COMPILE_FOR_IDLE_ERISC == 0
    return MEM_IERISC_STACK_TOP;
#elif COMPILE_FOR_IDLE_ERISC == 1
    return MEM_SLAVE_IERISC_STACK_TOP;
#else
#error "idle erisc get_stack_top unknown"
#endif
#elif defined(COMPILE_FOR_TRISC)
#if COMPILE_FOR_TRISC == 0
    return MEM_TRISC0_STACK_TOP;
#elif COMPILE_FOR_TRISC == 1
    return MEM_TRISC1_STACK_TOP;
#elif COMPILE_FOR_TRISC == 2
    return MEM_TRISC2_STACK_TOP;
#else
#error "trisc get_stack_top unknown"
#endif
#else
#error "get_stack_top unknown"
#endif
}

uint32_t get_dispatch_class() {
#if defined(COMPILE_FOR_BRISC)
    return DISPATCH_CLASS_TENSIX_DM0;
#elif defined(COMPILE_FOR_NCRISC)
    return DISPATCH_CLASS_TENSIX_DM1;
#elif defined(COMPILE_FOR_ERISC)
    return DISPATCH_CLASS_ETH_DM0;
#elif defined(COMPILE_FOR_IDLE_ERISC)
    return COMPILE_FOR_IDLE_ERISC == 0 ? DISPATCH_CLASS_ETH_DM0
        : DISPATCH_CLASS_ETH_DM1;
#elif defined(COMPILE_FOR_TRISC)
    return DISPATCH_CLASS_TENSIX_COMPUTE;
#else
#error "dispatch class not defined"
#endif
}

void dirty_stack_memory() {
    uint32_t stack_words = get_stack_size() / sizeof(uint32_t);
    uint32_t *base = (uint32_t tt_l1_ptr *)get_stack_top() - stack_words;

    uint32_t tt_l1_ptr *ptr;
    asm ("mov %,sp" : "=r"(ptr));

    while (ptr != base)
        *--ptr = STACK_DIRTY_PATTERN;
}

void record_stack_usage() {
    uint32_t stack_words = get_stack_size() / sizeof(uint32_t);
    uint32_t *base = (uint32_t tt_l1_ptr *)get_stack_top() - stack_words;

    uint32_t tt_l1_ptr* stack_ptr = base;
    // We don't need to check size here, as we know we'll hit a
    // non-dirty value at some point (a set of return addresses).
    while (*stack_ptr == STACK_DIRTY_PATTERN)
        stack_ptr++;
    uint32_t stack_usage = (stack_words - (stack_ptr - base)) * sizeof (uint32_t);
    unsigned idx = debug_get_which_riscv();
    debug_stack_usage_t tt_l1_ptr* stack_usage_msg = GET_MAILBOX_ADDRESS_DEV(watcher.stack_usage);
    if (stack_usage_msg->watcher_kernel_id[idx] == 0 ||  // No entry recorded
        stack_usage_msg->max_usage[idx] < stack_usage) {
        stack_usage_msg->max_usage[idx] = stack_usage;
        unsigned launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        launch_msg_t tt_l1_ptr* launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_idx]);
        stack_usage_msg->watcher_kernel_id[idx] =
            launch_msg->kernel_config.watcher_kernel_ids[get_dispatch_class()];
    }
}

#define DIRTY_STACK_MEMORY() dirty_stack_memory()
#define RECORD_STACK_USAGE() record_stack_usage()

#else  // !WATCHER_ENABLED

#define DIRTY_STACK_MEMORY()
#define RECORD_STACK_USAGE()

#endif  // WATCHER_ENABLED
