//  Copyright (c)      2025 Aditya Sapra
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// Required HPX execution headers
#include <hpx/config/forward.hpp>
#include <hpx/async_cuda/target.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/rebind_executor.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/executors/sequenced_executor.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/threading_base/execution_agent.hpp>
#include <thrust/execution_policy.h>

#include <memory>
#include <thrust/system/cuda/execution_policy.h>
#include <type_traits>

namespace hpx::thrust {

    struct thrust_task_policy;    //for async ops
    template <typename Executor, typename Parameters>
    struct thrust_task_policy_shim;

    struct thrust_policy;
    template <typename Executor, typename Parameters>
    struct thrust_policy_shim;

    struct thrust_host_policy;
    struct thrust_device_policy;

    struct thrust_task_policy
      : hpx::execution::detail::parallel_task_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>
    {
        using base_type = hpx::execution::detail::parallel_task_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>;

        using base_type::base_type;
        using base_type::with;
        constexpr thrust_task_policy() = default;

        // Bind a CUDA target explicitly for async GPU execution
        thrust_task_policy_shim<hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>
        on(hpx::cuda::experimental::target const& t) const;

        // Async helpers with default-target fallback for base policy
        bool has_target() const
        {
            return false;
        }

        hpx::cuda::experimental::target const& target_or_default() const
        {
            return hpx::cuda::experimental::get_default_target();
        }

        cudaStream_t stream() const
        {
            return target_or_default().native_handle().get_stream();
        }

        auto get() const
        {
            return ::thrust::cuda::par_nosync.on(stream());
        }

        hpx::future<void> get_future() const
        {
            return target_or_default().get_future_with_event();
        }
    };

    template <typename Executor, typename Parameters>
    struct thrust_task_policy_shim
      : hpx::execution::detail::parallel_task_policy_shim<Executor, Parameters>
    {
        using base_type =
            hpx::execution::detail::parallel_task_policy_shim<Executor, Parameters>;
        using base_type::base_type;
        using base_type::with;
        constexpr thrust_task_policy_shim() = default;

        template <typename Executor_, typename Parameters_>
        constexpr thrust_task_policy_shim(Executor_&& exec, Parameters_&& params)
          : base_type(HPX_FORWARD(Executor_, exec), HPX_FORWARD(Parameters_, params))
        {
        }

        // Bind a CUDA target explicitly for async GPU execution
        thrust_task_policy_shim on(
            hpx::cuda::experimental::target const& t) const
        {
            thrust_task_policy_shim copy = *this;
            copy.bound_target_ =
                std::make_shared<hpx::cuda::experimental::target>(t);
            return copy;
        }

        // Async helpers with default-target fallback
        bool has_target() const
        {
            return static_cast<bool>(bound_target_);
        }

        hpx::cuda::experimental::target const& target_or_default() const
        {
            return bound_target_ ?
                *bound_target_ :
                hpx::cuda::experimental::get_default_target();
        }

        cudaStream_t stream() const
        {
            return target_or_default().native_handle().get_stream();
        }

        auto get() const
        {
            return ::thrust::cuda::par_nosync.on(stream());
        }

        hpx::future<void> get_future() const
        {
            return target_or_default().get_future_with_event();
        }

        // Construct with an already bound CUDA target
        explicit thrust_task_policy_shim(
            std::shared_ptr<hpx::cuda::experimental::target> tgt)
          : bound_target_(std::move(tgt))
        {
        }

    private:
        std::shared_ptr<hpx::cuda::experimental::target> bound_target_{};
    };

    inline thrust_task_policy_shim<hpx::execution::parallel_executor,
        hpx::traits::executor_parameters_type_t<
            hpx::execution::parallel_executor>>
    thrust_task_policy::on(hpx::cuda::experimental::target const& t) const
    {
        using shim_type = thrust_task_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>;
        return shim_type(std::make_shared<hpx::cuda::experimental::target>(t));
    }

    // Base thrust_policy derived from HPX parallel policy shim
    struct thrust_policy
      : hpx::execution::detail::parallel_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>
    {
        using base_type = hpx::execution::detail::parallel_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>;
        using base_type::base_type;
        using base_type::on;
        using base_type::with;
        constexpr thrust_policy() = default;

        thrust_task_policy operator()(hpx::execution::experimental::to_task_t) const
        {
            return thrust_task_policy(
                this->executor(), this->parameters());
        }
    };


    // Host-specific policy
    struct thrust_host_policy
      : hpx::execution::detail::parallel_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>
    {
        using base_type = hpx::execution::detail::parallel_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>;
        using base_type::base_type;
        constexpr thrust_host_policy() = default;

        // Return thrust::host execution policy
        constexpr auto get() const
        {
            return ::thrust::host;
        }
    };

    // Device-specific policy
    struct thrust_device_policy
      : hpx::execution::detail::parallel_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>
    {
        using base_type = hpx::execution::detail::parallel_policy_shim<
            hpx::execution::parallel_executor,
            hpx::traits::executor_parameters_type_t<
                hpx::execution::parallel_executor>>;
        using base_type::base_type;
        constexpr thrust_device_policy() = default;

        // Return thrust::device execution policy
        constexpr auto get() const
        {
            return ::thrust::device;
        }
    };

    // Global policy instances
    inline constexpr thrust_host_policy thrust_host{};
    inline constexpr thrust_device_policy thrust_device{};

    static constexpr thrust_policy thrust;

    template <typename ExecutionPolicy>
    struct is_thrust_execution_policy : std::false_type
    {
    };

    template <>
    struct is_thrust_execution_policy<hpx::thrust::thrust_policy>
      : std::true_type
    {
    };


    template <>
    struct is_thrust_execution_policy<hpx::thrust::thrust_host_policy>
      : std::true_type
    {
    };

    template <>
    struct is_thrust_execution_policy<hpx::thrust::thrust_device_policy>
      : std::true_type
    {
    };

    template <>
    struct is_thrust_execution_policy<hpx::thrust::thrust_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_thrust_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_thrust_execution_policy_v =
        is_thrust_execution_policy<T>::value;

    namespace detail {
        template <typename ExecutionPolicy, typename Enable = void>
        struct get_policy_result;

        template <typename ExecutionPolicy>
        struct get_policy_result<ExecutionPolicy,
            std::enable_if_t<hpx::is_async_execution_policy_v<
                std::decay_t<ExecutionPolicy>>>>
        {
            static_assert(is_thrust_execution_policy<
                              std::decay_t<ExecutionPolicy>>::value,
                "get_policy_result can only be used with Thrust execution "
                "policies");

            using type = hpx::future<void>;

            template <typename Future>
            static constexpr decltype(auto) call(Future&& future)
            {
                return std::forward<Future>(future);
            }
        };

        template <typename ExecutionPolicy>
        struct get_policy_result<ExecutionPolicy,
            std::enable_if_t<!hpx::is_async_execution_policy_v<
                std::decay_t<ExecutionPolicy>>>>
        {
            static_assert(is_thrust_execution_policy<
                              std::decay_t<ExecutionPolicy>>::value,
                "get_policy_result can only be used with Thrust execution "
                "policies");

            template <typename Future>
            static constexpr decltype(auto) call(Future&& future)
            {
                return std::forward<Future>(future).get();
            }
        };
    }    // namespace detail

}    // namespace hpx::thrust

namespace hpx::detail {

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::thrust::thrust_policy> : std::true_type
    {
    };


    template <>
    struct is_execution_policy<hpx::thrust::thrust_host_policy> : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::thrust::thrust_device_policy>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::thrust::thrust_task_policy> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<hpx::thrust::thrust_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::thrust::thrust_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<hpx::thrust::thrust_host_policy>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<hpx::thrust::thrust_device_policy>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<hpx::thrust::thrust_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_async_execution_policy<hpx::thrust::thrust_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

}    // namespace hpx::detail
