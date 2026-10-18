# Only included when PVZ_CUDA_ROLLOUT=ON. The service uses POSIX sockets (including WSL).
if(CMAKE_VERSION VERSION_LESS 3.18)
	message(FATAL_ERROR "PVZ_CUDA_ROLLOUT needs CMake 3.18 or newer")
endif()
if(NOT UNIX)
	message(FATAL_ERROR "The CUDA continuation service currently needs POSIX sockets; use WSL on Windows")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures (RTX 5080: 120)")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(rollout_cuda ${PROJECT_SOURCE_DIR}/cuda/RolloutCuda.cu)
target_include_directories(rollout_cuda PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(rollout_cuda PRIVATE CUDA::cublas CUDA::cudart)
target_compile_options(rollout_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(rollout_cuda PROPERTIES
	CUDA_STANDARD 17
	CUDA_STANDARD_REQUIRED ON
	CUDA_EXTENSIONS OFF
	CUDA_RUNTIME_LIBRARY Shared
)

# Small teacher requests otherwise run the first 1024 activation slots even when only a few are
# occupied. Keep the large-wave target above, and provide the measured 128-slot layout explicitly.
add_executable(rollout_cuda_small ${PROJECT_SOURCE_DIR}/cuda/RolloutCuda.cu)
target_include_directories(rollout_cuda_small PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(rollout_cuda_small PRIVATE CUDA::cublas CUDA::cudart)
target_compile_definitions(rollout_cuda_small PRIVATE PVZ_NET_CHUNK_BITS=7)
target_compile_options(rollout_cuda_small PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(rollout_cuda_small PROPERTIES
	CUDA_STANDARD 17
	CUDA_STANDARD_REQUIRED ON
	CUDA_EXTENSIONS OFF
	CUDA_RUNTIME_LIBRARY Shared
)
