# cmake_minimum_required(VERSION 3.10)
# project(Try)

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_subdirectory(Tensor)
# add_subdirectory(nn)

# add_executable(Try Try.cc)

# target_link_libraries(Try PUBLIC Tensor)
# target_link_libraries(Try PUBLIC nn)
# target_include_directories(
#     Try PUBLIC 
#     "${PROJECT_BINARY_DIR}"
#     "${PROJECT_SOURCE_DIR}/TENSOR"
#     "${PROJECT_SOURCE_DIR}/nn"
# )

cmake_minimum_required(VERSION 3.10)
project(Try)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(Tensor)
add_executable(Try Try.cc)

target_link_libraries(Try PUBLIC Tensor)
target_include_directories(
    Try PUBLIC 
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/TENSOR"
)