cmake_minimum_required(VERSION 3.10)
project(Try)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


option(UNIT_TEST "Enable Unit Test" OFF)
option(COMPLETE_TEST "Enable Complete Test" OFF)

# configure the include directories.
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/
    ${CMAKE_CURRENT_LIST_DIR}/nn/
    ${CMAKE_CURRENT_LIST_DIR}/Tensor/
)


add_subdirectory(Tensor)
add_subdirectory(nn)
add_executable(Try Try.cc)


if(UNIT_TEST)
    target_compile_definitions(Try PUBLIC "UNIT_TEST")
    target_compile_definitions(Tensor PUBLIC "UNIT_TEST")
endif()
if(COMPLETE_TEST)
    target_compile_definitions(Try PUBLIC "COMPLETE_TEST")
endif()


target_link_libraries(Try PUBLIC Tensor)
if(COMPLETE_TEST)
    target_link_libraries(Try PUBLIC nn)
endif()


if(UNIT_TEST)
    target_include_directories(
        Try PUBLIC 
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/TENSOR"
    )
endif()
if(COMPELTE_TEST)
    target_include_directories(
        Try PUBLIC 
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/TENSOR"
        "${PROJECT_SOURCE_DIR}/nn"
    )
endif()