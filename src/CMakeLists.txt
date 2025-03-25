# HAL
set(HAL_SRC_FILES)
foreach(ext IN LISTS SRC_FILE_EXTENSIONS)
    file(GLOB_RECURSE tmp_files "hal/${ext}")
    list(APPEND HAL_SRC_FILES ${tmp_files})
endforeach()
message(STATUS "HAL_SRC_FILES: ${HAL_SRC_FILES}")

add_library(hal OBJECT
  ${HAL_SRC_FILES}
)
target_include_directories(hal
  PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/hal
)

# Kernel
set(KERNEL_SRC_FILES)
foreach(ext IN LISTS SRC_FILE_EXTENSIONS)
    file(GLOB_RECURSE tmp_files "kernel/${ext}")
    list(APPEND KERNEL_SRC_FILES ${tmp_files})
endforeach()
message(STATUS "KERNEL_SRC_FILES: ${KERNEL_SRC_FILES}")

add_library(kernel OBJECT
    ${KERNEL_SRC_FILES}
)
target_include_directories(kernel
#   PUBLIC ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/ 
  PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/kernel
)
target_link_libraries(kernel PRIVATE hal)

# API
set(API_SRC_FILES)
foreach(ext IN LISTS SRC_FILE_EXTENSIONS)
    file(GLOB_RECURSE tmp_files "api/${ext}")
    list(APPEND API_SRC_FILES ${tmp_files})
endforeach()
message(STATUS "API_SRC_FILES: ${API_SRC_FILES}")

add_library(${project_lib_name}
    ${API_SRC_FILES}
    $<TARGET_OBJECTS:hal>
    $<TARGET_OBJECTS:kernel>
)
target_include_directories(${project_lib_name}
  PRIVATE /usr/local/cuda/include
  PUBLIC ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/hal
  PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/kernel
)