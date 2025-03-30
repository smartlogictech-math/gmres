file(GLOB CHILD_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(SUBDIR ${CHILD_DIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR})
        set(TEST_SRC_FILES)
        foreach(ext IN LISTS SRC_FILE_EXTENSIONS)
            file(GLOB_RECURSE tmp_files "${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/${ext}")
            list(APPEND TEST_SRC_FILES ${tmp_files})
        endforeach()
        message(STATUS "TEST_SRC_FILES: ${TEST_SRC_FILES}")

        if(TEST_SRC_FILES)
            set(TEST_EXECUTABLE "test_${SUBDIR}")
            add_executable(${TEST_EXECUTABLE} ${TEST_SRC_FILES})
            
            target_include_directories(${TEST_EXECUTABLE}
                PRIVATE /usr/local/cuda/include
                PRIVATE ${CMAKE_SOURCE_DIR}/include
                PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/hal
                PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/kernel
            )

            target_link_libraries(${TEST_EXECUTABLE}
                PRIVATE
                ${project_lib_name}
                pthread
                gtest
                gtest_main
            )

            add_test(NAME ${TEST_EXECUTABLE} COMMAND ${TEST_EXECUTABLE})
        endif()
    endif()
endforeach()