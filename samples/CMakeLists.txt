set(DEMO_SRC_FILES)
foreach(ext IN LISTS SRC_FILE_EXTENSIONS)
    file(GLOB tmp_files "./${ext}")
    list(APPEND DEMO_SRC_FILES ${tmp_files})
endforeach()
message(STATUS "DEMO_SRC_FILES: ${DEMO_SRC_FILES}")

foreach(SRC_FILE IN LISTS DEMO_SRC_FILES)
    get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${SRC_FILE})
    target_include_directories(${EXE_NAME}
        PRIVATE /usr/local/cuda/include
    )
    target_link_libraries(${EXE_NAME} PRIVATE ${project_lib_name})
endforeach()