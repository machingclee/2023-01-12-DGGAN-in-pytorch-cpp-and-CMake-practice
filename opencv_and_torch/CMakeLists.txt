add_executable(opencv_and_torch main.cpp)
target_link_libraries(opencv_and_torch ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(opencv_and_torch PUBLIC ${OpenCV_INCLUDE_DIRS})

if (MSVC)
  message("copying dll files")
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET opencv_and_torch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:opencv_and_torch>)
endif (MSVC)

set(TARGET opencv_and_torch PROPERTY CXX_STANDARD 14)