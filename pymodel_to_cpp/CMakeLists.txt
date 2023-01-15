add_executable(pymode_to_cpp main.cpp)
target_link_libraries(pymode_to_cpp PUBLIC ${TORCH_LIBRARIES})

if (MSVC)
  message("copying dll files")
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pymode_to_cpp
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:pymode_to_cpp>)
endif (MSVC)