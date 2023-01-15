add_executable(app_exe main.cpp)

target_link_libraries(app_exe PUBLIC ${NET_LIB})
target_link_libraries(app_exe PUBLIC ${TORCH_LIBRARIES})

if (MSVC)
  message("copying dll files")
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET app_exe
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:app_exe>)
endif (MSVC)