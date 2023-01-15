set(
  SOURCES
  include/nets/net.h
  include/nets/dcgan.h
)

add_library(${NET_LIB} INTERFACE ${SOURCES})

target_include_directories(${NET_LIB} INTERFACE include)
target_link_libraries(${NET_LIB} INTERFACE ${TORCH_LIBRARIES})
