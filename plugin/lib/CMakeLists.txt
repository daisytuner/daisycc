list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")

find_package(LLVM 16 REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})

find_package(Polly REQUIRED CONFIG)
include_directories(${Polly_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

add_library(DaisyLLVMPlugin SHARED
  Daisy.cpp
  JScop.cpp
  RegisterPasses.cpp
)
target_link_libraries(DaisyLLVMPlugin "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

include(GNUInstallDirs)
install(TARGETS DaisyLLVMPlugin
    EXPORT "${DaisytunerLLVMPlugin}Targets"
    # these get default values from GNUInstallDirs, no need to set them
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
    #ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
    # PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} # include/SomeProject
    # INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # include
)

