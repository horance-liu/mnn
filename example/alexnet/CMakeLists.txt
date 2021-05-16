file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
    *.cpp
    *.cc
    *.c
)

add_executable(alexnet_cons ${srcs})

target_link_libraries(alexnet_cons
    PRIVATE mnn ${REQUIRED_LIBRARIES}
)
