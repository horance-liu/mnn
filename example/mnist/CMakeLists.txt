file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
    *.cpp
    *.cc
    *.c
)

add_executable(mnist_train ${srcs})

target_link_libraries(mnist_train
    PRIVATE mnn ${REQUIRED_LIBRARIES}
)
