if (NOT DEFINED ENV{TOOLCHAIN_LLVM_INSTALL_DIR})
  message(FATAL_ERROR "Environment variable TOOLCHAIN_LLVM_INSTALL_DIR not defined.")
endif()

set(TOOLCHAIN_BIN $ENV{TOOLCHAIN_LLVM_INSTALL_DIR}/bin)
set(PICOLIBC $ENV{TOOLCHAIN_LLVM_INSTALL_DIR}/picolibc/riscv)
set(COMPILER_RT $ENV{TOOLCHAIN_LLVM_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32imc/)

set(CMAKE_SYSTEM_NAME Generic)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_BIN}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_BIN}/clang++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_BIN}/clang)
set(CMAKE_OBJCOPY ${TOOLCHAIN_BIN}/llvm-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_BIN}/llvm-objdump)

set(ISA rv32imc_zfinx_xpulpv2)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

add_compile_options(
  -target riscv32-unknown-elf
  -march=${ISA}
  -ffunction-sections
  -fdata-sections
  -fomit-frame-pointer
  -mno-relax
  -O3
  -DNUM_CORES=${NUM_CORES}
  -MMD
  -MP
  --sysroot=${PICOLIBC}
  -fno-builtin-memcpy
  -fno-builtin-memset
)

add_link_options(
  -target riscv32-unknown-elf
  -MMD
  -MP
  -nostartfiles
  -march=${ISA}
  --sysroot=${PICOLIBC}
  -L${COMPILER_RT}
  -z norelro
  -fno-builtin-memcpy
  -fno-builtin-memset
)

link_libraries(
  -lm
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_LLVM__)
