if (NOT DEFINED ENV{LLVM_INSTALL_DIR})
  message(FATAL_ERROR "Environment variable LLVM_INSTALL_DIR not defined.")
endif()

set(LLVM_BIN $ENV{LLVM_INSTALL_DIR}/bin)
set(PICOLIBC_RISCV_ROOT $ENV{LLVM_INSTALL_DIR}/picolibc/riscv)
set(COMPILER_RT_RISCV_ROOT $ENV{LLVM_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32imc/)

set(CMAKE_SYSTEM_NAME Generic)

set(CMAKE_C_COMPILER   ${LLVM_BIN}/clang)
set(CMAKE_CXX_COMPILER ${LLVM_BIN}/clang++)
set(CMAKE_ASM_COMPILER ${LLVM_BIN}/clang)
set(CMAKE_OBJCOPY ${LLVM_BIN}/llvm-objcopy)
set(CMAKE_OBJDUMP ${LLVM_BIN}/llvm-objdump)

set(CMAKE_SYSTEM_PROCESSOR "pulp")

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
  -MMD
  -MP
  --sysroot=${PICOLIBC_RISCV_ROOT}
  -fno-builtin-memcpy
  -fno-builtin-memset
  -fcolor-diagnostics
)

add_link_options(
  -target riscv32-unknown-elf
  -MMD
  -MP
  -march=${ISA}
  --sysroot=${PICOLIBC_RISCV_ROOT}
  -L${COMPILER_RT_RISCV_ROOT}
  -z norelro
  -fno-builtin-memcpy
  -fno-builtin-memset
)

link_libraries(
  -lm
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_LLVM__)
