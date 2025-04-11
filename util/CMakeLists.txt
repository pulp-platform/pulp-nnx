add_library(pulp-nnx-util
	OBJECT
	hwpe.c
	pulp_nnx_util.c
)

target_include_directories(pulp-nnx-util PUBLIC .)
