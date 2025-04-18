# Copyright © 2017 Broadcom
# SPDX-License-Identifier: MIT

files_libvc4 = files(
  'kernel/vc4_drv.h',
  'kernel/vc4_gem.c',
  'kernel/vc4_packet.h',
  'kernel/vc4_render_cl.c',
  'kernel/vc4_validate.c',
  'kernel/vc4_validate_shaders.c',
  'vc4_blit.c',
  'vc4_bufmgr.c',
  'vc4_bufmgr.h',
  'vc4_cl.c',
  'vc4_cl_dump.c',
  'vc4_cl_dump.h',
  'vc4_cl.h',
  'vc4_context.c',
  'vc4_context.h',
  'vc4_draw.c',
  'vc4_emit.c',
  'vc4_fence.c',
  'vc4_formats.c',
  'vc4_job.c',
  'vc4_nir_lower_blend.c',
  'vc4_nir_lower_io.c',
  'vc4_nir_lower_txf_ms.c',
  'vc4_opt_algebraic.c',
  'vc4_opt_constant_folding.c',
  'vc4_opt_copy_propagation.c',
  'vc4_opt_dead_code.c',
  'vc4_opt_peephole_sf.c',
  'vc4_opt_small_immediates.c',
  'vc4_opt_vpm.c',
  'vc4_opt_coalesce_ff_writes.c',
  'vc4_program.c',
  'vc4_qir.c',
  'vc4_qir_emit_uniform_stream_resets.c',
  'vc4_qir_live_variables.c',
  'vc4_qir_lower_uniforms.c',
  'vc4_qir_schedule.c',
  'vc4_qir_validate.c',
  'vc4_qir.h',
  'vc4_qpu.c',
  'vc4_qpu_defines.h',
  'vc4_qpu_disasm.c',
  'vc4_qpu_emit.c',
  'vc4_qpu.h',
  'vc4_qpu_schedule.c',
  'vc4_qpu_validate.c',
  'vc4_query.c',
  'vc4_register_allocate.c',
  'vc4_reorder_uniforms.c',
  'vc4_resource.c',
  'vc4_resource.h',
  'vc4_screen.c',
  'vc4_screen.h',
  'vc4_simulator.c',
  'vc4_simulator_validate.h',
  'vc4_state.c',
  'vc4_tiling.c',
  'vc4_tiling_lt.c',
  'vc4_tiling.h',
  'vc4_uniforms.c',
)

vc4_c_args = []

vc4_deps = [
  dep_simpenrose, dep_libdrm, dep_valgrind,
  idep_nir_headers, idep_mesautil,
]

libvc4_neon = []
if host_machine.cpu_family() == 'arm'
  libvc4_neon = static_library(
    'vc4_neon',
    'vc4_tiling_lt.c',
    include_directories : [
      inc_src, inc_include, inc_gallium, inc_gallium_aux, inc_broadcom
    ],
    c_args : ['-mfpu=neon', '-DV3D_BUILD_NEON'],
    dependencies : vc4_deps,
  )
  vc4_c_args += '-DUSE_ARM_ASM'
endif

if dep_simpenrose.found()
  vc4_c_args += '-DUSE_VC4_SIMULATOR'
endif

libvc4 = static_library(
  'vc4',
  [files_libvc4, v3d_xml_pack],
  include_directories : [
    inc_src, inc_include, inc_gallium, inc_gallium_aux, inc_broadcom,
    inc_gallium_drivers,
  ],
  link_with: libvc4_neon,
  c_args : [vc4_c_args],
  gnu_symbol_visibility : 'hidden',
  dependencies : vc4_deps,
  build_by_default : false,
)

driver_vc4 = declare_dependency(
  compile_args : '-DGALLIUM_VC4',
  link_with : [libvc4, libvc4winsys, libbroadcom_cle, libbroadcom_v3d],
  dependencies : idep_nir,
)
