# Copyright © 2017 Intel Corporation
# SPDX-License-Identifier: MIT

es2_glapi_mapi_tmp_h = custom_target(
  'es2_glapi_mapi_tmp.h',
  input : [glapi_gen_mapi_py, glapi_gen_gl_xml],
  output : 'glapi_mapi_tmp.h',
  command : [prog_python, '@INPUT0@', 'glesv2', '@INPUT1@'],
  depend_files : glapi_gen_mapi_deps,
  capture : true,
)

_es2_c_args = []
if with_platform_windows
  _es2_c_args += ['-D_GDI32_']
endif

gles2_def = custom_target(
  'gles2.def',
  input: 'gles2.def.in',
  output : 'gles2.def',
  command : gen_vs_module_defs_normal_command,
)

libgles2 = shared_library(
  'GLESv2' + get_option('gles-lib-suffix'),
  ['../entry.c', es2_glapi_mapi_tmp_h],
  vs_module_defs : gles2_def,
  c_args : [
    c_msvc_compat_args,
    _es2_c_args,
    '-DMAPI_MODE_BRIDGE',
    '-DMAPI_ABI_HEADER="@0@"'.format(es2_glapi_mapi_tmp_h.full_path()),
    gcc_lto_quirk,
  ],
  gnu_symbol_visibility : 'hidden',
  link_args : [ld_args_gc_sections],
  include_directories : [inc_src, inc_include, inc_mapi],
  link_with : shared_glapi_lib,
  dependencies : [dep_thread, dep_libdrm, dep_m, dep_dl, idep_mesautilc11],
  soversion : host_machine.system() == 'windows' ? '' : '2',
  version : '2.0.0',
  darwin_versions : '3.0.0',
  name_prefix : host_machine.system() == 'windows' ? 'lib' : [],  # always use lib, but avoid warnings on !windows
  install : true,
)

pkg.generate(
  name : 'glesv2',
  filebase : 'glesv2',
  description : 'Mesa OpenGL ES 2.0 library',
  version : meson.project_version(),
  libraries : libgles2,
  libraries_private : gl_priv_libs,
)

if with_symbols_check
  test(
    'es2-ABI-check',
    symbols_check,
    args : [
      '--lib', libgles2,
      '--symbols-file', files('gles2-symbols.txt'),
      symbols_check_args,
    ],
    suite : ['mapi'],
  )
endif
