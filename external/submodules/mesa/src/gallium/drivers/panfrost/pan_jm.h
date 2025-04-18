/*
 * Copyright (C) 2023 Collabora Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef __PAN_JM_H__
#define __PAN_JM_H__

#include "pan_jc.h"

struct panfrost_jm_batch {
   /* Job related fields. */
   struct {
      /* Vertex/tiler/compute job chain. */
      struct pan_jc vtc_jc;

      /* Fragment job, only one per batch. */
      uint64_t frag;
   } jobs;
};

#if defined(PAN_ARCH) && PAN_ARCH < 10

#include "genxml/gen_macros.h"

struct panfrost_batch;
struct panfrost_context;
struct pan_fb_info;
struct pan_tls_info;
struct pipe_draw_info;
struct pipe_grid_info;
struct pipe_draw_start_count_bias;

static inline int
GENX(jm_init_context)(struct panfrost_context *ctx)
{
   return 0;
}

static inline void
GENX(jm_cleanup_context)(struct panfrost_context *ctx)
{
}

int
GENX(jm_init_batch)(struct panfrost_batch *batch);

static inline void
GENX(jm_cleanup_batch)(struct panfrost_batch *batch)
{
}

int GENX(jm_submit_batch)(struct panfrost_batch *batch);

static inline void
GENX(jm_prepare_tiler)(struct panfrost_batch *batch, struct pan_fb_info *fb)
{
}

void GENX(jm_preload_fb)(struct panfrost_batch *batch, struct pan_fb_info *fb);
void GENX(jm_emit_fbds)(struct panfrost_batch *batch, struct pan_fb_info *fb,
                        struct pan_tls_info *tls);
void GENX(jm_emit_fragment_job)(struct panfrost_batch *batch,
                                const struct pan_fb_info *pfb);

void GENX(jm_launch_xfb)(struct panfrost_batch *batch,
                         const struct pipe_draw_info *info, unsigned count);

void GENX(jm_launch_grid)(struct panfrost_batch *batch,
                          const struct pipe_grid_info *info);

void GENX(jm_launch_draw)(struct panfrost_batch *batch,
                          const struct pipe_draw_info *info,
                          unsigned drawid_offset,
                          const struct pipe_draw_start_count_bias *draw,
                          unsigned vertex_count);
void GENX(jm_launch_draw_indirect)(struct panfrost_batch *batch,
                                   const struct pipe_draw_info *info,
                                   unsigned drawid_offset,
                                   const struct pipe_draw_indirect_info *indirect);

void GENX(jm_emit_write_timestamp)(struct panfrost_batch *batch,
                                   struct panfrost_resource *dst,
                                   unsigned offset);

#endif /* PAN_ARCH < 10 */

#endif
