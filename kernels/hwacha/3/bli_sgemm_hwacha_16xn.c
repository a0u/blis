/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#define __vsetcfg(nvd, nvp) \
	__asm__ __volatile__ ("vsetcfg " #nvd ", " #nvp)
#define vsetcfg(nvd, nvp) __vsetcfg(nvd, nvp)

#define vsetvl(vl) __extension__ ({ \
	size_t n; \
	__asm__ __volatile__ ("vsetvl %0, %1" : "=r" (n) : "r" (vl)); \
	n; \
})

#define vmca(va, x) \
	__asm__ __volatile__ ("vmca " #va ", %0" : : "r" (x))

#define vmcs(vs, x) \
	__asm__ __volatile__ ("vmcs " #vs ", %0" : : "r" (x))

#define vf(p) \
	__asm__ __volatile__ ("vf %0" : : "A" (p))


extern void bli_sgemm_hwacha_16xn_vf_init(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_init_beta(void)__attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_inner_0(void)__attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_inner_1(void)__attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_tail_0(void)__attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_tail_1(void)__attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_end(void)__attribute__((visibility("protected")));

static inline void bli_sgemm_hwacha_16xn_load_a
     (
       const float* restrict a
     )
{
	vmcs(vs1, a[0]);
	vmcs(vs2, a[1]);
	vmcs(vs3, a[2]);
	vmcs(vs4, a[3]);
	vmcs(vs5, a[4]);
	vmcs(vs6, a[5]);
	vmcs(vs7, a[6]);
	vmcs(vs8, a[7]);
	vmcs(vs9, a[8]);
	vmcs(vs10, a[9]);
	vmcs(vs11, a[10]);
	vmcs(vs12, a[11]);
	vmcs(vs13, a[12]);
	vmcs(vs14, a[13]);
	vmcs(vs15, a[14]);
	vmcs(vs16, a[15]);
}

void bli_sgemm_hwacha_16xn
     (
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,
       float*     restrict b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	const num_t        dt     = BLIS_FLOAT;

//	const dim_t        mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t        nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );

//	const inc_t        packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx );
//	const inc_t        packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );

	vsetcfg(16+2, 1);
	if (vsetvl(nr) < nr) {
		bli_abort();
	}

	vmca(va0, c);
	vmca(va1, c += rs_c0);
	vmca(va2, c += rs_c0);
	vmca(va3, c += rs_c0);
	vmca(va4, c += rs_c0);
	vmca(va5, c += rs_c0);
	vmca(va6, c += rs_c0);
	vmca(va7, c += rs_c0);
	vmca(va8, c += rs_c0);
	vmca(va9, c += rs_c0);
	vmca(va10, c += rs_c0);
	vmca(va11, c += rs_c0);
	vmca(va12, c += rs_c0);
	vmca(va13, c += rs_c0);
	vmca(va14, c += rs_c0);
	vmca(va15, c += rs_c0);
	vmca(va16, b);
	b += rs_c0;

	if (*beta) {
		vmcs(vs63, *beta);
		vf(bli_sgemm_hwacha_16xn_vf_init_beta);
	} else {
		vf(bli_sgemm_hwacha_16xn_vf_init);
	}

	vmcs(vs63, *alpha);

	/*
	 * FIXME: Hack around suboptimal gcc code generation with
	 * for (; k0 > 2; k0 -= 2)
	 *
	 * When written as a for loop, the `a' and `b' pointers are
	 * unnecessarily recomputed in the software pipeline epilogue,
	 * rather than reusing the updated pointers from the loop.
	 */
	__asm__ goto (
		/* if (k0 <= 2) goto tail; */
		"bleu %0, %1, %l[tail]"
		:
		: "r" (k0), "r" (2)
		:
		: tail);

	do {

		bli_sgemm_hwacha_16xn_load_a(a);
		vmca(va16, b);
		vf(bli_sgemm_hwacha_16xn_vf_inner_0);
		b += rs_c0;

		bli_sgemm_hwacha_16xn_load_a(a + 16);
		vmca(va16, b);
		vf(bli_sgemm_hwacha_16xn_vf_inner_1);
		b += rs_c0;
		a += 32;
		k0 -= 2;
	} while (k0 > 2);

tail:
	bli_sgemm_hwacha_16xn_load_a(a);
	if (k0 > 1) {
		vmca(va16, b);
		vf(bli_sgemm_hwacha_16xn_vf_inner_0);

		bli_sgemm_hwacha_16xn_load_a(a + 16);
		vf(bli_sgemm_hwacha_16xn_vf_tail_1);
	} else {
		vf(bli_sgemm_hwacha_16xn_vf_tail_0);
	}

	vf(bli_sgemm_hwacha_16xn_vf_end);
	__asm__ __volatile__ ("fence" ::: "memory");
}
