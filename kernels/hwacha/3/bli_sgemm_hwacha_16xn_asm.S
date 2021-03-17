/* SPDX-License-Identifier: BSD-3-Clause */

	.set MR, 16
	.set NR, 16

	.text

	.altmacro

	.macro OP insn, r0, n0, r1, n1, r2, n2, r3, n3
	.ifb r1
	\insn \r0\n0
	.else
	.ifb r2
	\insn \r0\n0, \r1\n1
	.else
	.ifb r3
	\insn \r0\n0, \r1\n1, \r2\n2
	.else
	\insn \r0\n0, \r1\n1, \r2\n2, \r3\n3
	.endif
	.endif
	.endif
	.endm

	.align 3
	.global bli_sgemm_hwacha_16xn_vf_init
bli_sgemm_hwacha_16xn_vf_init:
	vpset vp0
	/* Software pipeline prologue: Load first row of B */
	vlw vv16, va16
	/* Initialize MR rows of C to 0.0f */
	.set i, 0
	.rept MR
	OP vadd vv,%i, vs0,, vs0
	.set i, i+1
	.endr
	vstop

	.align 3
	.global bli_sgemm_hwacha_16xn_vf_init_beta
bli_sgemm_hwacha_16xn_vf_init_beta:
	/* Load MR rows of C; multiply by beta (vs63)  */
	vpset vp0
	.set i, 0
	.rept MR
	OP vlw vv,%i, va,%i
	OP vfmul.s vv,%i, vv,%i, vs63
	.set i, i+1
	.endr
	/* Software pipeline prologue: Load first row of B */
	vlw vv16, va16
	vstop

	.align 3
	.global bli_sgemm_hwacha_16xn_vf_inner_0
	.global bli_sgemm_hwacha_16xn_vf_tail_0
bli_sgemm_hwacha_16xn_vf_inner_0:
	/* Software pipeline: Preload next row of B */
	vlw vv17, va16
bli_sgemm_hwacha_16xn_vf_tail_0:
	/* alpha * B[k][_] */
	vfmul.s vv16, vv16, vs63
	.set i, 0
	.rept MR
	/* C[i][_] += A[i][k] * B[k][_] */
	OP vfmadd.s vv,%i, vv16,, vs,%(i+1), vv,%i
	.set i, i+1
	.endr
	vstop

	.align 3
	.global bli_sgemm_hwacha_16xn_vf_inner_1
	.global bli_sgemm_hwacha_16xn_vf_tail_1
bli_sgemm_hwacha_16xn_vf_inner_1:
	/* Software pipeline: Preload next row of B */
	vlw vv16, va16
bli_sgemm_hwacha_16xn_vf_tail_1:
	/* alpha * B[k+1][_] */
	vfmul.s vv17, vv17, vs63
	.set i, 0
	.rept MR
	/* C[i][_] += A[i][k+1] * B[k+1][_] */
	OP vfmadd.s vv,%i, vv17,, vs,%(i+1), vv,%i
	.set i, i+1
	.endr
	vstop

	.align 3
	.global bli_sgemm_hwacha_16xn_vf_end
bli_sgemm_hwacha_16xn_vf_end:
	/* Store MR rows of C */
	.set i, 0
	.rept MR
	OP vsw vv,%i, va,%i
	.set i, i+1
	.endr
	vstop

