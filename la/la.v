module la

import vtl.num
import vsl.blas
import vsl.la

struct Workspace {
	size int
	work &f64
}

fn allocate_workspace(size int) Workspace {
	ptr := &f64(v_calloc(size * int(sizeof(f64))))
	return Workspace{
		size: size
		work: ptr
	}
}

fn fortran_view_or_copy(t num.NdArray) num.NdArray {
	if t.flags.fortran {
		return t.view()
	} else {
		return t.copy('F')
	}
}

fn fortran_copy(t num.NdArray) num.NdArray {
	return t.copy('F')
}

fn assert_square_matrix(a num.NdArray) {
	if a.ndims != 2 || a.shape[0] != a.shape[1] {
		panic('Matrix is not square')
	}
}

fn assert_matrix(a num.NdArray) {
	if a.ndims != 2 {
		panic('Tensor is not two-dimensional')
	}
}

pub fn ddot(a num.NdArray, b num.NdArray) f64 {
	if a.ndims != 1 || b.ndims != 1 {
		panic('Tensors must be one dimensional')
	} else if a.size != b.size {
		panic('Tensors must have the same shape')
	}
	return la.vector_dot(a.f64_array(), b.f64_array())
}

pub fn dger(a num.NdArray, b num.NdArray) num.NdArray {
	if a.ndims != 1 || b.ndims != 1 {
		panic('Tensors must be one dimensional')
	}
	m := la.vector_vector_tr_mul(1.0, a.f64_array(), b.f64_array())
	return num.from_f64_2d(m.get_deep2())
}

pub fn dnrm2(a num.NdArray) f64 {
	if a.ndims != 1 {
		panic('Tensor must be one dimensional')
	}
	return blas.dnrm2(a.size, a.f64_array(), a.strides[0])
}

pub fn dlange(a num.NdArray, norm byte) f64 {
	if a.ndims != 2 {
		panic('Tensor must be two-dimensional')
	}
	m := fortran_view_or_copy(a)
	workspace := []f64{len: m.shape[0] * int(sizeof(f64))}
	return blas.dlange(norm, m.shape[0], m.shape[1], m.f64_array(), m.shape[0], workspace)
}

pub fn dpotrf(a num.NdArray, up bool) num.NdArray {
	if a.ndims != 2 {
		panic('Tensor must be two-dimensional')
	}
	mut ret := fortran_copy(a)
	mut mut_ret := ret.f64_array()
	blas.dpotrf(up, ret.shape[0], mut mut_ret, ret.shape[0])
	ret.assign(num.from_f64(mut_ret, ret.shape))
	if up {
		num.triu_inpl(mut ret)
	} else {
		num.tril_inpl(mut ret)
	}
	return ret
}

pub fn det(a num.NdArray) f64 {
	assert_square_matrix(a)
	m := a.shape[0]
	n := a.shape[1]
	mat := la.matrix_raw(m, n, a.f64_array())
	return mat.det()
}

pub fn inv(a num.NdArray) num.NdArray {
	assert_square_matrix(a)
	ret := fortran_copy(a)
	n := a.shape[0]
	ipiv := []int{len: (n * int(sizeof(int)))}
	mut mut_ret := ret.f64_array()
	blas.dgetrf(n, n, mut mut_ret, n, ipiv)
	blas.dgetri(n, mut mut_ret, n, ipiv)
	ret.assign(num.from_f64(mut_ret, ret.shape))
	return ret
}

pub fn matmul(a num.NdArray, b num.NdArray) num.NdArray {
	mut dest := []f64{len: a.shape[0] * b.shape[1]}
	ma := match a.flags.contiguous {
		true { a }
		else { a.copy('C') }
	}
	mb := match b.flags.contiguous {
		true { b }
		else { b.copy('C') }
	}
	blas.dgemm(false, false, ma.shape[0], mb.shape[1], ma.shape[1], 1.0, ma.f64_array(),
		ma.shape[1], mb.f64_array(), mb.shape[1], 1.0, mut dest, mb.shape[1])
	return num.from_f64(dest, [a.shape[0], b.shape[1]])
}

pub fn eigh(a num.NdArray) []num.NdArray {
	assert_square_matrix(a)
	ret := fortran_copy(a)
	n := ret.shape[0]
	w := num.empty([n])
	jobz := blas.job_vlr(true)
	uplo := blas.l_uplo(false)
	info := 0
	workspace := allocate_workspace(3 * n - 1)
	C.LAPACKE_dsyev(jobz, uplo, &n, ret.buffer(), &n, w.buffer(), workspace.work)
	if info > 0 {
		panic('Failed to converge')
	}
	return [w, ret]
}

pub fn eig(a num.NdArray) []num.NdArray {
	assert_square_matrix(a)
	ret := fortran_copy(a)
	n := ret.shape[0]
	wr := num.empty([n])
	wl := wr.copy('C')
	vl := num.allocate_cpu([n, n], 'F')
	vr := vl.copy('C')
	workspace := allocate_workspace(n * 4)
	jobvr := blas.job_vlr(true)
	jobvl := blas.job_vlr(true)
	info := 0
	C.LAPACKE_dgeev(jobvl, jobvr, &n, ret.buffer(), &n, wr.buffer(), wl.buffer(), vl.buffer(),
		&n, vr.buffer(), &n, workspace.work)
	if info > 0 {
		panic('QR algorithm failed')
	}
	return [wr, vl]
}

pub fn eigvalsh(a num.NdArray) num.NdArray {
	assert_square_matrix(a)
	ret := fortran_view_or_copy(a)
	n := ret.shape[0]
	jobz := blas.job_vlr(true)
	uplo := blas.l_uplo(false)
	info := 0
	w := num.empty([n])
	workspace := allocate_workspace(3 * n - 1)
	C.LAPACKE_dsyev(jobz, uplo, &n, ret.buffer(), &n, w.buffer(), workspace.work)
	if info > 0 {
		panic('Failed to converge')
	}
	return w
}

pub fn eigvals(a num.NdArray) num.NdArray {
	assert_square_matrix(a)
	ret := fortran_copy(a)
	n := ret.shape[0]
	wr := num.empty([n])
	wl := wr.copy('C')
	vl := num.allocate_cpu([n, n], 'F')
	vr := vl.copy('C')
	workspace := allocate_workspace(n * 3)
	jobvr := blas.job_vlr(false)
	jobvl := blas.job_vlr(false)
	info := 0
	C.LAPACKE_dgeev(jobvl, jobvr, &n, ret.buffer(), &n, wr.buffer(), wl.buffer(), vl.buffer(),
		&n, vr.buffer(), &n, workspace.work)
	if info > 0 {
		panic('QR algorithm failed')
	}
	return wr
}

pub fn solve(a num.NdArray, b num.NdArray) num.NdArray {
	assert_square_matrix(a)
	af := fortran_view_or_copy(a)
	bf := fortran_copy(b)
	n := af.shape[0]
	mut m := bf.shape[0]
	if bf.ndims > 1 {
		m = bf.shape[1]
	}
	ipiv := &int(v_calloc(n * int(sizeof(int))))
	info := 0
	C.LAPACKE_dgesv(&n, &m, af.buffer(), &n, ipiv, bf.buffer(), &m, &info)
	return bf
}

pub fn hessenberg(a num.NdArray) num.NdArray {
	assert_square_matrix(a)
	mut ret := fortran_copy(a)
	if ret.shape[0] < 2 {
		return ret
	}
	n := ret.shape[0]
	s := num.empty([n])
	ilo := 0
	ihi := 0
	job := `B`
	info := 0
	C.LAPACKE_dgebal(job, &n, ret.buffer(), &n, &ilo, &ihi, s.buffer(), &info)
	tau := num.empty([n])
	workspace := allocate_workspace(n)
	C.LAPACKE_dgehrd(n, &ilo, &ihi, ret.buffer(), &n, tau.buffer(), workspace.work)
	num.triu_inpl_offset(mut ret, -1)
	return ret
}

fn int_prod(a []int) int {
	mut i := 1
	for el in a {
		i *= el
	}
	return i
}

pub fn tensordot(a num.NdArray, b num.NdArray, ax_a []int, ax_b []int) num.NdArray {
	as_ := a.shape
	nda := a.ndims
	bs := b.shape
	ndb := b.ndims
	mut equal := true
	mut axes_a := ax_a.clone()
	mut axes_b := ax_b.clone()
	if axes_a.len != axes_b.len {
		equal = false
	} else {
		for k in 0 .. axes_a.len {
			if as_[axes_a[k]] != bs[axes_b[k]] {
				equal = false
				break
			}
			if axes_a[k] < 0 {
				axes_a[k] += nda
			}
			if axes_b[k] < 0 {
				axes_b[k] += ndb
			}
		}
	}
	if !equal {
		panic('shape-mismatch for sum')
	}
	tmp := num.irange(0, nda)
	notin := tmp.filter(!(it in axes_a))
	mut newaxes_a := notin.clone()
	newaxes_a << axes_a
	mut n2 := 1
	for axis in axes_a {
		n2 *= as_[axis]
	}
	firstdim := notin.map(as_[it])
	val := int_prod(firstdim)
	newshape_a := [val, n2]
	tmpb := num.irange(0, ndb)
	notinb := tmpb.filter(!(it in axes_b))
	mut newaxes_b := axes_b.clone()
	newaxes_b << notinb
	n2 = 1
	for axis in axes_b {
		n2 *= bs[axis]
	}
	firstdimb := notin.map(bs[it])
	valb := int_prod(firstdimb)
	newshape_b := [n2, valb]
	mut outshape := []int{}
	outshape << firstdim
	outshape << firstdimb
	at := a.transpose(newaxes_a).reshape(newshape_a)
	bt := b.transpose(newaxes_b).reshape(newshape_b)
	res := matmul(at, bt)
	return res.reshape(outshape)
}
