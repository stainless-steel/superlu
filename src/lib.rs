//! Interface to [SuperLU][1].
//!
//! [1]: http://crd-legacy.lbl.gov/~xiaoye/SuperLU

extern crate libc;
extern crate matrix;
extern crate superlu_sys as ffi;

use matrix::Compressed;
use std::convert::From;
use std::mem;

/// A super matrix.
pub struct SuperMatrix {
    raw: ffi::SuperMatrix,
}

impl SuperMatrix {
    /// Create a matrix from a raw structure.
    ///
    /// The underlying memory is considered to be owned, and it will be freed
    /// when the object goes out of scope.
    pub unsafe fn from_raw(raw: ffi::SuperMatrix) -> SuperMatrix {
        SuperMatrix { raw: raw }
    }

    /// Consume the object returning the wrapped raw structure without freeing
    /// the underlying memory.
    pub fn into_raw(self) -> ffi::SuperMatrix {
        let raw = self.raw;
        mem::forget(self);
        raw
    }
}

impl Drop for SuperMatrix {
    fn drop(&mut self) {
        match self.raw.Stype {
            ffi::Stype_t::SLU_NC => unsafe {
                ffi::Destroy_CompCol_Matrix(&mut self.raw);
            },
            ffi::Stype_t::SLU_NCP => unsafe {
                ffi::Destroy_CompCol_Permuted(&mut self.raw);
            },
            ffi::Stype_t::SLU_NR => unsafe {
                ffi::Destroy_CompRow_Matrix(&mut self.raw);
            },
            ffi::Stype_t::SLU_SC | ffi::Stype_t::SLU_SCP | ffi::Stype_t::SLU_SR => unsafe {
                ffi::Destroy_SuperNode_Matrix(&mut self.raw);
            },
            ffi::Stype_t::SLU_DN => unsafe {
                ffi::Destroy_Dense_Matrix(&mut self.raw);
            },
            _ => {},
        }
    }
}

impl<'l> From<&'l SuperMatrix> for Option<Compressed<f64>> {
    fn from(matrix: &'l SuperMatrix) -> Option<Compressed<f64>> {
        let raw = &matrix.raw;

        let rows = raw.nrow as usize;
        let columns = raw.ncol as usize;

        match (raw.Stype, raw.Dtype, raw.Mtype) {
            (ffi::Stype_t::SLU_NC, ffi::Dtype_t::SLU_D, ffi::Mtype_t::SLU_GE) => unsafe {
                let store = &*(raw.Store as *const ffi::NCformat);
                let nonzeros = store.nnz as usize;

                let mut values = Vec::with_capacity(nonzeros);
                let mut indices = Vec::with_capacity(nonzeros);
                let mut offsets = Vec::with_capacity(columns + 1);

                for i in 0..nonzeros {
                    values.push(*(store.nzval.offset(i as isize) as *const libc::c_double));
                    indices.push(*store.rowind.offset(i as isize) as usize);
                }
                for i in 0..(columns + 1) {
                    offsets.push(*store.colptr.offset(i as isize) as usize);
                }

                Some(Compressed { rows: rows,
                    columns: columns,
                    nonzeros: nonzeros,
                    format: matrix::Major::Column,
                    data: values,
                    indices: indices,
                    offsets: offsets,
                })
            },
            (ffi::Stype_t::SLU_NC, ffi::Dtype_t::SLU_D, _) => unimplemented!(),
            (ffi::Stype_t::SLU_NCP, ffi::Dtype_t::SLU_D, _) => unimplemented!(),
            _ => return None,
        }
    }
}

impl From<SuperMatrix> for Option<Compressed<f64>> {
    #[inline]
    fn from(matrix: SuperMatrix) -> Option<Compressed<f64>> {
        (&matrix).into()
    }
}
