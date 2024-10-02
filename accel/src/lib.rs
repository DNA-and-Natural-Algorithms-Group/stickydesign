#![allow(non_snake_case)]
use pyo3::prelude::pymodule;

#[pymodule]
mod stickydesign_accel {

    use ndarray::{s, Array1, Array2, ArrayView2, Axis, Zip};
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
        PyUntypedArrayMethods,
    };
    use pyo3::prelude::*;

    #[pyfunction]
    fn fastsub(x: &Bound<PyArray2<f64>>, r: &Bound<PyArray1<f64>>) {
        let xro = x.readonly();
        let mut rrw = r.readwrite();
        let x = xro.as_array();
        let r = rrw.as_array_mut();

        Zip::from(x.rows()).and(r).for_each(|xr, rr| {
            let mut d = 0.0;
            let mut g = 0.0;
            xr.for_each(|&xv| {
                d += xv;
                if xv == 0.0 {
                    if d > g {
                        g = d;
                    };
                    d = 0.0;
                }
            });
            *rr = g;
        });
    }

    #[pyfunction]
    fn pytops<'py>(s: PyReadonlyArray2<u8>, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let s = s.as_array();
        let r = (s.slice(s![.., ..-1]).to_owned() * 4) + s.slice(s![.., 1..]);
        r.into_pyarray_bound(py)
    }

    #[pyfunction]
    fn pytorps<'py>(s: PyReadonlyArray2<u8>, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let s = s.as_array();
        let r = torps(&s);
        r.into_pyarray_bound(py)
    }

    fn tops(s: &ArrayView2<u8>) -> Array2<u8> {
        (s.slice(s![.., ..-1]).to_owned() * 4) + s.slice(s![.., 1..])
    }

    fn torps(s: &ArrayView2<u8>) -> Array2<u8> {
        let mut r = Array2::<u8>::zeros((s.nrows(), s.ncols() - 1));

        Zip::from(&mut r)
            .and(s.slice(s![.., ..-1;-1]))
            .and(s.slice(s![.., 1..;-1]))
            .for_each(|r, &a, &b| *r = 15 - 4 * b - a);

        r
    }

    #[pyfunction]
    fn fastuniform<'py>(
        seqs1: PyReadonlyArray2<u8>,
        seqs2: PyReadonlyArray2<u8>,
        nndG: PyReadonlyArray1<f64>,
        ltmm: PyReadonlyArray1<f64>,
        rtmm: PyReadonlyArray1<f64>,
        intmm: PyReadonlyArray1<f64>,
        looppenalty: f64,
        singlepair: bool,
        initdG: f64,
        allow_shifts: bool,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        let seqs1 = seqs1.as_array();
        let seqs2 = seqs2.as_array();

        let nndG = nndG.as_array();
        let ltmm = ltmm.as_array();
        let rtmm = rtmm.as_array();
        let intmm = intmm.as_array();

        let s1 = tops(&seqs1);
        let s2r = torps(&seqs2);

        let mut ens: Array1<f64> = Array1::zeros(seqs1.shape()[0]);

        let ln = s1.ncols();

        let offset_range = if allow_shifts {
            (1 - ln as i32)..(ln as i32)
        } else {
            0..1
        };

        ens.indexed_iter_mut()
            .zip(s1.axis_iter(Axis(0)))
            .zip(s2r.axis_iter(Axis(0)))
            .for_each(|(((_r, e), s1), s2)| {
                for offset in offset_range.clone() {
                    let s1s = if offset == 0 {
                        s1.slice(s![..])
                    } else if offset < 0 {
                        s1.slice(s![-offset..])
                    } else {
                        s1.slice(s![..-offset])
                    };
                    let s2s = if offset == 0 {
                        s2.slice(s![..])
                    } else if offset < 0 {
                        s2.slice(s![..offset])
                    } else {
                        s2.slice(s![offset..])
                    };
                    let a = Zip::indexed(&s1s).and(&s2s).fold(
                        (0., 0.0_f64),
                        |(acc, bindmax), i, &p1, &p2| {
                            let pv = (16 * p1 + p2) as usize;
                            if p1 == p2 {
                                (acc + nndG[p1 as usize], bindmax)
                            } else if (i > 0) && (s1s[i - 1] == s2s[i - 1]) && (rtmm[pv] != 0.) {
                                (acc + intmm[pv], bindmax.max(acc + rtmm[pv]))
                            } else if (ltmm[pv] != 0.) && (i < s1s.len() - 1) {
                                if (singlepair || (s1s[i + 1] == s2s[i + 1]))
                                    && (ltmm[pv] > acc + intmm[pv])
                                {
                                    (ltmm[pv], bindmax)
                                } else {
                                    (acc + intmm[pv], bindmax)
                                }
                            } else {
                                (acc - looppenalty, bindmax)
                            }
                        },
                    );
                    *e = (*e).max(a.0).max(a.1);
                }
                *e -= initdG;
            });

        ens.into_pyarray_bound(py)
    }

    #[pyfunction]
    fn rust_tightloop<'py>(
        ens: PyReadonlyArray2<f64>,
        ltmm: PyReadonlyArray2<f64>,
        rtmm: PyReadonlyArray2<f64>,
        intmm: PyReadonlyArray2<f64>,
        singlepair: bool,
        looppenalty: f64,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut acc: f64;
        let mut bindmax = Array1::zeros(ens.shape()[0]);
        let ens = ens.as_array();
        let ltmm = ltmm.as_array();
        let rtmm = rtmm.as_array();
        let intmm = intmm.as_array();

        for e in 0..ens.shape()[0] {
            acc = 0.;
            for i in 0..ens.shape()[1] {
                if ens[(e, i)] != 0. {
                    // We're matching: add the pair to the accumulator.
                    acc += ens[(e, i)]
                } else if (rtmm[(e, i)] != 0.) && (i > 0) && (ens[(e, i - 1)] > 0.) {
                    if acc + rtmm[(e, i)] > bindmax[e] {
                        // we're mismatching on the right: see if
                        // right-dangling is highest binding so far,
                        // and continue, adding intmm to accumulator.
                        // Update: we only want to do this if the last
                        // nnpair was bound, because otherwise, weD
                        // can't have a "right" mismatch.
                        bindmax[e] = acc + rtmm[(e, i)];
                    }
                    acc += intmm[(e, i)];
                } else if (ltmm[(e, i)] != 0.) && (i < ens.shape()[1] - 1) {
                    // don't do this for the last pair; we're mismatching on
                    // the left: see if our ltmm is stronger than our
                    // accumulated binding+intmm. If so, reset to ltmm and
                    // continue as left-dangling, or reset to 0 if ltmm+next
                    // is weaker than next dangle,or next is also a mismatch
                    // (fixme: good idea?). If not, continue as internal
                    // mismatch.
                    if ((ens[(e, i + 1)] > 0.) || singlepair)
                        && (ltmm[(e, i)] > acc + intmm[(e, i)])
                    {
                        acc = ltmm[(e, i)];
                    } else {
                        acc += intmm[(e, i)];
                    }
                } else {
                    acc -= looppenalty;
                }
            }
            bindmax[e] = bindmax[e].max(acc);
        }

        bindmax.into_pyarray_bound(py)
    }
}
