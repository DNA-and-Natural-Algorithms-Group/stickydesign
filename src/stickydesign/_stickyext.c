#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject* fastsub(PyObject* self, PyObject* args) {

    PyArrayObject *py_x;
    PyArrayObject      *py_r;
    PyArrayIterObject *itr;
    double *p1,*res;
    double g,d;
    int axis = 1;
    int go;
    int i;
    
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &py_x, &PyArray_Type, &py_r))
        return NULL;
    
    g = 0;
    d = 0;
    itr = (PyArrayIterObject *) PyArray_IterAllButAxis((PyObject*)py_x,&axis);
    while(PyArray_ITER_NOTDONE(itr)) {
        go = py_x->strides[axis]/sizeof(double);
        p1 = (double *) PyArray_ITER_DATA(itr);
        res = (double *) PyArray_GETPTR1(py_r,itr->index);
        g = 0;
        d = 0;
        for (i = 0; i < py_x->dimensions[axis]; i++) {
            d+=*p1;
            if ((*p1)==0) {
                if (d>g) g=d;
                d=0;
            }
            p1+=go;
        }
        if (d>g) g=d;
        *res = g;
        PyArray_ITER_NEXT(itr);
    }
    Py_DECREF(itr);
    Py_RETURN_NONE;
    
}

/*  define functions in module */
static PyMethodDef FastSub[] =
{
     {"fastsub", (PyCFunction)fastsub, METH_VARARGS,
         "fastsub!"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
#if PY_MAJOR_VERSION >= 3

static int FastSub_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int FastSub_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "stickyext",
        NULL,
        sizeof(struct module_state),
        FastSub,
        NULL,
        FastSub_traverse,
        FastSub_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit__stickyext(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_stickyext(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_stickyext", FastSub);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_stickyext.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
     /* IMPORTANT: this must be called */
    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

