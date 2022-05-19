from jax.core import Var, JaxprEqn, ClosedJaxpr, Jaxpr, eval_jaxpr
from jax._src import source_info_util
from jax import lax, make_jaxpr
import sys

DEBUG = False


def debug_message(s):
    if DEBUG:
        print(s, file=sys.stderr)


class PrimitiveWrapper:
    def __init__(self, var_context):
        self.var_context = var_context
        self.primitives = {
            'reduce_max': lax.reduce_max_p,
            'sub': lax.sub_p,
            'add': lax.add_p,
            'exp': lax.exp_p,
            'reduce_sum': lax.reduce_sum_p,
            'log': lax.log_p
        }

    def _make_call(self, primitive):
        def fn(*args, out_aval, params=None):
            if isinstance(args[0], list):
                args = args[0]
            invars = []
            for arg in args:
                invars += get_vars(arg)

            out_var = self.var_context.new(out_aval)
            return JaxprEqn(
                invars=invars,
                outvars=[out_var],
                primitive=primitive,
                params=params or {},
                source_info=source_info_util.new_source_info(),
                effects=set()
            )
        return fn

    def __getattr__(self, fn):
        if fn in self.primitives:
            return self._make_call(self.primitives[fn])
        else:
            raise AttributeError()


class VarContext:
    def __init__(self, eqns):
        self.variables = []
        self.counts = set()
        for eqn in eqns:
            for invar in eqn.invars:
                self.add(invar, allow_in_set=True)
            for outvar in eqn.outvars:
                self.add(outvar, allow_in_set=True)

        self.fns = PrimitiveWrapper(self)

    def add(self, var, allow_in_set=False):
        if isinstance(var, Var):
            assert var.suffix == ''
            assert var.count not in self.counts or allow_in_set
            self.variables.append(var)
            self.counts.add(var.count)

    def new(self, aval):
        var = Var(max(self.counts) + 1, '', aval)
        self.add(var)
        return var


def replace_closed_jaxpr_eqns(closed_jaxpr, eqns):
    return ClosedJaxpr(
        jaxpr=Jaxpr(
            constvars=closed_jaxpr.jaxpr.constvars,
            invars=closed_jaxpr.jaxpr.invars,
            outvars=eqns[-1].outvars,
            eqns=eqns,
            effects=closed_jaxpr.jaxpr.effects,
        ),
        consts=closed_jaxpr.consts
    )


def replace_eqn_outvars(eqn, outvars):
    return JaxprEqn(
        invars=eqn.invars,
        outvars=outvars,
        primitive=eqn.primitive,
        params=eqn.params,
        effects=eqn.effects,
        source_info=eqn.source_info
    )


def get_vars(inpt):
    if isinstance(inpt, JaxprEqn):
        return inpt.outvars
    elif isinstance(inpt, Var):
        return [inpt]
    else:
        raise NotImplemented()


class Eqns:
    def __init__(self, jaxpr):
        # var -> eqn idx
        self.source = {}
        # eqn -> set of (eqn idx, var)s
        self.influenced = {}
        self.eqns = [_ for _ in jaxpr.eqns]

        for var in jaxpr.jaxpr.invars:
            assert var not in self.source
            self.source[var] = 'arg'

        for i, eqn in enumerate(self.eqns):
            for var in eqn.outvars:
                assert var not in self.source
                if isinstance(var, Var):
                    self.source[var] = i
            for var in eqn.invars:
                if isinstance(var, Var):
                    source = self.source[var]
                    if source not in self.influenced:
                        self.influenced[source] = set()
                    self.influenced[source].add((i, var))

    def eqns_without(self, eqns, stop_vars):
        affected = set(eqns)
        frontier = [_ for _ in eqns]
        changed = True

        # A good old worklist algorithm to remove things downstream of our equations
        for start in frontier:
            if start in self.influenced:
                for (end, var) in self.influenced[start]:
                    if end not in affected and var not in stop_vars:
                        affected.add(end)
                        frontier.append(end)

        return [eqn for i, eqn in enumerate(self.eqns) if i not in affected], affected


def matches(eqn_tup, name):
    (_, eqn) = eqn_tup
    return eqn.primitive.name == name


def get_next(vs, eqn_tup):
    (_, eqn) = eqn_tup
    return vs[eqn.invars[0]] if eqn.invars[0] in vs and len(eqn.invars) == 1 else None


def search_logsumexp(ir):
    vs = {}
    for i, eqn in enumerate(ir.eqns):
        if len(eqn.outvars) == 1:
            vs[eqn.outvars[0]] = (i, eqn)

    for eqn_tup in enumerate(ir.eqns):
        if matches(eqn_tup, 'log'):
            log = eqn_tup
            sm = get_next(vs, log)
            if sm and matches(sm, 'reduce_sum'):
                exp = get_next(vs, sm)
                if matches(exp, 'exp'):
                    print(f"Found logsumexp {log, sm, exp}")
                    return (log, sm, exp), (exp[1].invars, log[1].outvars)


def maybe_peephole_logsumexp_trick(ir):
    # If we don't find our pattern, just return the input
    spec = search_logsumexp(ir)
    if spec is None:
        return None

    # Otherwise, unpack the spec
    (log, sm, exp), (inpt_vars, output_vars) = spec
    vec_shape = inpt_vars[0].aval
    squeeze_shape = output_vars[0].aval

    # Double-check that no intermediate variables are used by something else
    eqns = Eqns(ir)
    new_eqns, removed = eqns.eqns_without([log[0], sm[0], exp[0]], output_vars)
    if len(removed) != 3:
        # intermediate steps were used, so can't do optimization
        return None

    # Fix the variable setting
    context = VarContext(ir.eqns)

    max_eqn = context.fns.reduce_max(inpt_vars, out_aval=squeeze_shape, params=sm[1].params)
    sub_eqn = context.fns.sub(inpt_vars[0], max_eqn, out_aval=vec_shape)
    exp_eqn = context.fns.exp(sub_eqn, out_aval=vec_shape)
    sum_eqn = context.fns.reduce_sum(exp_eqn, out_aval=squeeze_shape, params=sm[1].params)
    log_eqn = context.fns.log(sum_eqn, out_aval=squeeze_shape)
    add_eqn = replace_eqn_outvars(
        context.fns.add(max_eqn, log_eqn, out_aval=squeeze_shape),
        output_vars
    )

    new_eqns += [max_eqn, sub_eqn, exp_eqn, sum_eqn, log_eqn, add_eqn]

    return replace_closed_jaxpr_eqns(ir, new_eqns)


class PeepholeContext:
    def __init__(self, fn, transforms=None):
        # original function
        self.fn = fn
        # intermediate representation
        self.ir = None
        # transformations to apply
        self.transforms = transforms or []

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            print("Warning, peephole context doesn't handle kwargs yet", file=sys.stderr)
        if self.ir is None:
            debug_message("JITting")
            self.ir = make_jaxpr(self.fn)(*args, **kwargs)
            debug_message(f"Starting Jaxpr: {self.ir}")
            for transform in self.transforms:
                self.ir = transform(self.ir) or self.ir
                debug_message(f"Transformed Jaxpr: {self.ir}")
        else:
            debug_message("Using ir")
        return eval_jaxpr(self.ir.jaxpr, self.ir.literals, *args)


def peephole_improve(fn):
    return PeepholeContext(fn, [
        maybe_peephole_logsumexp_trick
    ])