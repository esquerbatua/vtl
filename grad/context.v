module grad

import vnum.num

// A Context keeps track of the computational graph for
// a number of operations.  Variables that interact with each
// other must belong to the same context, or state will be
// lost while tracking operations done.
// The generic type of a context is always going to be a specific
// class of `NdArray`, to allow easy creation of gradients on the
// fly.  Unlike standard `NdArray` operations, a `Context` cannot
// shift it's generic type, and operations resulting in a different
// data type will raise.
pub struct Context {
pub mut:
        // A list of all variables present in an operation.
        // This list can contain duplicates
        nodes []Node

        // If no_grad is set to true, operations will not
        // be cached, and backpropogation will not be possible
        no_grad bool
}

// Contexts can only be initialized as empty, and
// a generic type must be provided
pub fn new_ctx() &Context {
        return &Context{
                no_grad: false,
                nodes: []Node{}
        }
}

pub fn (ctx Context) len() int {
        return len(ctx.nodes)
}

pub fn (mut ctx Context) push(node Node) {
        ctx.nodes << node
}

pub fn (ctx Context) last() Node {
        return ctx.nodes.last()
}

pub fn (mut ctx Context) pop() Node {
        return ctx.nodes.pop()
}

pub struct ContextVariableData {
pub:
	value         num.NdArray
	requires_grad bool = true
}

pub fn (ctx &Context) variable(data ContextVariableData) &Variable {
        return new_variable(context: ctx, value: data.value, requires_grad: data.requires_grad)
}

pub fn (ctx Context) str() string {
        mut str := ""
        for i, node in nodes {
                if len(node.parents) <= 1 {
                        str << node.parents[0].value.shape
                } else {
                        str << "("
                        for pi, parent in node.parents {
                                if pi != 0 {
                                        str << ", "
                                }
                                str << parent.value.shape
                        }
                        str << ")"
                }
                str << node.payload.variable.value.shape
                if i != len(ctx.nodes) - 1 {
                        str << "\n"
                }
        }
        return str
}
