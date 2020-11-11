module grad

import vnum.num

// A variable is an abstraction of a num.NdArray that tracks
// the operations done to the num.NdArray. It also keeps
// track of the gradient of the operation if a Variable
// needs to backpropogate.
// This is the fundamental object used in automatic
// differentiation, as well as the neural network aspects
// of Num.cr
pub struct Variable {
pub:
	// The graph the variable is associated with.  This is a reference,
	// as a variable does not own its context
	context       &Context
	// The value of the Variable.  This should not be edited outside
	// of Variable operations, as other edits will not be tracked
	// and will lead to incorrect results
	value         num.NdArray
mut:
	// The gradient of the Variable.  This is set as a reference to
	// the value of a Variable unless `backprop` has been called, in
	// which case all related Variables will have their gradient
	// updated correctly
	grad          T
	// If set to true, this variable will track its operations,
	// otherwise it will act similar to a num.NdArray, only calculating
	// forward operations
	requires_grad bool
}

pub struct VariableData {
pub:
	context       &Context
	value         num.NdArray
	requires_grad bool = true
}

// Initialization method for a Variable.
// This method should only be called by a context, as it creates
// a Variable.  Context provides a helper method to add a
// Variable to the computational graph that handles ownership
// of the context and other related instance variables
pub fn new_variable(data VariableData) &Variable {
	grad := if data.requires_grad { num.zeros_like(data.value) } else { data.value }
	return &Variable{
		context: context
		value: value
		grad: grad
		requires_grad: requires_grad
	}
}

pub fn (v Variable) is_grad_needed() bool {
	return v.requires_grad && !v.context.no_grad
}

pub fn (v Variable) str() string {
	return v.value.str()
}

// Back propogates an operation along a computational graph.
// This operation will destroy the operational graph, populating
// the gradients for all variables that are predecessors of
// the Variable this is called on.
// Even if this is called on the first node in a graph, it will
// destroy all descendents of this variable stored by the
// Context
pub fn (v &Variable) backprop(debug bool) {
	grad := num.ones_like(value)
	for v.context.len() > 0 && v.context.last().variable != v {
		node := v.context.pop()
		$if debug {
			print(node.name)
		}
	}
	for v.context.len() > 0 {
		cur_node := v.context.pop()
		$if debug {
			print(node.name)
		}
		diffs := cur_node.gate.backward(cur_node.payload)
		mut i := 0
		for iter := a.iter(); !iter.done; iter.next() {
			diff := *iter.ptr
			parent_i = cur_node.parents[i]
			if parent_i.requires_grad() {
				parent_i.grad += diff
			}
			i++
		}
	}
}
